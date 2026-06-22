"""Diagnose what compact-aware finite trajectory priors are really weighting."""

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
    from .analyze_compact_aware_prior import (
        _build_prior_entries,
        _build_shared_prior_pools,
        _family_config,
        _gain_basis,
        _json_ready,
        _stable_trajectory_key,
        _trajectory_key_matrix,
    )
    from .analyze_compact_mechanism import (
        _load_basis,
        _project_delta,
        _random_basis,
        _static_pc_basis,
        _unit_shuffle_basis,
        _validate_basis_mode,
    )
    from .analyze_feature_posterior import (
        _filter_manifest,
        _filter_response_cache_manifest,
        _load_npz,
        _parse_int_list,
        _parse_str_list,
    )
    from .likelihood import normalized_log_weights, poisson_expected_count_loglik, posterior_from_log_scores
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.backimage_trajectory_observer.analyze_compact_aware_prior import (
        _build_prior_entries,
        _build_shared_prior_pools,
        _family_config,
        _gain_basis,
        _json_ready,
        _stable_trajectory_key,
        _trajectory_key_matrix,
    )
    from declan.backimage_trajectory_observer.analyze_compact_mechanism import (
        _load_basis,
        _project_delta,
        _random_basis,
        _static_pc_basis,
        _unit_shuffle_basis,
        _validate_basis_mode,
    )
    from declan.backimage_trajectory_observer.analyze_feature_posterior import (
        _filter_manifest,
        _filter_response_cache_manifest,
        _load_npz,
        _parse_int_list,
        _parse_str_list,
    )
    from declan.backimage_trajectory_observer.likelihood import (
        normalized_log_weights,
        poisson_expected_count_loglik,
        posterior_from_log_scores,
    )


DEFAULT_PRIOR_FAMILIES = (
    "image_independent_compact_prior,"
    "random_subspace_aware,"
    "unit_shuffle_compact_aware,"
    "gain_axis_aware,"
    "static_pc_aware,"
    "inverse_compact_control"
)


def _progress(message: str) -> None:
    print(f"[compact-prior-diagnostics] {message}", flush=True)


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


def _scalar_int(table: dict[str, np.ndarray], key: str, default: int = -1) -> int:
    if key not in table:
        return int(default)
    arr = np.asarray(table[key]).reshape(-1)
    return int(arr[0]) if arr.size else int(default)


def _candidate_ids(table: dict[str, np.ndarray], n_candidates: int) -> list[str]:
    if "candidate_ids" not in table:
        return [str(i) for i in range(int(n_candidates))]
    return [str(v) for v in np.asarray(table["candidate_ids"]).tolist()]


def _basis_for_type(
    basis_type: str,
    *,
    compact_u: np.ndarray,
    static_basis: np.ndarray | None,
    n_units: int,
    seed: int,
    k_dim: int,
) -> np.ndarray | None:
    if basis_type == "compact":
        return compact_u
    if basis_type == "unit_shuffle_compact":
        basis, _perm = _unit_shuffle_basis(compact_u, np.random.default_rng(int(seed) + 10_000 + int(k_dim)))
        return basis
    if basis_type == "gain_ones":
        return _gain_basis(n_units)
    if basis_type == "static_pc":
        if static_basis is None:
            return None
        return static_basis[:, : int(k_dim)]
    return None


def _trajectory_catalog_lookup(path: Path) -> tuple[dict[tuple[int, int, str], dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}, []
    cols = [
        "trial_id",
        "trajectory_id",
        "trajectory_identity_id",
        "candidate_index",
        "candidate_id",
        "effective_rms_deg",
        "rendered_rms_displacement_deg",
        "path_length_deg",
        "rendered_path_length_deg",
        "speed_mean_deg_s",
        "speed_p95_deg_s",
        "generated_lag1_autocorr",
        "source_row",
        "sample_index",
        "axis_relation",
        "output_axis_deg",
    ]
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in cols if col in header.columns]
    df = pd.read_csv(path, usecols=usecols)
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        trial_id = int(row["trial_id"])
        candidate_index = -1
        if "candidate_index" in row and not pd.isna(row["candidate_index"]):
            candidate_index = int(row["candidate_index"])
        seen_stable_keys: set[str] = set()
        for key_col in ("trajectory_id", "trajectory_identity_id"):
            if key_col not in row or pd.isna(row[key_col]):
                continue
            key = _stable_trajectory_key(str(row[key_col]))
            if key.startswith("prior:"):
                stable = key
            else:
                stable = "prior:" + key
            if stable in seen_stable_keys:
                continue
            seen_stable_keys.add(stable)
            payload = row.to_dict()
            payload["stable_trajectory_key"] = stable
            payload["trajectory_catalog_candidate_index"] = int(candidate_index)
            out[(trial_id, int(candidate_index), stable)] = payload
            records.append(
                {
                    "trial_id": int(trial_id),
                    "candidate_index": int(candidate_index),
                    "stable_trajectory_key": stable,
                    **{
                        col: payload.get(col, np.nan)
                        for col in [
                            "effective_rms_deg",
                            "rendered_rms_displacement_deg",
                            "path_length_deg",
                            "rendered_path_length_deg",
                            "speed_mean_deg_s",
                            "speed_p95_deg_s",
                            "generated_lag1_autocorr",
                            "output_axis_deg",
                        ]
                    },
                }
            )

    ambiguity_rows: list[dict[str, Any]] = []
    if records:
        rec_df = pd.DataFrame(records)
        group_cols = ["trial_id", "stable_trajectory_key"]
        for col in [
            "effective_rms_deg",
            "rendered_rms_displacement_deg",
            "path_length_deg",
            "rendered_path_length_deg",
            "speed_mean_deg_s",
            "speed_p95_deg_s",
            "generated_lag1_autocorr",
            "output_axis_deg",
        ]:
            if col not in rec_df.columns:
                continue
            vals = pd.to_numeric(rec_df[col], errors="coerce")
            if not np.isfinite(vals.to_numpy()).any():
                continue
            tmp = rec_df.assign(_value=vals)
            ranges = tmp.groupby(group_cols, dropna=False)["_value"].max() - tmp.groupby(group_cols, dropna=False)[
                "_value"
            ].min()
            ranges = ranges.dropna()
            if ranges.empty:
                continue
            ambiguity_rows.append(
                {
                    "covariate": col,
                    "n_stable_key_groups": int(ranges.shape[0]),
                    "n_groups_vary_across_candidates": int((ranges > 1e-9).sum()),
                    "fraction_groups_vary_across_candidates": float(np.mean(ranges.to_numpy() > 1e-9)),
                    "max_within_stable_key_range": float(ranges.max()),
                }
            )
    return out, ambiguity_rows


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(xx.to_numpy()) & np.isfinite(yy.to_numpy())
    if int(mask.sum()) < 3:
        return float("nan")
    if float(np.nanstd(xx.to_numpy()[mask])) <= 1e-12 or float(np.nanstd(yy.to_numpy()[mask])) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx.to_numpy()[mask], yy.to_numpy()[mask])[0, 1])


def _slot_covariates(
    *,
    prior: np.ndarray,
    zero: np.ndarray,
    compact_u: np.ndarray,
    static_basis: np.ndarray | None,
    eps: float,
) -> dict[str, np.ndarray]:
    delta = np.asarray(prior, dtype=np.float64) - np.asarray(zero, dtype=np.float64)[:, None, :, :]
    total_energy = np.sum(delta * delta, axis=(-2, -1))
    compact_residual = delta - _project_delta(delta, compact_u)
    compact_residual_energy = np.sum(compact_residual * compact_residual, axis=(-2, -1))
    compact_leakage = compact_residual_energy / (total_energy + float(eps))
    gain_basis = _gain_basis(delta.shape[-1])
    gain_projected = _project_delta(delta, gain_basis)
    gain_energy_fraction = np.sum(gain_projected * gain_projected, axis=(-2, -1)) / (total_energy + float(eps))
    rate_delta = np.sum(delta, axis=(-2, -1))
    out = {
        "response_delta_energy": np.mean(total_energy, axis=0),
        "compact_leakage": np.mean(compact_leakage, axis=0),
        "compact_energy_fraction": np.mean(1.0 - compact_leakage, axis=0),
        "gain_energy_fraction": np.mean(gain_energy_fraction, axis=0),
        "abs_rate_delta": np.mean(np.abs(rate_delta), axis=0),
    }
    if static_basis is not None:
        static_projected = _project_delta(delta, static_basis[:, : compact_u.shape[1]])
        static_fraction = np.sum(static_projected * static_projected, axis=(-2, -1)) / (total_energy + float(eps))
        out["static_pc_energy_fraction"] = np.mean(static_fraction, axis=0)
    return out


def _entry_slot_arrays(entry: dict[str, Any], *, true_candidate_index: int, n_trajectories: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = entry["raw"]
    log_prior_raw = entry["log_prior"]
    if log_prior_raw is None:
        log_prior = normalized_log_weights(None, int(n_trajectories))
    else:
        log_arr = np.asarray(log_prior_raw, dtype=np.float64)
        n_rows = int(log_arr.shape[0]) if log_arr.ndim == 2 else None
        log_prior = normalized_log_weights(log_arr, int(n_trajectories), n_rows=n_rows)
    if raw is None:
        raw_slot = np.zeros(int(n_trajectories), dtype=np.float64)
    else:
        raw_arr = np.asarray(raw, dtype=np.float64)
        if raw_arr.ndim == 1:
            raw_slot = raw_arr
        elif raw_arr.ndim == 2:
            idx = int(np.clip(true_candidate_index, 0, raw_arr.shape[0] - 1))
            raw_slot = raw_arr[idx]
        else:
            raise ValueError(f"Unsupported raw prior shape {raw_arr.shape}")
    if log_prior_raw is None or np.asarray(log_prior_raw).ndim == 1:
        log_slot = np.asarray(log_prior, dtype=np.float64)
    else:
        idx = int(np.clip(true_candidate_index, 0, np.asarray(log_prior).shape[0] - 1))
        log_slot = np.asarray(log_prior, dtype=np.float64)[idx]
    prob_slot = np.exp(log_slot)
    return raw_slot, log_slot, prob_slot


def _family_label(row: pd.Series) -> str:
    family = str(row["trajectory_weight_family"])
    null_id = int(row.get("random_seed_or_null_id", -1))
    if family == "random_subspace_aware" and null_id >= 0:
        return f"{family}:{null_id}"
    return family


def _make_plots(out_dir: Path, slot_df: pd.DataFrame, corr_df: pd.DataFrame) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional in headless CI
        _progress(f"plotting skipped: {exc}")
        return paths

    plot_dir = out_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    focus_families = [
        "image_independent_compact_prior",
        "gain_axis_aware",
        "unit_shuffle_compact_aware",
        "static_pc_aware",
        "inverse_compact_control",
    ]
    focus = slot_df[slot_df["trajectory_weight_family"].isin(focus_families)].copy()
    if not focus.empty:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        specs = [
            ("response_delta_energy", "response-change energy"),
            ("compact_leakage", "compact residual fraction"),
            ("gain_energy_fraction", "gain energy fraction"),
            ("path_length_deg", "trajectory path length"),
        ]
        for ax, (col, label) in zip(axes.flat, specs, strict=True):
            for family, grp in focus.groupby("trajectory_weight_family"):
                x = pd.to_numeric(grp[col], errors="coerce")
                y = pd.to_numeric(grp["prior_probability"], errors="coerce")
                mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
                if int(mask.sum()) == 0:
                    continue
                ax.scatter(x.to_numpy()[mask], y.to_numpy()[mask], s=4, alpha=0.18, label=family)
            ax.set_xlabel(label)
            ax.set_ylabel("trajectory prior probability")
        axes.flat[0].legend(fontsize=7, loc="best", markerscale=2)
        path = plot_dir / "prior_probability_vs_slot_covariates.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    if not corr_df.empty:
        corr = corr_df[corr_df["weight_metric"].eq("prior_probability")].copy()
        corr = corr[corr["trajectory_weight_family"].isin(focus_families + ["random_subspace_aware"])]
        if not corr.empty:
            pivot = corr.pivot_table(
                index="trajectory_weight_family",
                columns="covariate",
                values="pearson_r",
                aggfunc="mean",
            )
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(12, max(3.5, 0.45 * len(pivot))), constrained_layout=True)
                im = ax.imshow(pivot.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        val = pivot.to_numpy(dtype=float)[i, j]
                        if np.isfinite(val):
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
                fig.colorbar(im, ax=ax, label="Pearson r")
                path = plot_dir / "prior_probability_covariate_correlations.png"
                fig.savefig(path, dpi=180)
                plt.close(fig)
                paths.append(str(path))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--compact-basis-path", type=Path, required=True)
    parser.add_argument("--basis-key", default="auto")
    parser.add_argument("--basis-mode", default="image_disjoint")
    parser.add_argument("--allow-unverified-image-disjoint-basis", action="store_true")
    parser.add_argument("--candidate-set-modes", default="")
    parser.add_argument("--priors", default="")
    parser.add_argument("--motion-scales", default="")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--k-dims", default="10")
    parser.add_argument("--prior-families", default=DEFAULT_PRIOR_FAMILIES)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--prior-beta-max", type=float, default=8.0)
    parser.add_argument("--entropy-match-target", default="image_independent_compact_prior")
    parser.add_argument("--n-random", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--likelihood-scale", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=64)
    return parser


def analyze(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(run_dir / "response_cache_manifest.csv")
    manifest = _filter_manifest(manifest, args)
    if manifest.empty:
        raise ValueError("No response cache rows remain after filtering")
    manifest, skipped_cache_rows = _filter_response_cache_manifest(manifest, run_dir)
    if manifest.empty:
        raise ValueError("No response cache files remain after cache filtering")

    first = _load_npz(run_dir / str(manifest.iloc[0]["response_cache_path"]))
    n_units = int(np.asarray(first["prior_lambda_counts"]).shape[-1])
    k_dims = _parse_int_list(args.k_dims)
    if not k_dims:
        raise ValueError("--k-dims must request at least one dimension")
    max_k = max(k_dims)
    prior_families = _parse_str_list(args.prior_families)
    basis_full, basis_meta = _load_basis(Path(args.compact_basis_path), n_units=n_units, basis_key=str(args.basis_key))
    _validate_basis_mode(args, basis_meta)
    if basis_full.shape[1] < max_k:
        raise ValueError(f"Basis has only {basis_full.shape[1]} columns, but max k={max_k}")

    static_basis = None
    if any(_family_config(fam)["basis_type"] == "static_pc" for fam in prior_families):
        _progress("building static response PC basis")
        zero_tables = []
        for _, row in manifest.iterrows():
            table = _load_npz(run_dir / str(row["response_cache_path"]))
            zero_tables.append(np.asarray(table["zero_lambda_counts"], dtype=np.float32))
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
    trajectory_lookup, trajectory_catalog_ambiguity = _trajectory_catalog_lookup(run_dir / "axis_trajectory_catalog.csv")
    _progress(
        f"diagnosing {manifest.shape[0]} tables; families={','.join(prior_families)}; "
        f"k={','.join(str(v) for v in k_dims)}"
    )

    rows: list[dict[str, Any]] = []
    progress_every = max(1, int(args.progress_every))
    for progress_i, (table_index, man_row) in enumerate(tqdm(list(manifest.iterrows()), desc="prior weight diagnostics"), start=1):
        table_key = str(man_row["response_cache_path"])
        table = _load_npz(run_dir / table_key)
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        y_obs = np.asarray(table["y_obs_counts"], dtype=np.float64)
        n_candidates, n_trajectories = int(prior.shape[0]), int(prior.shape[1])
        true_idx = _scalar_int(table, "true_candidate_index", 0)
        true_tau = _scalar_int(table, "true_trajectory_index", -1)
        nearest_tau = _scalar_int(table, "nearest_trajectory_index", -1)
        trajectory_keys = _trajectory_key_matrix(table, n_candidates=n_candidates, n_trajectories=n_trajectories)
        candidate_ids = _candidate_ids(table, n_candidates)
        prior_ll = poisson_expected_count_loglik(
            y_obs,
            prior,
            eps=float(args.eps),
            likelihood_scale=float(args.likelihood_scale),
        )
        uniform_tau_posterior = posterior_from_log_scores(prior_ll[int(true_idx)])
        for k_dim in k_dims:
            compact_u = basis_full[:, : int(k_dim)]
            slot_cov = _slot_covariates(
                prior=prior,
                zero=zero,
                compact_u=compact_u,
                static_basis=static_basis,
                eps=float(args.eps),
            )
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
                table_key=table_key,
                shared_prior_pools=shared_prior_pools,
            )
            basis_covariates: dict[tuple[str, int], np.ndarray] = {}
            for prior_entry in prior_entries:
                family = str(prior_entry["family"])
                basis_type = str(prior_entry["basis_type"])
                null_id = int(prior_entry["null_id"])
                basis = None
                if prior_entry.get("basis") is not None:
                    basis = np.asarray(prior_entry["basis"], dtype=np.float64)
                elif basis_type != "none":
                    basis = _basis_for_type(
                        basis_type,
                        compact_u=compact_u,
                        static_basis=static_basis,
                        n_units=n_units,
                        seed=int(args.seed),
                        k_dim=int(k_dim),
                    )
                if basis is not None and (basis_type, null_id) not in basis_covariates:
                    delta = prior - zero[:, None, :, :]
                    projected = _project_delta(delta, basis)
                    energy = np.sum(delta * delta, axis=(-2, -1))
                    projected_energy = np.sum(projected * projected, axis=(-2, -1))
                    basis_covariates[(basis_type, null_id)] = np.mean(projected_energy / (energy + float(args.eps)), axis=0)

                raw_slot, log_slot, prob_slot = _entry_slot_arrays(
                    prior_entry,
                    true_candidate_index=true_idx,
                    n_trajectories=n_trajectories,
                )
                tau_posterior = posterior_from_log_scores(prior_ll[int(true_idx)] + log_slot)
                basis_fraction = basis_covariates.get((basis_type, null_id), np.full(n_trajectories, np.nan))
                for slot in range(n_trajectories):
                    stable_key = str(trajectory_keys[int(true_idx), int(slot)])
                    catalog_key = (int(man_row["trial_id"]), int(true_idx), stable_key)
                    catalog = trajectory_lookup.get(catalog_key)
                    catalog_source = "true_candidate_stable_key"
                    if catalog is None:
                        catalog = trajectory_lookup.get((int(man_row["trial_id"]), -1, stable_key))
                        catalog_source = "candidate_unspecified_stable_key" if catalog is not None else "missing"
                    if catalog is None:
                        catalog = {}
                    row: dict[str, Any] = {
                        "table_index": int(table_index),
                        "trial_id": int(man_row["trial_id"]),
                        "response_cache_path": table_key,
                        "candidate_set_mode": str(man_row.get("candidate_set_mode", "")),
                        "prior_family": str(man_row.get("prior_family", "")),
                        "scale": float(man_row.get("scale", np.nan)),
                        "axis_catalog_mode": str(man_row.get("axis_catalog_mode", "")),
                        "k_dim": int(k_dim),
                        "n_candidates": int(n_candidates),
                        "n_trajectories": int(n_trajectories),
                        "true_candidate_index": int(true_idx),
                        "true_image_id": str(candidate_ids[int(true_idx)]),
                        "diagnostic_candidate_scope": "true_candidate_row",
                        "diagnostic_candidate_index": int(true_idx),
                        "trajectory_slot": int(slot),
                        "stable_trajectory_key": stable_key,
                        "trajectory_catalog_metadata_source": catalog_source,
                        "trajectory_catalog_metadata_candidate_index": int(
                            catalog.get("trajectory_catalog_candidate_index", -1)
                        ),
                        "is_true_trajectory_slot": bool(int(slot) == int(true_tau)) if int(true_tau) >= 0 else False,
                        "is_nearest_trajectory_slot": bool(int(slot) == int(nearest_tau)) if int(nearest_tau) >= 0 else False,
                        "trajectory_weight_family": family,
                        "trajectory_weight_shape": str(prior_entry["shape_kind"]),
                        "trajectory_weight_basis_type": basis_type,
                        "random_seed_or_null_id": int(null_id),
                        "prior_beta": float(prior_entry["beta"]),
                        "prior_raw_weight": float(raw_slot[int(slot)]),
                        "prior_log_probability": float(log_slot[int(slot)]),
                        "prior_probability": float(prob_slot[int(slot)]),
                        "true_candidate_tau_log_likelihood": float(prior_ll[int(true_idx), int(slot)]),
                        "true_candidate_tau_posterior_uniform": float(uniform_tau_posterior[int(slot)]),
                        "true_candidate_tau_posterior_with_prior": float(tau_posterior[int(slot)]),
                        "basis_energy_fraction": float(basis_fraction[int(slot)]),
                        "shared_prior_key_scope": str(prior_entry["shared_prior_key_scope"]),
                        "shared_prior_fallback_fraction": float(prior_entry["shared_prior_fallback_fraction"]),
                        "shared_prior_stable_key_fallback_fraction": float(
                            prior_entry["shared_prior_stable_key_fallback_fraction"]
                        ),
                    }
                    for cov_name, cov_values in slot_cov.items():
                        row[cov_name] = float(cov_values[int(slot)])
                    for cov_name in [
                        "effective_rms_deg",
                        "rendered_rms_displacement_deg",
                        "path_length_deg",
                        "rendered_path_length_deg",
                        "speed_mean_deg_s",
                        "speed_p95_deg_s",
                        "generated_lag1_autocorr",
                        "source_row",
                        "sample_index",
                        "axis_relation",
                        "output_axis_deg",
                    ]:
                        row[cov_name] = catalog.get(cov_name, np.nan)
                    rows.append(row)
        if progress_i % progress_every == 0:
            _progress(f"processed {progress_i}/{manifest.shape[0]} response tables")

    slot_df = pd.DataFrame(rows)
    slot_path = out_dir / "trajectory_prior_slot_diagnostics.csv"
    _write_csv(slot_path, rows)

    covariates = [
        "response_delta_energy",
        "compact_leakage",
        "compact_energy_fraction",
        "gain_energy_fraction",
        "static_pc_energy_fraction",
        "abs_rate_delta",
        "basis_energy_fraction",
        "effective_rms_deg",
        "rendered_rms_displacement_deg",
        "path_length_deg",
        "rendered_path_length_deg",
        "speed_mean_deg_s",
        "speed_p95_deg_s",
        "generated_lag1_autocorr",
        "true_candidate_tau_log_likelihood",
        "true_candidate_tau_posterior_uniform",
        "is_nearest_trajectory_slot",
    ]
    covariates = [col for col in covariates if col in slot_df.columns]
    corr_rows: list[dict[str, Any]] = []
    for (family, basis_type, null_id), grp in slot_df.groupby(
        ["trajectory_weight_family", "trajectory_weight_basis_type", "random_seed_or_null_id"],
        dropna=False,
    ):
        for metric in ("prior_raw_weight", "prior_probability"):
            for cov in covariates:
                corr_rows.append(
                    {
                        "trajectory_weight_family": str(family),
                        "trajectory_weight_basis_type": str(basis_type),
                        "random_seed_or_null_id": int(null_id),
                        "weight_metric": metric,
                        "covariate": cov,
                        "pearson_r": _safe_corr(grp[metric], grp[cov]),
                        "n_slots": int(len(grp)),
                    }
                )
    corr_df = pd.DataFrame(corr_rows)
    _write_csv(out_dir / "trajectory_prior_family_correlations.csv", corr_rows)

    summary_rows: list[dict[str, Any]] = []
    for (family, basis_type, null_id), grp in slot_df.groupby(
        ["trajectory_weight_family", "trajectory_weight_basis_type", "random_seed_or_null_id"],
        dropna=False,
    ):
        row = {
            "trajectory_weight_family": str(family),
            "trajectory_weight_basis_type": str(basis_type),
            "random_seed_or_null_id": int(null_id),
            "n_slot_rows": int(len(grp)),
            "n_tables": int(grp["response_cache_path"].nunique()),
            "mean_prior_probability": float(pd.to_numeric(grp["prior_probability"], errors="coerce").mean()),
            "median_prior_probability": float(pd.to_numeric(grp["prior_probability"], errors="coerce").median()),
            "mean_prior_probability_nearest_slot": float(
                pd.to_numeric(grp.loc[grp["is_nearest_trajectory_slot"].astype(bool), "prior_probability"], errors="coerce").mean()
            ),
            "mean_uniform_tau_posterior_nearest_slot": float(
                pd.to_numeric(
                    grp.loc[grp["is_nearest_trajectory_slot"].astype(bool), "true_candidate_tau_posterior_uniform"],
                    errors="coerce",
                ).mean()
            ),
            "mean_prior_tau_posterior_nearest_slot": float(
                pd.to_numeric(
                    grp.loc[grp["is_nearest_trajectory_slot"].astype(bool), "true_candidate_tau_posterior_with_prior"],
                    errors="coerce",
                ).mean()
            ),
            "mean_shared_prior_fallback_fraction": float(
                pd.to_numeric(grp["shared_prior_fallback_fraction"], errors="coerce").mean()
            ),
            "mean_shared_prior_stable_key_fallback_fraction": float(
                pd.to_numeric(grp["shared_prior_stable_key_fallback_fraction"], errors="coerce").mean()
            ),
        }
        summary_rows.append(row)
    _write_csv(out_dir / "trajectory_prior_family_summary.csv", summary_rows)
    figure_paths = _make_plots(out_dir, slot_df, corr_df)

    report_lines = [
        "# Compact-Aware Prior Weight Diagnostics",
        "",
        "This cache-only diagnostic asks what finite trajectory reweighting is actually tracking.",
        "The prior weights are rebuilt before looking at the observed response posterior diagnostics;",
        "posterior columns are marked as diagnostics, not inputs to the prior.",
        "",
        "## Inputs",
        "",
        f"- source run: `{run_dir}`",
        f"- compact basis: `{args.compact_basis_path}`",
        f"- k dimensions: `{args.k_dims}`",
        f"- prior families: `{args.prior_families}`",
        f"- entropy match target: `{args.entropy_match_target}`",
        "",
        "## Primary Outputs",
        "",
        "- `trajectory_prior_slot_diagnostics.csv`",
        "- `trajectory_prior_family_correlations.csv`",
        "- `trajectory_prior_family_summary.csv`",
        "- `figures/prior_probability_vs_slot_covariates.png`",
        "- `figures/prior_probability_covariate_correlations.png`",
        "",
        "## Family Summary Preview",
        "",
        "```text",
        pd.DataFrame(summary_rows).head(20).to_csv(index=False).strip(),
        "```",
    ]
    if trajectory_catalog_ambiguity:
        report_lines.extend(
            [
                "",
                "## Trajectory Catalog Metadata",
                "",
                "Catalog covariates are joined by true candidate index plus hash-stripped stable trajectory key.",
                "The table below reports where those stable keys still vary across candidate rows.",
                "",
                "```text",
                pd.DataFrame(trajectory_catalog_ambiguity).to_csv(index=False).strip(),
                "```",
            ]
        )
    if figure_paths:
        report_lines.extend(["", "## Figures", ""])
        for path in figure_paths:
            report_lines.append(f"- `{Path(path).relative_to(out_dir)}`")
    (out_dir / "trajectory_prior_weight_diagnostics_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    _write_json(
        out_dir / "trajectory_prior_weight_diagnostics_metadata.json",
        {
            "run_dir": run_dir,
            "compact_basis_path": args.compact_basis_path,
            "basis_mode": args.basis_mode,
            "k_dims": k_dims,
            "prior_families": prior_families,
            "n_selected_tables": int(manifest.shape[0]),
            "n_manifest_rows_without_response_cache": int(skipped_cache_rows),
            "trajectory_catalog_metadata_join": "trial_id,true_candidate_index,hash_stripped_stable_trajectory_key",
            "trajectory_catalog_ambiguity_summary": trajectory_catalog_ambiguity,
            "outputs": [
                "trajectory_prior_slot_diagnostics.csv",
                "trajectory_prior_family_correlations.csv",
                "trajectory_prior_family_summary.csv",
                "trajectory_prior_weight_diagnostics_report.md",
                "trajectory_prior_weight_diagnostics_metadata.json",
            ],
            "figures": figure_paths,
        },
    )
    _progress(f"wrote compact prior diagnostics to {out_dir}")
    return out_dir


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
