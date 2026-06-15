"""Audit which covariance target each tangent basis spans.

This is a provenance audit, not another closure-interpretation run.  It
compares trace capture for three covariance targets:

1. exact J-linear target reconstructed from cached row Jacobians and the
   deterministic eye trajectory used by that row;
2. cached within-pair movement covariance;
3. cached pooled-residual movement covariance.

The key metric is trace(P_B Sigma P_B) / trace(Sigma).  Closure scoring can be
enabled explicitly, but is off by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

VISIONCORE_ROOT = Path(__file__).resolve().parents[2]
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from jake.twininfo.common import load_digital_twin
from jake.twininfo.covariance_optimality import (
    covariance_fisher_by_time,
    covariance_residual_noise_side,
    movement_covariance_pooled_residual,
    orthonormalize_columns,
    top_eigenvectors,
    trajectories_for_scaled_family,
)
from jake.twininfo.pipeline import _example_seed
from jake.twininfo.run_covariance_optimality import _load_trace_examples_from_metadata


def _scale_label(scale: float) -> str:
    return str(float(scale)).replace(".", "p").replace("-", "m")


def _cov_key(family: str, kind: str, scale: float, estimator: str) -> str:
    return f"{family}__{kind or 'all'}__D{_scale_label(scale)}__{estimator}"


def _subset_manifest_tangent_basis(
    *,
    population_csv: Path,
    canonical_manifest_csv: Path,
    tangent_basis_npz: Path,
    basis_key: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    population = pd.read_csv(population_csv).reset_index(names="cov_index")
    canonical = pd.read_csv(canonical_manifest_csv)
    matched = population.merge(
        canonical,
        left_on=["session_name", "original_neuron_id"],
        right_on=["session_name", "unit_id"],
        how="inner",
        validate="many_to_one",
    ).sort_values("cov_index")
    if matched.empty:
        raise ValueError("No overlap between covariance population and canonical tangent manifest.")
    with np.load(tangent_basis_npz) as z:
        basis_756 = np.asarray(z[basis_key], dtype=np.float64)
    basis_overlap = basis_756[matched["canonical_unit_index"].to_numpy(dtype=int)]
    return orthonormalize_columns(basis_overlap), matched


def _cache_tangent_basis_for_group(j_group: list[np.ndarray], *, max_k: int) -> np.ndarray:
    columns = []
    for jac in j_group:
        arr = np.asarray(jac, dtype=np.float64)
        flat = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)
        flat = flat - flat.mean(axis=0, keepdims=True)
        columns.append(flat)
    return orthonormalize_columns(np.concatenate(columns, axis=1))[:, : int(max_k)]


def _random_basis(n_units: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _r = np.linalg.qr(rng.standard_normal((n_units, int(k))))
    return q[:, : int(k)]


def _trace_capture(covariance: np.ndarray, basis: np.ndarray) -> float:
    cov = np.asarray(covariance, dtype=np.float64)
    trace = float(np.trace(cov))
    if trace <= 1e-18:
        return np.nan
    u = orthonormalize_columns(basis, n_rows=cov.shape[0])
    if u.shape[1] == 0:
        return 0.0
    projector = u @ u.T
    return float(np.trace(projector @ cov @ projector) / trace)


def _subspace_overlap(reference: np.ndarray, basis: np.ndarray) -> float:
    ref = orthonormalize_columns(reference)
    u = orthonormalize_columns(basis, n_rows=ref.shape[0])
    if ref.shape[1] == 0 or u.shape[1] == 0:
        return np.nan
    return float(np.trace(ref.T @ u @ u.T @ ref) / ref.shape[1])


def _score(mu: np.ndarray, jac: np.ndarray, expected: np.ndarray, sigma_extra: np.ndarray | None) -> float:
    fisher_t = covariance_fisher_by_time(mu, jac, sigma_extra)
    trace = float(np.trace(np.sum(fisher_t, axis=0)))
    return trace / max(float(np.sum(expected)), 1e-12)


def _closure_for_group(
    *,
    group: pd.DataFrame,
    mu_all: np.ndarray,
    j_all: np.ndarray,
    expected_all: np.ndarray,
    cov_idx: np.ndarray,
    sigma_target: np.ndarray,
    basis: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    _compact, residual = covariance_residual_noise_side(sigma_target, basis)
    closures: list[float] = []
    residuals: list[float] = []
    for row_id in group["row_id"].to_numpy(dtype=int):
        mu = mu_all[row_id][:, cov_idx]
        jac = j_all[row_id][:, cov_idx, :]
        expected = expected_all[row_id]
        aware = _score(mu, jac, expected, None)
        blind = _score(mu, jac, expected, sigma_target)
        corrected = _score(mu, jac, expected, residual)
        gap = aware - blind
        if np.isfinite(gap) and gap > eps:
            closures.append((corrected - blind) / gap)
            residuals.append((aware - corrected) / gap)
    return (
        float(np.mean(closures)) if closures else np.nan,
        float(np.mean(residuals)) if residuals else np.nan,
    )


def _linear_prediction_rows(
    *,
    group: pd.DataFrame,
    j_all: np.ndarray,
    cov_idx: np.ndarray,
    trace_by_id: dict[str, dict[str, Any]],
    seed: int,
    t_max: int,
) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for _, rec in group.iterrows():
        row_id = int(rec["row_id"])
        example_id = str(rec["example_id"])
        image_index = int(rec["image_index"])
        crop_rank = int(rec["crop_rank"])
        family = str(rec["family"])
        scale = float(rec["scale_D"])
        pair_seed = _example_seed(int(seed), example_id, image_index, crop_rank)
        trace = trace_by_id[example_id]["trace"]
        trajectory = trajectories_for_scaled_family(
            trace,
            family,
            (scale,),
            t_max=int(t_max),
            seed=pair_seed,
        )[scale][0]
        delta = trajectory - np.mean(trajectory, axis=0, keepdims=True)
        jac = np.asarray(j_all[row_id][:, cov_idx, :], dtype=np.float64)
        pred = np.einsum("tnd,td->tn", jac, delta[: jac.shape[0]])
        rows.append(pred.astype(np.float64))
    return rows


def _write_markdown(path: Path, summary: pd.DataFrame, metadata: dict[str, Any]) -> None:
    lines = [
        "# Covariance Target Provenance Audit",
        "",
        "## Scope",
        "",
        f"- Covopt dir: `{metadata['covopt_dir']}`.",
        f"- Source run dir: `{metadata['from_run_dir']}`.",
        f"- Matched overlap: {metadata['n_matched_units']} / {metadata['n_covariance_units']} covariance units.",
        f"- Scales: {metadata['scales']}.",
        f"- k list: {metadata['k_list']}.",
        f"- Closure computed: {metadata['compute_closure']}.",
        "",
        "## Target Definitions",
        "",
        "- `exact_j_linear_pooled`: covariance of cached row-wise linear predictions `J_t delta_e(t)`, pooled after subtracting each row mean.",
        "- `within_pair`: cached movement covariance estimator that averages within-pair PSD covariances.",
        "- `pooled_residual`: cached movement covariance estimator used by the current corrected closure audit.",
        "",
        "## Table",
        "",
    ]
    cols = [
        "family",
        "kind",
        "scale_D",
        "target",
        "k",
        "target_trace",
        "capture_oracle_topPC",
        "capture_cache_tangent",
        "capture_manifest_tangent",
        "capture_random_mean",
        "capture_unitshuffled_manifest_mean",
        "overlap_cache_with_topPC",
        "overlap_manifest_with_topPC",
        "closure_cache_tangent",
        "closure_manifest_tangent",
        "closure_oracle_topPC",
    ]
    available = [col for col in cols if col in summary.columns]
    lines.append("| " + " | ".join(available) + " |")
    lines.append("| " + " | ".join("---" for _ in available) + " |")
    for _, row in summary[available].iterrows():
        vals = []
        for col in available:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append("nan" if not np.isfinite(value) else f"{float(value):.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--covopt-dir", type=Path, required=True)
    p.add_argument("--tangent-basis-npz", type=Path, required=True)
    p.add_argument("--canonical-manifest-csv", type=Path, required=True)
    p.add_argument("--basis-key", default="basis_delta_0p25")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scales", default="1")
    p.add_argument("--k-list", default="2,5,10,20")
    p.add_argument("--max-rows-per-group", type=int, default=0)
    p.add_argument("--n-random-subspaces", type=int, default=5)
    p.add_argument("--n-unit-shuffles", type=int, default=5)
    p.add_argument("--compute-closure", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=1e-9)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    covopt_dir = Path(args.covopt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    scales = tuple(float(x) for x in str(args.scales).split(",") if x.strip())
    k_list = tuple(int(float(x)) for x in str(args.k_list).split(",") if x.strip())

    config = json.loads((covopt_dir / "metadata" / "covopt_run_config.json").read_text(encoding="utf-8"))
    from_run_dir = Path(config["from_run_dir"])
    t_max = int(config.get("t_max", 128))
    seed = int(config.get("seed", args.seed))

    manifest_q, matched = _subset_manifest_tangent_basis(
        population_csv=covopt_dir / "metadata" / "covopt_population_units.csv",
        canonical_manifest_csv=Path(args.canonical_manifest_csv),
        tangent_basis_npz=Path(args.tangent_basis_npz),
        basis_key=str(args.basis_key),
    )
    cov_idx = matched["cov_index"].to_numpy(dtype=int)
    n_covariance_units = int(pd.read_csv(covopt_dir / "metadata" / "covopt_population_units.csv").shape[0])
    records = pd.read_csv(covopt_dir / "metadata" / "covopt_rate_records.csv")
    records = records[records["scale_D"].astype(float).isin(scales)].copy()
    with np.load(covopt_dir / "cache" / "covopt_mu_j.npz") as z:
        mu_all = np.asarray(z["mu"], dtype=np.float32)
        j_all = np.asarray(z["J"], dtype=np.float32)
        expected_all = np.asarray(z["expected_spikes_t"], dtype=np.float32)
    with np.load(covopt_dir / "cache" / "covopt_covariances.npz") as z:
        covs = {name: np.asarray(z[name], dtype=np.float64) for name in z.files}

    print("Loading digital twin only to reconstruct selected eye traces...", flush=True)
    model, _model_info, _device = load_digital_twin(device=str(args.device))
    trace_by_id = _load_trace_examples_from_metadata(from_run_dir, model, t_max=t_max)

    rows: list[dict[str, Any]] = []
    group_items = list(records.groupby(["family", "kind", "scale_D"], sort=True))
    for group_i, ((family, kind, scale), group) in enumerate(group_items, start=1):
        if int(args.max_rows_per_group) > 0 and len(group) > int(args.max_rows_per_group):
            group = group.sample(
                n=int(args.max_rows_per_group),
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).sort_values("row_id")
        row_ids = group["row_id"].to_numpy(dtype=int)
        j_group = [j_all[row_id][:, cov_idx, :] for row_id in row_ids]
        cache_q = _cache_tangent_basis_for_group(j_group, max_k=max(k_list))
        exact_rows = _linear_prediction_rows(
            group=group,
            j_all=j_all,
            cov_idx=cov_idx,
            trace_by_id=trace_by_id,
            seed=seed,
            t_max=t_max,
        )
        exact_j_cov, exact_j_samples, exact_j_pairs = movement_covariance_pooled_residual(exact_rows)
        targets = {
            "exact_j_linear_pooled": exact_j_cov,
            "within_pair": covs[_cov_key(str(family), str(kind), float(scale), "within_pair")][np.ix_(cov_idx, cov_idx)],
            "pooled_residual": covs[_cov_key(str(family), str(kind), float(scale), "pooled_residual")][np.ix_(cov_idx, cov_idx)],
        }
        print(
            f"[{group_i}/{len(group_items)}] {family} {kind} D={float(scale):g} rows={len(group)}",
            flush=True,
        )
        for target_name, sigma in targets.items():
            target_trace = float(np.trace(sigma))
            for k in k_list:
                top_q = top_eigenvectors(sigma, int(k))
                manifest_k = manifest_q[:, : int(k)]
                cache_k = cache_q[:, : int(k)]
                random_caps = []
                for _draw in range(int(args.n_random_subspaces)):
                    random_caps.append(_trace_capture(sigma, _random_basis(len(cov_idx), int(k), rng)))
                shuffled_caps = []
                for _draw in range(int(args.n_unit_shuffles)):
                    shuffled_caps.append(_trace_capture(sigma, manifest_q[rng.permutation(len(cov_idx)), : int(k)]))
                row: dict[str, Any] = {
                    "family": family,
                    "kind": kind,
                    "scale_D": float(scale),
                    "target": target_name,
                    "k": int(k),
                    "target_trace": target_trace,
                    "n_rows": int(len(group)),
                    "exact_j_n_samples": int(exact_j_samples) if target_name == "exact_j_linear_pooled" else np.nan,
                    "exact_j_n_pairs": int(exact_j_pairs) if target_name == "exact_j_linear_pooled" else np.nan,
                    "capture_oracle_topPC": _trace_capture(sigma, top_q),
                    "capture_cache_tangent": _trace_capture(sigma, cache_k),
                    "capture_manifest_tangent": _trace_capture(sigma, manifest_k),
                    "capture_random_mean": float(np.nanmean(random_caps)) if random_caps else np.nan,
                    "capture_random_sd": float(np.nanstd(random_caps, ddof=1)) if len(random_caps) > 1 else np.nan,
                    "capture_unitshuffled_manifest_mean": float(np.nanmean(shuffled_caps)) if shuffled_caps else np.nan,
                    "capture_unitshuffled_manifest_sd": float(np.nanstd(shuffled_caps, ddof=1)) if len(shuffled_caps) > 1 else np.nan,
                    "overlap_cache_with_topPC": _subspace_overlap(top_q, cache_k),
                    "overlap_manifest_with_topPC": _subspace_overlap(top_q, manifest_k),
                }
                if bool(args.compute_closure):
                    row["closure_oracle_topPC"], row["residual_frac_oracle_topPC"] = _closure_for_group(
                        group=group,
                        mu_all=mu_all,
                        j_all=j_all,
                        expected_all=expected_all,
                        cov_idx=cov_idx,
                        sigma_target=sigma,
                        basis=top_q,
                        eps=float(args.eps),
                    )
                    row["closure_cache_tangent"], row["residual_frac_cache_tangent"] = _closure_for_group(
                        group=group,
                        mu_all=mu_all,
                        j_all=j_all,
                        expected_all=expected_all,
                        cov_idx=cov_idx,
                        sigma_target=sigma,
                        basis=cache_k,
                        eps=float(args.eps),
                    )
                    row["closure_manifest_tangent"], row["residual_frac_manifest_tangent"] = _closure_for_group(
                        group=group,
                        mu_all=mu_all,
                        j_all=j_all,
                        expected_all=expected_all,
                        cov_idx=cov_idx,
                        sigma_target=sigma,
                        basis=manifest_k,
                        eps=float(args.eps),
                    )
                rows.append(row)

    summary = pd.DataFrame(rows)
    metadata = {
        "covopt_dir": str(covopt_dir),
        "from_run_dir": str(from_run_dir),
        "basis_key": str(args.basis_key),
        "n_covariance_units": n_covariance_units,
        "n_matched_units": int(len(cov_idx)),
        "scales": list(scales),
        "k_list": list(k_list),
        "max_rows_per_group": int(args.max_rows_per_group),
        "n_random_subspaces": int(args.n_random_subspaces),
        "n_unit_shuffles": int(args.n_unit_shuffles),
        "compute_closure": bool(args.compute_closure),
        "targets": ["exact_j_linear_pooled", "within_pair", "pooled_residual"],
    }
    summary.to_csv(out_dir / "covariance_target_provenance_summary.csv", index=False)
    _write_markdown(out_dir / "covariance_target_provenance_summary.md", summary, metadata)
    (out_dir / "covariance_target_provenance_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
