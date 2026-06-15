"""Minimal projection/provenance checks for covariance-target audits.

This script is deliberately narrower than the covariance provenance audit.  It
answers whether the projection math is sane, whether the current cache tangent
basis is a full 116D response-space basis at k=116, and whether an orth(J)
basis constructed inside the same code path as the exact J-linear covariance
captures that covariance target.
"""
from __future__ import annotations

import argparse
import hashlib
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
    covariance_residual_noise_side,
    movement_covariance_pooled_residual,
    orthonormalize_columns,
    top_eigenvectors,
)
from jake.twininfo.run_covariance_optimality import _load_trace_examples_from_metadata

from declan.active_sensing_movie_information.run_covariance_target_provenance_audit import (
    _cache_tangent_basis_for_group,
    _closure_for_group,
    _cov_key,
    _linear_prediction_rows,
    _subset_manifest_tangent_basis,
    _trace_capture,
)


def _hash_dataframe_rows(df: pd.DataFrame, cols: list[str]) -> str:
    present = [col for col in cols if col in df.columns]
    if not present:
        return "missing"
    payload = df[present].astype(str).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _j_column_matrix(j_group: list[np.ndarray], *, center_columns: bool) -> np.ndarray:
    cols = []
    for jac in j_group:
        arr = np.asarray(jac, dtype=np.float64)
        flat = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)
        if center_columns:
            flat = flat - flat.mean(axis=0, keepdims=True)
        cols.append(flat)
    return np.concatenate(cols, axis=1)


def _rank(matrix: np.ndarray, *, tol: float = 1e-10) -> int:
    vals = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if vals.size == 0:
        return 0
    cutoff = float(tol) * max(float(vals[0]), 1.0)
    return int(np.sum(vals > cutoff))


def _projector(basis: np.ndarray, n_units: int) -> np.ndarray:
    u = orthonormalize_columns(basis, n_rows=n_units)
    if u.shape[1] == 0:
        return np.zeros((n_units, n_units), dtype=np.float64)
    return u @ u.T


def _j_residual_frac(j_matrix: np.ndarray, basis: np.ndarray) -> float:
    j = np.asarray(j_matrix, dtype=np.float64)
    denom = float(np.sum(j * j))
    if denom <= 1e-18:
        return np.nan
    p = _projector(basis, j.shape[0])
    residual = j - p @ j
    return float(np.sum(residual * residual) / denom)


def _cov_residual_frac(covariance: np.ndarray, basis: np.ndarray) -> float:
    cov = np.asarray(covariance, dtype=np.float64)
    trace = float(np.trace(cov))
    if trace <= 1e-18:
        return np.nan
    _compact, residual = covariance_residual_noise_side(cov, basis)
    return float(np.trace(residual) / trace)


def _basis_diagnostics(basis: np.ndarray, n_units: int) -> dict[str, float | int]:
    u = orthonormalize_columns(basis, n_rows=n_units)
    utu_err = float(np.linalg.norm(u.T @ u - np.eye(u.shape[1]), ord="fro")) if u.shape[1] else 0.0
    uut_err = float(np.linalg.norm(u @ u.T - np.eye(n_units), ord="fro")) if u.shape[1] else float(np.sqrt(n_units))
    return {
        "basis_shape_rows": int(u.shape[0]),
        "basis_shape_cols": int(u.shape[1]),
        "basis_rank": _rank(u),
        "uT_u_minus_I_fro": utu_err,
        "u_uT_minus_I_fro": uut_err,
    }


def _assert_close(name: str, value: float, expected: float, tol: float) -> dict[str, Any]:
    ok = bool(np.isfinite(value) and abs(float(value) - float(expected)) <= float(tol))
    return {
        "assertion": name,
        "value": float(value) if np.isfinite(value) else np.nan,
        "expected": float(expected),
        "tolerance": float(tol),
        "passed": ok,
    }


def _write_markdown(path: Path, checks: pd.DataFrame, provenance: pd.DataFrame) -> None:
    lines = [
        "# Covariance Projection Debug",
        "",
        "## Assertion Summary",
        "",
    ]
    cols = [
        "family",
        "kind",
        "scale_D",
        "target",
        "basis_name",
        "assertion",
        "value",
        "expected",
        "tolerance",
        "passed",
    ]
    available = [col for col in cols if col in checks.columns]
    lines.append("| " + " | ".join(available) + " |")
    lines.append("| " + " | ".join("---" for _ in available) + " |")
    for _, row in checks[available].iterrows():
        vals = []
        for col in available:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append("nan" if not np.isfinite(value) else f"{float(value):.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(["", "## Provenance", ""])
    prov_cols = [
        "family",
        "kind",
        "scale_D",
        "target",
        "basis_name",
        "basis_source_file",
        "covariance_source_file",
        "n_units",
        "basis_shape",
        "basis_rank",
        "cov_shape",
        "basis_unit_hash",
        "covariance_unit_hash",
        "response_space_transform",
        "centering_residualization_flags",
        "normalization_whitening_flags",
    ]
    available = [col for col in prov_cols if col in provenance.columns]
    lines.append("| " + " | ".join(available) + " |")
    lines.append("| " + " | ".join("---" for _ in available) + " |")
    for _, row in provenance[available].iterrows():
        vals = []
        for col in available:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append("nan" if not np.isfinite(value) else f"{float(value):.6g}")
            else:
                vals.append(str(value).replace("|", "/"))
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
    p.add_argument("--max-rows-per-group", type=int, default=0)
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

    config = json.loads((covopt_dir / "metadata" / "covopt_run_config.json").read_text(encoding="utf-8"))
    from_run_dir = Path(config["from_run_dir"])
    t_max = int(config.get("t_max", 128))
    seed = int(config.get("seed", args.seed))

    population_csv = covopt_dir / "metadata" / "covopt_population_units.csv"
    rate_records_csv = covopt_dir / "metadata" / "covopt_rate_records.csv"
    mu_j_npz = covopt_dir / "cache" / "covopt_mu_j.npz"
    covariance_npz = covopt_dir / "cache" / "covopt_covariances.npz"

    population = pd.read_csv(population_csv).reset_index(names="cov_index")
    _manifest_q, matched = _subset_manifest_tangent_basis(
        population_csv=population_csv,
        canonical_manifest_csv=Path(args.canonical_manifest_csv),
        tangent_basis_npz=Path(args.tangent_basis_npz),
        basis_key=str(args.basis_key),
    )
    cov_idx = matched["cov_index"].to_numpy(dtype=int)
    matched_population = population.iloc[cov_idx].reset_index(drop=True)
    unit_hash = _hash_dataframe_rows(
        matched_population,
        ["session_name", "original_neuron_id", "global_unit_idx", "source_readout_index", "simulated_unit_idx"],
    )

    records = pd.read_csv(rate_records_csv)
    records = records[records["scale_D"].astype(float).isin(scales)].copy()
    with np.load(mu_j_npz) as z:
        mu_all = np.asarray(z["mu"], dtype=np.float32)
        j_all = np.asarray(z["J"], dtype=np.float32)
        expected_all = np.asarray(z["expected_spikes_t"], dtype=np.float32)
    with np.load(covariance_npz) as z:
        covs = {name: np.asarray(z[name], dtype=np.float64) for name in z.files}

    print("Loading digital twin only to reconstruct selected eye traces...", flush=True)
    model, _model_info, _device = load_digital_twin(device=str(args.device))
    trace_by_id = _load_trace_examples_from_metadata(from_run_dir, model, t_max=t_max)

    checks: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    group_items = list(records.groupby(["family", "kind", "scale_D"], sort=True))
    identity = np.eye(len(cov_idx), dtype=np.float64)

    for group_i, ((family, kind, scale), group) in enumerate(group_items, start=1):
        if int(args.max_rows_per_group) > 0 and len(group) > int(args.max_rows_per_group):
            group = group.sample(
                n=int(args.max_rows_per_group),
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).sort_values("row_id")
        row_ids = group["row_id"].to_numpy(dtype=int)
        j_group = [j_all[row_id][:, cov_idx, :] for row_id in row_ids]
        j_exact = _j_column_matrix(j_group, center_columns=False)
        j_centered = _j_column_matrix(j_group, center_columns=True)
        cache_q = _cache_tangent_basis_for_group(j_group, max_k=len(cov_idx))
        exact_j_q = orthonormalize_columns(j_exact, n_rows=len(cov_idx))
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
            "exact_j_linear_pooled": (
                exact_j_cov,
                f"{mu_j_npz}::J plus reconstructed deterministic trajectories",
            ),
            "within_pair": (
                covs[_cov_key(str(family), str(kind), float(scale), "within_pair")][np.ix_(cov_idx, cov_idx)],
                f"{covariance_npz}::{_cov_key(str(family), str(kind), float(scale), 'within_pair')}",
            ),
            "pooled_residual": (
                covs[_cov_key(str(family), str(kind), float(scale), "pooled_residual")][np.ix_(cov_idx, cov_idx)],
                f"{covariance_npz}::{_cov_key(str(family), str(kind), float(scale), 'pooled_residual')}",
            ),
        }
        print(
            f"[{group_i}/{len(group_items)}] {family} {kind} D={float(scale):g} rows={len(group)}",
            flush=True,
        )

        cache_diag = _basis_diagnostics(cache_q, len(cov_idx))
        exact_diag = _basis_diagnostics(exact_j_q, len(cov_idx))
        cache_j_resid = _j_residual_frac(j_exact, cache_q)
        cache_centered_j_resid = _j_residual_frac(j_centered, cache_q)
        exact_j_resid = _j_residual_frac(j_exact, exact_j_q)

        for target_name, (sigma, sigma_source) in targets.items():
            target_trace = float(np.trace(sigma))
            identity_capture = _trace_capture(sigma, identity)
            checks.append(
                {
                    "family": family,
                    "kind": kind,
                    "scale_D": float(scale),
                    "target": target_name,
                    "basis_name": "identity",
                    **_assert_close("trace_capture(I, Sigma) = 1", identity_capture, 1.0, 1e-10),
                }
            )

            cache_capture = _trace_capture(sigma, cache_q)
            cache_cov_resid = _cov_residual_frac(sigma, cache_q)
            checks.append(
                {
                    "family": family,
                    "kind": kind,
                    "scale_D": float(scale),
                    "target": target_name,
                    "basis_name": "cache_tangent_centered_k116",
                    "assertion": "1 - trace_capture equals residual_trace_frac",
                    "value": float(1.0 - cache_capture - cache_cov_resid),
                    "expected": 0.0,
                    "tolerance": 1e-8,
                    "passed": bool(abs(1.0 - cache_capture - cache_cov_resid) <= 1e-8),
                    "trace_capture": cache_capture,
                    "residual_trace_frac": cache_cov_resid,
                }
            )
            if target_name == "exact_j_linear_pooled":
                exact_capture = _trace_capture(sigma, exact_j_q)
                exact_cov_resid = _cov_residual_frac(sigma, exact_j_q)
                exact_closure, exact_residual_frac = _closure_for_group(
                    group=group,
                    mu_all=mu_all,
                    j_all=j_all,
                    expected_all=expected_all,
                    cov_idx=cov_idx,
                    sigma_target=sigma,
                    basis=exact_j_q,
                    eps=float(args.eps),
                )
                checks.extend(
                    [
                        {
                            "family": family,
                            "kind": kind,
                            "scale_D": float(scale),
                            "target": target_name,
                            "basis_name": "orth_J_exact",
                            **_assert_close("trace_capture(orth(J_exact), Sigma_J) = 1", exact_capture, 1.0, 1e-8),
                            "residual_trace_frac": exact_cov_resid,
                            "closure": exact_closure,
                        },
                        {
                            "family": family,
                            "kind": kind,
                            "scale_D": float(scale),
                            "target": target_name,
                            "basis_name": "orth_J_exact",
                            **_assert_close("residual_trace_frac(orth(J_exact), Sigma_J) = 0", exact_cov_resid, 0.0, 1e-8),
                            "trace_capture": exact_capture,
                            "closure": exact_closure,
                        },
                        {
                            "family": family,
                            "kind": kind,
                            "scale_D": float(scale),
                            "target": target_name,
                            "basis_name": "orth_J_exact",
                            **_assert_close("direct J residual frac for orth(J_exact) = 0", exact_j_resid, 0.0, 1e-8),
                            "trace_capture": exact_capture,
                            "closure": exact_closure,
                        },
                        {
                            "family": family,
                            "kind": kind,
                            "scale_D": float(scale),
                            "target": target_name,
                            "basis_name": "orth_J_exact",
                            **_assert_close("closure(orth(J_exact), Sigma_J) = 1", exact_closure, 1.0, 1e-8),
                            "residual_frac_fisher": exact_residual_frac,
                        },
                    ]
                )

            top_q = top_eigenvectors(sigma, len(cov_idx))
            basis_specs = [
                (
                    "identity",
                    identity,
                    "constructed in run_covariance_projection_debug.py",
                    "identity projection; no centering",
                ),
                (
                    "cache_tangent_centered_k116",
                    cache_q,
                    f"{mu_j_npz}::J",
                    "each cached J column unit-centered before SVD; covariance rows not unit-centered",
                ),
                (
                    "orth_J_exact",
                    exact_j_q,
                    f"{mu_j_npz}::J",
                    "same uncentered J matrix used to reconstruct exact J-linear target; Sigma_J rows are time-mean residualized",
                ),
                (
                    "oracle_topPC_full",
                    top_q,
                    sigma_source,
                    "top eigenvectors of this covariance target",
                ),
            ]
            for basis_name, basis, basis_source, centering_flags in basis_specs:
                diag = _basis_diagnostics(basis, len(cov_idx))
                provenance.append(
                    {
                        "family": family,
                        "kind": kind,
                        "scale_D": float(scale),
                        "target": target_name,
                        "basis_name": basis_name,
                        "basis_source_file": basis_source,
                        "basis_unit_hash": unit_hash,
                        "covariance_source_file": sigma_source,
                        "covariance_unit_hash": unit_hash,
                        "n_units": int(len(cov_idx)),
                        "basis_shape": f"{diag['basis_shape_rows']}x{diag['basis_shape_cols']}",
                        "basis_rank": int(diag["basis_rank"]),
                        "cov_shape": f"{sigma.shape[0]}x{sigma.shape[1]}",
                        "target_trace": target_trace,
                        "basis_trace_capture": _trace_capture(sigma, basis),
                        "response_space_transform": "covopt 256 sampled units intersected with canonical manifest to 116 matched units via cov_index",
                        "centering_residualization_flags": centering_flags,
                        "normalization_whitening_flags": "SVD orthonormalization tol=1e-10; no whitening; covariance PSD projection from estimator/helper",
                        "exact_j_n_samples": int(exact_j_samples) if target_name == "exact_j_linear_pooled" else np.nan,
                        "exact_j_n_pairs": int(exact_j_pairs) if target_name == "exact_j_linear_pooled" else np.nan,
                    }
                )

        checks.extend(
            [
                {
                    "family": family,
                    "kind": kind,
                    "scale_D": float(scale),
                    "target": "basis_only",
                    "basis_name": "cache_tangent_centered_k116",
                    **_assert_close("U.T @ U = I", float(cache_diag["uT_u_minus_I_fro"]), 0.0, 1e-8),
                    **cache_diag,
                },
                {
                    "family": family,
                    "kind": kind,
                    "scale_D": float(scale),
                    "target": "basis_only",
                    "basis_name": "cache_tangent_centered_k116",
                    "assertion": "rank(U_cache_k116) = 116",
                    "value": float(cache_diag["basis_rank"]),
                    "expected": float(len(cov_idx)),
                    "tolerance": 0.0,
                    "passed": bool(int(cache_diag["basis_rank"]) == int(len(cov_idx))),
                    **cache_diag,
                },
                {
                    "family": family,
                    "kind": kind,
                    "scale_D": float(scale),
                    "target": "basis_only",
                    "basis_name": "cache_tangent_centered_k116",
                    **_assert_close("U @ U.T = I", float(cache_diag["u_uT_minus_I_fro"]), 0.0, 1e-8),
                    **cache_diag,
                },
                {
                    "family": family,
                    "kind": kind,
                    "scale_D": float(scale),
                    "target": "J_columns",
                    "basis_name": "cache_tangent_centered_k116",
                    **_assert_close("direct uncentered J residual frac for cache basis = 0", cache_j_resid, 0.0, 1e-8),
                    "direct_centered_J_residual_frac": cache_centered_j_resid,
                },
                {
                    "family": family,
                    "kind": kind,
                    "scale_D": float(scale),
                    "target": "J_columns",
                    "basis_name": "cache_tangent_centered_k116",
                    **_assert_close("direct centered J residual frac for cache basis = 0", cache_centered_j_resid, 0.0, 1e-8),
                    "direct_uncentered_J_residual_frac": cache_j_resid,
                },
                {
                    "family": family,
                    "kind": kind,
                    "scale_D": float(scale),
                    "target": "basis_only",
                    "basis_name": "orth_J_exact",
                    **_assert_close("rank(orth(J_exact)) = 116", float(exact_diag["basis_rank"]), float(len(cov_idx)), 0.0),
                    **exact_diag,
                },
            ]
        )

    checks_df = pd.DataFrame(checks)
    provenance_df = pd.DataFrame(provenance)
    checks_df.to_csv(out_dir / "projection_debug_assertions.csv", index=False)
    provenance_df.to_csv(out_dir / "projection_debug_provenance.csv", index=False)
    _write_markdown(out_dir / "projection_debug_summary.md", checks_df, provenance_df)
    metadata = {
        "covopt_dir": str(covopt_dir),
        "from_run_dir": str(from_run_dir),
        "tangent_basis_npz": str(args.tangent_basis_npz),
        "canonical_manifest_csv": str(args.canonical_manifest_csv),
        "basis_key": str(args.basis_key),
        "population_csv": str(population_csv),
        "rate_records_csv": str(rate_records_csv),
        "mu_j_npz": str(mu_j_npz),
        "covariance_npz": str(covariance_npz),
        "scales": list(scales),
        "n_covariance_units": int(len(population)),
        "n_matched_units": int(len(cov_idx)),
        "matched_unit_hash": unit_hash,
        "max_rows_per_group": int(args.max_rows_per_group),
        "assertion_rows": int(len(checks_df)),
        "failed_assertion_rows": int((~checks_df["passed"].fillna(False).astype(bool)).sum()),
    }
    (out_dir / "projection_debug_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
