"""Compare centered and uncentered exact-J tangent closure.

This is the focused follow-up after the projection debug identified that the
old cache tangent basis was built from unit-centered Jacobian columns.  The
audit keeps the corrected noise-side-only semantics and compares covariance
targets against:

* oracle top-PC of the target covariance;
* uncentered exact-J tangent basis;
* centered exact-J tangent basis;
* manifest tangent basis;
* random orthonormal controls;
* unit-shuffled manifest controls.

The output table is intentionally compact:

    target | basis | k | trace_capture | closure | residual_trace
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
    covariance_residual_noise_side,
    movement_covariance_pooled_residual,
    orthonormalize_columns,
    top_eigenvectors,
)
from jake.twininfo.run_covariance_optimality import _load_trace_examples_from_metadata

from declan.active_sensing_movie_information.run_covariance_target_provenance_audit import (
    _cov_key,
    _linear_prediction_rows,
    _score,
    _subset_manifest_tangent_basis,
    _trace_capture,
)


def _j_column_matrix(j_group: list[np.ndarray], *, center_columns: bool) -> np.ndarray:
    columns = []
    for jac in j_group:
        arr = np.asarray(jac, dtype=np.float64)
        flat = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)
        if center_columns:
            flat = flat - flat.mean(axis=0, keepdims=True)
        columns.append(flat)
    return np.concatenate(columns, axis=1)


def _random_basis(n_units: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _r = np.linalg.qr(rng.standard_normal((n_units, int(k))))
    return q[:, : int(k)]


def _residual_trace_fraction(covariance: np.ndarray, basis: np.ndarray) -> float:
    cov = np.asarray(covariance, dtype=np.float64)
    trace = float(np.trace(cov))
    if trace <= 1e-18:
        return np.nan
    _compact, residual = covariance_residual_noise_side(cov, basis)
    return float(np.trace(residual) / trace)


def _baseline_scores_for_group(
    *,
    group: pd.DataFrame,
    mu_all: np.ndarray,
    j_all: np.ndarray,
    expected_all: np.ndarray,
    cov_idx: np.ndarray,
    sigma_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aware_scores: list[float] = []
    blind_scores: list[float] = []
    row_ids: list[int] = []
    for row_id in group["row_id"].to_numpy(dtype=int):
        mu = mu_all[row_id][:, cov_idx]
        jac = j_all[row_id][:, cov_idx, :]
        expected = expected_all[row_id]
        aware_scores.append(_score(mu, jac, expected, None))
        blind_scores.append(_score(mu, jac, expected, sigma_target))
        row_ids.append(int(row_id))
    return (
        np.asarray(row_ids, dtype=int),
        np.asarray(aware_scores, dtype=np.float64),
        np.asarray(blind_scores, dtype=np.float64),
    )


def _closure_from_baselines(
    *,
    row_ids: np.ndarray,
    aware_scores: np.ndarray,
    blind_scores: np.ndarray,
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
    for idx, row_id in enumerate(row_ids):
        gap = float(aware_scores[idx] - blind_scores[idx])
        if not (np.isfinite(gap) and gap > eps):
            continue
        mu = mu_all[int(row_id)][:, cov_idx]
        jac = j_all[int(row_id)][:, cov_idx, :]
        expected = expected_all[int(row_id)]
        corrected = _score(mu, jac, expected, residual)
        closures.append((corrected - float(blind_scores[idx])) / gap)
        residuals.append((float(aware_scores[idx]) - corrected) / gap)
    return (
        float(np.mean(closures)) if closures else np.nan,
        float(np.mean(residuals)) if residuals else np.nan,
    )


def _mean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else np.nan


def _sd_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanstd(arr, ddof=1)) if arr.size > 1 else np.nan


def _write_markdown(path: Path, summary: pd.DataFrame, metadata: dict[str, Any]) -> None:
    lines = [
        "# Uncentered Exact-J Tangent Closure Audit",
        "",
        "## Scope",
        "",
        f"- Covopt dir: `{metadata['covopt_dir']}`.",
        f"- Source run dir: `{metadata['from_run_dir']}`.",
        f"- Matched overlap: {metadata['n_matched_units']} / {metadata['n_covariance_units']} covariance units.",
        f"- Scales: {metadata['scales']}.",
        f"- k list: {metadata['k_list']}.",
        f"- Random draws: {metadata['n_random_subspaces']}.",
        f"- Unit-shuffled draws: {metadata['n_unit_shuffles']}.",
        "",
        "## Basis Definitions",
        "",
        "- `oracle_topPC`: top eigenvectors of the covariance target.",
        "- `exactJ_uncentered`: orthonormal left singular vectors of raw cached J columns.",
        "- `exactJ_centered`: orthonormal left singular vectors of unit-centered cached J columns.",
        "- `manifest_tangent`: canonical manifest tangent basis restricted to the 116 matched units.",
        "- `random`: random orthonormal basis controls.",
        "- `unitshuffled_manifest`: row-shuffled manifest tangent controls.",
        "",
        "## Table",
        "",
    ]
    cols = [
        "family",
        "kind",
        "scale_D",
        "target",
        "basis",
        "k",
        "target_trace",
        "trace_capture",
        "closure",
        "residual_trace",
        "residual_frac",
        "n_draws",
        "trace_capture_sd",
        "closure_sd",
        "residual_trace_sd",
    ]
    available = [col for col in cols if col in summary.columns]
    lines.append("| " + " | ".join(available) + " |")
    lines.append("| " + " | ".join("---" for _ in available) + " |")
    for _, row in summary[available].iterrows():
        vals = []
        for col in available:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append("nan" if not np.isfinite(value) else f"{float(value):.5g}")
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
    p.add_argument("--k-list", default="2,5,10,20,50,100,116")
    p.add_argument("--max-rows-per-group", type=int, default=0)
    p.add_argument("--n-random-subspaces", type=int, default=1)
    p.add_argument("--n-unit-shuffles", type=int, default=1)
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

    population_csv = covopt_dir / "metadata" / "covopt_population_units.csv"
    manifest_q, matched = _subset_manifest_tangent_basis(
        population_csv=population_csv,
        canonical_manifest_csv=Path(args.canonical_manifest_csv),
        tangent_basis_npz=Path(args.tangent_basis_npz),
        basis_key=str(args.basis_key),
    )
    cov_idx = matched["cov_index"].to_numpy(dtype=int)
    n_covariance_units = int(pd.read_csv(population_csv).shape[0])
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
        exact_j_uncentered_q = orthonormalize_columns(
            _j_column_matrix(j_group, center_columns=False),
            n_rows=len(cov_idx),
        )
        exact_j_centered_q = orthonormalize_columns(
            _j_column_matrix(j_group, center_columns=True),
            n_rows=len(cov_idx),
        )
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
            baseline_row_ids, aware_scores, blind_scores = _baseline_scores_for_group(
                group=group,
                mu_all=mu_all,
                j_all=j_all,
                expected_all=expected_all,
                cov_idx=cov_idx,
                sigma_target=sigma,
            )
            for k in k_list:
                basis_items: list[tuple[str, np.ndarray, int]] = [
                    ("oracle_topPC", top_eigenvectors(sigma, int(k)), 1),
                    ("exactJ_uncentered", exact_j_uncentered_q[:, : int(k)], 1),
                    ("exactJ_centered", exact_j_centered_q[:, : int(k)], 1),
                    ("manifest_tangent", manifest_q[:, : int(k)], 1),
                ]
                for basis_name, basis, n_draws in basis_items:
                    closure, residual_frac = _closure_from_baselines(
                        row_ids=baseline_row_ids,
                        aware_scores=aware_scores,
                        blind_scores=blind_scores,
                        mu_all=mu_all,
                        j_all=j_all,
                        expected_all=expected_all,
                        cov_idx=cov_idx,
                        sigma_target=sigma,
                        basis=basis,
                        eps=float(args.eps),
                    )
                    rows.append(
                        {
                            "family": family,
                            "kind": kind,
                            "scale_D": float(scale),
                            "target": target_name,
                            "basis": basis_name,
                            "k": int(k),
                            "target_trace": target_trace,
                            "trace_capture": _trace_capture(sigma, basis),
                            "closure": closure,
                            "residual_trace": _residual_trace_fraction(sigma, basis),
                            "residual_frac": residual_frac,
                            "n_draws": int(n_draws),
                            "exact_j_n_samples": int(exact_j_samples) if target_name == "exact_j_linear_pooled" else np.nan,
                            "exact_j_n_pairs": int(exact_j_pairs) if target_name == "exact_j_linear_pooled" else np.nan,
                        }
                    )

                random_caps: list[float] = []
                random_closures: list[float] = []
                random_residual_traces: list[float] = []
                random_residual_fracs: list[float] = []
                for _draw in range(int(args.n_random_subspaces)):
                    basis = _random_basis(len(cov_idx), int(k), rng)
                    closure, residual_frac = _closure_from_baselines(
                        row_ids=baseline_row_ids,
                        aware_scores=aware_scores,
                        blind_scores=blind_scores,
                        mu_all=mu_all,
                        j_all=j_all,
                        expected_all=expected_all,
                        cov_idx=cov_idx,
                        sigma_target=sigma,
                        basis=basis,
                        eps=float(args.eps),
                    )
                    random_caps.append(_trace_capture(sigma, basis))
                    random_closures.append(closure)
                    random_residual_traces.append(_residual_trace_fraction(sigma, basis))
                    random_residual_fracs.append(residual_frac)
                rows.append(
                    {
                        "family": family,
                        "kind": kind,
                        "scale_D": float(scale),
                        "target": target_name,
                        "basis": "random",
                        "k": int(k),
                        "target_trace": target_trace,
                        "trace_capture": _mean_or_nan(random_caps),
                        "closure": _mean_or_nan(random_closures),
                        "residual_trace": _mean_or_nan(random_residual_traces),
                        "residual_frac": _mean_or_nan(random_residual_fracs),
                        "n_draws": int(args.n_random_subspaces),
                        "trace_capture_sd": _sd_or_nan(random_caps),
                        "closure_sd": _sd_or_nan(random_closures),
                        "residual_trace_sd": _sd_or_nan(random_residual_traces),
                    }
                )

                shuffled_caps: list[float] = []
                shuffled_closures: list[float] = []
                shuffled_residual_traces: list[float] = []
                shuffled_residual_fracs: list[float] = []
                for _draw in range(int(args.n_unit_shuffles)):
                    basis = manifest_q[rng.permutation(len(cov_idx)), : int(k)]
                    closure, residual_frac = _closure_from_baselines(
                        row_ids=baseline_row_ids,
                        aware_scores=aware_scores,
                        blind_scores=blind_scores,
                        mu_all=mu_all,
                        j_all=j_all,
                        expected_all=expected_all,
                        cov_idx=cov_idx,
                        sigma_target=sigma,
                        basis=basis,
                        eps=float(args.eps),
                    )
                    shuffled_caps.append(_trace_capture(sigma, basis))
                    shuffled_closures.append(closure)
                    shuffled_residual_traces.append(_residual_trace_fraction(sigma, basis))
                    shuffled_residual_fracs.append(residual_frac)
                rows.append(
                    {
                        "family": family,
                        "kind": kind,
                        "scale_D": float(scale),
                        "target": target_name,
                        "basis": "unitshuffled_manifest",
                        "k": int(k),
                        "target_trace": target_trace,
                        "trace_capture": _mean_or_nan(shuffled_caps),
                        "closure": _mean_or_nan(shuffled_closures),
                        "residual_trace": _mean_or_nan(shuffled_residual_traces),
                        "residual_frac": _mean_or_nan(shuffled_residual_fracs),
                        "n_draws": int(args.n_unit_shuffles),
                        "trace_capture_sd": _sd_or_nan(shuffled_caps),
                        "closure_sd": _sd_or_nan(shuffled_closures),
                        "residual_trace_sd": _sd_or_nan(shuffled_residual_traces),
                    }
                )

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
        "targets": ["exact_j_linear_pooled", "within_pair", "pooled_residual"],
        "basis_ordering": {
            "oracle_topPC": "target covariance eigenvalue",
            "exactJ_uncentered": "left singular vectors of raw cached J columns",
            "exactJ_centered": "left singular vectors of unit-centered cached J columns",
            "manifest_tangent": "canonical manifest basis order restricted to overlap",
        },
    }
    summary.to_csv(out_dir / "uncentered_j_tangent_closure_summary.csv", index=False)
    _write_markdown(out_dir / "uncentered_j_tangent_closure_summary.md", summary, metadata)
    (out_dir / "uncentered_j_tangent_closure_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
