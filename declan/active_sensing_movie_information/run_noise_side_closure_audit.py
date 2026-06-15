"""Noise-side-only closure audit for covariance-aware Fisher hierarchy.

This script audits whether basis-aware covariance closure is being computed
with matched semantics across oracle top-PC, manifest tangent, cache tangent,
random, and unit-shuffled bases.

For every basis ``B`` it keeps the task signal/derivative untouched and only
changes the nuisance covariance term:

    Sigma_extra_k = R Sigma_FEM R.T, where R = I - B B.T

The observation covariance used by ``covariance_fisher_by_time`` is therefore
diag(mu) + Sigma_extra_k, matching the pose-aware/pose-blind convention already
used by the covariance-optimality runner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VISIONCORE_ROOT = Path(__file__).resolve().parents[2]
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from VisionCore.subspace import directional_variance_capture
from jake.twininfo.covariance_optimality import (
    coding_covariance_from_j,
    covariance_fisher_by_time,
    covariance_residual_noise_side,
    orthonormalize_columns,
    signal_covariance_from_pair_means,
    top_eigenvectors,
)


def _scale_label(scale: float) -> str:
    return str(float(scale)).replace(".", "p").replace("-", "m")


def _cov_key(family: str, kind: str, scale: float, estimator: str) -> str:
    return f"{family}__{kind or 'all'}__D{_scale_label(scale)}__{estimator}"


def _random_basis(n_units: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _r = np.linalg.qr(rng.standard_normal((n_units, int(k))))
    return q[:, : int(k)]


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
    tangent_matrix = np.concatenate(columns, axis=1)
    return orthonormalize_columns(tangent_matrix)[:, : int(max_k)]


def _score(mu: np.ndarray, jac: np.ndarray, expected: np.ndarray, sigma_extra: np.ndarray | None) -> tuple[float, float]:
    fisher_t = covariance_fisher_by_time(mu, jac, sigma_extra)
    trace = float(np.trace(np.sum(fisher_t, axis=0)))
    expected_total = float(np.sum(expected))
    return trace, trace / max(expected_total, 1e-12)


def _sem(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _nanmean(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _synthetic_sanity() -> dict[str, float]:
    rng = np.random.default_rng(123)
    j_cols = rng.normal(size=(8, 2))
    eye_cov = np.array([[1.0, 0.2], [0.2, 0.7]], dtype=np.float64)
    sigma_fem = j_cols @ eye_cov @ j_cols.T
    _compact, residual = covariance_residual_noise_side(sigma_fem, j_cols)
    mu = np.full((5, 8), 2.0, dtype=np.float64)
    jac = np.broadcast_to(j_cols[None, :, :], (5, 8, 2)).copy()
    aware = _score(mu, jac, np.ones(5), None)[1]
    blind = _score(mu, jac, np.ones(5), sigma_fem)[1]
    corrected = _score(mu, jac, np.ones(5), residual)[1]
    pose_gap = aware - blind
    closure = (corrected - blind) / pose_gap if pose_gap > 1e-12 else np.nan
    return {
        "synthetic_sigma_trace": float(np.trace(sigma_fem)),
        "synthetic_residual_trace_frac": float(np.trace(residual) / max(np.trace(sigma_fem), 1e-12)),
        "synthetic_pose_gap": float(pose_gap),
        "synthetic_closure": float(closure),
    }


def _summarize(row_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, grp in row_scores.groupby(["family", "kind", "scale_D", "k", "basis_label"], sort=True):
        family, kind, scale, k, basis_label = key
        rows.append({
            "family": family,
            "kind": kind,
            "scale_D": float(scale),
            "k": int(k),
            "basis_label": str(basis_label),
            "n_rows": int(len(grp)),
            "n_draws": int(grp["draw"].nunique()),
            "sigma_trace_mean": _nanmean(grp["sigma_fem_trace"]),
            "pose_gap_mean": _nanmean(grp["pose_gap"]),
            "pose_gap_min": float(np.nanmin(grp["pose_gap"])),
            "closure_mean": _nanmean(grp["closure"]),
            "closure_sem": _sem(grp["closure"]),
            "residual_frac_mean": _nanmean(grp["residual_frac"]),
            "residual_trace_frac_mean": _nanmean(grp["residual_trace_frac"]),
            "signal_frac_in_basis_mean": _nanmean(grp["signal_frac_in_basis"]),
            "pair_mean_signal_frac_in_basis_mean": _nanmean(grp["pair_mean_signal_frac_in_basis"]),
            "score_per_spike_mean": _nanmean(grp["score_per_spike"]),
        })
    return pd.DataFrame(rows)


def _decision_table(summary: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, grp in summary.groupby(["family", "kind", "scale_D", "k"], sort=True):
        family, kind, scale, k = key
        lookup = {str(row["basis_label"]): row for _, row in grp.iterrows()}

        def get(label: str, col: str) -> float:
            row = lookup.get(label)
            return float("nan") if row is None else float(row[col])

        pose_gap = get("oracle_topPC", "pose_gap_mean")
        if not np.isfinite(pose_gap):
            pose_gap = get("manifest_tangent", "pose_gap_mean")
        top_closure = get("oracle_topPC", "closure_mean")
        manifest_closure = get("manifest_tangent", "closure_mean")
        cache_closure = get("cache_tangent", "closure_mean")
        random_closure = get("random", "closure_mean")
        shuffled_closure = get("unitshuffled_manifest_tangent", "closure_mean")
        top_resid = get("oracle_topPC", "residual_trace_frac_mean")
        manifest_resid = get("manifest_tangent", "residual_trace_frac_mean")
        cache_resid = get("cache_tangent", "residual_trace_frac_mean")
        if not np.isfinite(pose_gap) or pose_gap <= float(eps):
            label = "gap_too_small_to_interpret"
        elif np.isfinite(cache_resid) and cache_resid <= 0.1 and np.isfinite(cache_closure) and cache_closure < 0.5:
            label = "invalid_signal_projection_confound"
        elif np.isfinite(cache_resid) and cache_resid > 0.5 and np.isfinite(cache_closure) and cache_closure < 0.5:
            label = "invalid_sigma_provenance_mismatch"
        elif np.isfinite(manifest_resid) and manifest_resid > 0.5 and np.isfinite(top_closure) and top_closure > 0.8 and np.isfinite(manifest_closure) and manifest_closure < 0.5:
            label = "valid_low_rank_only"
        elif np.isfinite(manifest_closure) and np.isfinite(random_closure) and np.isfinite(shuffled_closure) and manifest_closure > random_closure + 0.1 and manifest_closure > shuffled_closure + 0.05:
            label = "valid_geometry_specific"
        elif np.isfinite(manifest_closure) and np.isfinite(random_closure) and manifest_closure > random_closure + 0.05:
            label = "valid_partial_geometry"
        else:
            label = "valid_low_rank_only"

        rows.append({
            "family": family,
            "kind": kind,
            "scale_D": float(scale),
            "k": int(k),
            "pose_gap": pose_gap,
            "closure_oracle_topPC": top_closure,
            "closure_manifest_tangent": manifest_closure,
            "closure_cache_tangent": cache_closure,
            "closure_random": random_closure,
            "closure_unitshuffled_manifest_tangent": shuffled_closure,
            "residual_trace_oracle_topPC": top_resid,
            "residual_trace_manifest_tangent": manifest_resid,
            "residual_trace_cache_tangent": cache_resid,
            "signal_frac_manifest_tangent": get("manifest_tangent", "signal_frac_in_basis_mean"),
            "signal_frac_cache_tangent": get("cache_tangent", "signal_frac_in_basis_mean"),
            "decision_label": label,
        })
    return pd.DataFrame(rows)


def _plot(decision: pd.DataFrame, out: Path) -> None:
    positive = decision[decision["scale_D"] > 0].copy()
    if positive.empty:
        return
    families = sorted(positive["family"].unique())
    kinds = sorted(positive["kind"].unique())
    fig, axes = plt.subplots(len(families), len(kinds), figsize=(11, 11), sharex=True, sharey=True)
    if len(families) == 1 and len(kinds) == 1:
        axes = np.array([[axes]])
    elif len(families) == 1:
        axes = np.array([axes])
    elif len(kinds) == 1:
        axes = np.array([[ax] for ax in axes])
    labels = [
        ("closure_manifest_tangent", "manifest tangent"),
        ("closure_cache_tangent", "cache tangent"),
        ("closure_random", "random"),
        ("closure_unitshuffled_manifest_tangent", "unit-shuffled"),
        ("closure_oracle_topPC", "oracle top-PC"),
    ]
    for i, family in enumerate(families):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            sub = positive[(positive["family"] == family) & (positive["kind"] == kind)].sort_values("k")
            for col, label in labels:
                ax.plot(sub["k"], sub[col], marker="o", label=label)
            ax.set_title(f"{family}\n{kind}", fontsize=9)
            ax.set_xlabel("k")
            ax.set_ylabel("gap closure")
            ax.axhline(0, color="0.7", linewidth=1)
            ax.axhline(1, color="0.7", linewidth=1, linestyle=":")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center right")
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _write_md(path: Path, decision: pd.DataFrame, metadata: dict[str, Any], sanity: dict[str, Any]) -> None:
    lines = [
        "# Noise-Side Closure Audit",
        "",
        "## Scope",
        "",
        f"- Covariance estimator: `{metadata['covariance_estimator']}`.",
        f"- Matched overlap: {metadata['n_matched_units']} / {metadata['n_covariance_units']} covariance units.",
        f"- D scales: {metadata['scales']}.",
        f"- k list: {metadata['k_list']}.",
        "- All bases use the same `covariance_residual_noise_side` function: `R Sigma_FEM R.T`.",
        "",
        "## Sanity Checks",
        "",
    ]
    for key, value in sanity.items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: {value:.6g}")
        else:
            lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Decision Counts", ""])
    for label, count in decision["decision_label"].value_counts().sort_index().items():
        lines.append(f"- {label}: {int(count)}")
    cols = [
        "family",
        "kind",
        "scale_D",
        "k",
        "closure_oracle_topPC",
        "closure_manifest_tangent",
        "closure_cache_tangent",
        "closure_random",
        "closure_unitshuffled_manifest_tangent",
        "residual_trace_oracle_topPC",
        "residual_trace_manifest_tangent",
        "residual_trace_cache_tangent",
        "decision_label",
    ]
    lines.extend(["", "## Table", "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"])
    for _, row in decision[cols].iterrows():
        vals = []
        for col in cols:
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
    p.add_argument("--scales", default="0,1")
    p.add_argument("--k-list", default="2,5,10,20")
    p.add_argument("--covariance-estimator", default="pooled_residual", choices=("pooled_residual", "within_pair"))
    p.add_argument("--n-random-subspaces", type=int, default=5)
    p.add_argument("--n-unit-shuffles", type=int, default=5)
    p.add_argument("--max-rows-per-group", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=1e-9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    covopt_dir = Path(args.covopt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    scales = tuple(float(x) for x in str(args.scales).split(",") if x.strip())
    k_list = tuple(int(float(x)) for x in str(args.k_list).split(",") if x.strip())

    manifest_q, matched = _subset_manifest_tangent_basis(
        population_csv=covopt_dir / "metadata" / "covopt_population_units.csv",
        canonical_manifest_csv=Path(args.canonical_manifest_csv),
        tangent_basis_npz=Path(args.tangent_basis_npz),
        basis_key=str(args.basis_key),
    )
    cov_idx = matched["cov_index"].to_numpy(dtype=int)
    population_rows = pd.read_csv(covopt_dir / "metadata" / "covopt_population_units.csv").shape[0]
    records = pd.read_csv(covopt_dir / "metadata" / "covopt_rate_records.csv")
    records = records[records["scale_D"].astype(float).isin(scales)].copy()
    with np.load(covopt_dir / "cache" / "covopt_mu_j.npz") as z:
        mu_all = np.asarray(z["mu"], dtype=np.float32)
        j_all = np.asarray(z["J"], dtype=np.float32)
        expected_all = np.asarray(z["expected_spikes_t"], dtype=np.float32)
    with np.load(covopt_dir / "cache" / "covopt_covariances.npz") as z:
        covs = {name: np.asarray(z[name], dtype=np.float64) for name in z.files}

    print(
        f"Running noise-side closure audit on {len(cov_idx)} matched units, "
        f"scales={scales}, k={k_list}, estimator={args.covariance_estimator}",
        flush=True,
    )

    score_rows: list[dict[str, Any]] = []
    sanity: dict[str, Any] = _synthetic_sanity()
    d0_max_sigma_trace = 0.0
    d0_max_abs_aware_blind = 0.0
    d0_max_abs_aware_corrected = 0.0

    group_items = list(records.groupby(["family", "kind", "scale_D"], sort=True))
    for group_i, ((family, kind, scale), group) in enumerate(group_items, start=1):
        if int(args.max_rows_per_group) > 0 and len(group) > int(args.max_rows_per_group):
            group = group.sample(
                n=int(args.max_rows_per_group),
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).sort_values("row_id")
        key = _cov_key(str(family), str(kind), float(scale), str(args.covariance_estimator))
        sigma_full = covs[key]
        sigma_fem = sigma_full[np.ix_(cov_idx, cov_idx)]
        sigma_trace = float(np.trace(sigma_fem))
        row_ids = group["row_id"].to_numpy(dtype=int)
        mu_group = [mu_all[row_id][:, cov_idx] for row_id in row_ids]
        j_group = [j_all[row_id][:, cov_idx, :] for row_id in row_ids]
        coding = coding_covariance_from_j(j_group)
        pair_signal = signal_covariance_from_pair_means(mu_group)
        cache_q = _cache_tangent_basis_for_group(j_group, max_k=max(k_list))
        print(
            f"[{group_i}/{len(group_items)}] {family} {kind} D={float(scale):g} "
            f"rows={len(group)} trace={sigma_trace:.6g}",
            flush=True,
        )
        base_scores: dict[int, tuple[float, float]] = {}
        blind_scores: dict[int, tuple[float, float]] = {}
        for row_id in row_ids:
            mu = mu_all[row_id][:, cov_idx]
            jac = j_all[row_id][:, cov_idx, :]
            expected = expected_all[row_id]
            base_scores[int(row_id)] = _score(mu, jac, expected, None)
            blind_scores[int(row_id)] = _score(mu, jac, expected, sigma_fem)

        for k in k_list:
            basis_defs: list[tuple[str, int, np.ndarray]] = [
                ("oracle_topPC", 0, top_eigenvectors(sigma_fem, int(k))),
                ("manifest_tangent", 0, manifest_q[:, : int(k)]),
                ("cache_tangent", 0, cache_q[:, : int(k)]),
            ]
            for draw in range(int(args.n_unit_shuffles)):
                basis_defs.append((
                    "unitshuffled_manifest_tangent",
                    draw,
                    manifest_q[rng.permutation(len(cov_idx)), : int(k)],
                ))
            for draw in range(int(args.n_random_subspaces)):
                basis_defs.append(("random", draw, _random_basis(len(cov_idx), int(k), rng)))
            for basis_label, draw, basis in basis_defs:
                basis_q = orthonormalize_columns(basis, n_rows=len(cov_idx))
                _compact, sigma_residual = covariance_residual_noise_side(sigma_fem, basis_q)
                residual_trace_frac = float(np.trace(sigma_residual) / max(sigma_trace, 1e-12))
                signal_frac = float(directional_variance_capture(coding, basis_q))
                pair_signal_frac = float(directional_variance_capture(pair_signal, basis_q))
                for _, rec in group.iterrows():
                    row_id = int(rec["row_id"])
                    aware_trace, aware_per_spike = base_scores[row_id]
                    blind_trace, blind_per_spike = blind_scores[row_id]
                    score_trace, score_per_spike = _score(
                        mu_all[row_id][:, cov_idx],
                        j_all[row_id][:, cov_idx, :],
                        expected_all[row_id],
                        sigma_residual,
                    )
                    pose_gap = aware_per_spike - blind_per_spike
                    valid = bool(np.isfinite(pose_gap) and pose_gap > float(args.eps))
                    closure = (score_per_spike - blind_per_spike) / pose_gap if valid else np.nan
                    residual_frac = (aware_per_spike - score_per_spike) / pose_gap if valid else np.nan
                    if float(scale) == 0.0:
                        d0_max_sigma_trace = max(d0_max_sigma_trace, abs(sigma_trace))
                        d0_max_abs_aware_blind = max(d0_max_abs_aware_blind, abs(aware_per_spike - blind_per_spike))
                        d0_max_abs_aware_corrected = max(
                            d0_max_abs_aware_corrected,
                            abs(aware_per_spike - score_per_spike),
                        )
                    score_rows.append({
                        "row_id": row_id,
                        "example_id": rec["example_id"],
                        "kind": rec["kind"],
                        "image_index": int(rec["image_index"]),
                        "crop_rank": int(rec["crop_rank"]),
                        "family": rec["family"],
                        "scale_D": float(rec["scale_D"]),
                        "covariance_estimator": str(args.covariance_estimator),
                        "k": int(k),
                        "basis_label": basis_label,
                        "draw": int(draw),
                        "sigma_fem_trace": sigma_trace,
                        "residual_trace_frac": residual_trace_frac,
                        "signal_frac_in_basis": signal_frac,
                        "pair_mean_signal_frac_in_basis": pair_signal_frac,
                        "aware_per_spike": aware_per_spike,
                        "blind_per_spike": blind_per_spike,
                        "score_per_spike": score_per_spike,
                        "pose_gap": pose_gap,
                        "closure": closure,
                        "residual_frac": residual_frac,
                        "valid_pose_gap": valid,
                    })

    score_df = pd.DataFrame(score_rows)
    summary = _summarize(score_df)
    decision = _decision_table(summary, eps=float(args.eps))
    sanity.update({
        "d0_max_sigma_trace": float(d0_max_sigma_trace),
        "d0_max_abs_F_PA_minus_F_PB_per_spike": float(d0_max_abs_aware_blind),
        "d0_max_abs_F_PA_minus_F_k_per_spike": float(d0_max_abs_aware_corrected),
        "matched_semantics_function": "jake.twininfo.covariance_optimality.covariance_residual_noise_side",
    })
    metadata = {
        "covopt_dir": str(covopt_dir),
        "covariance_estimator": str(args.covariance_estimator),
        "tangent_basis_npz": str(args.tangent_basis_npz),
        "canonical_manifest_csv": str(args.canonical_manifest_csv),
        "basis_key": str(args.basis_key),
        "n_covariance_units": int(population_rows),
        "n_matched_units": int(len(cov_idx)),
        "scales": list(scales),
        "k_list": list(k_list),
        "n_random_subspaces": int(args.n_random_subspaces),
        "n_unit_shuffles": int(args.n_unit_shuffles),
        "max_rows_per_group": int(args.max_rows_per_group),
        "n_row_scores": int(len(score_df)),
        "n_summary_rows": int(len(summary)),
        "n_decision_rows": int(len(decision)),
        "note": "Noise-side-only audit; task derivatives/responses are not projected.",
    }

    score_df.to_csv(out_dir / "noise_side_closure_row_scores.csv", index=False)
    summary.to_csv(out_dir / "noise_side_closure_summary.csv", index=False)
    decision.to_csv(out_dir / "noise_side_closure_decision_table.csv", index=False)
    _write_md(out_dir / "noise_side_closure_decision_table.md", decision, metadata, sanity)
    _plot(decision, out_dir / "figures" / "noise_side_closure_vs_k.png")
    (out_dir / "noise_side_closure_sanity_checks.json").write_text(
        json.dumps(sanity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "noise_side_closure_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metadata": metadata, "sanity": sanity}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
