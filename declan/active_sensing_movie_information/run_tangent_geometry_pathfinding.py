"""Exact D-scale Fisher pilot for an independent tangent basis on matched units.

This pathfinding script reuses a completed covariance-optimality ``mu/J`` cache
and projects it to the overlap between that population and a 756D Figure 4/TFTS
tangent-basis manifest.  It compares:

- pose-aware covariance Fisher
- pose-blind covariance Fisher
- oracle top movement-PC residual covariance
- independent tangent-basis residual covariance
- same-cache Jacobian tangent-basis residual covariance
- unit-shuffled tangent basis
- random orthonormal bases

The intended use is a cheap specificity pilot before rebuilding a fully matched
756-channel covariance-optimality cache.
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
    covariance_residual_after_subspace,
    signal_covariance_from_pair_means,
    top_eigenvectors,
)


def _scale_label(scale: float) -> str:
    return str(float(scale)).replace(".", "p").replace("-", "m")


def _cov_key(family: str, kind: str, scale: float) -> str:
    return f"{family}__{kind or 'all'}__D{_scale_label(scale)}__pooled_residual"


def _random_basis(n_units: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _r = np.linalg.qr(rng.standard_normal((n_units, int(k))))
    return q[:, : int(k)]


def _subset_tangent_basis(
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
    canonical_idx = matched["canonical_unit_index"].to_numpy(dtype=int)
    basis_overlap = basis_756[canonical_idx]
    basis_q, _r = np.linalg.qr(basis_overlap)
    return basis_q, matched


def _cache_tangent_basis(
    *,
    j_all: np.ndarray,
    row_ids: np.ndarray,
    cov_idx: np.ndarray,
    max_k: int,
) -> np.ndarray:
    columns = []
    for row_id in row_ids:
        jac = np.asarray(j_all[int(row_id)][:, cov_idx, :], dtype=np.float64)
        flat = jac.transpose(1, 0, 2).reshape(len(cov_idx), -1)
        flat = flat - flat.mean(axis=0, keepdims=True)
        columns.append(flat)
    tangent_matrix = np.concatenate(columns, axis=1)
    u, _s, _vt = np.linalg.svd(tangent_matrix, full_matrices=False)
    return u[:, : min(int(max_k), u.shape[1])]


def _score(mu: np.ndarray, jac: np.ndarray, expected: np.ndarray, cov: np.ndarray | None) -> tuple[float, float]:
    f_by_time = covariance_fisher_by_time(mu, jac, cov)
    trace = float(np.trace(np.sum(f_by_time, axis=0)))
    expected_total = float(np.sum(expected))
    return trace, trace / max(expected_total, 1e-12)


def _basis_diagnostics(
    *,
    family: str,
    kind: str,
    scale: float,
    k: int,
    basis_label: str,
    basis: np.ndarray,
    sigma: np.ndarray,
    coding: np.ndarray,
    signal: np.ndarray,
) -> dict[str, Any]:
    compact, residual = covariance_residual_after_subspace(sigma, basis)
    sigma_trace = float(np.trace(sigma))
    return {
        "family": family,
        "kind": kind,
        "scale_D": float(scale),
        "k": int(k),
        "basis_label": basis_label,
        "sigma_trace": sigma_trace,
        "compact_covariance_trace": float(np.trace(compact)),
        "residual_covariance_trace": float(np.trace(residual)),
        "nuisance_variance_removed_fraction": float(np.trace(compact) / max(sigma_trace, 1e-12)),
        "nuisance_variance_remaining_fraction": float(np.trace(residual) / max(sigma_trace, 1e-12)),
        "signal_fraction_in_basis": float(directional_variance_capture(signal, basis)),
        "coding_fraction_in_basis": float(directional_variance_capture(coding, basis)),
    }


def _sem(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _summarize(row_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, grp in row_scores.groupby(["family", "kind", "scale_D", "k", "basis_label"], sort=True):
        family, kind, scale, k, basis_label = key
        rows.append({
            "family": family,
            "kind": kind,
            "scale_D": float(scale),
            "k": int(k),
            "basis_label": basis_label,
            "n_rows": int(len(grp)),
            "n_images": int(grp["image_index"].nunique()),
            "n_draws": int(grp["draw"].nunique()),
            "pose_gap_mean": float(np.nanmean(grp["pose_gap"])),
            "closure_mean": float(np.nanmean(grp["closure"])),
            "closure_sem": _sem(grp["closure"]),
            "residual_frac_mean": float(np.nanmean(grp["residual_frac"])),
            "score_per_spike_mean": float(np.nanmean(grp["score_per_spike"])),
            "score_per_spike_sem": _sem(grp["score_per_spike"]),
        })
    return pd.DataFrame(rows)


def _decision_table(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, grp in summary.groupby(["family", "kind", "scale_D", "k"], sort=True):
        family, kind, scale, k = key
        lookup = {str(row["basis_label"]): row for _, row in grp.iterrows()}
        tangent = lookup.get("tangent")
        if tangent is None:
            continue
        def get(label: str, col: str) -> float:
            row = lookup.get(label)
            return float("nan") if row is None else float(row[col])
        diag = diagnostics[
            (diagnostics["family"] == family)
            & (diagnostics["kind"] == kind)
            & np.isclose(diagnostics["scale_D"].astype(float), float(scale))
            & (diagnostics["k"].astype(int) == int(k))
            & (diagnostics["basis_label"] == "tangent")
        ]
        signal_fraction = float(diag["signal_fraction_in_basis"].iloc[0]) if not diag.empty else np.nan
        coding_fraction = float(diag["coding_fraction_in_basis"].iloc[0]) if not diag.empty else np.nan
        closure_tangent = float(tangent["closure_mean"])
        closure_random = get("random", "closure_mean")
        closure_unit = get("unitshuffled_tangent", "closure_mean")
        closure_top = get("oracle_topPC", "closure_mean")
        closure_cache_tangent = get("cache_tangent", "closure_mean")
        if not np.isfinite(float(tangent["pose_gap_mean"])) or float(tangent["pose_gap_mean"]) <= 1e-6:
            label = "gap_too_small_to_interpret"
        elif np.isfinite(closure_top) and closure_top - closure_tangent > 0.25:
            label = "topPC_much_better_than_tangent"
        elif np.isfinite(closure_random) and np.isfinite(closure_unit) and closure_tangent > closure_random + 0.05 and closure_tangent > closure_unit + 0.05:
            label = "partial_tangent_specificity"
        elif np.isfinite(closure_random) and closure_tangent <= closure_random + 0.05:
            label = "tangent_near_random"
        else:
            label = "inconclusive_pathfinding"
        rows.append({
            "family": family,
            "kind": kind,
            "scale_D": float(scale),
            "k": int(k),
            "pose_gap": float(tangent["pose_gap_mean"]),
            "closure_tangent": closure_tangent,
            "closure_random": closure_random,
            "closure_unitshuffled_tangent": closure_unit,
            "closure_cache_tangent": closure_cache_tangent,
            "closure_oracle_topPC": closure_top,
            "tangent_minus_random": closure_tangent - closure_random,
            "tangent_minus_unitshuffled": closure_tangent - closure_unit,
            "cache_tangent_minus_random": closure_cache_tangent - closure_random,
            "cache_tangent_minus_topPC": closure_cache_tangent - closure_top,
            "tangent_minus_topPC": closure_tangent - closure_top,
            "signal_fraction_in_tangent": signal_fraction,
            "coding_fraction_in_tangent": coding_fraction,
            "decision_label": label,
        })
    return pd.DataFrame(rows)


def _write_md(path: Path, decision: pd.DataFrame, metadata: dict[str, Any]) -> None:
    lines = [
        "# Tangent Geometry Pathfinding",
        "",
        "## Scope",
        "",
        f"- Matched overlap: {metadata['n_matched_units']} / {metadata['n_covariance_units']} covariance units.",
        f"- D scales: {metadata['scales']}",
        f"- Random draws: {metadata['n_random_subspaces']}; unit-shuffle draws: {metadata['n_unit_shuffles']}.",
        "- This is a partial-overlap pathfinding test, not a final matched 756-channel claim.",
        "",
        "## Decision Counts",
        "",
    ]
    for label, count in decision["decision_label"].value_counts().sort_index().items():
        lines.append(f"- {label}: {int(count)}")
    lines.extend(["", "## Table", ""])
    cols = [
        "family",
        "kind",
        "k",
        "pose_gap",
        "closure_tangent",
        "closure_random",
        "closure_unitshuffled_tangent",
        "closure_cache_tangent",
        "closure_oracle_topPC",
        "tangent_minus_random",
        "tangent_minus_unitshuffled",
        "cache_tangent_minus_random",
        "cache_tangent_minus_topPC",
        "tangent_minus_topPC",
        "signal_fraction_in_tangent",
        "decision_label",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
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


def _plot_closure(decision: pd.DataFrame, out: Path) -> None:
    families = sorted(decision["family"].unique())
    kinds = sorted(decision["kind"].unique())
    fig, axes = plt.subplots(len(families), len(kinds), figsize=(10, 11), sharex=True, sharey=True)
    labels = [
        ("closure_tangent", "tangent"),
        ("closure_cache_tangent", "same-cache tangent"),
        ("closure_random", "random"),
        ("closure_unitshuffled_tangent", "unit-shuffled"),
        ("closure_oracle_topPC", "oracle top-PC"),
    ]
    for i, family in enumerate(families):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            sub = decision[(decision["family"] == family) & (decision["kind"] == kind)].sort_values("k")
            for col, label in labels:
                if col not in sub.columns or sub[col].isna().all():
                    continue
                ax.plot(sub["k"], sub[col], marker="o", label=label)
            ax.set_title(f"{family}\n{kind}", fontsize=9)
            ax.set_xlabel("k")
            ax.set_ylabel("Gap closure")
            ax.axhline(0, color="0.7", linewidth=1)
            ax.axhline(1, color="0.7", linewidth=1, linestyle=":")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center right")
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--covopt-dir", type=Path, required=True)
    p.add_argument("--tangent-basis-npz", type=Path, required=True)
    p.add_argument("--canonical-manifest-csv", type=Path, required=True)
    p.add_argument("--basis-key", default="basis_delta_0p25")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scales", default="1")
    p.add_argument("--k-list", default="2,5,10,20")
    p.add_argument("--n-random-subspaces", type=int, default=5)
    p.add_argument("--n-unit-shuffles", type=int, default=5)
    p.add_argument("--max-rows-per-group", type=int, default=0)
    p.add_argument("--include-cache-tangent", action="store_true")
    p.add_argument("--seed", type=int, default=0)
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

    tangent_q, matched = _subset_tangent_basis(
        population_csv=covopt_dir / "metadata" / "covopt_population_units.csv",
        canonical_manifest_csv=Path(args.canonical_manifest_csv),
        tangent_basis_npz=Path(args.tangent_basis_npz),
        basis_key=str(args.basis_key),
    )
    cov_idx = matched["cov_index"].to_numpy(dtype=int)

    records = pd.read_csv(covopt_dir / "metadata" / "covopt_rate_records.csv")
    with np.load(covopt_dir / "cache" / "covopt_mu_j.npz") as z:
        mu_all = np.asarray(z["mu"], dtype=np.float32)
        j_all = np.asarray(z["J"], dtype=np.float32)
        expected_all = np.asarray(z["expected_spikes_t"], dtype=np.float32)
    with np.load(covopt_dir / "cache" / "covopt_covariances.npz") as z:
        covs = {name: np.asarray(z[name], dtype=np.float64) for name in z.files}

    records = records[records["scale_D"].astype(float).isin(scales)].copy()
    group_items = list(records.groupby(["family", "kind", "scale_D"], sort=True))
    cache_tangent_q = None
    if bool(args.include_cache_tangent):
        cache_tangent_q = _cache_tangent_basis(
            j_all=j_all,
            row_ids=records["row_id"].drop_duplicates().to_numpy(dtype=int),
            cov_idx=cov_idx,
            max_k=max(k_list),
        )
    print(
        f"Running tangent pathfinding on {len(cov_idx)} matched units, "
        f"{len(group_items)} groups, k={k_list}",
        flush=True,
    )
    if cache_tangent_q is not None:
        print(
            f"Included same-cache Jacobian tangent basis with rank {cache_tangent_q.shape[1]}",
            flush=True,
        )

    score_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for group_i, ((family, kind, scale), group) in enumerate(group_items, start=1):
        if int(args.max_rows_per_group) > 0 and len(group) > int(args.max_rows_per_group):
            group = group.sample(
                n=int(args.max_rows_per_group),
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).sort_values("row_id")
        sigma_full = covs[_cov_key(str(family), str(kind), float(scale))]
        sigma = sigma_full[np.ix_(cov_idx, cov_idx)]
        group_ids = group["row_id"].to_numpy(dtype=int)
        mu_group = [mu_all[row_id][:, cov_idx] for row_id in group_ids]
        j_group = [j_all[row_id][:, cov_idx, :] for row_id in group_ids]
        coding = coding_covariance_from_j(j_group)
        signal = signal_covariance_from_pair_means(mu_group)
        print(
            f"[{group_i}/{len(group_items)}] {family} {kind} D={float(scale):g} rows={len(group)}",
            flush=True,
        )
        base_scores: dict[int, tuple[float, float]] = {}
        blind_scores: dict[int, tuple[float, float]] = {}
        for row_id in group_ids:
            mu = mu_all[row_id][:, cov_idx]
            jac = j_all[row_id][:, cov_idx, :]
            expected = expected_all[row_id]
            base_scores[int(row_id)] = _score(mu, jac, expected, None)
            blind_scores[int(row_id)] = _score(mu, jac, expected, sigma)
        for k in k_list:
            print(f"  k={k}", flush=True)
            basis_defs: list[tuple[str, int, np.ndarray]] = [
                ("oracle_topPC", 0, top_eigenvectors(sigma, int(k))),
                ("tangent", 0, tangent_q[:, : int(k)]),
            ]
            if cache_tangent_q is not None:
                basis_defs.append(("cache_tangent", 0, cache_tangent_q[:, : int(k)]))
            for draw in range(int(args.n_unit_shuffles)):
                basis_defs.append(("unitshuffled_tangent", draw, tangent_q[rng.permutation(len(cov_idx)), : int(k)]))
            for draw in range(int(args.n_random_subspaces)):
                basis_defs.append(("random", draw, _random_basis(len(cov_idx), int(k), rng)))
            for basis_label, draw, basis in basis_defs:
                compact, residual = covariance_residual_after_subspace(sigma, basis)
                diag_rows.append(_basis_diagnostics(
                    family=str(family),
                    kind=str(kind),
                    scale=float(scale),
                    k=int(k),
                    basis_label=str(basis_label),
                    basis=basis,
                    sigma=sigma,
                    coding=coding,
                    signal=signal,
                ) | {"draw": int(draw)})
                for _, rec in group.iterrows():
                    row_id = int(rec["row_id"])
                    aware_trace, aware_per_spike = base_scores[row_id]
                    blind_trace, blind_per_spike = blind_scores[row_id]
                    score_trace, score_per_spike = _score(
                        mu_all[row_id][:, cov_idx],
                        j_all[row_id][:, cov_idx, :],
                        expected_all[row_id],
                        residual,
                    )
                    pose_gap = aware_per_spike - blind_per_spike
                    valid = bool(np.isfinite(pose_gap) and pose_gap > 1e-9)
                    closure = (score_per_spike - blind_per_spike) / pose_gap if valid else np.nan
                    score_rows.append({
                        "row_id": row_id,
                        "example_id": rec["example_id"],
                        "kind": rec["kind"],
                        "image_index": int(rec["image_index"]),
                        "crop_rank": int(rec["crop_rank"]),
                        "family": rec["family"],
                        "scale_D": float(rec["scale_D"]),
                        "k": int(k),
                        "basis_label": basis_label,
                        "draw": int(draw),
                        "aware_per_spike": aware_per_spike,
                        "blind_per_spike": blind_per_spike,
                        "score_per_spike": score_per_spike,
                        "pose_gap": pose_gap,
                        "closure": closure,
                        "residual_frac": 1.0 - closure if np.isfinite(closure) else np.nan,
                        "valid_pose_gap": valid,
                    })

    score_df = pd.DataFrame(score_rows)
    diag_df = pd.DataFrame(diag_rows)
    summary = _summarize(score_df)
    diag_summary = (
        diag_df.groupby(["family", "kind", "scale_D", "k", "basis_label"], as_index=False)
        .agg({
            "nuisance_variance_removed_fraction": "mean",
            "signal_fraction_in_basis": "mean",
            "coding_fraction_in_basis": "mean",
        })
    )
    decision = _decision_table(summary, diag_summary)

    score_df.to_csv(out_dir / "tangent_geometry_pathfinding_row_scores.csv", index=False)
    summary.to_csv(out_dir / "tangent_geometry_pathfinding_summary.csv", index=False)
    diag_df.to_csv(out_dir / "tangent_geometry_pathfinding_basis_diagnostics.csv", index=False)
    diag_summary.to_csv(out_dir / "tangent_geometry_pathfinding_basis_summary.csv", index=False)
    decision.to_csv(out_dir / "tangent_geometry_pathfinding_decision_table.csv", index=False)

    metadata = {
        "covopt_dir": str(covopt_dir),
        "tangent_basis_npz": str(args.tangent_basis_npz),
        "canonical_manifest_csv": str(args.canonical_manifest_csv),
        "basis_key": str(args.basis_key),
        "n_covariance_units": int(pd.read_csv(covopt_dir / "metadata" / "covopt_population_units.csv").shape[0]),
        "n_matched_units": int(len(cov_idx)),
        "scales": list(scales),
        "k_list": list(k_list),
        "n_random_subspaces": int(args.n_random_subspaces),
        "n_unit_shuffles": int(args.n_unit_shuffles),
        "include_cache_tangent": bool(args.include_cache_tangent),
        "max_rows_per_group": int(args.max_rows_per_group),
        "n_row_scores": int(len(score_df)),
        "n_summary_rows": int(len(summary)),
        "n_decision_rows": int(len(decision)),
        "note": "Partial-overlap pathfinding only; final claim requires matched tangent/covariance response space.",
    }
    _write_md(out_dir / "tangent_geometry_pathfinding_decision_table.md", decision, metadata)
    _plot_closure(decision, out_dir / "figures" / "tangent_geometry_closure_vs_k.png")
    (out_dir / "tangent_geometry_pathfinding_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
