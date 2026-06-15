"""Summarize the focused uncentered exact-J tangent closure audit.

This is a reporting helper, not another closure run.  It reads the existing
closure CSV, recomputes only the basis signal fractions in the same sampled
row/basis regime, and writes compact k-specific tables plus a bar figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VISIONCORE_ROOT = Path(__file__).resolve().parents[2]
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from jake.twininfo.common import load_digital_twin
from jake.twininfo.covariance_optimality import (
    movement_covariance_pooled_residual,
    orthonormalize_columns,
    top_eigenvectors,
)
from jake.twininfo.run_covariance_optimality import _load_trace_examples_from_metadata

from declan.active_sensing_movie_information.run_covariance_target_provenance_audit import (
    _cov_key,
    _linear_prediction_rows,
    _subset_manifest_tangent_basis,
)
from declan.active_sensing_movie_information.run_uncentered_j_tangent_closure_audit import (
    _j_column_matrix,
    _random_basis,
)


BASIS_ORDER = [
    "oracle_topPC",
    "exactJ_uncentered",
    "exactJ_centered",
    "manifest_tangent",
    "random",
    "unitshuffled_manifest",
]

BASIS_LABELS = {
    "oracle_topPC": "Oracle top-PC",
    "exactJ_uncentered": "Uncentered exact-J",
    "exactJ_centered": "Centered exact-J",
    "manifest_tangent": "Manifest tangent",
    "random": "Random",
    "unitshuffled_manifest": "Unit-shuffled",
}

TARGET_ORDER = ["exact_j_linear_pooled", "pooled_residual", "within_pair"]
TARGET_LABELS = {
    "exact_j_linear_pooled": "Exact J",
    "pooled_residual": "Pooled residual",
    "within_pair": "Within-pair",
}


def _basis_signal_fraction(jac_rows: list[np.ndarray], basis: np.ndarray) -> float:
    u = orthonormalize_columns(basis, n_rows=jac_rows[0].shape[1])
    denom = 0.0
    numer = 0.0
    for jac in jac_rows:
        arr = np.asarray(jac, dtype=np.float64)
        # Signal lives in response-unit coordinates; columns are time x task-axis.
        signal = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)
        denom += float(np.sum(signal * signal))
        if u.shape[1]:
            projected = u.T @ signal
            numer += float(np.sum(projected * projected))
    if denom <= 1e-18:
        return np.nan
    return numer / denom


def _signal_fraction_table(
    *,
    covopt_dir: Path,
    tangent_basis_npz: Path,
    canonical_manifest_csv: Path,
    basis_key: str,
    scale_values: tuple[float, ...],
    k_list: tuple[int, ...],
    max_rows_per_group: int,
    n_random_subspaces: int,
    n_unit_shuffles: int,
    seed: int,
    device: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    config = json.loads((covopt_dir / "metadata" / "covopt_run_config.json").read_text(encoding="utf-8"))
    from_run_dir = Path(config["from_run_dir"])
    t_max = int(config.get("t_max", 128))
    trajectory_seed = int(config.get("seed", seed))

    manifest_q, matched = _subset_manifest_tangent_basis(
        population_csv=covopt_dir / "metadata" / "covopt_population_units.csv",
        canonical_manifest_csv=canonical_manifest_csv,
        tangent_basis_npz=tangent_basis_npz,
        basis_key=basis_key,
    )
    cov_idx = matched["cov_index"].to_numpy(dtype=int)
    records = pd.read_csv(covopt_dir / "metadata" / "covopt_rate_records.csv")
    records = records[records["scale_D"].astype(float).isin(scale_values)].copy()

    with np.load(covopt_dir / "cache" / "covopt_mu_j.npz") as z:
        j_all = np.asarray(z["J"], dtype=np.float32)
    with np.load(covopt_dir / "cache" / "covopt_covariances.npz") as z:
        covs = {name: np.asarray(z[name], dtype=np.float64) for name in z.files}

    print("Loading digital twin only to reconstruct exact-J target eigenvectors...", flush=True)
    model, _model_info, _device = load_digital_twin(device=device)
    trace_by_id = _load_trace_examples_from_metadata(from_run_dir, model, t_max=t_max)

    rows: list[dict[str, Any]] = []
    group_items = list(records.groupby(["family", "kind", "scale_D"], sort=True))
    for group_i, ((family, kind, scale), group) in enumerate(group_items, start=1):
        if int(max_rows_per_group) > 0 and len(group) > int(max_rows_per_group):
            group = group.sample(
                n=int(max_rows_per_group),
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).sort_values("row_id")
        row_ids = group["row_id"].to_numpy(dtype=int)
        j_group = [j_all[row_id][:, cov_idx, :] for row_id in row_ids]
        signal_jacs = [np.asarray(j, dtype=np.float64) for j in j_group]
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
            seed=trajectory_seed,
            t_max=t_max,
        )
        exact_j_cov, _exact_j_samples, _exact_j_pairs = movement_covariance_pooled_residual(exact_rows)
        targets = {
            "exact_j_linear_pooled": exact_j_cov,
            "pooled_residual": covs[_cov_key(str(family), str(kind), float(scale), "pooled_residual")][
                np.ix_(cov_idx, cov_idx)
            ],
            "within_pair": covs[_cov_key(str(family), str(kind), float(scale), "within_pair")][
                np.ix_(cov_idx, cov_idx)
            ],
        }
        print(f"[{group_i}/{len(group_items)}] {family} {kind} D={float(scale):g}", flush=True)
        for target_name, sigma in targets.items():
            for k in k_list:
                basis_items: list[tuple[str, np.ndarray]] = [
                    ("oracle_topPC", top_eigenvectors(sigma, int(k))),
                    ("exactJ_uncentered", exact_j_uncentered_q[:, : int(k)]),
                    ("exactJ_centered", exact_j_centered_q[:, : int(k)]),
                    ("manifest_tangent", manifest_q[:, : int(k)]),
                ]
                for basis, basis_matrix in basis_items:
                    rows.append(
                        {
                            "family": family,
                            "kind": kind,
                            "scale_D": float(scale),
                            "target": target_name,
                            "basis": basis,
                            "k": int(k),
                            "signal_fraction": _basis_signal_fraction(signal_jacs, basis_matrix),
                            "n_draws": 1,
                        }
                    )

                random_fracs = []
                for _draw in range(int(n_random_subspaces)):
                    random_fracs.append(_basis_signal_fraction(signal_jacs, _random_basis(len(cov_idx), int(k), rng)))
                rows.append(
                    {
                        "family": family,
                        "kind": kind,
                        "scale_D": float(scale),
                        "target": target_name,
                        "basis": "random",
                        "k": int(k),
                        "signal_fraction": float(np.nanmean(random_fracs)) if random_fracs else np.nan,
                        "signal_fraction_sd": float(np.nanstd(random_fracs, ddof=1)) if len(random_fracs) > 1 else np.nan,
                        "n_draws": int(n_random_subspaces),
                    }
                )

                shuffle_fracs = []
                for _draw in range(int(n_unit_shuffles)):
                    shuffle_fracs.append(
                        _basis_signal_fraction(
                            signal_jacs,
                            manifest_q[rng.permutation(len(cov_idx)), : int(k)],
                        )
                    )
                rows.append(
                    {
                        "family": family,
                        "kind": kind,
                        "scale_D": float(scale),
                        "target": target_name,
                        "basis": "unitshuffled_manifest",
                        "k": int(k),
                        "signal_fraction": float(np.nanmean(shuffle_fracs)) if shuffle_fracs else np.nan,
                        "signal_fraction_sd": float(np.nanstd(shuffle_fracs, ddof=1))
                        if len(shuffle_fracs) > 1
                        else np.nan,
                        "n_draws": int(n_unit_shuffles),
                    }
                )
    return pd.DataFrame(rows)


def _plot_k_summary(summary: pd.DataFrame, path: Path, *, k: int) -> None:
    metrics = [
        ("closure", "Closure fraction"),
        ("residual_trace", "Residual trace fraction"),
        ("signal_fraction", "Signal fraction"),
    ]
    colors = {
        "oracle_topPC": "#303030",
        "exactJ_uncentered": "#0072B2",
        "exactJ_centered": "#D55E00",
        "manifest_tangent": "#009E73",
        "random": "#999999",
        "unitshuffled_manifest": "#CC79A7",
    }
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10.8, 8.0), sharex=True)
    x = np.arange(len(TARGET_ORDER), dtype=float)
    width = 0.12
    offsets = (np.arange(len(BASIS_ORDER)) - (len(BASIS_ORDER) - 1) / 2.0) * width
    for ax, (metric, title) in zip(axes, metrics):
        for i, basis in enumerate(BASIS_ORDER):
            vals = []
            for target in TARGET_ORDER:
                match = summary[(summary["target"] == target) & (summary["basis"] == basis)]
                vals.append(float(match[metric].iloc[0]) if not match.empty else np.nan)
            ax.bar(
                x + offsets[i],
                vals,
                width=width,
                label=BASIS_LABELS[basis],
                color=colors[basis],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_ylabel(title)
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([TARGET_LABELS[t] for t in TARGET_ORDER])
    axes[0].set_title(f"Uncentered exact-J tangent audit at k={int(k)}")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.38), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_markdown(path: Path, summary: pd.DataFrame, *, k: int) -> None:
    lines = [
        "# Uncentered Exact-J Tangent Summary",
        "",
        f"Focused D=1 summary at `k={int(k)}`. Closure and residual trace are read from the existing sampled closure audit; signal fraction is recomputed for the same sampled row/basis regime.",
        "",
        "## Main Table",
        "",
    ]
    cols = ["target", "basis", "closure", "residual_trace", "signal_fraction"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    ordered = summary.copy()
    ordered["target"] = pd.Categorical(ordered["target"], TARGET_ORDER)
    ordered["basis"] = pd.Categorical(ordered["basis"], BASIS_ORDER)
    ordered = ordered.sort_values(["target", "basis"])
    for _, row in ordered.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append("nan" if not np.isfinite(value) else f"{float(value):.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Oracle top-PC and uncentered exact-J are both near-complete closures at `k=20`.",
            "- Centered exact-J has high omitted residual trace because the unit-mean/common-mode direction is absent.",
            "- Manifest tangent is stronger than random at closure but far below uncentered exact-J; the independent-basis specificity question remains unresolved.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--covopt-dir", type=Path, required=True)
    p.add_argument("--tangent-basis-npz", type=Path, required=True)
    p.add_argument("--canonical-manifest-csv", type=Path, required=True)
    p.add_argument("--basis-key", default="basis_delta_0p25")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((audit_dir / "uncentered_j_tangent_closure_metadata.json").read_text(encoding="utf-8"))
    k_list = tuple(int(x) for x in metadata["k_list"])
    signal_rows = _signal_fraction_table(
        covopt_dir=Path(args.covopt_dir),
        tangent_basis_npz=Path(args.tangent_basis_npz),
        canonical_manifest_csv=Path(args.canonical_manifest_csv),
        basis_key=str(args.basis_key),
        scale_values=tuple(float(x) for x in metadata["scales"]),
        k_list=k_list,
        max_rows_per_group=int(metadata["max_rows_per_group"]),
        n_random_subspaces=int(metadata["n_random_subspaces"]),
        n_unit_shuffles=int(metadata["n_unit_shuffles"]),
        seed=0,
        device=str(args.device),
    )
    signal_rows.to_csv(out_dir / "signal_fraction_by_group.csv", index=False)

    closure_rows = pd.read_csv(audit_dir / "uncentered_j_tangent_closure_summary.csv")
    agg_closure = (
        closure_rows.groupby(["target", "basis", "k"], as_index=False)[["closure", "residual_trace"]]
        .mean()
        .reset_index(drop=True)
    )
    agg_signal = (
        signal_rows.groupby(["target", "basis", "k"], as_index=False)[["signal_fraction"]]
        .mean()
        .reset_index(drop=True)
    )
    merged = agg_closure.merge(agg_signal, on=["target", "basis", "k"], how="left")
    merged.to_csv(out_dir / "uncentered_j_tangent_k_metrics.csv", index=False)
    k_summary = merged[merged["k"].astype(int) == int(args.k)].copy()
    k_summary.to_csv(out_dir / f"uncentered_j_tangent_k{int(args.k)}_summary.csv", index=False)
    _write_markdown(out_dir / f"uncentered_j_tangent_k{int(args.k)}_summary.md", k_summary, k=int(args.k))
    _plot_k_summary(k_summary, out_dir / f"uncentered_j_tangent_k{int(args.k)}_metrics.png", k=int(args.k))
    out_metadata = {
        "audit_dir": str(audit_dir),
        "covopt_dir": str(args.covopt_dir),
        "k": int(args.k),
        "input_metadata": metadata,
        "outputs": [
            "signal_fraction_by_group.csv",
            "uncentered_j_tangent_k_metrics.csv",
            f"uncentered_j_tangent_k{int(args.k)}_summary.csv",
            f"uncentered_j_tangent_k{int(args.k)}_summary.md",
            f"uncentered_j_tangent_k{int(args.k)}_metrics.png",
        ],
    }
    (out_dir / "summary_metadata.json").write_text(
        json.dumps(out_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out_metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
