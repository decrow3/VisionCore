"""Summarize compact-mechanism promotion gates and basis-overlap diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analyze_compact_mechanism import _load_basis, _load_npz_table, _parse_float_list, _parse_int_list, _parse_list, _static_pc_basis


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _subspace_overlap(a: np.ndarray, b: np.ndarray, k: int) -> dict[str, float]:
    kk = min(int(k), a.shape[1], b.shape[1])
    if kk <= 0:
        return {
            "mean_cos2": float("nan"),
            "min_cos2": float("nan"),
            "max_cos2": float("nan"),
            "principal_angle_deg_max": float("nan"),
        }
    sv = np.linalg.svd(a[:, :kk].T @ b[:, :kk], compute_uv=False)
    cos2 = np.clip(sv * sv, 0.0, 1.0)
    return {
        "mean_cos2": float(np.mean(cos2)),
        "min_cos2": float(np.min(cos2)),
        "max_cos2": float(np.max(cos2)),
        "principal_angle_deg_max": float(np.degrees(np.arccos(np.sqrt(np.min(cos2))))),
    }


def _load_static_basis(base_run_dir: Path, summary: pd.DataFrame, k_max: int, filters: argparse.Namespace) -> np.ndarray:
    manifest = pd.read_csv(base_run_dir / "response_cache_manifest.csv")
    if filters.candidate_set_modes:
        manifest = manifest[manifest["candidate_set_mode"].astype(str).isin(set(_parse_list(filters.candidate_set_modes)))]
    if filters.motion_scales:
        scales = {round(v, 10) for v in _parse_float_list(filters.motion_scales)}
        manifest = manifest[manifest["scale"].astype(float).round(10).isin(scales)]
    if filters.priors:
        priors = {v.lower() for v in _parse_list(filters.priors)}
        manifest = manifest[manifest["prior_family"].astype(str).str.lower().isin(priors)]
    zero_tables = []
    for _, row in manifest.iterrows():
        table = _load_npz_table(base_run_dir / str(row["response_cache_path"]))
        zero_tables.append(np.asarray(table["zero_lambda_counts"], dtype=np.float32))
    n_units = int(summary["n_units"].dropna().iloc[0]) if "n_units" in summary.columns and summary["n_units"].notna().any() else 756
    return _static_pc_basis(zero_tables, n_units=n_units, k_max=k_max)


def _control_value(summary: pd.DataFrame, key: dict[str, Any], variant: str, k_dim: int | None = None) -> pd.Series | None:
    mask = np.ones(len(summary), dtype=bool)
    for col, value in key.items():
        mask &= summary[col].eq(value).to_numpy()
    mask &= summary["response_variant"].eq(variant).to_numpy()
    if k_dim is not None:
        mask &= summary["k_dim"].eq(k_dim).to_numpy()
    rows = summary[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _promotion_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    compact = summary[summary["response_variant"].eq("compact_only")]
    key_cols = ["candidate_set_mode", "prior_condition", "motion_scale", "likelihood_scale", "basis_mode", "k_dim"]
    for _, comp in compact.iterrows():
        key = {col: comp[col] for col in key_cols if col != "k_dim"}
        k_dim = int(comp["k_dim"])
        full = _control_value(summary, key, "full_exact", 0)
        zero = _control_value(summary, key, "zero_static", 0)
        random = _control_value(summary, key, "random_k", k_dim)
        shuffled = _control_value(summary, key, "unit_shuffle_compact", k_dim)
        static_pc = _control_value(summary, key, "static_pc_k", k_dim)
        gain = _control_value(summary, key, "gain_only", 1)
        removed = _control_value(summary, key, "compact_removed", k_dim)
        log_removed = _control_value(summary, key, "log_compact_removed", k_dim)

        def get(row: pd.Series | None, col: str) -> float:
            return float(row[col]) if row is not None and col in row and pd.notna(row[col]) else float("nan")

        rows.append(
            {
                **key,
                "k_dim": k_dim,
                "full_joint_accuracy": get(full, "joint_eye_accuracy"),
                "zero_joint_accuracy": get(zero, "joint_eye_accuracy"),
                "compact_joint_accuracy": float(comp["joint_eye_accuracy"]),
                "random_joint_accuracy": get(random, "joint_eye_accuracy"),
                "unit_shuffle_joint_accuracy": get(shuffled, "joint_eye_accuracy"),
                "gain_joint_accuracy": get(gain, "joint_eye_accuracy"),
                "static_pc_joint_accuracy": get(static_pc, "joint_eye_accuracy"),
                "compact_removed_joint_accuracy": get(removed, "joint_eye_accuracy"),
                "log_compact_removed_joint_accuracy": get(log_removed, "joint_eye_accuracy"),
                "compact_true_score_rescue": float(comp["median_joint_rescue_fraction_true_score"]),
                "random_true_score_rescue": get(random, "median_joint_rescue_fraction_true_score"),
                "unit_shuffle_true_score_rescue": get(shuffled, "median_joint_rescue_fraction_true_score"),
                "gain_true_score_rescue": get(gain, "median_joint_rescue_fraction_true_score"),
                "static_pc_true_score_rescue": get(static_pc, "median_joint_rescue_fraction_true_score"),
                "compact_removed_true_score_rescue": get(removed, "median_joint_rescue_fraction_true_score"),
                "log_compact_removed_true_score_rescue": get(log_removed, "median_joint_rescue_fraction_true_score"),
                "compact_minus_random_true_score_rescue": float(comp["median_joint_rescue_fraction_true_score"]) - get(random, "median_joint_rescue_fraction_true_score"),
                "compact_minus_unit_shuffle_true_score_rescue": float(comp["median_joint_rescue_fraction_true_score"]) - get(shuffled, "median_joint_rescue_fraction_true_score"),
                "compact_minus_gain_true_score_rescue": float(comp["median_joint_rescue_fraction_true_score"]) - get(gain, "median_joint_rescue_fraction_true_score"),
                "compact_minus_static_pc_true_score_rescue": float(comp["median_joint_rescue_fraction_true_score"]) - get(static_pc, "median_joint_rescue_fraction_true_score"),
                "compact_clipped_fraction": float(comp["median_clipped_rate_fraction"]),
                "compact_removed_clipped_fraction": get(removed, "median_clipped_rate_fraction"),
                "log_compact_removed_clipped_fraction": get(log_removed, "median_clipped_rate_fraction"),
            }
        )
    return rows


def _plot(summary: pd.DataFrame, out_dir: Path, *, candidate_mode: str, motion_scale: float, likelihood_scale: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_error.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    plot_dir = out_dir / "figures"
    plot_dir.mkdir(exist_ok=True)
    variants = ["full_exact", "zero_static", "compact_only", "compact_removed", "log_compact_removed", "random_k", "unit_shuffle_compact", "gain_only", "static_pc_k"]
    for prior in sorted(summary["prior_condition"].dropna().unique()):
        df = summary[
            summary["candidate_set_mode"].eq(candidate_mode)
            & summary["motion_scale"].astype(float).eq(float(motion_scale))
            & summary["likelihood_scale"].astype(float).eq(float(likelihood_scale))
            & summary["prior_condition"].eq(prior)
            & summary["response_variant"].isin(variants)
        ].copy()
        if df.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for variant, grp in df.groupby("response_variant"):
            x = grp["k_dim"].astype(float).to_numpy()
            if variant in {"full_exact", "zero_static", "gain_only"}:
                x = np.full_like(x, 1.0 if variant == "gain_only" else 0.0)
            order = np.argsort(x)
            label = variant
            axes[0].plot(x[order], grp["joint_eye_accuracy"].to_numpy()[order], marker="o", label=label)
            axes[1].plot(x[order], grp["median_joint_rescue_fraction_true_score"].to_numpy()[order], marker="o", label=label)
            axes[2].plot(x[order], grp["median_clipped_rate_fraction"].to_numpy()[order], marker="o", label=label)
        axes[0].set_title("Joint Accuracy")
        axes[1].set_title("True-Score Rescue")
        axes[2].set_title("Clipped Rate Fraction")
        for ax in axes:
            ax.set_xlabel("k")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("accuracy")
        axes[1].set_ylabel("fraction")
        axes[2].set_ylabel("fraction")
        axes[2].set_yscale("symlog", linthresh=1e-4)
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"{candidate_mode}, scale={motion_scale}, likelihood={likelihood_scale}, prior={prior}")
        fig.savefig(plot_dir / f"compact_mechanism_ksweep_{candidate_mode}_scale{motion_scale:g}_like{likelihood_scale:g}_{prior}.png", dpi=180)
        plt.close(fig)


def summarize(args: argparse.Namespace) -> Path:
    compact_run_dir = Path(args.compact_run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else compact_run_dir / "followup_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(compact_run_dir / "compact_mechanism_summary.csv")
    promotion = pd.DataFrame(_promotion_rows(summary))
    promotion.to_csv(out_dir / "compact_mechanism_promotion_gates.csv", index=False)

    basis, basis_meta = _load_basis(Path(args.compact_basis_path), n_units=int(args.n_units), basis_key=str(args.basis_key))
    k_list = _parse_int_list(args.k_dims)
    static_basis = _load_static_basis(Path(args.base_run_dir), summary, max(k_list), args)
    gain = np.ones((basis.shape[0], 1), dtype=np.float64)
    gain /= np.linalg.norm(gain)
    rng = np.random.default_rng(int(args.seed))
    overlap_rows = []
    for k in k_list:
        for name, other in [
            ("static_pc", static_basis),
            ("gain_ones", gain),
            ("random", np.linalg.qr(rng.standard_normal((basis.shape[0], int(k))))[0][:, : int(k)]),
        ]:
            row = {
                "comparison": f"compact_vs_{name}",
                "k_dim": int(k),
                **_subspace_overlap(basis, other, int(k)),
            }
            overlap_rows.append(row)
    pd.DataFrame(overlap_rows).to_csv(out_dir / "compact_staticpc_basis_overlap.csv", index=False)

    _plot(
        summary,
        out_dir,
        candidate_mode=str(args.primary_candidate_set_mode),
        motion_scale=float(args.primary_motion_scale),
        likelihood_scale=float(args.primary_likelihood_scale),
    )
    _write_json(
        out_dir / "compact_mechanism_followup_metadata.json",
        {
            "compact_run_dir": str(compact_run_dir),
            "base_run_dir": str(args.base_run_dir),
            "basis": basis_meta,
            "k_dims": k_list,
            "primary_candidate_set_mode": str(args.primary_candidate_set_mode),
            "primary_motion_scale": float(args.primary_motion_scale),
            "primary_likelihood_scale": float(args.primary_likelihood_scale),
        },
    )
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-run-dir", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--compact-basis-path", type=Path, required=True)
    parser.add_argument("--basis-key", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k-dims", default="2,5,10,20")
    parser.add_argument("--n-units", type=int, default=756)
    parser.add_argument("--candidate-set-modes", default="")
    parser.add_argument("--motion-scales", default="")
    parser.add_argument("--priors", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--primary-candidate-set-mode", default="matched_static_response")
    parser.add_argument("--primary-motion-scale", type=float, default=1.0)
    parser.add_argument("--primary-likelihood-scale", type=float, default=1.0)
    return parser


def main() -> None:
    out = summarize(build_parser().parse_args())
    print(f"Wrote compact mechanism follow-up summary to {out}")


if __name__ == "__main__":
    main()
