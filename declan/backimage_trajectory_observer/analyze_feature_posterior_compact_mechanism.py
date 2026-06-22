"""Feature-posterior compact-subspace mechanism analysis for BackImage tables."""

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
    from .analyze_compact_mechanism import (
        _load_basis,
        _orth_residual_basis,
        _project_delta,
        _random_basis,
        _rate_audit,
        _safe_for_likelihood,
        _static_pc_basis,
        _unit_shuffle_basis,
        _validate_basis_mode,
    )
    from .analyze_feature_posterior import (
        _add_uncertainty_fields,
        _auto_likelihood_scales,
        _candidate_set_lookup,
        _candidate_window_indices,
        _filter_manifest,
        _filter_response_cache_manifest,
        _fit_feature_spaces,
        _load_npz,
        _load_observer_trial_metadata,
        _load_or_compute_latents,
        _mode_row,
        _parse_float_list,
        _parse_int_list,
        _parse_str_list,
        _safe_bool,
        _safe_float,
    )
    from .observer import score_image_identity_score_vectors
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.backimage_trajectory_observer.analyze_compact_mechanism import (
        _load_basis,
        _orth_residual_basis,
        _project_delta,
        _random_basis,
        _rate_audit,
        _safe_for_likelihood,
        _static_pc_basis,
        _unit_shuffle_basis,
        _validate_basis_mode,
    )
    from declan.backimage_trajectory_observer.analyze_feature_posterior import (
        _add_uncertainty_fields,
        _auto_likelihood_scales,
        _candidate_set_lookup,
        _candidate_window_indices,
        _filter_manifest,
        _filter_response_cache_manifest,
        _fit_feature_spaces,
        _load_npz,
        _load_observer_trial_metadata,
        _load_or_compute_latents,
        _mode_row,
        _parse_float_list,
        _parse_int_list,
        _parse_str_list,
        _safe_bool,
        _safe_float,
    )
    from declan.backimage_trajectory_observer.observer import score_image_identity_score_vectors


CORE_RESPONSE_VARIANTS = (
    "full_exact",
    "zero_static",
    "compact_only",
    "compact_removed",
    "compact_addback",
    "static_pc_only",
    "static_pc_removed",
    "gain_only",
    "gain_removed",
    "unit_shuffle_only",
    "unit_shuffle_removed",
    "random_only",
    "random_removed",
    "compact_residualized_against_static_pc_only",
    "compact_residualized_against_static_pc_removed",
    "static_pc_residualized_against_compact_only",
    "static_pc_residualized_against_compact_removed",
)
KNOWN_EYE_VARIANT = "known_eye"
PAIRWISE_CONTRASTS = (
    ("full_exact", "zero_static"),
    ("compact_only", "zero_static"),
    ("compact_removed", "zero_static"),
    ("compact_only", "compact_removed"),
    ("full_exact", "compact_removed"),
    ("full_exact", "static_pc_removed"),
    ("full_exact", "gain_removed"),
    ("full_exact", "unit_shuffle_removed"),
    ("full_exact", "random_removed"),
    ("compact_removed", "static_pc_removed"),
    ("compact_residualized_against_static_pc_removed", "static_pc_residualized_against_compact_removed"),
    ("known_eye", "full_exact"),
)
CONTRAST_METRICS = (
    "feature_cosine",
    "feature_neg_mse",
    "candidate_posterior_true_mass",
)


def _progress(message: str) -> None:
    print(f"[feature-compact] {message}", flush=True)


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


def _candidate_ids(table: dict[str, np.ndarray], n_candidates: int) -> list[str]:
    if "candidate_ids" not in table:
        return [str(i) for i in range(int(n_candidates))]
    return [str(v) for v in np.asarray(table["candidate_ids"]).tolist()]


def _scalar_int(table: dict[str, np.ndarray], key: str, default: int = -1) -> int:
    if key not in table:
        return int(default)
    arr = np.asarray(table[key]).reshape(-1)
    return int(arr[0]) if arr.size else int(default)


def _response_variants(args: argparse.Namespace) -> list[str]:
    variants = _parse_str_list(args.variants)
    unknown = sorted(set(variants).difference(set(CORE_RESPONSE_VARIANTS) | {KNOWN_EYE_VARIANT}))
    if unknown:
        raise ValueError(f"Unsupported response variants: {unknown}")
    if not bool(args.no_known_eye) and KNOWN_EYE_VARIANT not in variants:
        variants.append(KNOWN_EYE_VARIANT)
    return variants


def _variant_tables(
    variant: str,
    *,
    prior_full: np.ndarray,
    known_full: np.ndarray,
    zero: np.ndarray,
    u: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return response tables for one compact feature-posterior curve."""
    if variant == "full_exact":
        return prior_full.astype(np.float64), known_full.astype(np.float64), zero.astype(np.float64)
    if variant == "zero_static":
        z = zero.astype(np.float64)
        return np.broadcast_to(z[:, None, :, :], prior_full.shape).copy(), z.copy(), z.copy()
    if variant == KNOWN_EYE_VARIANT:
        return prior_full.astype(np.float64), known_full.astype(np.float64), zero.astype(np.float64)
    if u is None:
        raise ValueError(f"response_variant={variant!r} requires a compact basis")

    z = zero.astype(np.float64)
    prior_delta = prior_full.astype(np.float64) - z[:, None, :, :]
    known_delta = known_full.astype(np.float64) - z
    prior_compact = _project_delta(prior_delta, u)
    known_compact = _project_delta(known_delta, u)
    prior_residual = prior_delta - prior_compact
    known_residual = known_delta - known_compact
    only_variants = {
        "compact_only",
        "static_pc_only",
        "gain_only",
        "unit_shuffle_only",
        "random_only",
        "compact_residualized_against_static_pc_only",
        "static_pc_residualized_against_compact_only",
    }
    removed_variants = {
        "compact_removed",
        "static_pc_removed",
        "gain_removed",
        "unit_shuffle_removed",
        "random_removed",
        "compact_residualized_against_static_pc_removed",
        "static_pc_residualized_against_compact_removed",
    }
    if variant in only_variants:
        return z[:, None, :, :] + prior_compact, z + known_compact, z.copy()
    if variant in removed_variants:
        return z[:, None, :, :] + prior_residual, z + known_residual, z.copy()
    if variant == "compact_addback":
        return (
            z[:, None, :, :] + prior_residual + prior_compact,
            z + known_residual + known_compact,
            z.copy(),
        )
    raise ValueError(f"Unsupported response_variant={variant!r}")


def _basis_type(variant: str) -> str:
    if variant in {"compact_only", "compact_removed", "compact_addback"}:
        return "compact"
    if variant in {"static_pc_only", "static_pc_removed"}:
        return "static_pc"
    if variant in {"gain_only", "gain_removed"}:
        return "gain_ones"
    if variant in {"unit_shuffle_only", "unit_shuffle_removed"}:
        return "unit_shuffle_compact"
    if variant in {"random_only", "random_removed"}:
        return "random"
    if variant in {"compact_residualized_against_static_pc_only", "compact_residualized_against_static_pc_removed"}:
        return "compact_residualized_against_static_pc"
    if variant in {"static_pc_residualized_against_compact_only", "static_pc_residualized_against_compact_removed"}:
        return "static_pc_residualized_against_compact"
    return "none"


def _score_source_mode(variant: str) -> str:
    return "known" if variant == KNOWN_EYE_VARIANT else "joint"


def _requires_static_basis(variants: list[str]) -> bool:
    return bool(
        set(variants).intersection(
            {
                "static_pc_only",
                "static_pc_removed",
                "compact_residualized_against_static_pc_only",
                "compact_residualized_against_static_pc_removed",
                "static_pc_residualized_against_compact_only",
                "static_pc_residualized_against_compact_removed",
            }
        )
    )


def _basis_specs_for_variants(
    *,
    variants: list[str],
    compact_u: np.ndarray,
    static_basis: np.ndarray | None,
    n_units: int,
    compact_k: int,
    seed: int,
) -> list[tuple[str, str, np.ndarray | None, int]]:
    specs: list[tuple[str, str, np.ndarray | None, int]] = []
    gain = np.ones((int(n_units), 1), dtype=np.float64)
    gain /= np.linalg.norm(gain)
    unit_shuffle, _ = _unit_shuffle_basis(compact_u, np.random.default_rng(int(seed) + 10_000 + int(compact_k)))
    random_u = _random_basis(int(n_units), int(compact_k), np.random.default_rng(int(seed) + 100_000 * int(compact_k)))
    for variant in variants:
        basis_type = _basis_type(variant)
        if basis_type == "none":
            specs.append((variant, basis_type, None, int(compact_k)))
        elif basis_type == "compact":
            specs.append((variant, basis_type, compact_u, int(compact_u.shape[1])))
        elif basis_type == "static_pc":
            if static_basis is None:
                raise ValueError(f"{variant} requested but static response PC basis was not built")
            specs.append((variant, basis_type, static_basis[:, : int(compact_k)], int(min(compact_k, static_basis.shape[1]))))
        elif basis_type == "gain_ones":
            specs.append((variant, basis_type, gain, 1))
        elif basis_type == "unit_shuffle_compact":
            specs.append((variant, basis_type, unit_shuffle, int(unit_shuffle.shape[1])))
        elif basis_type == "random":
            specs.append((variant, basis_type, random_u, int(random_u.shape[1])))
        elif basis_type == "compact_residualized_against_static_pc":
            if static_basis is None:
                raise ValueError(f"{variant} requested but static response PC basis was not built")
            u_resid = _orth_residual_basis(compact_u, static_basis[:, : int(compact_k)])
            specs.append((variant, basis_type, u_resid, int(u_resid.shape[1])))
        elif basis_type == "static_pc_residualized_against_compact":
            if static_basis is None:
                raise ValueError(f"{variant} requested but static response PC basis was not built")
            u_resid = _orth_residual_basis(static_basis[:, : int(compact_k)], compact_u)
            specs.append((variant, basis_type, u_resid, int(u_resid.shape[1])))
        else:
            raise ValueError(f"Unsupported basis type {basis_type!r} for variant {variant!r}")
    return specs


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = [
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_condition",
        "prior_family",
        "prior_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_prior_mode",
        "likelihood_scale",
        "latent",
        "requested_k",
        "k_eff",
        "response_variant",
        "score_source_mode",
        "requested_response_k_dim",
        "k_dim",
        "basis_mode",
        "basis_type",
    ]
    out: list[dict[str, Any]] = []
    for key, grp in df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, key, strict=True)}
        row["n_trial_rows"] = int(len(grp))
        row["n_trials"] = int(grp["trial_id"].nunique())
        row["mean_feature_cosine"] = float(grp["feature_cosine"].mean())
        row["median_feature_cosine"] = float(grp["feature_cosine"].median())
        row["mean_feature_neg_mse"] = float(grp["feature_neg_mse"].mean())
        row["median_feature_neg_mse"] = float(grp["feature_neg_mse"].median())
        row["mean_feature_rmse"] = float(grp["feature_rmse"].mean())
        row["median_feature_rmse"] = float(grp["feature_rmse"].median())
        row["mean_candidate_true_mass"] = float(grp["candidate_posterior_true_mass"].mean())
        row["median_candidate_true_mass"] = float(grp["candidate_posterior_true_mass"].median())
        row["mean_candidate_N_eff_fraction"] = float(grp["candidate_posterior_N_eff_fraction"].mean())
        row["median_candidate_N_eff_fraction"] = float(grp["candidate_posterior_N_eff_fraction"].median())
        row["median_score_true_rank"] = float(grp["score_true_rank"].median())
        row["mean_score_true_margin"] = float(grp["score_true_margin"].mean())
        row["median_negative_rate_fraction_before_clamp"] = float(grp["negative_rate_fraction_before_clamp"].median())
        row["median_clipped_rate_fraction"] = float(grp["clipped_rate_fraction"].median())
        row["median_negative_rate_min"] = float(grp["negative_rate_min"].median())
        row["median_negative_rate_mass"] = float(grp["negative_rate_mass"].median())
        out.append(row)
    return out


def _contrast_rows(
    rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    confidence: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    index_cols = [
        "table_index",
        "trial_id",
        "response_cache_path",
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_condition",
        "prior_family",
        "prior_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_prior_mode",
        "zero_reference_mode",
        "bin_seconds",
        "likelihood_scale",
        "latent",
        "requested_k",
        "k_eff",
        "requested_response_k_dim",
        "k_dim",
        "basis_mode",
    ]
    duplicate_key = index_cols + ["response_variant"]
    dupes = df.duplicated(duplicate_key, keep=False)
    if bool(dupes.any()):
        sample = df.loc[dupes, duplicate_key].head(5).to_dict("records")
        raise ValueError(f"Duplicate feature-compact rows for contrast key: {sample}")

    group_cols = [col for col in index_cols if col not in {"table_index", "trial_id", "response_cache_path"}]
    out: list[dict[str, Any]] = []
    for metric in CONTRAST_METRICS:
        wide = df.pivot(index=index_cols, columns="response_variant", values=metric).reset_index()
        for key, grp in wide.groupby(group_cols, dropna=False):
            context = {col: value for col, value in zip(group_cols, key, strict=True)}
            for lhs, rhs in PAIRWISE_CONTRASTS:
                if lhs not in grp.columns or rhs not in grp.columns:
                    continue
                vals = pd.to_numeric(grp[lhs], errors="coerce") - pd.to_numeric(grp[rhs], errors="coerce")
                finite = vals[np.isfinite(vals)]
                row = {
                    **context,
                    "metric": metric,
                    "lhs_response_variant": lhs,
                    "rhs_response_variant": rhs,
                    "contrast": f"{lhs}_minus_{rhs}",
                    "mean_lhs_minus_rhs": float(finite.mean()) if len(finite) else float("nan"),
                    "median_lhs_minus_rhs": float(finite.median()) if len(finite) else float("nan"),
                    "n_trials": int(grp["trial_id"].nunique()),
                    "n_paired_trials": int(len(finite)),
                }
                _add_uncertainty_fields(
                    row,
                    prefix="mean_lhs_minus_rhs",
                    values=vals,
                    rng=rng,
                    n_bootstrap=int(n_bootstrap),
                    n_permutations=int(n_permutations),
                    confidence=float(confidence),
                )
                out.append(row)
    return out


def _reference_comparison_rows(summary: list[dict[str, Any]], reference_path: Path | None) -> list[dict[str, Any]]:
    if reference_path is None:
        return []
    ref = pd.read_csv(reference_path)
    summary_df = pd.DataFrame(summary)
    if summary_df.empty:
        return []
    mode_by_variant = {
        "full_exact": "joint",
        "zero_static": "zero",
        KNOWN_EYE_VARIANT: "known",
    }
    compare_pairs = [
        ("mean_feature_cosine", "{mode}_mean_cosine"),
        ("mean_feature_neg_mse", "{mode}_mean_neg_mse"),
        ("mean_candidate_true_mass", "{mode}_mean_true_mass"),
        ("median_candidate_N_eff_fraction", "{mode}_median_candidate_N_eff_fraction"),
    ]
    out: list[dict[str, Any]] = []
    for _, row in summary_df[summary_df["response_variant"].isin(mode_by_variant)].iterrows():
        mode = mode_by_variant[str(row["response_variant"])]
        ref_rows = ref[
            (ref["candidate_set_mode"].astype(str) == str(row["candidate_set_mode"]))
            & (ref["observation_scale"].astype(float).round(10) == round(float(row["observation_scale"]), 10))
            & (ref["prior_family"].astype(str) == str(row["prior_family"]))
            & (ref["likelihood_scale"].astype(float).round(10) == round(float(row["likelihood_scale"]), 10))
            & (ref["latent"].astype(str) == str(row["latent"]))
            & (ref["requested_k"].astype(int) == int(row["requested_k"]))
        ]
        if ref_rows.empty:
            out.append(
                {
                    "qc_type": "reference_feature_posterior_match",
                    "response_variant": row["response_variant"],
                    "candidate_set_mode": row["candidate_set_mode"],
                    "observation_scale": row["observation_scale"],
                    "prior_family": row["prior_family"],
                    "latent": row["latent"],
                    "requested_k": row["requested_k"],
                    "status": "missing_reference_row",
                }
            )
            continue
        ref_row = ref_rows.iloc[0]
        for ours_key, ref_template in compare_pairs:
            ref_key = ref_template.format(mode=mode)
            ours = _safe_float(row.get(ours_key))
            expected = _safe_float(ref_row.get(ref_key))
            out.append(
                {
                    "qc_type": "reference_feature_posterior_match",
                    "response_variant": row["response_variant"],
                    "score_source_mode": mode,
                    "candidate_set_mode": row["candidate_set_mode"],
                    "observation_scale": row["observation_scale"],
                    "prior_family": row["prior_family"],
                    "likelihood_scale": row["likelihood_scale"],
                    "latent": row["latent"],
                    "requested_k": row["requested_k"],
                    "metric": ours_key,
                    "reference_metric": ref_key,
                    "observed": ours,
                    "reference": expected,
                    "abs_delta": abs(ours - expected) if np.isfinite(ours) and np.isfinite(expected) else float("nan"),
                    "status": "ok",
                }
            )
    return out


def _write_report(
    out_dir: Path,
    *,
    run_dir: Path,
    summary: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    reconstruction_rows: list[dict[str, Any]],
    basis_meta: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    summary_df = pd.DataFrame(summary)
    qc_df = pd.DataFrame(qc_rows)
    contrast_df = pd.DataFrame(contrasts)
    max_prior_error = max((float(r["prior_addback_reconstruction_max_abs_error"]) for r in reconstruction_rows), default=float("nan"))
    max_known_error = max((float(r["known_addback_reconstruction_max_abs_error"]) for r in reconstruction_rows), default=float("nan"))
    reference_delta = float("nan")
    if not qc_df.empty and "qc_type" in qc_df.columns:
        ref = qc_df[qc_df["qc_type"].eq("reference_feature_posterior_match")]
        if not ref.empty and "abs_delta" in ref.columns:
            vals = pd.to_numeric(ref["abs_delta"], errors="coerce").dropna()
            reference_delta = float(vals.max()) if len(vals) else float("nan")

    lines = [
        "# Feature-Posterior Compact Mechanism",
        "",
        "This cache-only analysis applies the compact-subspace intervention to the",
        "feature-posterior endpoint used by Figure 4C.",
        "",
        "## Inputs",
        "",
        f"- source run: `{run_dir}`",
        f"- compact basis: `{basis_meta.get('basis_path', '')}`",
        f"- basis key: `{basis_meta.get('basis_key', '')}`",
        f"- basis shape: `{basis_meta.get('basis_shape', '')}`",
        f"- basis mode: `{args.basis_mode}`",
        f"- image-disjoint provenance declared: `{bool(basis_meta.get('declares_image_disjoint', False))}`",
        f"- variants: `{', '.join(_response_variants(args))}`",
        f"- compact k dimensions: `{args.k_dims}`",
        f"- latent names: `{args.latent_names}`",
        f"- feature PCA k list: `{args.pca_k_list}`",
        "",
        "## Validation",
        "",
        f"- max prior addback reconstruction error: `{max_prior_error:.6g}`",
        f"- max known addback reconstruction error: `{max_known_error:.6g}`",
        f"- max reference feature-posterior summary delta: `{reference_delta:.6g}`",
        "",
        "## Primary Files",
        "",
        "- `feature_compact_mechanism_trials.csv`",
        "- `feature_compact_mechanism_summary.csv`",
        "- `feature_compact_mechanism_uncertainty.csv`",
        "- `feature_compact_mechanism_qc.csv`",
        "- `feature_compact_mechanism_metadata.json`",
        "",
        "## Claim Boundary",
        "",
        "If compact-only stays near full joint while compact-removed falls toward",
        "zero-eye, this supports the claim that the compact subspace carries",
        "much of the feature information used by hidden-eye joint decoding. It",
        "does not prove that the animal computes this posterior, that the",
        "posterior identifies the true eye trajectory, that the compact subspace",
        "is the only useful response structure, or that behavior optimizes this",
        "model objective.",
        "",
        "## Summary Preview",
        "",
    ]
    if not summary_df.empty:
        preview_cols = [
            "candidate_set_mode",
            "observation_scale",
            "prior_family",
            "response_variant",
            "requested_response_k_dim",
            "k_dim",
            "mean_feature_cosine",
            "median_feature_cosine",
            "mean_feature_neg_mse",
            "mean_candidate_true_mass",
            "median_candidate_N_eff_fraction",
            "median_clipped_rate_fraction",
        ]
        lines.append("```text")
        lines.append(summary_df[preview_cols].head(80).to_csv(index=False).strip())
        lines.append("```")
    if not contrast_df.empty:
        lines.extend(["", "## Contrast Preview", "", "```text"])
        preview = contrast_df[
            (contrast_df["metric"] == "feature_cosine")
            & (contrast_df["contrast"].isin(["compact_only_minus_compact_removed", "full_exact_minus_compact_removed"]))
        ]
        cols = [
            "observation_scale",
            "prior_family",
            "contrast",
            "mean_lhs_minus_rhs",
            "mean_lhs_minus_rhs_ci_low",
            "mean_lhs_minus_rhs_ci_high",
            "mean_lhs_minus_rhs_permutation_p_two_sided",
        ]
        lines.append(preview[cols].head(40).to_csv(index=False).strip())
        lines.append("```")
    (out_dir / "feature_compact_mechanism_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-npz", type=Path, default=None)
    parser.add_argument("--feature-manifest", type=Path, default=None)
    parser.add_argument("--latent-names", default="pyramid_local_field")
    parser.add_argument("--pca-k-list", default="8")
    parser.add_argument("--likelihood-scales", default="auto")
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--candidate-set-modes", default="")
    parser.add_argument("--priors", default="")
    parser.add_argument("--motion-scales", default="")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--latent-crop-px", type=int, default=151)
    parser.add_argument("--center-crop-px", type=int, default=41)
    parser.add_argument("--local-field-grid", type=int, default=8)
    parser.add_argument("--trust-feature-row-order", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--compact-basis-path", type=Path, required=True)
    parser.add_argument("--basis-key", default="auto")
    parser.add_argument("--basis-mode", default="global")
    parser.add_argument("--allow-unverified-image-disjoint-basis", action="store_true")
    parser.add_argument("--k-dims", default="10")
    parser.add_argument("--variants", default="full_exact,zero_static,compact_only,compact_removed,compact_addback")
    parser.add_argument("--no-known-eye", action="store_true")
    parser.add_argument("--reference-feature-summary", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--uncertainty-confidence", type=float, default=0.95)
    parser.add_argument("--uncertainty-seed", type=int, default=0)
    return parser


def analyze(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(run_dir / "selected_windows.csv")
    manifest = pd.read_csv(run_dir / "response_cache_manifest.csv")
    manifest = _filter_manifest(manifest, args)
    if manifest.empty:
        raise ValueError("No response cache rows remain after filtering")
    manifest, skipped_cache_rows = _filter_response_cache_manifest(manifest, run_dir)
    candidate_sets_path = run_dir / "candidate_sets.csv"
    candidate_sets = pd.read_csv(candidate_sets_path) if candidate_sets_path.exists() and candidate_sets_path.stat().st_size > 0 else pd.DataFrame()
    candidate_lookup = _candidate_set_lookup(candidate_sets)
    source_row_to_pos = {
        int(row["source_row"]): int(pos)
        for pos, row in windows.iterrows()
    } if "source_row" in windows.columns else {}

    first = _load_npz(run_dir / str(manifest.iloc[0]["response_cache_path"]))
    n_units = int(np.asarray(first["prior_lambda_counts"]).shape[-1])
    compact_k_dims = _parse_int_list(args.k_dims)
    if not compact_k_dims:
        raise ValueError("--k-dims must request at least one compact dimension")
    basis_full, basis_meta = _load_basis(Path(args.compact_basis_path), n_units=n_units, basis_key=str(args.basis_key))
    _validate_basis_mode(args, basis_meta)
    max_k = max(compact_k_dims)
    if basis_full.shape[1] < max_k:
        raise ValueError(f"Basis has only {basis_full.shape[1]} columns, but max k={max_k}")

    feature_k_list = _parse_int_list(args.pca_k_list)
    likelihood_scales = _auto_likelihood_scales(run_dir, str(args.likelihood_scales))
    trial_metadata = _load_observer_trial_metadata(run_dir)
    latent_arrays, latent_qc, feature_source = _load_or_compute_latents(args, windows, out_dir)
    feature_spaces, feature_qc = _fit_feature_spaces(latent_arrays, feature_k_list)
    variants = _response_variants(args)
    static_basis: np.ndarray | None = None
    if _requires_static_basis(variants):
        _progress("building static response PC basis")
        zero_tables = []
        for _, row in manifest.iterrows():
            tab = _load_npz(run_dir / str(row["response_cache_path"]))
            zero_tables.append(np.asarray(tab["zero_lambda_counts"], dtype=np.float32))
        static_basis = _static_pc_basis(zero_tables, n_units=n_units, k_max=max_k)
        _progress(f"static response PC basis ready with shape={static_basis.shape}")
    uncertainty_rng = np.random.default_rng(int(args.uncertainty_seed))
    _progress(
        f"selected tables={manifest.shape[0]}; variants={','.join(variants)}; "
        f"compact_k={','.join(str(k) for k in compact_k_dims)}; "
        f"feature_k={','.join(str(k) for k in feature_k_list)}"
    )

    trial_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = list(latent_qc) + list(feature_qc)
    reconstruction_rows: list[dict[str, Any]] = []
    progress_every = max(1, int(args.progress_every))
    manifest_items = list(manifest.iterrows())
    for progress_i, (table_index, man_row) in enumerate(tqdm(manifest_items, desc="feature compact tables"), start=1):
        table_path = run_dir / str(man_row["response_cache_path"])
        table = _load_npz(table_path)
        prior_full = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        known_full = np.asarray(table["known_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        y_obs = np.asarray(table["y_obs_counts"], dtype=np.float64)
        true_idx = _scalar_int(table, "true_candidate_index", 0)
        candidate_ids = _candidate_ids(table, prior_full.shape[0])
        candidate_indices, candidate_index_source = _candidate_window_indices(
            manifest_row=man_row,
            candidate_ids=candidate_ids,
            candidate_lookup=candidate_lookup,
            source_row_to_pos=source_row_to_pos,
            n_windows=int(windows.shape[0]),
        )
        qc_rows.append(
            {
                "qc_type": "candidate_alignment",
                "table_index": int(table_index),
                "trial_id": int(man_row["trial_id"]),
                "response_cache_path": str(man_row["response_cache_path"]),
                "candidate_set_mode": str(man_row["candidate_set_mode"]),
                "n_candidates": int(len(candidate_ids)),
                "candidate_index_source": candidate_index_source,
                "candidate_indices": ";".join(str(v) for v in candidate_indices),
                "candidate_ids": ";".join(candidate_ids),
            }
        )
        meta = trial_metadata.get(str(man_row["response_cache_path"]), {})

        for compact_k in compact_k_dims:
            compact_u = basis_full[:, : int(compact_k)]
            prior_delta = prior_full - zero[:, None, :, :]
            known_delta = known_full - zero
            prior_compact = _project_delta(prior_delta, compact_u)
            known_compact = _project_delta(known_delta, compact_u)
            prior_residual = prior_delta - prior_compact
            known_residual = known_delta - known_compact
            reconstruction_rows.append(
                {
                    "qc_type": "compact_addback_reconstruction",
                    "table_index": int(table_index),
                    "trial_id": int(man_row["trial_id"]),
                    "response_cache_path": str(man_row["response_cache_path"]),
                    "k_dim": int(compact_k),
                    "prior_addback_reconstruction_max_abs_error": float(
                        np.max(np.abs((zero[:, None, :, :] + prior_residual + prior_compact) - prior_full))
                    ),
                    "known_addback_reconstruction_max_abs_error": float(
                        np.max(np.abs((zero + known_residual + known_compact) - known_full))
                    ),
                }
            )

            for variant, basis_type, variant_u, effective_k_dim in _basis_specs_for_variants(
                variants=variants,
                compact_u=compact_u,
                static_basis=static_basis,
                n_units=n_units,
                compact_k=int(compact_k),
                seed=int(args.uncertainty_seed),
            ):
                prior_var, known_var, zero_var = _variant_tables(
                    variant,
                    prior_full=prior_full,
                    known_full=known_full,
                    zero=zero,
                    u=variant_u,
                )
                audit = _rate_audit(
                    np.concatenate([prior_var.reshape(-1), known_var.reshape(-1), zero_var.reshape(-1)]),
                    float(args.eps),
                )
                prior_score = _safe_for_likelihood(prior_var, float(args.eps))
                known_score = _safe_for_likelihood(known_var, float(args.eps))
                zero_score = _safe_for_likelihood(zero_var, float(args.eps))
                for likelihood_scale in likelihood_scales:
                    vectors = score_image_identity_score_vectors(
                        y_obs_counts=y_obs,
                        prior_lambda_counts=prior_score,
                        known_lambda_counts=known_score,
                        zero_lambda_counts=zero_score,
                        true_candidate_index=true_idx,
                        candidate_ids=candidate_ids,
                        eps=float(args.eps),
                        likelihood_scale=float(likelihood_scale),
                    )
                    source_mode = _score_source_mode(variant)
                    scores = np.asarray(vectors[f"{source_mode}_scores"], dtype=np.float64)
                    for (latent, k_requested), space in feature_spaces.items():
                        features_all = np.asarray(space["scores"], dtype=np.float64)
                        candidate_features = features_all[np.asarray(candidate_indices, dtype=int)]
                        z_true = candidate_features[int(true_idx)]
                        base_cols = {
                            "table_index": int(table_index),
                            "trial_id": int(man_row["trial_id"]),
                            "response_cache_path": str(man_row["response_cache_path"]),
                            "candidate_set_mode": str(man_row["candidate_set_mode"]),
                            "observation_condition": str(meta.get("observation_condition", man_row.get("observation_family", ""))),
                            "observation_family": str(man_row.get("observation_family", "")),
                            "observation_scale": float(meta.get("observation_scale", man_row.get("scale", np.nan))),
                            "prior_condition": str(meta.get("prior_condition", man_row.get("prior_family", ""))),
                            "prior_family": str(man_row.get("prior_family", "")),
                            "prior_scale": float(meta.get("prior_scale", man_row.get("scale", np.nan))),
                            "axis_catalog_mode": str(man_row.get("axis_catalog_mode", "shared")),
                            "axis_shared_source_catalog": _safe_bool(man_row.get("axis_shared_source_catalog", False)),
                            "trajectory_prior_mode": str(
                                meta.get("trajectory_prior_mode", man_row.get("trajectory_prior_mode", "unknown"))
                            ),
                            "zero_reference_mode": str(meta.get("zero_reference_mode", man_row.get("zero_reference_mode", ""))),
                            "bin_seconds": float(meta.get("bin_seconds", man_row.get("bin_seconds", np.nan))),
                            "likelihood_scale": float(likelihood_scale),
                            "likelihood_family": "poisson_expected_count",
                            "eps": float(args.eps),
                            "n_candidates": int(prior_full.shape[0]),
                            "n_trajectories": int(prior_full.shape[1]),
                            "n_timebins": int(prior_full.shape[2]),
                            "n_units": int(prior_full.shape[3]),
                            "true_candidate_index": int(true_idx),
                            "true_image_id": str(candidate_ids[int(true_idx)]),
                            "latent": str(latent),
                            "requested_k": int(k_requested),
                            "k_eff": int(space["k_eff"]),
                            "raw_feature_dim": int(space["raw_feature_dim"]),
                            "feature_variance_fraction": float(space["variance_fraction"]),
                            "feature_space": "selected_windows_zscore_pca",
                            "feature_source": feature_source,
                            "response_variant": str(variant),
                            "score_source_mode": str(source_mode),
                            "basis_type": str(basis_type),
                            "basis_mode": str(args.basis_mode),
                            "k_dim": int(effective_k_dim),
                            "requested_response_k_dim": int(compact_k),
                            "basis_path": str(args.compact_basis_path),
                            "basis_key": str(basis_meta.get("basis_key", "")),
                            "basis_shape": json.dumps(basis_meta.get("basis_shape", [])),
                            "image_disjoint_basis_verified": bool(
                                str(args.basis_mode) == "image_disjoint" and basis_meta.get("declares_image_disjoint", False)
                            ),
                            **audit,
                        }
                        trial_rows.append(
                            _mode_row(
                                base_cols=base_cols,
                                observer_mode=source_mode,
                                scores=scores,
                                candidate_features=candidate_features,
                                z_true=z_true,
                                true_idx=int(true_idx),
                                temperature=float(args.posterior_temperature),
                            )
                        )
        if progress_i == 1 or progress_i == len(manifest_items) or progress_i % progress_every == 0:
            _progress(f"scored {progress_i}/{len(manifest_items)} response tables")

    qc_rows.extend(reconstruction_rows)
    qc_rows.append(
        {
            "qc_type": "response_cache_manifest",
            "n_manifest_rows_after_cli_filters": int(manifest.shape[0] + skipped_cache_rows),
            "n_response_cache_rows_scored": int(manifest.shape[0]),
            "n_manifest_rows_without_response_cache": int(skipped_cache_rows),
        }
    )
    summary = _summary_rows(trial_rows)
    qc_rows.extend(_reference_comparison_rows(summary, args.reference_feature_summary))
    contrasts = _contrast_rows(
        trial_rows,
        rng=uncertainty_rng,
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        confidence=float(args.uncertainty_confidence),
    )

    _write_csv(out_dir / "feature_compact_mechanism_trials.csv", trial_rows)
    _write_csv(out_dir / "feature_compact_mechanism_summary.csv", summary)
    _write_csv(out_dir / "feature_compact_mechanism_uncertainty.csv", contrasts)
    _write_csv(out_dir / "feature_compact_mechanism_qc.csv", qc_rows)
    _write_json(
        out_dir / "feature_compact_mechanism_metadata.json",
        {
            "run_dir": run_dir,
            "n_selected_tables": int(manifest.shape[0]),
            "n_manifest_rows_without_response_cache": int(skipped_cache_rows),
            "feature_source": feature_source,
            "likelihood_scales": likelihood_scales,
            "compact_k_dims": compact_k_dims,
            "variants": variants,
            "basis": basis_meta,
            "basis_mode": str(args.basis_mode),
            "image_disjoint_basis_verified": bool(
                str(args.basis_mode) == "image_disjoint" and basis_meta.get("declares_image_disjoint", False)
            ),
            "outputs": [
                "feature_compact_mechanism_trials.csv",
                "feature_compact_mechanism_summary.csv",
                "feature_compact_mechanism_uncertainty.csv",
                "feature_compact_mechanism_qc.csv",
                "feature_compact_mechanism_report.md",
            ],
            "uncertainty": {
                "n_bootstrap": int(args.n_bootstrap),
                "n_permutations": int(args.n_permutations),
                "confidence": float(args.uncertainty_confidence),
                "seed": int(args.uncertainty_seed),
                "permutation_test": "paired random sign flip against zero mean contrast",
                "bootstrap_target": "paired trial-level mean contrast",
            },
            "config": vars(args),
        },
    )
    _write_report(
        out_dir,
        run_dir=run_dir,
        summary=summary,
        contrasts=contrasts,
        qc_rows=qc_rows,
        reconstruction_rows=reconstruction_rows,
        basis_meta=basis_meta,
        args=args,
    )
    _progress(f"wrote feature-compact outputs to {out_dir}")
    return out_dir


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
