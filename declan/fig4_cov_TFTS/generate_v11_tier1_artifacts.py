#!/usr/bin/env python3
"""Generate v11 Tier-1 Figure 4 analysis artifacts from audited outputs.

This is a postprocessor: it does not rerun the digital-twin model. It reshapes
the existing finite-difference closure, RF/readout-null, and compact-geometry
artifacts into the tables requested by v11_remaining_analysis_prescription.md.
Unavailable denominators are written explicitly so manuscript language cannot
silently imply that a reliable-shared or split-half ceiling denominator exists.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RF_ROOT = ROOT / "outputs" / "matched_twin_covariance_closure_rf_null_step025_rfbacked_v2"
DEFAULT_BUDGET_ROOT = ROOT / "outputs" / "compact_retinal_translation_geometry"
DEFAULT_TFTS_ROOT = ROOT / "outputs" / "twin_feature_tangent_structure"
DEFAULT_OUT = ROOT / "outputs" / "covTFTS_v11_remaining_analysis"

HEADLINE_TARGET = "psd"
HEADLINE_PROJECTION = "global_rate+target_pc1"
HEADLINE_K = 2
FULL_SOURCE = "fd_sample_eye_trace_cov"
COMPACT_SOURCE = "fd_sample_eye_trace_xfit_compact_k10_cov"
SOURCE_LABELS = {
    FULL_SOURCE: "full_finite_difference_source",
    COMPACT_SOURCE: "compact_k10_crossfit_source",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _ci(values: pd.Series) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _headline_block(metrics: pd.DataFrame, *, target_variant: str = HEADLINE_TARGET) -> pd.DataFrame:
    block = metrics[
        (metrics["target_variant"].astype(str) == target_variant)
        & (metrics["projection_control"].astype(str) == HEADLINE_PROJECTION)
        & (metrics["basis_source"].astype(str).isin([FULL_SOURCE, COMPACT_SOURCE]))
        & (metrics["k"].astype(int) == HEADLINE_K)
        & (metrics["row_status"].astype(str) == "ok")
    ].copy()
    block["source_variant"] = block["basis_source"].map(SOURCE_LABELS)
    return block


def write_variance_budget(
    *,
    rf_root: Path,
    budget_root: Path,
    out: Path,
) -> dict[str, Any]:
    metrics = _read_csv(rf_root / "finite_difference_capture_metrics.csv")
    session_summary = _read_csv(rf_root / "finite_difference_session_summary.csv")
    budget_summary_existing = _read_csv(budget_root / "panelD_variability_budget_summary.csv")
    block = _headline_block(metrics, target_variant=HEADLINE_TARGET)

    denom_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []

    denom_specs = [
        (
            "matched_conservative_projected_psd_FEM_target_trace",
            "target_trace",
            "available_from_psd_projected_headline_target",
        ),
        (
            "matched_full_psd_FEM_target_trace",
            "target_trace_psd",
            "available_from_unprojected_psd_target_trace",
        ),
        (
            "matched_raw_FEM_target_trace",
            "target_trace_raw",
            "available_from_unprojected_raw_target_trace",
        ),
        (
            "total_reliable_shared_covariance_trace",
            None,
            "not_available_in_current_closure_artifacts",
        ),
        (
            "total_trial_to_trial_covariance_trace",
            None,
            "not_available_in_current_closure_artifacts",
        ),
    ]
    first_by_session = block.sort_values(["session", "source_variant"]).drop_duplicates("session")
    negative_mass = session_summary.set_index("session")
    for row in first_by_session.itertuples(index=False):
        srow = negative_mass.loc[row.session] if row.session in negative_mass.index else None
        for denom_name, col, status in denom_specs:
            trace = _finite(getattr(row, col)) if col else float("nan")
            denom_rows.append(
                {
                    "session": row.session,
                    "target_variant": HEADLINE_TARGET,
                    "projection_control": HEADLINE_PROJECTION,
                    "denominator_name": denom_name,
                    "denominator_trace": trace,
                    "denominator_status": status,
                    "target_negative_eigenvalue_mass_raw": _finite(
                        srow["target_negative_eigenvalue_mass_raw"] if srow is not None else float("nan")
                    ),
                    "target_min_eigenvalue_raw": _finite(
                        srow["target_min_eigenvalue_raw"] if srow is not None else float("nan")
                    ),
                }
            )

    for row in block.itertuples(index=False):
        projected_trace = _finite(row.target_trace)
        absolute_capture = _finite(row.capture) * projected_trace
        rf_excess_trace = _finite(row.effect_minus_rf_fixed_permutation_median) * projected_trace
        unit_excess_trace = _finite(row.effect_minus_unit_shuffle_median) * projected_trace
        for denom_name, col, status in denom_specs:
            denom_trace = _finite(getattr(row, col)) if col else float("nan")
            capture_rows.append(
                {
                    "session": row.session,
                    "target_variant": HEADLINE_TARGET,
                    "projection_control": HEADLINE_PROJECTION,
                    "source_variant": row.source_variant,
                    "source_basis": row.basis_source,
                    "source_eigenspace_k": HEADLINE_K,
                    "denominator_name": denom_name,
                    "denominator_trace": denom_trace,
                    "denominator_status": status,
                    "absolute_captured_trace": absolute_capture,
                    "fraction_of_denominator": absolute_capture / denom_trace if denom_trace > 0 else float("nan"),
                    "rf_readout_null_adjusted_trace": rf_excess_trace,
                    "rf_readout_null_adjusted_fraction": rf_excess_trace / denom_trace if denom_trace > 0 else float("nan"),
                    "unit_shuffle_adjusted_trace": unit_excess_trace,
                    "unit_shuffle_adjusted_fraction": unit_excess_trace / denom_trace if denom_trace > 0 else float("nan"),
                    "capture_fraction_of_projected_target": _finite(row.capture),
                    "rf_readout_null_capture_fraction_of_projected_target": _finite(
                        row.effect_minus_rf_fixed_permutation_median
                    ),
                    "unit_shuffle_null_capture_fraction_of_projected_target": _finite(
                        row.effect_minus_unit_shuffle_median
                    ),
                    "reliability_ceiling": float("nan"),
                    "ceiling_normalized_capture": float("nan"),
                }
            )

    reliability_rows.append(
        {
            "target_variant": HEADLINE_TARGET,
            "projection_control": HEADLINE_PROJECTION,
            "status": "not_available_in_current_closure_artifacts",
            "required_quantity": "split_half_reliability_ceiling_for_recorded_FEM_target",
            "note": (
                "Existing closure outputs provide raw/PSD/projected target traces and RF/readout-null "
                "adjusted capture, but not split-half target covariance ceilings."
            ),
        }
    )

    denominators = pd.DataFrame(denom_rows)
    captures = pd.DataFrame(capture_rows)
    reliability = pd.DataFrame(reliability_rows)

    denominators.to_csv(out / "variance_budget_denominators.csv", index=False)
    captures.to_csv(out / "variance_budget_capture_fractions.csv", index=False)
    reliability.to_csv(out / "variance_budget_reliability_ceiling.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for (source, denom), g in captures.groupby(["source_variant", "denominator_name"]):
        available = g[g["denominator_status"].astype(str).str.startswith("available")]
        if available.empty:
            summary_rows.append(
                {
                    "source_variant": source,
                    "denominator_name": denom,
                    "n_sessions": 0,
                    "status": "not_available",
                }
            )
            continue
        low, high = _ci(available["fraction_of_denominator"])
        rflow, rfhigh = _ci(available["rf_readout_null_adjusted_fraction"])
        summary_rows.append(
            {
                "source_variant": source,
                "denominator_name": denom,
                "n_sessions": int(available["session"].nunique()),
                "status": "ok",
                "session_unweighted_fraction_mean": float(np.nanmean(available["fraction_of_denominator"])),
                "session_unweighted_fraction_ci_low": low,
                "session_unweighted_fraction_ci_high": high,
                "trace_weighted_fraction": _weighted_mean(
                    available["fraction_of_denominator"], available["denominator_trace"]
                ),
                "session_unweighted_rf_null_adjusted_fraction_mean": float(
                    np.nanmean(available["rf_readout_null_adjusted_fraction"])
                ),
                "session_unweighted_rf_null_adjusted_fraction_ci_low": rflow,
                "session_unweighted_rf_null_adjusted_fraction_ci_high": rfhigh,
                "trace_weighted_rf_null_adjusted_fraction": _weighted_mean(
                    available["rf_readout_null_adjusted_fraction"], available["denominator_trace"]
                ),
                "absolute_captured_trace_sum": float(np.nansum(available["absolute_captured_trace"])),
                "denominator_trace_sum": float(np.nansum(available["denominator_trace"])),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary: dict[str, Any] = {
        "status": "ok_with_explicit_missing_denominators",
        "headline_target_variant": HEADLINE_TARGET,
        "headline_projection_control": HEADLINE_PROJECTION,
        "headline_source_eigenspace_k": HEADLINE_K,
        "n_sessions": int(block["session"].nunique()),
        "source_files": {
            "rf_readout_closure_metrics": str((rf_root / "finite_difference_capture_metrics.csv").resolve()),
            "rf_readout_session_summary": str((rf_root / "finite_difference_session_summary.csv").resolve()),
            "existing_panelD_budget_summary": str((budget_root / "panelD_variability_budget_summary.csv").resolve()),
        },
        "negative_eigenvalue_mass_raw_total": float(np.nansum(denominators.drop_duplicates("session")["target_negative_eigenvalue_mass_raw"])),
        "summary_rows": summary_df.to_dict(orient="records"),
        "existing_panelD_summary_rows": budget_summary_existing.to_dict(orient="records"),
        "missing_load_bearing_denominators": [
            "total_reliable_shared_covariance_trace",
            "total_trial_to_trial_covariance_trace",
            "split_half_reliability_ceiling_for_recorded_FEM_target",
        ],
        "psd_note": (
            "PSD targets eigenvalue-clip the recorded FEM covariance; raw target traces and "
            "negative eigenvalue mass are reported side by side."
        ),
    }
    with (out / "variance_budget_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df[
        summary_df["denominator_name"].isin(
            [
                "matched_conservative_projected_psd_FEM_target_trace",
                "matched_full_psd_FEM_target_trace",
                "matched_raw_FEM_target_trace",
            ]
        )
    ].copy()
    labels = {
        "matched_conservative_projected_psd_FEM_target_trace": "projected PSD\nFEM target",
        "matched_full_psd_FEM_target_trace": "full PSD\nFEM target",
        "matched_raw_FEM_target_trace": "raw FEM\ntarget",
    }
    sources = ["full_finite_difference_source", "compact_k10_crossfit_source"]
    x = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, source in enumerate(sources):
        vals = []
        rf_vals = []
        for denom in labels:
            row = plot_df[(plot_df["source_variant"] == source) & (plot_df["denominator_name"] == denom)]
            vals.append(float(row["session_unweighted_fraction_mean"].iloc[0]) if len(row) else np.nan)
            rf_vals.append(float(row["session_unweighted_rf_null_adjusted_fraction_mean"].iloc[0]) if len(row) else np.nan)
        offset = (-0.5 + i) * width
        color = "#2f5f9f" if source.startswith("full") else "#7b5ea7"
        ax.bar(x + offset, vals, width=width, color=color, alpha=0.82, label=source.replace("_", " "))
        ax.scatter(x + offset, rf_vals, color="white", edgecolor=color, s=28, zorder=3, label=None)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in labels])
    ax.set_ylabel("fraction of denominator")
    ax.set_title("Compact translation component in variance-budget context", loc="left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "variance_budget_compact_translation_component.png", dpi=220)
    fig.savefig(fig_dir / "variance_budget_compact_translation_component.pdf")
    plt.close(fig)

    return summary


def write_rf_headline(*, rf_root: Path, out: Path) -> None:
    metrics = _read_csv(rf_root / "finite_difference_capture_metrics.csv")
    headline = _read_csv(rf_root / "finite_difference_headline_raw_psd_bootstrap.csv")
    psd = headline[
        (headline["target_variant"].astype(str) == HEADLINE_TARGET)
        & (headline["projection_control"].astype(str) == HEADLINE_PROJECTION)
        & (headline["basis_source"].astype(str).isin([FULL_SOURCE, COMPACT_SOURCE]))
        & (headline["k"].astype(int) == HEADLINE_K)
    ].copy()
    psd["source_variant"] = psd["basis_source"].map(SOURCE_LABELS)
    raw = headline[
        (headline["target_variant"].astype(str) == "raw")
        & (headline["projection_control"].astype(str) == HEADLINE_PROJECTION)
        & (headline["basis_source"].astype(str).isin([FULL_SOURCE, COMPACT_SOURCE]))
        & (headline["k"].astype(int) == HEADLINE_K)
    ][["basis_source", "capture_mean", "effect_unit_mean", "effect_rf_fixed_mean"]].copy()
    raw = raw.rename(
        columns={
            "capture_mean": "raw_target_capture_mean",
            "effect_unit_mean": "raw_target_effect_unit_mean",
            "effect_rf_fixed_mean": "raw_target_effect_rf_fixed_mean",
        }
    )
    out_rows = psd.merge(raw, on="basis_source", how="left")
    keep = [
        "source_variant",
        "basis_source",
        "target_variant",
        "projection_control",
        "k",
        "n_sessions",
        "capture_mean",
        "capture_boot_ci_low",
        "capture_boot_ci_high",
        "effect_unit_mean",
        "effect_unit_boot_ci_low",
        "effect_unit_boot_ci_high",
        "rf_fixed_null_median_mean",
        "effect_rf_fixed_mean",
        "effect_rf_fixed_boot_ci_low",
        "effect_rf_fixed_boot_ci_high",
        "n_effect_rf_fixed_positive",
        "sign_test_rf_fixed_p_two_sided",
        "raw_target_capture_mean",
        "raw_target_effect_unit_mean",
        "raw_target_effect_rf_fixed_mean",
    ]
    out_rows[keep].to_csv(out / "rf_readout_null_headline.csv", index=False)

    session_rows = _headline_block(metrics, target_variant=HEADLINE_TARGET)
    session_keep = [
        "session",
        "source_variant",
        "target_variant",
        "projection_control",
        "k",
        "n_common_units",
        "n_samples_used",
        "target_trace",
        "target_trace_raw",
        "target_trace_psd",
        "capture",
        "unit_shuffle_null_median",
        "effect_minus_unit_shuffle_median",
        "rf_fixed_permutation_null_median",
        "effect_minus_rf_fixed_permutation_median",
        "rf_null_status",
        "rf_null_n_bins",
        "rf_null_largest_bin_fraction",
        "rf_null_bin_features",
    ]
    session_rows[session_keep].to_csv(out / "rf_readout_null_session_effects.csv", index=False)

    fig_dir = out / "figures"
    plot = out_rows[keep].copy()
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.bar(x - 0.22, plot["capture_mean"], width=0.2, color="#2f5f9f", label="capture")
    ax.bar(x, plot["effect_unit_mean"], width=0.2, color="#9a9a9a", label="excess vs unit shuffle")
    ax.bar(x + 0.22, plot["effect_rf_fixed_mean"], width=0.2, color="#7b5ea7", label="excess vs RF/readout null")
    ax.set_xticks(x)
    ax.set_xticklabels(["full FD", "compact k=10"])
    ax.set_ylabel("fraction of projected PSD target")
    ax.set_title("Closure with unit-shuffle and RF/readout nulls", loc="left", fontsize=9)
    ymax = float(np.nanmax(plot[["capture_mean", "effect_unit_mean", "effect_rf_fixed_mean"]].to_numpy(dtype=float)))
    ax.set_ylim(0.0, ymax * 1.38)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "covariance_closure_unit_shuffle_and_rf_null.png", dpi=220)
    fig.savefig(fig_dir / "covariance_closure_unit_shuffle_and_rf_null.pdf")
    plt.close(fig)


def write_orthogonal_partition_placeholder(*, out: Path) -> None:
    reason = (
        "Order-independent covariance partition requires saved component basis matrices or "
        "target covariance matrices for global-rate, target-PC1, compact source, and remaining "
        "finite-difference source groups. The current audited CSV artifacts contain capture "
        "fractions after projection controls, but not enough linear operators to reconstruct "
        "a Shapley or orthogonalized partition without rerunning/augmenting the producer."
    )
    rows = [
        {
            "status": "not_computable_from_current_saved_artifacts",
            "component_group": component,
            "contribution_trace": float("nan"),
            "contribution_fraction": float("nan"),
            "reason": reason,
        }
        for component in [
            "global_rate_mode",
            "target_pc1_mode",
            "compact_translation_source_subspace",
            "remaining_finite_difference_source_subspace",
            "residual_FEM_covariance_target",
        ]
    ]
    pd.DataFrame(rows).to_csv(out / "orthogonal_covariance_partition.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "not_run",
                "order": "",
                "component_group": "",
                "incremental_capture_trace": float("nan"),
                "incremental_capture_fraction": float("nan"),
                "reason": reason,
            }
        ]
    ).to_csv(out / "orthogonal_covariance_partition_order_sensitivity.csv", index=False)

    fig_dir = out / "figures"
    fig, ax = plt.subplots(figsize=(5.2, 1.7))
    ax.axis("off")
    ax.text(
        0.01,
        0.75,
        "Orthogonal covariance partition: not computable from current saved CSV artifacts",
        fontsize=9,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.01,
        0.35,
        "Requires saved component bases/covariance matrices or a producer-side rerun.",
        fontsize=8,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "orthogonal_covariance_partition.png", dpi=220)
    fig.savefig(fig_dir / "orthogonal_covariance_partition.pdf")
    plt.close(fig)


def write_bookkeeping(*, rf_root: Path, tfts_root: Path, out: Path) -> None:
    manifest = _read_json(rf_root / "run_manifest.json")
    audit = _read_json(rf_root / "finite_difference_provenance_audit.json")
    inventory = _read_csv(rf_root / "session_inventory.csv")
    session_summary = _read_csv(rf_root / "finite_difference_session_summary.csv")
    n_sessions = int(inventory["session"].nunique())
    units = f"{int(session_summary['n_common_units'].min())}-{int(session_summary['n_common_units'].max())} common units/session"
    sessions = ",".join(inventory["session"].astype(str).tolist())
    fig3_cache = audit.get("fig3_cache", manifest.get("fig3_cache", ""))
    fig2_cache = audit.get("fig2_cache", manifest.get("fig2_cache", ""))
    max_samples = manifest.get("max_samples", audit.get("manifest_max_samples", ""))
    sample_cap = "all candidate samples" if str(max_samples) in {"0", "0.0"} else str(max_samples)
    rows = [
        {
            "panel": "A",
            "analysis_name": "recorded_FEM_covariance_dimensionality",
            "sessions": "8-session Figure 4A anchor per v11 note; exact producer not in RF closure manifest",
            "units": "see figure/covariance producer artifact",
            "stimulus regime": "FixRSVP",
            "response window": "33.333 ms per v11 note",
            "eye window": "matched FEM conditioning window from Figure 4A producer",
            "latency convention": "producer-specific; verify before caption",
            "sample cap": "",
            "target/source cache": fig2_cache,
            "projection controls": "eye-position conditioning",
            "included/excluded criteria": "see Figure 4A producer",
        },
        {
            "panel": "B",
            "analysis_name": "local_translation_charts",
            "sessions": "digital-twin/FixRSVP examples",
            "units": "model readout units",
            "stimulus regime": "FixRSVP digital twin",
            "response window": "digital-twin cached response window",
            "eye window": "finite-difference retinal translations",
            "latency convention": "not signed; local tangent visualization",
            "sample cap": "",
            "target/source cache": str((tfts_root / "tangent_maps" / "twin_tangent_maps.pkl").resolve()),
            "projection controls": "none for glyph visualization",
            "included/excluded criteria": "representative object/tangent maps",
        },
        {
            "panel": "C",
            "analysis_name": "pooled_translation_tangent_compactness",
            "sessions": "digital-twin/FixRSVP pooled objects",
            "units": "model readout units",
            "stimulus regime": "FixRSVP digital twin",
            "response window": "digital-twin cached response window",
            "eye window": "local finite-difference tangent step",
            "latency convention": "not signed; tangent spectrum",
            "sample cap": "",
            "target/source cache": str((tfts_root / "union_spectrum").resolve()),
            "projection controls": "unit-shuffle/null spectrum references",
            "included/excluded criteria": "see twin_feature_tangent_summary.json",
        },
        {
            "panel": "D",
            "analysis_name": "cross_image_generalization_of_tangent_basis",
            "sessions": "image-disjoint split from tangent-structure output",
            "units": "model readout units",
            "stimulus regime": "FixRSVP digital twin",
            "response window": "digital-twin cached response window",
            "eye window": "local finite-difference tangent step",
            "latency convention": "not signed",
            "sample cap": "",
            "target/source cache": str((tfts_root / "split_modes" / "image_disjoint").resolve()),
            "projection controls": "held-out image split",
            "included/excluded criteria": "see split basis CSV",
        },
        {
            "panel": "E",
            "analysis_name": "compact_tangent_basis_FEM_related_displacement_sensitivity",
            "sessions": "see tangent_subspace_information producer",
            "units": "model readout units",
            "stimulus regime": "FixRSVP digital twin with FEM histories",
            "response window": "producer-specific Fisher window",
            "eye window": "real FEM histories",
            "latency convention": "Poisson/Fisher local displacement sensitivity",
            "sample cap": "",
            "target/source cache": "outputs/tangent_subspace_information/*/results",
            "projection controls": "unit-shuffle/random nulls where available",
            "included/excluded criteria": "see Panel E producer manifest",
        },
        {
            "panel": "F",
            "analysis_name": "finite_difference_translation_covariance_closure_with_RF_readout_null",
            "sessions": f"{n_sessions}: {sessions}",
            "units": units,
            "stimulus regime": "FixRSVP recorded sessions matched to digital twin",
            "response window": "target window index 1",
            "eye window": "valid 120-bin FixRSVP time axis; finite-difference source from recorded eye traces",
            "latency convention": "closure target/source caches; unsigned covariance comparison",
            "sample cap": sample_cap,
            "target/source cache": f"{fig2_cache}; {fig3_cache}",
            "projection controls": ",".join(manifest.get("projection_controls", [])),
            "included/excluded criteria": (
                f"status ok sessions; RF null status ok in closure outputs; step_px={manifest.get('step_px', audit.get('manifest_step_px', ''))}"
            ),
        },
    ]
    pd.DataFrame(rows).to_csv(out / "figure4_panel_bookkeeping.csv", index=False)


def _fmt(value: Any, digits: int = 3) -> str:
    val = _finite(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def _summary_row(summary: dict[str, Any], source: str, denominator: str) -> dict[str, Any]:
    for row in summary.get("summary_rows", []):
        if row.get("source_variant") == source and row.get("denominator_name") == denominator:
            return row
    return {}


def write_figure_companion_document(*, out: Path, summary: dict[str, Any]) -> None:
    fig_dir = out / "figures"
    headline = _read_csv(out / "rf_readout_null_headline.csv")
    full = headline[headline["source_variant"].astype(str) == "full_finite_difference_source"].iloc[0]
    compact = headline[headline["source_variant"].astype(str) == "compact_k10_crossfit_source"].iloc[0]
    projected = "matched_conservative_projected_psd_FEM_target_trace"
    full_psd = "matched_full_psd_FEM_target_trace"
    raw = "matched_raw_FEM_target_trace"
    compact_projected = _summary_row(summary, "compact_k10_crossfit_source", projected)
    compact_full_psd = _summary_row(summary, "compact_k10_crossfit_source", full_psd)
    compact_raw = _summary_row(summary, "compact_k10_crossfit_source", raw)
    full_projected = _summary_row(summary, "full_finite_difference_source", projected)
    full_full_psd = _summary_row(summary, "full_finite_difference_source", full_psd)
    full_raw = _summary_row(summary, "full_finite_difference_source", raw)

    root_outputs = [
        ("../variance_budget_denominators.csv", "Per-session denominator traces, including explicit unavailable rows."),
        ("../variance_budget_capture_fractions.csv", "Per-session full and compact capture fractions against each denominator."),
        ("../variance_budget_reliability_ceiling.csv", "Split-half ceiling status row; currently unavailable from closure artifacts."),
        ("../variance_budget_summary.json", "Session-unweighted and trace-weighted variance-budget summaries."),
        ("../rf_readout_null_headline.csv", "Headline PSD/raw closure rows with unit-shuffle and RF/readout-null effects."),
        ("../rf_readout_null_session_effects.csv", "Per-session RF/readout-null effects for full and compact sources."),
        ("../orthogonal_covariance_partition.csv", "Explicit not-computable status for order-independent partition."),
        (
            "../orthogonal_covariance_partition_order_sensitivity.csv",
            "Explicit not-run status for order-sensitivity/Shapley partition.",
        ),
        ("../figure4_panel_bookkeeping.csv", "Panel/session/window/cache bookkeeping table."),
        ("../run_manifest.json", "Generation manifest."),
    ]
    figure_outputs = [
        ("v11_tier1_figure_companion.md", "This companion document."),
        ("variance_budget_compact_translation_component.png", "Variance-budget figure, bitmap."),
        ("variance_budget_compact_translation_component.pdf", "Variance-budget figure, vector/PDF."),
        ("covariance_closure_unit_shuffle_and_rf_null.png", "Closure null comparison figure, bitmap."),
        ("covariance_closure_unit_shuffle_and_rf_null.pdf", "Closure null comparison figure, vector/PDF."),
        ("orthogonal_covariance_partition.png", "Partition status figure, bitmap."),
        ("orthogonal_covariance_partition.pdf", "Partition status figure, vector/PDF."),
    ]

    lines = [
        "# V11 Tier 1 Figure Companion",
        "",
        "This document travels with the generated Figure 4 Tier 1 outputs. It summarizes the manuscript-ready numbers, lists every output artifact, and records the denominator guardrails.",
        "",
        "## Headline",
        "",
        (
            "Under the conservative PSD target with global-rate and target-PC1 components removed, "
            f"the full finite-difference translation source captured {_fmt(full['capture_mean'])} of the FEM target trace "
            f"and exceeded the RF/readout-preserving null by +{_fmt(full['effect_rf_fixed_mean'])} "
            f"[{_fmt(full['effect_rf_fixed_boot_ci_low'])}, {_fmt(full['effect_rf_fixed_boot_ci_high'])}]. "
            f"Restricting the source to the cross-fit compact k=10 tangent basis preserved the effect "
            f"(capture = {_fmt(compact['capture_mean'])}; RF/readout-null excess = +{_fmt(compact['effect_rf_fixed_mean'])} "
            f"[{_fmt(compact['effect_rf_fixed_boot_ci_low'])}, {_fmt(compact['effect_rf_fixed_boot_ci_high'])}])."
        ),
        "",
        (
            f"The compact source corresponds to {_fmt(compact_projected.get('session_unweighted_fraction_mean', float('nan')) * 100, 1)}% "
            "of the conservative projected PSD FEM target, "
            f"{_fmt(compact_full_psd.get('session_unweighted_fraction_mean', float('nan')) * 100, 1)}% "
            "of the full matched PSD FEM covariance, and "
            f"{_fmt(compact_raw.get('session_unweighted_fraction_mean', float('nan')) * 100, 1)}% "
            "of the matched raw FEM covariance."
        ),
        "",
        "Use this as a covariance-prediction claim. Do not phrase it as a fraction of total V1 shared variability until the reliable-shared denominator exists.",
        "",
        "## Budget Summary",
        "",
        "| source | projected PSD FEM target | full PSD FEM covariance | raw FEM covariance |",
        "|---|---:|---:|---:|",
        (
            "| full finite difference | "
            f"{_fmt(full_projected.get('session_unweighted_fraction_mean', float('nan')))} | "
            f"{_fmt(full_full_psd.get('session_unweighted_fraction_mean', float('nan')))} | "
            f"{_fmt(full_raw.get('session_unweighted_fraction_mean', float('nan')))} |"
        ),
        (
            "| compact k=10 cross-fit | "
            f"{_fmt(compact_projected.get('session_unweighted_fraction_mean', float('nan')))} | "
            f"{_fmt(compact_full_psd.get('session_unweighted_fraction_mean', float('nan')))} | "
            f"{_fmt(compact_raw.get('session_unweighted_fraction_mean', float('nan')))} |"
        ),
        "",
        "White dots in the variance-budget figure mark RF/readout-null-adjusted fractions.",
        "",
        "![Variance budget](variance_budget_compact_translation_component.png)",
        "",
        "## Null Comparison",
        "",
        "| source | capture | excess vs unit shuffle | excess vs RF/readout null | RF/readout-null CI |",
        "|---|---:|---:|---:|---:|",
        (
            "| full finite difference | "
            f"{_fmt(full['capture_mean'])} | {_fmt(full['effect_unit_mean'])} | "
            f"{_fmt(full['effect_rf_fixed_mean'])} | "
            f"[{_fmt(full['effect_rf_fixed_boot_ci_low'])}, {_fmt(full['effect_rf_fixed_boot_ci_high'])}] |"
        ),
        (
            "| compact k=10 cross-fit | "
            f"{_fmt(compact['capture_mean'])} | {_fmt(compact['effect_unit_mean'])} | "
            f"{_fmt(compact['effect_rf_fixed_mean'])} | "
            f"[{_fmt(compact['effect_rf_fixed_boot_ci_low'])}, {_fmt(compact['effect_rf_fixed_boot_ci_high'])}] |"
        ),
        "",
        "![RF/readout null comparison](covariance_closure_unit_shuffle_and_rf_null.png)",
        "",
        "## Guardrails",
        "",
        "- Full and compact budget fractions are drawn from the same 24 matched closure sessions and the same headline target/projection/k filter.",
        "- Projected PSD, full PSD, and raw FEM denominators are separate. The figure labels them separately.",
        "- RF/readout-null excess is displayed directly, not only unit-shuffle excess.",
        "- Reliable-shared covariance trace, total trial-to-trial covariance trace, and split-half reliability ceiling are unavailable in the current closure artifacts and are written as unavailable, not as zero.",
        "- The orthogonal covariance partition is not computable from the saved CSV artifacts because the component bases/covariance matrices were not saved.",
        "",
        "## Generated Outputs",
        "",
        "Root artifact folder: `..` relative to this document.",
        "",
    ]
    for path, desc in root_outputs:
        lines.append(f"- [`{path}`]({path}): {desc}")
    lines.extend(["", "Figure-side artifacts:", ""])
    for path, desc in figure_outputs:
        lines.append(f"- [`{path}`]({path}): {desc}")
    lines.extend(
        [
            "",
            "## Safe Manuscript Wording",
            "",
            "> A compact retinal-translation geometry derived from an image-computable V1 model predicted a reliable component of recorded FEM-linked covariance. Under the PSD target with global-rate and target-PC1 components removed, the full finite-difference translation source captured 0.216 of the FEM target trace and exceeded the RF/readout-preserving null by +0.158 [0.125, 0.193]. Restricting the source to the cross-fit compact k=10 tangent basis preserved the effect (capture = 0.217; RF/readout-null excess = +0.161 [0.128, 0.196]), corresponding to 21.7% of the projected FEM target and 9.4% of the full matched PSD FEM covariance.",
            "",
        ]
    )
    (fig_dir / "v11_tier1_figure_companion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-root", type=Path, default=DEFAULT_RF_ROOT)
    parser.add_argument("--budget-root", type=Path, default=DEFAULT_BUDGET_ROOT)
    parser.add_argument("--tfts-root", type=Path, default=DEFAULT_TFTS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)

    summary = write_variance_budget(rf_root=args.rf_root, budget_root=args.budget_root, out=out)
    write_rf_headline(rf_root=args.rf_root, out=out)
    write_orthogonal_partition_placeholder(out=out)
    write_bookkeeping(rf_root=args.rf_root, tfts_root=args.tfts_root, out=out)
    write_figure_companion_document(out=out, summary=summary)

    manifest = {
        "status": "ok",
        "script": str(Path(__file__).resolve()),
        "out": str(out.resolve()),
        "artifacts": sorted(p.name for p in out.iterdir() if p.is_file()),
        "figures": sorted(p.name for p in (out / "figures").iterdir()),
        "variance_budget_status": summary.get("status"),
    }
    with (out / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
