#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "outputs/phase1_fem_covariance"
DEFAULT_OUTPUT_DIR = INPUT_ROOT / "derivative_geometry_refinement_20260530"
MIN_RELIABILITY_CEILING = 0.20


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_single_row(path: Path) -> dict[str, str]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows[0]


def _maybe_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _maybe_int(value: str | float | int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return 0


def _regularized_delta(delta: float, ceiling: float) -> tuple[float, str]:
    if not np.isfinite(delta):
        return float("nan"), "delta_not_finite"
    if not np.isfinite(ceiling) or ceiling < MIN_RELIABILITY_CEILING:
        return float("nan"), "low_reliability_ceiling"
    return float(delta / ceiling), "ok"


def _support_label(session_row: dict[str, str], reg_r2: float, reg_subspace: float) -> str:
    mc_tier = str(session_row.get("mc_context_support_tier", "")).strip()
    if np.isfinite(reg_r2) and np.isfinite(reg_subspace) and reg_r2 > 0 and reg_subspace > 0:
        return "image_conditioned_support"
    if mc_tier == "strong" and np.isfinite(reg_r2) and reg_r2 > -0.05:
        return "mixed_but_supported_by_context"
    return "shared_covariance_supported"


def _session_bundle(session_dir: Path) -> dict[str, dict[str, str]]:
    covariance_dir = session_dir / "covariance_geometry"
    aggregation_dir = session_dir / "aggregation_scaling"
    return {
        "session_metrics": _read_single_row(covariance_dir / "covariance_geometry_session_metrics.csv"),
        "shared_vs_image_specific": _read_single_row(covariance_dir / "shared_vs_image_specific_metrics.csv"),
        "aggregation_session": _read_single_row(aggregation_dir / "aggregation_scaling_session_metrics.csv"),
        "aggregation_curve": _read_csv_rows(aggregation_dir / "reliability_vs_n_windows.csv"),
        "stage4_comparison": _read_single_row(aggregation_dir / "stage4_comparison_metrics.csv"),
    }


def _session_output_rows(session: str, bundle: dict[str, dict[str, str]]) -> dict[str, float | str]:
    sm = bundle["session_metrics"]
    svi = bundle["shared_vs_image_specific"]
    agg = bundle["aggregation_session"]
    stage4 = bundle["stage4_comparison"]

    reliability_ceiling = _maybe_float(sm.get("reliability_ceiling"))
    mean_r2_shared = _maybe_float(svi.get("mean_r2_shared"))
    mean_r2_image_specific = _maybe_float(svi.get("mean_r2_image_specific"))
    r2_delta = _maybe_float(svi.get("r2_delta_image_vs_shared"))
    subspace_delta = _maybe_float(svi.get("mean_subspace_delta"))
    regularized_r2_delta, regularized_r2_status = _regularized_delta(r2_delta, reliability_ceiling)
    regularized_subspace_delta, regularized_subspace_status = _regularized_delta(subspace_delta, reliability_ceiling)
    support_label = _support_label(sm, regularized_r2_delta, regularized_subspace_delta)

    return {
        "session": session,
        "n_units_primary": _maybe_int(sm.get("n_units_primary")),
        "n_valid_bins": _maybe_int(sm.get("n_valid_bins")),
        "reliability_ceiling": reliability_ceiling,
        "b_emp_top2_fraction": _maybe_float(sm.get("b_emp_top2_fraction")),
        "mc_top2_fraction": _maybe_float(sm.get("mc_top2_fraction")),
        "mc_context_support_tier": str(sm.get("mc_context_support_tier", "")),
        "mc_fit_mode": str(sm.get("mc_fit_mode", "")),
        "model_alignment": _maybe_float(sm.get("model_alignment")),
        "model_shuffle_alignment": _maybe_float(sm.get("model_shuffle_alignment")),
        "primary_vs_sensitivity_cov_corr": _maybe_float(sm.get("primary_vs_sensitivity_cov_corr")),
        "primary_vs_sensitivity_subspace_overlap": _maybe_float(sm.get("primary_vs_sensitivity_subspace_overlap")),
        "mean_r2_shared": mean_r2_shared,
        "mean_r2_image_specific": mean_r2_image_specific,
        "r2_delta_image_vs_shared": r2_delta,
        "mean_subspace_delta": subspace_delta,
        "regularized_r2_delta": regularized_r2_delta,
        "regularized_r2_status": regularized_r2_status,
        "regularized_subspace_delta": regularized_subspace_delta,
        "regularized_subspace_status": regularized_subspace_status,
        "aggregation_reliability_at_max_N": _maybe_float(agg.get("aggregation_reliability_at_max_N")),
        "aggregation_reliability_max": _maybe_float(agg.get("aggregation_reliability_max")),
        "max_N_windows": _maybe_float(agg.get("max_N_windows")),
        "stage4_explained_by_aggregation": str(agg.get("stage4_explained_by_aggregation", "")),
        "stage4_metric_name": str(stage4.get("stage4_metric_name", "")),
        "stage4_comparison_status": str(stage4.get("comparison_status", "")),
        "support_label": support_label,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output_dir: Path, session_rows: list[dict[str, object]]) -> None:
    sessions = "; ".join(str(row["session"]) for row in session_rows)
    reg_r2 = np.array([_maybe_float(row["regularized_r2_delta"]) for row in session_rows], dtype=np.float64)
    reg_sub = np.array([_maybe_float(row["regularized_subspace_delta"]) for row in session_rows], dtype=np.float64)
    support_count = sum(str(row["support_label"]) == "image_conditioned_support" for row in session_rows)
    lines = [
        "# Phase 1 Derivative Geometry Refinement",
        "",
        "This branch is a derivative-geometry summary over cached Phase 1 outputs.",
        "",
        "## Scope",
        "",
        "- shared-vs-image-specific cross-validation from the cached covariance geometry outputs",
        "- reliability-normalized deltas for cached image-specific and low-rank residual summary metrics",
        "- model-alignment diagnostics against the same cached sessions",
        "",
        "## Session Coverage",
        "",
        f"- sessions: {sessions}",
        f"- image-conditioned-support sessions: {support_count}/{len(session_rows)}",
        f"- median regularized r2 delta: {float(np.nanmedian(reg_r2)):.6f}",
        f"- median regularized subspace delta: {float(np.nanmedian(reg_sub)):.6f}",
        "",
        "## Interpretation",
        "",
        "The cached evidence continues to favor the shared covariance / supporting-geometry interpretation over a cleanly separated image-conditioned derivative-geometry claim.",
        "The regularized image-specific deltas are kept as diagnostics and not promoted to a headline mechanism.",
        "No new regularized/hierarchical fitting is performed in this script; it summarizes prior outputs.",
        "",
        "## Stop Rule",
        "",
        "If later runs do not turn the regularized deltas positive in a reproducible way, keep this branch in the supporting category and do not widen the slice without a new decision table.",
        "",
    ]
    (output_dir / "derivative_geometry_readme.md").write_text("\n".join(lines))


def _write_figures(output_dir: Path, session_rows: list[dict[str, object]]) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sessions = [str(row["session"]) for row in session_rows]
    reg_r2 = np.array([_maybe_float(row["regularized_r2_delta"]) for row in session_rows], dtype=np.float64)
    reg_sub = np.array([_maybe_float(row["regularized_subspace_delta"]) for row in session_rows], dtype=np.float64)
    agg_rel = np.array([_maybe_float(row["aggregation_reliability_at_max_N"]) for row in session_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    x = np.arange(len(session_rows))
    width = 0.36
    ax.bar(x - width / 2, reg_r2, width=width, label="regularized r2 delta")
    ax.bar(x + width / 2, reg_sub, width=width, label="regularized subspace delta")
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=20, ha="right")
    ax.set_ylabel("Reliability-normalized delta")
    ax.set_title("Derivative geometry refinement: regularized session deltas")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "regularized_session_deltas.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(agg_rel, reg_r2, s=60, label="r2 delta", alpha=0.85)
    for idx, session in enumerate(sessions):
        ax.annotate(session.replace("Allen_", "A"), (agg_rel[idx], reg_r2[idx]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    ax.axvline(float(np.nanmedian(agg_rel)), color="0.5", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Aggregation reliability at max N")
    ax.set_ylabel("Regularized r2 delta")
    ax.set_title("Image-specific gain vs aggregation reliability")
    fig.tight_layout()
    fig.savefig(figures_dir / "aggregation_vs_regularized_r2.png", dpi=200)
    plt.close(fig)


def _write_decision_table(output_dir: Path, session_rows: list[dict[str, object]]) -> None:
    support_count = sum(str(row["support_label"]) == "image_conditioned_support" for row in session_rows)
    rows = [
        {
            "row": "2D_covariance_geometry_refinement",
            "headline_worthy": "no",
            "supporting": "yes",
            "null": "no",
            "reason": (
                "Regularized image-specific and low-rank residual comparisons remain mixed after reliability normalization, "
                "so the shared covariance geometry interpretation stays the cleaner summary."
            ),
            "sessions_supporting": ";".join(str(row["session"]) for row in session_rows),
            "controls_passed": "partial",
            "manuscript_implication": "covariance_geometry_support",
            "next_action": "keep_as_supporting_and_prioritize_active_sensing_branch" if support_count < len(session_rows) else "document_as_stronger_image_conditioned_branch",
        }
    ]
    _write_csv(output_dir / "derivative_geometry_decision_table.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 1 derivative-geometry summary outputs from cached session artifacts.")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-label", type=str, default="20260530")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted(p for p in args.input_root.glob("Allen_*") if p.is_dir())
    session_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    regularized_rows: list[dict[str, object]] = []
    model_alignment_rows: list[dict[str, object]] = []

    for session_dir in session_dirs:
        session = session_dir.name
        bundle = _session_bundle(session_dir)
        row = _session_output_rows(session, bundle)
        session_rows.append(row)

        context_rows.append(
            {
                "session": session,
                "mc_context_support_tier": row["mc_context_support_tier"],
                "mc_fit_mode": row["mc_fit_mode"],
                "model_1_vs_2_status": str(bundle["session_metrics"].get("model_1_vs_2_status", "")),
                "aggregation_reliability_at_max_N": row["aggregation_reliability_at_max_N"],
                "max_N_windows": row["max_N_windows"],
                "regularized_r2_delta": row["regularized_r2_delta"],
                "regularized_subspace_delta": row["regularized_subspace_delta"],
                "support_label": row["support_label"],
            }
        )

        regularized_rows.append(
            {
                "session": session,
                "mean_r2_shared": row["mean_r2_shared"],
                "mean_r2_image_specific": row["mean_r2_image_specific"],
                "r2_delta_image_vs_shared": row["r2_delta_image_vs_shared"],
                "regularized_r2_delta": row["regularized_r2_delta"],
                "mean_subspace_delta": row["mean_subspace_delta"],
                "regularized_subspace_delta": row["regularized_subspace_delta"],
                "aggregation_reliability_at_max_N": row["aggregation_reliability_at_max_N"],
                "reliability_ceiling": row["reliability_ceiling"],
                "support_label": row["support_label"],
            }
        )

        model_alignment_rows.append(
            {
                "session": session,
                "model_alignment": row["model_alignment"],
                "model_shuffle_alignment": row["model_shuffle_alignment"],
                "primary_vs_sensitivity_cov_corr": row["primary_vs_sensitivity_cov_corr"],
                "primary_vs_sensitivity_subspace_overlap": row["primary_vs_sensitivity_subspace_overlap"],
                "b_emp_top2_fraction": row["b_emp_top2_fraction"],
                "mc_top2_fraction": row["mc_top2_fraction"],
                "reliability_ceiling": row["reliability_ceiling"],
                "support_label": row["support_label"],
            }
        )

    _write_csv(args.output_dir / "derivative_geometry_session_summary.csv", session_rows)
    _write_csv(args.output_dir / "derivative_geometry_context_summary.csv", context_rows)
    _write_csv(args.output_dir / "shared_vs_image_specific_regularized.csv", regularized_rows)
    _write_csv(args.output_dir / "model_derivative_alignment.csv", model_alignment_rows)
    _write_readme(args.output_dir, session_rows)
    _write_figures(args.output_dir, session_rows)
    _write_decision_table(args.output_dir, session_rows)

    print(f"Saved derivative-geometry refinement outputs to {args.output_dir}")


if __name__ == "__main__":
    main()