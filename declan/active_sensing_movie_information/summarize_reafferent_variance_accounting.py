"""Summarize existing reafferent variance-accounting outputs for Figure 5.

This is a denominator dashboard, not a raw covariance recomputation.  It pulls
existing Phase 1, direct recorded-derivative, and finite-difference closure
tables into a small set of explicitly labelled evidence rows.

The important guardrail is that rows do not all share the same denominator.
The output labels each row's evidence class and denominator so draft language
does not accidentally collapse heterogeneous fractions into one headline
number.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE1_DIR = ROOT / "outputs" / "phase1_fem_covariance"
DEFAULT_DIRECT_DIR = ROOT / "outputs" / "direct_recorded_derivative_twin_alignment_prod"
DEFAULT_FD_DIR = ROOT / "outputs" / "matched_twin_covariance_closure_fd_allen_step025"
DEFAULT_OUT_DIR = ROOT / "outputs" / "active_sensing_movie_information" / "reafferent_variance_accounting"


def parse_csv_text(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def stable_sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def status_from_fraction(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 0:
        return "negative_or_counterevidence"
    if value < 0.10:
        return "small"
    if value < 0.30:
        return "moderate"
    return "large"


def evidence_row(
    *,
    source_table: str,
    session: str,
    evidence_class: str,
    metric: str,
    numerator: float,
    denominator: float,
    fraction: float,
    row_status: str,
    notes: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_table": source_table,
        "session": session,
        "evidence_class": evidence_class,
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "fraction": fraction,
        "fraction_status": status_from_fraction(fraction),
        "row_status": row_status,
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def phase1_rows(phase1_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = phase1_dir / "summaries" / "phase1_master_summary.csv"
    rows = read_csv_rows(path)
    evidence: list[dict[str, Any]] = []
    rollup: list[dict[str, Any]] = []
    for row in rows:
        session = str(row.get("session", ""))
        aggregation = fnum(row, "aggregation_reliability_at_max_N")
        eye_shuffle = fnum(row, "eye_shuffle_reliability_at_max_N")
        true_minus_eye = fnum(row, "true_minus_eye_shuffle_reliability_at_max_N")
        if np.isfinite(aggregation) and aggregation != 0:
            frac = true_minus_eye / aggregation
        else:
            frac = float("nan")
        evidence.append(
            evidence_row(
                source_table=str(path),
                session=session,
                evidence_class="reliable_shared_denominator_proxy",
                metric="aggregation_true_minus_eye_shuffle_fraction",
                numerator=true_minus_eye,
                denominator=aggregation,
                fraction=frac,
                row_status=str(row.get("aggregation_status", "")),
                notes=(
                    "Fraction of aggregation reliability above eye-shuffle reliability. "
                    "This is denominator-like but not yet a raw covariance trace fraction."
                ),
                extra={
                    "n_units": fnum(row, "n_units"),
                    "n_valid_windows": fnum(row, "n_valid_windows"),
                    "aggregation_reliability_full": fnum(row, "aggregation_reliability_full"),
                    "aggregation_reliability_at_max_N": aggregation,
                    "eye_shuffle_reliability_at_max_N": eye_shuffle,
                },
            )
        )

        raw_noise_corr = fnum(row, "raw_noise_corr_median")
        eye_corrected_corr = fnum(row, "eye_corrected_corr_median")
        noise_delta = raw_noise_corr - eye_corrected_corr
        noise_frac = noise_delta / raw_noise_corr if np.isfinite(raw_noise_corr) and raw_noise_corr != 0 else fnum(row, "noise_corr_reduction_fraction")
        evidence.append(
            evidence_row(
                source_table=str(path),
                session=session,
                evidence_class="noise_correlation_eye_correction_proxy",
                metric="noise_corr_reduction_fraction",
                numerator=noise_delta,
                denominator=raw_noise_corr,
                fraction=noise_frac,
                row_status="ok" if np.isfinite(noise_frac) else "missing",
                notes=(
                    "Median pairwise noise-correlation reduction after eye correction. "
                    "Useful corroborating denominator evidence, but pairwise-correlation based."
                ),
                extra={
                    "raw_noise_corr_median": raw_noise_corr,
                    "eye_corrected_corr_median": eye_corrected_corr,
                    "source_noise_corr_reduction_field": fnum(row, "noise_corr_reduction"),
                    "noise_corr_fisher_reduction_fraction": fnum(row, "noise_corr_fisher_reduction_fraction"),
                },
            )
        )

        reliability_ceiling = fnum(row, "reliability_ceiling")
        model_alignment = fnum(row, "model_alignment")
        image_shuffle_alignment = fnum(row, "image_shuffle_alignment")
        alignment_excess = model_alignment - image_shuffle_alignment
        ceiling_frac = alignment_excess / reliability_ceiling if np.isfinite(reliability_ceiling) and reliability_ceiling != 0 else float("nan")
        evidence.append(
            evidence_row(
                source_table=str(path),
                session=session,
                evidence_class="model_alignment_excess_proxy",
                metric="model_alignment_excess_over_image_shuffle_per_reliability_ceiling",
                numerator=alignment_excess,
                denominator=reliability_ceiling,
                fraction=ceiling_frac,
                row_status=str(row.get("alignment_norm_status", "")),
                notes=(
                    "Existing model-alignment excess normalized by split-half reliability ceiling. "
                    "This is not a variance-trace denominator but tracks model-linked reafferent structure."
                ),
                extra={
                    "model_alignment": model_alignment,
                    "image_shuffle_alignment": image_shuffle_alignment,
                    "ceiling_normalized_alignment_existing": fnum(row, "ceiling_normalized_alignment"),
                },
            )
        )

        rollup.append(
            {
                "session": session,
                "n_units": fnum(row, "n_units"),
                "n_valid_windows": fnum(row, "n_valid_windows"),
                "aggregation_reliability_at_max_N": aggregation,
                "true_minus_eye_shuffle_reliability_at_max_N": true_minus_eye,
                "aggregation_true_minus_eye_fraction": frac,
                "noise_corr_reduction_fraction": noise_frac,
                "model_alignment_excess_ceiling_fraction": ceiling_frac,
                "phase1_recommendation": row.get("phase1_recommendation", ""),
                "status": row.get("status", ""),
            }
        )
    return evidence, rollup


def direct_derivative_rows(direct_dir: Path, k_values: set[int], projection_controls: set[str]) -> list[dict[str, Any]]:
    path = direct_dir / "tier1_compact_basis_capture.csv"
    rows = read_csv_rows(path)
    evidence: list[dict[str, Any]] = []
    for row in rows:
        k = int(fnum(row, "k", -1))
        projection = str(row.get("projection_control", ""))
        if k_values and k not in k_values:
            continue
        if projection_controls and projection not in projection_controls:
            continue
        if str(row.get("target_variant", "")) != "psd":
            continue
        if str(row.get("row_status", "")) != "ok":
            continue
        capture = fnum(row, "capture")
        evidence.append(
            evidence_row(
                source_table=str(path),
                session=str(row.get("session", "")),
                evidence_class="compact_reafferent_numerator_candidate",
                metric="direct_recorded_derivative_compact_capture",
                numerator=capture,
                denominator=1.0,
                fraction=capture,
                row_status=str(row.get("row_status", "")),
                notes=(
                    "Fraction of target covariance captured by compact derivative basis within a context. "
                    "Good numerator candidate; denominator is target covariance for that context, not full reliable shared covariance."
                ),
                extra={
                    "subject": row.get("subject", ""),
                    "target_variant": row.get("target_variant", ""),
                    "projection_control": projection,
                    "context_id": row.get("context_id", ""),
                    "context_label": row.get("context_label", ""),
                    "n_samples": fnum(row, "n_samples"),
                    "k": k,
                    "split_half_excess": fnum(row, "split_half_excess"),
                    "reliability_qualified": row.get("reliability_qualified", ""),
                    "effect_minus_random_subspace_median": fnum(row, "effect_minus_random_subspace_median"),
                    "effect_minus_unit_shuffle_median": fnum(row, "effect_minus_unit_shuffle_median"),
                    "effect_minus_rf_readout_median": fnum(row, "effect_minus_rf_readout_median"),
                },
            )
        )
    return evidence


def finite_difference_rows(fd_dir: Path, k_values: set[int], projection_controls: set[str]) -> list[dict[str, Any]]:
    path = fd_dir / "finite_difference_capture_metrics.csv"
    rows = read_csv_rows(path)
    evidence: list[dict[str, Any]] = []
    preferred_basis = {"fd_sample_eye_trace_cov", "fd_mean_tangent_cov", "fd_mean_tangent_matrix"}
    for row in rows:
        k = int(fnum(row, "k", -1))
        projection = str(row.get("projection_control", ""))
        if k_values and k not in k_values:
            continue
        if projection_controls and projection not in projection_controls:
            continue
        if str(row.get("target_variant", "")) != "psd":
            continue
        if str(row.get("row_status", "")) != "ok":
            continue
        basis_source = str(row.get("basis_source", ""))
        if basis_source not in preferred_basis:
            continue
        capture = fnum(row, "capture")
        evidence.append(
            evidence_row(
                source_table=str(path),
                session=str(row.get("session", "")),
                evidence_class="finite_difference_reafferent_numerator_candidate",
                metric="finite_difference_tangent_capture",
                numerator=capture,
                denominator=1.0,
                fraction=capture,
                row_status=str(row.get("row_status", "")),
                notes=(
                    "Fraction of target covariance captured by finite-difference tangent basis. "
                    "Good mechanistic numerator candidate; denominator is target covariance, not full reliable shared covariance."
                ),
                extra={
                    "subject": row.get("subject", ""),
                    "target_variant": row.get("target_variant", ""),
                    "projection_control": projection,
                    "basis_source": basis_source,
                    "window_idx": row.get("window_idx", ""),
                    "n_common_units": fnum(row, "n_common_units"),
                    "n_samples_used": fnum(row, "n_samples_used"),
                    "k": k,
                    "target_trace": fnum(row, "target_trace"),
                    "effect_minus_unit_shuffle_median": fnum(row, "effect_minus_unit_shuffle_median"),
                    "effect_minus_random_subspace_median": fnum(row, "effect_minus_random_subspace_median"),
                },
            )
        )
    return evidence


def _finite_difference_trace_source_rows(
    fd_dir: Path,
    k_values: set[int],
    projection_controls: set[str],
) -> list[dict[str, str]]:
    path = fd_dir / "finite_difference_capture_metrics.csv"
    rows = read_csv_rows(path)
    preferred_basis = {"fd_sample_eye_trace_cov", "fd_mean_tangent_cov", "fd_mean_tangent_matrix"}
    out: list[dict[str, str]] = []
    for row in rows:
        k = int(fnum(row, "k", -1))
        projection = str(row.get("projection_control", ""))
        if k_values and k not in k_values:
            continue
        if projection_controls and projection not in projection_controls:
            continue
        if str(row.get("target_variant", "")) != "psd":
            continue
        if str(row.get("row_status", "")) != "ok":
            continue
        if str(row.get("basis_source", "")) not in preferred_basis:
            continue
        out.append(row)
    return out


def finite_difference_trace_closure_rows(
    fd_dir: Path,
    k_values: set[int],
    projection_controls: set[str],
) -> list[dict[str, Any]]:
    path = fd_dir / "finite_difference_capture_metrics.csv"
    rows = _finite_difference_trace_source_rows(fd_dir, k_values, projection_controls)
    out: list[dict[str, Any]] = []
    for row in rows:
        target_trace = fnum(row, "target_trace")
        capture = fnum(row, "capture")
        unit_null = fnum(row, "unit_shuffle_null_median")
        random_null = fnum(row, "random_subspace_null_median")
        excess_unit = fnum(row, "effect_minus_unit_shuffle_median")
        excess_random = fnum(row, "effect_minus_random_subspace_median")
        out.append(
            {
                "source_table": str(path),
                "session": row.get("session", ""),
                "subject": row.get("subject", ""),
                "window_idx": row.get("window_idx", ""),
                "target_variant": row.get("target_variant", ""),
                "projection_control": row.get("projection_control", ""),
                "basis_source": row.get("basis_source", ""),
                "basis_rank": fnum(row, "basis_rank"),
                "k": int(fnum(row, "k", -1)),
                "n_common_units": fnum(row, "n_common_units"),
                "n_samples_used": fnum(row, "n_samples_used"),
                "target_trace": target_trace,
                "target_trace_raw": fnum(row, "target_trace_raw"),
                "target_trace_psd": fnum(row, "target_trace_psd"),
                "capture_fraction": capture,
                "captured_trace": capture * target_trace,
                "unit_shuffle_null_fraction": unit_null,
                "unit_shuffle_null_trace": unit_null * target_trace,
                "random_subspace_null_fraction": random_null,
                "random_subspace_null_trace": random_null * target_trace,
                "excess_over_unit_shuffle_fraction": excess_unit,
                "excess_over_unit_shuffle_trace": excess_unit * target_trace,
                "excess_over_random_subspace_fraction": excess_random,
                "excess_over_random_subspace_trace": excess_random * target_trace,
                "row_status": row.get("row_status", ""),
                "trace_denominator_scope": "matched_psd_target_covariance_trace",
                "guardrail": (
                    "Trace units are relative to the finite-difference target covariance, "
                    "not the full reliable shared covariance denominator."
                ),
            }
        )
    return out


def aggregate_trace_closure(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("projection_control", "")),
            str(row.get("basis_source", "")),
            int(fnum(row, "k", -1)),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (projection_control, basis_source, k), group in sorted(groups.items()):
        target_traces = [fnum(row, "target_trace") for row in group]
        captured_traces = [fnum(row, "captured_trace") for row in group]
        unit_null_traces = [fnum(row, "unit_shuffle_null_trace") for row in group]
        random_null_traces = [fnum(row, "random_subspace_null_trace") for row in group]
        excess_unit_traces = [fnum(row, "excess_over_unit_shuffle_trace") for row in group]
        excess_random_traces = [fnum(row, "excess_over_random_subspace_trace") for row in group]
        capture_fractions = [fnum(row, "capture_fraction") for row in group]
        excess_unit_fractions = [fnum(row, "excess_over_unit_shuffle_fraction") for row in group]
        excess_random_fractions = [fnum(row, "excess_over_random_subspace_fraction") for row in group]

        total_target_trace = float(np.nansum(np.asarray(target_traces, dtype=np.float64)))
        total_captured_trace = float(np.nansum(np.asarray(captured_traces, dtype=np.float64)))
        total_unit_null_trace = float(np.nansum(np.asarray(unit_null_traces, dtype=np.float64)))
        total_random_null_trace = float(np.nansum(np.asarray(random_null_traces, dtype=np.float64)))
        total_excess_unit_trace = float(np.nansum(np.asarray(excess_unit_traces, dtype=np.float64)))
        total_excess_random_trace = float(np.nansum(np.asarray(excess_random_traces, dtype=np.float64)))

        by_session: dict[str, list[float]] = {}
        for row in group:
            session = str(row.get("session", ""))
            if session:
                by_session.setdefault(session, []).append(fnum(row, "capture_fraction"))
        session_capture_means = [stable_mean(vals) for vals in by_session.values()]

        out.append(
            {
                "projection_control": projection_control,
                "basis_source": basis_source,
                "k": k,
                "n_rows": len(group),
                "n_sessions": len(by_session),
                "total_target_trace": total_target_trace,
                "total_captured_trace": total_captured_trace,
                "total_unit_shuffle_null_trace": total_unit_null_trace,
                "total_random_subspace_null_trace": total_random_null_trace,
                "total_excess_over_unit_shuffle_trace": total_excess_unit_trace,
                "total_excess_over_random_subspace_trace": total_excess_random_trace,
                "trace_weighted_capture_fraction": total_captured_trace / total_target_trace if total_target_trace else float("nan"),
                "trace_weighted_unit_shuffle_null_fraction": total_unit_null_trace / total_target_trace if total_target_trace else float("nan"),
                "trace_weighted_random_subspace_null_fraction": total_random_null_trace / total_target_trace if total_target_trace else float("nan"),
                "trace_weighted_excess_over_unit_shuffle_fraction": total_excess_unit_trace / total_target_trace if total_target_trace else float("nan"),
                "trace_weighted_excess_over_random_subspace_fraction": total_excess_random_trace / total_target_trace if total_target_trace else float("nan"),
                "capture_fraction_row_mean": stable_mean(capture_fractions),
                "capture_fraction_row_sem": stable_sem(capture_fractions),
                "capture_fraction_session_mean": stable_mean(session_capture_means),
                "capture_fraction_session_sem": stable_sem(session_capture_means),
                "excess_over_unit_shuffle_fraction_row_mean": stable_mean(excess_unit_fractions),
                "excess_over_random_subspace_fraction_row_mean": stable_mean(excess_random_fractions),
                "trace_denominator_scope": "matched_psd_target_covariance_trace",
                "guardrail": (
                    "This is a trace-unit finite-difference closure for the matched target covariance. "
                    "It is not yet tr(C_reaff_explained) / tr(C_reliable_shared)."
                ),
            }
        )
    return out


def aggregate_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("evidence_class", "")), str(row.get("metric", "")))
        groups.setdefault(key, []).append(row)
    out = []
    for (evidence_class, metric), group in sorted(groups.items()):
        vals = [fnum(row, "fraction") for row in group]
        by_session: dict[str, list[float]] = {}
        for row in group:
            session = str(row.get("session", ""))
            if not session:
                continue
            by_session.setdefault(session, []).append(fnum(row, "fraction"))
        session_vals = [stable_mean(session_group) for session_group in by_session.values()]
        out.append(
            {
                "evidence_class": evidence_class,
                "metric": metric,
                "n_rows": len(group),
                "n_sessions": len(by_session),
                "fraction_row_mean": stable_mean(vals),
                "fraction_row_sem": stable_sem(vals),
                "fraction_session_mean": stable_mean(session_vals),
                "fraction_session_sem": stable_sem(session_vals),
                "fraction_median": float(np.nanmedian(np.asarray(vals, dtype=np.float64))) if vals else float("nan"),
                "fraction_min": float(np.nanmin(np.asarray(vals, dtype=np.float64))) if vals else float("nan"),
                "fraction_max": float(np.nanmax(np.asarray(vals, dtype=np.float64))) if vals else float("nan"),
                "notes": str(group[0].get("notes", "")),
            }
        )
    return out


def write_summary_markdown(
    path: Path,
    aggregate_rows_out: list[dict[str, Any]],
    rollup_rows: list[dict[str, Any]],
    trace_summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Reafferent Variance Accounting Summary",
        "",
        "This is a fast summary over existing outputs. It is not a raw covariance recomputation.",
        "",
        "## Aggregate Evidence",
        "",
        "| Evidence class | Metric | n rows | n sessions | session-mean fraction | session SEM | row-mean fraction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows_out:
        lines.append(
            "| {evidence_class} | {metric} | {n_rows} | {n_sessions} | {session_mean:.3f} | {session_sem:.3f} | {row_mean:.3f} |".format(
                evidence_class=row["evidence_class"],
                metric=row["metric"],
                n_rows=int(row["n_rows"]),
                n_sessions=int(row["n_sessions"]),
                session_mean=fnum(row, "fraction_session_mean"),
                session_sem=fnum(row, "fraction_session_sem"),
                row_mean=fnum(row, "fraction_row_mean"),
            )
        )
    lines.extend([
        "",
        "## Session Rollup",
        "",
        "| Session | aggregation FEM-linked fraction | noise-corr reduction fraction | model alignment excess / reliability |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in rollup_rows:
        lines.append(
            "| {session} | {agg:.3f} | {noise:.3f} | {align:.3f} |".format(
                session=row.get("session", ""),
                agg=fnum(row, "aggregation_true_minus_eye_fraction"),
                noise=fnum(row, "noise_corr_reduction_fraction"),
                align=fnum(row, "model_alignment_excess_ceiling_fraction"),
            )
        )
    lines.extend([
        "",
        "## Finite-Difference Trace Closure",
        "",
        "Trace units below use the matched PSD target covariance saved by the finite-difference closure run.",
        "They are useful for numerator accounting, but they are not the full reliable-shared covariance denominator.",
        "",
        "| Projection | Basis | k | n rows | target trace | captured trace | capture frac | excess vs unit shuffle frac | excess vs random frac |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in trace_summary_rows:
        lines.append(
            "| {projection} | {basis} | {k} | {n_rows} | {target:.3f} | {captured:.3f} | {capture:.3f} | {unit:.3f} | {random:.3f} |".format(
                projection=row.get("projection_control", ""),
                basis=row.get("basis_source", ""),
                k=int(fnum(row, "k")),
                n_rows=int(fnum(row, "n_rows")),
                target=fnum(row, "total_target_trace"),
                captured=fnum(row, "total_captured_trace"),
                capture=fnum(row, "trace_weighted_capture_fraction"),
                unit=fnum(row, "trace_weighted_excess_over_unit_shuffle_fraction"),
                random=fnum(row, "trace_weighted_excess_over_random_subspace_fraction"),
            )
        )
    lines.extend([
        "",
        "## Guardrail",
        "",
        "Rows above do not all share the same denominator. Use the evidence-class labels.",
        "The denominator priority remains a raw reliable-shared-covariance trace fraction.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = {int(k) for k in parse_csv_text(args.k_list)}
    projection_controls = set(parse_csv_text(args.projection_controls))

    evidence_rows: list[dict[str, Any]] = []
    phase_evidence, rollup_rows = phase1_rows(Path(args.phase1_dir))
    evidence_rows.extend(phase_evidence)
    evidence_rows.extend(direct_derivative_rows(Path(args.direct_dir), k_values, projection_controls))
    evidence_rows.extend(finite_difference_rows(Path(args.fd_dir), k_values, projection_controls))
    aggregate_rows_out = aggregate_evidence(evidence_rows)
    trace_rows = finite_difference_trace_closure_rows(Path(args.fd_dir), k_values, projection_controls)
    trace_summary_rows = aggregate_trace_closure(trace_rows)

    write_csv_rows(out_dir / "variance_accounting_component_candidates.csv", evidence_rows)
    write_csv_rows(out_dir / "variance_accounting_session_rollup.csv", rollup_rows)
    write_csv_rows(out_dir / "variance_accounting_aggregate_summary.csv", aggregate_rows_out)
    write_csv_rows(out_dir / "variance_accounting_trace_closure.csv", trace_rows)
    write_csv_rows(out_dir / "variance_accounting_trace_closure_summary.csv", trace_summary_rows)
    write_summary_markdown(out_dir / "variance_accounting_summary.md", aggregate_rows_out, rollup_rows, trace_summary_rows)
    manifest = {
        "phase1_dir": str(args.phase1_dir),
        "direct_dir": str(args.direct_dir),
        "fd_dir": str(args.fd_dir),
        "out_dir": str(args.out_dir),
        "k_list": sorted(k_values),
        "projection_controls": sorted(projection_controls),
        "outputs": [
            "variance_accounting_component_candidates.csv",
            "variance_accounting_session_rollup.csv",
            "variance_accounting_aggregate_summary.csv",
            "variance_accounting_trace_closure.csv",
            "variance_accounting_trace_closure_summary.csv",
            "variance_accounting_summary.md",
        ],
        "guardrail": "Rows have heterogeneous denominators; do not collapse into one headline fraction without a matched denominator.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Reafferent variance-accounting summary complete")
    print(f"  out_dir: {out_dir}")
    print(f"  evidence rows: {len(evidence_rows)}")
    print(f"  session rollup rows: {len(rollup_rows)}")
    print(f"  aggregate rows: {len(aggregate_rows_out)}")
    print(f"  trace closure rows: {len(trace_rows)}")
    print(f"  trace closure summary rows: {len(trace_summary_rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--direct-dir", type=Path, default=DEFAULT_DIRECT_DIR)
    parser.add_argument("--fd-dir", type=Path, default=DEFAULT_FD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k-list", type=str, default="2,10,20")
    parser.add_argument("--projection-controls", type=str, default="none,global_rate,global_rate+target_pc1")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
