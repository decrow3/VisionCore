#!/usr/bin/env python3
"""Summarize correct-chart swap alignment outputs."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from declan.compact_retinal_translation_geometry.run_correct_chart_swap_alignment import (
    DEFAULT_OUTPUT_ROOT,
    _primary_decision,
    _summarize_pair_rows,
    _write_figures,
    write_csv,
    write_json,
)
from declan.direct_recorded_derivative_twin_alignment.run_direct_recorded_derivative_alignment import (
    bootstrap_mean_ci,
    sign_test_p_two_sided,
)


def _coerce_value(value: str) -> Any:
    text = str(value)
    if text == "":
        return ""
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.lower() == "nan":
        return float("nan")
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return value


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: _coerce_value(val) for key, val in row.items()} for row in csv.DictReader(handle)]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _value_stratification_rows(
    pair_rows: list[dict[str, Any]],
    *,
    value_column: str,
    bin_prefix: str,
    value_mean_column: str,
    seed: int,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    metrics = [
        "true_minus_wrong",
        "true_minus_gain",
        "true_minus_random",
        "true_minus_unit_shuffle",
        "true_minus_rf_readout",
        "true_minus_shuffled_eye",
    ]
    base_groups: dict[tuple[str, str, int, str, str, str], list[dict[str, Any]]] = {}
    for row in pair_rows:
        for sample_set in ("all", "drift_only"):
            if sample_set == "drift_only" and not bool(row.get("drift_mask", False)):
                continue
            key = (
                str(row.get("session")),
                str(row.get("projection_control")),
                int(row.get("basis_k", 0)),
                str(row.get("chart_space")),
                str(row.get("unit_score_subset", "all_units")),
                sample_set,
            )
            base_groups.setdefault(key, []).append(row)

    session_rows: list[dict[str, Any]] = []
    for (session, projection, k, chart_space, unit_subset, sample_set), rows in sorted(base_groups.items()):
        norms = np.asarray([float(r.get(value_column, float("nan"))) for r in rows], dtype=np.float64)
        finite_norms = norms[np.isfinite(norms)]
        if finite_norms.size == 0:
            continue
        thresholds = {
            "all": float("-inf"),
            f"{bin_prefix}_top50": float(np.nanpercentile(finite_norms, 50)),
            f"{bin_prefix}_top25": float(np.nanpercentile(finite_norms, 75)),
            f"{bin_prefix}_top10": float(np.nanpercentile(finite_norms, 90)),
        }
        for bin_name, threshold in thresholds.items():
            block = [
                row
                for row, norm in zip(rows, norms, strict=False)
                if np.isfinite(norm) and (bin_name == "all" or norm >= threshold)
            ]
            if not block:
                continue
            base: dict[str, Any] = {
                "row_type": "session",
                "session": session,
                "projection_control": projection,
                "basis_k": int(k),
                "chart_space": chart_space,
                "unit_score_subset": unit_subset,
                "sample_set": sample_set,
                "stratification": value_column,
                "bin": bin_name,
                "threshold": threshold if np.isfinite(threshold) else float("nan"),
                "n_pairs": int(len(block)),
                value_mean_column: float(np.nanmean([float(r.get(value_column, float("nan"))) for r in block])),
            }
            for metric in metrics:
                vals = np.asarray([float(r.get(metric, float("nan"))) for r in block], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                base[f"mean_{metric}"] = float(np.mean(vals)) if vals.size else float("nan")
                base[f"median_{metric}"] = float(np.median(vals)) if vals.size else float("nan")
                base[f"n_positive_{metric}"] = int(np.sum(vals > 0.0)) if vals.size else 0
                base[f"sign_test_{metric}_p_two_sided"] = sign_test_p_two_sided(int(np.sum(vals > 0.0)), int(vals.size))
            session_rows.append(base)

    rng = np.random.default_rng(int(seed))
    aggregate_groups: dict[tuple[str, int, str, str, str, str], list[dict[str, Any]]] = {}
    for row in session_rows:
        key = (
            str(row["projection_control"]),
            int(row["basis_k"]),
            str(row["chart_space"]),
            str(row.get("unit_score_subset", "all_units")),
            str(row["sample_set"]),
            str(row["bin"]),
        )
        aggregate_groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (projection, k, chart_space, unit_subset, sample_set, bin_name), rows in sorted(aggregate_groups.items()):
        for metric in metrics:
            vals = np.asarray([float(r.get(f"mean_{metric}", float("nan"))) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            mean, lo, hi = bootstrap_mean_ci(vals, rng=rng, n_bootstrap=int(n_bootstrap))
            aggregate_rows.append(
                {
                    "row_type": "aggregate",
                    "session": "ALL",
                    "projection_control": projection,
                    "basis_k": int(k),
                    "chart_space": chart_space,
                    "unit_score_subset": unit_subset,
                    "sample_set": sample_set,
                    "stratification": value_column,
                    "bin": bin_name,
                    "metric": metric,
                    "n_sessions": int(vals.size),
                    "session_mean": mean,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "n_positive_sessions": int(np.sum(vals > 0.0)),
                    "sign_test_p_two_sided": sign_test_p_two_sided(int(np.sum(vals > 0.0)), int(vals.size)),
                }
            )
    return session_rows + aggregate_rows


def _q_norm_stratification_rows(
    pair_rows: list[dict[str, Any]],
    *,
    seed: int,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    rows = _value_stratification_rows(
        pair_rows,
        value_column="prediction_norm_true",
        bin_prefix="qtrue",
        value_mean_column="prediction_norm_true_mean",
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    for row in rows:
        if "bin" in row:
            row["q_norm_bin"] = row["bin"]
        if "threshold" in row:
            row["q_norm_threshold"] = row["threshold"]
    return rows


def _pseudo_bootstrap_rows(session_rows: list[dict[str, Any]], *, seed: int, n_bootstrap: int) -> list[dict[str, Any]]:
    metrics = [
        "true_minus_wrong",
        "true_minus_gain",
        "true_minus_random",
        "true_minus_unit_shuffle",
        "true_minus_rf_readout",
        "true_minus_shuffled_eye",
    ]
    groups: dict[tuple[str, int, str, str, str, str, str, float, float, str], list[dict[str, Any]]] = {}
    for row in session_rows:
        if "mean_true_minus_wrong" not in row:
            continue
        key = (
            str(row.get("projection_control")),
            int(row.get("basis_k", 0)),
            str(row.get("chart_space")),
            str(row.get("unit_score_subset", "all_units")),
            str(row.get("wrong_chart_pool", "unknown")),
            str(row.get("wrong_chart_match_features", "norm_only")),
            str(row.get("pseudo_control_mode", "poisson")),
            float(row.get("pseudo_control_scale", 1.0)),
            float(row.get("pseudo_injection_noise_sd", 0.0)),
            str(row.get("sample_set")),
        )
        groups.setdefault(key, []).append(row)

    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    for (
        projection,
        k,
        chart_space,
        unit_subset,
        wrong_pool,
        wrong_match,
        pseudo_mode,
        pseudo_scale,
        pseudo_noise_sd,
        sample_set,
    ), rows in sorted(groups.items()):
        for metric in metrics:
            vals = np.asarray([float(r.get(f"mean_{metric}", float("nan"))) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            mean, lo, hi = bootstrap_mean_ci(vals, rng=rng, n_bootstrap=int(n_bootstrap))
            out.append(
                {
                    "projection_control": projection,
                    "basis_k": int(k),
                    "chart_space": chart_space,
                    "unit_score_subset": unit_subset,
                    "wrong_chart_pool": wrong_pool,
                    "wrong_chart_match_features": wrong_match,
                    "pseudo_control_mode": pseudo_mode,
                    "pseudo_control_scale": float(pseudo_scale),
                    "pseudo_injection_noise_sd": float(pseudo_noise_sd),
                    "sample_set": sample_set,
                    "metric": metric,
                    "n_sessions": int(vals.size),
                    "session_mean": mean,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "n_positive_sessions": int(np.sum(vals > 0.0)),
                    "sign_test_p_two_sided": sign_test_p_two_sided(int(np.sum(vals > 0.0)), int(vals.size)),
                }
            )
    return out


def _variant_label(projection: str, unit_subset: str) -> str:
    return f"{projection}|{unit_subset}"


def _interesting_variants(bootstrap_rows: list[dict[str, Any]], *, primary_k: int) -> list[tuple[str, str]]:
    explicit = [
        ("global_rate+target_pc1", "all_units"),
        ("target_pc1", "all_units"),
        ("global_rate", "fem_tangent_top50"),
    ]
    available = {
        (str(r.get("projection_control")), str(r.get("unit_score_subset", "all_units")))
        for r in bootstrap_rows
        if int(r.get("basis_k", -1)) == int(primary_k)
        and str(r.get("chart_space")) == "compact"
        and str(r.get("sample_set")) == "all"
        and str(r.get("metric")) == "true_minus_wrong"
    }
    selected = [v for v in explicit if v in available]
    for r in bootstrap_rows:
        if (
            int(r.get("basis_k", -1)) == int(primary_k)
            and str(r.get("chart_space")) == "compact"
            and str(r.get("sample_set")) == "all"
            and str(r.get("metric")) == "true_minus_wrong"
        ):
            key = (str(r.get("projection_control")), str(r.get("unit_score_subset", "all_units")))
            ci_low = float(r.get("bootstrap_ci_low", float("nan")))
            n_pos = int(r.get("n_positive_sessions", 0))
            n_sess = int(r.get("n_sessions", 0))
            if (np.isfinite(ci_low) and ci_low > 0.0) or (n_sess > 0 and n_pos == n_sess):
                if key not in selected:
                    selected.append(key)
    return selected


def _targeted_variant_audit_rows(
    *,
    pair_rows: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    primary_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = [
        "true_minus_wrong",
        "true_minus_gain",
        "true_minus_random",
        "true_minus_unit_shuffle",
        "true_minus_rf_readout",
        "true_minus_shuffled_eye",
    ]
    variants = _interesting_variants(bootstrap_rows, primary_k=int(primary_k))
    variant_set = set(variants)

    summary_rows: list[dict[str, Any]] = []
    for projection, unit_subset in variants:
        label = _variant_label(projection, unit_subset)
        rows = [
            r
            for r in bootstrap_rows
            if str(r.get("projection_control")) == projection
            and str(r.get("unit_score_subset", "all_units")) == unit_subset
            and int(r.get("basis_k", -1)) == int(primary_k)
            and str(r.get("chart_space")) == "compact"
            and str(r.get("sample_set")) == "all"
        ]
        by_metric = {str(r.get("metric")): r for r in rows}
        primary = by_metric.get("true_minus_wrong", {})
        required = ["true_minus_gain", "true_minus_random", "true_minus_unit_shuffle", "true_minus_rf_readout"]
        control_ci_lows = [float(by_metric.get(m, {}).get("bootstrap_ci_low", float("nan"))) for m in required]
        controls_pass = bool(control_ci_lows) and all(np.isfinite(v) and v > 0.0 for v in control_ci_lows)
        summary_rows.append(
            {
                "variant": label,
                "projection_control": projection,
                "unit_score_subset": unit_subset,
                "basis_k": int(primary_k),
                "n_unit_score_subset": primary.get("n_unit_score_subset", ""),
                "true_minus_wrong_mean": primary.get("session_mean", float("nan")),
                "true_minus_wrong_ci_low": primary.get("bootstrap_ci_low", float("nan")),
                "true_minus_wrong_ci_high": primary.get("bootstrap_ci_high", float("nan")),
                "true_minus_wrong_positive_sessions": primary.get("n_positive_sessions", 0),
                "n_sessions": primary.get("n_sessions", 0),
                "controls_all_ci_low_gt_zero": controls_pass,
                "min_required_control_ci_low": float(np.nanmin(control_ci_lows)) if control_ci_lows else float("nan"),
                "diagnostic_status": (
                    "specific_positive"
                    if float(primary.get("bootstrap_ci_low", float("nan"))) > 0.0 and controls_pass
                    else "positive_but_control_failed"
                    if float(primary.get("bootstrap_ci_low", float("nan"))) > 0.0
                    else "diagnostic"
                ),
            }
        )

    session_diag_rows: list[dict[str, Any]] = []
    for row in session_rows:
        key = (str(row.get("projection_control")), str(row.get("unit_score_subset", "all_units")))
        if key not in variant_set:
            continue
        if int(row.get("basis_k", -1)) != int(primary_k) or str(row.get("chart_space")) != "compact" or str(row.get("sample_set")) != "all":
            continue
        session_diag_rows.append({"variant": _variant_label(*key), **row})

    leave_one_rows: list[dict[str, Any]] = []
    for projection, unit_subset in variants:
        label = _variant_label(projection, unit_subset)
        rows = [
            r
            for r in session_rows
            if str(r.get("projection_control")) == projection
            and str(r.get("unit_score_subset", "all_units")) == unit_subset
            and int(r.get("basis_k", -1)) == int(primary_k)
            and str(r.get("chart_space")) == "compact"
            and str(r.get("sample_set")) == "all"
        ]
        for metric in metrics:
            vals_by_session = {
                str(r.get("session")): float(r.get(f"mean_{metric}", float("nan")))
                for r in rows
                if np.isfinite(float(r.get(f"mean_{metric}", float("nan"))))
            }
            all_vals = np.asarray(list(vals_by_session.values()), dtype=np.float64)
            if all_vals.size == 0:
                continue
            full_mean = float(np.mean(all_vals))
            for held_session in sorted(vals_by_session):
                kept = np.asarray([v for s, v in vals_by_session.items() if s != held_session], dtype=np.float64)
                leave_one_rows.append(
                    {
                        "variant": label,
                        "projection_control": projection,
                        "unit_score_subset": unit_subset,
                        "basis_k": int(primary_k),
                        "metric": metric,
                        "held_out_session": held_session,
                        "n_sessions_kept": int(kept.size),
                        "full_session_mean": full_mean,
                        "leave_one_session_mean": float(np.mean(kept)) if kept.size else float("nan"),
                        "held_out_session_mean": vals_by_session[held_session],
                    }
                )

    pair_diag_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    pair_keep_cols = [
        "session",
        "fold",
        "trial_i",
        "trial_j",
        "image_id",
        "wrong_image_id",
        "time_context",
        "wrong_time_context",
        "delta_eye_norm",
        "prediction_norm_true",
        "prediction_norm_wrong",
        "wrong_chart_norm_abs_diff",
        "image_structure_score",
        "local_image_structure_score",
        "score_true_chart",
        "score_wrong_chart",
        "score_gain_only",
        "score_random",
        "score_unit_shuffle",
        "score_rf_readout_null",
        "true_minus_wrong",
        "true_minus_gain",
        "true_minus_random",
        "true_minus_unit_shuffle",
        "true_minus_rf_readout",
    ]
    for projection, unit_subset in variants:
        label = _variant_label(projection, unit_subset)
        rows = [
            r
            for r in pair_rows
            if str(r.get("projection_control")) == projection
            and str(r.get("unit_score_subset", "all_units")) == unit_subset
            and int(r.get("basis_k", -1)) == int(primary_k)
            and str(r.get("chart_space")) == "compact"
        ]
        for metric in metrics:
            vals = np.asarray([float(r.get(metric, float("nan"))) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            lo, hi = np.nanpercentile(vals, [5, 95])
            clipped = np.clip(vals, lo, hi)
            pair_diag_rows.append(
                {
                    "variant": label,
                    "projection_control": projection,
                    "unit_score_subset": unit_subset,
                    "basis_k": int(primary_k),
                    "metric": metric,
                    "n_pairs": int(vals.size),
                    "pair_mean": float(np.mean(vals)),
                    "pair_median": float(np.median(vals)),
                    "pair_p05": float(lo),
                    "pair_p95": float(hi),
                    "pair_winsorized_5_95_mean": float(np.mean(clipped)),
                    "pair_n_positive": int(np.sum(vals > 0.0)),
                    "pair_sign_test_p_two_sided": sign_test_p_two_sided(int(np.sum(vals > 0.0)), int(vals.size)),
                    "pair_max_abs": float(np.max(np.abs(vals))),
                }
            )
        sorted_rows = sorted(
            rows,
            key=lambda r: abs(float(r.get("true_minus_wrong", 0.0))) if np.isfinite(float(r.get("true_minus_wrong", float("nan")))) else -1.0,
            reverse=True,
        )
        for rank, row in enumerate(sorted_rows[:20], start=1):
            out = {
                "variant": label,
                "projection_control": projection,
                "unit_score_subset": unit_subset,
                "basis_k": int(primary_k),
                "rank_abs_true_minus_wrong": int(rank),
            }
            for col in pair_keep_cols:
                out[col] = row.get(col, "")
            outlier_rows.append(out)

    return summary_rows, session_diag_rows, leave_one_rows, pair_diag_rows, outlier_rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize correct-chart swap alignment tables")
    p.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--primary-projection-control", type=str, default="global_rate+target_pc1")
    p.add_argument("--primary-k", type=int, default=10)
    p.add_argument("--min-sessions", type=int, default=3)
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    pair_rows = read_csv_rows(root / "chart_alignment_pair_metrics.csv")
    session_rows, bootstrap_rows = _summarize_pair_rows(
        pair_rows,
        seed=int(args.seed),
        n_bootstrap=int(args.n_bootstrap),
    )
    control_summary = [
        r
        for r in bootstrap_rows
        if str(r.get("metric")) in {"true_minus_wrong", "true_minus_random", "true_minus_unit_shuffle", "true_minus_rf_readout", "true_minus_shuffled_eye"}
    ]
    gain_summary = [r for r in bootstrap_rows if str(r.get("metric")) == "true_minus_gain"]
    compact_vs_full = [
        r
        for r in bootstrap_rows
        if str(r.get("metric")) == "true_minus_wrong"
        and int(r.get("basis_k", -1)) in {0, int(args.primary_k)}
    ]
    write_csv(root / "chart_alignment_session_summary.csv", session_rows)
    write_csv(root / "chart_alignment_bootstrap_summary.csv", bootstrap_rows)
    write_csv(root / "chart_swap_control_summary.csv", control_summary)
    write_csv(root / "gain_control_summary.csv", gain_summary)
    write_csv(root / "compact_vs_full_summary.csv", compact_vs_full)
    q_norm_rows = _q_norm_stratification_rows(
        pair_rows,
        seed=int(args.seed) + 101,
        n_bootstrap=int(args.n_bootstrap),
    )
    write_csv(root / "q_norm_stratification.csv", q_norm_rows)
    image_structure_rows = _value_stratification_rows(
        pair_rows,
        value_column="image_structure_score",
        bin_prefix="image_structure",
        value_mean_column="image_structure_score_mean",
        seed=int(args.seed) + 107,
        n_bootstrap=int(args.n_bootstrap),
    )
    write_csv(root / "image_structure_stratification.csv", image_structure_rows)
    local_image_structure_rows = _value_stratification_rows(
        pair_rows,
        value_column="local_image_structure_score",
        bin_prefix="local_image_structure",
        value_mean_column="local_image_structure_score_mean",
        seed=int(args.seed) + 109,
        n_bootstrap=int(args.n_bootstrap),
    )
    write_csv(root / "local_image_structure_stratification.csv", local_image_structure_rows)
    pseudo_rows = read_csv_rows(root / "pseudo_spike_positive_control.csv")
    pseudo_bootstrap_rows = _pseudo_bootstrap_rows(
        pseudo_rows,
        seed=int(args.seed) + 113,
        n_bootstrap=max(100, min(int(args.n_bootstrap), 1000)),
    )
    write_csv(root / "pseudo_spike_bootstrap_summary.csv", pseudo_bootstrap_rows)
    (
        targeted_variant_rows,
        targeted_session_rows,
        targeted_leave_one_rows,
        targeted_pair_rows,
        targeted_outlier_rows,
    ) = _targeted_variant_audit_rows(
        pair_rows=pair_rows,
        session_rows=session_rows,
        bootstrap_rows=bootstrap_rows,
        primary_k=int(args.primary_k),
    )
    write_csv(root / "targeted_variant_summary.csv", targeted_variant_rows)
    write_csv(root / "targeted_variant_session_diagnostics.csv", targeted_session_rows)
    write_csv(root / "targeted_variant_leave_one_session_out.csv", targeted_leave_one_rows)
    write_csv(root / "targeted_variant_pair_diagnostics.csv", targeted_pair_rows)
    write_csv(root / "targeted_variant_outlier_pairs.csv", targeted_outlier_rows)
    _write_figures(root, bootstrap_rows, str(args.primary_projection_control), int(args.primary_k))

    audit = _load_json(root / "audit.json")
    leakage_rows = read_csv_rows(root / "fold_leakage_audit.csv")
    leakage_failures = int(sum(1 for row in leakage_rows if str(row.get("status")) == "fail"))
    decision, decision_checks = _primary_decision(
        bootstrap_rows,
        leakage_failures=leakage_failures,
        primary_projection_control=str(args.primary_projection_control),
        primary_k=int(args.primary_k),
        min_sessions=int(args.min_sessions),
    )
    audit.update(
        {
            "summary_status": "ok",
            "summary_datetime_utc": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "decision_checks": decision_checks,
            "n_pair_metric_rows": int(len(pair_rows)),
            "n_leakage_failures": leakage_failures,
            "q_norm_stratification_rows": int(len(q_norm_rows)),
            "image_structure_stratification_rows": int(len(image_structure_rows)),
            "local_image_structure_stratification_rows": int(len(local_image_structure_rows)),
            "pseudo_spike_bootstrap_rows": int(len(pseudo_bootstrap_rows)),
            "targeted_variant_summary_rows": int(len(targeted_variant_rows)),
            "targeted_variant_session_diagnostics_rows": int(len(targeted_session_rows)),
            "targeted_variant_leave_one_session_out_rows": int(len(targeted_leave_one_rows)),
            "targeted_variant_pair_diagnostics_rows": int(len(targeted_pair_rows)),
            "targeted_variant_outlier_pair_rows": int(len(targeted_outlier_rows)),
            "primary_projection_control": str(args.primary_projection_control),
            "primary_k": int(args.primary_k),
            "primary_true_minus_wrong": decision_checks.get("primary", {}),
        }
    )
    write_json(root / "audit.json", audit)


if __name__ == "__main__":
    main()
