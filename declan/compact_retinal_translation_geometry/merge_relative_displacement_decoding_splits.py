#!/usr/bin/env python3
"""Merge relative-displacement decoder split outputs into a canonical artifact."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from declan.compact_retinal_translation_geometry.run_relative_displacement_decoding import (
    DEFAULT_OUTPUT_ROOT,
    _feature_comparison,
    _write_summary_figure,
    write_csv,
    write_json,
)


DEFAULT_INPUT_GLOB = "relative_displacement_decoding_prod_gpu*"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _concat_csv(input_roots: list[Path], name: str) -> pd.DataFrame:
    frames = []
    for root in input_roots:
        frame = _read_csv(root / name)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["split_source"] = root.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    write_csv(path, frame.to_dict("records"))


def _metric_summary(metrics: pd.DataFrame, nulls: pd.DataFrame, *, seed: int, n_bootstrap: int) -> list[dict[str, Any]]:
    observed = metrics[(metrics["metric_name"].astype(str) == "R2_mean") & (metrics["null_type"].astype(str) == "observed")]
    if observed.empty:
        return []
    null_lookup: dict[tuple[str, str, str, int, str], list[float]] = {}
    if not nulls.empty:
        null_r2 = nulls[nulls["metric_name"].astype(str) == "R2_mean"]
        for row in null_r2.itertuples(index=False):
            key = (
                str(row.session),
                str(row.projection_control),
                str(row.feature_space),
                int(row.k),
                str(row.null_type),
            )
            null_lookup.setdefault(key, []).append(float(row.metric_null))
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    group_cols = ["feature_space", "k", "projection_control", "metric_name"]
    for (feature_space, k, projection_control, metric_name), group in observed.groupby(group_cols, sort=True):
        by_session = group.groupby("session", sort=True)["metric_value"].mean()
        obs_vals = by_session.to_numpy(dtype=np.float64)
        finite = obs_vals[np.isfinite(obs_vals)]
        if finite.size:
            observed_mean = float(np.mean(finite))
            boot = np.mean(finite[rng.integers(0, finite.size, size=(int(n_bootstrap), finite.size))], axis=1) if finite.size > 1 and n_bootstrap > 0 else np.asarray([], dtype=np.float64)
            ci_low = float(np.percentile(boot, 2.5)) if boot.size else float("nan")
            ci_high = float(np.percentile(boot, 97.5)) if boot.size else float("nan")
        else:
            observed_mean = ci_low = ci_high = float("nan")
        row: dict[str, Any] = {
            "feature_space": str(feature_space),
            "k": int(k),
            "projection_control": str(projection_control),
            "metric_name": str(metric_name),
            "n_sessions": int(finite.size),
            "observed_mean": observed_mean,
            "observed_boot_ci_low": ci_low,
            "observed_boot_ci_high": ci_high,
        }
        for null_type in ("eye_label_shuffle", "response_pair_shuffle"):
            effects = []
            medians = []
            for session, obs in by_session.items():
                vals = null_lookup.get((str(session), str(projection_control), str(feature_space), int(k), null_type), [])
                vals = [v for v in vals if np.isfinite(v)]
                if not vals:
                    continue
                med = float(np.nanmedian(vals))
                medians.append(med)
                effects.append(float(obs) - med)
            eff = np.asarray(effects, dtype=np.float64)
            eff = eff[np.isfinite(eff)]
            med_arr = np.asarray(medians, dtype=np.float64)
            if eff.size:
                eff_mean = float(np.mean(eff))
                boot = np.mean(eff[rng.integers(0, eff.size, size=(int(n_bootstrap), eff.size))], axis=1) if eff.size > 1 and n_bootstrap > 0 else np.asarray([], dtype=np.float64)
                eff_low = float(np.percentile(boot, 2.5)) if boot.size else float("nan")
                eff_high = float(np.percentile(boot, 97.5)) if boot.size else float("nan")
            else:
                eff_mean = eff_low = eff_high = float("nan")
            row[f"{null_type}_median_mean"] = float(np.nanmean(med_arr)) if med_arr.size else float("nan")
            row[f"effect_minus_{null_type}_mean"] = eff_mean
            row[f"effect_minus_{null_type}_boot_ci_low"] = eff_low
            row[f"effect_minus_{null_type}_boot_ci_high"] = eff_high
            row[f"n_{null_type}_effect_positive"] = int(np.sum(eff > 0.0))
        out.append(row)
    return out


def _infer_decision(comparison_rows: list[dict[str, Any]], leakage: pd.DataFrame) -> tuple[str, int, int, str]:
    comp = comparison_rows[0] if comparison_rows else {}
    compact = float(comp.get("compact_R2_mean", float("nan"))) if comp else float("nan")
    compact_minus_orth = float(comp.get("compact_minus_orthogonal", float("nan"))) if comp else float("nan")
    compact_minus_random = float(comp.get("compact_minus_random", float("nan"))) if comp else float("nan")
    compact_minus_rf = float(comp.get("compact_minus_rf_readout", float("nan"))) if comp else float("nan")
    leakage_failures = int((leakage.get("status", pd.Series(dtype=str)).astype(str) == "fail").sum()) if not leakage.empty else 0
    if not leakage.empty and "n_shared_trials" in leakage:
        trial_overlap_folds = int((pd.to_numeric(leakage["n_shared_trials"], errors="coerce").fillna(0) > 0).sum())
        trial_status = "audited"
    else:
        trial_overlap_folds = -1
        trial_status = "missing_for_input_outputs"
    positive = (
        np.isfinite(compact)
        and compact > 0.0
        and np.isfinite(compact_minus_orth)
        and compact_minus_orth > 0.0
        and np.isfinite(compact_minus_random)
        and compact_minus_random > 0.0
        and (not np.isfinite(compact_minus_rf) or compact_minus_rf > 0.0)
        and leakage_failures == 0
        and trial_overlap_folds == 0
    )
    return ("candidate_positive" if positive else "diagnostic"), leakage_failures, trial_overlap_folds, trial_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge relative displacement decoding split outputs")
    parser.add_argument("--compact-root", type=Path, default=Path("outputs") / "compact_retinal_translation_geometry")
    parser.add_argument("--input-roots", type=str, default="")
    parser.add_argument("--input-glob", type=str, default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--primary-projection-control", type=str, default="global_rate+target_pc1")
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-write-parent-tables", dest="write_parent_tables", action="store_false")
    parser.set_defaults(write_parent_tables=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    compact_root = Path(args.compact_root)
    if args.input_roots:
        input_roots = [Path(v) for v in args.input_roots.split(",") if v.strip()]
    else:
        input_roots = sorted(compact_root.glob(str(args.input_glob)))
    input_roots = [p for p in input_roots if (p / "audit.json").exists()]
    if not input_roots:
        raise SystemExit("No split roots with audit.json found")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    session_summary = _concat_csv(input_roots, "session_summary.csv")
    pair_inventory = _concat_csv(input_roots, "pair_inventory.csv")
    metrics = _concat_csv(input_roots, "decoder_metrics.csv")
    nulls = _concat_csv(input_roots, "decoder_nulls.csv")
    leakage = _concat_csv(input_roots, "split_leakage_audit.csv")
    rf_bins = _concat_csv(input_roots, "rf_readout_unit_bins.csv")
    summary_rows = _metric_summary(metrics, nulls, seed=int(args.seed), n_bootstrap=int(args.n_bootstrap))
    comparison_rows = _feature_comparison(summary_rows, str(args.primary_projection_control), int(args.primary_k))
    projection_rows = [
        r for r in summary_rows if str(r.get("feature_space")) == "compact" and int(r.get("k", -1)) == int(args.primary_k)
    ]
    k_rows = [
        r
        for r in summary_rows
        if str(r.get("feature_space")) == "compact"
        and str(r.get("projection_control")) == str(args.primary_projection_control)
    ]

    _write_frame(out / "session_summary.csv", session_summary)
    _write_frame(out / "pair_inventory.csv", pair_inventory)
    _write_frame(out / "decoder_metrics.csv", metrics)
    _write_frame(out / "decoder_nulls.csv", nulls)
    write_csv(out / "decoder_bootstrap_summary.csv", summary_rows)
    write_csv(out / "feature_space_comparison.csv", comparison_rows)
    write_csv(out / "projection_control_comparison.csv", projection_rows)
    write_csv(out / "k_sweep.csv", k_rows)
    _write_frame(out / "split_leakage_audit.csv", leakage)
    _write_frame(out / "rf_readout_unit_bins.csv", rf_bins)
    write_csv(out / "decoder_reliability_ceiling.csv", [{"status": "not_run", "reason": "split-half decoding ceiling is not yet implemented for pairwise relative decoder"}])
    write_csv(out / "spectral_bridge.csv", [{"status": "not_run", "reason": "run after primary decoder passes support/null gates"}])
    write_csv(out / "information_gain_bridge.csv", [{"status": "not_run", "reason": "run after primary decoder passes support/null gates"}])
    write_csv(out / "compact_specific_bridge.csv", [{"status": "not_run", "reason": "run after primary decoder passes support/null gates"}])
    _write_summary_figure(out, comparison_rows, summary_rows, str(args.primary_projection_control), int(args.primary_k))

    decision, leakage_failures, trial_overlap_folds, trial_status = _infer_decision(comparison_rows, leakage)
    split_audits = [_read_json(root / "audit.json") for root in input_roots]
    audit = {
        "status": "ok",
        "decision": decision,
        "merged_from": [str(root) for root in input_roots],
        "n_sessions_requested": int(sum(int(a.get("n_sessions_requested", 0)) for a in split_audits)),
        "n_sessions_ok": int(sum(int(a.get("n_sessions_ok", 0)) for a in split_audits)),
        "n_metric_rows": int(len(metrics)),
        "n_null_rows": int(len(nulls)),
        "n_leakage_failures": leakage_failures,
        "n_trial_overlap_folds": trial_overlap_folds,
        "trial_overlap_audit_status": trial_status,
        "primary_projection_control": str(args.primary_projection_control),
        "primary_k": int(args.primary_k),
        "primary_feature_comparison": comparison_rows[0] if comparison_rows else {},
        "claim_guardrail": "Do not describe this as absolute or image-independent eye-position decoding.",
    }
    write_json(out / "audit.json", audit)
    write_json(
        out / "relative_displacement_decoding_manifest.json",
        {
            "analysis": "same_image_relative_displacement_decoding_merged",
            "status": "ok",
            "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
            "merged_from": [str(root) for root in input_roots],
            "claim_guardrail": audit["claim_guardrail"],
        },
    )
    (out / "README.md").write_text(
        "# Same-Condition Relative Displacement Decoding\n\n"
        "Canonical merged artifact from split runs. Primary tables are "
        "`decoder_bootstrap_summary.csv`, `feature_space_comparison.csv`, and `decoder_nulls.csv`. "
        "Older split outputs may lack trial-overlap fields; in that case run a trial-disjoint sensitivity pass "
        "before promoting Panel F claims.\n",
        encoding="utf-8",
    )
    if bool(args.write_parent_tables):
        tables = out.parent / "tables"
        _write_frame(tables / "displacement_decoding_metrics.csv", metrics)
        write_csv(tables / "displacement_decoding_bootstrap_summary.csv", summary_rows)
        _write_frame(tables / "displacement_decoding_nulls.csv", nulls)
        _write_frame(tables / "displacement_decoding_pair_inventory.csv", pair_inventory)
        write_csv(tables / "displacement_decoding_reliability_ceiling.csv", [{"status": "not_run", "reason": "split-half decoding ceiling is not yet implemented for pairwise relative decoder"}])
        write_csv(tables / "panelF_displacement_decoding_metrics.csv", comparison_rows)


if __name__ == "__main__":
    main()
