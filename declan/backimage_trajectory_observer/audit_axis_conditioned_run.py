"""Audit axis-conditioned BackImage trajectory observer runs before promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


AXIS_FAMILIES = ("axis_edge_parallel", "axis_edge_orthogonal")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce")


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    vals = df[col]
    if vals.dtype == bool:
        return vals.astype(float)
    if pd.api.types.is_numeric_dtype(vals):
        return vals.astype(float)
    normalized = vals.astype(str).str.strip().str.lower()
    out = normalized.map({"true": 1.0, "1": 1.0, "yes": 1.0, "false": 0.0, "0": 0.0, "no": 0.0})
    return out.astype(float)


def _normalize_axis_table(axis: pd.DataFrame) -> pd.DataFrame:
    if axis.empty:
        return axis
    out = axis.copy()
    if "axis_catalog_mode" not in out.columns:
        out["axis_catalog_mode"] = "shared"
    out["axis_catalog_mode"] = out["axis_catalog_mode"].fillna("shared").astype(str)
    if "role" not in out.columns:
        out["role"] = np.where(out.get("family", pd.Series("", index=out.index)).isin(AXIS_FAMILIES), "prior", "")
    if "axis_pair_id" not in out.columns:
        out["axis_pair_id"] = ""
    return out


def _normalize_manifest_table(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    out = manifest.copy()
    if "axis_catalog_mode" not in out.columns:
        out["axis_catalog_mode"] = "shared"
    out["axis_catalog_mode"] = out["axis_catalog_mode"].fillna("shared").astype(str)
    return out


def _normalize_trials_table(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return trials
    out = trials.copy()
    if "axis_catalog_mode" not in out.columns:
        out["axis_catalog_mode"] = "shared"
    out["axis_catalog_mode"] = out["axis_catalog_mode"].fillna("shared").astype(str)
    return out


def _family_summary(axis: pd.DataFrame) -> list[dict[str, Any]]:
    prior = axis[axis["role"].astype(str).eq("prior")].copy()
    rows: list[dict[str, Any]] = []
    metrics = [
        "requested_rms_deg",
        "effective_rms_deg",
        "path_length_deg",
        "speed_mean_deg_s",
        "speed_p95_deg_s",
        "rendered_rms_displacement_deg",
        "rendered_path_length_deg",
        "rendered_max_radius_deg",
        "rendered_duration_s",
        "clipping_fraction",
        "axis_match_rms_delta_deg",
        "axis_match_path_delta_deg",
        "axis_match_duration_delta_s",
        "axis_match_clipping_fraction_delta",
    ]
    for key, grp in prior.groupby(["family", "scale", "axis_catalog_mode"], dropna=False):
        family, scale, axis_catalog_mode = key
        row: dict[str, Any] = {
            "family": family,
            "scale": float(scale),
            "axis_catalog_mode": axis_catalog_mode,
            "n_prior_rows": int(len(grp)),
            "n_trials": int(grp["trial_id"].nunique()) if "trial_id" in grp else 0,
            "n_candidates": int(grp["candidate_index"].nunique()) if "candidate_index" in grp else 0,
            "n_source_rows": int(grp["source_row"].nunique()) if "source_row" in grp else 0,
            "rms_clipped_high_fraction": float(_bool_series(grp, "rms_clipped_high").mean()),
            "axis_match_degenerate_fraction": float(_bool_series(grp, "axis_match_degenerate").mean()),
        }
        for metric in metrics:
            vals = _num(grp, metric)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_median"] = float(vals.median())
            row[f"{metric}_p95"] = float(vals.quantile(0.95))
        rows.append(row)
    return rows


def _paired_family_deltas(axis: pd.DataFrame) -> pd.DataFrame:
    prior = axis[axis["role"].astype(str).eq("prior") & axis["family"].isin(AXIS_FAMILIES)].copy()
    if prior.empty:
        return pd.DataFrame()
    key_cols = [
        "trial_id",
        "candidate_index",
        "axis_candidate_source_row",
        "source_row",
        "scale",
    ]
    if "axis_pair_id" in prior.columns and prior["axis_pair_id"].notna().any():
        key_cols.append("axis_pair_id")
    missing = [col for col in key_cols if col not in prior.columns]
    if missing:
        return pd.DataFrame()
    metrics = [
        "requested_rms_deg",
        "effective_rms_deg",
        "path_length_deg",
        "speed_mean_deg_s",
        "speed_p95_deg_s",
        "rendered_rms_displacement_deg",
        "rendered_path_length_deg",
        "rendered_max_radius_deg",
        "rendered_duration_s",
        "clipping_fraction",
    ]
    keep_cols = key_cols + ["family"] + metrics
    wide = prior[keep_cols].copy()
    for metric in metrics:
        wide[metric] = pd.to_numeric(wide[metric], errors="coerce")
    piv = wide.pivot_table(index=key_cols, columns="family", values=metrics, aggfunc="mean")
    if not set(AXIS_FAMILIES).issubset(set(piv.columns.get_level_values(1))):
        return pd.DataFrame()
    rows = []
    for idx, row in piv.iterrows():
        first_metric = metrics[0]
        if not (
            np.isfinite(row[(first_metric, "axis_edge_parallel")])
            and np.isfinite(row[(first_metric, "axis_edge_orthogonal")])
        ):
            continue
        payload = {col: value for col, value in zip(key_cols, idx, strict=True)}
        for metric in metrics:
            par = row[(metric, "axis_edge_parallel")]
            orth = row[(metric, "axis_edge_orthogonal")]
            payload[f"{metric}_parallel"] = float(par)
            payload[f"{metric}_orthogonal"] = float(orth)
            payload[f"{metric}_parallel_minus_orthogonal"] = float(par - orth)
            payload[f"{metric}_abs_delta"] = float(abs(par - orth))
        rows.append(payload)
    return pd.DataFrame(rows)


def _source_overlap_summary(axis: pd.DataFrame) -> list[dict[str, Any]]:
    prior = axis[axis["role"].astype(str).eq("prior") & axis["family"].isin(AXIS_FAMILIES)].copy()
    if prior.empty or not {"trial_id", "candidate_index", "family", "source_row", "scale"}.issubset(prior.columns):
        return []
    rows = []
    for key, grp in prior.groupby(["trial_id", "candidate_index", "scale"], dropna=False):
        trial_id, candidate_index, scale = key
        sources = {
            family: set(pd.to_numeric(fgrp["source_row"], errors="coerce").dropna().astype(int).tolist())
            for family, fgrp in grp.groupby("family")
        }
        if not set(AXIS_FAMILIES).issubset(sources):
            continue
        par = sources["axis_edge_parallel"]
        orth = sources["axis_edge_orthogonal"]
        union = par | orth
        inter = par & orth
        rows.append(
            {
                "trial_id": int(trial_id),
                "candidate_index": int(candidate_index),
                "scale": float(scale),
                "n_parallel_sources": int(len(par)),
                "n_orthogonal_sources": int(len(orth)),
                "n_shared_sources": int(len(inter)),
                "source_jaccard": float(len(inter) / len(union)) if union else float("nan"),
            }
        )
    return rows


def _source_overlap_lookup(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["trial_id"]), int(row["candidate_index"]), float(row["scale"]))
        out[key] = row
    return out


def _source_overlap_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for scale, grp in df.groupby("scale", dropna=False):
        out.append(
            {
                "scale": float(scale),
                "n_trial_candidate_groups": int(len(grp)),
                "median_parallel_sources": float(grp["n_parallel_sources"].median()),
                "median_orthogonal_sources": float(grp["n_orthogonal_sources"].median()),
                "median_shared_sources": float(grp["n_shared_sources"].median()),
                "median_source_jaccard": float(grp["source_jaccard"].median()),
                "mean_source_jaccard": float(grp["source_jaccard"].mean()),
            }
        )
    return out


def _paired_delta_summary(paired: pd.DataFrame) -> list[dict[str, Any]]:
    if paired.empty:
        return []
    metrics = sorted(col.removesuffix("_parallel_minus_orthogonal") for col in paired.columns if col.endswith("_parallel_minus_orthogonal"))
    rows = []
    for scale, grp in paired.groupby("scale", dropna=False):
        row: dict[str, Any] = {
            "scale": float(scale),
            "n_paired_rows": int(len(grp)),
            "n_trials": int(grp["trial_id"].nunique()),
            "n_candidates": int(grp["candidate_index"].nunique()),
        }
        for metric in metrics:
            delta = pd.to_numeric(grp[f"{metric}_parallel_minus_orthogonal"], errors="coerce")
            abs_delta = pd.to_numeric(grp[f"{metric}_abs_delta"], errors="coerce")
            row[f"{metric}_delta_mean"] = float(delta.mean())
            row[f"{metric}_delta_median"] = float(delta.median())
            row[f"{metric}_abs_delta_median"] = float(abs_delta.median())
            row[f"{metric}_abs_delta_p95"] = float(abs_delta.quantile(0.95))
        rows.append(row)
    return rows


def _manifest_summary(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    if manifest.empty:
        return []
    rows = []
    for key, grp in manifest.groupby(["candidate_set_mode", "prior_family", "scale", "axis_catalog_mode"], dropna=False):
        candidate_set_mode, prior_family, scale, axis_catalog_mode = key
        row = {
            "candidate_set_mode": candidate_set_mode,
            "prior_family": prior_family,
            "scale": float(scale),
            "axis_catalog_mode": axis_catalog_mode,
            "n_rows": int(len(grp)),
            "n_trials": int(grp["trial_id"].nunique()) if "trial_id" in grp else 0,
            "n_candidates_median": float(pd.to_numeric(grp["n_candidates"], errors="coerce").median()) if "n_candidates" in grp else float("nan"),
            "n_prior_trajectories_median": float(pd.to_numeric(grp["n_prior_trajectories"], errors="coerce").median()) if "n_prior_trajectories" in grp else float("nan"),
            "n_timebins_median": float(pd.to_numeric(grp["n_timebins"], errors="coerce").median()) if "n_timebins" in grp else float("nan"),
            "n_units_median": float(pd.to_numeric(grp["n_units"], errors="coerce").median()) if "n_units" in grp else float("nan"),
            "prior_duplicate_trajectory_count_sum": int(pd.to_numeric(grp.get("prior_duplicate_trajectory_count", 0), errors="coerce").fillna(0).sum()),
            "excluded_exact_trace_hash_sum": int(pd.to_numeric(grp.get("excluded_exact_trace_hash", 0), errors="coerce").fillna(0).sum()),
            "excluded_near_duplicate_rmse_sum": int(pd.to_numeric(grp.get("excluded_near_duplicate_rmse", 0), errors="coerce").fillna(0).sum()),
        }
        if "axis_shared_source_catalog" in grp:
            row["axis_shared_source_catalog_fraction"] = float(_bool_series(grp, "axis_shared_source_catalog").mean())
        if "nearest_trajectory_distance" in grp:
            row["nearest_trajectory_distance_median"] = float(pd.to_numeric(grp["nearest_trajectory_distance"], errors="coerce").median())
        rows.append(row)
    return rows


def _observer_summary(trials: pd.DataFrame) -> list[dict[str, Any]]:
    if trials.empty:
        return []
    rows = []
    for key, grp in trials.groupby(["candidate_set_mode", "prior_family", "prior_scale", "axis_catalog_mode", "likelihood_scale"], dropna=False):
        candidate_set_mode, prior_family, prior_scale, axis_catalog_mode, likelihood_scale = key
        known = _bool_series(grp, "known_correct")
        zero = _bool_series(grp, "zero_correct")
        joint = _bool_series(grp, "joint_correct")
        rows.append(
            {
                "candidate_set_mode": candidate_set_mode,
                "prior_family": prior_family,
                "prior_scale": float(prior_scale),
                "axis_catalog_mode": axis_catalog_mode,
                "likelihood_scale": float(likelihood_scale),
                "n_trials": int(len(grp)),
                "known_eye_accuracy": float(known.mean()),
                "zero_eye_accuracy": float(zero.mean()),
                "joint_eye_accuracy": float(joint.mean()),
                "joint_minus_zero_accuracy": (
                    float(joint.mean() - zero.mean())
                    if {"joint_correct", "zero_correct"}.issubset(grp.columns)
                    else float("nan")
                ),
                "median_N_eff_fraction": float(pd.to_numeric(grp.get("N_eff_true_image_fraction", np.nan), errors="coerce").median()),
                "median_nearest_tau_rank": float(pd.to_numeric(grp.get("nearest_tau_rank", np.nan), errors="coerce").median()),
                "median_nearest_tau_distance": float(pd.to_numeric(grp.get("nearest_tau_distance", np.nan), errors="coerce").median()),
                "median_joint_true_margin": float(pd.to_numeric(grp.get("joint_true_margin", np.nan), errors="coerce").median()),
                "median_joint_minus_zero_true_score": float(pd.to_numeric(grp.get("joint_minus_zero_true_score", np.nan), errors="coerce").median()),
            }
        )
    return rows


def _paired_observer_delta(trials: pd.DataFrame, source_overlap_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if trials.empty or not {"trial_id", "prior_family"}.issubset(trials.columns):
        return []
    metric_cols = [
        "joint_correct",
        "joint_true_margin",
        "joint_true_score",
        "joint_minus_zero_true_score",
        "N_eff_true_image_fraction",
        "nearest_tau_rank",
        "nearest_tau_distance",
    ]
    base_cols = ["trial_id", "candidate_set_mode", "prior_scale", "likelihood_scale"]
    rows = []
    subset = trials[trials["prior_family"].isin(AXIS_FAMILIES)].copy()
    source_lookup = _source_overlap_lookup(source_overlap_rows or [])
    for col in metric_cols:
        if col == "joint_correct" and col in subset.columns:
            subset[col] = _bool_series(subset, col)
        elif col in subset.columns and subset[col].dtype == bool:
            subset[col] = subset[col].astype(float)
        elif col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
    for key, grp in subset.groupby(base_cols, dropna=False):
        if set(grp["prior_family"]) != set(AXIS_FAMILIES):
            continue
        payload = {col: value for col, value in zip(base_cols, key, strict=True)}
        trial_id = int(payload["trial_id"])
        scale = float(payload["prior_scale"])
        overlaps = [row for row_key, row in source_lookup.items() if row_key[0] == trial_id and abs(row_key[2] - scale) < 1e-12]
        if overlaps:
            payload["source_overlap_n_candidate_groups"] = int(len(overlaps))
            payload["source_overlap_median_jaccard"] = float(np.median([row["source_jaccard"] for row in overlaps]))
            payload["source_overlap_median_shared_sources"] = float(np.median([row["n_shared_sources"] for row in overlaps]))
        else:
            payload["source_overlap_n_candidate_groups"] = 0
            payload["source_overlap_median_jaccard"] = float("nan")
            payload["source_overlap_median_shared_sources"] = float("nan")
        for metric in metric_cols:
            if metric not in grp.columns:
                continue
            vals = grp.set_index("prior_family")[metric]
            if not set(AXIS_FAMILIES).issubset(vals.index):
                continue
            par = float(vals.loc["axis_edge_parallel"])
            orth = float(vals.loc["axis_edge_orthogonal"])
            payload[f"{metric}_parallel"] = par
            payload[f"{metric}_orthogonal"] = orth
            payload[f"{metric}_parallel_minus_orthogonal"] = par - orth
        rows.append(payload)
    return rows


def _write_report(
    out_dir: Path,
    run_dir: Path,
    family_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    source_overlap_rows: list[dict[str, Any]],
    observer_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Axis-Conditioned Trajectory Audit",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Family Balance",
        "",
    ]
    if family_rows:
        fam = pd.DataFrame(family_rows)
        cols = [
            "family",
            "scale",
            "n_prior_rows",
            "effective_rms_deg_median",
            "path_length_deg_median",
            "speed_mean_deg_s_median",
            "speed_p95_deg_s_median",
            "clipping_fraction_median",
            "rms_clipped_high_fraction",
        ]
        lines.append("```text")
        lines.append(fam[[c for c in cols if c in fam.columns]].to_csv(index=False).strip())
        lines.append("```")
    lines.extend(["", "## Paired Parallel Minus Orthogonal Deltas", ""])
    if paired_rows:
        pair = pd.DataFrame(paired_rows)
        cols = [
            "scale",
            "n_paired_rows",
            "effective_rms_deg_abs_delta_p95",
            "path_length_deg_abs_delta_p95",
            "speed_mean_deg_s_abs_delta_p95",
            "speed_p95_deg_s_abs_delta_p95",
            "clipping_fraction_abs_delta_p95",
        ]
        lines.append("```text")
        lines.append(pair[[c for c in cols if c in pair.columns]].to_csv(index=False).strip())
        lines.append("```")
    lines.extend(["", "## Source Overlap Across Axis Families", ""])
    if source_overlap_rows:
        overlap = pd.DataFrame(source_overlap_rows)
        lines.append("```text")
        lines.append(overlap.to_csv(index=False).strip())
        lines.append("```")
    lines.extend(["", "## Observer Readout", ""])
    if observer_rows:
        obs = pd.DataFrame(observer_rows)
        cols = [
            "candidate_set_mode",
            "prior_family",
            "prior_scale",
            "n_trials",
            "known_eye_accuracy",
            "zero_eye_accuracy",
            "joint_eye_accuracy",
            "joint_minus_zero_accuracy",
            "median_N_eff_fraction",
            "median_nearest_tau_distance",
        ]
        lines.append("```text")
        lines.append(obs[[c for c in cols if c in obs.columns]].to_csv(index=False).strip())
        lines.append("```")
    (out_dir / "axis_conditioned_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "axis_conditioned_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    axis_path = run_dir / "axis_trajectory_catalog.csv"
    motion_path = run_dir / "motion_catalog.csv"
    manifest_path = run_dir / "response_cache_manifest.csv"
    trials_path = run_dir / "observer_trials.csv"
    axis = _read_csv_safe(axis_path)
    if axis.empty and motion_path.exists():
        motion = _read_csv_safe(motion_path)
        axis = motion[motion.get("family", pd.Series(dtype=str)).isin(AXIS_FAMILIES)].copy()
    axis = _normalize_axis_table(axis)
    manifest = _normalize_manifest_table(_read_csv_safe(manifest_path))
    trials = _normalize_trials_table(_read_csv_safe(trials_path))

    family_rows = _family_summary(axis) if not axis.empty else []
    paired = _paired_family_deltas(axis) if not axis.empty else pd.DataFrame()
    paired_rows = _paired_delta_summary(paired)
    source_overlap = _source_overlap_summary(axis) if not axis.empty else []
    source_overlap_rows = _source_overlap_aggregate(source_overlap)
    manifest_rows = _manifest_summary(manifest)
    observer_rows = _observer_summary(trials)
    observer_delta_rows = _paired_observer_delta(trials, source_overlap)

    _write_csv(out_dir / "axis_family_balance_summary.csv", family_rows)
    paired.to_csv(out_dir / "axis_family_paired_deltas.csv", index=False)
    _write_csv(out_dir / "axis_family_paired_delta_summary.csv", paired_rows)
    _write_csv(out_dir / "axis_family_source_overlap.csv", source_overlap)
    _write_csv(out_dir / "axis_family_source_overlap_summary.csv", source_overlap_rows)
    _write_csv(out_dir / "axis_manifest_summary.csv", manifest_rows)
    _write_csv(out_dir / "axis_observer_summary.csv", observer_rows)
    _write_csv(out_dir / "axis_observer_parallel_minus_orthogonal.csv", observer_delta_rows)
    _write_json(
        out_dir / "axis_conditioned_audit_metadata.json",
        {
            "run_dir": str(run_dir),
            "axis_catalog_path": str(axis_path) if axis_path.exists() else None,
            "motion_catalog_path": str(motion_path) if motion_path.exists() else None,
            "manifest_path": str(manifest_path) if manifest_path.exists() else None,
            "observer_trials_path": str(trials_path) if trials_path.exists() else None,
            "n_axis_rows": int(len(axis)),
            "n_manifest_rows": int(len(manifest)),
            "n_observer_rows": int(len(trials)),
            "axis_families": list(AXIS_FAMILIES),
        },
    )
    _write_report(out_dir, run_dir, family_rows, paired_rows, source_overlap_rows, observer_rows)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    out = audit(build_parser().parse_args())
    print(f"Wrote axis-conditioned audit to {out}")


if __name__ == "__main__":
    main()
