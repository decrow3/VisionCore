"""Cache-first BackImage feature/readout decomposition adjudication.

This posthoc script inventories existing aggregate FEM, local I_z pairing, and
joint feature-posterior outputs, then ranks feature specifications by stability
across branches.  It intentionally does not launch new V1 forward passes; any
missing k/summary/scale cells are written to a completion manifest so the next
cache-only posthoc can be targeted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review")
DEFAULT_AGGREGATE_RUN_DIR = (
    DEFAULT_OUTPUT_ROOT
    / "backimage_aggregate_fem_information_n256_k48_rel025-2_drift_only_common_unclipped_patched"
)
DEFAULT_LOCAL_PAIRING_DIRS = (
    DEFAULT_OUTPUT_ROOT
    / "backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_1_seed7_v1"
)
DEFAULT_JOINT_FEATURE_DIRS = (
    DEFAULT_OUTPUT_ROOT
    / "backimage_axis_conditioned_hard_negative_n128_scale_sweep_feature_posterior_gabor_pyramid_k4_8_16_uncertainty_v1"
)
DEFAULT_OUT_DIR = DEFAULT_OUTPUT_ROOT / "backimage_feature_decomposition_adjudication_v1"

PRIMARY_AGGREGATE_CONTRASTS = {
    "empirical-ou": "aggregate_empirical_minus_ou",
    "empirical-brownian": "aggregate_empirical_minus_brownian",
    "empirical-rotated": "aggregate_empirical_minus_rotated",
    "empirical-static": "aggregate_empirical_minus_static",
}
PRIMARY_LOCAL_CONTRASTS = {
    "actual_paired_empirical-matched_unpaired_empirical": "local_actual_minus_matched_unpaired",
    "actual_paired_empirical-rotated_actual_90": "local_actual_minus_rotated",
    "actual_paired_empirical-ou_matched_actual": "local_actual_minus_ou",
    "actual_paired_empirical-brownian_matched_actual": "local_actual_minus_brownian",
    "edge_axis-edge_orthogonal": "local_edge_axis_minus_edge_orthogonal",
}
PRIMARY_JOINT_METRICS = {
    "joint_minus_zero_feature_gain_parallel_minus_orthogonal": "joint_parallel_minus_orthogonal_feature_gain",
    "joint_parallel_minus_orthogonal": "joint_parallel_minus_orthogonal",
    "joint_minus_zero_feature_gain": "joint_minus_zero_feature_gain",
}
JOINT_AXIS_METRICS = {
    "joint_parallel_minus_orthogonal_feature_gain",
    "joint_parallel_minus_orthogonal",
}
JOINT_GENERIC_METRICS = {
    "joint_minus_zero_feature_gain",
}


@dataclass(frozen=True)
class SourceFile:
    branch: str
    source_dir: Path
    source_file: Path
    table_role: str


def _parse_str_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_list(text: str | None) -> list[int]:
    return [int(part) for part in _parse_str_list(text)]


def _parse_path_list(text: str | Path | None) -> list[Path]:
    if text is None:
        return []
    if isinstance(text, Path):
        return [text]
    return [Path(part) for part in _parse_str_list(text)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_source_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    aggregate = _load_json(args.aggregate_run_dir / "run_metadata.json")
    if aggregate:
        configs["aggregate"] = dict(aggregate.get("config", aggregate))
    local_dirs = _parse_path_list(args.local_pairing_dirs)
    if local_dirs:
        local = _load_json(local_dirs[0] / "run_metadata.json")
        if local:
            configs["local_Iz"] = dict(local.get("config", local))
    joint_dirs = _parse_path_list(args.joint_feature_dirs)
    if joint_dirs:
        joint = _load_json(joint_dirs[0] / "feature_posterior_metadata.json")
        if joint:
            configs["joint_posterior"] = dict(joint.get("config", joint))
    return configs


def _config_value(configs: dict[str, dict[str, Any]], key: str, default: Any = None) -> Any:
    for branch in ["aggregate", "local_Iz", "joint_posterior"]:
        if key in configs.get(branch, {}):
            return configs[branch][key]
    return default


def _markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    if max_rows is not None:
        df = df.head(int(max_rows))
    cols = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                if np.isfinite(value):
                    vals.append(f"{value:.4g}")
                else:
                    vals.append("")
            else:
                vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _value(row: pd.Series, names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return default


def _scale_sort_key(scale: str) -> tuple[int, float, str]:
    text = str(scale)
    val = text.replace("rel_", "").replace("x", "").replace("p", ".")
    try:
        return (0, float(val), text)
    except ValueError:
        try:
            return (0, float(text), text)
        except ValueError:
            return (1, float("inf"), text)


def _normalize_scale(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("rel_"):
        return text
    try:
        val = float(text)
    except ValueError:
        return text
    token = f"{val:g}".replace(".", "p")
    return f"rel_{token}x"


def _estimate_se(ci_low: float, ci_high: float) -> float:
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return float("nan")
    width = float(ci_high) - float(ci_low)
    if width <= 0:
        return float("nan")
    return width / (2.0 * 1.96)


def _evidence_score(estimate: float, ci_low: float, ci_high: float, *, cap: float = 3.0) -> tuple[float, bool]:
    estimate = _safe_float(estimate)
    ci_low = _safe_float(ci_low)
    ci_high = _safe_float(ci_high)
    if not np.isfinite(estimate):
        return (0.0, False)
    ci_pass = bool(np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low > 0.0)
    se = _estimate_se(ci_low, ci_high)
    if np.isfinite(se) and se > 1e-12:
        z = estimate / se
        return (float(np.clip(z, -cap, cap)), ci_pass)
    return (float(np.sign(estimate)), ci_pass)


def _infer_estimate_columns(table_role: str, df: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None]:
    candidates = [
        "incremental_gain_delta_neg_mse",
        "incremental_gain_neg_mse",
        "estimate",
        "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal",
        "mean_joint_parallel_minus_orthogonal",
        "mean_joint_minus_zero_feature_gain",
        "neg_mse",
        "mean_neg_mse",
        "score",
        "score_mean",
        "signal_motion_ratio",
        "trace_signal",
    ]
    estimate_col = next((col for col in candidates if col in df.columns), None)
    low_col = next((col for col in ["ci95_low", "ci_low", f"{estimate_col}_ci_low"] if col in df.columns), None)
    high_col = next((col for col in ["ci95_high", "ci_high", f"{estimate_col}_ci_high"] if col in df.columns), None)
    p_col = next((col for col in ["p_value", "permutation_p_two_sided", f"{estimate_col}_permutation_p_two_sided"] if col in df.columns), None)
    return estimate_col, low_col, high_col, p_col


def _base_inventory_row(source: SourceFile, row: pd.Series) -> dict[str, Any]:
    latent = str(_value(row, ["latent", "latent_name"], ""))
    k = _safe_int(_value(row, ["k", "requested_k", "k_eff", "pca_k"], None), None)
    response_summary = str(_value(row, ["motion_summary", "response_summary", "metric", "posterior_mode"], ""))
    scale = _normalize_scale(_value(row, ["scale_id", "observation_scale", "prior_scale", "scale"], ""))
    return {
        "branch": source.branch,
        "source_dir": str(source.source_dir),
        "source_file": str(source.source_file),
        "table_role": source.table_role,
        "latent": latent,
        "k": k,
        "response_summary": response_summary,
        "scale": scale,
        "feature_space": "selected_windows_zscore_pca",
    }


def _claim_role(branch: str, control_contrast: str, response_summary: str, scale: str) -> str:
    roles: list[str] = []
    if branch == "aggregate" and control_contrast in PRIMARY_AGGREGATE_CONTRASTS:
        roles.append(PRIMARY_AGGREGATE_CONTRASTS[control_contrast])
    if branch == "local_Iz" and control_contrast in PRIMARY_LOCAL_CONTRASTS:
        roles.append(PRIMARY_LOCAL_CONTRASTS[control_contrast])
    if branch == "joint_posterior":
        for key, name in PRIMARY_JOINT_METRICS.items():
            if key in control_contrast or key == response_summary:
                roles.append(name)
    if response_summary in {"delta_mean", "temporal_pca"}:
        roles.append("primary_response_summary")
    if scale in {"rel_0p25x", "rel_0p5x", "rel_1x"}:
        roles.append("primary_scale")
    if scale == "rel_2x":
        roles.append("sentinel_scale")
    return ";".join(dict.fromkeys(roles))


def _normalize_decode_like(source: SourceFile) -> list[dict[str, Any]]:
    df = _read_csv(source.source_file)
    if df.empty:
        return []
    estimate_col, low_col, high_col, p_col = _infer_estimate_columns(source.table_role, df)
    rows: list[dict[str, Any]] = []
    for _, rec in df.iterrows():
        out = _base_inventory_row(source, rec)
        family = str(_value(rec, ["family", "prior_family", "condition"], ""))
        lhs = str(_value(rec, ["lhs_family"], ""))
        rhs = str(_value(rec, ["rhs_family"], ""))
        if lhs and rhs:
            contrast = f"{lhs}-{rhs}"
        elif family:
            contrast = f"{family}-static" if "gain_vs_static" in source.table_role else family
        else:
            contrast = str(_value(rec, ["contrast", "metric"], source.table_role))
        estimate = _safe_float(_value(rec, [estimate_col] if estimate_col else [], float("nan")))
        ci_low = _safe_float(_value(rec, [low_col] if low_col else [], float("nan")))
        ci_high = _safe_float(_value(rec, [high_col] if high_col else [], float("nan")))
        p_value = _safe_float(_value(rec, [p_col] if p_col else [], float("nan")))
        out.update(
            {
                "control_contrast": contrast,
                "metric": estimate_col or source.table_role,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "n_images": _safe_int(_value(rec, ["n_images"], None), None),
                "n_trials": _safe_int(_value(rec, ["n_trials"], None), None),
                "n_sessions": _safe_int(_value(rec, ["n_sessions"], None), None),
                "claim_role": _claim_role(source.branch, contrast, str(out["response_summary"]), str(out["scale"])),
                "known_caveat": _known_caveat(source.branch, contrast, str(out["scale"])),
            }
        )
        rows.append(out)
    return rows


def _normalize_joint_uncertainty(source: SourceFile) -> list[dict[str, Any]]:
    df = _read_csv(source.source_file)
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, rec in df.iterrows():
        out = _base_inventory_row(source, rec)
        metric = str(_value(rec, ["metric"], ""))
        scope = str(_value(rec, ["contrast_scope"], ""))
        if scope in {"axis", "axis_parallel_minus_orthogonal"}:
            lhs = str(_value(rec, ["axis_lhs", "lhs_prior_family"], ""))
            rhs = str(_value(rec, ["axis_rhs", "rhs_prior_family"], ""))
            contrast = f"{metric}_parallel_minus_orthogonal" if lhs or rhs else metric
        elif scope == "pairwise_prior_lhs_minus_rhs":
            lhs = str(_value(rec, ["lhs_prior_family"], ""))
            rhs = str(_value(rec, ["rhs_prior_family"], ""))
            contrast = f"{lhs}-{rhs}:{metric}" if lhs and rhs else f"pairwise_prior:{metric}"
        elif scope == "within_prior":
            prior = str(_value(rec, ["prior_family", "prior_condition"], ""))
            contrast = f"{prior}-zero:{metric}" if prior else metric
        else:
            contrast = metric
        out.update(
            {
                "response_summary": metric,
                "control_contrast": contrast,
                "metric": metric,
                "estimate": _safe_float(_value(rec, ["estimate"], float("nan"))),
                "ci_low": _safe_float(_value(rec, ["ci_low"], float("nan"))),
                "ci_high": _safe_float(_value(rec, ["ci_high"], float("nan"))),
                "p_value": _safe_float(_value(rec, ["permutation_p_two_sided"], float("nan"))),
                "n_images": None,
                "n_trials": _safe_int(_value(rec, ["n_trials", "paired_n"], None), None),
                "n_sessions": None,
                "claim_role": _claim_role(source.branch, contrast, metric, str(out["scale"])),
                "known_caveat": _known_caveat(source.branch, contrast, str(out["scale"])),
            }
        )
        rows.append(out)
    return rows


def _normalize_joint_axis_wide(source: SourceFile) -> list[dict[str, Any]]:
    df = _read_csv(source.source_file)
    if df.empty:
        return []
    metrics = [
        "mean_joint_parallel_minus_orthogonal",
        "mean_joint_minus_zero_feature_gain_parallel_minus_orthogonal",
        "mean_known_minus_joint_pose_cost_parallel_minus_orthogonal",
        "mean_motion_delta_parallel_minus_orthogonal",
    ]
    rows: list[dict[str, Any]] = []
    for _, rec in df.iterrows():
        for metric_col in metrics:
            if metric_col not in df.columns:
                continue
            metric = metric_col.removeprefix("mean_")
            out = _base_inventory_row(source, rec)
            out.update(
                {
                    "response_summary": metric,
                    "control_contrast": metric,
                    "metric": metric,
                    "estimate": _safe_float(rec[metric_col]),
                    "ci_low": _safe_float(_value(rec, [f"{metric_col}_ci_low"], float("nan"))),
                    "ci_high": _safe_float(_value(rec, [f"{metric_col}_ci_high"], float("nan"))),
                    "p_value": _safe_float(_value(rec, [f"{metric_col}_permutation_p_two_sided"], float("nan"))),
                    "n_images": None,
                    "n_trials": _safe_int(_value(rec, ["n_trials", f"{metric_col}_n"], None), None),
                    "n_sessions": None,
                    "claim_role": _claim_role(source.branch, metric, metric, str(out["scale"])),
                    "known_caveat": _known_caveat(source.branch, metric, str(out["scale"])),
                }
            )
            rows.append(out)
    return rows


def _known_caveat(branch: str, contrast: str, scale: str) -> str:
    caveats: list[str] = []
    if branch == "joint_posterior":
        caveats.append("hard-negative image-identity pressure can favor orthogonal discrimination")
    if scale == "rel_2x":
        caveats.append("2x is sentinel/over-large and should not be sole positive")
    if "rotated" in contrast:
        caveats.append("rotated controls can remain competitive in local branch")
    return "; ".join(caveats)


def _source_files(
    *,
    aggregate_run_dir: Path,
    aggregate_incremental_dir: Path,
    local_pairing_dirs: list[Path],
    joint_feature_dirs: list[Path],
) -> list[SourceFile]:
    sources: list[SourceFile] = []
    aggregate_files = [
        (aggregate_incremental_dir / "incremental_gain_vs_static.csv", "incremental_gain_vs_static"),
        (aggregate_incremental_dir / "incremental_gain_contrasts.csv", "incremental_gain_contrasts"),
        (aggregate_run_dir / "decode_summary.csv", "decode_summary"),
        (aggregate_run_dir / "decode_contrasts.csv", "decode_contrasts"),
        (aggregate_run_dir / "covariance_summary.csv", "covariance_summary"),
    ]
    for path, role in aggregate_files:
        sources.append(SourceFile("aggregate", aggregate_run_dir, path, role))
    for local_dir in local_pairing_dirs:
        for name in [
            "decode_summary.csv",
            "decode_contrasts.csv",
            "incremental_decode_summary.csv",
            "incremental_gain_vs_static.csv",
            "incremental_gain_contrasts.csv",
        ]:
            sources.append(SourceFile("local_Iz", local_dir, local_dir / name, name.removesuffix(".csv")))
    for joint_dir in joint_feature_dirs:
        for name in [
            "feature_posterior_summary.csv",
            "feature_axis_contrasts.csv",
            "feature_motion_evidence_contrasts.csv",
            "feature_posterior_uncertainty.csv",
        ]:
            sources.append(SourceFile("joint_posterior", joint_dir, joint_dir / name, name.removesuffix(".csv")))
    return sources


def build_inventory(sources: list[SourceFile]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    for source in sources:
        exists = source.source_file.exists()
        source_manifest.append(
            {
                "branch": source.branch,
                "source_dir": str(source.source_dir),
                "source_file": str(source.source_file),
                "table_role": source.table_role,
                "exists": bool(exists),
                "status": "source_file_present" if exists else "source_file_missing",
                "recommended_action": "" if exists else "verify path or complete upstream cache-only posthoc",
            }
        )
        if not exists:
            continue
        if source.table_role == "feature_posterior_uncertainty":
            rows.extend(_normalize_joint_uncertainty(source))
        elif source.table_role == "feature_axis_contrasts":
            rows.extend(_normalize_joint_axis_wide(source))
        else:
            rows.extend(_normalize_decode_like(source))
    return pd.DataFrame(rows), source_manifest


def _metric_name(row: pd.Series) -> str:
    branch = str(row["branch"])
    contrast = str(row["control_contrast"])
    response = str(row["response_summary"])
    if branch == "aggregate":
        return PRIMARY_AGGREGATE_CONTRASTS.get(contrast, "")
    if branch == "local_Iz":
        return PRIMARY_LOCAL_CONTRASTS.get(contrast, "")
    if branch == "joint_posterior":
        if "joint_minus_zero_feature_gain_parallel_minus_orthogonal" in contrast or response == "joint_minus_zero_feature_gain_parallel_minus_orthogonal":
            return "joint_parallel_minus_orthogonal_feature_gain"
        if "joint_parallel_minus_orthogonal" in contrast or response == "joint_parallel_minus_orthogonal":
            return "joint_parallel_minus_orthogonal"
        if response == "joint_minus_zero_feature_gain" and "-zero:" in contrast:
            return "joint_minus_zero_feature_gain"
    return ""


def build_branch_metrics(
    inventory: pd.DataFrame,
    *,
    requested_latents: list[str],
    requested_k: list[int],
    requested_summaries: list[str],
    primary_scales: list[str],
    sentinel_scales: list[str],
) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    latent_set = set(requested_latents)
    k_set = set(requested_k)
    summary_set = set(requested_summaries)
    scale_set = set(primary_scales + sentinel_scales)
    for _, rec in inventory.iterrows():
        branch = str(rec.get("branch"))
        table_role = str(rec.get("table_role"))
        if branch in {"aggregate", "local_Iz"} and table_role not in {
            "incremental_gain_vs_static",
            "incremental_gain_contrasts",
        }:
            continue
        if branch == "joint_posterior" and table_role not in {
            "feature_posterior_uncertainty",
            "feature_axis_contrasts",
        }:
            continue
        latent = str(rec.get("latent", ""))
        k = _safe_int(rec.get("k"), None)
        summary = str(rec.get("response_summary", ""))
        scale = str(rec.get("scale", ""))
        if latent_set and latent not in latent_set:
            continue
        if k_set and k not in k_set:
            continue
        if scale_set and scale and scale not in scale_set:
            continue
        metric = _metric_name(rec)
        if not metric:
            continue
        if branch in {"aggregate", "local_Iz"} and summary_set and summary not in summary_set:
            continue
        row_response_summary = metric if branch == "joint_posterior" else summary
        evidence, ci_pass = _evidence_score(rec.get("estimate"), rec.get("ci_low"), rec.get("ci_high"))
        rows.append(
            {
                "branch": rec.get("branch"),
                "latent": latent,
                "k": k,
                "response_summary": row_response_summary,
                "feature_space": rec.get("feature_space", "selected_windows_zscore_pca"),
                "scale": scale,
                "scale_role": "sentinel" if scale in sentinel_scales else "primary",
                "metric": metric,
                "control_contrast": rec.get("control_contrast"),
                "estimate": _safe_float(rec.get("estimate")),
                "ci_low": _safe_float(rec.get("ci_low")),
                "ci_high": _safe_float(rec.get("ci_high")),
                "p_value": _safe_float(rec.get("p_value")),
                "evidence_score": evidence,
                "ci_pass_expected_direction": ci_pass,
                "n_images": rec.get("n_images"),
                "n_trials": rec.get("n_trials"),
                "n_sessions": rec.get("n_sessions"),
                "source_file": rec.get("source_file"),
                "known_caveat": rec.get("known_caveat"),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.drop_duplicates(
        subset=[
            "branch",
            "latent",
            "k",
            "response_summary",
            "scale",
            "metric",
            "control_contrast",
        ]
    ).reset_index(drop=True)


def _mean_or_zero(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else 0.0


def _branch_score(metrics: pd.DataFrame, branch: str, *, include_sentinel: bool = False) -> float:
    if metrics.empty:
        return 0.0
    sub = metrics[metrics["branch"] == branch]
    if not include_sentinel:
        sub = sub[sub["scale_role"] == "primary"]
    if sub.empty:
        return 0.0
    by_metric = sub.groupby("metric")["evidence_score"].mean()
    return float(np.clip(by_metric.mean(), -3.0, 3.0))


def _metric_subset_score(metrics: pd.DataFrame, branch: str, metric_names: set[str]) -> float:
    if metrics.empty:
        return 0.0
    sub = metrics[
        (metrics["branch"] == branch)
        & (metrics["scale_role"] == "primary")
        & (metrics["metric"].isin(metric_names))
    ]
    if sub.empty:
        return 0.0
    by_metric = sub.groupby("metric")["evidence_score"].mean()
    return float(np.clip(by_metric.mean(), -3.0, 3.0))


def _sign_reversal_penalty(metrics: pd.DataFrame) -> float:
    if metrics.empty:
        return 0.0
    penalties = 0
    for _, group in metrics.groupby(["branch", "metric"], dropna=False):
        signs = np.sign(pd.to_numeric(group["estimate"], errors="coerce").to_numpy(dtype=np.float64))
        signs = signs[np.isfinite(signs) & (signs != 0)]
        if np.any(signs > 0) and np.any(signs < 0):
            penalties += 1
    return float(min(2.0, 0.25 * penalties))


def _sentinel_penalty(metrics: pd.DataFrame) -> float:
    if metrics.empty:
        return 0.0
    primary = metrics[metrics["scale_role"] == "primary"]
    sentinel = metrics[metrics["scale_role"] == "sentinel"]
    if sentinel.empty:
        return 0.0
    primary_score = _mean_or_zero(primary["evidence_score"]) if not primary.empty else 0.0
    sentinel_score = _mean_or_zero(sentinel["evidence_score"])
    if sentinel_score > 0.5 and primary_score <= 0.0:
        return 1.0
    if sentinel_score > primary_score + 1.0:
        return 0.5
    return 0.0


def _interpretability_penalty(latent: str, k: int, response_summary: str) -> float:
    penalty = 0.0
    if "local_field" not in latent:
        penalty += 0.35
    if latent.startswith("dct"):
        penalty += 0.25
    if k >= 32:
        penalty += 0.15
    if response_summary in {"temporal_dct", "temporal_dct_delta"}:
        penalty += 0.45
    if response_summary == "temporal_delta_pca":
        penalty += 0.2
    if response_summary == "mean":
        penalty += 0.4
    if response_summary == "delta_mean":
        penalty -= 0.1
    if response_summary == "temporal_pca":
        penalty -= 0.05
    return max(0.0, penalty)


def rank_feature_specs(
    metrics: pd.DataFrame,
    *,
    requested_latents: list[str],
    requested_k: list[int],
    requested_summaries: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for latent in requested_latents:
        for k in requested_k:
            joint_for_latent_k = metrics[
                (metrics["branch"] == "joint_posterior")
                & (metrics["latent"] == latent)
                & (metrics["k"] == k)
            ]
            for summary in requested_summaries:
                summary_sub = metrics[
                    (metrics["latent"] == latent)
                    & (metrics["k"] == k)
                    & (metrics["branch"].isin(["aggregate", "local_Iz"]))
                    & (metrics["response_summary"] == summary)
                ]
                if summary_sub.empty:
                    continue
                scored_sub = pd.concat([summary_sub, joint_for_latent_k], ignore_index=True, sort=False)
                aggregate_score = _branch_score(summary_sub, "aggregate")
                local_score = _branch_score(summary_sub, "local_Iz")
                joint_axis_score = _metric_subset_score(joint_for_latent_k, "joint_posterior", JOINT_AXIS_METRICS)
                joint_generic_score = _metric_subset_score(joint_for_latent_k, "joint_posterior", JOINT_GENERIC_METRICS)
                reversal_penalty = _sign_reversal_penalty(summary_sub)
                sentinel_penalty = _sentinel_penalty(summary_sub)
                interpretability_penalty = _interpretability_penalty(latent, int(k), summary)
                no_joint_score = float(aggregate_score + local_score - reversal_penalty - sentinel_penalty - interpretability_penalty)
                with_joint_score = float(no_joint_score + joint_axis_score)
                summary_branches = set(str(v) for v in summary_sub["branch"].dropna())
                latent_k_branches = set(str(v) for v in scored_sub["branch"].dropna())
                present_branches = ",".join(sorted(summary_branches))
                latent_k_present_branches = ",".join(sorted(latent_k_branches))
                n_present_branches = len(summary_branches)
                meets_min_branch_coverage = bool(n_present_branches >= 2)
                rows.append(
                    {
                        "latent": latent,
                        "k": int(k),
                        "response_summary": summary,
                        "feature_space": "selected_windows_zscore_pca",
                        "aggregate_score": aggregate_score,
                        "local_Iz_score": local_score,
                        "joint_axis_score": joint_axis_score,
                        "joint_generic_score": joint_generic_score,
                        "score_without_joint_axis_term": no_joint_score,
                        "score_with_joint_axis_term": with_joint_score,
                        "sentinel_penalty": sentinel_penalty,
                        "sign_reversal_penalty": reversal_penalty,
                        "interpretability_penalty": interpretability_penalty,
                        "present_branches": present_branches,
                        "latent_k_present_branches": latent_k_present_branches,
                        "n_present_branches": n_present_branches,
                        "meets_min_branch_coverage": meets_min_branch_coverage,
                        "n_metric_rows": int(len(scored_sub)),
                        "n_summary_metric_rows": int(len(summary_sub)),
                        "n_joint_metric_rows": int(len(joint_for_latent_k)),
                        "n_ci_pass_rows": int(np.sum(scored_sub["ci_pass_expected_direction"].astype(bool))),
                        "n_summary_ci_pass_rows": int(np.sum(summary_sub["ci_pass_expected_direction"].astype(bool))),
                        "n_joint_ci_pass_rows": int(np.sum(joint_for_latent_k["ci_pass_expected_direction"].astype(bool))),
                    }
                )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    return ranking.sort_values(
        ["meets_min_branch_coverage", "score_with_joint_axis_term", "score_without_joint_axis_term", "n_ci_pass_rows"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_completion_manifest(
    inventory: pd.DataFrame,
    source_manifest: list[dict[str, Any]],
    *,
    requested_latents: list[str],
    requested_k: list[int],
    requested_summaries: list[str],
    primary_scales: list[str],
    sentinel_scales: list[str],
) -> pd.DataFrame:
    rows = list(source_manifest)
    scales = primary_scales + sentinel_scales
    for branch in ["aggregate", "local_Iz"]:
        branch_inv = inventory[inventory["branch"] == branch] if not inventory.empty else pd.DataFrame()
        for latent in requested_latents:
            for k in requested_k:
                for summary in requested_summaries:
                    for scale in scales:
                        present = False
                        if not branch_inv.empty:
                            present = bool(
                                (
                                    (branch_inv["latent"] == latent)
                                    & (branch_inv["k"] == k)
                                    & (branch_inv["response_summary"] == summary)
                                    & (branch_inv["scale"] == scale)
                                ).any()
                            )
                        rows.append(
                            {
                                "branch": branch,
                                "latent": latent,
                                "k": k,
                                "response_summary": summary,
                                "scale": scale,
                                "status": "present" if present else "missing_cache_only_gap",
                                "recommended_action": "" if present else "run cache-only incremental decode/posthoc for this cell",
                            }
                        )
    joint_inv = inventory[inventory["branch"] == "joint_posterior"] if not inventory.empty else pd.DataFrame()
    for latent in requested_latents:
        for k in requested_k:
            for scale in scales:
                present = False
                if not joint_inv.empty:
                    present = bool(
                        (
                            (joint_inv["latent"] == latent)
                            & (joint_inv["k"] == k)
                            & (joint_inv["scale"] == scale)
                        ).any()
                    )
                rows.append(
                    {
                        "branch": "joint_posterior",
                        "latent": latent,
                        "k": k,
                        "response_summary": "feature_posterior",
                        "scale": scale,
                        "status": "present" if present else "missing_cache_only_gap",
                        "recommended_action": "" if present else "rerun analyze_feature_posterior.py on compatible cache",
                    }
                )
    return pd.DataFrame(rows)


def write_inventory_md(path: Path, inventory: pd.DataFrame, completion: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Existing Feature Result Inventory", ""]
    if inventory.empty:
        lines.append("No result rows were found.")
    else:
        lines.append(f"Inventoried rows: {len(inventory)}")
        lines.append("")
        by_branch = inventory.groupby("branch").size().reset_index(name="rows")
        lines.append(_markdown_table(by_branch))
        lines.append("")
        primary = inventory[inventory["claim_role"].astype(str) != ""].copy()
        if not primary.empty:
            cols = ["branch", "latent", "k", "response_summary", "scale", "control_contrast", "estimate", "ci_low", "ci_high", "claim_role"]
            lines.append("## Claim-Relevant Rows")
            lines.append("")
            lines.append(_markdown_table(primary[cols], max_rows=80))
            lines.append("")
    if not completion.empty and "status" in completion.columns:
        gaps = completion[completion["status"] == "missing_cache_only_gap"]
        lines.append("## Cache-Only Gaps")
        lines.append("")
        lines.append(f"Missing requested cells: {len(gaps)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ranking_report(
    path: Path,
    ranking: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    primary_scales: list[str],
    sentinel_scales: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Feature Spec Ranking Report", ""]
    lines.append(f"Primary scales: {', '.join(primary_scales)}")
    lines.append(f"Sentinel scales: {', '.join(sentinel_scales)}")
    lines.append("")
    lines.append(
        "Branch coverage for `response_summary` is counted from aggregate/local rows; joint-posterior rows add latent/k support only."
    )
    lines.append("")
    if ranking.empty:
        lines.append("No rankable feature specifications were found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    top = ranking.iloc[0].to_dict()
    lines.append("## Nominations")
    lines.append("")
    lines.append("primary_locked_feature_spec:")
    lines.append(f"  latent = {top['latent']}")
    lines.append(f"  k = {top['k']}")
    lines.append(f"  response_summary = {top['response_summary']}")
    lines.append("  feature_space = selected_windows_zscore_pca")
    lines.append(f"  primary_scales = {', '.join(primary_scales)}")
    lines.append("")
    if len(ranking) > 1:
        second = ranking.iloc[1].to_dict()
        lines.append("secondary_sensitivity_feature_spec:")
        lines.append(f"  latent = {second['latent']}")
        lines.append(f"  k = {second['k']}")
        lines.append(f"  response_summary = {second['response_summary']}")
        lines.append("")
    negative = ranking.sort_values("score_with_joint_axis_term", ascending=True).iloc[0].to_dict()
    lines.append("negative_control_feature_spec:")
    lines.append(f"  latent = {negative['latent']}")
    lines.append(f"  k = {negative['k']}")
    lines.append(f"  response_summary = {negative['response_summary']}")
    lines.append("")
    show_cols = [
        "latent",
        "k",
        "response_summary",
        "aggregate_score",
        "local_Iz_score",
        "joint_axis_score",
        "joint_generic_score",
        "score_without_joint_axis_term",
        "score_with_joint_axis_term",
        "sentinel_penalty",
        "sign_reversal_penalty",
        "present_branches",
        "latent_k_present_branches",
        "meets_min_branch_coverage",
        "n_ci_pass_rows",
        "n_summary_ci_pass_rows",
        "n_joint_ci_pass_rows",
    ]
    lines.append("## Top Ranked Specs")
    lines.append("")
    lines.append(_markdown_table(ranking[show_cols], max_rows=20))
    lines.append("")
    if not metrics.empty:
        top_metrics = metrics[
            (metrics["latent"] == top["latent"])
            & (metrics["k"] == top["k"])
            & (
                (metrics["response_summary"] == top["response_summary"])
                | (metrics["branch"] == "joint_posterior")
            )
        ].copy()
        if not top_metrics.empty:
            cols = ["branch", "metric", "scale", "estimate", "ci_low", "ci_high", "evidence_score", "control_contrast"]
            lines.append("## Primary Spec Evidence Rows")
            lines.append("")
            lines.append(_markdown_table(top_metrics[cols].sort_values(["branch", "metric", "scale"])))
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_locked_spec(
    md_path: Path,
    json_path: Path,
    ranking: pd.DataFrame,
    *,
    primary_scales: list[str],
    sentinel_scales: list[str],
    configs: dict[str, dict[str, Any]],
    lock_allowed: bool,
    missing_gap_count: int,
    args: argparse.Namespace,
) -> None:
    if ranking.empty:
        payload = {
            "status": "not_locked",
            "reason_for_locking": "No rankable cached feature specification was available.",
        }
    else:
        top = ranking.iloc[0].to_dict()
        meets_branch_coverage = bool(top.get("meets_min_branch_coverage", False))
        status = "locked" if lock_allowed and meets_branch_coverage and missing_gap_count == 0 else "provisional_not_locked"
        payload = {
            "status": status,
            "canonical_run_allowed": bool(status == "locked"),
            "latent_name": top["latent"],
            "latent_crop_px": _config_value(configs, "latent_crop_px"),
            "center_crop_px": _config_value(configs, "center_crop_px"),
            "local_field_grid": _config_value(configs, "local_field_grid"),
            "feature_space": top.get("feature_space", "selected_windows_zscore_pca"),
            "pca_k": int(top["k"]),
            "response_summary_primary": top["response_summary"],
            "response_summary_secondary": "temporal_pca" if top["response_summary"] != "temporal_pca" else "delta_mean",
            "ridge_alpha_policy": {
                "fixed_ridge_alpha": _config_value(configs, "fixed_ridge_alpha"),
                "ridge_alphas": _config_value(configs, "ridge_alphas"),
                "canonical_run_preference": "fixed/shared ridge",
            },
            "primary_scales": primary_scales,
            "sentinel_scales": sentinel_scales,
            "primary_control_contrasts": {
                "aggregate": list(PRIMARY_AGGREGATE_CONTRASTS),
                "local_Iz": list(PRIMARY_LOCAL_CONTRASTS),
                "joint_posterior": list(PRIMARY_JOINT_METRICS),
            },
            "score_with_joint_axis_term": float(top["score_with_joint_axis_term"]),
            "score_without_joint_axis_term": float(top["score_without_joint_axis_term"]),
            "joint_axis_score": float(top["joint_axis_score"]),
            "joint_generic_score": float(top["joint_generic_score"]),
            "present_branches": top["present_branches"],
            "latent_k_present_branches": top.get("latent_k_present_branches", top["present_branches"]),
            "n_present_branches": int(top["n_present_branches"]),
            "missing_cache_only_gap_count": int(missing_gap_count),
            "reason_for_locking": (
                "Highest currently eligible cache-first stability score. This is not a final "
                "lock unless status is 'locked' and canonical_run_allowed is true."
            ),
            "known_failure_modes": [
                "The first implementation ranks only existing cached outputs and does not render new responses.",
                "Joint posterior contributes latent/k support, but response_summary must be supported by aggregate/local rows.",
                "Joint hard-negative feature posterior may favor orthogonal across-edge discrimination.",
                "2x sentinel positives are penalized and should not be used as the sole claim source.",
                "Missing k or summary cells in posthoc_completion_manifest.csv should be filled before a manuscript-level lock.",
            ],
            "adjudication_out_dir": str(args.out_dir),
        }
    _write_json(json_path, payload)
    lines = ["# Locked Feature Decomposition Spec", ""]
    for key, value in payload.items():
        if isinstance(value, (list, dict)):
            lines.append(f"{key}:")
            lines.append("```json")
            lines.append(json.dumps(_json_ready(value), indent=2, sort_keys=True))
            lines.append("```")
        else:
            lines.append(f"{key}: {value}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _remove_generated_lock_files(out_dir: Path) -> None:
    for path in [
        out_dir / "locked_feature_decomposition_spec.md",
        out_dir / "locked_feature_decomposition_spec.json",
    ]:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        markers = [
            "provisional_cache_adjudication_lock",
            "provisional_not_locked",
            "cache-first stability score",
            "adjudication_out_dir",
        ]
        if any(marker in text for marker in markers):
            path.unlink()


def make_figures(out_dir: Path, ranking: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    made: list[str] = []
    if ranking.empty and metrics.empty:
        return made
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return made

    if not ranking.empty:
        fig, ax = plt.subplots(figsize=(9, max(3, min(10, 0.35 * len(ranking.head(20))))))
        top = ranking.head(20).iloc[::-1]
        labels = [f"{r.latent} k{int(r.k)} {r.response_summary}" for r in top.itertuples()]
        ax.barh(np.arange(len(top)), top["score_with_joint_axis_term"], color="#4c78a8", label="with joint")
        ax.barh(np.arange(len(top)), top["score_without_joint_axis_term"], color="#f58518", alpha=0.55, label="without joint")
        ax.set_yticks(np.arange(len(top)), labels=labels, fontsize=7)
        ax.set_xlabel("stability score")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out_dir / "fig_feature_spec_heatmap_by_branch.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(str(path))

    if not metrics.empty:
        curve = (
            metrics[metrics["scale_role"] == "primary"]
            .groupby(["branch", "latent", "k"], dropna=False)["evidence_score"]
            .mean()
            .reset_index()
        )
        if not curve.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for (branch, latent), sub in curve.groupby(["branch", "latent"]):
                sub = sub.sort_values("k")
                ax.plot(sub["k"], sub["evidence_score"], marker="o", label=f"{branch} {latent}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("PCA k")
            ax.set_ylabel("mean primary-scale evidence")
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            path = out_dir / "fig_k_stability_curves.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            made.append(str(path))

        sentinel = metrics.groupby(["scale_role", "branch"], dropna=False)["evidence_score"].mean().reset_index()
        if not sentinel.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            pivot = sentinel.pivot(index="branch", columns="scale_role", values="evidence_score").fillna(0.0)
            pivot.plot(kind="bar", ax=ax, color=["#54a24b", "#e45756"])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("mean evidence")
            ax.set_xlabel("")
            fig.tight_layout()
            path = out_dir / "fig_small_scale_vs_2x_sentinel.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            made.append(str(path))
    return made


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-run-dir", type=Path, default=DEFAULT_AGGREGATE_RUN_DIR)
    parser.add_argument("--aggregate-incremental-dir", type=Path, default=None)
    parser.add_argument("--local-pairing-dirs", default=str(DEFAULT_LOCAL_PAIRING_DIRS))
    parser.add_argument("--joint-feature-dirs", default=str(DEFAULT_JOINT_FEATURE_DIRS))
    parser.add_argument("--joint-run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--latent-names", default="gabor_local_field,pyramid_local_field")
    parser.add_argument("--k-list", default="2,4,8,16,32")
    parser.add_argument(
        "--summaries",
        default="delta_mean,temporal_pca,temporal_delta_pca,temporal_dct,temporal_dct_delta,mean",
    )
    parser.add_argument("--primary-scales", default="rel_0p25x,rel_0p5x,rel_1x")
    parser.add_argument("--sentinel-scales", default="rel_2x")
    parser.add_argument("--session-bootstrap-n", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--write-lock",
        action="store_true",
        help="Emit locked_feature_decomposition_spec.* only when all requested cache-only cells are present.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.aggregate_incremental_dir = args.aggregate_incremental_dir or (
        args.aggregate_run_dir / "incremental_static_plus_motion_relids"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requested_latents = _parse_str_list(args.latent_names)
    requested_k = _parse_int_list(args.k_list)
    requested_summaries = _parse_str_list(args.summaries)
    primary_scales = [_normalize_scale(scale) for scale in _parse_str_list(args.primary_scales)]
    sentinel_scales = [_normalize_scale(scale) for scale in _parse_str_list(args.sentinel_scales)]
    local_dirs = _parse_path_list(args.local_pairing_dirs)
    joint_dirs = _parse_path_list(args.joint_feature_dirs)

    sources = _source_files(
        aggregate_run_dir=args.aggregate_run_dir,
        aggregate_incremental_dir=args.aggregate_incremental_dir,
        local_pairing_dirs=local_dirs,
        joint_feature_dirs=joint_dirs,
    )
    inventory, source_manifest = build_inventory(sources)
    completion = build_completion_manifest(
        inventory,
        source_manifest,
        requested_latents=requested_latents,
        requested_k=requested_k,
        requested_summaries=requested_summaries,
        primary_scales=primary_scales,
        sentinel_scales=sentinel_scales,
    )
    metrics = build_branch_metrics(
        inventory,
        requested_latents=requested_latents,
        requested_k=requested_k,
        requested_summaries=requested_summaries,
        primary_scales=primary_scales,
        sentinel_scales=sentinel_scales,
    )
    ranking = rank_feature_specs(
        metrics,
        requested_latents=requested_latents,
        requested_k=requested_k,
        requested_summaries=requested_summaries,
    )
    missing_gap_count = 0
    if not completion.empty and "status" in completion.columns:
        missing_gap_count = int(np.sum(completion["status"] == "missing_cache_only_gap"))
    top_meets_branch_coverage = False
    if not ranking.empty and "meets_min_branch_coverage" in ranking.columns:
        top_meets_branch_coverage = bool(ranking.iloc[0]["meets_min_branch_coverage"])
    lock_allowed = bool(args.write_lock and missing_gap_count == 0 and top_meets_branch_coverage)
    configs = _load_source_configs(args)

    _write_json(
        args.out_dir / "run_metadata.json",
        {
            "aggregate_run_dir": args.aggregate_run_dir,
            "aggregate_incremental_dir": args.aggregate_incremental_dir,
            "local_pairing_dirs": local_dirs,
            "joint_feature_dirs": joint_dirs,
            "joint_run_dir": args.joint_run_dir,
            "out_dir": args.out_dir,
            "latent_names": requested_latents,
            "k_list": requested_k,
            "summaries": requested_summaries,
            "primary_scales": primary_scales,
            "sentinel_scales": sentinel_scales,
            "session_bootstrap_n": args.session_bootstrap_n,
            "random_seed": args.random_seed,
            "cache_first": True,
            "write_lock_requested": bool(args.write_lock),
            "lock_allowed": lock_allowed,
            "missing_cache_only_gap_count": missing_gap_count,
            "top_meets_branch_coverage": top_meets_branch_coverage,
        },
    )
    _write_csv(args.out_dir / "existing_feature_result_inventory.csv", inventory.to_dict("records"))
    write_inventory_md(args.out_dir / "existing_feature_result_inventory.md", inventory, completion)
    _write_csv(args.out_dir / "posthoc_completion_manifest.csv", completion.to_dict("records"))
    _write_csv(args.out_dir / "feature_spec_branch_metrics.csv", metrics.to_dict("records"))

    stability = pd.DataFrame()
    if not metrics.empty:
        stability = (
            metrics.groupby(["latent", "k", "response_summary", "branch", "metric"], dropna=False)
            .agg(
                mean_evidence_score=("evidence_score", "mean"),
                mean_estimate=("estimate", "mean"),
                n_rows=("estimate", "size"),
                n_ci_pass=("ci_pass_expected_direction", "sum"),
            )
            .reset_index()
        )
    _write_csv(args.out_dir / "feature_spec_stability_scores.csv", stability.to_dict("records"))
    _write_csv(args.out_dir / "feature_spec_ranking.csv", ranking.to_dict("records"))
    write_ranking_report(
        args.out_dir / "feature_spec_ranking_report.md",
        ranking,
        metrics,
        primary_scales=primary_scales,
        sentinel_scales=sentinel_scales,
    )
    spec_stem = "locked_feature_decomposition_spec" if lock_allowed else "provisional_feature_decomposition_spec"
    if not lock_allowed:
        _remove_generated_lock_files(args.out_dir)
    write_locked_spec(
        args.out_dir / f"{spec_stem}.md",
        args.out_dir / f"{spec_stem}.json",
        ranking,
        primary_scales=primary_scales,
        sentinel_scales=sentinel_scales,
        configs=configs,
        lock_allowed=lock_allowed,
        missing_gap_count=missing_gap_count,
        args=args,
    )
    _write_json(
        args.out_dir / "lock_readiness.json",
        {
            "write_lock_requested": bool(args.write_lock),
            "lock_allowed": lock_allowed,
            "missing_cache_only_gap_count": missing_gap_count,
            "top_meets_branch_coverage": top_meets_branch_coverage,
            "locked_files_written": lock_allowed,
            "provisional_files_written": not lock_allowed,
        },
    )
    figures = [] if args.skip_figures else make_figures(args.out_dir, ranking, metrics)
    if figures:
        _write_json(args.out_dir / "figure_manifest.json", {"figures": figures})

    print(f"[feature-adjudication] wrote {len(inventory)} inventory rows to {args.out_dir}", flush=True)
    print(f"[feature-adjudication] wrote {len(ranking)} ranked feature specs", flush=True)


if __name__ == "__main__":
    main()
