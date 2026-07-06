"""Audit inherited decoder contracts for the joint feature observer in 4C.

This is a cache-level integrity gate for the joint observer artifacts. It does
not refit the observer; it checks that existing outputs make their metric axis,
source grouping, and CI provenance explicit enough to avoid the issues found in
the aggregate and local feature-information analyses.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PANEL_C_DIAGNOSTICS = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_C"
    / "diagnostics"
)
CONTINUOUS_JOINT_DIR = PANEL_C_DIAGNOSTICS / "continuous_joint"
PROMOTED_MANIFEST = CONTINUOUS_JOINT_DIR / "continuous_joint_promoted_observer_manifest.json"
DEFAULT_OUT_DIR = CONTINUOUS_JOINT_DIR / "inherited_decoder_audit_20260630"

CI_SPECS = [
    ("mean_feature_cosine_delta", "ci_low", "ci_high"),
    ("mean_lhs_minus_rhs", "mean_lhs_minus_rhs_ci_low", "mean_lhs_minus_rhs_ci_high"),
    ("point_estimate", "ci_low", "ci_high"),
    ("mean_delta", "ci_low", "ci_high"),
]


@dataclass
class AuditRow:
    status: str
    category: str
    artifact: str
    check: str
    detail: str


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _add(rows: list[AuditRow], status: str, category: str, artifact: Path | str, check: str, detail: str) -> None:
    artifact_text = _rel(artifact) if isinstance(artifact, Path) else str(artifact)
    rows.append(
        AuditRow(
            status=str(status),
            category=str(category),
            artifact=artifact_text,
            check=str(check),
            detail=str(detail),
        )
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _source_row_from_id(value: object) -> int | None:
    match = re.fullmatch(r"source_row:(\d+)", str(value))
    if not match:
        return None
    return int(match.group(1))


def _status_counts(rows: list[AuditRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def _audit_metric_axis(rows: list[AuditRow], manifest_path: Path) -> dict[str, Any] | None:
    manifest = _read_json(manifest_path)
    if manifest is None:
        _add(rows, "FAIL", "metric_axis", manifest_path, "promoted_manifest_exists", "Missing promoted manifest.")
        return None
    text = json.dumps(manifest, sort_keys=True)
    if "feature-recovery" in text or "mean_feature_cosine" in text:
        _add(
            rows,
            "PASS",
            "metric_axis",
            manifest_path,
            "feature_recovery_axis_declared",
            "Promoted observer manifest declares feature-recovery metrics rather than information bits.",
        )
    else:
        _add(
            rows,
            "FAIL",
            "metric_axis",
            manifest_path,
            "feature_recovery_axis_declared",
            "Manifest does not explicitly expose the feature-recovery metric axis.",
        )
    if "bits" in text.lower():
        _add(
            rows,
            "WARN",
            "metric_axis",
            manifest_path,
            "bits_language_absent",
            "Manifest contains 'bits'; ensure 4C is not described as an information-bit endpoint.",
        )
    else:
        _add(rows, "PASS", "metric_axis", manifest_path, "bits_language_absent", "No bits language found.")
    return manifest


def _audit_promoted_observer_math(rows: list[AuditRow], run_dir: Path) -> None:
    posterior_path = run_dir / "continuous_joint_feature_posterior.csv"
    posterior = _read_csv(posterior_path)
    if posterior is None:
        _add(rows, "FAIL", "posterior_math", posterior_path, "posterior_exists", "Missing posterior CSV.")
        return
    required = {"table_index", "observer_mode", "candidate_score", "candidate_score_raw", "posterior_temperature", "candidate_posterior"}
    missing = required.difference(posterior.columns)
    if missing:
        _add(
            rows,
            "FAIL",
            "posterior_math",
            posterior_path,
            "required_columns",
            f"Missing columns: {sorted(missing)}.",
        )
        return
    _add(
        rows,
        "PASS",
        "posterior_math",
        posterior_path,
        "required_columns",
        f"Posterior table has {posterior.shape[0]} candidate rows.",
    )
    block = posterior[posterior["observer_mode"].eq("continuous_joint")].copy()
    if block.empty:
        _add(rows, "FAIL", "posterior_math", posterior_path, "continuous_joint_rows", "No continuous_joint rows found.")
        return
    expected_score = block["candidate_score_raw"].to_numpy(dtype=float) / block["posterior_temperature"].to_numpy(dtype=float)
    score_error = float(np.nanmax(np.abs(block["candidate_score"].to_numpy(dtype=float) - expected_score)))
    if score_error <= 1e-9:
        _add(
            rows,
            "PASS",
            "posterior_math",
            posterior_path,
            "raw_temperature_score_consistency",
            f"Max |score - raw/temp| = {score_error:.3g}.",
        )
    else:
        _add(
            rows,
            "FAIL",
            "posterior_math",
            posterior_path,
            "raw_temperature_score_consistency",
            f"Max |score - raw/temp| = {score_error:.3g}.",
        )
    posterior_sums = block.groupby("table_index", sort=False)["candidate_posterior"].sum().to_numpy(dtype=float)
    max_mass_error = float(np.nanmax(np.abs(posterior_sums - 1.0)))
    if max_mass_error <= 1e-9:
        _add(
            rows,
            "PASS",
            "posterior_math",
            posterior_path,
            "posterior_mass_sums_to_one",
            f"Max table posterior-mass error = {max_mass_error:.3g}.",
        )
    else:
        _add(
            rows,
            "FAIL",
            "posterior_math",
            posterior_path,
            "posterior_mass_sums_to_one",
            f"Max table posterior-mass error = {max_mass_error:.3g}.",
        )


def _audit_promoted_source_identity(rows: list[AuditRow], run_dir: Path) -> None:
    trials_path = run_dir / "continuous_joint_trials.csv"
    trials = _read_csv(trials_path)
    if trials is None:
        _add(rows, "FAIL", "source_identity", trials_path, "trials_exists", "Missing continuous joint trial CSV.")
        return
    if "true_image_id" not in trials.columns:
        _add(rows, "FAIL", "source_identity", trials_path, "true_source_parseable", "Missing true_image_id.")
        return
    source_rows = trials["true_image_id"].map(_source_row_from_id)
    missing = int(source_rows.isna().sum())
    if missing:
        _add(
            rows,
            "FAIL",
            "source_identity",
            trials_path,
            "true_source_parseable",
            f"{missing} rows have non-parseable true_image_id values.",
        )
    else:
        _add(
            rows,
            "PASS",
            "source_identity",
            trials_path,
            "true_source_parseable",
            f"Parsed {source_rows.nunique()} unique true source rows across {trials.shape[0]} tables.",
        )
    if "trial_id" in trials.columns and missing == 0:
        source_by_trial = pd.DataFrame({"trial_id": trials["trial_id"].astype(int), "source_row": source_rows.astype(int)})
        max_sources_per_trial = int(source_by_trial.groupby("trial_id")["source_row"].nunique().max())
        max_trials_per_source = int(source_by_trial.groupby("source_row")["trial_id"].nunique().max())
        if max_sources_per_trial == 1:
            _add(
                rows,
                "PASS",
                "source_identity",
                trials_path,
                "trial_id_groups_source",
                f"Each trial_id maps to one true source row; max trials per source = {max_trials_per_source}.",
            )
        else:
            _add(
                rows,
                "WARN",
                "source_identity",
                trials_path,
                "trial_id_groups_source",
                f"A trial_id can span {max_sources_per_trial} true source rows; prefer source_row split.",
            )
        if max_trials_per_source > 1:
            _add(
                rows,
                "WARN",
                "source_identity",
                trials_path,
                "source_row_groups_trial",
                f"A source row can span {max_trials_per_source} trial ids; source-row CV is stricter than trial-id CV.",
            )
        else:
            _add(
                rows,
                "PASS",
                "source_identity",
                trials_path,
                "source_row_groups_trial",
                "Each true source row maps to one trial_id in this cache.",
            )
    metadata_path = run_dir / "continuous_joint_metadata.json"
    metadata = _read_json(metadata_path)
    response_manifest_path = Path(str(metadata.get("response_manifest", ""))) if metadata else None
    if response_manifest_path is None or not response_manifest_path.exists():
        _add(rows, "WARN", "source_identity", metadata_path, "response_manifest_available", "Could not resolve response manifest.")
        return
    response_manifest = _read_csv(response_manifest_path)
    if response_manifest is None or "excluded_candidate_source_rows" not in response_manifest.columns:
        _add(
            rows,
            "WARN",
            "source_identity",
            response_manifest_path,
            "candidate_source_rows_available",
            "Response manifest lacks excluded_candidate_source_rows.",
        )
        return
    merged = trials[["response_cache_path", "true_image_id"]].copy()
    merged["true_source_row"] = merged["true_image_id"].map(_source_row_from_id)
    merged = merged.merge(
        response_manifest[["response_cache_path", "excluded_candidate_source_rows"]],
        on="response_cache_path",
        how="left",
    )
    def _in_candidate_list(row: pd.Series) -> bool:
        values = {int(part) for part in str(row["excluded_candidate_source_rows"]).split(",") if part.strip().isdigit()}
        return int(row["true_source_row"]) in values

    ok = merged.dropna(subset=["excluded_candidate_source_rows"]).apply(_in_candidate_list, axis=1)
    if bool(ok.all()) and ok.shape[0] == merged.shape[0]:
        _add(
            rows,
            "PASS",
            "source_identity",
            response_manifest_path,
            "true_source_in_candidate_set",
            "Every trial true source row is listed in the response-table candidate source rows.",
        )
    else:
        _add(
            rows,
            "FAIL",
            "source_identity",
            response_manifest_path,
            "true_source_in_candidate_set",
            f"{int((~ok).sum())} rows fail the response-manifest candidate-source check.",
        )


def _audit_supervised_decoder_splits(rows: list[AuditRow], trials_path: Path, label: str) -> None:
    trials = _read_csv(trials_path)
    if trials is None:
        _add(rows, "WARN", "supervised_split", trials_path, "trials_exists", f"{label} trial CSV is missing.")
        return
    if "true_source_row" not in trials.columns:
        _add(rows, "FAIL", "supervised_split", trials_path, "source_column_present", "Missing true_source_row.")
        return
    if "fold" not in trials.columns:
        _add(rows, "WARN", "supervised_split", trials_path, "fold_column_present", "Missing fold column.")
        return
    source_folds = trials.groupby("true_source_row")["fold"].nunique()
    leaking_sources = source_folds[source_folds > 1]
    if leaking_sources.empty:
        fold_counts = trials.groupby("fold")["true_source_row"].nunique().to_dict()
        _add(
            rows,
            "PASS",
            "supervised_split",
            trials_path,
            "source_row_disjoint_folds",
            f"{label}: each true source row appears in one fold; fold source counts={fold_counts}.",
        )
    else:
        _add(
            rows,
            "FAIL",
            "supervised_split",
            trials_path,
            "source_row_disjoint_folds",
            f"{label}: {leaking_sources.shape[0]} source rows span multiple folds.",
        )


def _audit_ci_file(rows: list[AuditRow], path: Path) -> None:
    data = _read_csv(path)
    if data is None:
        _add(rows, "WARN", "ci_integrity", path, "csv_exists", "CSV is missing.")
        return
    matched_specs = [spec for spec in CI_SPECS if all(col in data.columns for col in spec)]
    if not matched_specs:
        if "contrast" in path.name or "summary" in path.name:
            _add(rows, "WARN", "ci_integrity", path, "point_inside_ci", "No recognized point/CI columns found.")
        return
    for point_col, low_col, high_col in matched_specs:
        finite = data[[point_col, low_col, high_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if finite.empty:
            _add(
                rows,
                "WARN",
                "ci_integrity",
                path,
                f"{point_col}_inside_ci",
                "No finite point/CI rows found.",
            )
            continue
        inverted = finite[low_col] > finite[high_col]
        inverted_rows = finite[inverted]
        outside = finite[(finite[point_col] < finite[low_col] - 1e-12) | (finite[point_col] > finite[high_col] + 1e-12)]
        if not inverted_rows.empty:
            _add(
                rows,
                "FAIL",
                "ci_integrity",
                path,
                f"{point_col}_inside_ci",
                f"{inverted_rows.shape[0]} rows have inverted CI bounds.",
            )
        elif outside.empty:
            _add(
                rows,
                "PASS",
                "ci_integrity",
                path,
                f"{point_col}_inside_ci",
                f"All {finite.shape[0]} finite point estimates lie inside their CIs.",
            )
        else:
            preview = outside.head(3)[[point_col, low_col, high_col]].to_dict(orient="records")
            _add(
                rows,
                "FAIL",
                "ci_integrity",
                path,
                f"{point_col}_inside_ci",
                f"{outside.shape[0]} point estimates fall outside their CIs; preview={preview}.",
            )


def _audit_calibration_cv(rows: list[AuditRow], cv_path: Path, best_path: Path) -> None:
    cv = _read_csv(cv_path)
    if cv is None:
        _add(rows, "FAIL", "calibration_cv", cv_path, "cv_exists", "Promoted calibration CV CSV is missing.")
        return
    split_keys = sorted(str(value) for value in cv["split_key"].dropna().unique()) if "split_key" in cv.columns else []
    if "source_row" in split_keys:
        _add(
            rows,
            "PASS",
            "calibration_cv",
            cv_path,
            "source_row_split_present",
            f"Calibration CV contains split keys: {split_keys}.",
        )
    else:
        _add(
            rows,
            "FAIL",
            "calibration_cv",
            cv_path,
            "source_row_split_present",
            f"Calibration CV split keys are {split_keys}; rerun audit after source-row CV patch.",
        )
    best = _read_csv(best_path)
    if best is None or "split_key" not in best.columns:
        _add(rows, "WARN", "calibration_cv", best_path, "best_prefers_source_row", "Best CSV is missing or lacks split_key.")
        return
    best_splits = sorted(str(value) for value in best["split_key"].dropna().unique())
    if best_splits == ["source_row"]:
        _add(
            rows,
            "PASS",
            "calibration_cv",
            best_path,
            "best_prefers_source_row",
            "Promoted best rows are selected from source-row-heldout CV.",
        )
    else:
        _add(
            rows,
            "FAIL",
            "calibration_cv",
            best_path,
            "best_prefers_source_row",
            f"Promoted best rows use split keys {best_splits}; expected only source_row.",
        )


def run_audit(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[AuditRow] = []
    manifest_path = Path(args.promoted_manifest)
    manifest = _audit_metric_axis(rows, manifest_path)
    run_dir = Path(manifest["artifact"]["run_dir"]) if manifest else Path(args.promoted_run_dir)
    _audit_promoted_observer_math(rows, run_dir)
    _audit_promoted_source_identity(rows, run_dir)

    calibration_prefix = CONTINUOUS_JOINT_DIR / "continuous_joint_feature_calibration_audit_promoted"
    _audit_calibration_cv(
        rows,
        calibration_prefix.with_name(calibration_prefix.name + "_cv.csv"),
        calibration_prefix.with_name(calibration_prefix.name + "_best.csv"),
    )

    supervised_trials = [
        (
            PANEL_C_DIAGNOSTICS
            / "continuous_feature_embedding"
            / "continuous_feature_embedding_reconstruction_trials.csv",
            "continuous feature embedding",
        ),
        (
            PANEL_C_DIAGNOSTICS
            / "continuous_tau_mlp_feature_decoder_residual"
            / "continuous_tau_mlp_feature_decoder_trials.csv",
            "continuous tau MLP residual",
        ),
        (
            PANEL_C_DIAGNOSTICS
            / "continuous_tau_mlp_feature_decoder"
            / "continuous_tau_mlp_feature_decoder_trials.csv",
            "continuous tau MLP",
        ),
    ]
    for trials_path, label in supervised_trials:
        _audit_supervised_decoder_splits(rows, trials_path, label)

    ci_files = [
        PANEL_C_DIAGNOSTICS / "continuous_feature_embedding" / "continuous_feature_embedding_reconstruction_contrasts.csv",
        PANEL_C_DIAGNOSTICS / "continuous_feature_embedding_mlp_hammer" / "continuous_feature_embedding_reconstruction_contrasts.csv",
        PANEL_C_DIAGNOSTICS / "continuous_tau_mlp_feature_decoder" / "continuous_tau_mlp_feature_decoder_contrasts.csv",
        PANEL_C_DIAGNOSTICS
        / "continuous_tau_mlp_feature_decoder_residual"
        / "continuous_tau_mlp_feature_decoder_contrasts.csv",
        CONTINUOUS_JOINT_DIR / "continuous_joint_axis_trace_diagnostic_contrasts.csv",
    ]
    for path in ci_files:
        _audit_ci_file(rows, path)

    audit = pd.DataFrame([row.__dict__ for row in rows])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "panel_c_joint_inherited_decoder_audit.csv"
    summary_path = out_dir / "panel_c_joint_inherited_decoder_audit_summary.json"
    readme_path = out_dir / "README.md"
    audit.to_csv(audit_path, index=False)
    summary = {
        "status_counts": _status_counts(rows),
        "n_checks": int(audit.shape[0]),
        "promoted_manifest": manifest_path,
        "promoted_run_dir": run_dir,
        "outputs": {
            "audit_csv": audit_path,
            "summary_json": summary_path,
            "readme": readme_path,
        },
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Figure 4C Joint Decoder Inherited-Contract Audit",
        "",
        "This cache-level audit checks the same failure modes that were found in the aggregate/local feature-information analyses: metric-axis ambiguity, source-grouping provenance, stale point/CI summaries, and posterior temperature consistency.",
        "",
        f"Status counts: `{summary['status_counts']}`.",
        "",
        "The audit is intended as a gate before promoting new joint-observer artifacts. PASS means the cached output satisfies the contract; WARN means the artifact is diagnostic-only or lacks a non-blocking provenance field; FAIL requires regeneration or a code fix before promotion.",
        "",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    print(audit.to_string(index=False))
    print(f"wrote {audit_path}")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoted-manifest", type=Path, default=PROMOTED_MANIFEST)
    parser.add_argument(
        "--promoted-run-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "fixation_statistics_by_stimulus_all_sessions_after_review"
        / "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1"
        / "continuous_joint_quadratic_poisson_scale_conditioned_strict_scale_prior_predeclared_full",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main() -> None:
    run_audit(build_parser().parse_args())


if __name__ == "__main__":
    main()
