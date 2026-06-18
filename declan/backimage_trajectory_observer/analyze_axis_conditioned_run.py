"""Post-hoc analysis for axis-conditioned BackImage trajectory observer runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AXIS_PARALLEL = "axis_edge_parallel"
AXIS_ORTHOGONAL = "axis_edge_orthogonal"
AXIS_FAMILIES = (AXIS_PARALLEL, AXIS_ORTHOGONAL)

FEATURES = [
    ("image_patch_rms_contrast", "contrast"),
    ("image_gradient_energy", "gradient_energy"),
    ("image_edge_density", "edge_density"),
    ("image_orientation_coherence", "orientation_coherence"),
    ("image_spectrum_anisotropy", "spectrum_anisotropy"),
    ("image_high_freq_power_fraction", "high_freq_power"),
    ("image_power_8plus_cpd_fraction", "power_8plus_cpd"),
    ("drift_edge_cos2", "observed_drift_edge_parallelism"),
    ("drift_gradient_cos2", "observed_drift_gradient_parallelism"),
    ("drift_edge_delta_deg", "observed_drift_edge_delta_deg"),
    ("drift_gradient_delta_deg", "observed_drift_gradient_delta_deg"),
    ("static_response_distance_to_nearest_distractor", "nearest_static_response_distance"),
    ("structure_distance_to_nearest_distractor", "nearest_structure_distance"),
    ("contrast_distance_to_nearest_distractor", "nearest_contrast_distance"),
]

PAIR_KEYS = ["trial_id", "candidate_set_mode", "observation_scale", "prior_scale", "likelihood_scale"]


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({"true": 1.0, "1": 1.0, "yes": 1.0, "false": 0.0, "0": 0.0, "no": 0.0}).astype(float)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _pearson(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].corr(frame["y"]))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _observer_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, grp in trials.groupby(["candidate_set_mode", "prior_family", "prior_scale", "likelihood_scale"], dropna=False):
        candidate_set_mode, prior_family, prior_scale, likelihood_scale = key
        known = _bool_series(grp["known_correct"])
        zero = _bool_series(grp["zero_correct"])
        joint = _bool_series(grp["joint_correct"])
        rows.append(
            {
                "candidate_set_mode": candidate_set_mode,
                "prior_family": prior_family,
                "prior_scale": float(prior_scale),
                "likelihood_scale": float(likelihood_scale),
                "n_trials": int(len(grp)),
                "known_eye_accuracy": float(known.mean()),
                "zero_eye_accuracy": float(zero.mean()),
                "joint_eye_accuracy": float(joint.mean()),
                "joint_minus_zero_accuracy": float(joint.mean() - zero.mean()),
                "median_N_eff_fraction": float(_num(grp["N_eff_true_image_fraction"]).median()),
                "median_nearest_tau_rank": float(_num(grp["nearest_tau_rank"]).median()),
                "median_nearest_tau_distance": float(_num(grp["nearest_tau_distance"]).median()),
                "median_joint_true_margin": float(_num(grp["joint_true_margin"]).median()),
                "median_joint_minus_zero_true_score": float(_num(grp["joint_minus_zero_true_score"]).median()),
                "axis_shared_source_catalog_fraction": (
                    float(_bool_series(grp["axis_shared_source_catalog"]).mean())
                    if "axis_shared_source_catalog" in grp.columns
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate_set_mode", "prior_family", "prior_scale", "likelihood_scale"])


def _paired_trial_table(trials: pd.DataFrame) -> pd.DataFrame:
    subset = trials[trials["prior_family"].isin(AXIS_FAMILIES)].copy()
    if subset.empty:
        return pd.DataFrame()

    bool_cols = ["known_correct", "zero_correct", "joint_correct", "best_single_tau_correct"]
    for col in bool_cols:
        if col in subset.columns:
            subset[col] = _bool_series(subset[col])
    numeric_cols = [
        "joint_true_margin",
        "joint_true_score",
        "joint_minus_zero_true_score",
        "N_eff_true_image_fraction",
        "nearest_tau_rank",
        "nearest_tau_distance",
        "zero_true_margin",
        "static_response_distance_to_nearest_distractor",
        "structure_distance_to_nearest_distractor",
        "contrast_distance_to_nearest_distractor",
    ]
    for col in numeric_cols:
        if col in subset.columns:
            subset[col] = _num(subset[col])

    index_cols = [col for col in PAIR_KEYS if col in subset.columns]
    value_cols = [col for col in bool_cols + numeric_cols if col in subset.columns]
    wide = subset[index_cols + ["prior_family", "observation_source_row"] + value_cols].pivot_table(
        index=index_cols,
        columns="prior_family",
        values=["observation_source_row", *value_cols],
        aggfunc="first",
    )
    if not set(AXIS_FAMILIES).issubset(set(wide.columns.get_level_values(1))):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for idx, row in wide.iterrows():
        payload = {col: value for col, value in zip(index_cols, idx, strict=True)}
        payload["observation_source_row"] = int(row.get(("observation_source_row", AXIS_PARALLEL), -1))
        for metric in value_cols:
            par = _safe_float(row.get((metric, AXIS_PARALLEL), float("nan")))
            orth = _safe_float(row.get((metric, AXIS_ORTHOGONAL), float("nan")))
            payload[f"{metric}_parallel"] = par
            payload[f"{metric}_orthogonal"] = orth
            payload[f"{metric}_parallel_minus_orthogonal"] = par - orth
        par_correct = bool(payload.get("joint_correct_parallel", 0.0) >= 0.5)
        orth_correct = bool(payload.get("joint_correct_orthogonal", 0.0) >= 0.5)
        payload["axis_correct_case"] = (
            "parallel_only"
            if par_correct and not orth_correct
            else "orthogonal_only"
            if orth_correct and not par_correct
            else "both_correct"
            if par_correct and orth_correct
            else "both_wrong"
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def _join_features(paired: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    out = paired.copy()
    windows = _read_optional_csv(run_dir / "selected_windows.csv")
    candidates = _read_optional_csv(run_dir / "candidate_sets.csv")

    if not windows.empty and "source_row" in windows.columns:
        feature_cols = ["source_row"] + [col for col, _label in FEATURES if col in windows.columns]
        out = out.merge(
            windows[feature_cols].drop_duplicates("source_row"),
            left_on="observation_source_row",
            right_on="source_row",
            how="left",
        )
    if not candidates.empty and "trial_id" in candidates.columns:
        candidate_cols = [
            "trial_id",
            "static_response_distance_to_nearest_distractor",
            "structure_distance_to_nearest_distractor",
            "contrast_distance_to_nearest_distractor",
            "candidate_duplicate_flag",
            "near_duplicate_flag",
            "n_matched_distractors",
            "n_random_fallback_distractors",
            "random_fallback_used",
        ]
        candidate_cols = [col for col in candidate_cols if col in candidates.columns]
        out = out.merge(candidates[candidate_cols].drop_duplicates("trial_id"), on="trial_id", how="left", suffixes=("", "_candidate"))
        for col in [
            "static_response_distance_to_nearest_distractor",
            "structure_distance_to_nearest_distractor",
            "contrast_distance_to_nearest_distractor",
        ]:
            alt = f"{col}_candidate"
            if alt in out.columns:
                out[col] = _num(out.get(col, pd.Series(np.nan, index=out.index))).fillna(_num(out[alt]))
    return out


def _pair_summary(paired: pd.DataFrame, manifest_summary: pd.DataFrame, paired_delta_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, grp in paired.groupby(["candidate_set_mode", "prior_scale", "likelihood_scale"], dropna=False):
        candidate_set_mode, prior_scale, likelihood_scale = key
        row = {
            "candidate_set_mode": candidate_set_mode,
            "prior_scale": float(prior_scale),
            "likelihood_scale": float(likelihood_scale),
            "n_trials": int(len(grp)),
            "parallel_correct": int((grp["joint_correct_parallel"] >= 0.5).sum()),
            "orthogonal_correct": int((grp["joint_correct_orthogonal"] >= 0.5).sum()),
            "parallel_accuracy": float((grp["joint_correct_parallel"] >= 0.5).mean()),
            "orthogonal_accuracy": float((grp["joint_correct_orthogonal"] >= 0.5).mean()),
            "parallel_minus_orthogonal_accuracy": float(
                (grp["joint_correct_parallel"] >= 0.5).mean() - (grp["joint_correct_orthogonal"] >= 0.5).mean()
            ),
            "parallel_only_correct": int((grp["axis_correct_case"] == "parallel_only").sum()),
            "orthogonal_only_correct": int((grp["axis_correct_case"] == "orthogonal_only").sum()),
            "both_correct": int((grp["axis_correct_case"] == "both_correct").sum()),
            "both_wrong": int((grp["axis_correct_case"] == "both_wrong").sum()),
            "median_margin_delta_parallel_minus_orthogonal": float(grp["joint_true_margin_parallel_minus_orthogonal"].median()),
            "mean_margin_delta_parallel_minus_orthogonal": float(grp["joint_true_margin_parallel_minus_orthogonal"].mean()),
            "median_true_score_delta_parallel_minus_orthogonal": float(grp["joint_minus_zero_true_score_parallel_minus_orthogonal"].median()),
            "median_N_eff_delta_parallel_minus_orthogonal": float(grp["N_eff_true_image_fraction_parallel_minus_orthogonal"].median()),
            "median_nearest_tau_distance_delta_parallel_minus_orthogonal": float(
                grp["nearest_tau_distance_parallel_minus_orthogonal"].median()
            ),
        }
        if not manifest_summary.empty and "axis_shared_source_catalog_fraction" in manifest_summary.columns:
            row["axis_shared_source_catalog_fraction_min"] = float(manifest_summary["axis_shared_source_catalog_fraction"].min())
        if not paired_delta_summary.empty and "n_paired_rows" in paired_delta_summary.columns:
            row["axis_motion_paired_rows"] = int(paired_delta_summary["n_paired_rows"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _case_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_cols = [col for col, _label in FEATURES if col in joined.columns]
    for case, grp in joined.groupby("axis_correct_case", dropna=False):
        row: dict[str, Any] = {
            "axis_correct_case": case,
            "n_trials": int(len(grp)),
            "median_margin_delta_parallel_minus_orthogonal": float(grp["joint_true_margin_parallel_minus_orthogonal"].median()),
            "median_true_score_delta_parallel_minus_orthogonal": float(
                grp["joint_minus_zero_true_score_parallel_minus_orthogonal"].median()
            ),
            "median_N_eff_delta_parallel_minus_orthogonal": float(
                grp["N_eff_true_image_fraction_parallel_minus_orthogonal"].median()
            ),
        }
        for col in feature_cols:
            row[f"median_{col}"] = float(_num(grp[col]).median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("axis_correct_case")


def _feature_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    targets = [
        ("joint_correct_parallel_minus_orthogonal", "correct_delta"),
        ("joint_true_margin_parallel_minus_orthogonal", "margin_delta"),
        ("joint_minus_zero_true_score_parallel_minus_orthogonal", "true_score_delta"),
        ("N_eff_true_image_fraction_parallel_minus_orthogonal", "N_eff_delta"),
        ("nearest_tau_distance_parallel_minus_orthogonal", "nearest_tau_distance_delta"),
    ]
    rows: list[dict[str, Any]] = []
    for feature_col, feature_label in FEATURES:
        if feature_col not in joined.columns:
            continue
        for target_col, target_label in targets:
            if target_col not in joined.columns:
                continue
            frame = pd.DataFrame({"feature": _num(joined[feature_col]), "target": _num(joined[target_col])}).dropna()
            rows.append(
                {
                    "feature": feature_label,
                    "feature_column": feature_col,
                    "target": target_label,
                    "target_column": target_col,
                    "n_finite": int(len(frame)),
                    "spearman": _spearman(joined[feature_col], joined[target_col]),
                    "pearson": _pearson(joined[feature_col], joined[target_col]),
                }
            )
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def _feature_bin_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_col, feature_label in FEATURES:
        if feature_col not in joined.columns:
            continue
        values = _num(joined[feature_col])
        if values.notna().sum() < 8 or values.nunique(dropna=True) < 4:
            continue
        try:
            bins = pd.qcut(values, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        except ValueError:
            continue
        tmp = joined.copy()
        tmp["_bin"] = bins
        tmp["_feature_value"] = values
        for bin_name, grp in tmp.groupby("_bin", observed=True):
            rows.append(
                {
                    "feature": feature_label,
                    "feature_column": feature_col,
                    "feature_bin": str(bin_name),
                    "n_trials": int(len(grp)),
                    "feature_min": float(grp["_feature_value"].min()),
                    "feature_median": float(grp["_feature_value"].median()),
                    "feature_max": float(grp["_feature_value"].max()),
                    "parallel_accuracy": float((grp["joint_correct_parallel"] >= 0.5).mean()),
                    "orthogonal_accuracy": float((grp["joint_correct_orthogonal"] >= 0.5).mean()),
                    "parallel_minus_orthogonal_accuracy": float(
                        (grp["joint_correct_parallel"] >= 0.5).mean()
                        - (grp["joint_correct_orthogonal"] >= 0.5).mean()
                    ),
                    "median_margin_delta_parallel_minus_orthogonal": float(
                        grp["joint_true_margin_parallel_minus_orthogonal"].median()
                    ),
                    "median_true_score_delta_parallel_minus_orthogonal": float(
                        grp["joint_minus_zero_true_score_parallel_minus_orthogonal"].median()
                    ),
                    "median_N_eff_delta_parallel_minus_orthogonal": float(
                        grp["N_eff_true_image_fraction_parallel_minus_orthogonal"].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    run_dir: Path,
    out_dir: Path,
    observer_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    case_summary: pd.DataFrame,
    corr: pd.DataFrame,
) -> None:
    par = observer_summary[observer_summary["prior_family"].eq(AXIS_PARALLEL)].iloc[0]
    orth = observer_summary[observer_summary["prior_family"].eq(AXIS_ORTHOGONAL)].iloc[0]
    pair = pair_summary.iloc[0]
    corr_focus = corr[corr["target"].isin(["margin_delta", "true_score_delta"])].head(8)
    lines = [
        "# Axis-Conditioned Post-Hoc Analysis",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Observer Readout",
        "",
        "```text",
        (
            f"known-eye = {par['known_eye_accuracy']:.3f}; "
            f"zero-eye = {par['zero_eye_accuracy']:.3f}"
        ),
        (
            f"parallel joint = {par['joint_eye_accuracy']:.3f} "
            f"({int(pair['parallel_correct'])}/{int(pair['n_trials'])})"
        ),
        (
            f"orthogonal joint = {orth['joint_eye_accuracy']:.3f} "
            f"({int(pair['orthogonal_correct'])}/{int(pair['n_trials'])})"
        ),
        f"parallel - orthogonal accuracy = {pair['parallel_minus_orthogonal_accuracy']:+.3f}",
        "```",
        "",
        "## Paired Trial Cases",
        "",
        "```text",
        (
            f"parallel-only correct = {int(pair['parallel_only_correct'])}; "
            f"orthogonal-only correct = {int(pair['orthogonal_only_correct'])}; "
            f"both correct = {int(pair['both_correct'])}; both wrong = {int(pair['both_wrong'])}"
        ),
        f"median margin delta parallel-orthogonal = {pair['median_margin_delta_parallel_minus_orthogonal']:+.4f}",
        f"median true-score delta parallel-orthogonal = {pair['median_true_score_delta_parallel_minus_orthogonal']:+.4f}",
        f"median N_eff/K delta parallel-orthogonal = {pair['median_N_eff_delta_parallel_minus_orthogonal']:+.4f}",
        "```",
        "",
        "## Catalog Audit",
        "",
        "```text",
        f"axis_shared_source_catalog_fraction_min = {pair.get('axis_shared_source_catalog_fraction_min', float('nan')):.3f}",
        f"axis_motion_paired_rows = {int(pair.get('axis_motion_paired_rows', 0))}",
        "```",
        "",
        "## Case Summary",
        "",
        "```text",
        case_summary[
            [
                "axis_correct_case",
                "n_trials",
                "median_margin_delta_parallel_minus_orthogonal",
                "median_true_score_delta_parallel_minus_orthogonal",
                "median_N_eff_delta_parallel_minus_orthogonal",
            ]
        ].to_csv(index=False).strip(),
        "```",
        "",
        "## Top Exploratory Correlations",
        "",
    ]
    if corr_focus.empty:
        lines.append("- No finite feature correlations were available.")
    else:
        for _, row in corr_focus.iterrows():
            lines.append(
                "- `{feature}` vs `{target}`: Spearman `{spearman:.3f}`, Pearson `{pearson:.3f}` (`n={n}`)".format(
                    feature=row["feature"],
                    target=row["target"],
                    spearman=float(row["spearman"]) if np.isfinite(row["spearman"]) else float("nan"),
                    pearson=float(row["pearson"]) if np.isfinite(row["pearson"]) else float("nan"),
                    n=int(row["n_finite"]),
                )
            )
    acc_delta = float(pair["parallel_minus_orthogonal_accuracy"])
    score_delta = float(pair["median_true_score_delta_parallel_minus_orthogonal"])
    if acc_delta > 0:
        accuracy_read = "accuracy favors edge-parallel"
    elif acc_delta < 0:
        accuracy_read = "accuracy favors edge-orthogonal"
    else:
        accuracy_read = "accuracy is tied across edge-parallel and edge-orthogonal"
    if score_delta > 0:
        score_read = "median true-score delta favors edge-parallel"
    elif score_delta < 0:
        score_read = "median true-score delta favors edge-orthogonal"
    else:
        score_read = "median true-score delta is tied across axes"

    lines.extend(
        [
            "",
            "Interpretation: this is an exploratory posthoc on a clean shared-source",
            f"axis comparison. In this run, {accuracy_read}, while {score_read}.",
            "Treat this as a mixed pilot until the axis effect replicates across",
            "candidate modes, seeds, and larger sample sizes.",
            "",
            "## Outputs",
            "",
            "- `axis_observer_summary.csv`",
            "- `axis_pair_summary.csv`",
            "- `axis_paired_trial_feature_table.csv`",
            "- `axis_case_summary.csv`",
            "- `axis_feature_correlation_summary.csv`",
            "- `axis_feature_bin_summary.csv`",
        ]
    )
    (out_dir / "axis_posthoc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_run(run_dir: Path, out_dir: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    out_dir = run_dir / "axis_conditioned_posthoc" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(run_dir / "observer_trials.csv")
    manifest_summary = _read_optional_csv(run_dir / "axis_conditioned_audit" / "axis_manifest_summary.csv")
    paired_delta_summary = _read_optional_csv(run_dir / "axis_conditioned_audit" / "axis_family_paired_delta_summary.csv")

    observer_summary = _observer_summary(trials)
    paired = _paired_trial_table(trials)
    joined = _join_features(paired, run_dir)
    pair_summary = _pair_summary(joined, manifest_summary, paired_delta_summary)
    cases = _case_summary(joined)
    corr = _feature_correlations(joined)
    bins = _feature_bin_summary(joined)

    observer_summary.to_csv(out_dir / "axis_observer_summary.csv", index=False)
    pair_summary.to_csv(out_dir / "axis_pair_summary.csv", index=False)
    joined.to_csv(out_dir / "axis_paired_trial_feature_table.csv", index=False)
    cases.to_csv(out_dir / "axis_case_summary.csv", index=False)
    corr.to_csv(out_dir / "axis_feature_correlation_summary.csv", index=False)
    bins.to_csv(out_dir / "axis_feature_bin_summary.csv", index=False)
    _write_report(run_dir, out_dir, observer_summary, pair_summary, cases, corr)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = analyze_run(run_dir=args.run_dir, out_dir=args.output_dir)
    print(f"Wrote axis-conditioned posthoc analysis to {out}")


if __name__ == "__main__":
    main()
