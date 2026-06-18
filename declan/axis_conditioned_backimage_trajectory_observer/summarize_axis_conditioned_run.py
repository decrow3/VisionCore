"""Posthoc validation and paired statistics for axis-conditioned BackImage runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BOOL_COLUMNS = [
    "known_correct",
    "zero_correct",
    "joint_correct",
    "best_single_tau_correct",
    "best_trajectory_oracle_correct",
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: Any, digits: int = 4) -> str:
    val = _safe_float(value)
    if not np.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def _parse_source_rows(text: Any) -> set[int]:
    if not isinstance(text, str) or not text:
        return set()
    rows: set[int] = set()
    for piece in text.replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith("source_row:"):
            piece = piece.split(":", 1)[1]
        try:
            rows.add(int(float(piece)))
        except ValueError:
            continue
    return rows


def _exact_binom_sf(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0**n)


def _exact_binom_cdf(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    return sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)


def _mcnemar_p_two_sided(a_better: int, b_better: int) -> float:
    n = int(a_better + b_better)
    if n <= 0:
        return float("nan")
    lo = min(int(a_better), int(b_better))
    return min(1.0, 2.0 * _exact_binom_cdf(lo, n))


def _paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    reducer: str = "mean",
) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    diff = a[mask] - b[mask]
    if diff.size == 0:
        return {
            "n_pairs": 0,
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_p_delta_le_0": float("nan"),
            "bootstrap_p_delta_ge_0": float("nan"),
        }
    stat = np.mean if reducer == "mean" else np.median
    observed = float(stat(diff))
    if diff.size <= 1 or int(n_bootstrap) <= 0:
        return {
            "n_pairs": int(diff.size),
            "delta": observed,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_p_delta_le_0": float("nan"),
            "bootstrap_p_delta_ge_0": float("nan"),
        }
    draws = diff[rng.integers(0, diff.size, size=(int(n_bootstrap), diff.size))]
    boot = stat(draws, axis=1)
    return {
        "n_pairs": int(diff.size),
        "delta": observed,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "bootstrap_p_delta_le_0": float(np.mean(boot <= 0.0)),
        "bootstrap_p_delta_ge_0": float(np.mean(boot >= 0.0)),
    }


def _paired_accuracy_rows(
    trials: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    subset_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_pair(label: str, a_label: str, b_label: str, a: pd.Series, b: pd.Series) -> None:
        a_arr = a.astype(bool).to_numpy()
        b_arr = b.astype(bool).to_numpy()
        a_better = int(np.sum(a_arr & ~b_arr))
        b_better = int(np.sum(~a_arr & b_arr))
        both_correct = int(np.sum(a_arr & b_arr))
        both_wrong = int(np.sum(~a_arr & ~b_arr))
        boot = _paired_bootstrap_diff(
            a_arr.astype(float),
            b_arr.astype(float),
            rng=rng,
            n_bootstrap=n_bootstrap,
            reducer="mean",
        )
        rows.append(
            {
                "comparison": label,
                "analysis_subset": subset_label,
                "a": a_label,
                "b": b_label,
                "n_pairs": int(a_arr.size),
                "a_accuracy": float(np.mean(a_arr)),
                "b_accuracy": float(np.mean(b_arr)),
                "accuracy_delta_a_minus_b": float(np.mean(a_arr) - np.mean(b_arr)),
                "bootstrap_ci_low": boot["ci_low"],
                "bootstrap_ci_high": boot["ci_high"],
                "bootstrap_p_delta_le_0": boot["bootstrap_p_delta_le_0"],
                "bootstrap_p_delta_ge_0": boot["bootstrap_p_delta_ge_0"],
                "a_better_b_wrong": a_better,
                "b_better_a_wrong": b_better,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "mcnemar_exact_p_two_sided": _mcnemar_p_two_sided(a_better, b_better),
                "binomial_exact_p_a_greater_b": _exact_binom_sf(a_better, a_better + b_better)
                if (a_better + b_better) > 0
                else float("nan"),
            }
        )

    for family, grp in trials.groupby("prior_family", sort=True):
        grp = grp.sort_values("trial_id")
        add_pair(
            f"{family}: joint vs zero",
            f"{family}:joint",
            "zero",
            grp["joint_correct"],
            grp["zero_correct"],
        )
        add_pair(
            f"{family}: joint vs best_single_tau",
            f"{family}:joint",
            f"{family}:best_single_tau",
            grp["joint_correct"],
            grp["best_single_tau_correct"],
        )

    pivot = trials.pivot(index="trial_id", columns="prior_family", values="joint_correct")
    if {"axis_edge_orthogonal", "axis_edge_parallel"}.issubset(set(pivot.columns)):
        pivot = pivot.sort_index()
        add_pair(
            "axis_edge_orthogonal joint vs axis_edge_parallel joint",
            "axis_edge_orthogonal:joint",
            "axis_edge_parallel:joint",
            pivot["axis_edge_orthogonal"],
            pivot["axis_edge_parallel"],
        )

    return rows


def _continuous_delta_rows(
    trials: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    metrics = [
        "joint_true_margin",
        "joint_true_score",
        "N_eff_true_image_fraction",
        "joint_vs_best_single_tau_gap",
        "joint_vs_best_dilution_gap",
        "best_oracle_minus_joint_true_score",
        "nearest_tau_distance",
    ]
    rows: list[dict[str, Any]] = []
    families = set(trials["prior_family"].astype(str))
    if not {"axis_edge_orthogonal", "axis_edge_parallel"}.issubset(families):
        return rows
    wide = trials.pivot(index="trial_id", columns="prior_family", values=metrics)
    for metric in metrics:
        if (metric, "axis_edge_orthogonal") not in wide.columns:
            continue
        a = wide[(metric, "axis_edge_orthogonal")]
        b = wide[(metric, "axis_edge_parallel")]
        mean_boot = _paired_bootstrap_diff(a.to_numpy(), b.to_numpy(), rng=rng, n_bootstrap=n_bootstrap, reducer="mean")
        median_boot = _paired_bootstrap_diff(a.to_numpy(), b.to_numpy(), rng=rng, n_bootstrap=n_bootstrap, reducer="median")
        diff = a.to_numpy(dtype=float) - b.to_numpy(dtype=float)
        rows.append(
            {
                "comparison": "axis_edge_orthogonal minus axis_edge_parallel",
                "metric": metric,
                "n_pairs": mean_boot["n_pairs"],
                "orthogonal_mean": float(np.nanmean(a.to_numpy(dtype=float))),
                "parallel_mean": float(np.nanmean(b.to_numpy(dtype=float))),
                "mean_delta": mean_boot["delta"],
                "mean_delta_ci_low": mean_boot["ci_low"],
                "mean_delta_ci_high": mean_boot["ci_high"],
                "mean_bootstrap_p_delta_le_0": mean_boot["bootstrap_p_delta_le_0"],
                "orthogonal_median": float(np.nanmedian(a.to_numpy(dtype=float))),
                "parallel_median": float(np.nanmedian(b.to_numpy(dtype=float))),
                "median_delta": median_boot["delta"],
                "median_delta_ci_low": median_boot["ci_low"],
                "median_delta_ci_high": median_boot["ci_high"],
                "median_bootstrap_p_delta_le_0": median_boot["bootstrap_p_delta_le_0"],
                "sign_positive_count": int(np.sum(diff > 0)),
                "sign_negative_count": int(np.sum(diff < 0)),
                "sign_tie_count": int(np.sum(diff == 0)),
            }
        )
    return rows


def _summary_rows(trials: pd.DataFrame, summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in summary.sort_values("prior_family").iterrows():
        rows.append(
            {
                "prior_family": row["prior_family"],
                "n_trials": int(row["n_trials"]),
                "known_eye_accuracy": row["known_eye_accuracy"],
                "zero_eye_accuracy": row["zero_eye_accuracy"],
                "joint_eye_accuracy": row["joint_eye_accuracy"],
                "best_single_tau_accuracy": row["best_single_tau_accuracy"],
                "best_trajectory_oracle_accuracy": row["best_trajectory_oracle_accuracy"],
                "joint_minus_zero_accuracy": row["joint_minus_zero_accuracy"],
                "best_single_tau_minus_joint_accuracy": row["best_single_tau_minus_joint_accuracy"],
                "median_N_eff_fraction": row["median_N_eff_fraction"],
                "median_joint_true_margin": row["median_joint_true_margin"],
                "median_zero_true_margin": row["median_zero_true_margin"],
            }
        )
    return rows


def _catalog_qc_rows(
    *,
    out_dir: Path,
    trials: pd.DataFrame,
    manifest: pd.DataFrame,
    axis: pd.DataFrame,
    motion: pd.DataFrame,
    candidates: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check: str, value: Any, expected: Any = "", status: str = "info", detail: str = "") -> None:
        rows.append({"check": check, "value": value, "expected": expected, "status": status, "detail": detail})

    response_files = list((out_dir / "response_tables").glob("*.npz"))
    add("observer_trial_rows", len(trials), 256, "pass" if len(trials) == 256 else "fail")
    add("response_cache_files", len(response_files), len(manifest), "pass" if len(response_files) == len(manifest) else "fail")
    add("manifest_rows", len(manifest), 256, "pass" if len(manifest) == 256 else "fail")
    add("axis_catalog_rows", len(axis), 32768, "pass" if len(axis) == 32768 else "fail")
    add("motion_catalog_rows", len(motion), 32896, "pass" if len(motion) == 32896 else "fail")
    add("candidate_set_rows", len(candidates), 128, "pass" if len(candidates) == 128 else "fail")

    for col, expected in [
        ("axis_catalog_mode", {"per_candidate"}),
        ("trajectory_prior_mode", {"leave_one_out"}),
    ]:
        observed = set(trials[col].dropna().astype(str))
        add(col, ",".join(sorted(observed)), ",".join(sorted(expected)), "pass" if observed == expected else "fail")

    add("dry_run_false", int((manifest["dry_run"] == False).sum()), len(manifest), "pass" if not manifest["dry_run"].astype(bool).any() else "fail")
    add("n_candidates_all_4", int((manifest["n_candidates"] == 4).sum()), len(manifest), "pass" if (manifest["n_candidates"] == 4).all() else "fail")
    add(
        "n_prior_trajectories_all_32",
        int((manifest["n_prior_trajectories"] == 32).sum()),
        len(manifest),
        "pass" if (manifest["n_prior_trajectories"] == 32).all() else "fail",
    )
    add("n_timebins_all_40", int((manifest["n_timebins"] == 40).sum()), len(manifest), "pass" if (manifest["n_timebins"] == 40).all() else "fail")
    add("n_units_all_756", int((manifest["n_units"] == 756).sum()), len(manifest), "pass" if (manifest["n_units"] == 756).all() else "fail")

    add(
        "excluded_current_source_row_all_true",
        int((manifest["excluded_current_source_row"] == 1).sum()),
        len(manifest),
        "pass" if (manifest["excluded_current_source_row"] == 1).all() else "fail",
    )
    add(
        "excluded_candidate_source_row_count_all_4",
        int((manifest["excluded_candidate_source_row_count"] == 4).sum()),
        len(manifest),
        "pass" if (manifest["excluded_candidate_source_row_count"] == 4).all() else "fail",
    )
    add(
        "excluded_candidate_source_row_hits_all_4",
        int((manifest["excluded_candidate_source_row_hits"] == 4).sum()),
        len(manifest),
        "pass" if (manifest["excluded_candidate_source_row_hits"] == 4).all() else "fail",
    )
    add(
        "prior_duplicate_trajectory_count_sum",
        int(manifest["prior_duplicate_trajectory_count"].sum()),
        0,
        "pass" if int(manifest["prior_duplicate_trajectory_count"].sum()) == 0 else "fail",
    )
    add(
        "excluded_exact_trace_hash_sum",
        int(manifest["excluded_exact_trace_hash"].sum()),
        0,
        "pass" if int(manifest["excluded_exact_trace_hash"].sum()) == 0 else "warn",
        "Nonzero would be acceptable if it means duplicates were rejected before sampling.",
    )
    add(
        "excluded_near_duplicate_rmse_sum",
        int(manifest["excluded_near_duplicate_rmse"].sum()),
        0,
        "pass" if int(manifest["excluded_near_duplicate_rmse"].sum()) == 0 else "warn",
        "Nonzero would be acceptable if it means near-duplicates were rejected before sampling.",
    )

    candidate_source_by_trial = {
        int(row.trial_id): _parse_source_rows(row.candidate_ids)
        for row in candidates[["trial_id", "candidate_ids"]].itertuples(index=False)
    }
    source_in_candidate_set = 0
    for row in axis[["trial_id", "source_row"]].itertuples(index=False):
        try:
            source = int(row.source_row)
        except (TypeError, ValueError):
            continue
        if source in candidate_source_by_trial.get(int(row.trial_id), set()):
            source_in_candidate_set += 1
    add("axis_prior_source_rows_in_candidate_set", source_in_candidate_set, 0, "pass" if source_in_candidate_set == 0 else "fail")

    source_equals_candidate = int((axis["source_row"].astype(int) == axis["axis_candidate_source_row"].astype(int)).sum())
    add("axis_prior_source_equals_render_candidate_source", source_equals_candidate, 0, "pass" if source_equals_candidate == 0 else "fail")

    group_cols = ["trial_id", "family", "candidate_index"]
    for key in ["source_row", "trace_hash", "axis_pair_id", "trajectory_identity_id"]:
        dup_count = int(axis.duplicated(group_cols + [key]).sum())
        add(f"within_candidate_duplicate_{key}_rows", dup_count, 0, "pass" if dup_count == 0 else "fail")

    add(
        "axis_match_status_counts",
        axis["axis_match_status"].value_counts(dropna=False).to_dict(),
        "{'matched': 32768}",
        "pass" if set(axis["axis_match_status"].dropna().astype(str)) == {"matched"} else "fail",
    )
    add(
        "axis_match_degenerate_count",
        int(axis["axis_match_degenerate"].astype(bool).sum()),
        0,
        "pass" if int(axis["axis_match_degenerate"].astype(bool).sum()) == 0 else "fail",
    )
    add(
        "degenerate_requested_motion_count",
        int(axis["degenerate_requested_motion"].astype(bool).sum()),
        0,
        "pass" if int(axis["degenerate_requested_motion"].astype(bool).sum()) == 0 else "fail",
    )
    add(
        "rms_clipped_high_count",
        int(axis["rms_clipped_high"].astype(bool).sum()),
        0,
        "pass" if int(axis["rms_clipped_high"].astype(bool).sum()) == 0 else "warn",
    )
    for col in [
        "axis_match_rms_delta_deg",
        "axis_match_path_delta_deg",
        "axis_match_duration_delta_s",
        "axis_match_clipping_fraction_delta",
    ]:
        val = float(np.nanmax(np.abs(axis[col].to_numpy(dtype=float))))
        add(f"max_abs_{col}", val, "<=1e-6", "pass" if val <= 1e-6 else "warn")

    add(
        "candidate_duplicate_flag_count",
        int(candidates["candidate_duplicate_flag"].astype(bool).sum()),
        0,
        "pass" if not candidates["candidate_duplicate_flag"].astype(bool).any() else "fail",
    )
    add(
        "near_duplicate_flag_count",
        int(candidates["near_duplicate_flag"].astype(bool).sum()),
        0,
        "pass" if not candidates["near_duplicate_flag"].astype(bool).any() else "warn",
        "Candidate-set structure/contrast duplicate flag, not trajectory prior leakage.",
    )
    add(
        "random_fallback_used_count",
        int(candidates["random_fallback_used"].astype(bool).sum()),
        0,
        "pass" if not candidates["random_fallback_used"].astype(bool).any() else "warn",
    )
    add(
        "n_matched_distractors_all_3",
        int((candidates["n_matched_distractors"] == 3).sum()),
        len(candidates),
        "pass" if (candidates["n_matched_distractors"] == 3).all() else "warn",
    )

    return rows


def _validate_response_caches(out_dir: Path, manifest: pd.DataFrame, mode: str) -> list[dict[str, Any]]:
    if mode == "none":
        return []
    if mode == "sample":
        manifest = manifest.head(8).copy()
    rows: list[dict[str, Any]] = []
    checked = 0
    missing = 0
    shape_errors = 0
    nonfinite_files = 0
    negative_files = 0
    min_value = float("inf")
    max_value = float("-inf")
    for _, row in manifest.iterrows():
        checked += 1
        if checked == 1 or checked % 32 == 0:
            print(f"[axis-posthoc] validating response cache {checked}/{len(manifest)}", flush=True)
        path = out_dir / str(row["response_cache_path"])
        if not path.exists():
            missing += 1
            continue
        expected = {
            "prior_lambda_counts": (
                int(row["n_candidates"]),
                int(row["n_prior_trajectories"]),
                int(row["n_timebins"]),
                int(row["n_units"]),
            ),
            "known_lambda_counts": (int(row["n_candidates"]), int(row["n_timebins"]), int(row["n_units"])),
            "zero_lambda_counts": (int(row["n_candidates"]), int(row["n_timebins"]), int(row["n_units"])),
            "y_obs_counts": (int(row["n_timebins"]), int(row["n_units"])),
        }
        file_nonfinite = False
        file_negative = False
        with np.load(path) as z:
            for key, shape in expected.items():
                arr = z[key]
                if tuple(arr.shape) != tuple(shape):
                    shape_errors += 1
                if not np.isfinite(arr).all():
                    file_nonfinite = True
                arr_min = float(np.nanmin(arr))
                arr_max = float(np.nanmax(arr))
                min_value = min(min_value, arr_min)
                max_value = max(max_value, arr_max)
                if arr_min < 0.0:
                    file_negative = True
        nonfinite_files += int(file_nonfinite)
        negative_files += int(file_negative)
    status = "pass" if missing == 0 and shape_errors == 0 and nonfinite_files == 0 and negative_files == 0 else "fail"
    rows.append(
        {
            "check": f"response_cache_validation_{mode}",
            "value": {
                "checked": checked,
                "missing": missing,
                "shape_errors": shape_errors,
                "nonfinite_files": nonfinite_files,
                "negative_files": negative_files,
                "min_numeric_value": min_value if np.isfinite(min_value) else float("nan"),
                "max_numeric_value": max_value if np.isfinite(max_value) else float("nan"),
            },
            "expected": "all shapes match manifest; all numeric arrays finite and nonnegative",
            "status": status,
            "detail": "",
        }
    )
    return rows


def _write_report(
    *,
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    accuracy_rows: list[dict[str, Any]],
    continuous_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    n_bootstrap: int,
    cache_validation: str,
) -> None:
    summary_by_family = {str(row["prior_family"]): row for row in summary_rows}
    orth = summary_by_family.get("axis_edge_orthogonal", {})
    par = summary_by_family.get("axis_edge_parallel", {})
    acc_by_name = {(str(row.get("analysis_subset", "all_trials")), str(row["comparison"])): row for row in accuracy_rows}
    orth_vs_par = acc_by_name.get(("all_trials", "axis_edge_orthogonal joint vs axis_edge_parallel joint"), {})
    orth_vs_zero = acc_by_name.get(("all_trials", "axis_edge_orthogonal: joint vs zero"), {})
    par_vs_zero = acc_by_name.get(("all_trials", "axis_edge_parallel: joint vs zero"), {})
    sensitivity_orth_vs_par = acc_by_name.get(
        (
            "excluding_candidate_near_duplicate_flag",
            "axis_edge_orthogonal joint vs axis_edge_parallel joint",
        ),
        {},
    )
    qc_status = pd.DataFrame(qc_rows)["status"].value_counts().to_dict() if qc_rows else {}
    failing = [row for row in qc_rows if row.get("status") == "fail"]
    warnings = [row for row in qc_rows if row.get("status") == "warn"]

    lines = [
        "# Axis-Conditioned BackImage Posthoc Validation",
        "",
        f"- Output directory: `{out_dir}`",
        f"- Bootstrap: paired trial/window resampling, `n={n_bootstrap}`.",
        f"- Response-cache validation mode: `{cache_validation}`.",
        "",
        "## Headline Accuracy",
        "",
        "| prior family | joint | zero | joint-zero | best single tau | best tau minus joint | median N_eff frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['prior_family']}` | {_fmt(row['joint_eye_accuracy'])} | {_fmt(row['zero_eye_accuracy'])} | "
            f"{_fmt(row['joint_minus_zero_accuracy'])} | {_fmt(row['best_single_tau_accuracy'])} | "
            f"{_fmt(row['best_single_tau_minus_joint_accuracy'])} | {_fmt(row['median_N_eff_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired Tests",
            "",
            "| subset | comparison | delta | 95% bootstrap CI | discordance A>B / B>A | exact McNemar p | one-sided p(A>B) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in accuracy_rows:
        lines.append(
            f"| `{row.get('analysis_subset', 'all_trials')}` | {row['comparison']} | {_fmt(row['accuracy_delta_a_minus_b'])} | "
            f"[{_fmt(row['bootstrap_ci_low'])}, {_fmt(row['bootstrap_ci_high'])}] | "
            f"{row['a_better_b_wrong']} / {row['b_better_a_wrong']} | "
            f"{_fmt(row['mcnemar_exact_p_two_sided'])} | {_fmt(row['binomial_exact_p_a_greater_b'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- Orthogonal joint accuracy is `{_fmt(orth.get('joint_eye_accuracy'))}` versus "
                f"parallel `{_fmt(par.get('joint_eye_accuracy'))}`; the paired orthogonal-minus-parallel "
                f"delta is `{_fmt(orth_vs_par.get('accuracy_delta_a_minus_b'))}` "
                f"with 95% bootstrap CI `[{_fmt(orth_vs_par.get('bootstrap_ci_low'))}, "
                f"{_fmt(orth_vs_par.get('bootstrap_ci_high'))}]`."
            ),
            (
                f"- Orthogonal beats the zero-eye baseline by `{_fmt(orth_vs_zero.get('accuracy_delta_a_minus_b'))}` "
                f"(exact McNemar p `{_fmt(orth_vs_zero.get('mcnemar_exact_p_two_sided'))}`); "
                f"parallel beats zero by `{_fmt(par_vs_zero.get('accuracy_delta_a_minus_b'))}` "
                f"(exact McNemar p `{_fmt(par_vs_zero.get('mcnemar_exact_p_two_sided'))}`)."
            ),
            "- These are trial/window-level paired statistics; they should be treated as run-level evidence, not session-clustered population inference.",
            (
                f"- Excluding candidate-set rows flagged as structure/contrast near-duplicates leaves the "
                f"orthogonal-minus-parallel joint delta at `{_fmt(sensitivity_orth_vs_par.get('accuracy_delta_a_minus_b'))}` "
                f"with exact McNemar p `{_fmt(sensitivity_orth_vs_par.get('mcnemar_exact_p_two_sided'))}`."
            )
            if sensitivity_orth_vs_par
            else "- No candidate-near-duplicate sensitivity subset was generated.",
            "",
            "## QC",
            "",
            f"- QC status counts: `{qc_status}`.",
        ]
    )
    if failing:
        lines.append("- Failing checks:")
        for row in failing:
            lines.append(f"  - `{row['check']}` value `{row['value']}` expected `{row['expected']}`.")
    else:
        lines.append("- No failing QC checks.")
    if warnings:
        lines.append("- Warning checks:")
        for row in warnings:
            lines.append(f"  - `{row['check']}` value `{row['value']}` expected `{row['expected']}`.")
    else:
        lines.append("- No QC warnings.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `posthoc_accuracy_tests.csv`",
            "- `posthoc_continuous_metric_deltas.csv`",
            "- `posthoc_qc_checks.csv`",
            "- `posthoc_validation_report.md`",
        ]
    )
    (out_dir / "posthoc_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache-validation", choices=["none", "sample", "all"], default="none")
    args = parser.parse_args()

    out_dir = args.out_dir
    rng = np.random.default_rng(int(args.seed))
    trials = pd.read_csv(out_dir / "observer_trials.csv")
    summary = pd.read_csv(out_dir / "observer_summary.csv")
    manifest = pd.read_csv(out_dir / "response_cache_manifest.csv")
    axis = pd.read_csv(out_dir / "axis_trajectory_catalog.csv")
    motion = pd.read_csv(out_dir / "motion_catalog.csv", low_memory=False)
    candidates = pd.read_csv(out_dir / "candidate_sets.csv")
    for col in BOOL_COLUMNS:
        if col in trials:
            trials[col] = trials[col].astype(bool)

    summary_out = _summary_rows(trials, summary)
    accuracy_rows = _paired_accuracy_rows(
        trials,
        rng=rng,
        n_bootstrap=int(args.n_bootstrap),
        subset_label="all_trials",
    )
    if "near_duplicate_flag" in trials and trials["near_duplicate_flag"].astype(bool).any():
        accuracy_rows.extend(
            _paired_accuracy_rows(
                trials[~trials["near_duplicate_flag"].astype(bool)].copy(),
                rng=rng,
                n_bootstrap=int(args.n_bootstrap),
                subset_label="excluding_candidate_near_duplicate_flag",
            )
        )
    continuous_rows = _continuous_delta_rows(trials, rng=rng, n_bootstrap=int(args.n_bootstrap))
    qc_rows = _catalog_qc_rows(
        out_dir=out_dir,
        trials=trials,
        manifest=manifest,
        axis=axis,
        motion=motion,
        candidates=candidates,
    )
    qc_rows.extend(_validate_response_caches(out_dir, manifest, str(args.cache_validation)))

    _write_csv(out_dir / "posthoc_family_summary.csv", summary_out)
    _write_csv(out_dir / "posthoc_accuracy_tests.csv", accuracy_rows)
    _write_csv(out_dir / "posthoc_continuous_metric_deltas.csv", continuous_rows)
    _write_csv(out_dir / "posthoc_qc_checks.csv", qc_rows)
    _write_report(
        out_dir=out_dir,
        summary_rows=summary_out,
        accuracy_rows=accuracy_rows,
        continuous_rows=continuous_rows,
        qc_rows=qc_rows,
        n_bootstrap=int(args.n_bootstrap),
        cache_validation=str(args.cache_validation),
    )
    print(f"[axis-posthoc] wrote posthoc validation outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
