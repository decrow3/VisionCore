#!/usr/bin/env python3
"""Adapt completed recorded pose-aware prediction runs to prescription outputs.

The completed runner ``run_recorded_pose_aware_prediction.py`` writes a
cache-first held-out Poisson response-prediction ladder.  This adapter maps
those outputs onto the ``recorded_pose_info_*`` artifact names requested by the
non-circular FEM information prescription without rerunning the decoder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from non_circular_fem_common import (
    DEFAULT_STACK_OUT_DIR,
    mean_sem,
    parse_float,
    parse_int,
    read_csv_rows,
    write_csv_rows,
    write_json,
)


DEFAULT_SOURCE_DIR = (
    DEFAULT_STACK_OUT_DIR / "recorded_pose_aware_prediction_multisession_6pilot"
)
DEFAULT_OUT_DIR = DEFAULT_STACK_OUT_DIR / "recorded_pose_information"
LN2 = float(np.log(2.0))

MODEL_ROLES = {
    "M0_psth_glm": "pose_blind_psth",
    "M_eye_only": "eye_only_control",
    "M1_additive_eye": "pose_aware_additive_eye",
    "M2_scalar_eye_gain": "pose_aware_scalar_gain",
    "M3_time_by_eye_interaction": "pose_aware_time_by_eye_interaction",
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_manifest(source_dir: Path) -> dict[str, Any]:
    path = require_file(source_dir / "recorded_pose_aware_prediction_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def bits_from_delta_ll(delta_ll: float, spikes: float) -> float:
    if not np.isfinite(delta_ll) or not np.isfinite(spikes) or spikes <= 0:
        return float("nan")
    return float(delta_ll / (spikes * LN2))


def ll_per_spike(ll_sum: float, spikes: float) -> float:
    if not np.isfinite(ll_sum) or not np.isfinite(spikes) or spikes <= 0:
        return float("nan")
    return float(ll_sum / spikes)


def session_metric_rows(source_dir: Path) -> list[dict[str, Any]]:
    observed = read_csv_rows(require_file(source_dir / "fold_model_metrics.csv"))
    nulls = read_csv_rows(source_dir / "fold_eye_shuffle_nulls.csv")
    all_rows = observed + nulls
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in all_rows:
        key = (
            str(row.get("session", "")),
            str(row.get("subject", "")),
            str(row.get("model_name", "")),
            str(row.get("null_type", "observed")),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (session, subject, model, null_type), rows in sorted(groups.items()):
        delta_ll = float(sum(parse_float(row.get("delta_ll_vs_psth")) for row in rows))
        ll_sum = float(sum(parse_float(row.get("ll_sum")) for row in rows))
        spikes = float(sum(parse_float(row.get("total_spikes")) for row in rows))
        eval_obs = int(sum(parse_int(row.get("n_eval_observations")) for row in rows))
        unit_counts = [parse_float(row.get("n_units_fit")) for row in rows]
        failures = int(sum(parse_int(row.get("n_fit_failures")) for row in rows))
        constants = int(sum(parse_int(row.get("n_constant_units")) for row in rows))
        out.append(
            {
                "session": session,
                "subject": subject,
                "model_name": model,
                "model_role": MODEL_ROLES.get(model, "unknown"),
                "null_type": null_type,
                "aggregation": "spike_weighted_sum_over_folds",
                "n_folds": len(rows),
                "n_units_fit_mean": float(np.nanmean(unit_counts)) if unit_counts else float("nan"),
                "n_fit_failures": failures,
                "n_constant_units": constants,
                "n_eval_observations": eval_obs,
                "total_spikes": spikes,
                "heldout_log_likelihood_sum": ll_sum,
                "heldout_log_likelihood_per_spike": ll_per_spike(ll_sum, spikes),
                "delta_ll_vs_pose_blind_psth": delta_ll,
                "bits_per_spike_delta_vs_pose_blind_psth": bits_from_delta_ll(delta_ll, spikes),
                "effective_alpha_mean": float(
                    np.nanmean([parse_float(row.get("effective_alpha")) for row in rows])
                ),
            }
        )
    return out


def unit_count_rows(source_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(require_file(source_dir / "session_summary.csv"))
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "session": row.get("session", ""),
                "subject": row.get("subject", ""),
                "status": row.get("status", ""),
                "n_trials": parse_int(row.get("n_trials")),
                "n_time": parse_int(row.get("n_time")),
                "n_units": parse_int(row.get("n_units")),
                "n_folds_requested": parse_int(row.get("n_folds_requested")),
                "n_folds_evaluable": parse_int(row.get("n_folds_evaluable")),
                "valid_sample_fraction": parse_float(row.get("valid_sample_fraction")),
                "dfs_gt_threshold_fraction": parse_float(row.get("dfs_gt_threshold_fraction")),
                "include_trial_drift": row.get("include_trial_drift", ""),
            }
        )
    return out


def metric_lookup(
    session_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row["session"]), str(row["model_name"]), str(row["null_type"])): row
        for row in session_rows
    }


def source_style_session_model_stats(source_dir: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[tuple[str, str]]]:
    """Reproduce the source runner's session model mean/median convention."""
    rows = read_csv_rows(require_file(source_dir / "fold_model_metrics.csv")) + read_csv_rows(
        require_file(source_dir / "fold_eye_shuffle_nulls.csv")
    )
    groups: dict[tuple[str, str, str], list[float]] = {}
    subject_by_session: dict[str, str] = {}
    for row in rows:
        if parse_float(row.get("total_spikes")) <= 0:
            continue
        session = str(row.get("session", ""))
        subject_by_session.setdefault(session, str(row.get("subject", "")))
        key = (session, str(row.get("model_name", "")), str(row.get("null_type", "")))
        groups.setdefault(key, []).append(parse_float(row.get("bits_per_spike_delta_vs_psth")))
    stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        stats[key] = {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "n": int(arr.size)}
    sessions = sorted((session, subject) for session, subject in subject_by_session.items())
    return stats, sessions


def add_contrast(
    out: list[dict[str, Any]],
    lookup: dict[tuple[str, str, str], dict[str, Any]],
    *,
    session: str,
    subject: str,
    comparison: str,
    model_a: str,
    null_a: str,
    model_b: str,
    null_b: str,
) -> None:
    a = lookup.get((session, model_a, null_a))
    b = lookup.get((session, model_b, null_b))
    if a is None or b is None:
        return
    va = parse_float(a.get("mean"))
    vb = parse_float(b.get("median" if null_b != "observed" else "mean"))
    out.append(
        {
            "scope": "session",
            "session": session,
            "subject": subject,
            "comparison": comparison,
            "metric_name": "bits_per_spike_delta_difference",
            "model_a": model_a,
            "null_a": null_a,
            "model_b": model_b,
            "null_b": null_b,
            "model_a_value": va,
            "model_b_value": vb,
            "value": va - vb,
            "aggregation": "source_runner_unweighted_fold_stats",
            "source_session_stat": "model_a_mean_minus_model_b_median_for_null_else_mean",
        }
    )


def paired_contrast_rows(source_dir: Path, session_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup, sessions = source_style_session_model_stats(source_dir)
    out: list[dict[str, Any]] = []
    for session, subject in sessions:
        add_contrast(
            out,
            lookup,
            session=session,
            subject=subject,
            comparison="M1_additive_eye_minus_M0_psth_glm",
            model_a="M1_additive_eye",
            null_a="observed",
            model_b="M0_psth_glm",
            null_b="observed",
        )
        add_contrast(
            out,
            lookup,
            session=session,
            subject=subject,
            comparison="M2_scalar_eye_gain_minus_M0_psth_glm",
            model_a="M2_scalar_eye_gain",
            null_a="observed",
            model_b="M0_psth_glm",
            null_b="observed",
        )
        add_contrast(
            out,
            lookup,
            session=session,
            subject=subject,
            comparison="M3_time_by_eye_interaction_minus_M0_psth_glm",
            model_a="M3_time_by_eye_interaction",
            null_a="observed",
            model_b="M0_psth_glm",
            null_b="observed",
        )
        add_contrast(
            out,
            lookup,
            session=session,
            subject=subject,
            comparison="M1_minus_eye_shuffle",
            model_a="M1_additive_eye",
            null_a="observed",
            model_b="M1_additive_eye",
            null_b="trial_trace_eye_shuffle",
        )
        add_contrast(
            out,
            lookup,
            session=session,
            subject=subject,
            comparison="M3_minus_eye_shuffle",
            model_a="M3_time_by_eye_interaction",
            null_a="observed",
            model_b="M3_time_by_eye_interaction",
            null_b="trial_trace_eye_shuffle",
        )
        add_contrast(
            out,
            lookup,
            session=session,
            subject=subject,
            comparison="M3_minus_M1",
            model_a="M3_time_by_eye_interaction",
            null_a="observed",
            model_b="M1_additive_eye",
            null_b="observed",
        )
        add_contrast(
            out,
            lookup,
            session=session,
            subject=subject,
            comparison="M3_minus_M2",
            model_a="M3_time_by_eye_interaction",
            null_a="observed",
            model_b="M2_scalar_eye_gain",
            null_b="observed",
        )
        add_contrast(
            out,
            lookup,
            session=session,
            subject=subject,
            comparison="M1_minus_eye_only",
            model_a="M1_additive_eye",
            null_a="observed",
            model_b="M_eye_only",
            null_b="observed",
        )

    for row in read_csv_rows(source_dir / "model_comparison_summary.csv"):
        out.append(
            {
                "scope": "aggregate_session_bootstrap",
                "session": "",
                "subject": "",
                "comparison": row.get("comparison", ""),
                "metric_name": row.get("metric_name", ""),
                "model_a": "",
                "null_a": "",
                "model_b": "",
                "null_b": "",
                "model_a_value": "",
                "model_b_value": "",
                "value": parse_float(row.get("mean")),
                "aggregation": "source_runner_session_bootstrap",
                "boot_ci_low": parse_float(row.get("boot_ci_low")),
                "boot_ci_high": parse_float(row.get("boot_ci_high")),
                "n_sessions": parse_int(row.get("n_sessions")),
                "n_positive_sessions": parse_int(row.get("n_positive_sessions")),
                "sign_test_p_two_sided": parse_float(row.get("sign_test_p_two_sided")),
            }
        )
    return out


def decoder_qc_rows(source_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in read_csv_rows(require_file(source_dir / "leakage_audit.csv")):
        out.append(
            {
                "qc_type": "trial_split_leakage",
                "session": row.get("session", ""),
                "subject": row.get("subject", ""),
                "fold_id": parse_int(row.get("fold_id")),
                "status": row.get("status", ""),
                "n_train_trials": parse_int(row.get("n_train_trials")),
                "n_test_trials": parse_int(row.get("n_test_trials")),
                "n_shared_trials": parse_int(row.get("n_shared_trials")),
                "n_train_samples_base": parse_int(row.get("n_train_samples_base")),
                "n_test_samples_base": parse_int(row.get("n_test_samples_base")),
            }
        )
    for row in read_csv_rows(require_file(source_dir / "fold_model_metrics.csv")):
        out.append(
            {
                "qc_type": "model_fit",
                "session": row.get("session", ""),
                "subject": row.get("subject", ""),
                "fold_id": parse_int(row.get("fold_id")),
                "model_name": row.get("model_name", ""),
                "status": "pass" if parse_int(row.get("n_fit_failures")) == 0 else "fit_failures",
                "n_fit_failures": parse_int(row.get("n_fit_failures")),
                "n_constant_units": parse_int(row.get("n_constant_units")),
                "n_units_fit": parse_int(row.get("n_units_fit")),
                "mean_n_iter": parse_float(row.get("mean_n_iter")),
                "n_max_iter": parse_int(row.get("n_max_iter")),
            }
        )
    return out


def shuffle_control_rows(source_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(require_file(source_dir / "fold_eye_shuffle_nulls.csv"))
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "session": row.get("session", ""),
                "subject": row.get("subject", ""),
                "fold_id": parse_int(row.get("fold_id")),
                "model_name": row.get("model_name", ""),
                "null_type": row.get("null_type", ""),
                "null_draw": parse_int(row.get("null_draw")),
                "n_shuffle_self_donors": parse_int(row.get("n_shuffle_self_donors")),
                "total_spikes": parse_float(row.get("total_spikes")),
                "delta_ll_vs_pose_blind_psth": parse_float(row.get("delta_ll_vs_psth")),
                "bits_per_spike_delta_vs_pose_blind_psth": parse_float(row.get("bits_per_spike_delta_vs_psth")),
            }
        )
    return out


def aggregate_contrast_summary(contrast_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    session_rows = [row for row in contrast_rows if row.get("scope") == "session"]
    comparisons = sorted({str(row["comparison"]) for row in session_rows})
    out: list[dict[str, Any]] = []
    for comparison in comparisons:
        vals = [parse_float(row.get("value")) for row in session_rows if row["comparison"] == comparison]
        mean, sem, n = mean_sem(vals)
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        out.append(
            {
                "comparison": comparison,
                "metric_name": "bits_per_spike_delta_difference",
                "n_sessions": n,
                "aggregation": "source_runner_unweighted_fold_stats",
                "mean": mean,
                "sem": sem,
                "min": float(np.min(arr)) if arr.size else float("nan"),
                "max": float(np.max(arr)) if arr.size else float("nan"),
                "n_positive_sessions": int(np.sum(arr > 0)) if arr.size else 0,
            }
        )
    return out


def plot_session_pairs(out_dir: Path, contrast_rows: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    session_rows = [row for row in contrast_rows if row.get("scope") == "session"]
    comparisons = [
        "M1_additive_eye_minus_M0_psth_glm",
        "M1_minus_eye_shuffle",
        "M3_minus_M1",
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(comparisons))
    for i, comparison in enumerate(comparisons):
        vals = [parse_float(row.get("value")) for row in session_rows if row.get("comparison") == comparison]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.asarray([0.0])
        ax.scatter(np.full(len(vals), x[i]) + jitter, vals, color="#2f6fa5", alpha=0.65, s=24)
        ax.plot([x[i] - 0.22, x[i] + 0.22], [np.mean(vals), np.mean(vals)], color="#222222", linewidth=1.5)
    ax.axhline(0, color="#aaaaaa", linewidth=0.8)
    ax.set_xticks(x, ["M1-M0", "M1-shuffle", "M3-M1"])
    ax.set_ylabel("bits/spike difference")
    ax.set_title("recorded pose-aware prediction contrasts")
    fig.tight_layout()
    fig.savefig(fig_dir / "recorded_pose_info_session_pairs.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    by_subject: dict[str, list[float]] = {}
    for row in session_rows:
        if row.get("comparison") == "M1_additive_eye_minus_M0_psth_glm":
            by_subject.setdefault(str(row.get("subject", "")), []).append(parse_float(row.get("value")))
    labels = sorted(by_subject)
    means = [float(np.nanmean(by_subject[label])) for label in labels]
    sems = [
        float(np.nanstd(by_subject[label], ddof=1) / np.sqrt(len(by_subject[label])))
        if len(by_subject[label]) > 1
        else 0.0
        for label in labels
    ]
    ax.bar(np.arange(len(labels)), means, yerr=sems, color="#9fb8cc", edgecolor="#315f7d", capsize=3)
    ax.axhline(0, color="#aaaaaa", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("M1-M0 bits/spike")
    ax.set_title("additive eye model by subject")
    fig.tight_layout()
    fig.savefig(fig_dir / "recorded_pose_info_by_subject.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    out_dir: Path,
    *,
    source_dir: Path,
    manifest: dict[str, Any],
    session_rows: list[dict[str, Any]],
    contrast_summary: list[dict[str, Any]],
) -> None:
    def find(comparison: str) -> dict[str, Any] | None:
        for row in contrast_summary:
            if row.get("comparison") == comparison:
                return row
        return None

    m1 = find("M1_additive_eye_minus_M0_psth_glm")
    m1_shuffle = find("M1_minus_eye_shuffle")
    m3_m1 = find("M3_minus_M1")
    lines = [
        "# Recorded Pose Information Summary",
        "",
        f"Source run: `{source_dir}`",
        f"Status: `{manifest.get('status', 'unknown')}`; sessions ok: {manifest.get('n_sessions_ok', 'NA')} / {manifest.get('n_sessions_requested', 'NA')}.",
        "",
        "This adapter summarizes the completed held-out Poisson response-prediction ladder. The blind baseline is `M0_psth_glm`; pose-aware variants add measured eye state.",
        "",
        "Aggregation note: `recorded_pose_info_session_metrics.csv` uses spike-weighted summed held-out likelihoods over folds. `recorded_pose_info_paired_contrasts.csv` and the headline summary use the source runner's unweighted fold mean/median convention so they agree with `model_comparison_summary.csv` from the completed run.",
        "",
        "## Headline Contrasts",
        "",
    ]
    for label, row in [
        ("M1 additive eye minus blind PSTH", m1),
        ("M1 additive eye minus trial-shuffled eye", m1_shuffle),
        ("M3 time-by-eye interaction minus M1 additive eye", m3_m1),
    ]:
        if row is None:
            lines.append(f"- {label}: unavailable")
        else:
            lines.append(
                f"- {label}: mean={row['mean']:.6g} bits/spike, SEM={row['sem']:.6g}, "
                f"n={row['n_sessions']}, positive_sessions={row['n_positive_sessions']}"
            )
    lines.extend(
        [
            "",
            "Interpretation: this completed run is a guardrail/negative recorded anchor in its current form. Measured eye covariates did not improve held-out response prediction over the PSTH baseline, and the richer time-by-eye interaction was worse than the additive model.",
            "",
            "Caveat: this is response prediction, not a direct stimulus-decoding likelihood. It is still useful for the stack because it tests whether simple measured-pose covariates add recoverable recorded-V1 signal under trial-disjoint splits.",
            "",
        ]
    )
    (out_dir / "recorded_pose_info_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(source_dir)
    session_metrics = session_metric_rows(source_dir)
    unit_counts = unit_count_rows(source_dir)
    contrasts = paired_contrast_rows(source_dir, session_metrics)
    contrast_summary = aggregate_contrast_summary(contrasts)
    decoder_qc = decoder_qc_rows(source_dir)
    shuffle_controls = shuffle_control_rows(source_dir)

    write_csv_rows(out_dir / "recorded_pose_info_session_metrics.csv", session_metrics)
    write_csv_rows(out_dir / "recorded_pose_info_unit_counts.csv", unit_counts)
    write_csv_rows(out_dir / "recorded_pose_info_paired_contrasts.csv", contrasts)
    write_csv_rows(out_dir / "recorded_pose_info_paired_contrast_summary.csv", contrast_summary)
    write_csv_rows(out_dir / "recorded_pose_info_decoder_qc.csv", decoder_qc)
    write_csv_rows(out_dir / "recorded_pose_info_shuffle_controls.csv", shuffle_controls)
    plot_session_pairs(out_dir, contrasts)
    write_summary(
        out_dir,
        source_dir=source_dir,
        manifest=manifest,
        session_rows=session_metrics,
        contrast_summary=contrast_summary,
    )
    write_json(
        out_dir / "recorded_pose_info_manifest.json",
        {
            "analysis": "recorded_pose_information_adapter",
            "source_dir": source_dir,
            "source_manifest": source_dir / "recorded_pose_aware_prediction_manifest.json",
            "out_dir": out_dir,
            "n_session_metric_rows": len(session_metrics),
            "n_unit_count_rows": len(unit_counts),
            "n_paired_contrast_rows": len(contrasts),
            "n_decoder_qc_rows": len(decoder_qc),
            "n_shuffle_control_rows": len(shuffle_controls),
            "source_status": manifest.get("status", "unknown"),
            "session_metrics_aggregation": "spike_weighted_sum_over_folds",
            "paired_contrasts_aggregation": "source_runner_unweighted_fold_stats",
            "aggregate_comparisons_aggregation": "source_runner_session_bootstrap",
            "claim_guardrail": manifest.get("claim_guardrail", ""),
        },
    )
    print(f"Wrote recorded pose-information adapter outputs to {out_dir}")


if __name__ == "__main__":
    main()
