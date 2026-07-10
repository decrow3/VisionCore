"""Audit whether the candidate-free Figure 4C observer is ready for latent-eye claims.

This script is intentionally diagnostic.  It does not try to rescue the
candidate-free observer by changing the model or score.  It reads an existing
linear synthetic-prior observer run and writes compact CSV/README outputs that
make the promotion gate, nested-known fallback behavior, and tau-validity
diagnostics explicit.  A failed gate is treated as unresolved validity, not as
proof that the latent is non-eye-related.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_KNOWN_MODE = "pose_known_nested_tau_interactions"
DEFAULT_JOINT_MODE = "hidden_joint_forward_model"
DEFAULT_ZERO_MODE = "zero_static"
DEFAULT_RESPONSE_MODE = "response_only"
DEFAULT_RECORDED_FORWARD_MODE = "pose_known_forward_model"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-c-dir", type=Path, default=None)
    parser.add_argument("--trials-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--gate-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--known-mode", default=DEFAULT_KNOWN_MODE)
    parser.add_argument("--joint-mode", default=DEFAULT_JOINT_MODE)
    parser.add_argument("--zero-mode", default=DEFAULT_ZERO_MODE)
    parser.add_argument("--response-mode", default=DEFAULT_RESPONSE_MODE)
    parser.add_argument("--recorded-forward-mode", default=DEFAULT_RECORDED_FORWARD_MODE)
    parser.add_argument("--denominator-min", type=float, default=1e-6)
    return parser.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    base = args.four_c_dir
    trials_csv = args.trials_csv or (
        base / "linear_synthetic_prior_feature_observer_trials.csv" if base is not None else None
    )
    summary_csv = args.summary_csv or (
        base / "linear_synthetic_prior_feature_observer_summary.csv" if base is not None else None
    )
    gate_csv = args.gate_csv or (
        base / "gates" / "unified_feature_observer_gate_table.csv" if base is not None else None
    )
    if trials_csv is None or not trials_csv.exists():
        raise FileNotFoundError("Provide --trials-csv or --four-c-dir with observer trials.")
    if summary_csv is None or not summary_csv.exists():
        raise FileNotFoundError("Provide --summary-csv or --four-c-dir with observer summary.")
    if gate_csv is not None and not gate_csv.exists():
        gate_csv = None
    return trials_csv, summary_csv, gate_csv


def _pooled_r2(frame: pd.DataFrame, mode: str, *, sse_col: str = "feature_sse") -> float:
    subset = frame[frame["observer_mode"].astype(str).eq(mode)]
    if subset.empty or sse_col not in subset.columns or "feature_sst_train_baseline" not in subset.columns:
        return float("nan")
    sse = float(np.nansum(subset[sse_col].to_numpy(dtype=np.float64)))
    sst = float(np.nansum(subset["feature_sst_train_baseline"].to_numpy(dtype=np.float64)))
    return float(1.0 - sse / sst) if np.isfinite(sst) and sst > 1e-12 else float("nan")


def _all_gate_row(
    trials: pd.DataFrame,
    gate: pd.DataFrame | None,
    *,
    known_mode: str,
    joint_mode: str,
    zero_mode: str,
    response_mode: str,
    denominator_min: float,
) -> dict[str, Any]:
    if gate is not None and not gate.empty and "group_kind" in gate.columns:
        all_rows = gate[gate["group_kind"].astype(str).eq("all")]
        if not all_rows.empty:
            row = all_rows.iloc[0].to_dict()
            return {
                "score": row.get("score", "R2_cv"),
                "S_known": row.get("S_known", np.nan),
                "S_joint": row.get("S_joint", np.nan),
                "S_zero": row.get("S_zero", np.nan),
                "S_response": row.get("S_response", np.nan),
                "known_minus_zero": row.get("known_minus_zero", np.nan),
                "joint_minus_zero": row.get("joint_minus_zero", np.nan),
                "known_minus_joint": row.get("known_minus_joint", np.nan),
                "joint_minus_response": row.get("joint_minus_response", np.nan),
                "gap_recovered_reportable": bool(row.get("gap_recovered_reportable", False)),
                "gate_source": "gate_table",
            }
    s_known = _pooled_r2(trials, known_mode)
    s_joint = _pooled_r2(trials, joint_mode)
    s_zero = _pooled_r2(trials, zero_mode)
    s_response = _pooled_r2(trials, response_mode)
    known_minus_zero = s_known - s_zero
    joint_minus_zero = s_joint - s_zero
    return {
        "score": "R2_cv",
        "S_known": s_known,
        "S_joint": s_joint,
        "S_zero": s_zero,
        "S_response": s_response,
        "known_minus_zero": known_minus_zero,
        "joint_minus_zero": joint_minus_zero,
        "known_minus_joint": s_known - s_joint,
        "joint_minus_response": s_joint - s_response,
        "gap_recovered_reportable": bool(
            np.isfinite(known_minus_zero) and known_minus_zero > denominator_min and joint_minus_zero > 0.0
        ),
        "gate_source": "trials_recomputed",
    }


def _key_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "decoder_mode",
        "latent",
        "feature_space_mode",
        "observation_scale",
        "prior_family",
        "table_index",
        "trial_id",
        "true_source_row",
    ]
    return [col for col in candidates if col in frame.columns]


def _nested_known_audit(
    trials: pd.DataFrame,
    *,
    known_mode: str,
    response_mode: str,
) -> pd.DataFrame:
    known = trials[trials["observer_mode"].astype(str).eq(known_mode)].copy()
    if known.empty:
        return pd.DataFrame()
    group_cols = [col for col in ["observation_scale", "prior_family", "fold"] if col in known.columns]
    if {"first_pass_feature_sse", "feature_sse"}.issubset(known.columns):
        rows = []
        for values, group in known.groupby(group_cols, dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            first_sse = float(np.nansum(group["first_pass_feature_sse"].to_numpy(dtype=np.float64)))
            final_sse = float(np.nansum(group["feature_sse"].to_numpy(dtype=np.float64)))
            alpha = float(group["residual_shrinkage_lambda"].dropna().iloc[0]) if "residual_shrinkage_lambda" in group else np.nan
            rows.append(
                {
                    **dict(zip(group_cols, values)),
                    "comparison": "known_nested_vs_internal_z0",
                    "n": int(len(group)),
                    "residual_shrinkage_lambda": alpha,
                    "known_sse": final_sse,
                    "baseline_sse": first_sse,
                    "known_minus_baseline_sse": final_sse - first_sse,
                    "known_sse_le_baseline_sse": bool(final_sse <= first_sse + 1e-9),
                    "row_fraction_known_sse_le_baseline": float(
                        np.mean(
                            group["feature_sse"].to_numpy(dtype=np.float64)
                            <= group["first_pass_feature_sse"].to_numpy(dtype=np.float64) + 1e-9
                        )
                    ),
                }
            )
        return pd.DataFrame(rows)

    keys = _key_columns(known)
    if not keys:
        return pd.DataFrame()
    paired = trials[trials["observer_mode"].astype(str).isin([known_mode, response_mode])].copy()
    wide = paired.pivot_table(index=keys, columns="observer_mode", values="feature_sse", aggfunc="first")
    required = [known_mode, response_mode]
    if any(mode not in wide.columns for mode in required):
        return pd.DataFrame()
    wide = wide.dropna(subset=required).reset_index()
    rows = []
    group_cols = [col for col in ["observation_scale", "prior_family"] if col in wide.columns]
    for values, group in wide.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        known_sse = float(np.nansum(group[known_mode].to_numpy(dtype=np.float64)))
        response_sse = float(np.nansum(group[response_mode].to_numpy(dtype=np.float64)))
        rows.append(
            {
                **dict(zip(group_cols, values)),
                "comparison": "known_nested_vs_public_response_only",
                "n": int(len(group)),
                "known_sse": known_sse,
                "baseline_sse": response_sse,
                "known_minus_baseline_sse": known_sse - response_sse,
                "known_sse_le_baseline_sse": bool(known_sse <= response_sse + 1e-9),
                "row_fraction_known_sse_le_baseline": float(
                    np.mean(group[known_mode].to_numpy(dtype=np.float64) <= group[response_mode].to_numpy(dtype=np.float64) + 1e-9)
                ),
            }
        )
    return pd.DataFrame(rows)


def _tau_validity_audit(
    trials: pd.DataFrame,
    *,
    joint_mode: str,
    recorded_forward_mode: str,
) -> pd.DataFrame:
    if "forward_response_residual_mse" not in trials.columns:
        return pd.DataFrame()
    modes = [joint_mode, recorded_forward_mode]
    subset = trials[trials["observer_mode"].astype(str).isin(modes)].copy()
    if subset.empty:
        return pd.DataFrame()
    keys = _key_columns(subset)
    if not keys:
        return pd.DataFrame()
    wide = subset.pivot_table(index=keys, columns="observer_mode", values="forward_response_residual_mse", aggfunc="first")
    if any(mode not in wide.columns for mode in modes):
        return pd.DataFrame()
    wide = wide.dropna(subset=modes).reset_index()
    metric_cols = [
        col
        for col in [
            "trajectory_rmse",
            "trajectory_corr_mean",
            "trajectory_r2",
            "profile_energy",
            "forward_profile_energy",
        ]
        if col in subset.columns
    ]
    joint_metrics = (
        subset[subset["observer_mode"].astype(str).eq(joint_mode)]
        .groupby(keys, dropna=False, sort=False)[metric_cols]
        .first()
        .reset_index()
        if metric_cols
        else pd.DataFrame()
    )
    if not joint_metrics.empty:
        wide = wide.merge(joint_metrics, on=keys, how="left")
    group_cols = [col for col in ["observation_scale", "prior_family"] if col in wide.columns]
    rows = []
    for values, group in wide.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        recorded = group[recorded_forward_mode].to_numpy(dtype=np.float64)
        optimized = group[joint_mode].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            **dict(zip(group_cols, values)),
            "comparison": "optimized_tau_vs_recorded_tau_forward_residual",
            "n": int(len(group)),
            "mean_recorded_tau_forward_residual_mse": float(np.nanmean(recorded)),
            "mean_optimized_tau_forward_residual_mse": float(np.nanmean(optimized)),
            "optimized_minus_recorded_forward_residual_mse": float(np.nanmean(optimized - recorded)),
            "fraction_optimized_residual_lt_recorded": float(np.nanmean(optimized < recorded)),
        }
        for col in metric_cols:
            values_arr = group[col].to_numpy(dtype=np.float64)
            row[f"mean_joint_{col}"] = float(np.nanmean(values_arr))
            row[f"median_joint_{col}"] = float(np.nanmedian(values_arr))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_readme(
    path: Path,
    *,
    gate_row: dict[str, Any],
    nested_audit: pd.DataFrame,
    tau_audit: pd.DataFrame,
    branch_status: str,
    latent_label: str,
    failure_mode: str,
    model_error_warning: bool,
) -> None:
    known_minus_zero = float(gate_row.get("known_minus_zero", np.nan))
    joint_minus_zero = float(gate_row.get("joint_minus_zero", np.nan))
    optimized_better = (
        bool((tau_audit["optimized_minus_recorded_forward_residual_mse"] < 0).all())
        if "optimized_minus_recorded_forward_residual_mse" in tau_audit
        else False
    )
    internal_text = "not available"
    if not nested_audit.empty:
        bad = int((~nested_audit["known_sse_le_baseline_sse"].astype(bool)).sum())
        total = int(len(nested_audit))
        internal_text = f"{bad} / {total} groups worse than their baseline comparison"
    lines = [
        "# Candidate-free observer validity audit",
        "",
        f"Branch status: `{branch_status}`",
        f"Current latent label: `{latent_label}`",
        f"Current failure mode: `{failure_mode}`",
        "",
        "Promotion gate:",
        f"- `known - zero`: {known_minus_zero:.6g}",
        f"- `joint - zero`: {joint_minus_zero:.6g}",
        f"- gap reportable: `{bool(gate_row.get('gap_recovered_reportable', False))}`",
        "",
        "Nested-known fallback audit:",
        f"- {internal_text}",
        "",
        "Tau-validity audit:",
        f"- optimized tau residual lower than recorded tau in all reported groups: `{optimized_better}`",
        f"- model-error-latent warning raised: `{model_error_warning}`",
        "",
        "Decision rule used here:",
        "- Mark the branch paper-facing only if the known-trajectory reference beats zero-eye and the joint observer also beats zero-eye.",
        "- If the known-trajectory ceiling is not established, mark validity unresolved rather than drawing a final latent-identity conclusion.",
        "- Treat optimized-tau residuals beating recorded-tau residuals as a warning that the observation model may be confounded.",
        "- Do not call the latent an inferred retinal trajectory until self-consistency, direct known-tau decoding, and coordinate-alignment controls pass.",
        "",
        "Recommended recovery ladder:",
        "- Synthetic self-consistency positive control: generate responses from the fitted observer model and require known > joint > zero.",
        "- Direct known-tau feature decoder: train `g(r, tau_true)` with response-only fallback and the locked pooled R2_cv score.",
        "- Recorded-tau alignment audit: predeclared sign, x/y swap, lag, scale, and coordinate-convention checks.",
        "- Nuisance-slack control: separate eye trajectory from a small explicitly labeled model-error latent.",
        "- Per-scale and per-feature decomposition before interpreting all-scale failures.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _classify_validity(
    *,
    known_valid: bool,
    joint_valid: bool,
    optimized_beats_recorded: bool,
) -> tuple[str, str, str, bool]:
    if known_valid and joint_valid and not optimized_beats_recorded:
        return (
            "paper_facing_candidate",
            "candidate_eye_trajectory",
            "none_detected_by_gate",
            False,
        )
    failure_bits: list[str] = []
    if not known_valid:
        failure_bits.append("true_tau_ceiling_not_established_under_observation_model")
    if not joint_valid:
        failure_bits.append("joint_does_not_beat_zero_under_locked_score")
    if optimized_beats_recorded:
        failure_bits.append("optimized_tau_fits_compact_response_better_than_recorded_tau")
    return (
        "validity_unresolved",
        "candidate_eye_state_unresolved",
        "+".join(failure_bits) if failure_bits else "unresolved_control_battery_pending",
        bool(optimized_beats_recorded),
    )


def main() -> None:
    args = _parse_args()
    trials_csv, summary_csv, gate_csv = _resolve_inputs(args)
    trials = pd.read_csv(trials_csv)
    _summary = pd.read_csv(summary_csv)
    gate = pd.read_csv(gate_csv) if gate_csv is not None else None

    gate_row = _all_gate_row(
        trials,
        gate,
        known_mode=str(args.known_mode),
        joint_mode=str(args.joint_mode),
        zero_mode=str(args.zero_mode),
        response_mode=str(args.response_mode),
        denominator_min=float(args.denominator_min),
    )
    nested_audit = _nested_known_audit(
        trials,
        known_mode=str(args.known_mode),
        response_mode=str(args.response_mode),
    )
    tau_audit = _tau_validity_audit(
        trials,
        joint_mode=str(args.joint_mode),
        recorded_forward_mode=str(args.recorded_forward_mode),
    )

    known_valid = bool(np.isfinite(gate_row["known_minus_zero"]) and float(gate_row["known_minus_zero"]) > 0.0)
    joint_valid = bool(np.isfinite(gate_row["joint_minus_zero"]) and float(gate_row["joint_minus_zero"]) > 0.0)
    optimized_beats_recorded = (
        bool((tau_audit["optimized_minus_recorded_forward_residual_mse"] < 0.0).all())
        if "optimized_minus_recorded_forward_residual_mse" in tau_audit
        else False
    )
    branch_status, latent_label, failure_mode, model_error_warning = _classify_validity(
        known_valid=known_valid,
        joint_valid=joint_valid,
        optimized_beats_recorded=optimized_beats_recorded,
    )

    validity_summary = pd.DataFrame(
        [
            {
                **gate_row,
                "known_reference_valid": known_valid,
                "joint_beats_zero": joint_valid,
                "optimized_tau_beats_recorded_tau_forward_residual": optimized_beats_recorded,
                "model_error_latent_warning": model_error_warning,
                "model_error_latent_conclusion": False,
                "current_failure_mode": failure_mode,
                "candidate_free_branch_status": branch_status,
                "recommended_latent_label": latent_label,
            }
        ]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    validity_summary.to_csv(args.out_dir / "candidate_free_observer_validity_summary.csv", index=False)
    nested_audit.to_csv(args.out_dir / "nested_known_fallback_audit.csv", index=False)
    tau_audit.to_csv(args.out_dir / "tau_validity_audit.csv", index=False)
    _write_readme(
        args.out_dir / "README.md",
        gate_row=gate_row,
        nested_audit=nested_audit,
        tau_audit=tau_audit,
        branch_status=branch_status,
        latent_label=latent_label,
        failure_mode=failure_mode,
        model_error_warning=model_error_warning,
    )
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
