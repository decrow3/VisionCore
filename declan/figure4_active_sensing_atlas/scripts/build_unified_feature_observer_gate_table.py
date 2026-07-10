"""Build the Figure 4 unified feature-observer promotion gate table.

The gate table is intentionally score-only: it consumes observer trial or
summary rows that already contain per-row or pooled SSE/SST in the locked,
train-normalized feature space, then reports the pooled multi-output R2_cv
contrasts needed for deciding whether a candidate-free joint observer has
earned promotion beyond zero-eye and response-only baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from declan.feature_recovery_scores import R2_CV_METHOD, pooled_multioutput_r2_from_sse_sst


DEFAULT_KNOWN_MODE_CANDIDATES = (
    "pose_known_nested_tau_interactions",
    "pose_known_nested_tau_linear",
    "true_tau_interactions",
    "true_tau_linear",
    "pose_known_tau_interactions",
    "pose_known_tau_linear",
    "pose_known_forward_model",
)
DEFAULT_JOINT_MODE = "hidden_joint_forward_model"
DEFAULT_ZERO_MODE = "zero_static"
DEFAULT_RESPONSE_MODE = "response_only"
STATIC_MATCHED_MEAN_MODE = "static_matched_mean"
STATIC_MATCHED_MEAN_ALIASES = (
    "zero_static",
    "static_zero",
    "static_crop_center",
    "static_trace_mean_centered",
    "stabilized_at_mean_position",
)

GROUP_COLUMNS = (
    "decoder_mode",
    "latent",
    "feature_space_mode",
    "observation_scale",
    "prior_family",
)
TRIAL_KEY_CANDIDATES = (
    "decoder_mode",
    "latent",
    "feature_space_mode",
    "observation_scale",
    "prior_family",
    "table_index",
    "trial_id",
    "true_source_row",
    "response_cache_path",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--known-mode",
        default="auto",
        help=(
            "Known-trajectory reference mode. Use 'auto' to prefer "
            "pose_known_nested_tau_interactions, then pose_known_tau_interactions, "
            "then pose_known_forward_model."
        ),
    )
    parser.add_argument("--joint-mode", default=DEFAULT_JOINT_MODE)
    parser.add_argument("--zero-mode", default=DEFAULT_ZERO_MODE)
    parser.add_argument("--response-mode", default=DEFAULT_RESPONSE_MODE)
    parser.add_argument("--sse-column", default="feature_sse")
    parser.add_argument("--sst-column", default="feature_sst_train_baseline")
    parser.add_argument("--score-name", default="R2_cv")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--denominator-min",
        type=float,
        default=1e-6,
        help="Minimum S_known - S_zero contrast required before reporting gap recovered.",
    )
    return parser.parse_args()


def _read_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.trials_csv is not None and args.trials_csv.exists():
        frame = pd.read_csv(args.trials_csv)
        required = {"observer_mode", str(args.sse_column), str(args.sst_column)}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{args.trials_csv} lacks required columns: {sorted(missing)}")
        return frame, "trials"
    if args.summary_csv is not None and args.summary_csv.exists():
        frame = pd.read_csv(args.summary_csv)
        required = {"observer_mode", "R2_cv"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{args.summary_csv} lacks required columns: {sorted(missing)}")
        return frame, "summary"
    raise FileNotFoundError("Provide an existing --trials-csv or --summary-csv.")


def _select_mode(frame: pd.DataFrame, requested: str, candidates: tuple[str, ...]) -> str:
    if requested != "auto":
        return _resolve_mode_alias(frame, requested)
    observed = set(frame["observer_mode"].astype(str))
    for candidate in candidates:
        if candidate in observed:
            return candidate
    return candidates[-1]


def _resolve_mode_alias(frame: pd.DataFrame, requested: str) -> str:
    mode = str(requested)
    observed = set(frame["observer_mode"].astype(str))
    if mode in observed:
        return mode
    if mode == STATIC_MATCHED_MEAN_MODE or mode in STATIC_MATCHED_MEAN_ALIASES:
        static_candidates = (STATIC_MATCHED_MEAN_MODE, *STATIC_MATCHED_MEAN_ALIASES)
        for candidate in static_candidates:
            if candidate in observed:
                return candidate
    return mode


def _available_group_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in GROUP_COLUMNS if col in frame.columns]


def _all_group_values(frame: pd.DataFrame, group_cols: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for col in group_cols:
        unique = frame[col].dropna().unique()
        values[col] = unique[0] if len(unique) == 1 else "all"
    return values


def _iter_groups(frame: pd.DataFrame, group_cols: list[str]):
    yield _all_group_values(frame, group_cols), frame.copy(), "all"
    if group_cols:
        grouped = frame.groupby(group_cols, dropna=False, sort=True)
        for values, group in grouped:
            if not isinstance(values, tuple):
                values = (values,)
            yield dict(zip(group_cols, values)), group.copy(), "group"


def _pooled_score_from_trials(
    frame: pd.DataFrame,
    mode: str,
    *,
    sse_column: str,
    sst_column: str,
) -> dict[str, float]:
    subset = frame[frame["observer_mode"].astype(str).eq(mode)]
    if subset.empty:
        return {"R2_cv": np.nan, "sse": np.nan, "sst": np.nan, "n_rows": 0}
    score = pooled_multioutput_r2_from_sse_sst(
        subset[sse_column].to_numpy(dtype=np.float64),
        subset[sst_column].to_numpy(dtype=np.float64),
    )
    return {"R2_cv": score.r2, "sse": score.sse, "sst": score.sst, "n_rows": score.n_samples}


def _pooled_score_from_summary(frame: pd.DataFrame, mode: str) -> dict[str, float]:
    subset = frame[frame["observer_mode"].astype(str).eq(mode)]
    if subset.empty:
        return {"R2_cv": np.nan, "sse": np.nan, "sst": np.nan, "n_rows": 0}
    if {"feature_sse", "feature_sst_train_baseline"}.issubset(subset.columns):
        score = pooled_multioutput_r2_from_sse_sst(
            subset["feature_sse"].to_numpy(dtype=np.float64),
            subset["feature_sst_train_baseline"].to_numpy(dtype=np.float64),
        )
        return {
            "R2_cv": score.r2,
            "sse": score.sse,
            "sst": score.sst,
            "n_rows": int(subset["n"].sum()) if "n" in subset.columns else score.n_samples,
        }
    return {
        "R2_cv": float(subset["R2_cv"].iloc[0]),
        "sse": np.nan,
        "sst": np.nan,
        "n_rows": int(subset["n"].sum()) if "n" in subset.columns else len(subset),
    }


def _score_by_mode(
    frame: pd.DataFrame,
    mode: str,
    *,
    source: str,
    sse_column: str,
    sst_column: str,
) -> dict[str, float]:
    if source == "trials":
        return _pooled_score_from_trials(
            frame,
            mode,
            sse_column=sse_column,
            sst_column=sst_column,
        )
    return _pooled_score_from_summary(frame, mode)


def _trial_key_columns(frame: pd.DataFrame) -> list[str]:
    cols = set(frame.columns)
    if {"table_index", "trial_id"}.issubset(cols):
        return ["table_index", "trial_id"]
    fallback = [col for col in ("true_source_row", "response_cache_path") if col in cols]
    return fallback


def _bootstrap_gate(
    frame: pd.DataFrame,
    *,
    known_mode: str,
    joint_mode: str,
    zero_mode: str,
    response_mode: str,
    n_bootstrap: int,
    seed: int,
    denominator_min: float,
    sse_column: str,
    sst_column: str,
) -> dict[str, float]:
    if n_bootstrap <= 0:
        return {"bootstrap_n": 0}
    key_cols = _trial_key_columns(frame)
    if not key_cols:
        return {"bootstrap_n": 0, "bootstrap_skipped_no_shared_trial_keys": 1}

    modes = [known_mode, joint_mode, zero_mode, response_mode]
    tmp = frame[frame["observer_mode"].astype(str).isin(modes)].copy()
    if tmp.empty:
        return {"bootstrap_n": 0}
    grouped = (
        tmp.groupby(key_cols + ["observer_mode"], dropna=False, sort=False)[[sse_column, sst_column]]
        .sum()
        .reset_index()
    )
    wide_sse = grouped.pivot(index=key_cols, columns="observer_mode", values=sse_column)
    wide_sst = grouped.pivot(index=key_cols, columns="observer_mode", values=sst_column)
    required = [known_mode, joint_mode, zero_mode]
    if any(mode not in wide_sse.columns or mode not in wide_sst.columns for mode in required):
        return {"bootstrap_n": 0, "bootstrap_skipped_missing_required_mode": 1}
    complete = wide_sse[required].notna().all(axis=1) & wide_sst[required].notna().all(axis=1)
    wide_sse = wide_sse.loc[complete]
    wide_sst = wide_sst.loc[complete]
    if wide_sse.empty:
        return {"bootstrap_n": 0, "bootstrap_skipped_no_complete_mode_triplets": 1}

    rng = np.random.default_rng(seed)
    n = len(wide_sse)
    values: dict[str, list[float]] = {
        "known_minus_zero": [],
        "joint_minus_zero": [],
        "known_minus_joint": [],
        "joint_minus_response": [],
        "gap_recovered": [],
    }
    for _ in range(int(n_bootstrap)):
        sample = rng.integers(0, n, size=n)
        scores: dict[str, float] = {}
        for mode in modes:
            if mode not in wide_sse.columns or mode not in wide_sst.columns:
                scores[mode] = np.nan
                continue
            sse = float(np.nansum(wide_sse[mode].to_numpy(dtype=np.float64)[sample]))
            sst = float(np.nansum(wide_sst[mode].to_numpy(dtype=np.float64)[sample]))
            scores[mode] = float(1.0 - sse / sst) if np.isfinite(sst) and sst > 1e-12 else np.nan
        known_minus_zero = scores[known_mode] - scores[zero_mode]
        joint_minus_zero = scores[joint_mode] - scores[zero_mode]
        known_minus_joint = scores[known_mode] - scores[joint_mode]
        joint_minus_response = scores[joint_mode] - scores[response_mode]
        denom = known_minus_zero
        gap = joint_minus_zero / denom if np.isfinite(denom) and denom > denominator_min else np.nan
        values["known_minus_zero"].append(float(known_minus_zero))
        values["joint_minus_zero"].append(float(joint_minus_zero))
        values["known_minus_joint"].append(float(known_minus_joint))
        values["joint_minus_response"].append(float(joint_minus_response))
        values["gap_recovered"].append(float(gap))

    out: dict[str, float] = {"bootstrap_n": int(n_bootstrap), "bootstrap_trial_units": int(n)}
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[f"{name}_ci_low"] = np.nan
            out[f"{name}_ci_high"] = np.nan
            out[f"{name}_bootstrap_mean"] = np.nan
        else:
            out[f"{name}_ci_low"] = float(np.quantile(arr, 0.025))
            out[f"{name}_ci_high"] = float(np.quantile(arr, 0.975))
            out[f"{name}_bootstrap_mean"] = float(np.mean(arr))
    return out


def _gate_row(
    frame: pd.DataFrame,
    *,
    group_values: dict[str, Any],
    group_kind: str,
    source: str,
    known_mode: str,
    joint_mode: str,
    zero_mode: str,
    response_mode: str,
    n_bootstrap: int,
    seed: int,
    denominator_min: float,
    sse_column: str,
    sst_column: str,
    score_name: str,
) -> dict[str, Any]:
    known = _score_by_mode(
        frame,
        known_mode,
        source=source,
        sse_column=sse_column,
        sst_column=sst_column,
    )
    joint = _score_by_mode(
        frame,
        joint_mode,
        source=source,
        sse_column=sse_column,
        sst_column=sst_column,
    )
    zero = _score_by_mode(
        frame,
        zero_mode,
        source=source,
        sse_column=sse_column,
        sst_column=sst_column,
    )
    response = _score_by_mode(
        frame,
        response_mode,
        source=source,
        sse_column=sse_column,
        sst_column=sst_column,
    )

    s_known = known["R2_cv"]
    s_joint = joint["R2_cv"]
    s_zero = zero["R2_cv"]
    s_response = response["R2_cv"]
    known_minus_zero = s_known - s_zero
    joint_minus_zero = s_joint - s_zero
    known_minus_joint = s_known - s_joint
    joint_minus_response = s_joint - s_response
    gap = (
        joint_minus_zero / known_minus_zero
        if np.isfinite(known_minus_zero) and known_minus_zero > denominator_min
        else np.nan
    )

    row: dict[str, Any] = {
        **group_values,
        "group_kind": group_kind,
        "score": score_name,
        "score_method": R2_CV_METHOD,
        "sse_column": sse_column,
        "sst_column": sst_column,
        "score_space": "locked_train_normalized_feature_space",
        "score_aggregation": "pooled_multioutput_out_of_fold_sse_sst",
        "known_mode": known_mode,
        "joint_mode": joint_mode,
        "zero_mode": zero_mode,
        "response_mode": response_mode,
        "S_known": s_known,
        "S_joint": s_joint,
        "S_zero": s_zero,
        "S_response": s_response,
        "known_sse": known["sse"],
        "known_sst": known["sst"],
        "joint_sse": joint["sse"],
        "joint_sst": joint["sst"],
        "zero_sse": zero["sse"],
        "zero_sst": zero["sst"],
        "response_sse": response["sse"],
        "response_sst": response["sst"],
        "known_n_rows": known["n_rows"],
        "joint_n_rows": joint["n_rows"],
        "zero_n_rows": zero["n_rows"],
        "response_n_rows": response["n_rows"],
        "known_minus_zero": known_minus_zero,
        "joint_minus_zero": joint_minus_zero,
        "known_minus_joint": known_minus_joint,
        "joint_minus_response": joint_minus_response,
        "known_minus_zero_denominator_positive": bool(
            np.isfinite(known_minus_zero) and known_minus_zero > denominator_min
        ),
        "joint_gt_zero": bool(np.isfinite(joint_minus_zero) and joint_minus_zero > 0.0),
        "known_gt_zero": bool(np.isfinite(known_minus_zero) and known_minus_zero > 0.0),
        "known_gt_joint_gt_zero": bool(
            np.isfinite(known_minus_joint)
            and np.isfinite(joint_minus_zero)
            and known_minus_joint > 0.0
            and joint_minus_zero > 0.0
        ),
        "joint_gt_response": bool(np.isfinite(joint_minus_response) and joint_minus_response > 0.0),
        "gap_recovered": gap,
        "gap_recovered_denominator": "S_known_minus_S_zero",
        "gap_recovered_denominator_min": float(denominator_min),
    }
    if source == "trials":
        row.update(
            _bootstrap_gate(
                frame,
                known_mode=known_mode,
                joint_mode=joint_mode,
                zero_mode=zero_mode,
                response_mode=response_mode,
                n_bootstrap=n_bootstrap,
                seed=seed,
                denominator_min=denominator_min,
                sse_column=sse_column,
                sst_column=sst_column,
            )
        )
    else:
        row["bootstrap_n"] = 0
    low = row.get("known_minus_zero_ci_low", np.nan)
    row["gap_recovered_reportable"] = bool(
        row["known_minus_zero_denominator_positive"]
        and (not np.isfinite(low) or float(low) > float(denominator_min))
    )
    return row


def _write_readme(path: Path, *, args: argparse.Namespace, output_csv: Path, known_mode: str, source: str) -> None:
    lines = [
        "# Unified feature-observer gate table",
        "",
        f"Input source: `{source}`",
        f"Output CSV: `{output_csv.name}`",
        "",
        "The score is pooled multi-output `R2_cv` in the locked, train-normalized feature space.",
        "Rows report raw contrasts alongside the gap-recovered ratio:",
        "",
        "`gap_recovered = (S_joint - S_zero) / (S_known - S_zero)`",
        "",
        "The ratio is not capped. It is marked reportable only when the known-minus-zero denominator is positive and, when bootstrapped, its lower interval is also positive.",
        "",
        f"Known mode: `{known_mode}`",
        f"Joint mode: `{args.joint_mode}`",
        f"Zero mode: `{args.zero_mode}`",
        f"Response mode: `{args.response_mode}`",
        f"SSE column: `{args.sse_column}`",
        f"SST column: `{args.sst_column}`",
        "",
        "`zero_static`, `static_zero`, and related legacy static names are resolved to `static_matched_mean` when that mode is present.",
        "",
        "A failed gate is diagnostic, not a reason to discard the observer: it means the relevant known, joint, zero, and response contracts need to be read before making a claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    frame, source = _read_input(args)
    known_mode = _select_mode(frame, str(args.known_mode), DEFAULT_KNOWN_MODE_CANDIDATES)
    joint_mode = _resolve_mode_alias(frame, str(args.joint_mode))
    zero_mode = _resolve_mode_alias(frame, str(args.zero_mode))
    response_mode = _resolve_mode_alias(frame, str(args.response_mode))
    group_cols = _available_group_columns(frame)

    rows = []
    for offset, (group_values, group, group_kind) in enumerate(_iter_groups(frame, group_cols)):
        rows.append(
            _gate_row(
                group,
                group_values=group_values,
                group_kind=group_kind,
                source=source,
                known_mode=known_mode,
                joint_mode=joint_mode,
                zero_mode=zero_mode,
                response_mode=response_mode,
                n_bootstrap=int(args.n_bootstrap),
                seed=int(args.seed) + offset,
                denominator_min=float(args.denominator_min),
                sse_column=str(args.sse_column),
                sst_column=str(args.sst_column),
                score_name=str(args.score_name),
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "unified_feature_observer_gate_table.csv"
    out_readme = args.out_dir / "README.md"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    args.joint_mode = joint_mode
    args.zero_mode = zero_mode
    args.response_mode = response_mode
    _write_readme(out_readme, args=args, output_csv=out_csv, known_mode=known_mode, source=source)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
