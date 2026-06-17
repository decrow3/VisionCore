#!/usr/bin/env python3
"""Run the cached-trajectory Vernier likelihood-ratio observer.

This is the exact-cache counterpart to the local geometry pilot.  It reads
cached ConvGRU rates, treats trajectory identity as nuisance state, and scores
Vernier sign by marginalizing over the empirical trajectory catalog.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .joint_observer import THETA_LABELS, THETA_MINUS, THETA_PLUS
from .metrics import expected_counts
from .trajectory_table_observer import (
    SUPPORTED_TABLE_LIKELIHOODS,
    score_trajectory_table_vernier_observer_trial,
    summarize_trajectory_table_rows,
    table_score_family,
)


def parse_csv_str(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_csv_float(text: str) -> list[float]:
    return [float(part) for part in parse_csv_str(text)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _condition_from_cache_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("rates_") or "_fd" not in stem:
        raise ValueError(f"Unexpected rate cache filename: {path.name}")
    return stem[len("rates_") : stem.rindex("_fd")]


def _unpadded_rates(arr: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(arr[i, : int(lengths[i])], dtype=np.float64) for i in range(arr.shape[0])]


def _load_rate_caches(source_dir: Path) -> dict[tuple[str, float, str], dict[str, Any]]:
    caches: dict[tuple[str, float, str], dict[str, Any]] = {}
    for path in sorted((source_dir / "cache").glob("rates_*_fd*arcmin.npz")):
        with np.load(path, allow_pickle=True) as npz:
            condition = str(npz["condition"][0]) if "condition" in npz else _condition_from_cache_path(path)
            fd_step = float(np.asarray(npz["fd_step_arcmin"])[0])
            inference_mode = str(npz["inference_mode"][0]) if "inference_mode" in npz else "framewise"
            lengths = np.asarray(npz["lengths"], dtype=np.int32)
            plus_rates = _unpadded_rates(np.asarray(npz["plus"], dtype=np.float64), lengths)
            minus_rates = _unpadded_rates(np.asarray(npz["minus"], dtype=np.float64), lengths)
        caches[(condition, fd_step, inference_mode)] = {
            "path": path,
            "condition": condition,
            "fd_step_arcmin": fd_step,
            "inference_mode": inference_mode,
            "plus_rates": plus_rates,
            "minus_rates": minus_rates,
        }
    if not caches:
        raise FileNotFoundError(f"No rate caches found under {source_dir / 'cache'}")
    return caches


def _stack_counts(
    rates: list[np.ndarray],
    *,
    bin_seconds: float,
    max_timebins: int,
) -> np.ndarray:
    if not rates:
        raise ValueError("Cannot stack an empty rate list")
    t = min(arr.shape[0] for arr in rates)
    if int(max_timebins) > 0:
        t = min(t, int(max_timebins))
    if t <= 0:
        raise ValueError("No time bins available after truncation")
    units = {int(arr.shape[1]) for arr in rates}
    if len(units) != 1:
        raise ValueError(f"Rate arrays must have one unit count, got {sorted(units)}")
    stacked = np.stack([arr[:t] for arr in rates], axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError("Rate cache contains non-finite values")
    return expected_counts(stacked, float(bin_seconds))


def _mean_reference_counts(
    cache: dict[str, Any],
    *,
    bin_seconds: float,
    max_timebins: int,
    target_timebins: int,
) -> dict[str, np.ndarray]:
    plus = _stack_counts(cache["plus_rates"], bin_seconds=bin_seconds, max_timebins=max_timebins)
    minus = _stack_counts(cache["minus_rates"], bin_seconds=bin_seconds, max_timebins=max_timebins)
    t = min(int(target_timebins), plus.shape[1], minus.shape[1])
    return {
        THETA_PLUS: np.mean(plus[:, :t], axis=0),
        THETA_MINUS: np.mean(minus[:, :t], axis=0),
    }


def _select_caches(
    caches: dict[tuple[str, float, str], dict[str, Any]],
    *,
    conditions: list[str],
    fd_steps: list[float],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    condition_filter = set(conditions)
    fd_filter = {round(float(fd), 10) for fd in fd_steps}
    for (condition, fd_step, _inference), cache in sorted(caches.items()):
        if condition_filter and condition not in condition_filter:
            continue
        if fd_filter and round(float(fd_step), 10) not in fd_filter:
            continue
        selected.append(cache)
    return selected


def _cache_counts(cache: dict[str, Any], *, bin_seconds: float, max_timebins: int) -> dict[str, np.ndarray]:
    return {
        THETA_PLUS: _stack_counts(cache["plus_rates"], bin_seconds=bin_seconds, max_timebins=max_timebins),
        THETA_MINUS: _stack_counts(cache["minus_rates"], bin_seconds=bin_seconds, max_timebins=max_timebins),
    }


def _truncate_label_tables(
    tables: list[dict[str, np.ndarray] | None],
) -> tuple[list[dict[str, np.ndarray] | None], int]:
    finite_tables = [table for table in tables if table is not None]
    if not finite_tables:
        raise ValueError("At least one trajectory table is required")
    t = min(arr.shape[1] if arr.ndim == 3 else arr.shape[0] for table in finite_tables for arr in table.values())
    out: list[dict[str, np.ndarray] | None] = []
    for table in tables:
        if table is None:
            out.append(None)
        else:
            out.append({label: arr[:, :t] if arr.ndim == 3 else arr[:t] for label, arr in table.items()})
    return out, int(t)


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else source_dir
    caches = _load_rate_caches(source_dir)
    selected = _select_caches(caches, conditions=args.conditions, fd_steps=args.fd_steps_arcmin)
    if not selected:
        raise ValueError("No selected rate caches matched the requested conditions/fd steps")
    prior_conditions = list(args.prior_conditions)

    trial_rows: list[dict[str, Any]] = []
    for cache in selected:
        condition = str(cache["condition"])
        fd_step = float(cache["fd_step_arcmin"])
        inference_mode = str(cache["inference_mode"])
        observed_counts = _cache_counts(
            cache,
            bin_seconds=float(args.bin_seconds),
            max_timebins=int(args.max_timebins),
        )
        zero_key = (str(args.reference_condition), fd_step, inference_mode)
        zero_counts = None
        zero_ref_available = zero_key in caches
        if not zero_ref_available and not bool(args.allow_missing_reference):
            raise FileNotFoundError(
                f"Missing zero-eye reference cache for condition={args.reference_condition!r}, "
                f"fd_step={fd_step:g}, inference_mode={inference_mode!r}"
            )
        if zero_ref_available:
            zero_counts = _mean_reference_counts(
                caches[zero_key],
                bin_seconds=float(args.bin_seconds),
                max_timebins=int(args.max_timebins),
                target_timebins=observed_counts[THETA_PLUS].shape[1],
            )

        effective_prior_conditions = prior_conditions or [condition]
        for prior_condition in effective_prior_conditions:
            prior_key = (str(prior_condition), fd_step, inference_mode)
            if prior_key not in caches:
                raise FileNotFoundError(
                    f"Missing prior-condition cache for condition={prior_condition!r}, "
                    f"fd_step={fd_step:g}, inference_mode={inference_mode!r}"
                )
            prior_cache = caches[prior_key]
            prior_counts = _cache_counts(
                prior_cache,
                bin_seconds=float(args.bin_seconds),
                max_timebins=int(args.max_timebins),
            )
            (obs_table, prior_table, zero_table), t = _truncate_label_tables([observed_counts, prior_counts, zero_counts])

            for trace_idx in range(obs_table[THETA_PLUS].shape[0]):
                for true_label in THETA_LABELS:
                    observed = obs_table[true_label][trace_idx]
                    result = score_trajectory_table_vernier_observer_trial(
                        observed,
                        true_label,
                        prior_table,
                        true_trace_index=trace_idx,
                        known_counts_by_theta=obs_table,
                        zero_counts_by_theta=zero_table,
                        include_self=bool(args.include_self),
                        phi=float(args.phi),
                        likelihood_normalization=str(args.likelihood_normalization),
                        likelihood_scale=float(args.likelihood_scale),
                    )
                    trial_rows.append(
                        {
                            "condition": condition,
                            "prior_condition": str(prior_condition),
                            "fd_step_arcmin": fd_step,
                            "inference_mode": inference_mode,
                            "trace_index": trace_idx,
                            "n_timebins": int(t),
                            "n_units": int(observed.shape[1]),
                            "source_cache": str(cache["path"]),
                            "prior_cache": str(prior_cache["path"]),
                            "zero_eye_reference_condition": str(args.reference_condition),
                            "zero_eye_reference_available": bool(zero_ref_available),
                            **result,
                        }
                    )

    summary_rows = summarize_trajectory_table_rows(trial_rows)
    write_csv(out_dir / "trajectory_table_observer_trials.csv", trial_rows)
    write_csv(out_dir / "trajectory_table_observer_summary.csv", summary_rows)
    write_json(
        out_dir / "trajectory_table_observer_manifest.json",
        {
            "source_dir": source_dir,
            "out_dir": out_dir,
            "conditions": args.conditions,
            "prior_conditions": args.prior_conditions,
            "fd_steps_arcmin": args.fd_steps_arcmin,
            "reference_condition": args.reference_condition,
            "include_self": bool(args.include_self),
            "leave_one_out": not bool(args.include_self),
            "likelihood_normalization": str(args.likelihood_normalization),
            "joint_score_family": table_score_family(str(args.likelihood_normalization)),
            "likelihood_scale": float(args.likelihood_scale),
            "phi": float(args.phi),
            "bin_seconds": float(args.bin_seconds),
            "max_timebins": int(args.max_timebins),
            "n_trial_rows": len(trial_rows),
            "n_summary_rows": len(summary_rows),
            "observer_interpretation": (
                "Vernier likelihood ratio with empirical trajectory nuisance marginalization"
                if table_score_family(str(args.likelihood_normalization)) == "poisson_log_likelihood"
                else "Gaussian Vernier likelihood-ratio diagnostic over empirical trajectory nuisance catalog"
                if table_score_family(str(args.likelihood_normalization)) == "gaussian_log_likelihood"
                else "Residual-energy diagnostic over empirical trajectory nuisance catalog"
            ),
            "implementation_provenance": "Implemented independently from specification; no GPL-covered source code copied.",
        },
    )
    return trial_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--conditions", type=str, default="")
    parser.add_argument("--prior-conditions", type=str, default="")
    parser.add_argument("--fd-steps-arcmin", type=str, default="")
    parser.add_argument("--reference-condition", type=str, default="static_center")
    parser.add_argument("--include-self", dest="include_self", action="store_true", default=True)
    parser.add_argument("--leave-one-out", dest="include_self", action="store_false")
    parser.add_argument("--allow-missing-reference", action="store_true")
    parser.add_argument("--likelihood-normalization", choices=SUPPORTED_TABLE_LIKELIHOODS, default="poisson")
    parser.add_argument("--likelihood-scale", type=float, default=1.0)
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--max-timebins", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.conditions = parse_csv_str(args.conditions)
    args.prior_conditions = parse_csv_str(args.prior_conditions)
    args.fd_steps_arcmin = parse_csv_float(args.fd_steps_arcmin)
    trial_rows, summary_rows = run(args)
    print(
        f"Wrote {len(trial_rows)} trajectory-table trials and {len(summary_rows)} summary rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
