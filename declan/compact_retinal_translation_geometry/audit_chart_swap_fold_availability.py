#!/usr/bin/env python3
"""Audit chart-swap fold availability without computing finite-difference charts."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from declan.compact_retinal_translation_geometry.run_correct_chart_swap_alignment import (
    build_chart_pair_dataset,
    write_csv,
    _condition_keys,
    _image_ids_for_samples,
    _sample_drift_mask,
)
from declan.compact_retinal_translation_geometry.run_relative_displacement_decoding import (
    _trial_pair_keys,
    _trial_set,
    parse_int_list,
    parse_str_list,
)
from declan.matched_twin_covariance_closure.run_cache_closure import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_FIG3_CACHE,
    _fig2_by_session,
    _load_pickle,
)
from declan.matched_twin_covariance_closure.run_finite_difference_closure import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL_CONFIG,
    _collect_samples,
    _load_twin_model,
    _target_for_session,
)


DEFAULT_OUTPUT_ROOT = Path("outputs") / "compact_retinal_translation_geometry" / "chart_swap_fold_availability_audit"


@dataclass
class FoldCandidate:
    mode: str
    n_folds: int
    fold: int
    held_trials: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    status: str


def _condition_folds(ids: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return []
    rng = np.random.default_rng(int(seed))
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    return [fold.astype(np.int64) for fold in np.array_split(shuffled, min(int(n_folds), ids.size)) if fold.size]


def _drift_trial_candidates(pairs: dict[str, np.ndarray], n_folds: int, seed: int) -> list[FoldCandidate]:
    drift_idx = np.flatnonzero(np.asarray(pairs["drift_mask"], dtype=bool))
    trial_a = np.asarray(pairs["trial_a"], dtype=np.int64)
    trial_b = np.asarray(pairs["trial_b"], dtype=np.int64)
    out: list[FoldCandidate] = []
    for fold_i, fold_pair_indices in enumerate(_condition_folds(drift_idx, int(n_folds), int(seed))):
        held = np.unique(np.concatenate([trial_a[fold_pair_indices], trial_b[fold_pair_indices]])).astype(np.int64)
        test_mask = np.zeros(trial_a.size, dtype=bool)
        test_mask[fold_pair_indices] = True
        train_mask = (~np.isin(trial_a, held)) & (~np.isin(trial_b, held))
        status = "pass" if int(np.sum(train_mask)) >= 5 and int(np.sum(test_mask)) >= 1 else "reject"
        out.append(FoldCandidate("drift_trial_disjoint", int(n_folds), int(fold_i), held, train_mask, test_mask, status))
    return out


def _trial_disjoint_candidates(pairs: dict[str, np.ndarray], n_folds: int, seed: int) -> list[FoldCandidate]:
    trial_a = np.asarray(pairs["trial_a"], dtype=np.int64)
    trial_b = np.asarray(pairs["trial_b"], dtype=np.int64)
    trials = np.unique(np.concatenate([trial_a, trial_b])).astype(np.int64)
    out: list[FoldCandidate] = []
    for fold_i, held in enumerate(_condition_folds(trials, int(n_folds), int(seed))):
        a_test = np.isin(trial_a, held)
        b_test = np.isin(trial_b, held)
        test_mask = a_test & b_test
        train_mask = (~a_test) & (~b_test)
        status = "pass" if int(np.sum(train_mask)) >= 5 and int(np.sum(test_mask)) >= 3 else "reject"
        out.append(FoldCandidate("trial_disjoint", int(n_folds), int(fold_i), held, train_mask, test_mask, status))
    return out


def _drift_pair_holdout_candidates(pairs: dict[str, np.ndarray], n_folds: int, seed: int) -> list[FoldCandidate]:
    drift_idx = np.flatnonzero(np.asarray(pairs["drift_mask"], dtype=bool))
    n_pairs = int(np.asarray(pairs["trial_a"]).size)
    out: list[FoldCandidate] = []
    for fold_i, fold_pair_indices in enumerate(_condition_folds(drift_idx, int(n_folds), int(seed))):
        test_mask = np.zeros(n_pairs, dtype=bool)
        test_mask[fold_pair_indices] = True
        train_mask = ~test_mask
        status = "pass" if int(np.sum(train_mask)) >= 5 and int(np.sum(test_mask)) >= 1 else "reject"
        out.append(
            FoldCandidate(
                "drift_pair_holdout_not_trial_disjoint",
                int(n_folds),
                int(fold_i),
                np.zeros(0, dtype=np.int64),
                train_mask,
                test_mask,
                status,
            )
        )
    return out


def _wrong_condition_count(
    *,
    cond: int,
    true_image: int,
    true_time: int,
    chart_conditions: set[int],
    condition_meta: dict[int, tuple[int, int]],
    pool: str,
) -> int:
    n = 0
    for other in chart_conditions:
        if int(other) == int(cond):
            continue
        img, tt = condition_meta.get(int(other), (-999, -999))
        if pool == "same_time_different_image" and int(tt) != int(true_time):
            continue
        if pool == "same_image_wrong_time" and (int(img) != int(true_image) or int(tt) == int(true_time)):
            continue
        if pool == "different_image" and int(img) == int(true_image):
            continue
        n += 1
    return int(n)


def _fold_row(
    *,
    session: str,
    subject: str,
    candidate: FoldCandidate,
    pairs: dict[str, np.ndarray],
    labels: np.ndarray,
    samples: Any,
    condition_meta: dict[int, tuple[int, int]],
    min_train_samples_per_chart: int,
    wrong_chart_pool: str,
) -> dict[str, Any]:
    train_mask = np.asarray(candidate.train_mask, dtype=bool)
    test_mask = np.asarray(candidate.test_mask, dtype=bool)
    trial_a = np.asarray(pairs["trial_a"], dtype=np.int64)
    trial_b = np.asarray(pairs["trial_b"], dtype=np.int64)
    if candidate.mode.endswith("not_trial_disjoint"):
        train_sample_mask = np.ones(np.asarray(labels).shape, dtype=bool)
    else:
        train_sample_mask = ~np.isin(np.asarray(samples.trial_ids, dtype=np.int64), candidate.held_trials)
    chart_counts: dict[int, int] = {}
    for cond in np.unique(np.asarray(labels)[(np.asarray(labels) >= 0) & train_sample_mask]):
        chart_counts[int(cond)] = int(np.sum((np.asarray(labels) == int(cond)) & train_sample_mask))
    chart_conditions = {int(c) for c, n in chart_counts.items() if int(n) >= int(min_train_samples_per_chart)}

    test_idx = np.flatnonzero(test_mask)
    true_available = 0
    wrong_available = 0
    scoreable = 0
    wrong_counts: list[int] = []
    for idx in test_idx:
        cond = int(pairs["condition_id"][idx])
        img = int(pairs["image_id"][idx])
        tt = int(pairs["time_context"][idx])
        has_true = int(chart_counts.get(cond, 0)) >= int(min_train_samples_per_chart)
        n_wrong = _wrong_condition_count(
            cond=cond,
            true_image=img,
            true_time=tt,
            chart_conditions=chart_conditions,
            condition_meta=condition_meta,
            pool=str(wrong_chart_pool),
        )
        true_available += int(has_true)
        wrong_available += int(n_wrong > 0)
        scoreable += int(has_true and n_wrong > 0)
        wrong_counts.append(int(n_wrong))

    train_trials = _trial_set(trial_a, trial_b, train_mask)
    test_trials = _trial_set(trial_a, trial_b, test_mask)
    train_pair_keys = _trial_pair_keys(trial_a, trial_b, train_mask)
    test_pair_keys = _trial_pair_keys(trial_a, trial_b, test_mask)
    held_fraction = (
        float(candidate.held_trials.size / max(np.unique(np.concatenate([trial_a, trial_b])).size, 1))
        if candidate.held_trials.size
        else 0.0
    )
    reject_reasons: list[str] = []
    if int(np.sum(train_mask)) < 5:
        reject_reasons.append("too_few_train_pairs")
    if int(np.sum(test_mask)) < (3 if candidate.mode == "trial_disjoint" else 1):
        reject_reasons.append("too_few_test_pairs")
    if test_idx.size and scoreable == 0:
        reject_reasons.append("no_approx_scoreable_test_pairs")
    return {
        "session": session,
        "subject": subject,
        "fold_mode": candidate.mode,
        "n_folds": int(candidate.n_folds),
        "fold": int(candidate.fold),
        "status": candidate.status,
        "reject_reasons": ",".join(reject_reasons),
        "n_total_pairs": int(trial_a.size),
        "n_drift_pairs": int(np.sum(np.asarray(pairs["drift_mask"], dtype=bool))),
        "n_train_pairs": int(np.sum(train_mask)),
        "n_test_pairs": int(np.sum(test_mask)),
        "n_held_trials": int(candidate.held_trials.size),
        "held_trial_fraction": held_fraction,
        "n_train_trials": int(len(train_trials)),
        "n_test_trials": int(len(test_trials)),
        "n_shared_trials": int(len(train_trials.intersection(test_trials))),
        "n_shared_trial_pairs": int(len(train_pair_keys.intersection(test_pair_keys))),
        "n_train_conditions": int(np.unique(np.asarray(pairs["condition_id"])[train_mask]).size) if np.any(train_mask) else 0,
        "n_test_conditions": int(np.unique(np.asarray(pairs["condition_id"])[test_mask]).size) if np.any(test_mask) else 0,
        "n_chart_conditions": int(len(chart_conditions)),
        "n_test_pairs_true_chart_available": int(true_available),
        "n_test_pairs_wrong_chart_available": int(wrong_available),
        "n_test_pairs_approx_scoreable": int(scoreable),
        "test_pair_scoreable_fraction": float(scoreable / max(test_idx.size, 1)) if test_idx.size else 0.0,
        "wrong_chart_pool": str(wrong_chart_pool),
        "wrong_conditions_per_test_pair_min": int(min(wrong_counts)) if wrong_counts else 0,
        "wrong_conditions_per_test_pair_median": float(np.median(wrong_counts)) if wrong_counts else 0.0,
        "wrong_conditions_per_test_pair_max": int(max(wrong_counts)) if wrong_counts else 0,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fig3-cache", type=Path, default=DEFAULT_FIG3_CACHE)
    p.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--sessions", type=str, default="Logan_2020-01-07")
    p.add_argument("--window-idx", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pixels-per-degree-fallback", type=float, default=37.5)
    p.add_argument("--fixation-radius-deg", type=float, default=1.0)
    p.add_argument("--sample-dfs-mode", choices=["all", "any", "none"], default="all")
    p.add_argument("--context-mode", choices=["time_bin", "time_window", "image_only"], default="time_window")
    p.add_argument("--context-bin-size", type=int, default=10)
    p.add_argument("--min-repeats-per-condition", type=int, default=3)
    p.add_argument("--max-pairs-per-condition", type=int, default=100)
    p.add_argument("--min-train-samples-per-chart", type=int, default=2)
    p.add_argument("--wrong-chart-pool", choices=["any", "different_image", "same_time_different_image", "same_image_wrong_time"], default="different_image")
    p.add_argument("--drift-speed-threshold-px", type=float, default=2.0)
    p.add_argument("--drift-pair-delta-threshold-px", type=float, default=5.0)
    p.add_argument("--fold-list", type=str, default="2,3,5,10,20,50,100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose-model-load", action="store_true")
    return p


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    fig3_rows = _load_pickle(Path(args.fig3_cache))
    fig2_rows = _load_pickle(Path(args.fig2_cache))
    fig2 = _fig2_by_session(fig2_rows)
    fig3_by_session = {str(row["session"]): row for row in fig3_rows}
    sessions = parse_str_list(args.sessions)
    fold_list = parse_int_list(args.fold_list)
    model, _ = _load_twin_model(args)

    session_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    mode_summary_rows: list[dict[str, Any]] = []

    for session in sessions:
        if session not in fig3_by_session or session not in fig2:
            session_rows.append({"session": session, "status": "missing_fig2_or_fig3"})
            continue
        if session not in getattr(model, "names", []):
            session_rows.append({"session": session, "status": "missing_model_session"})
            continue
        dataset_idx = int(model.names.index(session))
        sr = fig3_by_session[session]
        common_units, _, _, target_meta = _target_for_session(fig2[session], sr, args)
        dset, _, samples = _collect_samples(model=model, dataset_idx=dataset_idx, common_units=common_units, args=args)
        eye_px = samples.eyepos_deg * float(samples.pixels_per_degree)
        image_ids, image_meta = _image_ids_for_samples(dset, samples)
        labels, sample_image_ids, time_contexts, condition_rows = _condition_keys(
            image_ids=image_ids,
            time_indices=samples.time_indices,
            mode=str(args.context_mode),
            bin_size=int(args.context_bin_size),
        )
        condition_meta = {int(r["condition_id"]): (int(r["image_id"]), int(r["time_context"])) for r in condition_rows}
        sample_drift_mask, sample_speed = _sample_drift_mask(
            dset=dset,
            samples=samples,
            pixels_per_degree=float(samples.pixels_per_degree),
            speed_threshold_px=float(args.drift_speed_threshold_px),
        )
        pairs, inventory = build_chart_pair_dataset(
            samples=samples,
            eye_px=eye_px,
            labels=labels,
            image_ids=sample_image_ids,
            time_contexts=time_contexts,
            sample_drift_mask=sample_drift_mask,
            drift_pair_delta_threshold_px=float(args.drift_pair_delta_threshold_px),
            min_repeats_per_condition=int(args.min_repeats_per_condition),
            max_pairs_per_condition=int(args.max_pairs_per_condition),
            seed=int(args.seed) + dataset_idx * 101,
        )
        for row in inventory:
            row.update({"session": session, "subject": sr.get("subject", "")})
        inventory_rows.extend(inventory)

        session_rows.append(
            {
                "session": session,
                "subject": sr.get("subject", ""),
                "status": "ok",
                "dataset_idx": int(dataset_idx),
                "n_common_units": int(common_units.size),
                "n_samples": int(samples.source_indices.size),
                "n_candidate_samples": int(samples.n_candidate_samples),
                "n_unique_trials": int(np.unique(samples.trial_ids).size),
                "n_pair_conditions": int(np.unique(pairs["condition_id"]).size) if pairs["condition_id"].size else 0,
                "n_pairs": int(pairs["delta_y"].shape[0]),
                "n_drift_pairs": int(np.sum(pairs["drift_mask"])),
                "sample_speed_px_p90": float(np.nanpercentile(sample_speed, 90)) if np.any(np.isfinite(sample_speed)) else float("nan"),
                **target_meta,
                **image_meta,
            }
        )

        candidates: list[FoldCandidate] = []
        for n_folds in fold_list:
            candidates.extend(_drift_trial_candidates(pairs, int(n_folds), int(args.seed) + dataset_idx * 1009))
            candidates.extend(_trial_disjoint_candidates(pairs, int(n_folds), int(args.seed) + dataset_idx * 1009))
            candidates.extend(_drift_pair_holdout_candidates(pairs, int(n_folds), int(args.seed) + dataset_idx * 1009))
        start = len(fold_rows)
        for cand in candidates:
            fold_rows.append(
                _fold_row(
                    session=session,
                    subject=str(sr.get("subject", "")),
                    candidate=cand,
                    pairs=pairs,
                    labels=labels,
                    samples=samples,
                    condition_meta=condition_meta,
                    min_train_samples_per_chart=int(args.min_train_samples_per_chart),
                    wrong_chart_pool=str(args.wrong_chart_pool),
                )
            )
        by_mode: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in fold_rows[start:]:
            by_mode.setdefault((str(row["fold_mode"]), int(row["n_folds"])), []).append(row)
        for (mode, n_folds), rows in sorted(by_mode.items()):
            valid = [r for r in rows if r["status"] == "pass"]
            scoreable = [r for r in rows if int(r["n_test_pairs_approx_scoreable"]) > 0]
            mode_summary_rows.append(
                {
                    "session": session,
                    "subject": sr.get("subject", ""),
                    "fold_mode": mode,
                    "n_folds": int(n_folds),
                    "n_candidate_folds": int(len(rows)),
                    "n_valid_folds_by_pair_threshold": int(len(valid)),
                    "n_folds_with_scoreable_test_pairs": int(len(scoreable)),
                    "total_test_pairs": int(sum(int(r["n_test_pairs"]) for r in rows)),
                    "total_train_pairs_valid_folds": int(sum(int(r["n_train_pairs"]) for r in valid)),
                    "total_approx_scoreable_test_pairs": int(sum(int(r["n_test_pairs_approx_scoreable"]) for r in rows)),
                    "median_held_trial_fraction": float(np.median([float(r["held_trial_fraction"]) for r in rows])) if rows else float("nan"),
                    "median_train_pairs": float(np.median([int(r["n_train_pairs"]) for r in rows])) if rows else float("nan"),
                    "median_test_pairs": float(np.median([int(r["n_test_pairs"]) for r in rows])) if rows else float("nan"),
                }
            )

    write_csv(out / "session_inventory.csv", session_rows)
    write_csv(out / "condition_pair_inventory.csv", inventory_rows)
    write_csv(out / "fold_candidate_audit.csv", fold_rows)
    write_csv(out / "fold_mode_summary.csv", mode_summary_rows)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
