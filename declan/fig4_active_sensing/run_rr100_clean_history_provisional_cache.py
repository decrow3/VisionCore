#!/usr/bin/env python3
"""Resume the RR100 provisional cache while excluding pre-fixation histories.

This runner deliberately targets the quarantined 100-image x 577-trace subset
and never writes a response for any trace flagged by ``CACHE_WARNING.json``.
It reuses the frozen inputs, baselines, and clean responses already computed by
the original production runner.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from declan.fig4_active_sensing.run_rr100_corrected_production_cache import (
    DEFAULT_OUT,
    N_HISTORY,
    N_SCORE,
    N_UNITS,
    SUMMARY_ARRAYS,
    atomic_json,
    atomic_npz,
    baseline_path,
    baseline_valid,
    render_scored_embedding,
    response_timecourses,
    summarize_timecourses,
)
from declan.fig4_active_sensing.run_rr100_corrected_ssi_map_first_smoke import MAPPING
from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
)
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--half-index", choices=("0", "1", "all"), default="0")
    parser.add_argument("--round-start", type=int, default=0)
    parser.add_argument("--round-stop", type=int, default=0)
    parser.add_argument("--max-new-blocks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_valid(path: Path, expected: np.ndarray, policy: str, request_sha256: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            n = len(expected)
            return (
                str(data["request_sha256"].item()) == request_sha256
                and np.array_equal(data["trace_index"], expected.astype(np.int64))
                and data["rate_timecourse_hz"].shape == (n, N_SCORE, N_UNITS)
                and data["instantaneous_ssi_bits_per_spike"].shape == (n, N_SCORE, N_UNITS)
                and all(data[name].shape == (n, N_UNITS) for name in SUMMARY_ARRAYS)
                and (
                    "history_qc_policy_sha256" not in data
                    or str(data["history_qc_policy_sha256"].item()) == policy
                )
            )
    except Exception:
        return False


def write_progress(out: Path, *, status: str, rounds: list[int], policy: str) -> None:
    schedule = pd.read_csv(out / "balanced_round_schedule.csv")
    bad = set(
        pd.read_csv(out / "quality_control/pre_fixation_history_trace_flags.csv")
        .query("not history_within_selected_fixation")
        .trace_index.astype(int)
    )
    complete: list[int] = []
    completed_conditions = 0
    completed_blocks = 0
    request_sha = json.loads((out / "request_identity.json").read_text())["request_sha256"]
    for round_index, rows in schedule.groupby("round_index", sort=True):
        ok = True
        for image_index, block in rows.groupby("image_index", sort=True):
            expected = block.loc[
                ~block.trace_index.astype(int).isin(bad)
            ].sort_values("within_block").trace_index.to_numpy(np.int64)
            path = out / "moving" / f"round_{int(round_index):03d}" / f"image_{int(image_index):03d}.npz"
            valid = path.exists() and clean_valid(path, expected, policy, request_sha)
            ok &= valid
            if valid:
                completed_blocks += 1
                completed_conditions += len(expected)
        if ok and rows.image_index.nunique() == 100:
            complete.append(int(round_index))
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "warning": "PROVISIONAL CLEAN-HISTORY 100x577 SUBSET; NOT FINAL 100x1000 PRODUCTION CACHE",
        "history_qc_policy_sha256": policy,
        "active_selected_rounds": rounds,
        "complete_clean_history_rounds": complete,
        "completed_clean_atomic_blocks": completed_blocks,
        "completed_clean_conditions": completed_conditions,
        "target_conditions_all_rounds": int(100 * 577),
        "trace_flags": str((out / "quality_control/pre_fixation_history_trace_flags.csv").resolve()),
        "quarantine_root": str((out / "quarantine/pre_fixation_history_contaminated_original_blocks").resolve()),
    }
    atomic_json(out / "clean_history_manifest.json", payload)


def main() -> None:
    args = parse_args()
    out = args.out_dir.resolve()
    warning_path = out / "CACHE_WARNING.json"
    if not warning_path.exists():
        raise FileNotFoundError("Run quarantine_rr100_pre_fixation_history.py --apply first")
    warning = json.loads(warning_path.read_text())
    policy = str(warning["policy_sha256"])
    bad = set(map(int, warning["must_exclude_trace_indices"]))
    if len(bad) != 423:
        raise RuntimeError("Expected exactly 423 quarantined trace identities")
    schedule = pd.read_csv(out / "balanced_round_schedule.csv")
    clean_schedule = schedule.loc[~schedule.trace_index.astype(int).isin(bad)]
    if clean_schedule.trace_index.nunique() != 577:
        raise RuntimeError("Schedule does not produce the expected 577-trace clean subset")
    all_rounds = sorted(map(int, schedule.round_index.unique()))
    half = len(all_rounds) // 2
    rounds = all_rounds[:half] if args.half_index == "0" else all_rounds[half:]
    if args.half_index == "all":
        rounds = all_rounds
    upper = int(args.round_stop) if int(args.round_stop) > 0 else len(all_rounds)
    rounds = [r for r in rounds if int(args.round_start) <= r < upper]
    if not rounds:
        raise ValueError("No rounds selected")
    request_sha = json.loads((out / "request_identity.json").read_text())["request_sha256"]
    trace_table = pd.read_csv(out.parent / "rr100_corrected100x1000_production_cohort_v1/corrected1000_traces.csv")
    trace_ids_all = trace_table.sort_values("trace_index").trace_index.to_numpy(int)
    trace_ordinal = {int(value): ordinal for ordinal, value in enumerate(trace_ids_all)}
    with np.load(out / "input_cache/corrected_trace_segments.npz", allow_pickle=False) as data:
        history = np.asarray(data["history_xy_deg"], dtype=np.float32)
        score = np.asarray(data["score_xy_deg"], dtype=np.float32)
        if not np.array_equal(data["trace_index"], trace_ids_all):
            raise RuntimeError("Frozen trace ordering mismatch")
    image_rows = pd.read_csv(out.parent / "rr100_corrected100x1000_production_cohort_v1/corrected100_images.csv")
    patches: dict[int, np.ndarray] = {}
    ppds: dict[int, float] = {}
    for image_index in image_rows.image_index.astype(int):
        with np.load(out / "input_cache/images" / f"image_{image_index:03d}.npz", allow_pickle=False) as data:
            patches[image_index] = np.asarray(data["corrected_patch"], dtype=np.float32)
            ppds[image_index] = float(data["patch_ppd"].item())

    pending = 0
    for round_index in rounds:
        rows = schedule[schedule.round_index.eq(round_index)]
        for image_index, block in rows.groupby("image_index", sort=True):
            expected = block.loc[
                ~block.trace_index.astype(int).isin(bad)
            ].sort_values("within_block").trace_index.to_numpy(np.int64)
            path = out / "moving" / f"round_{round_index:03d}" / f"image_{int(image_index):03d}.npz"
            if not (path.exists() and clean_valid(path, expected, policy, request_sha)):
                pending += 1
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
        "selected_rounds": list(map(int, rounds)),
            "pending_blocks": pending,
            "target_traces_per_round": 577,
            "excluded_trace_count": len(bad),
            "policy_sha256": policy,
        }, indent=2))
        return

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(MAPPING).sort_values("rr100_index")
    if int(view.n_units) != N_UNITS or not np.array_equal(
        np.argmax(view.membership, axis=1), mapping.canonical_channel.to_numpy(int)
    ):
        raise ValueError("RR100 population mapping mismatch")
    scorer = CanonicalTwinScorer(
        device=str(args.device), batch_size=int(args.frame_batch_size), empty_cache_every_batch=True
    )
    started = time.perf_counter()
    new_blocks = 0
    write_progress(out, status="clean_history_scoring_in_progress", rounds=rounds, policy=policy)
    for round_index in rounds:
        rows = schedule[schedule.round_index.eq(round_index)]
        for image_index, block in rows.groupby("image_index", sort=True):
            image_index = int(image_index)
            expected = block.loc[
                ~block.trace_index.astype(int).isin(bad)
            ].sort_values("within_block").trace_index.to_numpy(np.int64)
            destination = out / "moving" / f"round_{round_index:03d}" / f"image_{image_index:03d}.npz"
            if destination.exists():
                if clean_valid(destination, expected, policy, request_sha):
                    continue
                raise RuntimeError(f"Existing shard violates clean-history contract: {destination}")
            base_file = baseline_path(out, image_index)
            if not baseline_valid(base_file, image_index=image_index, request_sha256=request_sha):
                raise RuntimeError(f"Missing or invalid original stabilized baseline: {base_file}")
            with np.load(base_file, allow_pickle=False) as data:
                baseline_rate = np.asarray(data["rate_timecourse_hz"], dtype=np.float32)
            rates = np.empty((len(expected), N_SCORE, N_UNITS), dtype=np.float32)
            ssis = np.empty_like(rates)
            summaries = {name: np.empty((len(expected), N_UNITS), np.float32) for name in SUMMARY_ARRAYS}
            for local, trace_index in enumerate(expected):
                if int(trace_index) in bad:
                    raise AssertionError("Quarantined trace reached scoring loop")
                ordinal = trace_ordinal[int(trace_index)]
                trace72 = np.concatenate([history[ordinal], score[ordinal]], axis=0)
                stim = render_scored_embedding(scorer.common, scorer.torch, patches[image_index], trace72, ppds[image_index])
                rate, ssi = response_timecourses(scorer, view, stim)
                summary = summarize_timecourses(rate, ssi, baseline_rate)
                rates[local] = rate
                ssis[local] = ssi
                for name in SUMMARY_ARRAYS:
                    summaries[name][local] = summary[name]
            atomic_npz(
                destination,
                request_sha256=np.asarray(request_sha),
                round_index=np.asarray(round_index, dtype=np.int64),
                half_index=np.asarray(int(round_index >= len(all_rounds) // 2), dtype=np.int64),
                image_index=np.asarray(image_index, dtype=np.int64),
                trace_index=expected,
                rate_timecourse_hz=rates,
                instantaneous_ssi_bits_per_spike=ssis,
                history_qc_policy_sha256=np.asarray(policy),
                cache_subset_status=np.asarray("provisional_within_fixation_history_only"),
                **summaries,
            )
            new_blocks += 1
            print(
                f"round {round_index:03d} image {image_index:03d}: wrote {len(expected)} clean movies; "
                f"new_blocks={new_blocks}; elapsed={(time.perf_counter()-started)/60:.1f} min",
                flush=True,
            )
            write_progress(out, status="clean_history_scoring_in_progress", rounds=rounds, policy=policy)
            if int(args.max_new_blocks) > 0 and new_blocks >= int(args.max_new_blocks):
                write_progress(out, status="stopped_at_requested_clean_block_limit", rounds=rounds, policy=policy)
                return
    write_progress(out, status="selected_clean_history_rounds_complete", rounds=rounds, policy=policy)
    del scorer, view, patches, history, score
    gc.collect()


if __name__ == "__main__":
    main()
