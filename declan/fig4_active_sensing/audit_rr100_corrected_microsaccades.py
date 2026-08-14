#!/usr/bin/env python3
"""Detect microsaccades in corrected RR100 scored traces.

The event threshold is estimated from each complete trial's calibrated
``dpi_pix`` trajectory at global-even 120-Hz sampling. Event detection is then
applied to the corresponding 40-frame scored trace, which is the trajectory
used by the corrected response cache.
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import (
    corrected_crop_xy_deg,
    load_dset,
)
from declan.fixation_statistics_by_stimulus.extraction import _speed_threshold_mad_valid_pairs
from jake.twininfo.eye_controls import detect_microsaccade_events


ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
TRACE_CACHE = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "input_cache/corrected_trace_segments.npz"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_microsaccade_audit_v1"
MODEL_HZ = 120.0
THRESHOLD_Z = 6.0
PAD_SAMPLES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT / "corrected1000_traces.csv")
    parser.add_argument("--trace-cache", type=Path, default=TRACE_CACHE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    traces = pd.read_csv(args.cohort).sort_values("trace_index").reset_index(drop=True)
    with np.load(args.trace_cache, allow_pickle=False) as archive:
        frozen_ids = np.asarray(archive["trace_index"], dtype=int)
        score_xy_deg = np.asarray(archive["score_xy_deg"], dtype=float)
    if not np.array_equal(frozen_ids, traces.trace_index.to_numpy(dtype=int)):
        raise ValueError("Frozen trace order differs from corrected cohort")

    records: list[dict[str, float | int | str | bool]] = []
    for session, rows in traces.groupby("session", sort=True):
        dset = load_dset(str(session), {})
        crop_xy_deg = corrected_crop_xy_deg(dset)
        trial_index = np.asarray(dset.covariates["trial_inds"]).reshape(-1).astype(int)
        dpi_valid = np.asarray(dset.covariates["dpi_valid"]).reshape(-1).astype(bool)
        thresholds: dict[int, float] = {}
        for trial in rows.trial_idx.astype(int).unique():
            sample = np.flatnonzero(trial_index == int(trial))
            sample = sample[sample % 2 == 0]
            thresholds[int(trial)] = _speed_threshold_mad_valid_pairs(
                crop_xy_deg[sample], dpi_valid[sample], dt=1.0 / MODEL_HZ, z=THRESHOLD_Z
            )
        for row in rows.itertuples(index=False):
            trace_id = int(row.trace_index)
            threshold = thresholds[int(row.trial_idx)]
            events, event_mask, _ = detect_microsaccade_events(
                score_xy_deg[trace_id],
                dt=1.0 / MODEL_HZ,
                threshold_deg_s=threshold,
                min_samples=1,
                pad_samples=PAD_SAMPLES,
            )
            records.append(
                {
                    "trace_index": trace_id,
                    "session": str(session),
                    "trial_idx": int(row.trial_idx),
                    "scored_microsaccade_threshold_deg_s": float(threshold),
                    "scored_n_microsaccade_events": int(len(events)),
                    "scored_contains_microsaccade": bool(events),
                    "scored_microsaccade_frames": int(event_mask.sum()),
                    "scored_peak_microsaccade_speed_deg_s": max(
                        (float(event["peak_speed_deg_s"]) for event in events), default=0.0
                    ),
                }
            )
        del dset, crop_xy_deg, trial_index, dpi_valid
        gc.collect()

    labels = pd.DataFrame(records).sort_values("trace_index")
    labels.to_csv(args.out_dir / "corrected_scored_microsaccade_labels.csv", index=False)
    joined = traces[["trace_index", "corrected_dpi_crop120_path_length_arcmin"]].merge(
        labels, on="trace_index", validate="one_to_one"
    )
    path_summary = (
        joined.groupby("scored_contains_microsaccade")
        .corrected_dpi_crop120_path_length_arcmin.agg(["count", "min", "median", "mean", "max"])
        .reset_index()
    )
    path_summary.to_csv(args.out_dir / "path_length_by_microsaccade_context.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "detector": "jake.twininfo.eye_controls.detect_microsaccade_events",
        "threshold": "per complete trial, global-even calibrated dpi_pix at 120 Hz; median + 6 * 1.4826 * MAD speed",
        "classification": "event detection within the cached 40-frame scored trajectory",
        "pad_samples": PAD_SAMPLES,
        "n_traces": int(len(labels)),
        "n_drift_only": int((labels.scored_n_microsaccade_events == 0).sum()),
        "n_one_event": int((labels.scored_n_microsaccade_events == 1).sum()),
        "n_multiple_event": int((labels.scored_n_microsaccade_events > 1).sum()),
        "n_event_containing": int(labels.scored_contains_microsaccade.sum()),
        "outputs": {
            "labels": str((args.out_dir / "corrected_scored_microsaccade_labels.csv").resolve()),
            "path_summary": str((args.out_dir / "path_length_by_microsaccade_context.csv").resolve()),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
