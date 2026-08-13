#!/usr/bin/env python3
"""Resumable native-readout production sweep for zero-gaze RR100 SF/TF tuning.

This promotes the validated native smoke-test path to the full signed-TF,
orientation, spatial-frequency, and carrier-phase grid.  Every session is
checkpointed independently so an interrupted multi-hour GPU run can resume
without discarding completed conditions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import torch

from DataYatesV1.exp.gratings import GratingsTrial
from DataYatesV1.utils.io import get_session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.run_rr100_original_sequence_dense_sf_native_readout import (  # noqa: E402
    RR100_VERSION,
    load_rr100_rows,
)
from declan.run_rr100_zero_gaze_separable_sf_tf_input_checkpoint import (  # noqa: E402
    FRAME_RATE_HZ,
    HISTORY_FRAMES,
    derive_zero_gaze_roi,
)
from declan.run_rr100_zero_gaze_separable_sf_tf_native_smoke import (  # noqa: E402
    build_movie,
    embed_native_history,
    predict_native,
    resolve_device,
    summarize_condition,
)

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.utils import get_model_and_dataset_configs  # noqa: E402


DEFAULT_OUT = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_native_production_v1"
)
DEFAULT_SFS = "1,1.4142135623730951,2,2.8284271247461903,4,5.656854249492381,8,11.313708498984761,16"
DEFAULT_TFS = "0.5,0.7071067811865476,1,1.4142135623730951,2,2.8284271247461903,4,5.656854249492381,8,11.313708498984761,16,22.627416997969522,32,45.254833995939045"
DEFAULT_ORIENTATIONS = "0,45,90,135"
DT = 1.0 / FRAME_RATE_HZ


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--population-version", default=RR100_VERSION)
    parser.add_argument("--sessions", default="", help="Comma-separated subset; empty means all RR100 sessions")
    parser.add_argument("--spatial-cpds", default=DEFAULT_SFS)
    parser.add_argument("--temporal-hz-magnitudes", default=DEFAULT_TFS)
    parser.add_argument("--orientation-deg", default=DEFAULT_ORIENTATIONS)
    parser.add_argument("--static-phases", type=int, default=4)
    parser.add_argument("--low-tf-phases", type=int, default=2)
    parser.add_argument("--high-tf-phases", type=int, default=1)
    parser.add_argument("--low-tf-cutoff-hz", type=float, default=4.0)
    parser.add_argument("--base-duration-s", type=float, default=2.0)
    parser.add_argument("--minimum-dynamic-cycles", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=0, help="Testing only; zero means all")
    parser.add_argument("--max-conditions", type=int, default=0, help="Testing only; zero means all")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def condition_table(args: argparse.Namespace) -> pd.DataFrame:
    spatial_cpds = parse_float_list(args.spatial_cpds)
    temporal_magnitudes = parse_float_list(args.temporal_hz_magnitudes)
    orientations = parse_float_list(args.orientation_deg)
    if any(tf <= 0 for tf in temporal_magnitudes):
        raise ValueError("Temporal magnitudes must be positive; static TF=0 is added separately")
    rows: list[dict[str, Any]] = []

    def append_condition(
        kind: str,
        sf: float,
        signed_tf: float,
        orientation: float,
        phase_index: int,
        n_phases: int,
        duration_s: float,
    ) -> None:
        n_valid_frames = int(math.ceil(duration_s * FRAME_RATE_HZ))
        actual_duration_s = n_valid_frames / FRAME_RATE_HZ
        abs_tf = abs(float(signed_tf))
        rows.append(
            {
                "condition_id": len(rows),
                "condition_kind": kind,
                "spatial_cpd": float(sf),
                "signed_temporal_hz": float(signed_tf),
                "temporal_hz_magnitude": abs_tf,
                "orientation_deg": float(orientation),
                "phase_index": int(phase_index),
                "n_phases": int(n_phases),
                "phase_rad": float(2.0 * math.pi * phase_index / max(n_phases, 1)),
                "n_valid_response_frames": n_valid_frames,
                "valid_response_duration_s": actual_duration_s,
                "dynamic_cycles_observed": abs_tf * actual_duration_s,
                "spatial_edge_control": bool(np.isclose(sf, 16.0)),
                "temporal_edge_control": bool(abs_tf > 32.0),
                "primary_fit_support": bool(sf <= 11.3137086 and abs_tf <= 32.0),
            }
        )

    append_condition("gray_blank", 0.0, 0.0, 0.0, 0, 1, float(args.base_duration_s))
    for orientation in orientations:
        for sf in spatial_cpds:
            for phase_index in range(int(args.static_phases)):
                append_condition(
                    "static_grating",
                    sf,
                    0.0,
                    orientation,
                    phase_index,
                    int(args.static_phases),
                    float(args.base_duration_s),
                )
            for magnitude in temporal_magnitudes:
                duration_s = max(
                    float(args.base_duration_s),
                    float(args.minimum_dynamic_cycles) / float(magnitude),
                )
                n_phases = int(args.low_tf_phases if magnitude < args.low_tf_cutoff_hz else args.high_tf_phases)
                for direction in (-1.0, 1.0):
                    for phase_index in range(n_phases):
                        append_condition(
                            "drifting_grating",
                            sf,
                            direction * magnitude,
                            orientation,
                            phase_index,
                            n_phases,
                            duration_s,
                        )
    conditions = pd.DataFrame(rows)
    if conditions["condition_id"].duplicated().any():
        raise AssertionError("Condition identifiers are not unique")
    return conditions


def request_identity(args: argparse.Namespace, sessions: list[str], conditions: pd.DataFrame) -> dict[str, Any]:
    contract = {
        "analysis": "rr100_zero_gaze_separable_sf_tf_native_production",
        "population_version": str(args.population_version),
        "sessions": sessions,
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "frame_rate_hz": FRAME_RATE_HZ,
        "history_frames": HISTORY_FRAMES,
        "lag_order": "current,t-1,...,t-32",
        "stimulus_normalization": "(uint8-127)/255",
        "zero_gaze_rule": "session-specific ROI center at eyepos=(0,0)",
        "conditions": conditions.to_dict(orient="records"),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "contract": contract}


def save_progress(rows: list[dict[str, Any]], path: Path) -> None:
    if rows:
        frame = pd.DataFrame(rows).sort_values(["condition_id", "rr100_index"])
        frame = frame.drop_duplicates(["condition_id", "rr100_index"], keep="last")
        atomic_csv(frame, path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session_root = args.out_dir / "sessions"
    session_root.mkdir(parents=True, exist_ok=True)

    mapping_rows, _, spec_json, spec_npz = load_rr100_rows(str(args.population_version))
    mapping = pd.DataFrame(
        [
            {
                "rr100_index": int(row["rep_idx"]),
                "session": str(row["selected_session"]),
                "source_unit_index": int(row["selected_source_unit_index"]),
                "canonical_channel": int(row["selected_channel"]),
            }
            for row in mapping_rows
        ]
    )
    all_sessions = list(dict.fromkeys(mapping["session"].tolist()))
    requested = [part.strip() for part in str(args.sessions).split(",") if part.strip()]
    sessions = requested or all_sessions
    missing = sorted(set(sessions) - set(all_sessions))
    if missing:
        raise ValueError(f"Requested sessions do not contribute RR100 units: {missing}")
    if args.max_sessions > 0:
        sessions = sessions[: int(args.max_sessions)]

    conditions = condition_table(args)
    if args.max_conditions > 0:
        conditions = conditions.iloc[: int(args.max_conditions)].copy()
    identity = request_identity(args, sessions, conditions)
    identity_path = args.out_dir / "request_identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing.get("sha256") != identity["sha256"]:
            raise RuntimeError(
                f"Existing output has request identity {existing.get('sha256')}; requested {identity['sha256']}"
            )
    else:
        atomic_json(identity, identity_path)
    atomic_csv(conditions, args.out_dir / "condition_table.csv")
    atomic_csv(mapping[mapping["session"].isin(sessions)], args.out_dir / "rr100_unit_mapping.csv")

    incomplete_sessions: list[str] = []
    pending_by_session: dict[str, list[int]] = {}
    for session in sessions:
        session_dir = session_root / session
        summary_path = session_dir / "condition_unit_summary.csv"
        complete: set[int] = set()
        if summary_path.exists():
            previous = pd.read_csv(summary_path)
            expected_units = int(mapping["session"].eq(session).sum())
            counts = previous.groupby("condition_id")["rr100_index"].nunique()
            complete = set(counts[counts.eq(expected_units)].index.astype(int).tolist())
            if not (session_dir / "unused_behavior_audit.csv").exists():
                complete.discard(0)
        pending = [int(v) for v in conditions.loc[~conditions["condition_id"].isin(complete), "condition_id"]]
        pending_by_session[session] = pending
        if pending:
            incomplete_sessions.append(session)

    if incomplete_sessions:
        device = resolve_device(str(args.device))
        load_start = time.perf_counter()
        model, _ = get_model_and_dataset_configs(mode="standard")
        model = model.to(device)
        model.model.eval()
        model_load_seconds = time.perf_counter() - load_start
    else:
        device = str(args.device)
        model = None
        model_load_seconds = 0.0

    invocation_start = time.perf_counter()
    runtime_rows: list[dict[str, Any]] = []
    for session_number, session in enumerate(sessions, start=1):
        pending = pending_by_session[session]
        if not pending:
            print(f"[{session_number}/{len(sessions)}] {session}: already complete", flush=True)
            continue
        session_start = time.perf_counter()
        session_dir = session_root / session
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "condition_unit_summary.csv"
        rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []

        subject, date = session.split("_", maxsplit=1)
        sess = get_session(subject, date)
        dset = sess.get_dataset("gratings", strict=True)
        roi, roi_row = derive_zero_gaze_roi(dset, session)
        atomic_csv(pd.DataFrame([roi_row]), session_dir / "zero_gaze_roi_audit.csv")
        trial_index = int(np.asarray(dset["trial_inds"])[0])
        trial = GratingsTrial(sess.exp["D"][trial_index], sess.exp["S"])
        unit_map = mapping[mapping["session"].eq(session)].sort_values("rr100_index")
        rr100_indices = unit_map["rr100_index"].to_numpy(dtype=np.int64)
        source_indices = unit_map["source_unit_index"].to_numpy(dtype=np.int64)
        dataset_idx = list(model.names).index(session)
        print(
            f"[{session_number}/{len(sessions)}] {session}: {len(rr100_indices)} RR100 units, "
            f"{len(pending)}/{len(conditions)} pending conditions on {device}",
            flush=True,
        )

        processed = 0
        for condition_id in pending:
            condition = conditions.loc[conditions["condition_id"].eq(condition_id)].iloc[0]
            n_valid_frames = int(condition["n_valid_response_frames"])
            movie = build_movie(condition, trial, roi, n_valid_frames)
            stimulus = embed_native_history(movie)
            evaluate_start = time.perf_counter()
            response = predict_native(
                model,
                stimulus,
                dataset_idx,
                source_indices,
                device=device,
                batch_size=int(args.batch_size),
            )
            elapsed = time.perf_counter() - evaluate_start
            if condition_id == 0:
                audit_samples = min(len(stimulus), int(args.batch_size))
                response_zeros = predict_native(
                    model,
                    stimulus[:audit_samples],
                    dataset_idx,
                    source_indices,
                    device=device,
                    batch_size=int(args.batch_size),
                    behavior_mode="zeros",
                )
                difference = np.abs(response[:audit_samples] - response_zeros)
                audit = pd.DataFrame(
                    [
                        {
                            "session": session,
                            "n_samples": audit_samples,
                            "maximum_abs_count_difference_none_vs_zero_behavior": float(np.max(difference)),
                            "all_exact": bool(not np.any(difference)),
                        }
                    ]
                )
                if not bool(audit["all_exact"].iloc[0]):
                    raise AssertionError(f"{session}: prediction changed with explicit zero behavior")
                atomic_csv(audit, session_dir / "unused_behavior_audit.csv")
            rows.extend(summarize_condition(condition, response, rr100_indices, session, elapsed))
            processed += 1
            print(
                f"  [{processed}/{len(pending)}; id={condition_id}] sf={condition['spatial_cpd']:g} "
                f"tf={condition['signed_temporal_hz']:+g} ori={condition['orientation_deg']:g} "
                f"phase={int(condition['phase_index'])} frames={n_valid_frames} {elapsed:.2f}s",
                flush=True,
            )
            if processed % max(int(args.checkpoint_every), 1) == 0:
                save_progress(rows, summary_path)
        save_progress(rows, summary_path)
        runtime_rows.append(
            {
                "session": session,
                "conditions_evaluated_this_invocation": processed,
                "conditions_total": len(conditions),
                "wall_seconds_this_invocation": time.perf_counter() - session_start,
            }
        )
        atomic_csv(pd.DataFrame(runtime_rows), args.out_dir / "invocation_session_runtimes.csv")
        del dset, sess, trial, rows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completed_frames: list[pd.DataFrame] = []
    session_complete: dict[str, bool] = {}
    for session in sessions:
        path = session_root / session / "condition_unit_summary.csv"
        if not path.exists():
            session_complete[session] = False
            continue
        frame = pd.read_csv(path)
        expected_units = int(mapping["session"].eq(session).sum())
        counts = frame.groupby("condition_id")["rr100_index"].nunique()
        session_complete[session] = bool(
            len(counts) == len(conditions) and counts.eq(expected_units).all()
        )
        completed_frames.append(frame)

    status = "production_complete" if session_complete and all(session_complete.values()) else "production_partial_resumable"
    if completed_frames:
        summary = pd.concat(completed_frames, ignore_index=True)
        summary = summary.drop_duplicates(["session", "condition_id", "rr100_index"], keep="last")
        blank = summary[summary["condition_kind"].eq("gray_blank")][
            ["session", "rr100_index", "mean_rate_hz"]
        ].rename(columns={"mean_rate_hz": "blank_rate_hz"})
        summary = summary.merge(blank, on=["session", "rr100_index"], how="left", validate="many_to_one")
        summary["mean_rate_above_blank_hz"] = summary["mean_rate_hz"] - summary["blank_rate_hz"]
        atomic_csv(summary, args.out_dir / "native_condition_unit_summary.csv")

    manifest = {
        "analysis": "rr100_zero_gaze_separable_sf_tf_native_production",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "request_sha256": identity["sha256"],
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "model_load_seconds": model_load_seconds,
        "invocation_wall_seconds": time.perf_counter() - invocation_start,
        "population_version": str(args.population_version),
        "population_spec_json": str(spec_json.resolve()),
        "population_spec_npz": str(spec_npz.resolve()),
        "n_sessions": len(sessions),
        "n_rr100_units": int(mapping[mapping["session"].isin(sessions)]["rr100_index"].nunique()),
        "n_conditions_per_session": len(conditions),
        "session_complete": session_complete,
        "resumability": "session condition summaries are atomically checkpointed and complete condition IDs are skipped",
        "raw_trace_policy": "population sweep stores absolute rate summaries; selected raw traces are rerendered in the unit drill-down",
        "edge_policy": "16 cpd and 45.254834 Hz retained as diagnostic controls and excluded from primary_fit_support",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "artifacts": {
            "request_identity": "request_identity.json",
            "condition_table": "condition_table.csv",
            "unit_mapping": "rr100_unit_mapping.csv",
            "aggregate_summary": "native_condition_unit_summary.csv",
            "session_checkpoints": "sessions/<session>/condition_unit_summary.csv",
        },
    }
    atomic_json(manifest, args.out_dir / "analysis_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
