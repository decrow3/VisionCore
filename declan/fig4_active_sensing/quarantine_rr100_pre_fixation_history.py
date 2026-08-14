#!/usr/bin/env python3
"""Quarantine pre-fixation-history rows in the provisional RR100 cache.

This is a recoverable triage operation: every contaminated source NPZ is moved
intact beneath ``quarantine/`` before a clean-row-only derivative is installed
at its former path.  The quarantine tree must not be used for analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
DEFAULT_AUDIT = (
    ROOT
    / "outputs/fig4_active_sensing/rr100_pre_fixation_history_boundary_audit_v1"
    / "trace_history_boundary_audit.csv"
)
ROW_ARRAYS = (
    "trace_index",
    "rate_timecourse_hz",
    "instantaneous_ssi_bits_per_spike",
    "information_numerator_bits_spikes",
    "expected_spikes",
    "mean_rate_hz",
    "movie_ssi_bits_per_spike",
    "temporal_sd_rate_hz",
    "temporal_rms_delta_from_stabilized_hz",
    "temporal_mean_abs_delta_from_stabilized_hz",
)
WARNING_NAME = "WARNING_PRE_FIXATION_HISTORY.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--apply", action="store_true", help="Apply the recoverable quarantine operation")
    return parser.parse_args()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def policy_digest(flags: pd.DataFrame) -> str:
    payload = flags.sort_values("trace_index").to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    cache = args.cache_dir.resolve()
    audit = pd.read_csv(args.audit_csv.resolve()).sort_values("trace_index")
    required = {
        "trace_index",
        "history_within_selected_fixation",
        "history_frames_before_fixation",
        "history_contains_detected_event",
        "history_detected_event_frames",
    }
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"Audit table lacks required columns: {sorted(missing)}")
    if len(audit) != 1000 or audit.trace_index.nunique() != 1000:
        raise ValueError("Expected exactly 1,000 unique audited traces")

    flags = audit[
        [
            "trace_index",
            "session",
            "trial_idx",
            "corrected_history_global_start",
            "corrected_scored_global_start",
            "fixation_global_start",
            "history_frames_before_fixation",
            "history_within_selected_fixation",
            "history_contains_detected_event",
            "history_detected_event_frames",
        ]
    ].copy()
    flags["cache_eligibility"] = np.where(
        flags.history_within_selected_fixation.astype(bool),
        "clean_within_fixation_history",
        "quarantined_pre_fixation_history",
    )
    flags["quarantine_reason"] = np.where(
        flags.history_within_selected_fixation.astype(bool),
        "",
        "recorded model history begins before selected fixation onset",
    )
    digest = policy_digest(flags)
    bad = set(
        flags.loc[~flags.history_within_selected_fixation.astype(bool), "trace_index"].astype(int)
    )
    good = set(flags.trace_index.astype(int)).difference(bad)
    if len(bad) != 423 or len(good) != 577:
        raise RuntimeError(f"Unexpected clean/bad split: {len(good)}/{len(bad)}")

    moving = cache / "moving"
    quarantine = cache / "quarantine" / "pre_fixation_history_contaminated_original_blocks"
    files = sorted(moving.glob("round_*/image_*.npz"))
    contaminated_files = 0
    affected_rows = 0
    retained_rows = 0
    already_sanitized = 0
    report_rows: list[dict[str, object]] = []
    for path in files:
        relative = path.relative_to(moving)
        quarantine_path = quarantine / relative
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        trace_ids = arrays["trace_index"].astype(int)
        if "history_qc_policy_sha256" in arrays:
            if str(arrays["history_qc_policy_sha256"].item()) != digest:
                raise RuntimeError(f"Different history QC policy already applied: {path}")
            if any(int(value) in bad for value in trace_ids):
                raise RuntimeError(f"Sanitized shard still contains a quarantined trace: {path}")
            already_sanitized += 1
            continue
        mask = np.asarray([int(value) in good for value in trace_ids], dtype=bool)
        n_bad = int((~mask).sum())
        if n_bad == 0:
            continue
        contaminated_files += 1
        affected_rows += n_bad
        retained_rows += int(mask.sum())
        report_rows.append(
            {
                "moving_path": str(path),
                "quarantine_path": str(quarantine_path),
                "n_original_rows": int(len(trace_ids)),
                "n_quarantined_rows": n_bad,
                "n_retained_rows": int(mask.sum()),
                "quarantined_trace_indices": ";".join(map(str, trace_ids[~mask].tolist())),
                "retained_trace_indices": ";".join(map(str, trace_ids[mask].tolist())),
            }
        )
        if not args.apply:
            continue
        if quarantine_path.exists():
            raise FileExistsError(f"Refusing to overwrite prior quarantine source: {quarantine_path}")
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, quarantine_path)
        for name in ROW_ARRAYS:
            arrays[name] = arrays[name][mask]
        arrays["history_qc_policy_sha256"] = np.asarray(digest)
        arrays["cache_subset_status"] = np.asarray("provisional_within_fixation_history_only")
        atomic_npz(path, arrays)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "applied" if args.apply else "dry_run",
        "policy_sha256": digest,
        "n_clean_traces": len(good),
        "n_quarantined_traces": len(bad),
        "n_existing_moving_files": len(files),
        "n_contaminated_existing_files": contaminated_files,
        "n_already_sanitized_files": already_sanitized,
        "n_existing_rows_quarantined": affected_rows,
        "n_existing_rows_retained_in_sanitized_derivatives": retained_rows,
        "quarantine_is_recoverable": True,
    }
    print(json.dumps(summary, indent=2))
    if not args.apply:
        return

    qc = cache / "quality_control"
    qc.mkdir(parents=True, exist_ok=True)
    flags.to_csv(qc / "pre_fixation_history_trace_flags.csv", index=False)
    pd.DataFrame(report_rows).to_csv(qc / "quarantined_existing_blocks.csv", index=False)
    schedule = pd.read_csv(cache / "balanced_round_schedule.csv")
    clean_schedule = schedule[schedule.trace_index.astype(int).isin(good)].copy()
    clean_schedule.to_csv(qc / "balanced_round_schedule_clean_history.csv", index=False)
    atomic_text(qc / "pre_fixation_history_quarantine_manifest.json", json.dumps(summary, indent=2) + "\n")
    warning = f"""# WARNING: provisional clean-history subset only

This cache is **not** the final corrected 100 image x 1,000 trace production cache.

An audit found that 423/1,000 trace histories begin before the selected fixation.
Those trace identities are flagged in `quality_control/pre_fixation_history_trace_flags.csv`.
Original NPZ shards containing them were moved intact beneath
`quarantine/pre_fixation_history_contaminated_original_blocks/`; do not analyze that tree.
Replacement shards under `moving/` contain only the valid rows, and resumed scoring skips
all 423 flagged traces.  The usable provisional target is therefore 100 images x 577 traces.

Policy SHA-256: `{digest}`

Qualitative quartile slopes were stable when restricted to these clean histories, but a
fresh, fully stratified 100 x 1,000 production cohort is still required later.
"""
    atomic_text(cache / WARNING_NAME, warning)
    warning_payload = {
        **summary,
        "warning": "PROVISIONAL 100x577 WITHIN-FIXATION-HISTORY SUBSET; NOT FINAL 100x1000 CACHE",
        "must_exclude_trace_indices": sorted(bad),
        "trace_flags": str((qc / "pre_fixation_history_trace_flags.csv").resolve()),
        "clean_schedule": str((qc / "balanced_round_schedule_clean_history.csv").resolve()),
        "quarantine_root": str(quarantine.resolve()),
    }
    atomic_text(cache / "CACHE_WARNING.json", json.dumps(warning_payload, indent=2) + "\n")
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status_before_history_quarantine"] = manifest.get("status")
    manifest["status"] = "quarantined_clean_history_provisional_cache"
    manifest["warning"] = warning_payload["warning"]
    manifest["history_qc_policy_sha256"] = digest
    manifest["n_usable_trace_identities"] = len(good)
    manifest["n_quarantined_trace_identities"] = len(bad)
    manifest["cache_warning"] = str((cache / "CACHE_WARNING.json").resolve())
    manifest["clean_history_progress_manifest"] = str((cache / "clean_history_manifest.json").resolve())
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
