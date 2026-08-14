#!/usr/bin/env python3
"""Freeze and audit the interruption-safe corrected 100 x 1,000 cache contract."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from declan.fig4_active_sensing.run_rr100_corrected_production_cache import (
    DEFAULT_COHORT,
    DEFAULT_OUT,
    PRODUCTION_BLOCK_SIZE,
    PRODUCTION_IMAGES,
    PRODUCTION_STATUS,
    PRODUCTION_TRACES,
    make_balanced_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_production_cache_contract_checkpoint_37_v1"
RUNNER = ROOT / "declan/fig4_active_sensing/run_rr100_corrected_production_cache.py"
ASSEMBLER = ROOT / "declan/fig4_active_sensing/assemble_rr100_corrected_production_cache.py"
TEST = ROOT / "declan/fig4_active_sensing/tests/test_corrected_production_cache_contract.py"


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest}


def connected(frame: pd.DataFrame, images: np.ndarray, traces: np.ndarray) -> bool:
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for row in frame.itertuples(index=False):
        image = ("image", int(row.image_index))
        trace = ("trace", int(row.trace_index))
        adjacency.setdefault(image, set()).add(trace)
        adjacency.setdefault(trace, set()).add(image)
    expected = {("image", int(value)) for value in images} | {("trace", int(value)) for value in traces}
    stack = [next(iter(expected))]
    seen: set[tuple[str, int]] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency[node].difference(seen))
    return seen == expected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    images = np.arange(PRODUCTION_IMAGES, dtype=int)
    traces = np.arange(PRODUCTION_TRACES, dtype=int)
    schedule = make_balanced_schedule(images, traces, block_size=PRODUCTION_BLOCK_SIZE)
    schedule.rename(
        columns={"image_index": "image_identity_template", "trace_index": "trace_identity_template"}
    ).to_csv(OUT / "production_ordinal_schedule_template.csv", index=False)
    schedule_for_check = schedule
    half_checks = {}
    for half_index in (0, 1):
        half = schedule_for_check[schedule_for_check.half_index.eq(half_index)]
        half_checks[str(half_index)] = {
            "n_pairs": int(len(half)),
            "image_degree_min": int(half.image_index.value_counts().min()),
            "image_degree_max": int(half.image_index.value_counts().max()),
            "trace_degree_min": int(half.trace_index.value_counts().min()),
            "trace_degree_max": int(half.trace_index.value_counts().max()),
            "connected": connected(half, images, traces),
        }
    cohort_manifest = DEFAULT_COHORT / "manifest.json"
    cohort_ready = False
    cohort_status = None
    if cohort_manifest.exists():
        payload = json.loads(cohort_manifest.read_text(encoding="utf-8"))
        cohort_status = payload.get("status")
        cohort_ready = cohort_status == PRODUCTION_STATUS
    cache_manifest = DEFAULT_OUT / "manifest.json"
    cache_status = None
    completed_blocks = 0
    completed_movies = 0
    if cache_manifest.exists():
        cache_payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
        cache_status = cache_payload.get("status")
        completed_blocks = int(cache_payload.get("completed_atomic_blocks", 0))
        completed_movies = int(cache_payload.get("completed_movies", 0))

    run_base = (
        f"{ROOT / '.venv/bin/python'} -m declan.fig4_active_sensing.run_rr100_corrected_production_cache "
        f"--cohort-dir {DEFAULT_COHORT} --out-dir {DEFAULT_OUT} --device cuda:0 "
        "--frame-batch-size 16 --trace-block-size 10"
    )
    commands = {
        "prepare_and_freeze_inputs": run_base + " --half-index 0 --prepare-only",
        "run_or_resume_half0": run_base + " --half-index 0",
        "run_or_resume_half1": run_base + " --half-index 1",
        "bounded_one_block_preflight": run_base + " --half-index 0 --max-new-blocks 1",
        "assemble_any_complete_rounds": (
            f"{ROOT / '.venv/bin/python'} -m declan.fig4_active_sensing.assemble_rr100_corrected_production_cache "
            f"--cache-dir {DEFAULT_OUT}"
        ),
        "assemble_independent_half": (
            f"{ROOT / '.venv/bin/python'} -m declan.fig4_active_sensing.assemble_rr100_corrected_production_cache "
            f"--cache-dir {DEFAULT_OUT} --require-complete-half"
        ),
    }
    for name, command in commands.items():
        (OUT / f"{name}.txt").write_text(command + "\n", encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "bounded_preflight_complete_production_not_launched"
            if cohort_ready and completed_blocks > 0
            else (
                "interruption_safe_cache_code_ready_cohort_ready_not_launched"
                if cohort_ready
                else "interruption_safe_cache_code_ready_waiting_for_final_cohort_not_launched"
            )
        ),
        "production_contract": {
            "n_images": PRODUCTION_IMAGES,
            "n_traces": PRODUCTION_TRACES,
            "n_pairs": PRODUCTION_IMAGES * PRODUCTION_TRACES,
            "n_rounds": 100,
            "movies_per_round": 1000,
            "atomic_movies_per_image_round_block": PRODUCTION_BLOCK_SIZE,
            "rounds_per_independent_half": 50,
            "movies_per_independent_half": 50000,
            "half_checks": half_checks,
        },
        "resume_contract": (
            "Each ten-movie image-within-round block is an atomic NPZ. Existing blocks must pass request, image, "
            "round, trace-order, timecourse-shape, and sufficient-statistic-shape checks before they are skipped."
        ),
        "partial_analysis_contract": (
            "One complete round is the smallest balanced interim cohort. It spans all 100 images and all 1,000 traces. "
            "Any completed set of rounds is balanced; either 50-round half is a connected independently analyzable half-bank."
        ),
        "saved_response_objects": {
            "all_movies": [
                "40-frame mean spatial rate timecourse for every RR100 unit",
                "40-frame instantaneous spatial SSI for every RR100 unit",
                "information numerator",
                "expected spikes",
                "mean rate",
                "movie SSI",
                "temporal response SD",
                "RMS and mean-absolute rate difference from stabilized",
            ],
            "per_image_baseline": "zero-relative-translation explicit-history response and timecourses",
            "full_spatial_maps": "not stored production-wide; render later for predeclared selected examples",
        },
        "cohort_gate": {
            "path": str(cohort_manifest.resolve()),
            "required_status": PRODUCTION_STATUS,
            "present": cohort_manifest.exists(),
            "current_status": cohort_status,
            "pass": cohort_ready,
        },
        "launch_status": (
            "bounded_preflight_complete_production_not_launched"
            if completed_blocks > 0
            else "not_launched"
        ),
        "cache_checkpoint": {
            "path": str(cache_manifest.resolve()),
            "present": cache_manifest.exists(),
            "status": cache_status,
            "completed_atomic_blocks": completed_blocks,
            "completed_movies": completed_movies,
        },
        "sources": {
            "runner": file_identity(RUNNER),
            "assembler": file_identity(ASSEMBLER),
            "contract_test": file_identity(TEST),
        },
        "commands": commands,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Corrected RR100 production-cache contract

Status: **{manifest['status']}**

The final cohort gate is {'complete' if cohort_ready else 'not complete'}.
The response cache currently contains {completed_blocks} completed atomic
block(s), or {completed_movies} movie(s). A bounded preflight is not a balanced
analysis cohort and must not be promoted as a neural result.

## Recovery granularity

- Atomic save: one image × ten traces = ten movies.
- Balanced partial-analysis checkpoint: one complete round = 1,000 movies,
  covering every image ten times and every trace once.
- Independent half-bank: 50 rounds = 50,000 movies, covering every image × 500
  traces and every trace × 50 images.
- Full bank: 100 rounds = 100,000 movies.

Rerunning the same command validates and skips completed atomic blocks. The
assembler promotes only complete balanced rounds by default.

## Remaining production-launch gates

1. Inspect the bounded preflight outputs and timing.
2. Explicitly approve resumable production scoring on the free GPU.
3. Assemble and analyze only complete balanced rounds.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
