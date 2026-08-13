#!/usr/bin/env python3
"""Merge completed Panel G exact-pair production shards without summarizing them."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_exact_pair_fig4_trace_bank_n1000_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cohort = pd.read_csv(run_dir / "exact_pair_cohort_manifest.csv")
    shard_dirs = sorted(
        path.parent for path in (run_dir / "shards").glob("pairs_*/run_metadata.json")
    )
    if not shard_dirs:
        raise RuntimeError(f"No completed production shards under {run_dir / 'shards'}")
    manifests = [pd.read_csv(path / "exact_pair_shard_manifest.csv") for path in shard_dirs]
    scored = pd.concat(manifests, ignore_index=True).sort_values("pair_index").reset_index(drop=True)
    if scored["pair_index"].duplicated().any():
        duplicates = scored.loc[scored["pair_index"].duplicated(), "pair_index"].astype(int).tolist()
        raise RuntimeError(f"Overlapping completed shards for pair indices: {duplicates[:20]}")
    expected = cohort["pair_index"].astype(int).to_numpy()
    observed = scored["pair_index"].astype(int).to_numpy()
    missing = [int(value) for value in sorted(set(expected) - set(observed))]
    unexpected = [int(value) for value in sorted(set(observed) - set(expected))]
    if unexpected:
        raise RuntimeError(f"Completed shards contain unexpected pair indices: {unexpected[:20]}")
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"Production run is incomplete: {len(missing)} of {len(expected)} pairs missing; "
            "pass --allow-incomplete only for a labeled partial merge"
        )
    population = pd.concat(
        [pd.read_csv(path / "direct_pair_population_metrics.csv") for path in shard_dirs],
        ignore_index=True,
    ).sort_values(["pair_index", "population", "condition_index"]).reset_index(drop=True)
    contrasts = pd.concat(
        [pd.read_csv(path / "direct_pair_rotation_contrasts.csv") for path in shard_dirs],
        ignore_index=True,
    ).sort_values(["pair_index", "population"]).reset_index(drop=True)
    arrays = [_load_npz(path / "direct_pair_unit_metrics.npz") for path in shard_dirs]
    for key in ("condition_id", "rotation_angle_deg"):
        reference = arrays[0][key]
        if any(not np.array_equal(reference, item[key], equal_nan=True) for item in arrays[1:]):
            raise RuntimeError(f"Shard condition mismatch for {key}")
    pair_index = np.concatenate([item["pair_index"] for item in arrays])
    order = np.argsort(pair_index, kind="stable")
    merged_dir = run_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        merged_dir / "direct_pair_unit_metrics.npz",
        pair_index=pair_index[order],
        condition_id=arrays[0]["condition_id"],
        rotation_angle_deg=arrays[0]["rotation_angle_deg"],
        unit_bits_per_spike=np.concatenate([item["unit_bits_per_spike"] for item in arrays], axis=0)[order],
        unit_information_bits=np.concatenate([item["unit_information_bits"] for item in arrays], axis=0)[order],
        unit_expected_spikes=np.concatenate([item["unit_expected_spikes"] for item in arrays], axis=0)[order],
        unit_mean_rate=np.concatenate([item["unit_mean_rate"] for item in arrays], axis=0)[order],
    )
    population.to_csv(merged_dir / "direct_pair_population_metrics.csv", index=False)
    contrasts.to_csv(merged_dir / "direct_pair_rotation_contrasts.csv", index=False)
    scored.to_csv(merged_dir / "merged_pair_manifest.csv", index=False)
    metadata = {
        "analysis": "panel_g_exact_native_pair_production_merge",
        "artifact_type": "complete_raw_merge" if not missing else "partial_raw_merge",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "shard_dirs": [str(path) for path in shard_dirs],
        "n_cohort_pairs": int(len(expected)),
        "n_merged_pairs": int(len(observed)),
        "missing_pair_indices": missing,
        "population_summary_performed": False,
        "contract": "Raw exact-pair shard merge only; no interpolation and no population inference",
    }
    (merged_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"[panel-g-merge] merged {len(observed)}/{len(expected)} pairs from {len(shard_dirs)} shards "
        f"into {merged_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
