#!/usr/bin/env python3
"""Freeze balanced, interruption-safe halves for the next RR100 GPU work."""
from __future__ import annotations

import hashlib
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TF_REQUEST = ROOT / "outputs/fig4_active_sensing/rr100_extended_tf_probe_manifest_checkpoint_33_v6"
BRIDGE = ROOT / "outputs/fig4_active_sensing/rr100_interim_bridge_selection_checkpoint_32_v2"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_gpu_interruption_safe_contract_checkpoint_36_v3"
TF_OUTPUT_BASE = ROOT / "outputs/active_sensing_movie_information"
RUNNER = ROOT / "declan/active_sensing_movie_information/run_backimage_rr100_dense_sf_tf_grating_probe.py"


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def replace_option(command: list[str], option: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(option)
    result[index + 1] = value
    return result


def bipartite_is_connected(frame: pd.DataFrame, left: str, right: str) -> bool:
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for row in frame[[left, right]].itertuples(index=False, name=None):
        left_node = ("left", int(row[0]))
        right_node = ("right", int(row[1]))
        adjacency.setdefault(left_node, set()).add(right_node)
        adjacency.setdefault(right_node, set()).add(left_node)
    if not adjacency:
        return False
    start = next(iter(adjacency))
    visited = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    return len(visited) == len(adjacency)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen contract: {OUT}")
    OUT.mkdir(parents=True)

    pairs = pd.read_csv(TF_REQUEST / "extended_tf_pair_table.csv")
    sf_values = sorted(pairs["spatial_cpd"].unique())
    tf_values = sorted(pairs["temporal_hz"].unique())
    sf_lookup = {value: index for index, value in enumerate(sf_values)}
    tf_lookup = {value: index for index, value in enumerate(tf_values)}
    pairs["sf_grid_index"] = pairs["spatial_cpd"].map(sf_lookup).astype(int)
    pairs["tf_grid_index"] = pairs["temporal_hz"].map(tf_lookup).astype(int)
    pairs["balanced_half"] = ((pairs["sf_grid_index"] + pairs["tf_grid_index"]) % 2).astype(int)
    tf_connectivity_swap = pairs["sf_grid_index"].lt(2) & pairs["tf_grid_index"].lt(2)
    pairs.loc[tf_connectivity_swap, "balanced_half"] = 1 - pairs.loc[tf_connectivity_swap, "balanced_half"]
    pairs["atomic_pair_complete_conditions"] = 4 * 2 * 4
    pairs["atomic_pair_complete_unit_rows"] = pairs["atomic_pair_complete_conditions"] * 100
    tf_assignments = OUT / "extended_tf_connected_balanced_halves.csv"
    pairs.to_csv(tf_assignments, index=False)

    images = pd.read_csv(BRIDGE / "bridge20_images.csv").sort_values("selection_order")
    traces = pd.read_csv(BRIDGE / "bridge50_traces.csv").sort_values("selection_order")
    bridge_rows = []
    for image in images.itertuples(index=False):
        for trace in traces.itertuples(index=False):
            bridge_rows.append(
                {
                    "image_index": int(image.image_index),
                    "image_selection_order": int(image.selection_order),
                    "image_selection_role": str(image.selection_role),
                    "trace_index": int(trace.trace_index),
                    "trace_selection_order": int(trace.selection_order),
                    "trace_selection_role": str(trace.selection_role),
                    "balanced_half": int(
                        (
                            (int(image.selection_order) + int(trace.selection_order)) % 2
                            + int(int(image.selection_order) < 2 and int(trace.selection_order) < 2)
                        )
                        % 2
                    ),
                    "atomic_bundle_contract": (
                        "all predeclared neural conditions and the stabilized baseline for this image-trace pair"
                    ),
                }
            )
    bridge_assignments = pd.DataFrame(bridge_rows)
    bridge_path = OUT / "bridge20x50_connected_balanced_halves.csv"
    bridge_assignments.to_csv(bridge_path, index=False)

    base_command = shlex.split((TF_REQUEST / "run_command.txt").read_text(encoding="utf-8").strip())
    command_records = {}
    for half in (0, 1):
        half_dir = TF_OUTPUT_BASE / f"backimage_rr100_extended_tf_f0_connected_half{half}_v1"
        command = replace_option(base_command, "--out-dir", str(half_dir))
        command.extend(["--pair-shard-count", "2", "--pair-shard-index", str(half)])
        dry_run = [*command, "--dry-run"]
        assemble = [*command, "--assemble-only"]
        (OUT / f"tf_half{half}_run_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        (OUT / f"tf_half{half}_dry_run_command.txt").write_text(shlex.join(dry_run) + "\n", encoding="utf-8")
        (OUT / f"tf_half{half}_assemble_partial_command.txt").write_text(
            shlex.join(assemble) + "\n", encoding="utf-8"
        )
        half_pairs = pairs[pairs["balanced_half"].eq(half)]
        command_records[f"half{half}"] = {
            "out_dir": str(half_dir.resolve()),
            "n_pairs": int(len(half_pairs)),
            "n_spatial_frequencies": int(half_pairs["spatial_cpd"].nunique()),
            "n_temporal_frequencies": int(half_pairs["temporal_hz"].nunique()),
            "grating_movies": int(len(half_pairs) * 4 * 2 * 4),
            "unit_rows": int(len(half_pairs) * 4 * 2 * 4 * 100),
            "run_command_file": str((OUT / f"tf_half{half}_run_command.txt").resolve()),
            "assemble_partial_command_file": str(
                (OUT / f"tf_half{half}_assemble_partial_command.txt").resolve()
            ),
        }

    tf_half_checks = []
    tf_dry_run_checks = []
    for half in (0, 1):
        subset = pairs[pairs["balanced_half"].eq(half)]
        tf_half_checks.append(
            len(subset) == 56
            and subset["spatial_cpd"].nunique() == 8
            and subset["temporal_hz"].nunique() == 14
            and subset.groupby("spatial_cpd").size().eq(7).all()
            and subset.groupby("temporal_hz").size().eq(4).all()
            and bipartite_is_connected(subset, "sf_grid_index", "tf_grid_index")
        )
        actual_path = TF_OUTPUT_BASE / f"backimage_rr100_extended_tf_f0_connected_half{half}_v1/dense_sf_tf_pair_table.csv"
        if actual_path.exists():
            actual = pd.read_csv(actual_path)
            expected = subset[["pair_id", "spatial_cpd", "temporal_hz"]].reset_index(drop=True)
            observed = actual[["pair_id", "spatial_cpd", "temporal_hz"]].reset_index(drop=True)
            tf_dry_run_checks.append(expected.equals(observed))
        else:
            tf_dry_run_checks.append(False)
    bridge_half_checks = []
    for half in (0, 1):
        subset = bridge_assignments[bridge_assignments["balanced_half"].eq(half)]
        bridge_half_checks.append(
            len(subset) == 500
            and subset["image_index"].nunique() == 20
            and subset["trace_index"].nunique() == 50
            and subset.groupby("image_index").size().eq(25).all()
            and subset.groupby("trace_index").size().eq(10).all()
            and bipartite_is_connected(subset, "image_selection_order", "trace_selection_order")
        )
    if not all(tf_half_checks + bridge_half_checks):
        raise RuntimeError("Balanced-half coverage validation failed")

    readme = """# Interruption-safe GPU contract

## Extended TF probe

The 112 SF x TF pairs begin with `(sf_grid_index + tf_grid_index) % 2`, then
the top-left 2 x 2 block is swapped between halves. This preserves all row and
column degrees while making both bipartite sampling graphs connected.
Each 56-pair half contains every one of the 8 SFs and all 14 TFs. Each completed
pair is written atomically only after all four orientations, two drift
directions, four phases, and 100 RR100 units are present. Rerunning a half skips
valid pair shards. `tf_half*_assemble_partial_command.txt` creates provisional
tables and plots from whatever complete pair shards exist without loading the
neural model.

A completed half supports provisional F0 distributions, TF marginals, and
sparse checkerboard SF x TF fits. It does not have the precision or local grid
coverage of the completed 112-pair probe.

## Corrected neural bridge

The 20 x 50 image-trace crossing uses the same parity split plus a 2 x 2
connectivity swap. In each 500-pair half,
every image is paired with 25 traces and every trace with 10 images. The bridge runner must
write one image-trace bundle atomically, with every requested history/baseline
condition together. One half is sufficient for balanced image and trace main
effects but not for a complete interaction matrix.

Checkpoint numbers are provenance labels, not a pre-existing framework.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "balanced_interruption_safe_gpu_contract_dry_run_validated_not_launched",
        "extended_tf": command_records,
        "bridge": {
            "n_total_pairs": 1000,
            "pairs_per_half": 500,
            "images_per_half": 20,
            "traces_per_half": 50,
            "traces_per_image_per_half": 25,
            "images_per_trace_per_half": 10,
            "runner_gate": (
                "bridge runner must consume bridge20x50_connected_balanced_halves.csv and atomically save "
                "all neural conditions for one image-trace bundle"
            ),
        },
        "validation": {
            "tf_halves_balanced": all(tf_half_checks),
            "tf_halves_connected": all(
                bipartite_is_connected(
                    pairs[pairs["balanced_half"].eq(half)], "sf_grid_index", "tf_grid_index"
                )
                for half in (0, 1)
            ),
            "tf_half_dry_runs_match_assignments": all(tf_dry_run_checks),
            "tf_halves_disjoint": not bool(
                set(pairs[pairs["balanced_half"].eq(0)]["pair_id"])
                & set(pairs[pairs["balanced_half"].eq(1)]["pair_id"])
            ),
            "bridge_halves_balanced": all(bridge_half_checks),
            "bridge_halves_connected": all(
                bipartite_is_connected(
                    bridge_assignments[bridge_assignments["balanced_half"].eq(half)],
                    "image_selection_order",
                    "trace_selection_order",
                )
                for half in (0, 1)
            ),
        },
        "sources": {
            "tf_pairs": file_identity(TF_REQUEST / "extended_tf_pair_table.csv"),
            "tf_command": file_identity(TF_REQUEST / "run_command.txt"),
            "tf_runner": file_identity(RUNNER),
            "bridge_images": file_identity(BRIDGE / "bridge20_images.csv"),
            "bridge_traces": file_identity(BRIDGE / "bridge50_traces.csv"),
        },
        "outputs": {
            "tf_assignments": file_identity(tf_assignments),
            "bridge_assignments": file_identity(bridge_path),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
