#!/usr/bin/env python3
"""Cross-image, input-only visual checkpoint for accepted 3-D IAAFT controls."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import torch

from declan.fig4_active_sensing.generate_rr100_map_support_stage2a_iaaft_accepted_ensemble_v2 import (
    FROZEN_THRESHOLDS,
    PHASE_SUPPORT_RELATIVE_THRESHOLD,
    acceptance_decision,
    target_energy_in_weak_phase_bins,
)
from declan.fig4_active_sensing.run_rr100_corrected_production_cache import render_scored_embedding
from declan.fig4_active_sensing.run_rr100_map_support_amplitude_phase_factorial_stage2a import (
    file_identity,
    load_development_input,
)
from declan.fig4_active_sensing.run_rr100_map_support_stage2a_distribution_constrained_phase_v1 import (
    CANONICAL_HIGH,
    CANONICAL_LOW,
    distribution_audit,
    iaaft_3d,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _load_twin_common


ROOT = Path(__file__).resolve().parents[2]
STAGE2 = ROOT / "outputs/fig4_active_sensing/rr100_clean_history_whole_movie_power_stage2_v1"
RESPONSE_CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
V2_SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_map_support_stage2a_iaaft_accepted_ensemble_v2"
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/rr100_map_support_stage2a_iaaft_cross_image_input_checkpoint_v3"
)
TRACE_INDEX = 121
SCORED_FRAME = 20
EXCLUDED_ALREADY_INSPECTED_IMAGE = 94
SELECTION_QUANTILES = (0.10, 0.35, 0.65, 0.90)
TARGET_ACCEPTED_PER_IMAGE_POWER = 1
MAX_ATTEMPTS_PER_IMAGE_POWER = 50
ITERATIONS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, default=STAGE2)
    parser.add_argument("--response-cache-dir", type=Path, default=RESPONSE_CACHE)
    parser.add_argument("--v2-source", type=Path, default=V2_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--max-attempts-per-image-power", type=int, default=MAX_ATTEMPTS_PER_IMAGE_POWER)
    parser.add_argument("--maximum-selected-images", type=int, default=len(SELECTION_QUANTILES))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def select_images(stage2_dir: Path, maximum: int) -> pd.DataFrame:
    condition = pd.read_csv(stage2_dir / "development_condition_index.csv")
    with np.load(stage2_dir / "development_predictors_and_first_split_predictions.npz", allow_pickle=False) as archive:
        rows = np.asarray(archive["matrix_row_index"], dtype=int)
        amplitude = np.asarray(archive["global_supported_dynamic_power_amplitude"], dtype=float)
    amplitude_by_row = dict(zip(rows.tolist(), amplitude.tolist(), strict=True))
    candidates = condition.loc[condition.trace_index.eq(TRACE_INDEX)].copy()
    candidates["global_supported_dynamic_power_amplitude"] = [
        amplitude_by_row[int(row)] for row in candidates.matrix_row_index
    ]
    full_values = candidates.global_supported_dynamic_power_amplitude.to_numpy(float)
    available = candidates.loc[~candidates.image_index.eq(EXCLUDED_ALREADY_INSPECTED_IMAGE)].copy()
    selected: list[pd.Series] = []
    used: set[int] = set()
    for quantile in SELECTION_QUANTILES[: int(maximum)]:
        target = float(np.quantile(full_values, quantile))
        work = available.loc[~available.image_index.astype(int).isin(used)].copy()
        index = (work.global_supported_dynamic_power_amplitude - target).abs().idxmin()
        row = work.loc[index].copy()
        row["selection_role"] = f"trace-fixed input-power quantile {quantile:.2f}"
        row["selection_criterion"] = (
            "closest global supported dynamic-power amplitude to the predeclared quantile among trace-121 "
            "development conditions, excluding already-inspected image 94"
        )
        row["selection_quantile"] = quantile
        row["selection_target_amplitude"] = target
        row["selection_absolute_error"] = abs(
            float(row.global_supported_dynamic_power_amplitude) - target
        )
        selected.append(row)
        used.add(int(row.image_index))
    table = pd.DataFrame(selected).reset_index(drop=True)
    split = pd.read_csv(stage2_dir / "frozen_image_and_trace_identity_split.csv")
    for image in table.image_index.astype(int):
        record = split.loc[split.identity_type.eq("image") & split.identity.eq(image)]
        if len(record) != 1 or record.iloc[0].split != "development":
            raise ValueError(f"Image {image} is not a declared development identity")
    trace = split.loc[split.identity_type.eq("trace") & split.identity.eq(TRACE_INDEX)]
    if len(trace) != 1 or trace.iloc[0].split != "development":
        raise ValueError(f"Trace {TRACE_INDEX} is not a declared development identity")
    return table


def render_image_cubes(selection: pd.Series, cache_dir: Path) -> dict[str, np.ndarray]:
    patch, ppd, trace72 = load_development_input(selection, cache_dir)
    common = _load_twin_common()
    fem = render_scored_embedding(common, torch, patch, trace72, ppd)
    stabilized = render_scored_embedding(common, torch, patch, np.zeros_like(trace72), ppd)
    cubes = {
        "FEM": fem[SCORED_FRAME, 0].detach().cpu().numpy().astype(np.float32),
        "stabilized": stabilized[SCORED_FRAME, 0].detach().cpu().numpy().astype(np.float32),
    }
    for power, cube in cubes.items():
        if cube.shape != (32, 151, 151):
            raise ValueError(f"Unexpected {power} cube shape {cube.shape}")
    return cubes


def seed_start(image_index: int, power: str) -> int:
    return 20263000 + int(image_index) * 200 + (1 if power == "FEM" else 101)


def find_accepted(
    image_index: int,
    power: str,
    target: np.ndarray,
    *,
    iterations: int,
    max_attempts: int,
) -> tuple[np.ndarray | None, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    convergence_rows: list[pd.DataFrame] = []
    weak_energy = target_energy_in_weak_phase_bins(target)
    accepted: np.ndarray | None = None
    for attempt in range(1, int(max_attempts) + 1):
        seed = seed_start(image_index, power) + attempt - 1
        candidate, convergence = iaaft_3d(target, seed, int(iterations))
        convergence["image_index"] = int(image_index)
        convergence["target_power"] = power
        convergence["attempt"] = attempt
        convergence_rows.append(convergence)
        metrics = distribution_audit(
            target,
            candidate,
            seed=seed,
            condition=f"image_{image_index:03d}_{power.lower()}_iaaft_candidate",
            source_kind="cross-image natural map-support cube",
        )
        metrics["image_index"] = int(image_index)
        metrics["trace_index"] = TRACE_INDEX
        metrics["scored_frame"] = SCORED_FRAME
        metrics["target_power"] = power
        metrics["attempt"] = attempt
        metrics["phase_support_relative_threshold"] = PHASE_SUPPORT_RELATIVE_THRESHOLD
        metrics["phase_support_target_energy_in_weak_bins"] = weak_energy
        passed, failed = acceptance_decision(metrics)
        metrics["accepted"] = bool(passed)
        metrics["failed_acceptance_checks"] = ";".join(failed)
        rows.append(metrics)
        print(
            f"[image {image_index} {power}] attempt={attempt} seed={seed} accepted={passed} "
            f"energy_phase={metrics['global_energy_weighted_phase_coherence']:.4f}",
            flush=True,
        )
        if passed:
            accepted = candidate
            break
    return accepted, pd.DataFrame(rows), pd.concat(convergence_rows, ignore_index=True)


def make_movie(
    image_index: int,
    originals: dict[str, np.ndarray],
    accepted: dict[str, np.ndarray],
    accepted_rows: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    pairs = []
    for power in ("FEM", "stabilized"):
        row = accepted_rows.loc[
            accepted_rows.image_index.eq(image_index) & accepted_rows.target_power.eq(power)
        ].iloc[0]
        pairs.append((originals[power], accepted[power], power, int(row.seed)))
    difference_limit = max(
        float(np.quantile(np.abs(candidate - original), 0.995))
        for original, candidate, _, _ in pairs
    )
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    images = []
    for row_index, (original, candidate, power, seed) in enumerate(pairs):
        images.extend([
            axes[row_index, 0].imshow(
                original[0], cmap="gray", vmin=CANONICAL_LOW, vmax=CANONICAL_HIGH, origin="lower"
            ),
            axes[row_index, 1].imshow(
                candidate[0], cmap="gray", vmin=CANONICAL_LOW, vmax=CANONICAL_HIGH, origin="lower"
            ),
            axes[row_index, 2].imshow(
                candidate[0] - original[0], cmap="RdBu_r", origin="lower",
                norm=TwoSlopeNorm(vmin=-difference_limit, vcenter=0, vmax=difference_limit),
            ),
        ])
        axes[row_index, 0].set_ylabel(f"{power}\nseed {seed}")
    for column, title in enumerate(("original", "accepted IAAFT", "accepted − original")):
        axes[0, column].set_title(title)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    title = figure.suptitle(f"Cross-image input checkpoint: image {image_index}; frame 1/32")

    def update(frame: int):
        index = 0
        for original, candidate, _, _ in pairs:
            for values in (original[frame], candidate[frame], candidate[frame] - original[frame]):
                images[index].set_data(values)
                index += 1
        title.set_text(f"Cross-image input checkpoint: image {image_index}; frame {frame + 1}/32")
        return [*images, title]

    writer = animation.FFMpegWriter(fps=12, bitrate=2200)
    animation.FuncAnimation(figure, update, frames=32, interval=80, blit=False).save(
        path, writer=writer, dpi=dpi
    )
    plt.close(figure)


def make_contact_sheet(
    selections: pd.DataFrame,
    originals_by_image: dict[int, dict[str, np.ndarray]],
    accepted_by_image: dict[int, dict[str, np.ndarray]],
    path: Path,
    dpi: int,
) -> None:
    frame = 15
    figure, axes = plt.subplots(len(selections), 4, figsize=(14, 3.5 * len(selections)), constrained_layout=True)
    if len(selections) == 1:
        axes = axes[None, :]
    columns = (
        ("FEM", "original FEM"),
        ("FEM", "accepted FEM IAAFT"),
        ("stabilized", "original stabilized"),
        ("stabilized", "accepted stabilized IAAFT"),
    )
    for row_index, selection in enumerate(selections.itertuples(index=False)):
        image_index = int(selection.image_index)
        for column_index, (power, title) in enumerate(columns):
            source = originals_by_image if "original" in title else accepted_by_image
            axes[row_index, column_index].imshow(
                source[image_index][power][frame], cmap="gray",
                vmin=CANONICAL_LOW, vmax=CANONICAL_HIGH, origin="lower",
            )
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
            if row_index == 0:
                axes[row_index, column_index].set_title(title)
        axes[row_index, 0].set_ylabel(
            f"image {image_index}\nq={selection.selection_quantile:.2f}\n"
            f"power={selection.global_supported_dynamic_power_amplitude:.0f}"
        )
    figure.suptitle(
        "Cross-image accepted IAAFT checkpoint; trace 121, scored frame 20, history frame 16",
        weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed checkpoint exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selections = select_images(args.stage2_dir, int(args.maximum_selected_images))
    selections.to_csv(args.out_dir / "selected_images_frozen_before_generation.csv", index=False)
    contract = {
        "created_utc_before_surrogate_generation": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_surrogate_generation",
        "selection_quantiles": list(SELECTION_QUANTILES[: int(args.maximum_selected_images)]),
        "selected_images": selections.image_index.astype(int).tolist(),
        "fixed_trace_index": TRACE_INDEX,
        "fixed_scored_frame": SCORED_FRAME,
        "excluded_already_inspected_image": EXCLUDED_ALREADY_INSPECTED_IMAGE,
        "target_accepted_per_image_power": TARGET_ACCEPTED_PER_IMAGE_POWER,
        "max_attempts_per_image_power": int(args.max_attempts_per_image_power),
        "iterations": int(args.iterations),
        "thresholds": FROZEN_THRESHOLDS,
        "neural_scoring": False,
        "runner": file_identity(Path(__file__)),
    }
    (args.out_dir / "cross_image_contract_frozen_before_generation.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )

    originals_by_image: dict[int, dict[str, np.ndarray]] = {}
    accepted_by_image: dict[int, dict[str, np.ndarray]] = {}
    audit_frames: list[pd.DataFrame] = []
    convergence_frames: list[pd.DataFrame] = []
    completed = True
    for selection in selections.itertuples(index=False):
        image_index = int(selection.image_index)
        originals = render_image_cubes(pd.Series(selection._asdict()), args.response_cache_dir)
        originals_by_image[image_index] = originals
        accepted_by_image[image_index] = {}
        for power in ("FEM", "stabilized"):
            candidate, audit, convergence = find_accepted(
                image_index,
                power,
                originals[power],
                iterations=int(args.iterations),
                max_attempts=int(args.max_attempts_per_image_power),
            )
            audit_frames.append(audit)
            convergence_frames.append(convergence)
            if candidate is None:
                completed = False
            else:
                accepted_by_image[image_index][power] = candidate

    audit = pd.concat(audit_frames, ignore_index=True)
    convergence = pd.concat(convergence_frames, ignore_index=True)
    audit.to_csv(args.out_dir / "all_cross_image_candidate_audit.csv", index=False)
    convergence.to_csv(args.out_dir / "all_cross_image_iaaft_convergence.csv", index=False)
    accepted_rows = audit.loc[audit.accepted].copy()
    accepted_rows.to_csv(args.out_dir / "accepted_cross_image_members.csv", index=False)
    audit.loc[~audit.accepted].to_csv(args.out_dir / "rejected_cross_image_candidates.csv", index=False)

    completed_images = [
        int(image)
        for image in selections.image_index
        if set(accepted_by_image[int(image)]) == {"FEM", "stabilized"}
    ]
    if completed:
        image_order = selections.image_index.astype(int).to_numpy()
        np.savez_compressed(
            args.out_dir / "accepted_cross_image_input_cubes.npz",
            image_index=image_order,
            trace_index=np.asarray(TRACE_INDEX),
            scored_frame=np.asarray(SCORED_FRAME),
            fem_original=np.stack([originals_by_image[int(i)]["FEM"] for i in image_order]),
            stabilized_original=np.stack([originals_by_image[int(i)]["stabilized"] for i in image_order]),
            fem_accepted=np.stack([accepted_by_image[int(i)]["FEM"] for i in image_order]),
            stabilized_accepted=np.stack([accepted_by_image[int(i)]["stabilized"] for i in image_order]),
        )
        make_contact_sheet(
            selections,
            originals_by_image,
            accepted_by_image,
            args.out_dir / "01_cross_image_contact_sheet",
            int(args.dpi),
        )
    for image_index in completed_images:
        make_movie(
            int(image_index),
            originals_by_image[int(image_index)],
            accepted_by_image[int(image_index)],
            accepted_rows,
            args.out_dir / f"02_image_{int(image_index):03d}_accepted_iaaft.mp4",
            int(args.dpi),
        )

    summary = []
    for (image_index, power), frame in audit.groupby(["image_index", "target_power"], sort=False):
        summary.append({
            "image_index": int(image_index),
            "target_power": power,
            "attempts": int(len(frame)),
            "accepted": bool(frame.accepted.any()),
            "acceptance_rate_through_first_success": float(frame.accepted.sum() / len(frame)),
            "accepted_seed": int(frame.loc[frame.accepted, "seed"].iloc[0]) if frame.accepted.any() else None,
        })
    summary_table = pd.DataFrame(summary)
    summary_table.to_csv(args.out_dir / "cross_image_acceptance_summary.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_map_support_stage2a_iaaft_cross_image_input_checkpoint_v3",
        "status": "cross_image_input_checkpoint_complete" if completed else "cross_image_acceptance_target_not_reached",
        "scope": {
            "selected_images": selections.image_index.astype(int).tolist(),
            "trace_index": TRACE_INDEX,
            "scored_frame": SCORED_FRAME,
            "neural_scoring": False,
            "population_inference": False,
            "final_test_identities_opened": False,
        },
        "contract": contract,
        "acceptance_summary": summary,
        "sources": {
            "stage2_manifest": file_identity(args.stage2_dir / "manifest.json"),
            "stage2_identity_split": file_identity(args.stage2_dir / "frozen_image_and_trace_identity_split.csv"),
            "trace_cache": file_identity(args.response_cache_dir / "input_cache/corrected_trace_segments.npz"),
            "v2_accepted_ensemble_manifest": file_identity(args.v2_source / "manifest.json"),
            "runner": file_identity(Path(__file__)),
        },
        "artifacts": {
            "selection": "selected_images_frozen_before_generation.csv",
            "frozen_contract": "cross_image_contract_frozen_before_generation.json",
            "candidate_audit": "all_cross_image_candidate_audit.csv",
            "accepted_cubes": "accepted_cross_image_input_cubes.npz" if completed else None,
            "contact_sheet": "01_cross_image_contact_sheet.pdf" if completed else None,
            "movies": [f"02_image_{image:03d}_accepted_iaaft.mp4" for image in completed_images],
        },
        "decision_gate": (
            "stop for visual comparison across images; do not score the twin until the human reviewer judges whether "
            "the accepted manipulations consistently destroy recognizable geometry without an unacceptable new artifact"
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Cross-image accepted IAAFT input checkpoint v3",
        "",
        "Four additional development images were selected before surrogate generation at fixed trace 121 and scored ",
        "frame 20. Selection spans predeclared quantiles of the existing map-input power distribution and excludes ",
        "the already-inspected image 94.",
        "",
        f"Status: **{'complete' if completed else 'acceptance target not reached'}**.",
        "",
    ]
    for row in summary:
        lines.append(
            f"- image {row['image_index']} {row['target_power']}: "
            f"{'accepted' if row['accepted'] else 'not accepted'} after {row['attempts']} attempt(s)."
        )
    lines.extend([
        "",
        "No surrogate in this checkpoint was scored by the twin. Stop here for visual review.",
    ])
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
