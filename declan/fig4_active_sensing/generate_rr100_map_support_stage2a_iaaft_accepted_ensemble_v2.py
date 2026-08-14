#!/usr/bin/env python3
"""Generate a frozen, input-only ensemble of accepted 3-D IAAFT surrogates.

Candidate seeds are rejected only by predeclared stimulus-level manipulation
checks.  No twin response is evaluated in this runner.  FEM-amplitude and
stabilized-amplitude ensembles use independent, previously unseen seed streams.
"""
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

from declan.fig4_active_sensing.run_rr100_map_support_stage2a_distribution_constrained_phase_v1 import (
    CANONICAL_HIGH,
    CANONICAL_LOW,
    distribution_audit,
    file_identity,
    iaaft_3d,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "outputs/fig4_active_sensing/rr100_map_support_stage2a_distribution_constrained_phase_v1"
)
DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/rr100_map_support_stage2a_iaaft_accepted_ensemble_v2"
)
TARGET_ACCEPTED = 10
MAX_ATTEMPTS_PER_POWER = 100
ITERATIONS = 64
SEED_START = {"FEM": 20261101, "stabilized": 20262101}
PHASE_SUPPORT_RELATIVE_THRESHOLD = 1e-4
PHASE_SUPPORT_MAX_TARGET_ENERGY = 5e-4
FROZEN_THRESHOLDS = {
    "fourier_amplitude_relative_error_max": 1e-6,
    "fraction_outside_canonical_input_range_max": 1e-4,
    "histogram_wasserstein_distance_max": 1e-3,
    "absolute_global_excess_kurtosis_difference_max": 0.02,
    "patch_kurtosis_wasserstein_distance_max": 0.5,
    "spatial_gradient_wasserstein_over_reference_std_max": 0.15,
    "temporal_gradient_wasserstein_over_reference_std_max": 0.15,
    "global_unweighted_phase_coherence_max": 0.05,
    "global_energy_weighted_phase_coherence_max": 0.10,
    "phase_support_target_energy_in_weak_bins_max": PHASE_SUPPORT_MAX_TARGET_ENERGY,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-accepted", type=int, default=TARGET_ACCEPTED)
    parser.add_argument("--max-attempts-per-power", type=int, default=MAX_ATTEMPTS_PER_POWER)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def target_energy_in_weak_phase_bins(cube: np.ndarray) -> float:
    amplitude = np.abs(np.fft.fftn(np.asarray(cube, dtype=np.float64)))
    weak = amplitude <= max(1e-12, PHASE_SUPPORT_RELATIVE_THRESHOLD * float(amplitude.max()))
    energy = np.square(amplitude)
    return float(energy[weak].sum() / max(float(energy.sum()), np.finfo(float).tiny))


def acceptance_decision(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    checks = {
        "fourier_amplitude": (
            metrics["fourier_amplitude_relative_error"]
            <= FROZEN_THRESHOLDS["fourier_amplitude_relative_error_max"]
        ),
        "canonical_range": (
            metrics["fraction_outside_canonical_input_range"]
            <= FROZEN_THRESHOLDS["fraction_outside_canonical_input_range_max"]
        ),
        "histogram": (
            metrics["histogram_wasserstein_distance"]
            <= FROZEN_THRESHOLDS["histogram_wasserstein_distance_max"]
        ),
        "global_higher_order": (
            metrics["absolute_global_excess_kurtosis_difference"]
            <= FROZEN_THRESHOLDS["absolute_global_excess_kurtosis_difference_max"]
        ),
        "patch_higher_order": (
            metrics["patch_kurtosis_wasserstein_distance"]
            <= FROZEN_THRESHOLDS["patch_kurtosis_wasserstein_distance_max"]
        ),
        "spatial_gradient_distribution": (
            metrics["spatial_gradient_wasserstein_over_reference_std"]
            <= FROZEN_THRESHOLDS["spatial_gradient_wasserstein_over_reference_std_max"]
        ),
        "temporal_gradient_distribution": (
            metrics["temporal_gradient_wasserstein_over_reference_std"]
            <= FROZEN_THRESHOLDS["temporal_gradient_wasserstein_over_reference_std_max"]
        ),
        "unweighted_phase_decorrelation": (
            metrics["global_unweighted_phase_coherence"]
            <= FROZEN_THRESHOLDS["global_unweighted_phase_coherence_max"]
        ),
        "energy_weighted_phase_decorrelation": (
            metrics["global_energy_weighted_phase_coherence"]
            <= FROZEN_THRESHOLDS["global_energy_weighted_phase_coherence_max"]
        ),
        "phase_source_support": (
            metrics["phase_support_target_energy_in_weak_bins"]
            <= FROZEN_THRESHOLDS["phase_support_target_energy_in_weak_bins_max"]
        ),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    metrics.update({f"acceptance_check__{name}": bool(passed) for name, passed in checks.items()})
    return len(failed) == 0, failed


def load_targets(source: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    with np.load(source / "distribution_constrained_input_cubes.npz", allow_pickle=False) as archive:
        targets = {
            "FEM": np.asarray(archive["fem_original"], dtype=np.float32),
            "stabilized": np.asarray(archive["stabilized_original"], dtype=np.float32),
        }
        identity = {
            "image_index": int(archive["image_index"]),
            "trace_index": int(archive["trace_index"]),
            "scored_frame": int(archive["scored_frame"]),
        }
    return targets, identity


def generate_ensemble(
    power: str,
    target: np.ndarray,
    *,
    target_accepted: int,
    max_attempts: int,
    iterations: int,
) -> tuple[list[np.ndarray], pd.DataFrame, pd.DataFrame]:
    accepted: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    convergence_rows: list[pd.DataFrame] = []
    weak_energy = target_energy_in_weak_phase_bins(target)
    for attempt in range(1, int(max_attempts) + 1):
        if len(accepted) >= int(target_accepted):
            break
        seed = int(SEED_START[power] + attempt - 1)
        candidate, convergence = iaaft_3d(target, seed, int(iterations))
        convergence["target_power"] = power
        convergence["attempt"] = attempt
        convergence_rows.append(convergence)
        metrics = distribution_audit(
            target,
            candidate,
            seed=seed,
            condition=f"{power.lower()}_power_iaaft_candidate",
            source_kind="natural map-support cube rejection-sampling candidate",
        )
        metrics["target_power"] = power
        metrics["attempt"] = attempt
        metrics["phase_support_relative_threshold"] = PHASE_SUPPORT_RELATIVE_THRESHOLD
        metrics["phase_support_target_energy_in_weak_bins"] = weak_energy
        passed, failed = acceptance_decision(metrics)
        metrics["accepted"] = bool(passed)
        metrics["accepted_ordinal"] = len(accepted) if passed else -1
        metrics["failed_acceptance_checks"] = ";".join(failed)
        rows.append(metrics)
        if passed:
            accepted.append(candidate)
        print(
            f"[{power}] attempt={attempt} seed={seed} accepted={passed} "
            f"n_accepted={len(accepted)}/{target_accepted} "
            f"energy_phase={metrics['global_energy_weighted_phase_coherence']:.4f}",
            flush=True,
        )
    convergence_table = (
        pd.concat(convergence_rows, ignore_index=True) if convergence_rows else pd.DataFrame()
    )
    return accepted, pd.DataFrame(rows), convergence_table


def make_movie(
    targets: dict[str, np.ndarray],
    ensembles: dict[str, np.ndarray],
    audit: pd.DataFrame,
    ordinal: int,
    path: Path,
    dpi: int,
) -> None:
    pairs = []
    for power in ("FEM", "stabilized"):
        row = audit.loc[audit.target_power.eq(power) & audit.accepted & audit.accepted_ordinal.eq(ordinal)].iloc[0]
        pairs.append((targets[power], ensembles[power][ordinal], power, int(row.seed)))
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
                candidate[0] - original[0],
                cmap="RdBu_r",
                origin="lower",
                norm=TwoSlopeNorm(vmin=-difference_limit, vcenter=0, vmax=difference_limit),
            ),
        ])
        axes[row_index, 0].set_ylabel(f"{power}\nseed {seed}")
    for column, title in enumerate(("original", "accepted IAAFT", "accepted − original")):
        axes[0, column].set_title(title)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    title = figure.suptitle(f"Accepted ensemble member {ordinal + 1}: frame 1/32")

    def update(frame: int):
        index = 0
        for original, candidate, _, _ in pairs:
            for values in (original[frame], candidate[frame], candidate[frame] - original[frame]):
                images[index].set_data(values)
                index += 1
        title.set_text(f"Accepted ensemble member {ordinal + 1}: frame {frame + 1}/32")
        return [*images, title]

    writer = animation.FFMpegWriter(fps=12, bitrate=2200)
    animation.FuncAnimation(figure, update, frames=32, interval=80, blit=False).save(
        path, writer=writer, dpi=dpi
    )
    plt.close(figure)


def plot_acceptance(audit: pd.DataFrame, path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    colors = {"FEM": "#0072B2", "stabilized": "#E69F00"}
    for power, frame in audit.groupby("target_power", sort=False):
        axes[0].scatter(
            frame.attempt,
            frame.global_energy_weighted_phase_coherence,
            color=[colors[power] if passed else "0.75" for passed in frame.accepted],
            edgecolor="k",
            linewidth=0.4,
            label=power,
        )
        axes[1].step(
            frame.attempt,
            frame.accepted.astype(int).cumsum(),
            where="post",
            color=colors[power],
            label=power,
        )
        accepted = frame.loc[frame.accepted]
        axes[2].scatter(
            accepted.histogram_wasserstein_distance,
            accepted.patch_kurtosis_wasserstein_distance,
            color=colors[power],
            label=power,
        )
    axes[0].axhline(
        FROZEN_THRESHOLDS["global_energy_weighted_phase_coherence_max"],
        color="#D55E00",
        linestyle="--",
        label="frozen threshold",
    )
    axes[0].set(xlabel="attempt", ylabel="energy-weighted phase coherence", title="Phase acceptance")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="attempt", ylabel="cumulative accepted", title="Acceptance rate")
    axes[1].legend(fontsize=8)
    axes[2].set(
        xlabel="histogram Wasserstein",
        ylabel="patch-kurtosis Wasserstein",
        title="Accepted distribution checks",
    )
    axes[2].legend(fontsize=8)
    figure.suptitle("Input-only rejection sampling; gray points failed at least one frozen gate", weight="bold")
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed ensemble exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "created_utc_before_candidate_generation": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_candidate_generation",
        "target_accepted_per_power": int(args.target_accepted),
        "max_attempts_per_power": int(args.max_attempts_per_power),
        "iterations_per_candidate": int(args.iterations),
        "seed_start": SEED_START,
        "thresholds": FROZEN_THRESHOLDS,
        "selection_inputs": "stimulus audits only; no twin response is loaded or scored",
        "ensembles": "FEM and stabilized phase fields are optimized and accepted independently",
        "recorded_grating": (
            "the v1 generic-degradation calibration is inherited as context; its discrete histogram is not an "
            "exact natural-input acceptance gate"
        ),
        "runner": file_identity(Path(__file__)),
    }
    (args.out_dir / "acceptance_contract_frozen_before_generation.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )

    targets, identity = load_targets(args.source)
    ensembles: dict[str, np.ndarray] = {}
    audit_frames: list[pd.DataFrame] = []
    convergence_frames: list[pd.DataFrame] = []
    completed = True
    for power in ("FEM", "stabilized"):
        accepted, audit, convergence = generate_ensemble(
            power,
            targets[power],
            target_accepted=int(args.target_accepted),
            max_attempts=int(args.max_attempts_per_power),
            iterations=int(args.iterations),
        )
        audit_frames.append(audit)
        convergence_frames.append(convergence)
        if len(accepted) < int(args.target_accepted):
            completed = False
        if accepted:
            ensembles[power] = np.stack(accepted).astype(np.float32)

    audit = pd.concat(audit_frames, ignore_index=True)
    convergence = pd.concat(convergence_frames, ignore_index=True)
    audit.to_csv(args.out_dir / "all_candidate_acceptance_audit.csv", index=False)
    convergence.to_csv(args.out_dir / "all_candidate_iaaft_convergence.csv", index=False)
    accepted_rows = audit.loc[audit.accepted].copy()
    accepted_rows.to_csv(args.out_dir / "accepted_ensemble_members.csv", index=False)
    rejected_rows = audit.loc[~audit.accepted].copy()
    rejected_rows.to_csv(args.out_dir / "rejected_candidates.csv", index=False)

    if completed:
        np.savez_compressed(
            args.out_dir / "accepted_iaaft_ensemble.npz",
            fem_original=targets["FEM"],
            stabilized_original=targets["stabilized"],
            fem_accepted=ensembles["FEM"],
            stabilized_accepted=ensembles["stabilized"],
            fem_seed=accepted_rows.loc[accepted_rows.target_power.eq("FEM"), "seed"].to_numpy(np.int64),
            stabilized_seed=accepted_rows.loc[
                accepted_rows.target_power.eq("stabilized"), "seed"
            ].to_numpy(np.int64),
            image_index=np.asarray(identity["image_index"]),
            trace_index=np.asarray(identity["trace_index"]),
            scored_frame=np.asarray(identity["scored_frame"]),
        )
        plot_acceptance(audit, args.out_dir / "01_ensemble_acceptance_audit", int(args.dpi))
        for ordinal in range(min(3, int(args.target_accepted))):
            make_movie(
                targets,
                ensembles,
                audit,
                ordinal,
                args.out_dir / f"02_accepted_member_{ordinal + 1:02d}.mp4",
                int(args.dpi),
            )

    acceptance_summary = {}
    for power, frame in audit.groupby("target_power", sort=False):
        n_accepted = int(frame.accepted.sum())
        acceptance_summary[power] = {
            "attempts": int(len(frame)),
            "accepted": n_accepted,
            "acceptance_rate": float(n_accepted / max(len(frame), 1)),
            "last_accepted_seed": (
                int(frame.loc[frame.accepted, "seed"].iloc[-1]) if n_accepted else None
            ),
        }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_map_support_stage2a_iaaft_accepted_ensemble_v2",
        "status": "input_only_ensemble_complete" if completed else "input_only_ensemble_target_not_reached",
        "scope": {
            **identity,
            "neural_scoring": False,
            "population_inference": False,
            "final_test_identities_opened": False,
        },
        "acceptance_summary": acceptance_summary,
        "contract": contract,
        "sources": {
            "distribution_constrained_v1_manifest": file_identity(args.source / "manifest.json"),
            "distribution_constrained_v1_cubes": file_identity(
                args.source / "distribution_constrained_input_cubes.npz"
            ),
            "distribution_constrained_v1_grating_calibration": file_identity(
                args.source / "recorded_grating_generic_degradation_summary.json"
            ),
            "runner": file_identity(Path(__file__)),
        },
        "artifacts": {
            "frozen_contract": "acceptance_contract_frozen_before_generation.json",
            "candidate_audit": "all_candidate_acceptance_audit.csv",
            "accepted_members": "accepted_ensemble_members.csv",
            "rejected_candidates": "rejected_candidates.csv",
            "ensemble": "accepted_iaaft_ensemble.npz" if completed else None,
            "acceptance_figure": "01_ensemble_acceptance_audit.pdf" if completed else None,
            "example_movies": (
                [f"02_accepted_member_{ordinal + 1:02d}.mp4" for ordinal in range(min(3, int(args.target_accepted)))]
                if completed else []
            ),
        },
        "decision_gate": (
            "stop at the input-only ensemble checkpoint; inspect acceptance rate, candidate audits, and example movies "
            "before any accepted surrogate is exposed to the twin"
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage 2A accepted IAAFT ensemble v2",
        "",
        "This checkpoint used only frozen stimulus-level acceptance rules. No accepted or rejected candidate was ",
        "scored by the twin. FEM and stabilized ensembles are independent because a shared final phase field was ",
        "pathological under simultaneous distribution constraints.",
        "",
        f"Status: **{'complete' if completed else 'target not reached'}**.",
        "",
    ]
    for power, summary in acceptance_summary.items():
        lines.append(
            f"- {power}: {summary['accepted']} accepted in {summary['attempts']} attempts "
            f"(rate {summary['acceptance_rate']:.3f})."
        )
    lines.extend([
        "",
        "The previously recorded grating calibration remains a generic-degradation check, not proof that a ",
        "discrete grating histogram can satisfy the same exact natural-movie projection contract.",
        "",
        "Stop here for human review before raw maps or SSI are computed.",
    ])
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
