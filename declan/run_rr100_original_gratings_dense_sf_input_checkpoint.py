"""Audit and visualize dense SF substitution on the recorded grating sequence.

This is deliberately an input-only, map-first checkpoint.  It reconstructs
stored 240 Hz retinal frames with the original ForageGrating renderer, verifies
pixel equality, and then changes only positive carrier SF values.  Blank
carrier slots (sf == 0), orientation, timing, gaze-dependent ROI, contrast,
and probe/face overlays are inherited from the recorded sequence.

No neural data or digital-twin model is evaluated here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DataYatesV1.exp.gratings import GratingsTrial
from DataYatesV1.utils.general import get_clock_functions
from DataYatesV1.utils.io import get_session


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "rr100_original_gratings_dense_sf_input_checkpoint"
DEFAULT_CONFIG = ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long_legacy.yaml"
DEFAULT_SESSIONS = ("Allen_2022-02-16", "Logan_2020-02-29")
TARGET_SFS = np.power(2.0, np.arange(0.0, 4.0 + 0.5, 0.5))


@dataclass
class SessionCheckpoint:
    session: str
    ppd: float
    selected_global_index: int
    selected_trial_index: int
    selected_flip_index: int
    selected_source_sf: float
    selected_orientation: float
    selected_probe_index: float
    selected_invalid_fraction: float
    stored_frame: np.ndarray
    regenerated_frame: np.ndarray
    target_frames: np.ndarray
    sequence_source_sfs: np.ndarray
    sequence_orientations: np.ndarray
    sequence_probe_indices: np.ndarray
    sequence_original: np.ndarray
    sequence_targets: dict[float, np.ndarray]
    audit_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", default=",".join(DEFAULT_SESSIONS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-frames", type=int, default=128)
    parser.add_argument("--sequence-frames", type=int, default=8)
    return parser.parse_args()


def map_bins_to_flips(trial: GratingsTrial, trial_times: np.ndarray, ptb2ephys) -> np.ndarray:
    flip_times = np.asarray(ptb2ephys(trial.flip_times), dtype=float)
    frame_inds = np.searchsorted(flip_times, trial_times) - 1
    return np.maximum(frame_inds, 0).astype(int)


def substitute_positive_sf(
    trial: GratingsTrial,
    frame_inds: np.ndarray,
    rois: np.ndarray,
    target_sf: float,
) -> np.ndarray:
    """Render frames after replacing every positive trial SF by target_sf."""
    original = trial.spatial_frequencies.copy()
    try:
        trial.spatial_frequencies = np.where(original > 0, float(target_sf), 0.0)
        return trial.get_frames(frame_inds, roi=rois)
    finally:
        trial.spatial_frequencies = original


def choose_example_index(dset, n_sequence_frames: int) -> int:
    """Choose an auditable model-rate frame with a common 4-cpd carrier and overlay."""
    sf = dset["sf"].numpy()
    trials = dset["trial_inds"].numpy().astype(int)
    candidates = np.flatnonzero(np.isclose(sf, 4.0) & (np.arange(len(sf)) % 2 == 0))
    candidates = candidates[candidates >= 2 * (n_sequence_frames - 1)]
    candidates = candidates[
        trials[candidates] == trials[candidates - 2 * (n_sequence_frames - 1)]
    ]
    if not len(candidates):
        raise RuntimeError("No within-trial 4-cpd example supports the requested sequence.")

    # Evaluate a bounded, evenly spaced subset.  A 10% invalid phase fraction
    # usually exposes a probe without allowing it to dominate the retinal crop.
    if len(candidates) > 2000:
        candidates = candidates[np.linspace(0, len(candidates) - 1, 2000).astype(int)]
    phase = dset["stim_phase"][candidates].numpy()
    invalid_fraction = np.mean(phase < 0, axis=(1, 2))
    return int(candidates[np.argmin(np.abs(invalid_fraction - 0.10))])


def build_session_checkpoint(
    session_name: str,
    audit_frames: int,
    sequence_frames: int,
) -> SessionCheckpoint:
    subject, date = session_name.split("_", maxsplit=1)
    sess = get_session(subject, date)
    dset = sess.get_dataset("gratings", strict=True)
    ptb2ephys, _ = get_clock_functions(sess.exp)

    chosen = choose_example_index(dset, sequence_frames)
    trial_index = int(dset["trial_inds"][chosen])
    trial = GratingsTrial(sess.exp["D"][trial_index], sess.exp["S"])

    all_trial_indices = np.flatnonzero(dset["trial_inds"].numpy().astype(int) == trial_index)
    audit_positions = np.linspace(0, len(all_trial_indices) - 1, audit_frames).round().astype(int)
    audit_global = np.unique(all_trial_indices[audit_positions])
    audit_times = dset["t_bins"][audit_global].numpy()
    audit_flips = map_bins_to_flips(trial, audit_times, ptb2ephys)
    audit_rois = dset["roi"][audit_global].numpy()
    regenerated = trial.get_frames(audit_flips, roi=audit_rois)
    stored = dset["stim"][audit_global].numpy()
    difference = np.abs(regenerated.astype(np.int16) - stored.astype(np.int16))

    source_sf = trial.spatial_frequencies[audit_flips]
    source_ori = trial.orientations[audit_flips]
    stored_sf = dset["sf"][audit_global].numpy()
    stored_ori = dset["ori"][audit_global].numpy()
    if not np.allclose(source_sf, stored_sf, atol=1e-10, rtol=0):
        raise AssertionError(f"{session_name}: reconstructed SF labels differ from stored labels")
    if not np.allclose(source_ori, stored_ori, atol=1e-10, rtol=0):
        raise AssertionError(f"{session_name}: reconstructed orientation labels differ from stored labels")
    if np.max(difference) != 0:
        raise AssertionError(
            f"{session_name}: original generator did not reproduce stored frames; "
            f"max pixel difference={int(np.max(difference))}"
        )

    audit_rows: list[dict[str, Any]] = []
    for j, global_index in enumerate(audit_global):
        audit_rows.append(
            {
                "session": session_name,
                "global_240hz_index": int(global_index),
                "trial_index": trial_index,
                "flip_index": int(audit_flips[j]),
                "source_sf_cpd": float(source_sf[j]),
                "orientation_deg": float(source_ori[j]),
                "probe_index": float(trial.p_index[audit_flips[j]]),
                "max_abs_pixel_difference": int(np.max(difference[j])),
                "n_different_pixels": int(np.count_nonzero(difference[j])),
                "exact_match": bool(not np.any(difference[j])),
            }
        )

    # Match the model's global 240 -> 120 Hz stimulus decimation (indices 0,2,4,...).
    sequence_global = chosen - 2 * np.arange(sequence_frames - 1, -1, -1)
    if np.any(dset["trial_inds"][sequence_global].numpy().astype(int) != trial_index):
        raise AssertionError("Selected temporal example crosses a trial boundary.")
    sequence_times = dset["t_bins"][sequence_global].numpy()
    sequence_flips = map_bins_to_flips(trial, sequence_times, ptb2ephys)
    sequence_rois = dset["roi"][sequence_global].numpy()
    sequence_original = dset["stim"][sequence_global].numpy()

    chosen_flip = int(map_bins_to_flips(
        trial,
        np.asarray([float(dset["t_bins"][chosen])]),
        ptb2ephys,
    )[0])
    chosen_roi = dset["roi"][chosen : chosen + 1].numpy()
    target_frames = np.stack(
        [substitute_positive_sf(trial, np.asarray([chosen_flip]), chosen_roi, sf)[0] for sf in TARGET_SFS]
    )
    sequence_targets = {
        sf: substitute_positive_sf(trial, sequence_flips, sequence_rois, sf)
        for sf in (1.0, 4.0, 16.0)
    }

    selected_phase = dset["stim_phase"][chosen].numpy()
    selected_stored = dset["stim"][chosen].numpy()
    selected_regenerated = trial.get_frames(
        np.asarray([chosen_flip]), roi=chosen_roi
    )[0]
    checkpoint = SessionCheckpoint(
        session=session_name,
        ppd=float(np.asarray(dset.metadata["ppd"])),
        selected_global_index=chosen,
        selected_trial_index=trial_index,
        selected_flip_index=chosen_flip,
        selected_source_sf=float(trial.spatial_frequencies[chosen_flip]),
        selected_orientation=float(trial.orientations[chosen_flip]),
        selected_probe_index=float(trial.p_index[chosen_flip]),
        selected_invalid_fraction=float(np.mean(selected_phase < 0)),
        stored_frame=selected_stored,
        regenerated_frame=selected_regenerated,
        target_frames=target_frames,
        sequence_source_sfs=trial.spatial_frequencies[sequence_flips].copy(),
        sequence_orientations=trial.orientations[sequence_flips].copy(),
        sequence_probe_indices=trial.p_index[sequence_flips].copy(),
        sequence_original=sequence_original,
        sequence_targets=sequence_targets,
        audit_rows=audit_rows,
    )

    # Release the large stored phase tensor before moving to the next session.
    del dset, sess, trial
    return checkpoint


def make_support_table(checkpoints: list[SessionCheckpoint]) -> pd.DataFrame:
    rows = []
    for checkpoint in checkpoints:
        fov_deg = 51.0 / checkpoint.ppd
        for sf in TARGET_SFS:
            pixels_per_cycle = checkpoint.ppd / sf
            rows.append(
                {
                    "session": checkpoint.session,
                    "target_sf_cpd": float(sf),
                    "pixels_per_degree": checkpoint.ppd,
                    "retinal_crop_pixels": 51,
                    "retinal_crop_degrees": fov_deg,
                    "cycles_across_crop": float(sf * fov_deg),
                    "pixels_per_cycle": float(pixels_per_cycle),
                    "at_least_one_cycle_across_crop": bool(sf * fov_deg >= 1.0),
                    "below_pixel_nyquist": bool(sf <= checkpoint.ppd / 2.0),
                    "at_least_three_pixels_per_cycle": bool(pixels_per_cycle >= 3.0),
                }
            )
    return pd.DataFrame(rows)


def plot_reconstruction(checkpoints: list[SessionCheckpoint], output: Path) -> None:
    fig, axes = plt.subplots(len(checkpoints), 3, figsize=(8.2, 2.8 * len(checkpoints)), squeeze=False)
    for row, checkpoint in enumerate(checkpoints):
        diff = np.abs(
            checkpoint.stored_frame.astype(np.int16) - checkpoint.regenerated_frame.astype(np.int16)
        )
        for col, (image, title, cmap, vmax) in enumerate(
            [
                (checkpoint.stored_frame, "stored 240 Hz frame", "gray", 255),
                (checkpoint.regenerated_frame, "original renderer", "gray", 255),
                (diff, "absolute difference", "magma", 1),
            ]
        ):
            axes[row, col].imshow(image, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(checkpoint.session.replace("_", "\n"), fontsize=9)
    fig.suptitle("Pixel audit: stored retinal crop is reproduced exactly", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sf_frame_grid(checkpoints: list[SessionCheckpoint], output: Path) -> None:
    ncols = len(TARGET_SFS) + 1
    fig, axes = plt.subplots(len(checkpoints), ncols, figsize=(2.0 * ncols, 2.55 * len(checkpoints)), squeeze=False)
    for row, checkpoint in enumerate(checkpoints):
        images = [checkpoint.stored_frame, *list(checkpoint.target_frames)]
        titles = [f"original\n{checkpoint.selected_source_sf:g} cpd", *[f"{sf:.3g} cpd" for sf in TARGET_SFS]]
        for col, (image, title) in enumerate(zip(images, titles)):
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(
            f"{checkpoint.session}\nori={checkpoint.selected_orientation:g}°\n"
            f"probe={checkpoint.selected_probe_index:g}",
            fontsize=8,
        )
    fig.suptitle(
        "Dense SF substitution at one matched recorded moment\n"
        "Only positive carrier SF changes; ROI, phase origin, orientation, contrast, and overlay are fixed",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sequence_strip(checkpoints: list[SessionCheckpoint], output: Path) -> None:
    row_specs = [("original sequence", None), ("substitute 1 cpd", 1.0), ("substitute 4 cpd", 4.0), ("substitute 16 cpd", 16.0)]
    nframes = checkpoints[0].sequence_original.shape[0]
    fig, axes = plt.subplots(
        len(checkpoints) * len(row_specs),
        nframes,
        figsize=(1.55 * nframes, 1.42 * len(checkpoints) * len(row_specs)),
        squeeze=False,
    )
    for session_i, checkpoint in enumerate(checkpoints):
        for spec_i, (label, target) in enumerate(row_specs):
            row = session_i * len(row_specs) + spec_i
            images = checkpoint.sequence_original if target is None else checkpoint.sequence_targets[target]
            for col, image in enumerate(images):
                axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(f"t={col / 120:.3f}s", fontsize=8)
                if target is None:
                    axes[row, col].text(
                        0.03,
                        0.04,
                        f"{checkpoint.sequence_source_sfs[col]:g} cpd\n"
                        f"{checkpoint.sequence_orientations[col]:g}°",
                        transform=axes[row, col].transAxes,
                        color="yellow",
                        fontsize=6,
                        va="bottom",
                        bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
                    )
            axes[row, 0].set_ylabel(
                f"{checkpoint.session}\n{label}" if spec_i == 0 else label,
                fontsize=8,
            )
    fig.suptitle(
        "Original 120 Hz sequence replayed with SF substitution\n"
        "Orientation changes, blank slots, gaze-dependent crops, and probe states retain their recorded order",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = [value.strip() for value in args.sessions.split(",") if value.strip()]

    checkpoints = [
        build_session_checkpoint(name, args.audit_frames, args.sequence_frames)
        for name in sessions
    ]

    audit = pd.DataFrame([row for checkpoint in checkpoints for row in checkpoint.audit_rows])
    support = make_support_table(checkpoints)
    selection = pd.DataFrame(
        [
            {
                "selection_role": "representative_recorded_moment_with_visible_overlay",
                "session": checkpoint.session,
                "global_240hz_index": checkpoint.selected_global_index,
                "trial_index": checkpoint.selected_trial_index,
                "flip_index": checkpoint.selected_flip_index,
                "source_sf_cpd": checkpoint.selected_source_sf,
                "orientation_deg": checkpoint.selected_orientation,
                "probe_index": checkpoint.selected_probe_index,
                "phase_invalid_fraction": checkpoint.selected_invalid_fraction,
                "selection_reason": "4 cpd is shared by Allen and Logan; phase-invalid fraction nearest 0.10",
            }
            for checkpoint in checkpoints
        ]
    )

    audit_path = output_dir / "original_renderer_pixel_audit.csv"
    support_path = output_dir / "dense_sf_support.csv"
    selection_path = output_dir / "example_selection.csv"
    audit.to_csv(audit_path, index=False)
    support.to_csv(support_path, index=False)
    selection.to_csv(selection_path, index=False)

    reconstruction_path = output_dir / "original_renderer_pixel_audit.png"
    frame_grid_path = output_dir / "dense_sf_substitution_frame_grid.png"
    sequence_path = output_dir / "dense_sf_substitution_sequence_strip.png"
    plot_reconstruction(checkpoints, reconstruction_path)
    plot_sf_frame_grid(checkpoints, frame_grid_path)
    plot_sequence_strip(checkpoints, sequence_path)

    np.savez_compressed(
        output_dir / "dense_sf_substitution_examples.npz",
        sessions=np.asarray(sessions),
        target_sf_cpd=TARGET_SFS,
        target_frames=np.stack([checkpoint.target_frames for checkpoint in checkpoints]),
        sequence_original=np.stack([checkpoint.sequence_original for checkpoint in checkpoints]),
        sequence_1_cpd=np.stack([checkpoint.sequence_targets[1.0] for checkpoint in checkpoints]),
        sequence_4_cpd=np.stack([checkpoint.sequence_targets[4.0] for checkpoint in checkpoints]),
        sequence_16_cpd=np.stack([checkpoint.sequence_targets[16.0] for checkpoint in checkpoints]),
    )

    manifest = {
        "analysis": "rr100_original_gratings_dense_sf_input_checkpoint",
        "scope": "stimulus construction and pixel audit only; no neural/model responses",
        "sessions": sessions,
        "dataset_config": str(args.dataset_config.resolve()),
        "source_grating_renderer": str(
            Path(__import__("DataYatesV1.exp.gratings", fromlist=["__file__"]).__file__).resolve()
        ),
        "source_dataset_generator": str(
            (Path(__import__("DataYatesV1.exp.dataset_generation", fromlist=["__file__"]).__file__)).resolve()
        ),
        "source_rate_hz": 240,
        "model_rate_hz": 120,
        "model_rate_rule": "global stimulus decimation: retain source indices 0,2,4,...",
        "target_sf_cpd": TARGET_SFS.tolist(),
        "substitution_rule": "target_sf if original_sf > 0 else 0",
        "held_fixed": [
            "recorded trial and flip order",
            "recorded orientation sequence",
            "sf=0 blank-carrier slots",
            "absolute screen-coordinate phase origin",
            "grating contrast and background",
            "gaze-dependent 51x51 ROI",
            "Gabor/face probe identity, position, and alpha blend",
        ],
        "pixel_audit": {
            "n_frames": int(len(audit)),
            "all_exact": bool(audit["exact_match"].all()),
            "maximum_abs_pixel_difference": int(audit["max_abs_pixel_difference"].max()),
        },
        "files": {
            "pixel_audit_csv": str(audit_path),
            "support_csv": str(support_path),
            "selection_csv": str(selection_path),
            "pixel_audit_figure": str(reconstruction_path),
            "frame_grid_figure": str(frame_grid_path),
            "sequence_figure": str(sequence_path),
            "example_arrays": str(output_dir / "dense_sf_substitution_examples.npz"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest["pixel_audit"], indent=2))
    print(f"Wrote checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
