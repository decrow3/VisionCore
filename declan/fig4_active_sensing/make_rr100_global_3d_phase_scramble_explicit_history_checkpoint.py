#!/usr/bin/env python3
"""Explicit-history validation of the raw global 3-D phase control.

The recurrent twin scores 40 frames using 32 preceding retinal frames.  This
checkpoint therefore renders the full corrected 72-frame history+score movie,
phase-scrambles it as one real 3-D volume with exact Fourier amplitude, and
constructs the 40 x 32 lag stack consumed by the model.  It validates the
intact lag stack against the canonical helper but makes no neural-model call.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch

from declan.fig4_active_sensing.input_only_retinal_renderer import render_retinal_frames_lag_zero
from declan.fig4_active_sensing.make_rr100_global_3d_phase_scramble_checkpoint import (
    amplitude_relative_error,
    global_3d_phase_scramble,
    phase_relationship_audit,
)
from declan.fig4_active_sensing.make_rr100_global_source_phase_scramble_checkpoint import (
    _correlation,
    _difference,
    _font,
    _gray,
    _write_mp4,
    file_identity,
)
from declan.fig4_active_sensing.make_rr100_phase_surrogate_input_checkpoint import (
    PPD,
    movie_audit,
    power_audit,
)
from declan.fig4_active_sensing.run_rr100_corrected_production_cache import render_scored_embedding
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _load_twin_common


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_global_source_phase_scramble_checkpoint_42_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_global_3d_phase_scramble_explicit_history_checkpoint_45_v4"
N_HISTORY = 32
N_SCORE = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def scored_lag_stack(movie72: np.ndarray) -> np.ndarray:
    """Return model-order current-to-past lags for scored frames 32:72."""
    movie = np.asarray(movie72, dtype=np.float32)
    if movie.ndim != 3 or movie.shape[0] != N_HISTORY + N_SCORE:
        raise ValueError(f"Expected a 72 x H x W movie, got {movie.shape}")
    stack = np.stack(
        [np.stack([movie[t - lag] for lag in range(N_HISTORY)], axis=0) for t in range(N_HISTORY, N_HISTORY + N_SCORE)],
        axis=0,
    )
    return stack[:, None].astype(np.float32)


def _montage(movie: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(movie[index]) for index in indices], axis=1)


def _video_frames(intact: np.ndarray, scrambled: np.ndarray, *, title: str, repeat: int = 1) -> Iterable[Image.Image]:
    delta = np.asarray(scrambled, dtype=np.float64) - np.asarray(intact, dtype=np.float64)
    limit = float(np.quantile(np.abs(delta), 0.99))
    title_font = _font(21)
    label_font = _font(16)
    body_font = _font(14)
    scale = 2
    panel = intact.shape[-1] * scale
    gap = 16
    margin = 14
    width = 2 * margin + 3 * panel + 2 * gap
    height = 103 + panel
    for frame_index in range(intact.shape[0]):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        phase = "history" if frame_index < N_HISTORY else "scored"
        draw.text((margin, 5), title, fill="black", font=title_font)
        positions = [margin + index * (panel + gap) for index in range(3)]
        panels = (
            _gray(intact[frame_index], scale),
            _gray(scrambled[frame_index], scale),
            _difference(delta[frame_index], limit, scale),
        )
        labels = ("Intact full movie", "Raw exact-power surrogate", f"Difference (±{limit:.1f})")
        for x, image, label in zip(positions, panels, labels, strict=True):
            canvas.paste(image, (x, 103))
            draw.text((x, 50), label, fill="black", font=label_font)
        draw.text(
            (margin, 78),
            f"frame {frame_index:02d}/71 ({phase}) | frame r={_correlation(intact[frame_index], scrambled[frame_index]):+.3f}",
            fill="black",
            font=body_font,
        )
        for _ in range(int(repeat)):
            yield canvas


def plot_checkpoint(payloads: list[dict[str, Any]], audit: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(len(payloads), 4, figsize=(17.2, 3.45 * len(payloads)), constrained_layout=True)
    history_indices = (0, 15, 31)
    score_indices = (32, 51, 71)
    for row_index, payload in enumerate(payloads):
        record = audit.iloc[row_index]
        panels = (
            (_montage(payload["intact"], history_indices), "Intact history\nt=0,15,31"),
            (_montage(payload["scrambled"], history_indices), "Scrambled history\nt=0,15,31"),
            (_montage(payload["intact"], score_indices), "Intact scored\nt=32,51,71"),
            (_montage(payload["scrambled"], score_indices), "Scrambled scored\nt=32,51,71"),
        )
        for column, (values, title) in enumerate(panels):
            ax = axes[row_index, column]
            ax.imshow(values, cmap="gray", vmin=0, vmax=255, origin="lower")
            if column == 3:
                title += (
                    f"\nfull amp err={record.full72_3d_amplitude_relative_error:.1e}; "
                    f"scored Hann ratio={record.scored40_hann_supported_power_ratio:.2f}; "
                    f"OOR={100*record.full72_out_of_0_255_fraction:.1f}%"
                )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_index, 0].set_ylabel(
            f"{payload['selection_role']}\nimage {payload['image_index']}",
            fontsize=9,
        )
    fig.suptitle(
        "Checkpoint 45: exact-power 3-D phase scramble with genuine 32-frame history\n"
        "Full 72-frame movie is scrambled before constructing the twin's current-to-past lag stack; no model call",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path, dpi=185)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {out_dir}")
    (out_dir / "movies").mkdir(parents=True)
    (out_dir / "data").mkdir()
    selections_path = args.source_checkpoint / "selected_images.csv"
    trace_path = args.source_checkpoint / "selected_trace.csv"
    selections = pd.read_csv(selections_path)
    pd.read_csv(trace_path).to_csv(out_dir / "selected_trace.csv", index=False)
    selections.to_csv(out_dir / "selected_images.csv", index=False)
    common = _load_twin_common()

    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for example_number, selection in enumerate(selections.itertuples(index=False), start=1):
        image_index = int(selection.image_index)
        source_path = args.source_checkpoint / "data" / f"example_{example_number}_image_{image_index:03d}.npz"
        with np.load(source_path, allow_pickle=False) as data:
            source = np.asarray(data["source_intact"], dtype=np.float32)
            history = np.asarray(data["history_xy_deg"], dtype=np.float32)
            score = np.asarray(data["score_xy_deg"], dtype=np.float32)
            prior_scored = np.asarray(data["movie_intact"], dtype=np.float32)
            trace_index = int(data["trace_index"].item())
        trace72 = np.concatenate([history, score], axis=0)
        retinal72 = -trace72
        intact72 = (
            render_retinal_frames_lag_zero(
                common,
                source,
                retinal72,
                ppd=PPD,
                out_size=tuple(int(value) for value in common.OUT_SIZE),
            )
            .detach().cpu().numpy().astype(np.float32)
        )
        half = prior_scored.shape[-1] // 2
        cy, cx = intact72.shape[-2] // 2, intact72.shape[-1] // 2
        central_scored = intact72[
            N_HISTORY:,
            cy - half:cy + half + 1,
            cx - half:cx + half + 1,
        ]
        # The prior visualization used a native 51-pixel output grid.  It is
        # not exactly the center crop of the 151-pixel production grid because
        # the helper defines grid endpoints from out_size.  Retain this only as
        # an explicit cross-resolution diagnostic, not an equality assertion.
        prior_scored_error = float(np.max(np.abs(central_scored - prior_scored)))
        intact_stack = scored_lag_stack(intact72)
        canonical = render_scored_embedding(common, torch, source, trace72, PPD).detach().cpu().numpy()
        helper_error = float(np.max(np.abs((intact_stack - 127.0) / 255.0 - canonical)))
        if helper_error > 1e-5:
            raise AssertionError(f"Direct lag-stack mismatch against canonical helper: {helper_error}")
        seed = int(args.seed) + image_index * 1009
        scrambled72 = global_3d_phase_scramble(intact72, np.random.default_rng(seed))
        scrambled_stack = scored_lag_stack(scrambled72)
        scored_power, _ = power_audit(intact72[N_HISTORY:], scrambled72[N_HISTORY:], ppd=PPD)
        scored_movie = movie_audit(intact72[N_HISTORY:], scrambled72[N_HISTORY:])
        phase = phase_relationship_audit(intact72, scrambled72)
        row = {
            "example_number": example_number,
            "selection_role": selection.selection_role,
            "image_index": image_index,
            "trace_index": trace_index,
            "seed": seed,
            "legacy51_vs_production151_center_max_abs_difference_expected_grid_mismatch": prior_scored_error,
            "canonical_scored_lag_stack_max_abs_error_after_normalization": helper_error,
            "full72_3d_amplitude_relative_error": amplitude_relative_error(intact72, scrambled72),
            "full72_fourier_phase_retention_coherence": phase["fourier_phase_retention_coherence"],
            "full72_max_adjacent_frequency_phase_relation_retention_coherence": max(
                value for key, value in phase.items() if key.startswith("adjacent_")
            ),
            **{f"full72_{key}": value for key, value in movie_audit(intact72, scrambled72).items()},
            **{f"scored40_{key}": value for key, value in scored_movie.items()},
            **{f"scored40_{key}": value for key, value in scored_power.items()},
        }
        rows.append(row)
        payloads.append(
            {
                "selection_role": selection.selection_role,
                "image_index": image_index,
                "intact": intact72,
                "scrambled": scrambled72,
            }
        )
        np.savez_compressed(
            out_dir / "data" / f"example_{example_number}_image_{image_index:03d}.npz",
            movie_intact_full72=intact72,
            movie_global_3d_phase_scrambled_full72=scrambled72,
            history_xy_deg=history,
            score_xy_deg=score,
            image_index=np.asarray(image_index),
            trace_index=np.asarray(trace_index),
            seed=np.asarray(seed),
        )
        _write_mp4(
            _video_frames(
                intact72,
                scrambled72,
                title=f"Example {example_number}: image {image_index}, trace {trace_index}, full history+score",
            ),
            out_dir / "movies" / f"example_{example_number}_image_{image_index:03d}_full72_global_3d_phase_scramble.mp4",
            fps=int(args.fps),
        )
    audit = pd.DataFrame(rows)
    audit_path = out_dir / "explicit_history_global_3d_phase_scramble_audit.csv"
    audit.to_csv(audit_path, index=False)
    figure_path = out_dir / "explicit_history_global_3d_phase_scramble_checkpoint.png"
    plot_checkpoint(payloads, audit, figure_path)
    manifest = {
        "analysis": "rr100_global_3d_phase_scramble_explicit_history_input_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_only_human_checkpoint_no_neural_scoring",
        "contract": (
            "the genuine 32-frame history and 40 scored lag-zero frames are rendered as one 72-frame retinal "
            "movie; one real global 3-D phase scramble exactly retains its full Fourier amplitude; the scored "
            "40 x 32 current-to-past lag stack is then constructed directly"
        ),
        "maximum_full72_amplitude_relative_error": float(audit.full72_3d_amplitude_relative_error.max()),
        "maximum_full72_phase_retention_coherence": float(audit.full72_fourier_phase_retention_coherence.max()),
        "maximum_full72_adjacent_phase_relation_retention_coherence": float(
            audit.full72_max_adjacent_frequency_phase_relation_retention_coherence.max()
        ),
        "maximum_canonical_lag_stack_validation_error": float(
            audit.canonical_scored_lag_stack_max_abs_error_after_normalization.max()
        ),
        "maximum_full72_out_of_range_fraction": float(audit.full72_out_of_0_255_fraction.max()),
        "scored40_hann_supported_power_ratios": audit.scored40_hann_supported_power_ratio.tolist(),
        "inputs": {
            "source_manifest": file_identity(args.source_checkpoint / "manifest.json"),
            "selected_images": file_identity(selections_path),
            "selected_trace": file_identity(trace_path),
        },
        "outputs": {
            "audit": file_identity(audit_path),
            "figure": file_identity(figure_path),
            "movies": [file_identity(path) for path in sorted((out_dir / "movies").glob("*.mp4"))],
        },
        "next_checkpoint_if_approved": "targeted intact-versus-raw-surrogate RR100 activation maps on a small role-based unit panel",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Checkpoint 45: explicit-history exact-power phase control\n\n"
        "This corrects the model-interface omission in the 40-frame input checkpoint. The full genuine "
        "32-frame history plus 40 scored frames is scrambled as one movie before constructing the twin's "
        "32 lag channels. The intact direct lag construction is numerically validated against the canonical "
        "stimulus helper. No neural model or SSI was evaluated.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
