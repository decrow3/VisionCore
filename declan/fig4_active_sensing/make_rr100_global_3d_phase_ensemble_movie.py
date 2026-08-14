#!/usr/bin/env python3
"""Render an unfiltered ensemble of exact-power 3-D phase surrogates.

The median-orientation-coherence checkpoint image is shown intact beside five
predeclared, independent random-phase realizations.  All realizations scramble
the same production-sized 72 x 151 x 151 history+score retinal movie and none
is selected or rejected using power, range, or response criteria.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from declan.fig4_active_sensing.make_rr100_global_3d_phase_scramble_checkpoint import (
    amplitude_relative_error,
    global_3d_phase_scramble,
    phase_relationship_audit,
)
from declan.fig4_active_sensing.make_rr100_global_source_phase_scramble_checkpoint import (
    _font,
    _write_mp4,
    file_identity,
)
from declan.fig4_active_sensing.make_rr100_phase_surrogate_input_checkpoint import (
    PPD,
    movie_audit,
    power_audit,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_global_3d_phase_scramble_explicit_history_checkpoint_45_v4"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_global_3d_phase_ensemble_movie_checkpoint_46_v3"
EXAMPLE_NUMBER = 2
IMAGE_INDEX = 68
SEEDS = (20260820, 20260821, 20260822, 20260823, 20260824)
N_HISTORY = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args()


def display_panel(values: np.ndarray, *, scale: int = 2, mark_oor: bool = False) -> Image.Image:
    """Shared 0--255 display; optionally mark low/high OOD pixels cyan/magenta."""
    array = np.asarray(values, dtype=np.float64)
    clipped = np.clip(array, 0.0, 255.0).astype(np.uint8)
    rgb = np.repeat(clipped[..., None], 3, axis=2)
    if mark_oor:
        rgb[array < 0.0] = np.asarray([0, 235, 255], dtype=np.uint8)
        rgb[array > 255.0] = np.asarray([255, 0, 210], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def ensemble_frames(
    intact: np.ndarray,
    surrogates: list[np.ndarray],
    audit: pd.DataFrame,
    *,
    repeat: int,
) -> Iterable[Image.Image]:
    scale = 2
    panel = int(intact.shape[-1] * scale)
    margin = 18
    column_gap = 18
    row_gap = 88
    header_height = 110
    footer_height = 86
    width = 2 * margin + 3 * panel + 2 * column_gap
    height = header_height + 2 * panel + row_gap + footer_height
    title_font = _font(24)
    label_font = _font(17)
    body_font = _font(14)
    phase_font = _font(17)
    records = audit.set_index("seed")
    entries: list[tuple[str, np.ndarray, bool, int | None]] = [
        ("Intact", intact, False, None),
        *[(f"Random phase {index + 1}", surrogate, True, int(SEEDS[index])) for index, surrogate in enumerate(surrogates)],
    ]
    for frame_index in range(intact.shape[0]):
        phase = "GENUINE HISTORY" if frame_index < N_HISTORY else "SCORED RESPONSE WINDOW"
        phase_color = (45, 105, 170) if frame_index < N_HISTORY else (185, 78, 28)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (margin, 7),
            "Unfiltered global 3-D phase ensemble — image 68, exact same 72-frame power",
            fill="black",
            font=title_font,
        )
        draw.rounded_rectangle((margin, 47, width - margin, 81), radius=7, fill=phase_color)
        draw.text(
            (margin + 12, 53),
            f"frame {frame_index:02d}/71  |  {phase}",
            fill="white",
            font=phase_font,
        )
        for entry_index, (label, movie, mark_oor, seed) in enumerate(entries):
            row = entry_index // 3
            column = entry_index % 3
            x = margin + column * (panel + column_gap)
            y = header_height + row * (panel + row_gap)
            canvas.paste(display_panel(movie[frame_index], scale=scale, mark_oor=mark_oor), (x, y))
            draw.text((x, y - 25), label, fill="black", font=label_font)
            if seed is None:
                stats = "natural retinal movie"
            else:
                record = records.loc[int(seed)]
                stats = (
                    f"seed {seed} | OOD {100 * record.full72_out_of_0_255_fraction:.1f}%\n"
                    f"scored Hann power {record.scored40_hann_supported_power_ratio:.2f}× intact"
                )
            draw.multiline_text((x, y + panel + 6), stats, fill="black", font=body_font, spacing=2)
        draw.text(
            (margin, height - 29),
            "Cyan: value < 0    Magenta: value > 255    Grayscale uses the same fixed 0–255 display for every panel.",
            fill=(35, 35, 35),
            font=body_font,
        )
        for _ in range(int(repeat)):
            yield canvas


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {out_dir}")
    out_dir.mkdir(parents=True)
    source_path = args.source_checkpoint / "data" / f"example_{EXAMPLE_NUMBER}_image_{IMAGE_INDEX:03d}.npz"
    with np.load(source_path, allow_pickle=False) as data:
        intact = np.asarray(data["movie_intact_full72"], dtype=np.float32)
        trace_index = int(data["trace_index"].item())

    surrogates: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for ensemble_index, seed in enumerate(SEEDS, start=1):
        surrogate = global_3d_phase_scramble(intact, np.random.default_rng(int(seed)))
        phase = phase_relationship_audit(intact, surrogate)
        scored_power, _ = power_audit(intact[N_HISTORY:], surrogate[N_HISTORY:], ppd=PPD)
        rows.append(
            {
                "ensemble_index": ensemble_index,
                "seed": int(seed),
                "image_index": IMAGE_INDEX,
                "trace_index": trace_index,
                "selection_policy": "predeclared_seed_unfiltered_no_rejection",
                "full72_3d_amplitude_relative_error": amplitude_relative_error(intact, surrogate),
                "full72_fourier_phase_retention_coherence": phase["fourier_phase_retention_coherence"],
                "full72_max_adjacent_frequency_phase_relation_retention_coherence": max(
                    value for key, value in phase.items() if key.startswith("adjacent_")
                ),
                **{f"full72_{key}": value for key, value in movie_audit(intact, surrogate).items()},
                **{f"scored40_{key}": value for key, value in scored_power.items()},
            }
        )
        surrogates.append(surrogate)
    audit = pd.DataFrame(rows)
    audit_path = out_dir / "phase_ensemble_audit.csv"
    audit.to_csv(audit_path, index=False)
    arrays_path = out_dir / "phase_ensemble_movies.npz"
    np.savez_compressed(
        arrays_path,
        movie_intact_full72=intact,
        movie_phase_ensemble_full72=np.stack(surrogates),
        seeds=np.asarray(SEEDS, dtype=np.int64),
        image_index=np.asarray(IMAGE_INDEX),
        trace_index=np.asarray(trace_index),
    )
    movie_path = out_dir / "image_068_unfiltered_global_3d_phase_ensemble.mp4"
    _write_mp4(
        ensemble_frames(intact, surrogates, audit, repeat=int(args.repeat)),
        movie_path,
        fps=int(args.fps),
    )
    manifest = {
        "analysis": "rr100_unfiltered_global_3d_phase_ensemble_movie",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_only_human_checkpoint_no_neural_scoring",
        "example_selection": "median orientation coherence from the predeclared three-image input checkpoint",
        "image_index": IMAGE_INDEX,
        "trace_index": trace_index,
        "seeds": list(SEEDS),
        "seed_policy": "predeclared consecutive seeds; no seed selected or rejected using any audit or response",
        "display_contract": "shared 0--255 grayscale; values below 0 cyan; values above 255 magenta",
        "inputs": {"explicit_history_movie": file_identity(source_path)},
        "outputs": {
            "movie": file_identity(movie_path),
            "audit": file_identity(audit_path),
            "arrays": file_identity(arrays_path),
        },
        "not_run": "No neural response, activation map, or SSI was computed.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Unfiltered exact-power phase ensemble movie\n\n"
        "The median-coherence production input is shown intact beside five independent random-phase "
        "realizations. Every seed was declared before synthesis and retained. The complete 72-frame "
        "spatiotemporal Fourier amplitude is identical to intact in every surrogate. Cyan and magenta "
        "make values outside the training range visible. No neural model was evaluated.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
