#!/usr/bin/env python3
"""Save human-eye-check movies for the RR100 phase-surrogate input checkpoint."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_phase_surrogate_input_checkpoint_40_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_phase_surrogate_eyecheck_movies_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--comparison-fps", type=int, default=12)
    parser.add_argument("--raw-fps", type=int, default=8)
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def as_display_gray(frame: np.ndarray, scale: int) -> Image.Image:
    clipped = np.clip(np.asarray(frame, dtype=np.float64), 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(clipped, mode="L").convert("RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def signed_difference_image(
    difference: np.ndarray,
    *,
    limit: float,
    scale: int,
) -> Image.Image:
    normalized = np.clip(np.asarray(difference, dtype=np.float64) / max(limit, 1e-12), -1.0, 1.0)
    rgba = matplotlib.colormaps["coolwarm"]((normalized + 1.0) / 2.0)
    rgb = np.round(255.0 * rgba[..., :3]).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def frame_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    if float(aa.std()) == 0 or float(bb.std()) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def write_mp4(frames: Iterable[Image.Image], destination: Path, *, fps: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rr100_phase_video_", dir="/tmp") as temporary:
        temp = Path(temporary)
        count = 0
        for count, frame in enumerate(frames, start=1):
            canvas = frame
            if canvas.width % 2 or canvas.height % 2:
                even = Image.new(
                    "RGB",
                    (canvas.width + canvas.width % 2, canvas.height + canvas.height % 2),
                    "black",
                )
                even.paste(canvas, (0, 0))
                canvas = even
            canvas.save(temp / f"frame_{count - 1:04d}.png")
        if count == 0:
            raise ValueError("No frames supplied")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            str(temp / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        subprocess.run(command, check=True)


def raw_movie_frames(movie: np.ndarray, *, title: str, scale: int = 8) -> list[Image.Image]:
    frames: list[Image.Image] = []
    title_font = font(22)
    body_font = font(17)
    for index, values in enumerate(movie):
        panel = as_display_gray(values, scale)
        canvas = Image.new("RGB", (panel.width, panel.height + 58), "white")
        canvas.paste(panel, (0, 58))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), title, fill="black", font=title_font)
        draw.text(
            (8, 32),
            f"scored frame {index:02d}/39 | display clipped to [0,255]",
            fill="black",
            font=body_font,
        )
        frames.append(canvas)
    return frames


def comparison_frames(
    intact: np.ndarray,
    surrogate: np.ndarray,
    *,
    label: str,
    repeat: int = 3,
    scale: int = 5,
) -> list[Image.Image]:
    differences = np.asarray(surrogate, dtype=np.float64) - np.asarray(intact, dtype=np.float64)
    difference_limit = float(np.quantile(np.abs(differences), 0.99))
    global_correlation = frame_correlation(intact, surrogate)
    out_of_range = float(np.mean((surrogate < 0.0) | (surrogate > 255.0)))
    title_font = font(21)
    label_font = font(18)
    body_font = font(16)
    panel_size = 51 * scale
    gap = 18
    margin = 14
    width = 2 * margin + 3 * panel_size + 2 * gap
    height = 106 + panel_size
    frames: list[Image.Image] = []
    for index, (original, control, difference) in enumerate(
        zip(intact, surrogate, differences, strict=True)
    ):
        original_image = as_display_gray(original, scale)
        control_image = as_display_gray(control, scale)
        difference_image = signed_difference_image(
            difference, limit=difference_limit, scale=scale
        )
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 5), label, fill="black", font=title_font)
        positions = [margin, margin + panel_size + gap, margin + 2 * (panel_size + gap)]
        for x, image in zip(
            positions, (original_image, control_image, difference_image), strict=True
        ):
            canvas.paste(image, (x, 106))
        draw.text((positions[0], 54), "Intact FEM", fill="black", font=label_font)
        draw.text((positions[1], 54), "All-pass raw", fill="black", font=label_font)
        draw.text(
            (positions[2], 54),
            f"Signed difference (±{difference_limit:.1f})",
            fill="black",
            font=label_font,
        )
        draw.text(
            (margin, 80),
            f"frame {index:02d}/39 | frame r={frame_correlation(original, control):+.3f} | "
            f"movie r={global_correlation:+.3f} | OOR={100 * out_of_range:.2f}%",
            fill="black",
            font=body_font,
        )
        frames.extend([canvas] * int(repeat))
    return frames


def flicker_frames(
    intact: np.ndarray,
    surrogate: np.ndarray,
    *,
    label: str,
    fps: int = 12,
    selected_frames: tuple[int, ...] = (5, 19, 34),
    scale: int = 8,
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    hold = max(int(round(0.55 * fps)), 1)
    title_font = font(23)
    body_font = font(18)
    for frame_index in selected_frames:
        for _cycle in range(3):
            for condition, movie in (("INTACT", intact), ("ALL-PASS", surrogate)):
                panel = as_display_gray(movie[frame_index], scale)
                canvas = Image.new("RGB", (panel.width, panel.height + 70), "white")
                canvas.paste(panel, (0, 70))
                draw = ImageDraw.Draw(canvas)
                color = "#1b7837" if condition == "INTACT" else "#762a83"
                draw.text((8, 5), f"{label} — {condition}", fill=color, font=title_font)
                draw.text(
                    (8, 38),
                    f"same scored timepoint t={frame_index}; display clipped to [0,255]",
                    fill="black",
                    font=body_font,
                )
                frames.extend([canvas] * hold)
    return frames


def mechanism_figure(
    intact: np.ndarray,
    surrogate: np.ndarray,
    impulse: np.ndarray,
    destination: Path,
    *,
    frame_index: int = 19,
) -> None:
    original = np.asarray(intact[frame_index], dtype=np.float64)
    control = np.asarray(surrogate[frame_index], dtype=np.float64)
    difference = control - original
    original_fft = np.fft.fftshift(np.fft.fft2(original))
    control_fft = np.fft.fftshift(np.fft.fft2(control))
    amplitude_error = float(
        np.linalg.norm(np.abs(original_fft) - np.abs(control_fft))
        / np.linalg.norm(np.abs(original_fft))
    )
    phase_change = np.angle(control_fft * np.conj(original_fft))
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.2), constrained_layout=True)
    axes[0, 0].imshow(original, cmap="gray", vmin=0, vmax=255, origin="lower")
    axes[0, 0].set_title("Intact frame")
    axes[0, 1].imshow(control, cmap="gray", vmin=0, vmax=255, origin="lower")
    axes[0, 1].set_title(f"All-pass frame\nraw range {control.min():.1f}–{control.max():.1f}")
    limit = float(np.quantile(np.abs(difference), 0.99))
    image = axes[0, 2].imshow(
        difference, cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower"
    )
    axes[0, 2].set_title(f"All-pass − intact\n99% |Δ| ≤ {limit:.1f}")
    fig.colorbar(image, ax=axes[0, 2], shrink=0.8)
    image = axes[0, 3].imshow(impulse, cmap="coolwarm", origin="lower")
    axes[0, 3].set_title("All-pass impulse response\n90% energy radius = 6 px")
    fig.colorbar(image, ax=axes[0, 3], shrink=0.8)
    log_original = np.log10(np.maximum(np.abs(original_fft), 1e-6))
    log_control = np.log10(np.maximum(np.abs(control_fft), 1e-6))
    low = min(float(log_original.min()), float(log_control.min()))
    high = max(float(log_original.max()), float(log_control.max()))
    axes[1, 0].imshow(log_original, cmap="magma", vmin=low, vmax=high, origin="lower")
    axes[1, 0].set_title("log Fourier magnitude: intact")
    axes[1, 1].imshow(log_control, cmap="magma", vmin=low, vmax=high, origin="lower")
    axes[1, 1].set_title(f"log Fourier magnitude: all-pass\nrelative error {amplitude_error:.2e}")
    axes[1, 2].imshow(np.angle(original_fft), cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower")
    axes[1, 2].set_title("Fourier phase: intact")
    phase_image = axes[1, 3].imshow(
        phase_change, cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower"
    )
    axes[1, 3].set_title("Phase added by all-pass filter")
    fig.colorbar(phase_image, ax=axes[1, 3], shrink=0.8, label="radians")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Why an all-pass filter changes the image: magnitude is preserved, phase alignment is not\n"
        "‘All-pass’ means unit gain at every frequency, not an identity transformation",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(destination, dpi=190)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite eye-check output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    selections = pd.read_csv(args.source_dir / "selected_conditions.csv")
    audit = pd.read_csv(args.source_dir / "phase_surrogate_control_audit.csv")
    with np.load(args.source_dir / "localized_allpass_filter.npz", allow_pickle=False) as data:
        impulse = np.asarray(data["impulse"], dtype=np.float64)
    records: list[dict[str, object]] = []
    first_pair: tuple[np.ndarray, np.ndarray] | None = None
    for example_number, selection in enumerate(selections.itertuples(index=False), start=1):
        condition = int(selection.condition_row)
        source = args.source_dir / f"condition_{condition:04d}_movies_and_power.npz"
        with np.load(source, allow_pickle=False) as data:
            intact = np.asarray(data["movie_intact"], dtype=np.float32)
            allpass = np.asarray(data["movie_allpass_raw"], dtype=np.float32)
        if first_pair is None:
            first_pair = (intact, allpass)
        label = (
            f"Example {example_number}: image {int(selection.image_index)}, "
            f"trace {int(selection.trace_index)}"
        )
        stem = f"example_{example_number}_condition_{condition:04d}"
        files = {
            "intact": args.out_dir / f"{stem}_intact.mp4",
            "allpass": args.out_dir / f"{stem}_allpass_raw.mp4",
            "comparison": args.out_dir / f"{stem}_side_by_side_difference.mp4",
            "flicker": args.out_dir / f"{stem}_selected_frame_flicker.mp4",
        }
        write_mp4(
            raw_movie_frames(intact, title=f"{label} — intact FEM"),
            files["intact"],
            fps=int(args.raw_fps),
        )
        write_mp4(
            raw_movie_frames(allpass, title=f"{label} — all-pass raw"),
            files["allpass"],
            fps=int(args.raw_fps),
        )
        write_mp4(
            comparison_frames(intact, allpass, label=label),
            files["comparison"],
            fps=int(args.comparison_fps),
        )
        write_mp4(
            flicker_frames(
                intact,
                allpass,
                label=label,
                fps=int(args.comparison_fps),
            ),
            files["flicker"],
            fps=int(args.comparison_fps),
        )
        row = audit[
            audit.condition_row.eq(condition) & audit.control.eq("allpass_raw")
        ].iloc[0]
        records.append(
            {
                "example_number": example_number,
                "condition_row": condition,
                "image_index": int(selection.image_index),
                "trace_index": int(selection.trace_index),
                "rr100_index_from_invalid_upstream_selection": int(selection.rr100_index),
                "former_selection_role_invalid_due_to_spectral_row_bug": selection.selection_role,
                "centered_movie_pixel_correlation": float(
                    row.centered_pixel_correlation_with_intact
                ),
                "out_of_0_255_fraction": float(row.out_of_0_255_fraction),
                "full_st_amplitude_relative_error": float(
                    row.unwindowed_full_st_amplitude_relative_error
                ),
                "hann_supported_power_cosine": float(row.hann_supported_power_cosine),
                "hann_supported_power_ratio": float(row.hann_supported_power_ratio),
                **{f"{key}_movie": str(value.resolve()) for key, value in files.items()},
            }
        )
    if first_pair is None:
        raise RuntimeError("No examples found")
    mechanism_figure(
        first_pair[0],
        first_pair[1],
        impulse,
        args.out_dir / "allpass_mechanism_explainer.png",
    )
    pd.DataFrame(records).to_csv(args.out_dir / "movie_manifest.csv", index=False)
    paper_note = """# Comparison with Broderick et al., Foveated metamers

The paper's energy model does not directly Fourier-scramble a target image.
It represents the image with complex-steerable-pyramid energy at six scales
and four orientations plus luminance, pools those statistics in overlapping
Gaussian log-polar windows, and optimizes pixels from white-noise or natural
image initializations until the pooled target statistics are matched.

Transferable ideas for our control:

- define the representation that must be matched before synthesizing;
- use overlapping local windows to avoid blocking and ringing;
- include luminance/range constraints explicitly;
- generate multiple seeds, because initialization and optimization select
  different points from the matched-statistic set;
- compare intact-vs-synthetic and synthetic-vs-synthetic outcomes;
- audit residual representation error rather than assuming synthesis success.

Limits on direct transfer:

- their targets are static wide-field images, while ours are 40-frame retinal
  movies with explicit neural history;
- their pooling is foveated and perceptual, while our immediate scale should
  be tied to retinal/RF support;
- their model matches pooled spatial energy, not our measured SF×TF movie
  power or the frozen twin's input distribution;
- the revised paper explicitly weakens direct physiological interpretation of
  model pooling scales.

A later optimization-based movie control should therefore match local complex
spatial energy, temporal-frequency energy, luminance, and valid pixel range,
while preserving the recorded FEM path and reporting every residual mismatch.
"""
    (args.out_dir / "paper_method_comparison.md").write_text(paper_note, encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "human_eye_check_movies_complete",
        "scope": {
            "examples": len(records),
            "movies_per_example": 4,
            "neural_model_calls": False,
        },
        "display_contract": (
            "Video pixels are displayed with a fixed 0..255 mapping; raw all-pass values outside this range are clipped only for video encoding."
        ),
        "selection_guardrail": (
            "Examples retain their original image/trace identities, but former routing-based roles are invalid because of the upstream spectral-row bug."
        ),
        "source": str(args.source_dir.resolve()),
        "outputs": {
            "movie_manifest": str((args.out_dir / "movie_manifest.csv").resolve()),
            "mechanism_explainer": str(
                (args.out_dir / "allpass_mechanism_explainer.png").resolve()
            ),
            "paper_method_comparison": str(
                (args.out_dir / "paper_method_comparison.md").resolve()
            ),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
