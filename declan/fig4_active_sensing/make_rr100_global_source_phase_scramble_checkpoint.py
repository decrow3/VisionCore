#!/usr/bin/env python3
"""Global source-image phase-scramble checkpoint for corrected RR100 inputs.

This is deliberately the simplest valid phase control.  A corrected source
patch is Fourier phase-scrambled once with a Hermitian phase field, preserving
its global 2-D amplitude spectrum, mean, and standard deviation.  The intact
and scrambled source patches are then translated by the identical corrected
40-frame FEM trace.  No neural model is evaluated here.

The checkpoint distinguishes exact *source spatial-power* matching from the
retinal movie's resulting SF x TF power, which can differ after finite-window
cropping, interpolation, and boundary handling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from declan.fig4_active_sensing.input_only_retinal_renderer import (
    render_retinal_frames_lag_zero,
)
from declan.fig4_active_sensing.make_rr100_phase_surrogate_input_checkpoint import (
    EPS,
    N_SCORE,
    PPD,
    movie_audit,
    power_audit,
    relative_db,
)
from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    FRAME_RATE_HZ,
    SF_EDGES_CPD,
    spectral_statistics,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
    _standardize_uint_like,
)
from jake.twininfo.stimuli import (
    amplitude_spectrum_relative_error,
    phase_scramble_image,
)


ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
INPUT_CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/input_cache"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_global_source_phase_scramble_checkpoint_42_v1"
IMAGE_METRIC = "image_orientation_coherence"
TRACE_METRIC = "corrected_dpi_crop120_path_length_arcmin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT)
    parser.add_argument("--input-cache", type=Path, default=INPUT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def _gray(frame: np.ndarray, scale: int = 6) -> Image.Image:
    values = np.clip(np.asarray(frame, dtype=np.float64), 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(values, mode="L").convert("RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _difference(frame: np.ndarray, limit: float, scale: int = 6) -> Image.Image:
    values = np.clip(np.asarray(frame, dtype=np.float64) / max(float(limit), EPS), -1.0, 1.0)
    rgb = np.round(255 * matplotlib.colormaps["coolwarm"]((values + 1) / 2)[..., :3]).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    if float(aa.std()) <= 0 or float(bb.std()) <= 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _video_frames(
    intact: np.ndarray,
    scrambled: np.ndarray,
    *,
    title: str,
    repeat: int = 2,
) -> Iterable[Image.Image]:
    delta = np.asarray(scrambled, dtype=np.float64) - np.asarray(intact, dtype=np.float64)
    limit = float(np.quantile(np.abs(delta), 0.99))
    movie_r = _correlation(intact, scrambled)
    title_font = _font(21)
    label_font = _font(17)
    body_font = _font(15)
    scale = 6
    panel = 51 * scale
    gap = 16
    margin = 14
    width = 2 * margin + 3 * panel + 2 * gap
    height = 103 + panel
    for frame_index in range(intact.shape[0]):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 5), title, fill="black", font=title_font)
        positions = [margin, margin + panel + gap, margin + 2 * (panel + gap)]
        panels = (
            _gray(intact[frame_index], scale),
            _gray(scrambled[frame_index], scale),
            _difference(delta[frame_index], limit, scale),
        )
        labels = ("Intact source + FEM", "Global scramble + same FEM", f"Difference (±{limit:.1f})")
        for x, image, label in zip(positions, panels, labels, strict=True):
            canvas.paste(image, (x, 103))
            draw.text((x, 50), label, fill="black", font=label_font)
        draw.text(
            (margin, 78),
            f"frame {frame_index:02d}/39 | frame r={_correlation(intact[frame_index], scrambled[frame_index]):+.3f} | movie r={movie_r:+.3f}",
            fill="black",
            font=body_font,
        )
        for _ in range(int(repeat)):
            yield canvas


def _write_mp4(frames: Iterable[Image.Image], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rr100_global_scramble_", dir="/tmp") as temporary:
        temp = Path(temporary)
        n_frames = 0
        for n_frames, frame in enumerate(frames, start=1):
            canvas = frame
            if canvas.width % 2 or canvas.height % 2:
                even = Image.new(
                    "RGB",
                    (canvas.width + canvas.width % 2, canvas.height + canvas.height % 2),
                    "white",
                )
                even.paste(canvas, (0, 0))
                canvas = even
            canvas.save(temp / f"frame_{n_frames - 1:04d}.png")
        if n_frames == 0:
            raise ValueError("No video frames generated")
        subprocess.run(
            [
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
                str(path),
            ],
            check=True,
        )


def _nearest_unused(frame: pd.DataFrame, metric: str, target: float, used: set[int], id_column: str) -> pd.Series:
    candidates = frame.loc[~frame[id_column].astype(int).isin(used)].copy()
    if candidates.empty:
        raise ValueError("Ran out of distinct candidates")
    index = (candidates[metric].astype(float) - float(target)).abs().idxmin()
    row = candidates.loc[index]
    used.add(int(row[id_column]))
    return row


def select_examples(images: pd.DataFrame, traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    valid_images = images.loc[images[IMAGE_METRIC].notna()].copy()
    if len(valid_images) < 3:
        raise ValueError(f"Too few images with {IMAGE_METRIC}")
    roles = (
        ("low orientation coherence", 0.10),
        ("median orientation coherence", 0.50),
        ("high orientation coherence", 0.90),
    )
    used: set[int] = set()
    rows: list[dict[str, Any]] = []
    for role, quantile in roles:
        target = float(valid_images[IMAGE_METRIC].quantile(quantile))
        row = _nearest_unused(valid_images, IMAGE_METRIC, target, used, "image_index")
        rows.append(
            {
                "selection_role": role,
                "selection_metric": IMAGE_METRIC,
                "selection_quantile": quantile,
                "selection_target": target,
                **row.to_dict(),
            }
        )
    valid_traces = traces.loc[traces[TRACE_METRIC].notna()].copy()
    target_trace = float(valid_traces[TRACE_METRIC].median())
    trace_row = valid_traces.loc[(valid_traces[TRACE_METRIC] - target_trace).abs().idxmin()]
    return pd.DataFrame(rows), trace_row


def _phase_difference_coherence(intact: np.ndarray, scrambled: np.ndarray) -> float:
    a = np.fft.fft2(np.asarray(intact, dtype=np.float64) - float(np.mean(intact)))
    b = np.fft.fft2(np.asarray(scrambled, dtype=np.float64) - float(np.mean(scrambled)))
    valid = (np.abs(a) > EPS) & (np.abs(b) > EPS)
    valid[0, 0] = False
    delta = np.angle(b[valid] * np.conj(a[valid]))
    return float(np.abs(np.mean(np.exp(1j * delta))))


def _montage(movie: np.ndarray, indices: tuple[int, ...] = (5, 19, 34)) -> np.ndarray:
    return np.concatenate([np.asarray(movie[index]) for index in indices], axis=1)


def plot_checkpoint(
    selections: pd.DataFrame,
    payloads: list[dict[str, Any]],
    audit: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(len(payloads), 5, figsize=(17.2, 3.35 * len(payloads)), constrained_layout=True)
    for row_index, (selection, payload) in enumerate(zip(selections.itertuples(index=False), payloads, strict=True)):
        source_intact = payload["source_intact"]
        source_scrambled = payload["source_scrambled"]
        movie_intact = payload["movie_intact"]
        movie_scrambled = payload["movie_scrambled"]
        record = audit.iloc[row_index]
        panels = (
            (source_intact, "Corrected source"),
            (source_scrambled, "Global phase scramble"),
            (_montage(movie_intact), "Intact retinal movie\nt=5, 19, 34"),
            (_montage(movie_scrambled), "Scrambled retinal movie\nsame FEM frames"),
        )
        for column, (values, title) in enumerate(panels):
            ax = axes[row_index, column]
            ax.imshow(values, cmap="gray", vmin=0, vmax=255, origin="lower")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = axes[row_index, 4]
        sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
        tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]
        delta_db = relative_db(payload["radial_scrambled"]) - relative_db(payload["radial_intact"])
        mesh = ax.pcolormesh(sf_centers, tf_hz, delta_db, cmap="coolwarm", vmin=-12, vmax=12, shading="nearest")
        ax.set_xscale("log")
        ax.set_xlabel("SF (cpd)")
        ax.set_ylabel("TF (Hz)")
        ax.set_title(
            f"Retinal SFxTF difference\ncos={record.hann_supported_power_cosine:.3f}; ratio={record.hann_supported_power_ratio:.2f}",
            fontsize=9,
        )
        axes[row_index, 0].set_ylabel(
            f"{selection.selection_role}\nimage {int(selection.image_index)}\ncoherence={float(getattr(selection, IMAGE_METRIC)):.3f}",
            fontsize=9,
        )
    fig.colorbar(mesh, ax=axes[:, 4].ravel().tolist(), label="scrambled minus intact relative power (dB)")
    fig.suptitle(
        "Checkpoint 42: simplest global source-image phase scramble\n"
        "Source spatial amplitude is exact; identical corrected FEM trace is applied afterward; no neural scoring",
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

    images_path = args.cohort_dir / "corrected100_images.csv"
    traces_path = args.cohort_dir / "corrected1000_traces.csv"
    images = pd.read_csv(images_path)
    traces = pd.read_csv(traces_path)
    selections, trace_selection = select_examples(images, traces)
    trace_index = int(trace_selection.trace_index)

    trace_cache_path = args.input_cache / "corrected_trace_segments.npz"
    with np.load(trace_cache_path, allow_pickle=False) as data:
        trace_ids = np.asarray(data["trace_index"], dtype=int)
        history = np.asarray(data["history_xy_deg"], dtype=np.float32)
        score = np.asarray(data["score_xy_deg"], dtype=np.float32)
        request_sha256 = str(data["request_sha256"].item())
    positions = np.where(trace_ids == trace_index)[0]
    if positions.size != 1:
        raise ValueError(f"Expected one trace-cache row for trace {trace_index}, got {positions.size}")
    position = int(positions[0])
    history_trace = history[position]
    score_trace = score[position]
    retinal_score_trace = -score_trace
    common = _load_twin_common()

    audit_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for example_number, selection in enumerate(selections.itertuples(index=False), start=1):
        image_index = int(selection.image_index)
        patch_path = args.input_cache / "images" / f"image_{image_index:03d}.npz"
        with np.load(patch_path, allow_pickle=False) as data:
            source = _standardize_uint_like(np.asarray(data["corrected_patch"], dtype=np.float32))
            ppd = float(data["patch_ppd"].item())
        if not np.isclose(ppd, PPD):
            raise ValueError(f"Unexpected PPD for image {image_index}: {ppd}")
        scramble_seed = int(args.seed) + image_index * 1009
        scrambled = phase_scramble_image(source, np.random.default_rng(scramble_seed))
        movie_intact = (
            render_retinal_frames_lag_zero(common, source, retinal_score_trace, ppd=ppd).cpu().numpy().astype(np.float32)
        )
        movie_scrambled = (
            render_retinal_frames_lag_zero(common, scrambled, retinal_score_trace, ppd=ppd).cpu().numpy().astype(np.float32)
        )
        power_metrics, radial_scrambled = power_audit(movie_intact, movie_scrambled, ppd=ppd)
        radial_intact, _, _ = spectral_statistics(movie_intact, ppd=ppd)
        source_amp_error = amplitude_spectrum_relative_error(source, scrambled)
        source_phase_coherence = _phase_difference_coherence(source, scrambled)
        record = {
            "example_number": example_number,
            "selection_role": selection.selection_role,
            "image_index": image_index,
            "trace_index": trace_index,
            "scramble_seed": scramble_seed,
            "image_orientation_coherence": float(getattr(selection, IMAGE_METRIC)),
            "trace_path_length_arcmin": float(trace_selection[TRACE_METRIC]),
            "source_spatial_amplitude_relative_error": source_amp_error,
            "source_mean_error": abs(float(source.mean()) - float(scrambled.mean())),
            "source_std_error": abs(float(source.std()) - float(scrambled.std())),
            "source_pixel_correlation": _correlation(source, scrambled),
            "source_phase_difference_vector_coherence": source_phase_coherence,
            **movie_audit(movie_intact, movie_scrambled),
            **power_metrics,
        }
        audit_rows.append(record)
        payload = {
            "source_intact": source,
            "source_scrambled": scrambled,
            "movie_intact": movie_intact,
            "movie_scrambled": movie_scrambled,
            "radial_intact": radial_intact,
            "radial_scrambled": radial_scrambled,
        }
        payloads.append(payload)
        data_path = out_dir / "data" / f"example_{example_number}_image_{image_index:03d}.npz"
        np.savez_compressed(
            data_path,
            source_intact=source,
            source_global_phase_scrambled=scrambled,
            movie_intact=movie_intact,
            movie_global_phase_scrambled=movie_scrambled,
            history_xy_deg=history_trace,
            score_xy_deg=score_trace,
            retinal_score_trace_xy_deg=retinal_score_trace,
            radial_power_intact=radial_intact,
            radial_power_global_phase_scrambled=radial_scrambled,
            image_index=np.asarray(image_index),
            trace_index=np.asarray(trace_index),
            scramble_seed=np.asarray(scramble_seed),
        )
        movie_path = out_dir / "movies" / f"example_{example_number}_image_{image_index:03d}_global_phase_scramble.mp4"
        _write_mp4(
            _video_frames(movie_intact, movie_scrambled, title=f"Example {example_number}: image {image_index}, trace {trace_index}"),
            movie_path,
            fps=int(args.fps),
        )
        record["data_npz"] = str(data_path.resolve())
        record["comparison_movie"] = str(movie_path.resolve())

    audit = pd.DataFrame(audit_rows)
    audit_path = out_dir / "global_source_phase_scramble_audit.csv"
    audit.to_csv(audit_path, index=False)
    selections_path = out_dir / "selected_images.csv"
    selections.to_csv(selections_path, index=False)
    trace_row_path = out_dir / "selected_trace.csv"
    pd.DataFrame([trace_selection.to_dict()]).to_csv(trace_row_path, index=False)
    figure_path = out_dir / "global_source_phase_scramble_checkpoint.png"
    plot_checkpoint(selections, payloads, audit, figure_path)

    manifest = {
        "analysis": "rr100_global_source_image_phase_scramble_input_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_only_human_checkpoint_no_neural_scoring",
        "control_contract": (
            "one Hermitian global Fourier phase scramble per corrected source patch, then identical corrected FEM history"
        ),
        "power_contract": (
            "exact global 2-D source spatial amplitude; retinal full spatiotemporal and Hann-windowed SFxTF power audited but not constrained"
        ),
        "n_examples": len(audit),
        "trace_index_shared_across_images": trace_index,
        "trace_selection_metric": TRACE_METRIC,
        "trace_selection_value": float(trace_selection[TRACE_METRIC]),
        "image_selection_metric": IMAGE_METRIC,
        "seed": int(args.seed),
        "source_request_sha256": request_sha256,
        "next_checkpoint_if_approved": "local complex-steerable-pyramid source scramble on the same image/trace identities",
        "inputs": {
            "images": file_identity(images_path),
            "traces": file_identity(traces_path),
            "trace_cache": file_identity(trace_cache_path),
        },
        "outputs": {
            "audit": file_identity(audit_path),
            "selected_images": file_identity(selections_path),
            "selected_trace": file_identity(trace_row_path),
            "figure": file_identity(figure_path),
            "movies": [file_identity(Path(path)) for path in audit.comparison_movie],
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "# Checkpoint 42: global source-image phase scramble\n\n"
        "This is the simplest valid control. Each corrected source patch is globally phase-scrambled once, "
        "then translated with the identical corrected FEM trace. The source image's 2-D Fourier amplitude, "
        "mean, and standard deviation are matched. The retinal movie's full spatiotemporal power is audited "
        "rather than claimed to be exact, because finite retinal cropping and interpolation intervene.\n\n"
        "No digital-twin response or SSI was computed. Review the three MP4s and checkpoint figure before "
        "proceeding to the local complex-steerable-pyramid version.\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
