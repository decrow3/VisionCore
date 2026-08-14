#!/usr/bin/env python3
"""Exact global 3-D phase-scramble checkpoint for RR100 retinal movies.

The already-rendered intact retinal movie is treated as one (time, y, x)
volume.  Its complete 3-D Fourier amplitude is combined with the phase of a
real white-noise volume, guaranteeing a real surrogate with the same global
spatiotemporal power, mean, and variance.  A simple rank-histogram-matched
version is also saved to expose the tradeoff between exact power and valid
pixel range.  No digital twin is evaluated.
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

from declan.fig4_active_sensing.make_rr100_global_source_phase_scramble_checkpoint import (
    _correlation,
    _difference,
    _font,
    _gray,
    _montage,
    _write_mp4,
    file_identity,
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


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_global_source_phase_scramble_checkpoint_42_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_global_3d_phase_scramble_checkpoint_44_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def global_3d_phase_scramble(movie: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomize 3-D Fourier phase while retaining exact amplitude and reality."""
    values = np.asarray(movie, dtype=np.float64)
    mean = float(values.mean())
    centered = values - mean
    amplitude = np.abs(np.fft.fftn(centered))
    amplitude[(0, 0, 0)] = 0.0
    noise_phase = np.angle(np.fft.fftn(rng.standard_normal(values.shape)))
    surrogate = np.fft.ifftn(amplitude * np.exp(1j * noise_phase)).real + mean
    return surrogate.astype(np.float32)


def rank_histogram_match(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Give values exactly the empirical pixel histogram of reference."""
    reference_flat = np.asarray(reference, dtype=np.float32).ravel()
    values_flat = np.asarray(values, dtype=np.float32).ravel()
    order = np.argsort(values_flat, kind="stable")
    out = np.empty_like(values_flat)
    out[order] = np.sort(reference_flat, kind="stable")
    return out.reshape(np.asarray(values).shape).astype(np.float32)


def amplitude_relative_error(reference: np.ndarray, control: np.ndarray) -> float:
    reference_amp = np.abs(np.fft.fftn(np.asarray(reference, dtype=np.float64)))
    control_amp = np.abs(np.fft.fftn(np.asarray(control, dtype=np.float64)))
    return float(np.linalg.norm(control_amp - reference_amp) / max(np.linalg.norm(reference_amp), EPS))


def _weighted_coherence(delta: np.ndarray, weight: np.ndarray) -> float:
    phase = np.asarray(delta, dtype=np.float64).ravel()
    weights = np.asarray(weight, dtype=np.float64).ravel()
    valid = np.isfinite(phase) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    vector = np.sum(weights[valid] * np.exp(1j * phase[valid])) / np.sum(weights[valid])
    return float(np.abs(vector))


def _supported_unweighted_coherence(delta: np.ndarray, valid: np.ndarray) -> float:
    phase = np.asarray(delta, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(phase)
    if not np.any(mask):
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * phase[mask]))))


def phase_relationship_audit(reference: np.ndarray, control: np.ndarray) -> dict[str, float]:
    """Measure retained phase and adjacent-frequency relations in one FFT octant."""
    reference_fft = np.fft.fftn(np.asarray(reference, dtype=np.float64))
    control_fft = np.fft.fftn(np.asarray(control, dtype=np.float64))
    # Restrict to the nonnegative-frequency octant to avoid counting conjugate
    # copies as independent observations.
    stops = tuple(int(size // 2 + 1) for size in reference_fft.shape)
    index = tuple(slice(0, stop) for stop in stops)
    before = reference_fft[index]
    after = control_fft[index]
    supported = (
        (np.abs(before) > 1e-6 * float(np.abs(before).max()))
        & (np.abs(after) > 1e-6 * float(np.abs(after).max()))
    )
    supported[(0, 0, 0)] = False
    direct = _supported_unweighted_coherence(
        np.angle(after * np.conj(before)),
        supported,
    )
    result = {"fourier_phase_retention_coherence": direct}
    axis_names = ("temporal", "vertical_spatial", "horizontal_spatial")
    for axis, name in enumerate(axis_names):
        low_index = [slice(None)] * 3
        high_index = [slice(None)] * 3
        low_index[axis] = slice(0, -1)
        high_index[axis] = slice(1, None)
        low_index = tuple(low_index)
        high_index = tuple(high_index)
        before_low = before[low_index]
        before_high = before[high_index]
        after_low = after[low_index]
        after_high = after[high_index]
        before_relation = np.angle(before_high * np.conj(before_low))
        after_relation = np.angle(after_high * np.conj(after_low))
        before_product = before_high * np.conj(before_low)
        after_product = after_high * np.conj(after_low)
        relation_supported = (
            (np.abs(before_product) > 1e-6 * float(np.abs(before_product).max()))
            & (np.abs(after_product) > 1e-6 * float(np.abs(after_product).max()))
        )
        result[f"adjacent_{name}_frequency_phase_relation_retention_coherence"] = _supported_unweighted_coherence(
            after_relation - before_relation,
            relation_supported,
        )
    return result


def histogram_audit(reference: np.ndarray, control: np.ndarray) -> dict[str, float]:
    reference_sorted = np.sort(np.asarray(reference, dtype=np.float64).ravel())
    control_sorted = np.sort(np.asarray(control, dtype=np.float64).ravel())
    return {
        "histogram_sorted_rmse": float(np.sqrt(np.mean((control_sorted - reference_sorted) ** 2))),
        "histogram_sorted_rmse_over_intact_sd": float(
            np.sqrt(np.mean((control_sorted - reference_sorted) ** 2))
            / max(float(np.std(reference_sorted)), EPS)
        ),
        "mean_error": abs(float(np.mean(control_sorted)) - float(np.mean(reference_sorted))),
        "std_error": abs(float(np.std(control_sorted)) - float(np.std(reference_sorted))),
    }


def _video_frames(
    intact: np.ndarray,
    raw: np.ndarray,
    histogram_matched: np.ndarray,
    *,
    title: str,
    repeat: int = 2,
) -> Iterable[Image.Image]:
    delta = np.asarray(histogram_matched, dtype=np.float64) - np.asarray(intact, dtype=np.float64)
    limit = float(np.quantile(np.abs(delta), 0.99))
    title_font = _font(21)
    label_font = _font(16)
    body_font = _font(14)
    scale = 5
    panel = 51 * scale
    gap = 14
    margin = 14
    width = 2 * margin + 4 * panel + 3 * gap
    height = 103 + panel
    for frame_index in range(intact.shape[0]):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 5), title, fill="black", font=title_font)
        positions = [margin + index * (panel + gap) for index in range(4)]
        panels = (
            _gray(intact[frame_index], scale),
            _gray(raw[frame_index], scale),
            _gray(histogram_matched[frame_index], scale),
            _difference(delta[frame_index], limit, scale),
        )
        labels = (
            "Intact retinal movie",
            "Raw exact 3-D power",
            "Exact histogram",
            f"Histogram matched - intact (±{limit:.1f})",
        )
        for x, image, label in zip(positions, panels, labels, strict=True):
            canvas.paste(image, (x, 103))
            draw.text((x, 50), label, fill="black", font=label_font)
        draw.text(
            (margin, 78),
            f"frame {frame_index:02d}/39 | raw frame r={_correlation(intact[frame_index], raw[frame_index]):+.3f}"
            f" | histogram-matched frame r={_correlation(intact[frame_index], histogram_matched[frame_index]):+.3f}",
            fill="black",
            font=body_font,
        )
        for _ in range(int(repeat)):
            yield canvas


def plot_checkpoint(payloads: list[dict[str, Any]], audit: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(len(payloads), 5, figsize=(17.8, 3.45 * len(payloads)), constrained_layout=True)
    sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]
    mesh = None
    for row_index, payload in enumerate(payloads):
        panels = (
            (payload["intact"], "Intact retinal movie"),
            (payload["raw"], "Raw exact 3-D power"),
            (payload["histogram"], "Rank histogram matched"),
        )
        for column, (movie, title) in enumerate(panels):
            ax = axes[row_index, column]
            ax.imshow(_montage(movie), cmap="gray", vmin=0, vmax=255, origin="lower")
            record = audit[(audit.example_number == row_index + 1) & (audit.control == ("raw_exact_power" if column == 1 else "histogram_matched"))]
            if column > 0:
                values = record.iloc[0]
                title += f"\namp err={values.full_3d_amplitude_relative_error:.1e}; OOR={100*values.out_of_0_255_fraction:.1f}%"
            ax.set_title(title + "\nt=5, 19, 34", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        for column, (control, radial_key, title) in enumerate(
            (
                ("raw_exact_power", "radial_raw", "Raw: Hann SF×TF difference"),
                ("histogram_matched", "radial_histogram", "Histogram: Hann SF×TF difference"),
            ),
            start=3,
        ):
            record = audit[(audit.example_number == row_index + 1) & (audit.control == control)].iloc[0]
            delta_db = relative_db(payload[radial_key]) - relative_db(payload["radial_intact"])
            ax = axes[row_index, column]
            mesh = ax.pcolormesh(sf_centers, tf_hz, delta_db, cmap="coolwarm", vmin=-12, vmax=12, shading="nearest")
            ax.set_xscale("log")
            ax.set_xlabel("SF (cpd)")
            ax.set_ylabel("TF (Hz)")
            ax.set_title(
                f"{title}\nHann cos={record.hann_supported_power_cosine:.3f}; ratio={record.hann_supported_power_ratio:.2f}",
                fontsize=9,
            )
        axes[row_index, 0].set_ylabel(
            f"{payload['selection_role']}\nimage {payload['image_index']}",
            fontsize=9,
        )
    assert mesh is not None
    fig.colorbar(mesh, ax=axes[:, 3:].ravel().tolist(), label="control minus intact relative power (dB)")
    fig.suptitle(
        "Checkpoint 44: global 3-D retinal-movie phase scramble\n"
        "Raw branch exactly matches unwindowed spatiotemporal power; histogram branch exposes range tradeoff; no neural scoring",
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
    trace = pd.read_csv(trace_path)
    selections.to_csv(out_dir / "selected_images.csv", index=False)
    trace.to_csv(out_dir / "selected_trace.csv", index=False)

    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for example_number, selection in enumerate(selections.itertuples(index=False), start=1):
        image_index = int(selection.image_index)
        source_path = args.source_checkpoint / "data" / f"example_{example_number}_image_{image_index:03d}.npz"
        with np.load(source_path, allow_pickle=False) as data:
            intact = np.asarray(data["movie_intact"], dtype=np.float32)
            history = np.asarray(data["history_xy_deg"], dtype=np.float32)
            score = np.asarray(data["score_xy_deg"], dtype=np.float32)
            trace_index = int(data["trace_index"].item())
        seed = int(args.seed) + image_index * 1009
        raw = global_3d_phase_scramble(intact, np.random.default_rng(seed))
        histogram = rank_histogram_match(intact, raw)
        radial_intact, _, _ = spectral_statistics(intact, ppd=PPD)
        for control, movie in (("raw_exact_power", raw), ("histogram_matched", histogram)):
            power, radial = power_audit(intact, movie, ppd=PPD)
            rows.append(
                {
                    "example_number": example_number,
                    "selection_role": selection.selection_role,
                    "image_index": image_index,
                    "trace_index": trace_index,
                    "seed": seed,
                    "control": control,
                    "full_3d_amplitude_relative_error": amplitude_relative_error(intact, movie),
                    **phase_relationship_audit(intact, movie),
                    **histogram_audit(intact, movie),
                    **movie_audit(intact, movie),
                    **power,
                }
            )
            if control == "raw_exact_power":
                radial_raw = radial
            else:
                radial_histogram = radial
        payloads.append(
            {
                "selection_role": selection.selection_role,
                "image_index": image_index,
                "intact": intact,
                "raw": raw,
                "histogram": histogram,
                "radial_intact": radial_intact,
                "radial_raw": radial_raw,
                "radial_histogram": radial_histogram,
            }
        )
        data_path = out_dir / "data" / f"example_{example_number}_image_{image_index:03d}.npz"
        np.savez_compressed(
            data_path,
            movie_intact=intact,
            movie_global_3d_phase_scrambled_raw=raw,
            movie_global_3d_phase_scrambled_histogram_matched=histogram,
            history_xy_deg=history,
            score_xy_deg=score,
            radial_power_intact=radial_intact,
            radial_power_raw=radial_raw,
            radial_power_histogram_matched=radial_histogram,
            image_index=np.asarray(image_index),
            trace_index=np.asarray(trace_index),
            seed=np.asarray(seed),
        )
        movie_path = out_dir / "movies" / f"example_{example_number}_image_{image_index:03d}_global_3d_phase_scramble.mp4"
        _write_mp4(
            _video_frames(
                intact,
                raw,
                histogram,
                title=f"Example {example_number}: image {image_index}, trace {trace_index}",
            ),
            movie_path,
            fps=int(args.fps),
        )

    audit = pd.DataFrame(rows)
    audit_path = out_dir / "global_3d_phase_scramble_audit.csv"
    audit.to_csv(audit_path, index=False)
    figure_path = out_dir / "global_3d_phase_scramble_checkpoint.png"
    plot_checkpoint(payloads, audit, figure_path)
    raw = audit[audit.control.eq("raw_exact_power")]
    histogram = audit[audit.control.eq("histogram_matched")]
    manifest = {
        "analysis": "rr100_global_3d_retinal_movie_phase_scramble_input_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_only_human_checkpoint_no_neural_scoring",
        "raw_contract": (
            "random phase from a real white-noise volume is combined with the intact retinal movie's full "
            "3-D Fourier amplitude; mean and variance retained; exact unwindowed spatiotemporal power"
        ),
        "histogram_contract": (
            "raw surrogate ranks receive the sorted intact movie pixels, giving exact empirical histogram "
            "and range but relaxing the Fourier-amplitude match"
        ),
        "n_examples": int(len(payloads)),
        "seed": int(args.seed),
        "maximum_raw_full_3d_amplitude_relative_error": float(raw.full_3d_amplitude_relative_error.max()),
        "maximum_raw_fourier_phase_retention_coherence": float(raw.fourier_phase_retention_coherence.max()),
        "maximum_raw_adjacent_frequency_phase_relation_retention_coherence": float(
            raw[
                [
                    "adjacent_temporal_frequency_phase_relation_retention_coherence",
                    "adjacent_vertical_spatial_frequency_phase_relation_retention_coherence",
                    "adjacent_horizontal_spatial_frequency_phase_relation_retention_coherence",
                ]
            ].to_numpy().max()
        ),
        "maximum_raw_out_of_range_fraction": float(raw.out_of_0_255_fraction.max()),
        "maximum_histogram_amplitude_relative_error": float(histogram.full_3d_amplitude_relative_error.max()),
        "maximum_histogram_sorted_rmse": float(histogram.histogram_sorted_rmse.max()),
        "median_raw_hann_supported_power_ratio": float(raw.hann_supported_power_ratio.median()),
        "median_histogram_hann_supported_power_ratio": float(histogram.hann_supported_power_ratio.median()),
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
        "next_checkpoint_if_approved": (
            "choose raw exact-power versus an iterative amplitude+histogram compromise before any twin scoring; "
            "only then consider model-radius pooled-energy synthesis"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Checkpoint 44: global 3-D retinal-movie phase scramble\n\n"
        "Each intact 40×51×51 retinal movie is scrambled as one spatiotemporal volume. The raw branch "
        "exactly retains the complete unwindowed 3-D Fourier amplitude, mean, and variance. The rank-matched "
        "branch exactly retains the intact empirical pixel histogram and range, while its power mismatch is "
        "audited rather than hidden.\n\n"
        "Review the checkpoint figure and three MP4s. No digital-twin response or SSI was computed.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
