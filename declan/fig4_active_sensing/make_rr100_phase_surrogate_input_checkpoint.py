#!/usr/bin/env python3
"""Build the map-first input checkpoint for RR100 phase-surrogate controls.

This script deliberately stops before neural-model scoring.  It compares two
phase manipulations on the three auditable response examples selected by the
current FEM power-routing figure series:

1. a temporally coherent, spatially localized all-pass filter whose Fourier
   magnitude is exactly one, hence preserving the unwindowed 3-D movie power;
2. the existing local complex-steerable-pyramid source-image scramble.

The script also materializes clipped and contrast-only controls so that loss
of in-range contrast is visible before any SSI comparison is attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.input_only_retinal_renderer import (
    render_retinal_frames_lag_zero,
)
from declan.fig4_active_sensing.spectral_cache_contract import (
    validate_artifact_not_superseded,
    validate_spectral_cache,
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
from jake.twininfo.retinal_examples import pyramid_local_image_controls


ROOT = Path(__file__).resolve().parents[2]
SELECTIONS = ROOT / (
    "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1/"
    "03_response_examples/selected_response_examples.csv"
)
INPUT_CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/input_cache"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_phase_surrogate_input_checkpoint_40_v1"
N_SCORE = 40
PPD = 37.5047661706098
EPS = np.finfo(np.float64).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, default=SELECTIONS)
    parser.add_argument("--input-cache", type=Path, default=INPUT_CACHE)
    parser.add_argument(
        "--spectral-cache", type=Path, required=True,
        help="Explicit frozen corrected spectral cache; superseded caches are rejected.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--kernel-sigma-px", type=float, default=2.5)
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


def localized_allpass_transfer(
    shape: tuple[int, int],
    *,
    sigma_px: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return a Hermitian unit-magnitude transfer and its localized impulse.

    A real Gaussian-windowed noise kernel supplies the phase.  Normalizing its
    Fourier transform to unit magnitude turns it into an all-pass filter.  The
    inverse transform is not strictly compact, so its empirical energy radii
    are recorded rather than assuming locality from the generating envelope.
    """
    height, width = (int(shape[0]), int(shape[1]))
    if height < 3 or width < 3 or sigma_px <= 0:
        raise ValueError(f"Invalid all-pass configuration: shape={shape}, sigma={sigma_px}")
    yy, xx = np.mgrid[:height, :width]
    cy, cx = height // 2, width // 2
    radius = np.hypot(yy - cy, xx - cx)
    envelope = np.exp(-(radius**2) / (2.0 * float(sigma_px) ** 2))
    rng = np.random.default_rng(int(seed))
    generating_kernel = rng.normal(size=shape) * envelope
    spectrum = np.fft.fft2(np.fft.ifftshift(generating_kernel))
    transfer = spectrum / np.maximum(np.abs(spectrum), EPS)
    transfer[0, 0] = 1.0 + 0.0j
    impulse = np.fft.fftshift(np.fft.ifft2(transfer).real)
    energy = impulse**2
    total = max(float(energy.sum()), EPS)

    def energy_radius(fraction: float) -> int:
        for candidate in range(max(height, width)):
            if float(energy[radius <= candidate].sum()) / total >= fraction:
                return candidate
        return max(height, width) - 1

    audit = {
        "kernel_sigma_px": float(sigma_px),
        "transfer_magnitude_relative_error": float(
            np.linalg.norm(np.abs(transfer) - 1.0) / np.sqrt(transfer.size)
        ),
        "impulse_energy_radius_50_px": float(energy_radius(0.50)),
        "impulse_energy_radius_80_px": float(energy_radius(0.80)),
        "impulse_energy_radius_90_px": float(energy_radius(0.90)),
        "impulse_energy_radius_95_px": float(energy_radius(0.95)),
        "impulse_energy_radius_90_deg": float(energy_radius(0.90) / PPD),
        "impulse_center_value": float(impulse[cy, cx]),
    }
    return transfer, impulse, audit


def apply_coherent_allpass(movie: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    """Apply one spatial all-pass filter to every frame of a T x H x W movie."""
    arr = np.asarray(movie, dtype=np.float64)
    if arr.ndim != 3 or transfer.shape != arr.shape[1:]:
        raise ValueError(f"Incompatible movie/transfer shapes: {arr.shape}, {transfer.shape}")
    transformed = np.fft.fft2(arr, axes=(1, 2)) * transfer[None, :, :]
    return np.fft.ifft2(transformed, axes=(1, 2)).real.astype(np.float32)


def spatial_contrast_rms(movie: np.ndarray) -> float:
    arr = np.asarray(movie, dtype=np.float64)
    centered = arr - arr.mean(axis=(1, 2), keepdims=True)
    return float(np.sqrt(np.mean(centered**2)))


def contrast_match_intact(intact: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale intact framewise contrast to target while preserving phase and means."""
    source = np.asarray(intact, dtype=np.float64)
    means = source.mean(axis=(1, 2), keepdims=True)
    scale = spatial_contrast_rms(target) / max(spatial_contrast_rms(source), EPS)
    matched = means + scale * (source - means)
    return matched.astype(np.float32), float(scale)


def full_st_amplitude_relative_error(intact: np.ndarray, control: np.ndarray) -> float:
    a = np.abs(np.fft.fftn(np.asarray(intact, dtype=np.float64)))
    b = np.abs(np.fft.fftn(np.asarray(control, dtype=np.float64)))
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(a)), EPS))


def frame_spatial_amplitude_relative_error(intact: np.ndarray, control: np.ndarray) -> float:
    a = np.abs(np.fft.fft2(np.asarray(intact, dtype=np.float64), axes=(1, 2)))
    b = np.abs(np.fft.fft2(np.asarray(control, dtype=np.float64), axes=(1, 2)))
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(a)), EPS))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(aa, bb) / max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), EPS))


def centered_pixel_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa - aa.mean(axis=(1, 2), keepdims=True)
    bb = bb - bb.mean(axis=(1, 2), keepdims=True)
    if float(aa.std()) <= 0 or float(bb.std()) <= 0:
        return float("nan")
    return float(np.corrcoef(aa.ravel(), bb.ravel())[0, 1])


def power_audit(
    intact: np.ndarray,
    control: np.ndarray,
    *,
    ppd: float,
) -> tuple[dict[str, float], np.ndarray]:
    radial_intact, _, _ = spectral_statistics(intact, ppd=ppd)
    radial_control, _, _ = spectral_statistics(control, ppd=ppd)
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]
    sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    supported = (
        (tf_hz[:, None] > 0)
        & (tf_hz[:, None] <= 56.0)
        & (sf_centers[None, :] >= 1.0)
        & (sf_centers[None, :] <= 11.3137085)
    )
    a = radial_intact.astype(np.float64)
    b = radial_control.astype(np.float64)
    a_supported = a[supported]
    b_supported = b[supported]
    audit = {
        "unwindowed_full_st_amplitude_relative_error": full_st_amplitude_relative_error(intact, control),
        "unwindowed_frame_spatial_amplitude_relative_error": frame_spatial_amplitude_relative_error(intact, control),
        "hann_radial_power_relative_l2_error": float(np.linalg.norm(b - a) / max(float(np.linalg.norm(a)), EPS)),
        "hann_radial_power_cosine": cosine_similarity(a, b),
        "hann_total_positive_tf_power_ratio": float(b.sum() / max(float(a.sum()), EPS)),
        "hann_supported_power_relative_l2_error": float(
            np.linalg.norm(b_supported - a_supported) / max(float(np.linalg.norm(a_supported)), EPS)
        ),
        "hann_supported_power_cosine": cosine_similarity(a_supported, b_supported),
        "hann_supported_power_ratio": float(b_supported.sum() / max(float(a_supported.sum()), EPS)),
    }
    return audit, radial_control


def movie_audit(intact: np.ndarray, control: np.ndarray) -> dict[str, float]:
    arr = np.asarray(control, dtype=np.float64)
    return {
        "movie_mean": float(arr.mean()),
        "movie_sd": float(arr.std()),
        "spatial_contrast_rms": spatial_contrast_rms(arr),
        "minimum": float(arr.min()),
        "maximum": float(arr.max()),
        "out_of_0_255_fraction": float(np.mean((arr < 0.0) | (arr > 255.0))),
        "centered_pixel_correlation_with_intact": centered_pixel_correlation(intact, arr),
    }


def load_inputs(
    selections_path: Path,
    input_cache: Path,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, float], np.ndarray, np.ndarray, np.ndarray]:
    selections = pd.read_csv(selections_path)
    required = {
        "condition_row",
        "selection_role",
        "image_index",
        "trace_index",
        "round_index",
        "rr100_index",
    }
    if missing := required.difference(selections.columns):
        raise ValueError(f"Selection table lacks columns: {sorted(missing)}")
    if len(selections) < 3:
        raise ValueError("Expected agreement and two dissociation examples")
    patches: dict[int, np.ndarray] = {}
    ppds: dict[int, float] = {}
    for image_index in selections.image_index.astype(int).unique():
        path = input_cache / "images" / f"image_{image_index:03d}.npz"
        with np.load(path, allow_pickle=False) as data:
            patches[image_index] = _standardize_uint_like(
                np.asarray(data["corrected_patch"], dtype=np.float32)
            )
            ppds[image_index] = float(data["patch_ppd"].item())
    with np.load(input_cache / "corrected_trace_segments.npz", allow_pickle=False) as data:
        trace_ids = np.asarray(data["trace_index"], dtype=int)
        history = np.asarray(data["history_xy_deg"], dtype=np.float32)
        score = np.asarray(data["score_xy_deg"], dtype=np.float32)
    return selections, patches, ppds, trace_ids, history, score


def relative_db(power: np.ndarray) -> np.ndarray:
    arr = np.asarray(power, dtype=np.float64)
    scale = max(float(np.max(arr)), EPS)
    return 10.0 * np.log10(np.maximum(arr / scale, 1e-5))


def montage(movie: np.ndarray, frames: tuple[int, ...] = (5, 19, 34)) -> np.ndarray:
    return np.concatenate([np.asarray(movie[index]) for index in frames], axis=1)


def plot_movie_examples(
    selections: pd.DataFrame,
    movies: dict[int, dict[str, np.ndarray]],
    audit: pd.DataFrame,
    out_dir: Path,
) -> None:
    methods = [
        ("intact", "Intact FEM movie"),
        ("allpass_raw", "Localized all-pass\nraw power-exact"),
        ("allpass_clipped", "All-pass clipped\nto [0,255]"),
        ("contrast_to_allpass", "Intact phase\ncontrast matched"),
        ("pyramid_source", "Existing local pyramid\nsource scramble"),
        ("contrast_to_pyramid", "Intact phase\npyramid-contrast matched"),
    ]
    fig, axes = plt.subplots(
        len(selections),
        len(methods),
        figsize=(18.5, 3.5 * len(selections)),
        constrained_layout=True,
    )
    for row_index, selection in enumerate(selections.itertuples(index=False)):
        condition = int(selection.condition_row)
        for column, (method, title) in enumerate(methods):
            axis = axes[row_index, column]
            movie = movies[condition][method]
            axis.imshow(montage(movie), cmap="gray", vmin=0, vmax=255, origin="lower")
            axis.set_xticks([25, 76, 127], ["t=5", "t=19", "t=34"])
            axis.set_yticks([])
            record = audit[
                audit.condition_row.eq(condition) & audit.control.eq(method)
            ].iloc[0]
            axis.set_title(
                f"{title}\ncontrast={record.spatial_contrast_rms:.1f}, "
                f"OOR={100 * record.out_of_0_255_fraction:.1f}%",
                fontsize=9,
            )
            if column == 0:
                axis.set_ylabel(
                    f"{selection.selection_role}\n"
                    f"img {int(selection.image_index)}, trace {int(selection.trace_index)}\n"
                    f"RR100 {int(selection.rr100_index)}",
                    fontsize=9,
                )
    fig.suptitle(
        "Input checkpoint: phase destruction, clipping, and matched-contrast controls\n"
        "Three frames per 40-frame scored movie; no neural response has been recomputed",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(out_dir / "phase_surrogate_movie_examples.png", dpi=190)
    fig.savefig(out_dir / "phase_surrogate_movie_examples.pdf")
    plt.close(fig)


def plot_power_audit(
    selections: pd.DataFrame,
    radial: dict[int, dict[str, np.ndarray]],
    audit: pd.DataFrame,
    out_dir: Path,
) -> None:
    methods = [
        ("intact", "intact"),
        ("allpass_raw", "all-pass raw"),
        ("allpass_clipped", "all-pass clipped"),
        ("pyramid_source", "pyramid source"),
    ]
    sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]
    fig, axes = plt.subplots(
        len(selections),
        len(methods),
        figsize=(13.8, 3.55 * len(selections)),
        constrained_layout=True,
    )
    for row_index, selection in enumerate(selections.itertuples(index=False)):
        condition = int(selection.condition_row)
        for column, (method, label) in enumerate(methods):
            axis = axes[row_index, column]
            values = radial[condition][method]
            mesh = axis.pcolormesh(
                sf_centers,
                tf_hz,
                relative_db(values),
                cmap="magma",
                shading="nearest",
                vmin=-45,
                vmax=0,
            )
            axis.set_xscale("log")
            axis.set_xlabel("SF (cpd)")
            axis.set_ylabel("TF (Hz)")
            record = audit[
                audit.condition_row.eq(condition) & audit.control.eq(method)
            ].iloc[0]
            axis.set_title(
                f"{label}\ncurrent-power cosine={record.hann_supported_power_cosine:.3f}, "
                f"ratio={record.hann_supported_power_ratio:.2f}",
                fontsize=9,
            )
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="within-condition relative power (dB)")
    fig.suptitle(
        "Audit against the current 40-frame Hann-windowed SF×TF statistic\n"
        "Raw all-pass power is exact before windowing; the displayed statistic also includes a spatial Hann",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(out_dir / "phase_surrogate_power_audit.png", dpi=190)
    fig.savefig(out_dir / "phase_surrogate_power_audit.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_spectral_cache(args.spectral_cache)
    validate_artifact_not_superseded(args.selections, label="condition-selection table")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    selections, patches, ppds, trace_ids, history, score = load_inputs(
        args.selections, args.input_cache
    )
    selections.to_csv(args.out_dir / "selected_conditions.csv", index=False)
    transfer, impulse, kernel_audit = localized_allpass_transfer(
        (51, 51), sigma_px=float(args.kernel_sigma_px), seed=int(args.seed)
    )
    np.savez_compressed(
        args.out_dir / "localized_allpass_filter.npz",
        transfer=transfer,
        impulse=impulse,
        **{key: np.asarray(value) for key, value in kernel_audit.items()},
    )
    common = _load_twin_common()
    trace_position = {int(value): position for position, value in enumerate(trace_ids)}
    all_movies: dict[int, dict[str, np.ndarray]] = {}
    all_radial: dict[int, dict[str, np.ndarray]] = {}
    audit_rows: list[dict[str, Any]] = []
    pyramid_rows: list[dict[str, Any]] = []

    for selection in selections.itertuples(index=False):
        condition = int(selection.condition_row)
        image_index = int(selection.image_index)
        trace_index = int(selection.trace_index)
        position = trace_position[trace_index]
        ppd = float(ppds[image_index])
        if not np.isclose(ppd, PPD):
            raise ValueError(f"Unexpected PPD for image {image_index}: {ppd}")
        scored_retinal_trace = -score[position]
        full_retinal_trace = -np.concatenate([history[position], score[position]], axis=0)
        intact = (
            render_retinal_frames_lag_zero(
                common,
                patches[image_index],
                scored_retinal_trace,
                ppd=ppd,
                device="cpu",
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        allpass_raw = apply_coherent_allpass(intact, transfer)
        allpass_clipped = np.clip(allpass_raw, 0.0, 255.0).astype(np.float32)
        contrast_to_allpass, allpass_contrast_scale = contrast_match_intact(
            intact, allpass_clipped
        )

        pyramid_controls, pyramid_audits = pyramid_local_image_controls(
            patches[image_index],
            full_retinal_trace,
            np.random.default_rng(int(args.seed) + condition),
            clip=True,
            ppd=ppd,
            out_size=(51, 51),
            height=3,
            order=3,
            sf_bands=(),
        )
        pyramid_source = (
            render_retinal_frames_lag_zero(
                common,
                pyramid_controls["pyramid_phase_scrambled"],
                scored_retinal_trace,
                ppd=ppd,
                device="cpu",
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        contrast_to_pyramid, pyramid_contrast_scale = contrast_match_intact(
            intact, pyramid_source
        )
        movies = {
            "intact": intact,
            "allpass_raw": allpass_raw,
            "allpass_clipped": allpass_clipped,
            "contrast_to_allpass": contrast_to_allpass,
            "pyramid_source": pyramid_source,
            "contrast_to_pyramid": contrast_to_pyramid,
        }
        all_movies[condition] = movies
        all_radial[condition] = {}
        for control, movie in movies.items():
            power_metrics, radial_values = power_audit(intact, movie, ppd=ppd)
            all_radial[condition][control] = radial_values
            extra = {
                "contrast_scale_from_intact": (
                    allpass_contrast_scale
                    if control == "contrast_to_allpass"
                    else pyramid_contrast_scale
                    if control == "contrast_to_pyramid"
                    else 1.0
                )
            }
            audit_rows.append(
                {
                    "condition_row": condition,
                    "selection_role": selection.selection_role,
                    "image_index": image_index,
                    "trace_index": trace_index,
                    "round_index": int(selection.round_index),
                    "rr100_index": int(selection.rr100_index),
                    "control": control,
                    **movie_audit(intact, movie),
                    **power_metrics,
                    **extra,
                }
            )
        pyramid_rows.append(
            {
                "condition_row": condition,
                "selection_role": selection.selection_role,
                "image_index": image_index,
                "trace_index": trace_index,
                **pyramid_audits[0],
            }
        )
        np.savez_compressed(
            args.out_dir / f"condition_{condition:04d}_movies_and_power.npz",
            scored_retinal_trace_xy_deg=scored_retinal_trace,
            full_retinal_trace_xy_deg=full_retinal_trace,
            sf_edges_cpd=SF_EDGES_CPD,
            tf_hz=np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:],
            **{f"movie_{key}": value for key, value in movies.items()},
            **{f"radial_power_{key}": value for key, value in all_radial[condition].items()},
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.out_dir / "phase_surrogate_control_audit.csv", index=False)
    pd.DataFrame(pyramid_rows).to_csv(
        args.out_dir / "steerable_pyramid_source_audit.csv", index=False
    )
    pd.DataFrame([kernel_audit]).to_csv(
        args.out_dir / "localized_allpass_filter_audit.csv", index=False
    )
    plot_movie_examples(selections, all_movies, audit, args.out_dir)
    plot_power_audit(selections, all_radial, audit, args.out_dir)

    raw = audit[audit.control.eq("allpass_raw")]
    clipped = audit[audit.control.eq("allpass_clipped")]
    pyramid = audit[audit.control.eq("pyramid_source")]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_manipulation_checkpoint_complete_stop_before_neural_scoring",
        "scope": {
            "conditions": int(len(selections)),
            "selected_roles": selections.selection_role.tolist(),
            "scored_frames_per_movie": N_SCORE,
            "neural_model_calls": False,
        },
        "hypothesis": (
            "If FEM-dependent SSI sharpening requires higher-order spatial phase alignment, "
            "it should weaken under phase-scrambled movies even when second-order movie power is retained."
        ),
        "counterevidence_prediction": (
            "Persistence of SSI sharpening under a genuinely power-matched phase surrogate would count "
            "against phase alignment being necessary."
        ),
        "allpass_contract": {
            "description": (
                "One localized real all-pass spatial filter is applied identically to every frame; "
                "unwindowed per-frame spatial and full movie spatiotemporal Fourier amplitudes are preserved."
            ),
            "seed": int(args.seed),
            **kernel_audit,
            "maximum_full_st_amplitude_relative_error": float(
                raw.unwindowed_full_st_amplitude_relative_error.max()
            ),
            "median_current_hann_supported_power_cosine": float(
                raw.hann_supported_power_cosine.median()
            ),
            "median_current_hann_supported_power_ratio": float(
                raw.hann_supported_power_ratio.median()
            ),
            "maximum_out_of_range_fraction": float(raw.out_of_0_255_fraction.max()),
        },
        "clipping_complication": {
            "median_current_hann_supported_power_cosine": float(
                clipped.hann_supported_power_cosine.median()
            ),
            "median_current_hann_supported_power_ratio": float(
                clipped.hann_supported_power_ratio.median()
            ),
            "contrast_matched_intact_control_materialized": True,
        },
        "existing_pyramid_contract": {
            "description": (
                "Complex steerable-pyramid coefficient phases are randomized only in the source ROI "
                "swept by the trace; source pixels outside the ROI are preserved."
            ),
            "median_current_hann_supported_power_cosine": float(
                pyramid.hann_supported_power_cosine.median()
            ),
            "median_current_hann_supported_power_ratio": float(
                pyramid.hann_supported_power_ratio.median()
            ),
            "contrast_matched_intact_control_materialized": True,
        },
        "guardrails": [
            "The exact all-pass claim refers to the unwindowed movie Fourier amplitude.",
            "The latest power-routing analysis uses temporal and spatial Hann windows, so its realized mismatch is audited separately.",
            "Raw power-exact movies may leave the [0,255] training range; clipped and contrast-only controls are shown explicitly.",
            "The steerable-pyramid coefficient-magnitude guarantee is pre-reconstruction and does not imply matched retinal SFxTF power.",
            "No SSI or neural response is inferred at this checkpoint.",
        ],
        "sources": {
            "selections": file_identity(args.selections),
            "trace_cache": file_identity(args.input_cache / "corrected_trace_segments.npz"),
            "spectral_manifest": file_identity(args.spectral_cache / "manifest.json"),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# RR100 phase-surrogate input checkpoint

This is the input/manipulation checkpoint only. It does not call the frozen
neural model and does not yet test SSI.

The localized all-pass branch is the pragmatic power-exact candidate. It uses
the same phase-only spatial filter at every timepoint, preserving coherent FEM
motion and the complete unwindowed 3-D Fourier amplitude. The raw version is
therefore the clean second-order control, while its out-of-range pixels are a
model-distribution caveat. The clipped version and its matched-contrast intact
control expose that caveat rather than hiding it.

The steerable-pyramid branch reuses the existing receptive-field-scale source
ROI control. Its coefficient magnitudes are preserved before reconstruction,
but the reconstructed retinal movie need not retain the current SF×TF power.

Read `phase_surrogate_movie_examples.pdf`, then
`phase_surrogate_power_audit.pdf`, and use
`phase_surrogate_control_audit.csv` for exact values.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
