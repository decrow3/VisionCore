#!/usr/bin/env python3
"""Verify RF-local SFxorientationxTF routing on recorded grating movies.

This is an input/mechanism checkpoint.  It reuses the exact held-out retinal
grating movies and the unit-specific RF apertures from the three-way grating
analysis, but replaces the radial spectrum with the orientation-resolved
Fourier tensor and direct-positive-F0 orientation weights.  No response is fit
or scored here; that is deliberately reserved for the next checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.make_rr100_orientation_routing_input_checkpoint import (
    four_grating_channels,
)
from declan.fig4_active_sensing.make_rr100_recorded_grating_retinal_power_input_checkpoint import (
    candidate_windows,
    load_heldout_grating_dataset,
)
from declan.fig4_active_sensing.make_rr100_recorded_grating_three_way_response_checkpoint import (
    indices_for_support,
)
from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    FRAME_RATE_HZ,
    N_SCORE,
    ORIENTATION_EDGES_DEG,
    SF_EDGES_CPD,
    spatial_lookup,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/dataset_configs/multi_basic_120_long_legacy.yaml"
DEFAULT_RF = ROOT / "outputs/fig4_active_sensing/rr100_recorded_grating_three_way_response_rf_local_v2"
DEFAULT_TUNING = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_f0_map_checkpoint_v1"
DEFAULT_OUT = ROOT / "outputs/fig4_active_sensing/rr100_recorded_grating_oriented_power_input_checkpoint_v2"
GRATING_ORIENTATIONS = np.asarray([0.0, 45.0, 90.0, 135.0])
EPS = np.finfo(float).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rf-dir", type=Path, default=DEFAULT_RF)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--session", default="Logan_2020-02-29")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stride", type=int, default=N_SCORE)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def localized_oriented_spectrum(
    movie: np.ndarray, *, ppd: float, spatial_aperture: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-TF radial and orientation-resolved RF-local power."""
    arr = np.asarray(movie, dtype=np.float64)
    aperture = np.asarray(spatial_aperture, dtype=np.float64)
    if arr.shape != (N_SCORE, 51, 51) or aperture.shape != (51, 51):
        raise ValueError(f"Unexpected movie/aperture shapes {arr.shape}/{aperture.shape}")
    residual = arr - arr.mean(axis=0, keepdims=True)
    temporal_window = np.hanning(N_SCORE)[:, None, None]
    temporal_fft = np.fft.rfft(residual * temporal_window * aperture[None, :, :], axis=0)
    spectrum = np.fft.fftshift(np.fft.fft2(temporal_fft, axes=(1, 2)), axes=(1, 2))
    power = np.abs(spectrum) ** 2
    temporal_weights = np.ones(power.shape[0], dtype=np.float64)
    temporal_weights[1:-1] = 2.0
    power *= temporal_weights[:, None, None]
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)
    flat = power[tf_hz > 0].reshape(np.count_nonzero(tf_hz > 0), -1)
    sf_bin, orientation_bin, _ = spatial_lookup(float(ppd))
    n_sf = len(SF_EDGES_CPD) - 1
    n_orientation = len(ORIENTATION_EDGES_DEG) - 1
    joint_bin = sf_bin * n_orientation + orientation_bin
    radial = np.empty((len(flat), n_sf), dtype=np.float64)
    oriented = np.empty((len(flat), n_sf, n_orientation), dtype=np.float64)
    for index, values in enumerate(flat):
        radial[index] = np.bincount(sf_bin, weights=values, minlength=n_sf)
        oriented[index] = np.bincount(
            joint_bin, weights=values, minlength=n_sf * n_orientation
        ).reshape(n_sf, n_orientation)
    if not np.allclose(oriented.sum(axis=-1), radial, rtol=1e-10, atol=1e-8):
        raise ValueError("Orientation-resolved power does not reproduce radial power")
    return radial, oriented


def synthetic_drifting_grating(
    *, ppd: float, grating_orientation_deg: float, sf_cpd: float = 2.4, tf_hz: float = 12.0
) -> np.ndarray:
    coordinates = (np.arange(51, dtype=float) - 25.0) / float(ppd)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    wavevector_deg = (90.0 - float(grating_orientation_deg)) % 180.0
    wavevector = np.deg2rad(wavevector_deg)
    spatial_phase = float(sf_cpd) * (xx * np.cos(wavevector) + yy * np.sin(wavevector))
    time = np.arange(N_SCORE, dtype=float) / FRAME_RATE_HZ
    return np.cos(2.0 * np.pi * (spatial_phase[None, :, :] - float(tf_hz) * time[:, None, None]))


def synthetic_orientation_audit(ppd: float, apertures: dict[int, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for unit, aperture in apertures.items():
        for orientation in GRATING_ORIENTATIONS:
            movie = synthetic_drifting_grating(ppd=ppd, grating_orientation_deg=float(orientation))
            radial, oriented = localized_oriented_spectrum(movie, ppd=ppd, spatial_aperture=aperture)
            channels = four_grating_channels(oriented, ORIENTATION_EDGES_DEG).sum(axis=(0, 1))
            predicted = float(GRATING_ORIENTATIONS[int(np.argmax(channels))])
            expected_index = int(np.flatnonzero(np.isclose(GRATING_ORIENTATIONS, orientation))[0])
            rows.append(
                {
                    "rr100_index": int(unit),
                    "presented_grating_orientation_deg": float(orientation),
                    "maximum_power_channel_deg": predicted,
                    "orientation_channel_correct": bool(np.isclose(predicted, orientation)),
                    "expected_channel_power_fraction": float(channels[expected_index] / max(channels.sum(), EPS)),
                    "radial_reproduction_relative_error": float(
                        np.max(np.abs(oriented.sum(axis=-1) - radial)) / max(float(radial.max()), EPS)
                    ),
                }
            )
    return pd.DataFrame(rows)


def display_channel_fractions(displayed_orientation_deg: np.ndarray) -> np.ndarray:
    """Assign arbitrary grating-bar angles to the four measured neural channels."""
    displayed = np.mod(np.asarray(displayed_orientation_deg, dtype=float).ravel(), 180.0)
    difference = np.abs(displayed[:, None] - GRATING_ORIENTATIONS[None, :]) % 180.0
    distance = np.minimum(difference, 180.0 - difference)
    minimum = distance.min(axis=1, keepdims=True)
    assignment = np.isclose(distance, minimum).astype(float)
    assignment /= assignment.sum(axis=1, keepdims=True)
    return assignment.mean(axis=0)


def load_rf_and_tuning(
    rf_dir: Path, tuning_dir: Path, session: str
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    units = pd.read_csv(rf_dir / "unit_rf_metadata.csv")
    units = units.loc[units.session.eq(session)].sort_values("rr100_index").reset_index(drop=True)
    with np.load(rf_dir / "unit_rf_apertures.npz", allow_pickle=False) as archive:
        aperture_units = archive["rr100_index"].astype(int)
        aperture_values = archive["spectral_aperture"].astype(float)
    apertures = {int(unit): aperture_values[index] for index, unit in enumerate(aperture_units)}
    tuning_path = tuning_dir / "orientation_aware_f0_tuning_and_routing.npz"
    with np.load(tuning_path, allow_pickle=False) as archive:
        tuning_units = archive["rr100_index"].astype(int)
        sf = archive["movie_sf_cpd"].astype(float)
        tf = archive["movie_tf_hz"].astype(float)
        fourier_orientation = archive["movie_fourier_orientation_deg"].astype(float)
        radial_values = archive["smoothed_radial_f0_weight"].astype(float)
        oriented_values = archive["orientation_aware_f0_weight"].astype(float)
    expected_orientation = 0.5 * (ORIENTATION_EDGES_DEG[:-1] + ORIENTATION_EDGES_DEG[1:])
    if not np.allclose(fourier_orientation, expected_orientation):
        raise ValueError("Tuning and Fourier orientation axes do not match")
    radial_weights = {int(unit): radial_values[index] for index, unit in enumerate(tuning_units)}
    oriented_weights = {int(unit): oriented_values[index] for index, unit in enumerate(tuning_units)}
    missing = [int(unit) for unit in units.rr100_index if int(unit) not in apertures or int(unit) not in oriented_weights]
    if missing:
        raise ValueError(f"Missing RF aperture or oriented tuning for units {missing}")
    for unit in units.rr100_index.astype(int):
        if not np.allclose(oriented_weights[unit].mean(axis=-1), radial_weights[unit], rtol=2e-6, atol=1e-8):
            raise ValueError(f"Orientation weights fail radial nesting for RR100 {unit}")
    return units, apertures, radial_weights, oriented_weights, sf, tf


def build_metrics(
    window_metrics: pd.DataFrame,
    payload: dict[int, dict[str, np.ndarray]],
    units: pd.DataFrame,
    apertures: dict[int, np.ndarray],
    radial_weights: dict[int, np.ndarray],
    oriented_weights: dict[int, np.ndarray],
    tuning_sf: np.ndarray,
    tuning_tf: np.ndarray,
    candidate_sf: np.ndarray,
    candidate_tf: np.ndarray,
    ppd: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sf_index = indices_for_support(candidate_sf, tuning_sf, "SF")
    tf_index = indices_for_support(candidate_tf, tuning_tf, "TF")
    rows: list[dict[str, object]] = []
    radial_spectra: list[np.ndarray] = []
    oriented_spectra: list[np.ndarray] = []
    for window in window_metrics.itertuples(index=False):
        item = payload[int(window.window_index)]
        movie = (item["movie_uint8"].astype(np.float32) - 127.0) / 255.0
        displayed = np.mod(item["orientation_deg"].astype(float), 180.0)
        display_fraction = display_channel_fractions(displayed)
        for unit in units.itertuples(index=False):
            rr100_index = int(unit.rr100_index)
            radial_full, oriented_full = localized_oriented_spectrum(
                movie, ppd=float(ppd), spatial_aperture=apertures[rr100_index]
            )
            radial = radial_full[np.ix_(tf_index, sf_index)]
            oriented = oriented_full[tf_index][:, sf_index, :]
            radial_weight = radial_weights[rr100_index]
            oriented_weight = oriented_weights[rr100_index]
            radial_map = radial * radial_weight
            oriented_map = np.sum(oriented * oriented_weight, axis=-1)
            radial_drive = float(radial_map.sum())
            oriented_drive = float(oriented_map.sum())
            channels = four_grating_channels(oriented, ORIENTATION_EDGES_DEG).sum(axis=(0, 1))
            channel_fraction = channels / max(float(channels.sum()), EPS)
            dominant_display = int(np.argmax(display_fraction))
            row_position = len(rows)
            rows.append(
                {
                    "array_row": row_position,
                    "window_index": int(window.window_index),
                    "trial_index": int(window.trial_index),
                    "start_index_120hz": int(window.start_index_120hz),
                    "rr100_index": rr100_index,
                    "radial_direct_f0_drive": radial_drive,
                    "oriented_direct_f0_drive": oriented_drive,
                    "orientation_delta_drive": oriented_drive - radial_drive,
                    "orientation_to_radial_ratio": oriented_drive / max(radial_drive, EPS),
                    "log2_orientation_to_radial": float(np.log2(max(oriented_drive, EPS) / max(radial_drive, EPS))),
                    "dominant_display_grating_orientation_deg": float(GRATING_ORIENTATIONS[dominant_display]),
                    "dominant_display_fraction": float(display_fraction[dominant_display]),
                    "matched_fourier_channel_fraction": float(channel_fraction[dominant_display]),
                    **{f"display_fraction_{int(value)}deg": float(display_fraction[index]) for index, value in enumerate(GRATING_ORIENTATIONS)},
                    **{f"power_fraction_{int(value)}deg": float(channel_fraction[index]) for index, value in enumerate(GRATING_ORIENTATIONS)},
                }
            )
            radial_spectra.append(radial.astype(np.float32))
            oriented_spectra.append(oriented.astype(np.float32))
    return pd.DataFrame(rows), np.stack(radial_spectra), np.stack(oriented_spectra)


def select_examples(metrics: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    used: set[int] = set()

    def add(role: str, values: pd.Series, criterion: str, direction: str) -> None:
        available = metrics.loc[~metrics.array_row.isin(used)].copy()
        score = values.reindex(available.index)
        index = score.idxmax() if direction == "max" else score.idxmin()
        row = metrics.loc[index].copy()
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        row["selection_value"] = float(values.loc[index])
        selected.append(row)
        used.add(int(row.array_row))

    add(
        "orientation-aligned gain",
        metrics.log2_orientation_to_radial,
        "largest log2 oriented/radial direct-F0 drive",
        "max",
    )
    add(
        "orientation-mismatch loss",
        metrics.log2_orientation_to_radial,
        "smallest log2 oriented/radial direct-F0 drive",
        "min",
    )
    add(
        "radial-equivalent control",
        -metrics.log2_orientation_to_radial.abs(),
        "smallest absolute log2 oriented/radial direct-F0 drive",
        "max",
    )
    add(
        "input orientation agreement",
        metrics.matched_fourier_channel_fraction,
        "largest RF-local Fourier-power fraction in the dominant displayed grating channel",
        "max",
    )
    return pd.DataFrame(selected)


def relative_db(values: np.ndarray, maximum: float) -> np.ndarray:
    return np.maximum(10.0 * np.log10(np.maximum(values, EPS) / max(float(maximum), EPS)), -50.0)


def plot_checkpoint(
    selected: pd.DataFrame,
    payload: dict[int, dict[str, np.ndarray]],
    units: pd.DataFrame,
    apertures: dict[int, np.ndarray],
    radial_weights: dict[int, np.ndarray],
    oriented_weights: dict[int, np.ndarray],
    radial_spectra: np.ndarray,
    oriented_spectra: np.ndarray,
    sf: np.ndarray,
    tf: np.ndarray,
    out: Path,
    dpi: int,
) -> None:
    unit_rows = units.set_index("rr100_index")
    figure, axes = plt.subplots(len(selected), 7, figsize=(26, 3.55 * len(selected)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_number, selection in enumerate(selected.itertuples(index=False)):
        array_row = int(selection.array_row)
        unit = int(selection.rr100_index)
        item = payload[int(selection.window_index)]
        movie = item["movie_uint8"]
        aperture = apertures[unit]
        radial = radial_spectra[array_row].astype(float)
        oriented = oriented_spectra[array_row].astype(float)
        radial_map = radial * radial_weights[unit]
        oriented_map = np.sum(oriented * oriented_weights[unit], axis=-1)
        difference = oriented_map - radial_map
        maximum = max(float(radial_map.max()), float(oriented_map.max()), EPS)

        axes[row_number, 0].imshow(movie[len(movie) // 2], cmap="gray", vmin=0, vmax=255)
        axes[row_number, 0].imshow(aperture, cmap="viridis", alpha=0.48 * aperture / max(float(aperture.max()), EPS))
        axes[row_number, 0].set_title(
            f"observed retinal frame + RF aperture\nRR100 {unit}; window {int(selection.window_index)}"
        )
        axes[row_number, 0].axis("off")

        t_ms = np.arange(N_SCORE) / FRAME_RATE_HZ * 1000.0
        color = np.mod(item["orientation_deg"], 180.0)
        scatter = axes[row_number, 1].scatter(t_ms, item["sf_cpd"], c=color, cmap="hsv", vmin=0, vmax=180, s=22)
        axes[row_number, 1].set_yscale("symlog", linthresh=0.5)
        axes[row_number, 1].set(
            xlabel="time (ms)", ylabel="displayed SF (cpd)", title=f"observed display sequence\n{selection.selection_role}"
        )
        figure.colorbar(scatter, ax=axes[row_number, 1], label="grating orientation", fraction=0.046)

        x = np.arange(4)
        display_fraction = np.asarray([getattr(selection, f"display_fraction_{int(value)}deg") for value in GRATING_ORIENTATIONS])
        power_fraction = np.asarray([getattr(selection, f"power_fraction_{int(value)}deg") for value in GRATING_ORIENTATIONS])
        axes[row_number, 2].bar(x - 0.18, display_fraction, width=0.36, color="0.55", label="displayed frames")
        axes[row_number, 2].bar(x + 0.18, power_fraction, width=0.36, color="#009E73", label="RF-local power")
        axes[row_number, 2].set_xticks(x, [f"{value:g}°" for value in GRATING_ORIENTATIONS])
        axes[row_number, 2].set_ylim(0, 1)
        axes[row_number, 2].set(ylabel="fraction", title="observed labels vs derived power")
        axes[row_number, 2].legend(frameon=False, fontsize=7)

        for column, values, title in (
            (3, radial_map, "radial SF×TF accepted power"),
            (4, oriented_map, "oriented SF×θ×TF accepted power"),
        ):
            image = axes[row_number, column].imshow(
                relative_db(values, maximum), origin="lower", aspect="auto", cmap="magma", vmin=-50, vmax=0
            )
            axes[row_number, column].set_xticks(range(len(sf)), [f"{value:.2g}" for value in sf], rotation=45)
            axes[row_number, column].set_yticks([0, 5, 10, 15, 17], [f"{tf[index]:g}" for index in [0, 5, 10, 15, 17]])
            axes[row_number, column].set(xlabel="SF (cpd)", ylabel="TF (Hz)" if column == 3 else "", title=title)
        figure.colorbar(image, ax=[axes[row_number, 3], axes[row_number, 4]], label="dB (shared row scale)", fraction=0.03)

        scale = max(float(np.max(np.abs(difference))), EPS)
        image = axes[row_number, 5].imshow(
            difference, origin="lower", aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale
        )
        axes[row_number, 5].set_xticks(range(len(sf)), [f"{value:.2g}" for value in sf], rotation=45)
        axes[row_number, 5].set_yticks([0, 5, 10, 15, 17], [f"{tf[index]:g}" for index in [0, 5, 10, 15, 17]])
        axes[row_number, 5].set(xlabel="SF (cpd)", title="orientation correction\noriented − radial")
        figure.colorbar(image, ax=axes[row_number, 5], label="signed accepted power", fraction=0.046)

        values = [selection.radial_direct_f0_drive, selection.oriented_direct_f0_drive]
        axes[row_number, 6].bar(["radial", "oriented"], values, color=["0.58", "#D55E00"])
        axes[row_number, 6].set_title(
            f"direct-F0 routing\nratio={selection.orientation_to_radial_ratio:.2f}; "
            f"RF r95={unit_rows.loc[unit, 'rf_radius95_pixel']:.1f}px"
        )
        axes[row_number, 6].set_yticks([])

    figure.suptitle(
        "Recorded-grating orientation checkpoint: exact retinal movies routed through unit-specific RF-local SF×orientation×TF filters\n"
        "Left panels are observed inputs; right panels are derived power proxies; no neural response is fit or selected",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dset, local, _ = load_heldout_grating_dataset(args.dataset_config, args.session)
    window_metrics, payload, candidate_sf, candidate_tf, ppd = candidate_windows(
        dset, local, int(args.stride), int(args.max_windows), session=args.session
    )
    units, apertures, radial_weights, oriented_weights, tuning_sf, tuning_tf = load_rf_and_tuning(
        args.rf_dir, args.tuning_dir, args.session
    )
    audit = synthetic_orientation_audit(float(ppd), apertures)
    if not audit.orientation_channel_correct.all():
        raise ValueError("Synthetic grating orientation audit failed")
    metrics, radial_spectra, oriented_spectra = build_metrics(
        window_metrics,
        payload,
        units,
        apertures,
        radial_weights,
        oriented_weights,
        tuning_sf,
        tuning_tf,
        candidate_sf,
        candidate_tf,
        float(ppd),
    )
    selected = select_examples(metrics)
    metrics.to_csv(args.out_dir / "window_unit_oriented_power_metrics.csv", index=False)
    selected.to_csv(args.out_dir / "selected_window_units.csv", index=False)
    audit.to_csv(args.out_dir / "synthetic_orientation_audit.csv", index=False)
    np.savez_compressed(
        args.out_dir / "rf_local_oriented_power_arrays.npz",
        radial_power=radial_spectra,
        orientation_power=oriented_spectra,
        sf_cpd=tuning_sf,
        tf_hz=tuning_tf,
        fourier_orientation_deg=0.5 * (ORIENTATION_EDGES_DEG[:-1] + ORIENTATION_EDGES_DEG[1:]),
        array_row=metrics.array_row.to_numpy(int),
        window_index=metrics.window_index.to_numpy(int),
        rr100_index=metrics.rr100_index.to_numpy(int),
    )
    figure_base = args.out_dir / "recorded_grating_rf_local_oriented_power_checkpoint"
    plot_checkpoint(
        selected,
        payload,
        units,
        apertures,
        radial_weights,
        oriented_weights,
        radial_spectra,
        oriented_spectra,
        tuning_sf,
        tuning_tf,
        figure_base,
        int(args.dpi),
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_recorded_grating_rf_local_oriented_power_input_checkpoint",
        "status": "input_mechanism_checkpoint_complete",
        "supersedes": "rr100_recorded_grating_oriented_power_input_checkpoint_v1 (display-angle audit did not support the dataset's offset eight-angle grid)",
        "session": args.session,
        "scope": {
            "n_windows": int(window_metrics.window_index.nunique()),
            "n_units": int(units.rr100_index.nunique()),
            "n_window_unit_pairs": int(len(metrics)),
            "window_frames": N_SCORE,
        },
        "contracts": {
            "spatial_localization": "unit-specific 51x51 RF aperture frozen by the existing RF-local grating checkpoint",
            "spectral_tensor": "positive-TF P(SF magnitude, Fourier-wavevector orientation modulo 180, TF)",
            "orientation_bins": int(len(ORIENTATION_EDGES_DEG) - 1),
            "grating_to_fourier_conversion": "theta_fourier_wavevector=(90-theta_grating_bar) mod 180",
            "neural_weight": "direct positive-F0 SFxorientationxTF weight; not squared",
            "radial_nesting": "orientation weights have mean one relative to the direct-F0 radial weight at every SFxTF cell",
            "response_use": "none; this checkpoint does not read, fit, select, or score neural responses",
        },
        "verification": {
            "synthetic_grating_cases": int(len(audit)),
            "synthetic_orientation_channel_accuracy": float(audit.orientation_channel_correct.mean()),
            "minimum_expected_channel_power_fraction": float(audit.expected_channel_power_fraction.min()),
            "maximum_radial_reproduction_relative_error": float(audit.radial_reproduction_relative_error.max()),
            "maximum_absolute_log2_oriented_radial_drive": float(metrics.log2_orientation_to_radial.abs().max()),
        },
        "inputs": {
            "dataset_config": file_identity(args.dataset_config),
            "rf_metadata": file_identity(args.rf_dir / "unit_rf_metadata.csv"),
            "rf_apertures": file_identity(args.rf_dir / "unit_rf_apertures.npz"),
            "orientation_tuning": file_identity(args.tuning_dir / "orientation_aware_f0_tuning_and_routing.npz"),
        },
        "artifacts": {
            "figure_png": figure_base.with_suffix(".png").name,
            "figure_pdf": figure_base.with_suffix(".pdf").name,
            "metrics": "window_unit_oriented_power_metrics.csv",
            "selected_examples": "selected_window_units.csv",
            "synthetic_audit": "synthetic_orientation_audit.csv",
            "arrays": "rf_local_oriented_power_arrays.npz",
        },
        "next_checkpoint": "compare trial-held-out radial and oriented power predictions with the full twin and recorded response",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
