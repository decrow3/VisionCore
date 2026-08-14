#!/usr/bin/env python3
"""Input-only checkpoint for orientation-aware SFxTF routing."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.spectral_cache_contract import validated_spectral_cache_from_environment


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_routing_input_checkpoint_v1"
GRATING_ORIENTATIONS = np.asarray([0.0, 45.0, 90.0, 135.0])


def circular_distance_180(a: np.ndarray, b: float) -> np.ndarray:
    difference = np.abs(np.asarray(a) - float(b)) % 180.0
    return np.minimum(difference, 180.0 - difference)


def four_grating_channels(oriented: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Collapse 12 wave-vector bins into four nearest grating-bar channels.

    Bins exactly between two channels are split equally, so the four output
    channels form a power-preserving partition rather than double-counting
    boundary orientations.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    output = np.zeros(oriented.shape[:-1] + (4,), dtype=np.float64)
    wavevector_targets = (90.0 - GRATING_ORIENTATIONS) % 180.0
    distances = np.stack([circular_distance_180(centers, target) for target in wavevector_targets], axis=1)
    minimum = distances.min(axis=1, keepdims=True)
    weights = np.isclose(distances, minimum).astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    output[...] = np.einsum("...o,oc->...c", oriented, weights)
    return output


def relative_db(values: np.ndarray) -> np.ndarray:
    maximum = max(float(np.nanmax(values)), np.finfo(float).tiny)
    return 10.0 * np.log10(np.maximum(values / maximum, 1e-5))


def main() -> None:
    spectral = validated_spectral_cache_from_environment()
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(spectral / "condition_spectra.npz", allow_pickle=False) as data:
        oriented = np.asarray(data["orientation_power"], dtype=np.float64)
        radial = np.asarray(data["radial_power"], dtype=np.float64)
        sf_edges = np.asarray(data["sf_edges_cpd"], dtype=float)
        tf = np.asarray(data["tf_hz"], dtype=float)
        ori_edges = np.asarray(data["orientation_edges_deg"], dtype=float)
        image = np.asarray(data["image_index"], dtype=int)
        trace = np.asarray(data["trace_index"], dtype=int)
        rounds = np.asarray(data["round_index"], dtype=int)
    conditions = pd.read_csv(
        ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/assembled/rounds_000_002_n003/condition_index.csv"
    )
    images = pd.read_csv(
        ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1/corrected100_images.csv"
    ).set_index("image_index")
    traces = pd.read_csv(
        ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1/corrected1000_traces.csv"
    ).set_index("trace_index")
    channels = four_grating_channels(oriented, ori_edges)
    fitted_sf = (0.5 * (sf_edges[:-1] + sf_edges[1:]) >= 1.0) & (0.5 * (sf_edges[:-1] + sf_edges[1:]) <= 11.3137085)
    fitted_tf = (tf > 0) & (tf <= 56)
    channel_power = channels[:, fitted_tf][:, :, fitted_sf].sum(axis=(1, 2))
    total = channel_power.sum(axis=1)
    concentration = channel_power.max(axis=1) / np.maximum(total, np.finfo(float).tiny)
    central = (total >= np.quantile(total, 0.25)) & (total <= np.quantile(total, 0.75))
    selected_row = int(np.flatnonzero(central)[np.argmax(concentration[central])])
    selected = pd.DataFrame(
        [{
            "matrix_row_index": selected_row,
            "round_index": int(rounds[selected_row]),
            "image_index": int(image[selected_row]),
            "trace_index": int(trace[selected_row]),
            "selection_role": "orientation-structured typical-amplitude input",
            "selection_criterion": "maximum four-channel orientation concentration among middle 50% of supported total power",
            "orientation_concentration": float(concentration[selected_row]),
            "supported_total_power": float(total[selected_row]),
        }]
    )
    selected.to_csv(OUT / "selected_input_condition.csv", index=False)

    image_row = images.loc[int(image[selected_row])]
    trace_row = traces.loc[int(trace[selected_row])]
    patch_path = Path(str(image_row.corrected_patch_npz))
    with np.load(patch_path, allow_pickle=False) as data:
        patch = np.asarray(data[str(image_row.corrected_patch_key)], dtype=float)
    example_path = spectral / "example_movies" / f"condition_{selected_row:04d}.npz"
    movie = None
    retinal_trace = None
    if example_path.exists():
        with np.load(example_path, allow_pickle=False) as data:
            movie = np.asarray(data["scored_movie"], dtype=float)
            retinal_trace = np.asarray(data["retinal_trace"], dtype=float)
    if retinal_trace is None:
        trace_cache = (
            ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
            "input_cache/corrected_trace_segments.npz"
        )
        with np.load(trace_cache, allow_pickle=False) as data:
            cached_ids = np.asarray(data["trace_index"], dtype=int)
            position = np.flatnonzero(cached_ids == int(trace[selected_row]))
            if len(position) != 1:
                raise ValueError("Selected trace is not unique in the frozen corrected trace cache")
            # The renderer applies the negative crop trajectory.
            retinal_trace = -np.asarray(data["score_xy_deg"], dtype=float)[int(position[0])]

    sf = 0.5 * (sf_edges[:-1] + sf_edges[1:])
    maps = channels[selected_row]
    common_max = max(float(maps[:, :, index].max()) for index in range(4))
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[0.9, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(patch, cmap="gray", origin="upper")
    ax.set(title=f"corrected source image {int(image[selected_row])}")
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    if retinal_trace is not None:
        ax.plot(retinal_trace[:, 0] * 60, retinal_trace[:, 1] * 60, color="0.2", lw=1.4)
        ax.scatter(retinal_trace[0, 0] * 60, retinal_trace[0, 1] * 60, color="#009E73", s=32, label="start")
        ax.set(xlabel="retinal shift x (arcmin)", ylabel="retinal shift y (arcmin)", title=f"corrected trace {int(trace[selected_row])}")
        ax.legend(frameon=False)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "trace/movie not retained\nfor this selected condition", ha="center", va="center")
    ax = fig.add_subplot(gs[0, 2:])
    ax.axis("off")
    ax.text(0.02, 0.82, r"Radial analysis: $P(SF,TF)=\sum_{\theta_k}P(SF,\theta_k,TF)$", fontsize=15, weight="bold")
    ax.text(0.02, 0.53, r"Orientation-aware analysis retains $P(SF,\theta_k,TF)$", fontsize=15, weight="bold")
    ax.text(0.02, 0.18, "Neural orientation labels grating bars; FFT orientation labels the wave vector.\n"
            r"Matching rule: $\theta_k=(90^\circ-\theta_{grating})\ \mathrm{mod}\ 180^\circ$.", fontsize=12)

    for index, grating_orientation in enumerate(GRATING_ORIENTATIONS):
        ax = fig.add_subplot(gs[1, index])
        wavevector_orientation = (90.0 - grating_orientation) % 180.0
        values = maps[:, :, index]
        shown = 10 * np.log10(np.maximum(values / max(common_max, np.finfo(float).tiny), 1e-5))
        im = ax.pcolormesh(sf, tf, shown, shading="nearest", cmap="magma", vmin=-50, vmax=0)
        ax.set_xscale("log")
        ax.set(xlabel="SF (cpd)", ylabel="TF (Hz)" if index == 0 else "",
               title=f"{grating_orientation:.0f}° grating-axis channel\nFourier normal {wavevector_orientation:.0f}°")
    fig.colorbar(im, ax=[fig.axes[-4], fig.axes[-3], fig.axes[-2], fig.axes[-1]], label="power relative to common maximum (dB)")
    ax = fig.add_subplot(gs[1, 4])
    fraction = channel_power[selected_row] / max(float(channel_power[selected_row].sum()), np.finfo(float).tiny)
    ax.bar(["0°", "45°", "90°", "135°"], 100 * fraction, color=["#0072B2", "#E69F00", "#009E73", "#D55E00"])
    ax.set(xlabel="grating-axis channel", ylabel="supported dynamic power (%)", title="Orientation composition")
    fig.suptitle("Orientation checkpoint — FEM-created power occupies distinct SF×TF regions at different image orientations", fontsize=15, weight="bold")
    fig.savefig(OUT / "orientation_aware_input_checkpoint.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "orientation_aware_input_checkpoint.pdf", bbox_inches="tight")
    plt.close(fig)

    availability = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "orientation_input_checkpoint_complete",
        "movie_tensor": {"conditions": 3000, "tf_bins": 20, "sf_bins": 13, "fourier_orientation_bins": 12},
        "neural_tensor": {
            "units": 100,
            "spatial_frequencies": 8,
            "grating_orientations": [0, 45, 90, 135],
            "fit_temporal_frequencies_hz": "0.5–32 established (13 points) plus 34–56 extension (12 points)",
            "controls": "repeated 32-Hz audit and 60-Hz Nyquist control",
            "response_measure": "direction-folded positive F0 (mean rate above blank)",
        },
        "orientation_conversion": "theta_fourier_wavevector=(90-theta_grating_bar) mod 180",
        "neural_model_calls": False,
        "selected_condition": selected.iloc[0].to_dict(),
        "unsupported_yet": "whether orientation-aware overlap predicts activation or SSI",
    }
    (OUT / "manifest.json").write_text(json.dumps(availability, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(availability, indent=2))


if __name__ == "__main__":
    main()
