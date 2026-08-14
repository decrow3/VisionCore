#!/usr/bin/env python3
"""Figure 01: show the physical retinal-power redistribution before neural tuning."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.fig4_active_sensing.input_only_retinal_renderer import render_retinal_frames_lag_zero
from declan.fig4_active_sensing.spectral_cache_contract import validated_spectral_cache_from_environment
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
    _standardize_uint_like,
)


COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
TRACE_CACHE = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/input_cache/corrected_trace_segments.npz"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_power_routing_figure_series_v1/01_input_redistribution"
PPD = 37.50476617


def choose_examples(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    frame["supported_high_tf_power"] = (
        frame.power_32_45p25_fitted_sf
        + frame.power_45p25_60_fitted_sf
        - frame.power_at_60_fitted_sf
    )
    frame["supported_high_tf_fraction"] = frame.supported_high_tf_power / np.maximum(
        frame.total_positive_tf_power_fitted_sf, np.finfo(float).tiny
    )
    used: set[int] = set()
    rows: list[pd.Series] = []

    def add(role: str, index: int, criterion: str, metric: str) -> None:
        row = frame.loc[index].copy()
        if int(row.matrix_row_index) in used:
            candidates = frame[~frame.matrix_row_index.isin(used)]
            index2 = (candidates[metric] - row[metric]).abs().idxmin()
            row = frame.loc[index2].copy()
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        row["selection_metric"] = metric
        row["selection_value"] = float(row[metric])
        rows.append(row)
        used.add(int(row.matrix_row_index))

    add("low dynamic power", frame.total_positive_tf_power_fitted_sf.idxmin(), "minimum fitted-SF dynamic power", "total_positive_tf_power_fitted_sf")
    median = float(frame.total_positive_tf_power_fitted_sf.median())
    add("typical dynamic power", (frame.total_positive_tf_power_fitted_sf - median).abs().idxmin(), "closest to median fitted-SF dynamic power", "total_positive_tf_power_fitted_sf")
    add("high dynamic power", frame.total_positive_tf_power_fitted_sf.idxmax(), "maximum fitted-SF dynamic power", "total_positive_tf_power_fitted_sf")
    add("largest high-TF fraction", frame.supported_high_tf_fraction.idxmax(), "maximum fraction of fitted-SF power between 32 and 56 Hz", "supported_high_tf_fraction")
    return pd.DataFrame(rows)


def load_patch(row: pd.Series) -> np.ndarray:
    with np.load(str(row.corrected_patch_npz), allow_pickle=False) as data:
        return _standardize_uint_like(np.asarray(data[str(row.corrected_patch_key)], dtype=np.float32))


def main() -> None:
    spectral = validated_spectral_cache_from_environment()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(spectral / "condition_spectral_metrics.csv")
    images = pd.read_csv(COHORT / "corrected100_images.csv").set_index("image_index")
    with np.load(spectral / "condition_spectra.npz", allow_pickle=False) as data:
        radial = np.asarray(data["radial_power"], dtype=float)
        sf_edges = np.asarray(data["sf_edges_cpd"], dtype=float)
        tf = np.asarray(data["tf_hz"], dtype=float)
    with np.load(TRACE_CACHE, allow_pickle=False) as data:
        trace_ids = np.asarray(data["trace_index"], dtype=int)
        score_xy = np.asarray(data["score_xy_deg"], dtype=np.float32)
    trace_lookup = {int(identity): position for position, identity in enumerate(trace_ids)}
    selected = choose_examples(metrics)
    selected.to_csv(OUT / "selected_input_conditions.csv", index=False)

    common = _load_twin_common()
    payloads = []
    for row in selected.itertuples(index=False):
        image_row = images.loc[int(row.image_index)]
        patch = load_patch(image_row)
        retinal_trace = -score_xy[trace_lookup[int(row.trace_index)]]
        with torch.no_grad():
            tensor = render_retinal_frames_lag_zero(common, patch, retinal_trace, ppd=PPD, device="cpu")
        movie = tensor.detach().cpu().numpy().astype(float)
        payloads.append((row, patch, retinal_trace, movie, radial[int(row.matrix_row_index)]))

    global_reference = max(float(power.max()) for *_rest, power in payloads)
    sf_centers = 0.5 * (sf_edges[:-1] + sf_edges[1:])
    fig, axes = plt.subplots(len(payloads), 5, figsize=(18, 3.35 * len(payloads)), constrained_layout=True)
    band_colors = ["#0072B2", "#E69F00", "#D55E00"]
    for row_index, (row, patch, trace, movie, power) in enumerate(payloads):
        ax = axes[row_index, 0]
        # Photographic display convention only; Fourier calculations retain
        # the array coordinates used by the renderer.
        ax.imshow(patch, cmap="gray", origin="upper")
        ax.set_title(f"{row.selection_role}\nsource image {int(row.image_index)}", loc="left", weight="bold", fontsize=9.5)
        ax.axis("off")

        ax = axes[row_index, 1]
        ax.plot(trace[:, 0] * 60, trace[:, 1] * 60, color="#222", lw=1.2)
        ax.scatter(trace[0, 0] * 60, trace[0, 1] * 60, color="#009E73", s=22, label="start")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="retinal shift x (arcmin)", ylabel="retinal shift y (arcmin)", title=f"eye trace {int(row.trace_index)}")
        ax.legend(frameon=False, fontsize=7)

        ax = axes[row_index, 2]
        strip = np.concatenate([movie[0], movie[len(movie) // 2], movie[-1]], axis=1)
        ax.imshow(strip, cmap="gray", origin="upper")
        ax.set_title("retinal frames: start · middle · end", fontsize=9.5)
        ax.axis("off")

        ax = axes[row_index, 3]
        db = 10 * np.log10(np.maximum(power / max(global_reference, np.finfo(float).tiny), 1e-6))
        mesh = ax.pcolormesh(sf_centers, tf, db, shading="nearest", cmap="magma", vmin=-60, vmax=0)
        ax.axhline(32, color="cyan", ls="--", lw=0.9)
        ax.axhline(56, color="white", ls=":", lw=0.9)
        ax.axvspan(1, 11.3137085, color="white", alpha=0.06)
        ax.set_xscale("log")
        ax.set(xlabel="spatial frequency (cpd)", ylabel="temporal frequency (Hz)", title="FEM-created dynamic power")
        if row_index == 0:
            fig.colorbar(mesh, ax=axes[:, 3], shrink=0.72, label="power (dB relative to selected-set maximum)")

        ax = axes[row_index, 4]
        total = max(float(row.total_positive_tf_power_fitted_sf), np.finfo(float).tiny)
        bands = np.asarray([
            float(row.power_le_32_fitted_sf),
            float(row.power_32_45p25_fitted_sf),
            float(row.power_45p25_60_fitted_sf - row.power_at_60_fitted_sf),
        ]) / total
        ax.bar(["3–30", "33–45", "48–57"], 100 * bands, color=band_colors)
        ax.set_ylim(0, 100)
        ax.set_ylabel("fraction of fitted-SF dynamic power (%)")
        ax.set_title(f"total amplitude = {np.sqrt(total):.2e} a.u.\nstabilized dynamic power = 0", fontsize=9.5)
    fig.suptitle(
        "Figure 01 — Eye motion turns static spatial structure into condition-specific temporal power\n"
        "Observed retinal inputs only; no neural tuning or response is used to select these examples",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(OUT / "figure01_retinal_power_redistribution.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / "figure01_retinal_power_redistribution.pdf", bbox_inches="tight")
    plt.close(fig)

    selected_rows = selected.matrix_row_index.to_numpy(int)
    np.savez_compressed(
        OUT / "figure01_source_values.npz",
        matrix_row_index=selected_rows,
        radial_power=radial[selected_rows].astype(np.float32),
        sf_edges_cpd=sf_edges,
        tf_hz=tf,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "figure01_input_redistribution_complete",
        "selection": "input-only roles defined before neural responses are inspected",
        "visible_claim": "corrected FEM trajectories create different amounts and distributions of nonzero-TF retinal power",
        "not_tested": "unit tuning, neural response, SSI, gain, or routing",
        "artifacts": {
            "figure": str((OUT / "figure01_retinal_power_redistribution.pdf").resolve()),
            "selection": str((OUT / "selected_input_conditions.csv").resolve()),
            "values": str((OUT / "figure01_source_values.npz").resolve()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
