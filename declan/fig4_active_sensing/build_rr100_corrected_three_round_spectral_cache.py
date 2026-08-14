#!/usr/bin/env python3
"""Build exact-condition retinal SFxTF spectra for corrected rounds 0--2.

This is input-only.  It reconstructs the same deterministic lag-zero retinal
frames used by the corrected production renderer for the 3,000 currently
assembled image--trace conditions.  Neural-model history is deliberately not
included in the retinal power spectrum.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    FRAME_RATE_HZ,
    ORIENTATION_EDGES_DEG,
    SCALAR_NAMES,
    SF_EDGES_CPD,
    materialize_trace_arrays,
    spectral_statistics,
)
from declan.fig4_active_sensing.input_only_retinal_renderer import render_retinal_frames_lag_zero
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
    _standardize_uint_like,
)


ASSEMBLED = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/assembled/rounds_000_002_n003"
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_three_round_spectral_cache_v2"
PPD = 37.50476617


def identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_patch(row: pd.Series) -> np.ndarray:
    path = Path(str(row.corrected_patch_npz))
    with np.load(path, allow_pickle=False) as data:
        patch = np.asarray(data[str(row.corrected_patch_key)], dtype=np.float32)
    return _standardize_uint_like(patch)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "condition_spectra.npz"
    condition_path = ASSEMBLED / "condition_index.csv"
    image_path = COHORT / "corrected100_images.csv"
    trace_path = COHORT / "corrected1000_traces.csv"
    conditions = pd.read_csv(condition_path).sort_values("matrix_row_index").reset_index(drop=True)
    images = pd.read_csv(image_path).sort_values("image_index").reset_index(drop=True)
    traces = pd.read_csv(trace_path).sort_values("trace_index").reset_index(drop=True)
    if len(conditions) != 3000 or len(images) != 100 or len(traces) != 1000:
        raise ValueError("Expected the frozen 3,000-condition / 100-image / 1,000-trace cohort")
    if not np.array_equal(conditions.matrix_row_index.to_numpy(int), np.arange(len(conditions))):
        raise ValueError("Condition rows are not aligned to response arrays")

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as existing:
            if "orientation_power" in existing.files:
                print(f"orientation-aware cache already exists: {cache_path}")
                return
        print(f"upgrading radial-only cache with exact orientation power: {cache_path}")

    _history, score, _ppd_by_session = materialize_trace_arrays(traces)
    trace_position = {int(trace): position for position, trace in enumerate(traces.trace_index.astype(int))}
    image_lookup = images.set_index("image_index")
    common = _load_twin_common()
    # Conditions are iterated image-major for rendering efficiency, while the
    # response cache is matrix-row-major.  Write by the declared matrix row so
    # the spectral and response axes cannot silently diverge.
    radial_rows = np.full((len(conditions), 20, len(SF_EDGES_CPD) - 1), np.nan, dtype=np.float32)
    orientation_rows = np.full(
        (len(conditions), 20, len(SF_EDGES_CPD) - 1, len(ORIENTATION_EDGES_DEG) - 1),
        np.nan,
        dtype=np.float32,
    )
    scalar_rows = np.full((len(conditions), len(SCALAR_NAMES)), np.nan, dtype=np.float64)
    example_dir = OUT / "example_movies"
    example_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for image_index, frame in conditions.groupby("image_index", sort=True):
            image_row = image_lookup.loc[int(image_index)]
            patch = load_patch(image_row)
            for row in frame.sort_values("matrix_row_index").itertuples(index=False):
                trace_pos = trace_position[int(row.trace_index)]
                retinal_trace = -score[trace_pos]
                tensor = render_retinal_frames_lag_zero(common, patch, retinal_trace, ppd=PPD, device="cpu")
                movie = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                radial, oriented, scalar = spectral_statistics(movie, ppd=PPD)
                matrix_row = int(row.matrix_row_index)
                radial_rows[matrix_row] = radial
                orientation_rows[matrix_row] = oriented
                scalar_rows[matrix_row] = scalar
                if int(row.matrix_row_index) in (0, 1000, 2000):
                    atomic_npz(
                        example_dir / f"condition_{int(row.matrix_row_index):04d}.npz",
                        matrix_row_index=np.asarray([int(row.matrix_row_index)]),
                        image_index=np.asarray([int(row.image_index)]),
                        trace_index=np.asarray([int(row.trace_index)]),
                        source_patch=patch,
                        retinal_trace=retinal_trace,
                        scored_movie=movie,
                        radial_power=radial,
                        orientation_power=oriented,
                    )
                del tensor, movie
            print(f"spectralized image {int(image_index) + 1}/100", flush=True)

    radial = radial_rows
    oriented = orientation_rows
    scalar = scalar_rows
    if radial.shape != (3000, 20, 13) or oriented.shape != (3000, 20, 13, 12) or scalar.shape != (3000, len(SCALAR_NAMES)):
        raise ValueError(f"Unexpected spectral arrays {radial.shape}, {oriented.shape}, {scalar.shape}")
    if not np.isfinite(radial).all() or not np.isfinite(oriented).all() or not np.isfinite(scalar).all():
        raise ValueError("At least one declared condition row was not populated")
    atomic_npz(
        cache_path,
        matrix_row_index=conditions.matrix_row_index.to_numpy(int),
        round_index=conditions.round_index.to_numpy(int),
        image_index=conditions.image_index.to_numpy(int),
        trace_index=conditions.trace_index.to_numpy(int),
        radial_power=radial,
        orientation_power=oriented,
        scalar_metrics=scalar,
        scalar_names=np.asarray(SCALAR_NAMES),
        sf_edges_cpd=SF_EDGES_CPD,
        tf_hz=np.fft.rfftfreq(40, d=1.0 / FRAME_RATE_HZ)[1:],
        orientation_edges_deg=ORIENTATION_EDGES_DEG,
    )
    table = conditions.copy()
    for column, name in enumerate(SCALAR_NAMES):
        table[name] = scalar[:, column]
    total = np.maximum(table.total_positive_tf_power.to_numpy(float), np.finfo(float).tiny)
    table["fraction_le_32_all_sf"] = table.power_le_32_all_sf / total
    table["fraction_32_45p25_all_sf"] = table.power_32_45p25_all_sf / total
    table["fraction_45p25_60_all_sf"] = table.power_45p25_60_all_sf / total
    table.to_csv(OUT / "condition_spectral_metrics.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "corrected_three_round_input_spectral_cache_complete",
        "scope": {"conditions": 3000, "rounds": 3, "images": 100, "traces": 1000},
        "contract": {
            "retinal_motion": "negative corrected dpi_pix crop trajectory",
            "frames": "40 scored lag-zero retinal frames at 120 Hz",
            "history": "32 recorded history frames validated for the neural response but excluded from input power",
            "spatial_window": "separable 2D Hann",
            "temporal_window": "Hann",
            "neural_model_calls": False,
            "orientation": "12 Fourier-wavevector bins retained; grating-bar orientation maps as theta_k=(90-theta_grating) mod 180",
        },
        "sources": {"conditions": identity(condition_path), "images": identity(image_path), "traces": identity(trace_path)},
        "outputs": {"arrays": str(cache_path.resolve()), "metrics": str((OUT / "condition_spectral_metrics.csv").resolve())},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
