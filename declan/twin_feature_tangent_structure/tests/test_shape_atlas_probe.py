from __future__ import annotations

import csv
import pickle
import tempfile
from pathlib import Path

import numpy as np

from declan.twin_feature_tangent_structure.run_shape_atlas_probe import (
    _classify_shape,
    _pca_rank_metrics,
    _ring_fourier_metrics,
    _sheet_fit_metrics,
    analyze,
    build_parser,
)


def test_ring_fourier_metrics_identifies_ellipse() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    dz = np.stack([2.0 * np.cos(theta), 0.6 * np.sin(theta), np.zeros_like(theta)], axis=1)
    metrics = _ring_fourier_metrics(theta, dz)
    assert metrics["ring_planarity_fraction"] > 0.999
    assert metrics["ring_first_harmonic_fraction"] > 0.999
    assert metrics["ellipse_like_score"] > 0.999


def test_sheet_and_pca_metrics_identify_flat_plane_and_line() -> None:
    shifts = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float64,
    )
    plane = np.stack([shifts[:, 0], shifts[:, 1], np.zeros(shifts.shape[0])], axis=1)
    pca = _pca_rank_metrics(plane)
    sheet = _sheet_fit_metrics(shifts, plane)
    row = {**pca, **sheet, **_ring_fourier_metrics(np.zeros(0), np.zeros((0, 3)))}
    assert pca["plane_fraction"] > 0.999
    assert sheet["linear_sheet_r2_energy"] > 0.999
    assert _classify_shape(row) == "flat_translation_sheet"

    line = np.stack([shifts[:, 0] + 0.5 * shifts[:, 1], np.zeros(shifts.shape[0])], axis=1)
    assert _pca_rank_metrics(line)["line_fraction"] > 0.999


def _write_shape_fixture(root: Path) -> Path:
    input_root = root / "tfts"
    pkl_dir = input_root / "tangent_maps"
    pkl_dir.mkdir(parents=True)
    payload = {}
    delta = 0.25
    delta_px = 0.1
    for idx, phi in enumerate(np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)):
        r0 = np.asarray([np.cos(phi), np.sin(phi), 0.2 * np.cos(2.0 * phi), 0.2 * np.sin(2.0 * phi)], dtype=np.float64)
        bx = np.asarray([1.0, 0.0, 0.2, 0.0], dtype=np.float64)
        by = np.asarray([0.0, 1.0, 0.0, 0.2], dtype=np.float64)
        frame = np.sin(np.linspace(0.0, 2.0 * np.pi, 12))[None, :] * np.cos(np.linspace(0.0, 2.0 * np.pi, 12))[:, None]
        history = np.stack([frame for _ in range(4)], axis=0).astype(np.float32)
        payload[f"0/{idx}/{idx + 2}"] = {
            "r0": r0.astype(np.float32),
            "bx": bx.astype(np.float32),
            "by": by.astype(np.float32),
            "rx_p": (r0 + bx * delta_px).astype(np.float32),
            "rx_m": (r0 - bx * delta_px).astype(np.float32),
            "ry_p": (r0 + by * delta_px).astype(np.float32),
            "ry_m": (r0 - by * delta_px).astype(np.float32),
            "image_id": idx % 4,
            "trial_index": idx,
            "time_index": idx + 2,
            "delta_arcmin": delta,
            "delta_model_px": delta_px,
            "history": history,
        }
    with (pkl_dir / "twin_tangent_maps.pkl").open("wb") as handle:
        pickle.dump({"delta_arcmins": [delta], "object_payload": {delta: payload}}, handle)
    return input_root


def test_analyze_shape_fixture_writes_expected_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        input_root = _write_shape_fixture(Path(tmp))
        out_dir = Path(tmp) / "out"
        args = build_parser().parse_args(
            [
                "--input-root",
                str(input_root),
                "--output-dir",
                str(out_dir),
                "--basis-dims",
                "2,4",
                "--random-basis-repeats",
                "4",
                "--min-objects",
                "8",
            ]
        )
        analyze(args)

        with (out_dir / "shape_atlas_summary.csv").open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        assert len(summary_rows) == 2
        assert {row["status"] for row in summary_rows} == {"ok"}
        assert max(float(row["compact_orbit_energy_fraction_median"]) for row in summary_rows) > 0.99
        assert (out_dir / "README.md").exists()
