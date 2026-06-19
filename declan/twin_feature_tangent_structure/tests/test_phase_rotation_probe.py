from __future__ import annotations

import csv
import pickle
import tempfile
from pathlib import Path

import numpy as np

from declan.twin_feature_tangent_structure.run_phase_rotation_probe import (
    _ellipse_metrics,
    _fit_generator,
    _generator_shape_metrics,
    _valid_object_ids,
    analyze,
    build_parser,
)


def test_ellipse_metrics_identifies_circle_and_line() -> None:
    circle = _ellipse_metrics(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]), radius_px=1.0)
    assert np.isclose(circle["ellipse_aspect_minor_over_major"], 1.0)
    assert np.isclose(circle["ellipse_circularity"], 1.0)

    line = _ellipse_metrics(np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0]), radius_px=1.0)
    assert line["ellipse_aspect_minor_over_major"] < 1e-8
    assert line["ellipse_circularity"] < 1e-8


def test_fit_generator_recovers_phase_rotation_block() -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    z = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    generator = np.asarray([[0.0, -2.0], [2.0, 0.0]])
    dz = z @ generator.T
    fit = _fit_generator(z, dz, ridge=1e-10)
    shape = _generator_shape_metrics(fit.generator)
    assert fit.r2 > 0.999999
    assert np.allclose(fit.generator, generator, atol=1e-8)
    assert shape["generator_skew_energy_fraction"] > 0.999999
    assert shape["generator_symmetric_energy_fraction"] < 1e-8


def test_valid_object_filter_keeps_one_axis_zero_tangent() -> None:
    payload = {
        "flat-x": {
            "r0": np.asarray([1.0, 2.0]),
            "bx": np.asarray([1.0, 0.0]),
            "by": np.asarray([0.0, 0.0]),
        },
        "dead": {
            "r0": np.asarray([1.0, 2.0]),
            "bx": np.asarray([0.0, 0.0]),
            "by": np.asarray([0.0, 0.0]),
        },
    }
    assert _valid_object_ids(payload) == ["flat-x"]


def _write_phase_fixture(root: Path) -> Path:
    input_root = root / "tfts"
    pkl_dir = input_root / "tangent_maps"
    pkl_dir.mkdir(parents=True)
    payload = {}
    delta = 0.25
    kx = np.asarray([1.5, 0.4])
    ky = np.asarray([0.2, 1.0])
    for idx, phi in enumerate(np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)):
        phase = np.asarray([phi, 2.0 * phi + 0.3])
        amp = np.asarray([1.0, 0.6])
        z0 = np.asarray(
            [
                amp[0] * np.cos(phase[0]),
                amp[0] * np.sin(phase[0]),
                amp[1] * np.cos(phase[1]),
                amp[1] * np.sin(phase[1]),
            ],
            dtype=np.float64,
        )
        gx = np.asarray(
            [
                [0.0, -kx[0], 0.0, 0.0],
                [kx[0], 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -kx[1]],
                [0.0, 0.0, kx[1], 0.0],
            ],
            dtype=np.float64,
        )
        gy = np.asarray(
            [
                [0.0, -ky[0], 0.0, 0.0],
                [ky[0], 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -ky[1]],
                [0.0, 0.0, ky[1], 0.0],
            ],
            dtype=np.float64,
        )
        frame = np.sin(np.linspace(0.0, 2.0 * np.pi, 16))[None, :] * np.cos(np.linspace(0.0, 2.0 * np.pi, 16))[:, None]
        history = np.stack([frame for _ in range(3)], axis=0).astype(np.float32)
        payload[f"0/{idx}/{idx + 3}"] = {
            "r0": z0.astype(np.float32),
            "bx": (gx @ z0).astype(np.float32),
            "by": (gy @ z0).astype(np.float32),
            "image_id": idx % 5,
            "trial_index": idx,
            "time_index": idx + 3,
            "delta_arcmin": delta,
            "delta_model_px": 0.1,
            "history": history,
        }
    with (pkl_dir / "twin_tangent_maps.pkl").open("wb") as handle:
        pickle.dump({"delta_arcmins": [delta], "object_payload": {delta: payload}}, handle)
    return input_root


def test_analyze_phase_fixture_writes_expected_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        input_root = _write_phase_fixture(Path(tmp))
        out_dir = Path(tmp) / "out"
        args = build_parser().parse_args(
            [
                "--input-root",
                str(input_root),
                "--output-dir",
                str(out_dir),
                "--basis-dims",
                "4",
                "--pair-null-repeats",
                "4",
                "--min-objects",
                "8",
            ]
        )
        analyze(args)

        with (out_dir / "phase_rotation_generator_metrics.csv").open(newline="", encoding="utf-8") as handle:
            generator_rows = list(csv.DictReader(handle))
        assert {row["fit_axis"] for row in generator_rows} == {"x", "y"}
        assert min(float(row["generator_fit_r2"]) for row in generator_rows) > 0.99
        assert min(float(row["generator_skew_energy_fraction"]) for row in generator_rows) > 0.99

        with (out_dir / "phase_rotation_summary.csv").open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        assert len(summary_rows) == 1
        assert summary_rows[0]["status"] == "ok"
        assert float(summary_rows[0]["tangent_union_capture_by_basis"]) > 0.999
        assert (out_dir / "README.md").exists()
