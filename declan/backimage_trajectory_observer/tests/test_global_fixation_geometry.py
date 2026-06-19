from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from declan.backimage_trajectory_observer.analyze_global_fixation_geometry import analyze, build_parser


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_fixture(root: Path) -> Path:
    run_dir = root / "backimage_run"
    table_dir = run_dir / "response_tables"
    table_dir.mkdir(parents=True)
    rows = []
    window_rows = []
    n_trials = 3
    n_candidates = 4
    n_traj = 3
    n_time = 6
    n_units = 5
    for trial in range(n_trials):
        prior = np.zeros((n_candidates, n_traj, n_time, n_units), dtype=np.float32)
        zero = np.zeros((n_candidates, n_time, n_units), dtype=np.float32)
        known = np.zeros((n_candidates, n_time, n_units), dtype=np.float32)
        candidate_ids = []
        for c in range(n_candidates):
            source_row = 1000 + trial * n_candidates + c
            candidate_ids.append(f"source_row:{source_row}")
            content = np.asarray([float(c), float(trial), 0.2 * c, 0.1 * trial, 1.0], dtype=np.float32)
            for t in range(n_time):
                zero[c, t] = content + 0.01 * t
                known[c, t] = content + np.asarray([0.1 * t, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                for k in range(n_traj):
                    motion = np.asarray([0.1 * t, 0.2 * k, 0.02 * t * k, 0.0, 0.0], dtype=np.float32)
                    prior[c, k, t] = zero[c, t] + motion
            window_rows.append(
                {
                    "source_row": source_row,
                    "session": "fixture",
                    "image_index": trial,
                    "image_patch_center_x_px": 10.0 * c,
                    "image_patch_center_y_px": 20.0 * trial,
                    "image_patch_rms_contrast": 0.5 + 0.1 * c,
                    "image_gradient_energy": 1.0 + c,
                    "image_edge_density": 0.1,
                    "image_orientation_coherence": 0.2,
                    "image_dominant_orientation_deg": 30.0,
                }
            )
        path = table_dir / f"trial_{trial:05d}.npz"
        np.savez_compressed(
            path,
            prior_lambda_counts=prior,
            known_lambda_counts=known,
            zero_lambda_counts=zero,
            y_obs_counts=known[0],
            candidate_ids=np.asarray(candidate_ids),
            prior_trajectory_ids=np.asarray([f"tau:{i}" for i in range(n_traj)]),
            true_candidate_index=np.asarray([0], dtype=np.int32),
            true_trajectory_index=np.asarray([-1], dtype=np.int32),
            nearest_trajectory_index=np.asarray([0], dtype=np.int32),
            zero_reference_mode=np.asarray(["patch_center_static_tau_zero"]),
        )
        rows.append(
            {
                "trial_id": trial,
                "candidate_set_mode": "hard_negative_structure",
                "observation_family": "empirical",
                "prior_family": "empirical",
                "scale": 0.5,
                "response_cache_path": str(path.relative_to(run_dir)),
                "n_candidates": n_candidates,
                "n_prior_trajectories": n_traj,
                "n_timebins": n_time,
                "n_units": n_units,
                "dry_run": False,
            }
        )
    _write_csv(run_dir / "response_cache_manifest.csv", rows)
    _write_csv(run_dir / "selected_windows.csv", window_rows)
    return run_dir


def test_analyze_global_fixation_geometry_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _write_fixture(Path(tmp))
        out_dir = Path(tmp) / "out"
        args = build_parser().parse_args(
            [
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(out_dir),
                "--candidate-set-modes",
                "hard_negative_structure",
                "--prior-families",
                "empirical",
                "--scales",
                "0.5",
                "--response-variants",
                "motion_delta,zero_static",
                "--point-mode",
                "state_timepoints",
                "--max-points",
                "10000",
                "--min-points",
                "10",
                "--min-local-points",
                "3",
                "--n-pcs",
                "5",
                "--max-pca-rows",
                "100",
            ]
        )
        analyze(args)

        with (out_dir / "global_fixation_geometry_summary.csv").open(newline="", encoding="utf-8") as handle:
            summary = list(csv.DictReader(handle))
        assert {row["response_variant"] for row in summary} == {"motion_delta", "zero_static"}
        assert {row["status"] for row in summary} == {"ok"}
        assert min(float(row["n_unique_candidates"]) for row in summary) == 12
        assert max(float(row["local_plane_fraction_median"]) for row in summary) > 0.9
        assert (out_dir / "global_fixation_geometry_points_pca.csv").exists()
        assert (out_dir / "README.md").exists()
