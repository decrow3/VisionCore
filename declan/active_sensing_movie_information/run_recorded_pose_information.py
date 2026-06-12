#!/usr/bin/env python3
"""Inventory and bridge for recorded-cortex pose-aware information analyses.

The prescription asks for a recorded V1 pose-aware information test.  A related
Poisson prediction ladder already exists in
``run_recorded_pose_aware_prediction.py``; this script records the available
banked caches/outputs and provides the intended output location for the decoder
implementation without duplicating that mature runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

from non_circular_fem_common import DEFAULT_STACK_OUT_DIR, read_csv_rows, write_csv_rows, write_json


DEFAULT_PHASE1_DIR = Path("outputs/phase1_fem_covariance_sensitivity_bemp")
DEFAULT_EXISTING_POSE_RUN = Path("outputs/active_sensing_movie_information")
DEFAULT_OUT_DIR = DEFAULT_STACK_OUT_DIR / "recorded_pose_information"
SESSION_QC_FILES = (
    "qc/session_qc.csv",
    "qc/unit_qc.csv",
    "qc/image_repeat_support.csv",
    "qc/residual_metadata.csv",
    "aggregation_scaling/stage4_comparison_metrics.csv",
    "aggregation_scaling/aggregation_scaling_session_metrics.csv",
)
SESSION_RE = re.compile(r"^[A-Za-z]+_\d{4}-\d{2}-\d{2}$")


def inventory_phase1_outputs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for session_dir in sorted(path for path in root.iterdir() if path.is_dir() and SESSION_RE.match(path.name)):
        session = session_dir.name
        for rel in SESSION_QC_FILES:
            path = session_dir / rel
            rows.append(
                {
                    "source": "phase1_fem_covariance",
                    "session": session,
                    "artifact": rel,
                    "path": str(path),
                    "exists": path.exists(),
                    "n_rows": len(read_csv_rows(path)) if path.exists() else 0,
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                }
            )
    return rows


def pose_prediction_roots(root: Path) -> list[Path]:
    """Return existing recorded-pose prediction output directories."""
    if not root.exists():
        return []
    if (root / "recorded_pose_aware_prediction_manifest.json").exists():
        return [root]
    return sorted(
        path
        for path in root.glob("recorded_pose_aware_prediction*")
        if path.is_dir() and (path / "recorded_pose_aware_prediction_manifest.json").exists()
    )


def inventory_existing_pose_prediction(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in pose_prediction_roots(root):
        manifest_path = run_dir / "recorded_pose_aware_prediction_manifest.json"
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".csv", ".json", ".md", ".pdf", ".png"}:
                continue
            rows.append(
                {
                    "source": "recorded_pose_aware_prediction",
                    "run": run_dir.name,
                    "session": "",
                    "artifact": path.name,
                    "path": str(path),
                    "exists": True,
                    "is_manifest": path == manifest_path,
                    "n_rows": len(read_csv_rows(path)) if path.suffix.lower() == ".csv" else "",
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    n_phase1 = sum(1 for row in rows if row["source"] == "phase1_fem_covariance" and row["exists"])
    n_pose = sum(1 for row in rows if row["source"] == "recorded_pose_aware_prediction" and row["exists"])
    sessions = sorted({str(row["session"]) for row in rows if row.get("session")})
    pose_runs = sorted({str(row.get("run", "")) for row in rows if row["source"] == "recorded_pose_aware_prediction" and row.get("run")})
    lines = [
        "# Recorded Pose Information Inventory",
        "",
        f"- Phase1 covariance artifacts present: {n_phase1}",
        f"- Existing pose-aware prediction artifacts present: {n_pose}",
        f"- Pose-aware prediction runs found: {', '.join(pose_runs) if pose_runs else 'none found'}",
        f"- Sessions with phase1 artifacts: {', '.join(sessions) if sessions else 'none found'}",
        "",
        "Recommended next implementation step: adapt or symlink the strongest completed pose-aware prediction run into the prescription's recorded_pose_info_* schema, rather than rerunning the decoder.",
        "",
    ]
    (out_dir / "recorded_pose_info_inventory_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--existing-pose-dir", type=Path, default=DEFAULT_EXISTING_POSE_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = inventory_phase1_outputs(Path(args.phase1_dir)) + inventory_existing_pose_prediction(Path(args.existing_pose_dir))
    write_csv_rows(out_dir / "recorded_pose_info_cache_inventory.csv", rows)
    write_summary(out_dir, rows)
    write_json(
        out_dir / "recorded_pose_info_inventory_manifest.json",
        {
            "phase1_dir": Path(args.phase1_dir),
            "existing_pose_dir": Path(args.existing_pose_dir),
            "out_dir": out_dir,
            "n_inventory_rows": len(rows),
        },
    )
    print(f"Wrote recorded pose-information inventory to {out_dir}")


if __name__ == "__main__":
    main()
