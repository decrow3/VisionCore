#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from eval.fixrsvp import get_fixrsvp_data


DEFAULT_OUTPUT_DIR = Path("outputs/jacobian_predictive_framework")


@dataclass(frozen=True)
class AnalysisUnit:
    unit_id: str
    image_id: int
    trial_indices: tuple[int, ...]
    time_indices: tuple[int, ...]
    n_samples: int
    median_radius_deg: float
    p75_radius_deg: float
    p90_radius_deg: float


class Step01Backend:
    """
    Backend seam for model-response and Jacobian evaluation.

    The current scaffold deliberately stops after data collation, analysis-unit
    definition, and empirical displacement gating. The next edit should attach a
    backend here that takes an AnalysisUnit and returns model responses under
    shifted inputs plus a local Jacobian estimate on the same unit definition.
    """

    def describe(self) -> str:
        return "manifest-only"

    def run(self, units: Iterable[AnalysisUnit]) -> dict:
        return {
            "status": "manifest_only",
            "backend": self.describe(),
            "n_units": sum(1 for _ in units),
        }


def _valid_xy(eyepos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(eyepos).all(axis=-1)
    x = eyepos[..., 0]
    y = eyepos[..., 1]
    return x, y, finite_mask


def compute_centered_radius_deg(eyepos: np.ndarray) -> np.ndarray:
    x, y, finite_mask = _valid_xy(eyepos)
    centered = np.full_like(eyepos, np.nan, dtype=np.float64)
    for trial_idx in range(eyepos.shape[0]):
        valid = finite_mask[trial_idx]
        if not np.any(valid):
            continue
        xy = eyepos[trial_idx, valid]
        xy_centered = xy - np.nanmedian(xy, axis=0, keepdims=True)
        centered[trial_idx, valid] = xy_centered
    return np.linalg.norm(centered, axis=-1)


def compute_step_displacement_deg(eyepos: np.ndarray) -> np.ndarray:
    steps = np.full(eyepos.shape[:2], np.nan, dtype=np.float64)
    for trial_idx in range(eyepos.shape[0]):
        trial = eyepos[trial_idx]
        finite = np.isfinite(trial).all(axis=-1)
        valid_idx = np.flatnonzero(finite)
        if valid_idx.size < 2:
            continue
        diffs = np.diff(trial[valid_idx], axis=0)
        steps[trial_idx, valid_idx[1:]] = np.linalg.norm(diffs, axis=-1)
    return steps


def summarize_percentiles(values: np.ndarray, percentiles: tuple[int, ...] = (50, 75, 90, 95)) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"p{pct}": float("nan") for pct in percentiles}
    return {f"p{pct}": float(np.percentile(finite, pct)) for pct in percentiles}


def build_image_units(
    image_ids: np.ndarray,
    radius_deg: np.ndarray,
    min_samples: int,
) -> list[AnalysisUnit]:
    units: list[AnalysisUnit] = []
    finite_radius = np.isfinite(radius_deg)
    finite_image = image_ids >= 0
    usable = finite_radius & finite_image
    image_values = np.unique(image_ids[usable])

    for image_id in image_values:
        trial_idx, time_idx = np.where(usable & (image_ids == image_id))
        if trial_idx.size < min_samples:
            continue
        radii = radius_deg[trial_idx, time_idx]
        units.append(
            AnalysisUnit(
                unit_id=f"image_{int(image_id):05d}",
                image_id=int(image_id),
                trial_indices=tuple(int(x) for x in trial_idx.tolist()),
                time_indices=tuple(int(x) for x in time_idx.tolist()),
                n_samples=int(trial_idx.size),
                median_radius_deg=float(np.nanmedian(radii)),
                p75_radius_deg=float(np.nanpercentile(radii, 75)),
                p90_radius_deg=float(np.nanpercentile(radii, 90)),
            )
        )

    units.sort(key=lambda unit: (-unit.n_samples, unit.image_id))
    return units


def write_units_csv(path: Path, units: list[AnalysisUnit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unit_id",
                "image_id",
                "n_samples",
                "median_radius_deg",
                "p75_radius_deg",
                "p90_radius_deg",
            ],
        )
        writer.writeheader()
        for unit in units:
            writer.writerow(
                {
                    "unit_id": unit.unit_id,
                    "image_id": unit.image_id,
                    "n_samples": unit.n_samples,
                    "median_radius_deg": unit.median_radius_deg,
                    "p75_radius_deg": unit.p75_radius_deg,
                    "p90_radius_deg": unit.p90_radius_deg,
                }
            )


def write_summary_md(
    path: Path,
    *,
    subject: str,
    date: str,
    dataset_configs_path: str,
    radius_summary: dict,
    step_summary: dict,
    n_trials: int,
    n_units: int,
    min_samples: int,
    backend_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# fixRSVP Step 0/1 scaffold summary

## Scope

- subject: {subject}
- date: {date}
- dataset config: {dataset_configs_path}
- backend: {backend_name}

## Analysis unit

Primary unit for this scaffold is pooled image identity over all valid fixRSVP bins with finite eye position.
Units are retained only if they have at least {min_samples} valid samples across trials.

## Data summary

- trials after fixRSVP preprocessing: {n_trials}
- retained image units: {n_units}

## Empirical displacement summaries

Centered eye-position radius in degrees, pooled over valid bins:

- median: {radius_summary['p50']:.6f}
- p75: {radius_summary['p75']:.6f}
- p90: {radius_summary['p90']:.6f}
- p95: {radius_summary['p95']:.6f}

Frame-to-frame eye-step magnitude in degrees, pooled over valid bins:

- median: {step_summary['p50']:.6f}
- p75: {step_summary['p75']:.6f}
- p90: {step_summary['p90']:.6f}
- p95: {step_summary['p95']:.6f}

## Status

This run establishes the fixRSVP analysis-unit manifest and the Step 0 displacement gates.
The remaining implementation step is to attach a model-response backend that computes local Jacobians and finite-difference response changes on these same units.
"""
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold Step 0 and Step 1 fixRSVP Jacobian analyses."
    )
    parser.add_argument("--subject", required=True, help="Session subject name.")
    parser.add_argument("--date", required=True, help="Session date in YYYY-MM-DD format.")
    parser.add_argument(
        "--dataset-configs-path",
        required=True,
        help="Dataset config YAML used for fixRSVP collation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for manifests and summaries.",
    )
    parser.add_argument(
        "--min-samples-per-unit",
        type=int,
        default=100,
        help="Minimum valid bins required for a pooled image unit.",
    )
    parser.add_argument(
        "--use-cached-data",
        action="store_true",
        help="Use cached fixRSVP collation when available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print fixRSVP preprocessing diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    data = get_fixrsvp_data(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=args.dataset_configs_path,
        use_cached_data=args.use_cached_data,
        verbose=args.verbose,
    )

    radius_deg = compute_centered_radius_deg(data["eyepos"])
    step_deg = compute_step_displacement_deg(data["eyepos"])
    radius_summary = summarize_percentiles(radius_deg)
    step_summary = summarize_percentiles(step_deg)
    units = build_image_units(
        image_ids=data["image_ids"],
        radius_deg=radius_deg,
        min_samples=args.min_samples_per_unit,
    )

    backend = Step01Backend()
    backend_result = backend.run(units)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "step01_manifest.npz",
        centered_radius_deg=radius_deg,
        step_displacement_deg=step_deg,
        image_ids=data["image_ids"],
        fix_dur=data["fix_dur"],
        radius_p50=radius_summary["p50"],
        radius_p75=radius_summary["p75"],
        radius_p90=radius_summary["p90"],
        radius_p95=radius_summary["p95"],
        step_p50=step_summary["p50"],
        step_p75=step_summary["p75"],
        step_p90=step_summary["p90"],
        step_p95=step_summary["p95"],
        retained_unit_ids=np.array([unit.unit_id for unit in units], dtype=object),
        retained_image_ids=np.array([unit.image_id for unit in units], dtype=np.int32),
        retained_unit_samples=np.array([unit.n_samples for unit in units], dtype=np.int32),
    )
    write_units_csv(output_dir / "step01_units.csv", units)
    write_summary_md(
        output_dir / "README.md",
        subject=args.subject,
        date=args.date,
        dataset_configs_path=args.dataset_configs_path,
        radius_summary=radius_summary,
        step_summary=step_summary,
        n_trials=int(data["robs"].shape[0]),
        n_units=len(units),
        min_samples=args.min_samples_per_unit,
        backend_name=backend.describe(),
    )
    (output_dir / "backend_status.json").write_text(json.dumps(backend_result, indent=2))

    print(f"Saved Step 0/1 scaffold outputs to {output_dir}")
    print(f"Retained {len(units)} pooled image units with >= {args.min_samples_per_unit} samples")


if __name__ == "__main__":
    main()