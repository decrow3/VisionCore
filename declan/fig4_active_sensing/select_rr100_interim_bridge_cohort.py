#!/usr/bin/env python3
"""Freeze an outcome-independent, multivariate 20-image x 50-trace bridge.

Selection uses only corrected input/provenance variables. It seeds both tails
of each declared feature and fills the remaining slots with deterministic
maximin coverage in percentile-rank space. No cached neural response quantity
is read or used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_interim49x973_bridge_cohort_checkpoint_28_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_interim_bridge_selection_checkpoint_32_v2"
SEED = 20260813

IMAGE_FEATURES = (
    "reconstruction_exact_pixel_r",
    "exact_saved_rms_contrast",
    "exact_saved_orientation_coherence",
    "exact_saved_sf_centroid_cpd",
    "exact_saved_high_sf_fraction",
)
TRACE_FEATURES = (
    "corrected_dpi_crop120_path_length_arcmin",
    "corrected_dpi_crop120_rms_radius_arcmin",
    "corrected_dpi_crop120_position_power_fraction_32plus_hz",
    "corrected_dpi_crop120_position_power_centroid_hz",
    "corrected_dpi_crop120_cov_anisotropy",
    "corrected_minus_legacy_path_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--n-traces", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
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
        "sha256": digest.hexdigest(),
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def percentile_features(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    missing = [feature for feature in features if feature not in frame]
    if missing:
        raise KeyError(f"Missing selection features: {missing}")
    values = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        bad = values.columns[values.isna().any()].to_list()
        raise ValueError(f"Selection features contain missing/non-numeric values: {bad}")
    return np.column_stack(
        [values[column].rank(method="average", pct=True).to_numpy(float) for column in features]
    )


def select_maximin(
    frame: pd.DataFrame,
    *,
    key: str,
    features: tuple[str, ...],
    n_select: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame[key].duplicated().any():
        raise ValueError(f"Candidate key {key!r} is not unique")
    if not 1 <= int(n_select) <= len(frame):
        raise ValueError(f"Invalid n_select={n_select} for {len(frame)} candidates")
    work = frame.sort_values(key).reset_index(drop=True).copy()
    ranked = percentile_features(work, features)
    selected: list[int] = []
    reason: dict[int, tuple[str, str, float]] = {}

    # Explicitly seed both tails of every scientific coverage coordinate.
    for column_index, feature in enumerate(features):
        for label, index in (
            ("low", int(np.argmin(ranked[:, column_index]))),
            ("high", int(np.argmax(ranked[:, column_index]))),
        ):
            if index not in selected and len(selected) < int(n_select):
                selected.append(index)
                reason[index] = (
                    "feature_extreme",
                    f"{label} tail of {feature}",
                    float(ranked[index, column_index]),
                )

    rng = np.random.default_rng(int(seed))
    tie_break = rng.uniform(0.0, 1e-12, size=len(work))
    while len(selected) < int(n_select):
        chosen = ranked[np.asarray(selected, dtype=int)]
        squared = ((ranked[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2)
        min_distance = np.sqrt(squared.min(axis=1))
        min_distance[np.asarray(selected, dtype=int)] = -np.inf
        # A tiny deterministic bonus improves session diversity without
        # allowing session identity to override the continuous input features.
        if "session" in work:
            used_sessions = set(work.iloc[selected]["session"].astype(str))
            new_session = ~work["session"].astype(str).isin(used_sessions).to_numpy()
            score = min_distance + 0.02 * new_session.astype(float) + tie_break
        else:
            score = min_distance + tie_break
        index = int(np.argmax(score))
        selected.append(index)
        reason[index] = (
            "multivariate_maximin",
            "largest minimum distance in declared percentile-rank feature space",
            float(min_distance[index]),
        )

    selected_frame = work.iloc[selected].copy().reset_index(drop=True)
    selected_frame["selection_order"] = np.arange(len(selected_frame), dtype=int)
    selected_frame["selection_role"] = [reason[index][0] for index in selected]
    selected_frame["selection_criterion"] = [reason[index][1] for index in selected]
    selected_frame["selection_criterion_value"] = [reason[index][2] for index in selected]
    selected_frame["selection_is_algorithmic"] = True
    selected_frame["selection_seed"] = int(seed)

    audit = work[[key, "session", *features]].copy()
    audit["selected"] = audit[key].isin(set(selected_frame[key]))
    for column_index, feature in enumerate(features):
        audit[f"rankpct__{feature}"] = ranked[:, column_index]
    return selected_frame, audit


def coverage_summary(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    features: tuple[str, ...],
    kind: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        full = pd.to_numeric(candidates[feature], errors="raise").to_numpy(float)
        chosen = pd.to_numeric(selected[feature], errors="raise").to_numpy(float)
        rows.append(
            {
                "kind": kind,
                "feature": feature,
                "n_candidates": int(len(full)),
                "n_selected": int(len(chosen)),
                "candidate_min": float(np.min(full)),
                "candidate_median": float(np.median(full)),
                "candidate_max": float(np.max(full)),
                "selected_min": float(np.min(chosen)),
                "selected_median": float(np.median(chosen)),
                "selected_max": float(np.max(chosen)),
                "selected_fraction_below_candidate_q10": float(np.mean(chosen <= np.quantile(full, 0.10))),
                "selected_fraction_above_candidate_q90": float(np.mean(chosen >= np.quantile(full, 0.90))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite bridge checkpoint: {args.out_dir}")
    image_path = args.cohort_dir / "interim49_images.csv"
    trace_path = args.cohort_dir / "interim973_traces.csv"
    images = pd.read_csv(image_path)
    traces = pd.read_csv(trace_path)
    if len(images) != 49 or images["image_index"].nunique() != 49:
        raise ValueError("Expected exactly 49 unique interim images")
    if len(traces) != 973 or traces["trace_index"].nunique() != 973:
        raise ValueError("Expected exactly 973 unique interim traces")
    if not images["corrected_crop_valid"].astype(bool).all():
        raise ValueError("Interim image candidates include invalid corrected crops")
    if not traces["explicit_history_valid"].astype(bool).all():
        raise ValueError("Interim trace candidates include invalid explicit histories")

    selected_images, image_audit = select_maximin(
        images,
        key="image_index",
        features=IMAGE_FEATURES,
        n_select=int(args.n_images),
        seed=int(args.seed),
    )
    selected_traces, trace_audit = select_maximin(
        traces,
        key="trace_index",
        features=TRACE_FEATURES,
        n_select=int(args.n_traces),
        seed=int(args.seed) + 1,
    )
    coverage = pd.concat(
        [
            coverage_summary(images, selected_images, IMAGE_FEATURES, "image"),
            coverage_summary(traces, selected_traces, TRACE_FEATURES, "trace"),
        ],
        ignore_index=True,
    )

    args.out_dir.mkdir(parents=True)
    image_out = args.out_dir / "bridge20_images.csv"
    trace_out = args.out_dir / "bridge50_traces.csv"
    atomic_csv(selected_images, image_out)
    atomic_csv(selected_traces, trace_out)
    atomic_csv(image_audit, args.out_dir / "image_candidate_selection_audit.csv")
    atomic_csv(trace_audit, args.out_dir / "trace_candidate_selection_audit.csv")
    atomic_csv(coverage, args.out_dir / "feature_coverage_summary.csv")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "small_bridge_cohort_frozen_not_launched",
        "selection_algorithm": "tail-seeded deterministic multivariate maximin in percentile-rank space",
        "selection_seed": int(args.seed),
        "selection_outcome_independence": "input/provenance variables only; no neural response cache is read",
        "counts": {
            "candidate_images": int(len(images)),
            "selected_images": int(len(selected_images)),
            "candidate_traces": int(len(traces)),
            "selected_traces": int(len(selected_traces)),
            "movies_per_condition": int(len(selected_images) * len(selected_traces)),
        },
        "image_features": list(IMAGE_FEATURES),
        "trace_features": list(TRACE_FEATURES),
        "conditions": [
            "legacy renderer",
            "corrected synthetic history",
            "corrected 32-frame recorded history",
            "zero-relative-translation explicit-history baseline",
        ],
        "sources": {
            "images": file_identity(image_path),
            "traces": file_identity(trace_path),
            "selector": file_identity(Path(__file__)),
        },
        "outputs": {
            "images": file_identity(image_out),
            "traces": file_identity(trace_out),
        },
        "neural_scoring": "not authorized by this manifest",
    }
    atomic_text(args.out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
