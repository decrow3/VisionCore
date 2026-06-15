"""Rebuild canonical manifest tangent bases under centered/uncentered conventions.

The historical compact manifest basis was exported from
``twin_tangent_maps.pkl`` after stacking ``bx/by`` tangent vectors across
objects into an ``n_tangents x n_units`` matrix and subtracting the per-unit
mean tangent vector across objects.  This helper rebuilds that convention and
also exports the raw uncentered convention from the same cached tangent maps.

This is a diagnostic/export utility only; it does not run covariance closure.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np

VISIONCORE_ROOT = Path(__file__).resolve().parents[2]
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))


def _delta_key(delta: float) -> str:
    return str(float(delta)).replace(".", "p").replace("-", "m")


def _read_dropped(root: Path) -> dict[float, set[str]]:
    path = root / "dropped_objects_union_basis.csv"
    dropped: dict[float, set[str]] = {}
    if not path.exists():
        return dropped
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                delta = float(row["delta"])
                object_id = str(row["object_id"])
            except Exception:
                continue
            dropped.setdefault(delta, set()).add(object_id)
    return dropped


def _stack_tangents(payload: dict[str, dict[str, Any]], dropped_ids: set[str]) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    object_ids: list[str] = []
    for object_id in sorted(str(k) for k in payload.keys()):
        if object_id in dropped_ids:
            continue
        rec = payload[object_id]
        if "bx" not in rec or "by" not in rec:
            continue
        bx = np.asarray(rec["bx"], dtype=np.float64)
        by = np.asarray(rec["by"], dtype=np.float64)
        if bx.ndim != 1 or by.ndim != 1 or bx.shape != by.shape:
            continue
        rows.extend([bx, by])
        object_ids.append(object_id)
    if not rows:
        raise ValueError("No valid bx/by tangent vectors found.")
    return np.stack(rows, axis=0), object_ids


def _svd_basis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _u, singular_values, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    return vt.T, singular_values


def _variance_fractions(singular_values: np.ndarray, k_values: tuple[int, ...]) -> dict[str, float]:
    s2 = np.asarray(singular_values, dtype=np.float64) ** 2
    total = float(np.sum(s2))
    out: dict[str, float] = {}
    for k in k_values:
        kk = min(int(k), s2.size)
        out[f"top{int(k)}_variance_fraction"] = float(np.sum(s2[:kk]) / total) if total > 0 else float("nan")
    return out


def _subspace_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    kk = min(int(k), a.shape[1], b.shape[1])
    if kk <= 0:
        return float("nan")
    sv = np.linalg.svd(a[:, :kk].T @ b[:, :kk], compute_uv=False)
    return float(np.mean(sv * sv))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tfts-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--reference-basis-npz", type=Path)
    p.add_argument("--reference-basis-key-prefix", default="basis_delta_")
    p.add_argument("--k-list", default="2,10,20,50,100,126")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tfts_root = Path(args.tfts_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = tuple(int(float(x)) for x in str(args.k_list).split(",") if x.strip())
    with (tfts_root / "tangent_maps" / "twin_tangent_maps.pkl").open("rb") as handle:
        cached = pickle.load(handle)
    dropped_by_delta = _read_dropped(tfts_root)
    ref = np.load(args.reference_basis_npz) if args.reference_basis_npz else None

    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {
        "source_tfts_root": str(tfts_root),
        "source_tangent_maps": str(tfts_root / "tangent_maps" / "twin_tangent_maps.pkl"),
        "reference_basis_npz": str(args.reference_basis_npz) if args.reference_basis_npz else None,
        "conventions": {
            "uncentered": "stack bx/by rows directly, SVD, use V.T as response-space basis",
            "centered_across_tangents_per_unit": "stack bx/by rows, subtract per-unit mean tangent vector across rows, SVD, use V.T",
        },
        "deltas": {},
    }
    summary_rows: list[dict[str, Any]] = []
    for delta in [float(v) for v in cached["delta_arcmins"]]:
        payload = cached["object_payload"][delta]
        matrix, object_ids = _stack_tangents(payload, dropped_by_delta.get(delta, set()))
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        uncentered_basis, uncentered_s = _svd_basis(matrix)
        centered_basis, centered_s = _svd_basis(centered)
        key = _delta_key(delta)
        arrays[f"basis_delta_{key}_uncentered"] = uncentered_basis
        arrays[f"singular_values_delta_{key}_uncentered"] = uncentered_s
        arrays[f"basis_delta_{key}_centered_across_tangents_per_unit"] = centered_basis
        arrays[f"singular_values_delta_{key}_centered_across_tangents_per_unit"] = centered_s
        if abs(delta - 0.25) < 1e-9:
            arrays["basis_uncentered"] = uncentered_basis
            arrays["singular_values_uncentered"] = uncentered_s
            arrays["basis_centered_across_tangents_per_unit"] = centered_basis
            arrays["singular_values_centered_across_tangents_per_unit"] = centered_s

        delta_meta: dict[str, Any] = {
            "n_objects": int(len(object_ids)),
            "matrix_shape": [int(v) for v in matrix.shape],
            "uncentered_basis_shape": [int(v) for v in uncentered_basis.shape],
            "centered_basis_shape": [int(v) for v in centered_basis.shape],
            "uncentered": _variance_fractions(uncentered_s, k_values),
            "centered_across_tangents_per_unit": _variance_fractions(centered_s, k_values),
            "uncentered_vs_centered_overlap": {
                f"k{int(k)}": _subspace_overlap(uncentered_basis, centered_basis, int(k)) for k in k_values
            },
        }
        if ref is not None:
            ref_key = f"{args.reference_basis_key_prefix}{key}"
            if ref_key in ref.files:
                ref_basis = np.asarray(ref[ref_key], dtype=np.float64)
                delta_meta["reference_overlap"] = {
                    f"k{int(k)}": _subspace_overlap(centered_basis, ref_basis, int(k)) for k in k_values
                }
        manifest["deltas"][str(delta)] = delta_meta
        for convention, basis, svals in (
            ("uncentered", uncentered_basis, uncentered_s),
            ("centered_across_tangents_per_unit", centered_basis, centered_s),
        ):
            row: dict[str, Any] = {
                "delta_arcmin": float(delta),
                "convention": convention,
                "n_objects": int(len(object_ids)),
                "n_tangent_rows": int(matrix.shape[0]),
                "n_units": int(matrix.shape[1]),
                "rank_numeric_gt_1e-10": int(np.sum(svals > 1e-10)),
            }
            row.update(_variance_fractions(svals, k_values))
            summary_rows.append(row)

    np.savez(out_dir / "manifest_tangent_basis_conventions.npz", **arrays)
    import pandas as pd

    pd.DataFrame(summary_rows).to_csv(out_dir / "manifest_tangent_basis_conventions_summary.csv", index=False)
    (out_dir / "manifest_tangent_basis_conventions_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
