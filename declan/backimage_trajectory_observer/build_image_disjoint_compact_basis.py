"""Build image-disjoint compact tangent bases from cached TFTS tangent maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _delta_key(delta: float) -> str:
    return str(float(delta)).replace(".", "p").replace("-", "m")


def _stable_fold(label: str, n_folds: int, seed: int) -> int:
    payload = f"{int(seed)}::{label}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % int(n_folds)


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
            except (KeyError, TypeError, ValueError):
                continue
            dropped.setdefault(delta, set()).add(object_id)
    return dropped


def _object_group(object_id: str, rec: dict[str, Any], split_by: str) -> str:
    if split_by == "image_id":
        return str(rec.get("image_id", object_id))
    if split_by == "object_id":
        return str(object_id)
    raise ValueError("split_by must be 'image_id' or 'object_id'")


def _stack_tangents(
    payload: dict[str, dict[str, Any]],
    *,
    dropped_ids: set[str],
    heldout_fold: int,
    n_folds: int,
    seed: int,
    split_by: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []
    for object_id in sorted(str(k) for k in payload.keys()):
        if object_id in dropped_ids:
            continue
        rec = payload[object_id]
        group = _object_group(object_id, rec, split_by)
        fold = _stable_fold(group, n_folds, seed)
        if fold == int(heldout_fold):
            continue
        bx = np.asarray(rec.get("bx"), dtype=np.float64)
        by = np.asarray(rec.get("by"), dtype=np.float64)
        if bx.ndim != 1 or by.ndim != 1 or bx.shape != by.shape:
            continue
        rows.extend([bx, by])
        kept.append(
            {
                "object_id": object_id,
                "image_id": str(rec.get("image_id", "")),
                "split_group": group,
                "fold": int(fold),
            }
        )
    if not rows:
        raise ValueError("No valid tangent vectors remain after image-disjoint split")
    return np.stack(rows, axis=0), kept


def _svd_basis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _u, singular_values, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    return vt.T, singular_values


def _variance_fraction(singular_values: np.ndarray, k: int) -> float:
    s2 = np.asarray(singular_values, dtype=np.float64) ** 2
    total = float(np.sum(s2))
    kk = min(int(k), s2.size)
    return float(np.sum(s2[:kk]) / total) if total > 0.0 else float("nan")


def build(args: argparse.Namespace) -> Path:
    tfts_root = Path(args.tfts_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (tfts_root / "tangent_maps" / "twin_tangent_maps.pkl").open("rb") as handle:
        cached = pickle.load(handle)
    dropped_by_delta = _read_dropped(tfts_root)
    delta = float(args.delta_arcmin)
    if delta not in cached["object_payload"]:
        available = ", ".join(str(v) for v in cached["object_payload"].keys())
        raise ValueError(f"delta_arcmin={delta} unavailable; available: {available}")
    payload = cached["object_payload"][delta]
    n_folds = int(args.n_folds)
    heldout_fold = int(args.heldout_fold)
    if heldout_fold < 0 or heldout_fold >= n_folds:
        raise ValueError("heldout_fold must be in [0, n_folds)")
    matrix, kept = _stack_tangents(
        payload,
        dropped_ids=dropped_by_delta.get(delta, set()),
        heldout_fold=heldout_fold,
        n_folds=n_folds,
        seed=int(args.seed),
        split_by=str(args.split_by),
    )
    if str(args.centering) == "centered_across_tangents_per_unit":
        basis_matrix = matrix - matrix.mean(axis=0, keepdims=True)
    elif str(args.centering) == "uncentered":
        basis_matrix = matrix
    else:
        raise ValueError("centering must be 'centered_across_tangents_per_unit' or 'uncentered'")
    basis, singular_values = _svd_basis(basis_matrix)
    key = _delta_key(delta)
    basis_path = out_dir / f"image_disjoint_compact_basis_delta{key}_fold{heldout_fold}of{n_folds}.npz"
    np.savez(
        basis_path,
        basis=basis,
        singular_values=singular_values,
        image_disjoint=np.asarray([True]),
        basis_mode=np.asarray(["image_disjoint"]),
        basis_provenance=np.asarray(
            [
                f"image_disjoint {args.split_by} heldout_fold={heldout_fold} n_folds={n_folds} "
                f"seed={int(args.seed)} centering={args.centering}"
            ]
        ),
        source_tangent_maps=np.asarray([str(tfts_root / "tangent_maps" / "twin_tangent_maps.pkl")]),
        delta_arcmin=np.asarray([delta], dtype=np.float64),
        heldout_fold=np.asarray([heldout_fold], dtype=np.int32),
        n_folds=np.asarray([n_folds], dtype=np.int32),
    )
    pd.DataFrame(kept).to_csv(out_dir / f"image_disjoint_compact_basis_delta{key}_fold{heldout_fold}of{n_folds}_objects.csv", index=False)
    manifest = {
        "basis_path": str(basis_path),
        "source_tfts_root": str(tfts_root),
        "source_tangent_maps": str(tfts_root / "tangent_maps" / "twin_tangent_maps.pkl"),
        "delta_arcmin": delta,
        "basis_mode": "image_disjoint",
        "split_by": str(args.split_by),
        "n_folds": n_folds,
        "heldout_fold": heldout_fold,
        "seed": int(args.seed),
        "centering": str(args.centering),
        "matrix_shape": [int(v) for v in matrix.shape],
        "basis_shape": [int(v) for v in basis.shape],
        "n_train_objects": int(len(kept)),
        "n_train_split_groups": int(pd.Series([row["split_group"] for row in kept]).nunique()) if kept else 0,
        "top_variance_fraction": {
            f"k{int(k)}": _variance_fraction(singular_values, int(k))
            for k in [2, 5, 10, 20, 50, 100, min(126, singular_values.size)]
        },
    }
    (out_dir / f"image_disjoint_compact_basis_delta{key}_fold{heldout_fold}of{n_folds}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not bool(args.quiet):
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return basis_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfts-root", type=Path, default=Path("outputs/twin_feature_tangent_structure_prod_limited_synth"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--delta-arcmin", type=float, default=0.25)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--heldout-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-by", choices=["image_id", "object_id"], default="image_id")
    parser.add_argument("--centering", choices=["centered_across_tangents_per_unit", "uncentered"], default="centered_across_tangents_per_unit")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
