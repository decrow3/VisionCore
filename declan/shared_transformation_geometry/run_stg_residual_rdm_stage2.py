from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from eval.fixrsvp import get_fixrsvp_data

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.shared_transformation_geometry.utils import DEFAULT_OUT_ROOT, harmonize_fixrsvp_arrays  # type: ignore
    from declan.twin_covariance_structure.run_a3_fixrsvp_audit import _predict_twin_rates  # type: ignore
else:
    from .utils import DEFAULT_OUT_ROOT, harmonize_fixrsvp_arrays
    from declan.twin_covariance_structure.run_a3_fixrsvp_audit import _predict_twin_rates  # type: ignore


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def _pairwise_features(eyepos: np.ndarray) -> dict[str, np.ndarray]:
    n = int(eyepos.shape[0])
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = eyepos[i_idx, 0] - eyepos[j_idx, 0]
    dy = eyepos[i_idx, 1] - eyepos[j_idx, 1]
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    radial = np.sqrt(dx * dx + dy * dy)
    return {
        "i_idx": i_idx,
        "j_idx": j_idx,
        "dx": dx,
        "dy": dy,
        "abs_dx": abs_dx,
        "abs_dy": abs_dy,
        "radial": radial,
        "dx2": dx * dx,
        "dy2": dy * dy,
        "abs_dxdy": np.abs(dx * dy),
    }


def _distance_vector(x: np.ndarray, metric: str) -> np.ndarray:
    i_idx, j_idx = np.triu_indices(x.shape[0], k=1)
    a = x[i_idx]
    b = x[j_idx]
    if metric == "euclidean_zscored":
        mu = np.mean(x, axis=0, keepdims=True)
        sd = np.std(x, axis=0, keepdims=True) + 1e-8
        xz = (x - mu) / sd
        a = xz[i_idx]
        b = xz[j_idx]
        return np.linalg.norm(a - b, axis=1)
    if metric == "correlation_distance":
        a0 = a - np.mean(a, axis=1, keepdims=True)
        b0 = b - np.mean(b, axis=1, keepdims=True)
        den = (np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1)) + 1e-12
        c = np.sum(a0 * b0, axis=1) / den
        return 1.0 - c
    raise ValueError(f"Unsupported metric: {metric}")


def _regress_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - (x @ beta)


def _smoothness_rows(
    *,
    session_id: str,
    subject: str,
    date: str,
    source: str,
    image_set: str,
    image_id: int,
    n_samples: int,
    metric: str,
    d: np.ndarray,
    feats: dict[str, np.ndarray],
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "subject": subject,
        "date": date,
        "source": source,
        "image_set": image_set,
        "analysis_representation": "raw_samples",
        "analysis_role": "diagnostic_only",
        "image_id": int(image_id),
        "n_samples": int(n_samples),
        "distance_metric": metric,
        "corr_vecD_radial_distance": _corr(d, feats["radial"]),
        "corr_vecD_abs_dx": _corr(d, feats["abs_dx"]),
        "corr_vecD_abs_dy": _corr(d, feats["abs_dy"]),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 STG residual RDM geometry (diagnostic-only; use Stage 2+3 runner for inference)")
    p.add_argument("--subject", type=str, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--dataset-configs-path", type=Path, default=Path("experiments") / "dataset_configs" / "multi_basic_240_rsvp.yaml")
    p.add_argument("--source", choices=("recorded", "twin"), required=True)
    p.add_argument("--image-set", type=str, default="high_support")
    p.add_argument("--n-samples", type=int, default=320)
    p.add_argument("--min-images", type=int, default=8)
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--predict-batch-size", type=int, default=64)
    p.add_argument("--model-device", type=str, default="cuda")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--use-cached-data", action="store_true", default=True)
    return p


def _build_source_rates(args: argparse.Namespace, data: dict[str, Any]) -> np.ndarray:
    if args.source == "recorded":
        return np.asarray(data["robs"], dtype=np.float64)
    rates, _ = _predict_twin_rates(
        data=data,
        subject=args.subject,
        date=args.date,
        dataset_configs_path=str(args.dataset_configs_path),
        model_type=None,
        model_index=None,
        checkpoint_path=None,
        dataset_idx=None,
        model_device=str(args.model_device),
        predict_batch_size=int(args.predict_batch_size),
    )
    return np.asarray(rates, dtype=np.float64)


def main() -> None:
    args = build_parser().parse_args()
    session_id = f"{args.subject}_{args.date}"
    print("[diagnostic] Standalone Stage 2 is diagnostic-only; use run_stg_residual_rdm_stage23.py for inferential outputs.")

    data = get_fixrsvp_data(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=str(args.dataset_configs_path),
        use_cached_data=bool(args.use_cached_data),
    )
    data = harmonize_fixrsvp_arrays(data)

    rates = _build_source_rates(args, data)
    eyepos = np.asarray(data["eyepos"], dtype=np.float64)
    image_ids = np.asarray(data["image_ids"], dtype=np.int64)

    valid = np.isfinite(rates).all(axis=2) & np.isfinite(eyepos).all(axis=2) & (image_ids >= 0)

    img_support = []
    for img in sorted(int(i) for i in np.unique(image_ids[valid])):
        n = int(np.sum(valid & (image_ids == img)))
        img_support.append((img, n))
    selected_images = [img for img, n in img_support if n >= int(args.n_samples)]
    if len(selected_images) < int(args.min_images):
        raise ValueError(
            f"Underpowered: only {len(selected_images)} images with >= {args.n_samples} samples; need >= {args.min_images}"
        )

    distances_by_metric: dict[str, dict[int, np.ndarray]] = {
        "euclidean_zscored": {},
        "correlation_distance": {},
    }
    residuals_radial: dict[str, dict[int, np.ndarray]] = {
        "euclidean_zscored": {},
        "correlation_distance": {},
    }
    residuals_expanded: dict[str, dict[int, np.ndarray]] = {
        "euclidean_zscored": {},
        "correlation_distance": {},
    }
    smoothness_rows: list[dict[str, object]] = []

    for img in selected_images:
        mask = valid & (image_ids == img)
        r = rates[mask]
        e = eyepos[mask]

        rng = np.random.default_rng(int(args.selection_seed) + int(img) * 1009)
        idx = rng.permutation(r.shape[0])[: int(args.n_samples)]
        r = r[idx]
        e = e[idx]

        feats = _pairwise_features(e)
        x_radial = np.stack([np.ones_like(feats["radial"]), feats["radial"]], axis=1)
        x_expanded = np.stack(
            [
                np.ones_like(feats["radial"]),
                feats["radial"],
                feats["abs_dx"],
                feats["abs_dy"],
                feats["dx2"],
                feats["dy2"],
                feats["abs_dxdy"],
            ],
            axis=1,
        )

        for metric in ("euclidean_zscored", "correlation_distance"):
            d = _distance_vector(r, metric)
            distances_by_metric[metric][img] = d
            residuals_radial[metric][img] = _regress_residual(d, x_radial)
            residuals_expanded[metric][img] = _regress_residual(d, x_expanded)

            smoothness_rows.append(
                _smoothness_rows(
                    session_id=session_id,
                    subject=args.subject,
                    date=args.date,
                    source=args.source,
                    image_set=args.image_set,
                    image_id=img,
                    n_samples=int(args.n_samples),
                    metric=metric,
                    d=d,
                    feats=feats,
                )
            )

    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for metric in ("euclidean_zscored", "correlation_distance"):
        corr_radial = []
        corr_expanded = []
        for i, j in itertools.combinations(selected_images, 2):
            c_radial = _corr(residuals_radial[metric][i], residuals_radial[metric][j])
            c_expanded = _corr(residuals_expanded[metric][i], residuals_expanded[metric][j])
            corr_radial.append(c_radial)
            corr_expanded.append(c_expanded)
            pair_rows.append(
                {
                    "session_id": session_id,
                    "subject": args.subject,
                    "date": args.date,
                    "source": args.source,
                    "image_set": args.image_set,
                    "analysis_representation": "raw_samples",
                    "analysis_role": "diagnostic_only",
                    "n_samples": int(args.n_samples),
                    "distance_metric": metric,
                    "image_i": int(i),
                    "image_j": int(j),
                    "residual_model": "radial_only",
                    "residual_similarity": c_radial,
                }
            )
            pair_rows.append(
                {
                    "session_id": session_id,
                    "subject": args.subject,
                    "date": args.date,
                    "source": args.source,
                    "image_set": args.image_set,
                    "analysis_representation": "raw_samples",
                    "analysis_role": "diagnostic_only",
                    "n_samples": int(args.n_samples),
                    "distance_metric": metric,
                    "image_i": int(i),
                    "image_j": int(j),
                    "residual_model": "expanded_eye_geometry",
                    "residual_similarity": c_expanded,
                }
            )

        for model_name, vals in (("radial_only", corr_radial), ("expanded_eye_geometry", corr_expanded)):
            arr = np.asarray(vals, dtype=np.float64)
            summary_rows.append(
                {
                    "session_id": session_id,
                    "subject": args.subject,
                    "date": args.date,
                    "source": args.source,
                    "image_set": args.image_set,
                    "analysis_representation": "raw_samples",
                    "analysis_role": "diagnostic_only",
                    "n_samples": int(args.n_samples),
                    "distance_metric": metric,
                    "residual_model": model_name,
                    "n_images": int(len(selected_images)),
                    "n_pairs": int(arr.size),
                    "mean_residual_similarity": float(np.mean(arr)) if arr.size else float("nan"),
                    "median_residual_similarity": float(np.median(arr)) if arr.size else float("nan"),
                    "interpretation_label": "diagnostic_only",
                }
            )

    out_dir = Path(args.out_dir) / session_id / f"source_{args.source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "stg2_diagnostic_image_smoothness_metrics.csv", smoothness_rows)
    _write_csv(out_dir / "stg2_diagnostic_cross_image_residual_rdm.csv", pair_rows)
    _write_csv(out_dir / "stg2_diagnostic_session_rdm_summary.csv", summary_rows)

    print(str(out_dir / "stg2_diagnostic_image_smoothness_metrics.csv"))
    print(str(out_dir / "stg2_diagnostic_cross_image_residual_rdm.csv"))
    print(str(out_dir / "stg2_diagnostic_session_rdm_summary.csv"))


if __name__ == "__main__":
    main()
