from __future__ import annotations

import argparse
import csv
import itertools
import json
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


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(keep)) < 2:
        return 0.0
    a = a[keep]
    b = b[keep]
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 0.0 or (not np.isfinite(den)):
        return 0.0
    return float(np.dot(aa, bb) / den)


def _pair_features(centers: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = int(centers.shape[0])
    i, j = np.triu_indices(n, k=1)
    dx = centers[i, 0] - centers[j, 0]
    dy = centers[i, 1] - centers[j, 1]
    radial = np.sqrt(dx * dx + dy * dy)
    feats = {
        "radial": radial,
        "abs_dx": np.abs(dx),
        "abs_dy": np.abs(dy),
        "dx2": dx * dx,
        "dy2": dy * dy,
        "abs_dxdy": np.abs(dx * dy),
    }
    return np.stack([i, j], axis=1), feats


def _distance_vector(responses: np.ndarray, metric: str) -> np.ndarray:
    x = np.asarray(responses, dtype=np.float64)
    if metric == "euclidean_zscored":
        x = (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + 1e-12)
        i, j = np.triu_indices(x.shape[0], k=1)
        return np.linalg.norm(x[i] - x[j], axis=1)
    if metric == "correlation_distance":
        x = x - np.mean(x, axis=1, keepdims=True)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        c = x @ x.T
        i, j = np.triu_indices(x.shape[0], k=1)
        return 1.0 - np.clip(c[i, j], -1.0, 1.0)
    raise ValueError(f"Unknown distance metric: {metric}")


def _residualize(y: np.ndarray, design_cols: list[np.ndarray]) -> np.ndarray:
    cols = [np.ones_like(y)] + [np.asarray(c, dtype=np.float64) for c in design_cols]
    x = np.stack(cols, axis=1)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - (x @ beta)


def _build_image_bin_map(
    rates: np.ndarray,
    eyepos: np.ndarray,
    image_ids: np.ndarray,
    valid: np.ndarray,
    *,
    img_id: int,
    min_samples: int,
    eye_bin_step: float,
    min_bin_count: int,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], dict[str, object]] | None:
    mask = valid & (image_ids == int(img_id))
    if int(np.sum(mask)) < int(min_samples):
        return None
    x = np.asarray(eyepos[mask], dtype=np.float64)
    y = np.asarray(rates[mask], dtype=np.float64)

    x_centered = x - np.mean(x, axis=0, keepdims=True)
    bins = np.rint(x_centered / float(eye_bin_step)).astype(np.int64)

    sums_resp: dict[tuple[int, int], np.ndarray] = {}
    sums_eye: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], int] = {}

    for b, e, r in zip(bins, x_centered, y, strict=False):
        key = (int(b[0]), int(b[1]))
        if key not in sums_resp:
            sums_resp[key] = np.zeros_like(r, dtype=np.float64)
            sums_eye[key] = np.zeros(2, dtype=np.float64)
            counts[key] = 0
        sums_resp[key] += r
        sums_eye[key] += e
        counts[key] += 1

    resp_by_bin: dict[tuple[int, int], np.ndarray] = {}
    eye_by_bin: dict[tuple[int, int], np.ndarray] = {}
    for key, n in counts.items():
        if int(n) < int(min_bin_count):
            continue
        resp_by_bin[key] = sums_resp[key] / float(n)
        eye_by_bin[key] = sums_eye[key] / float(n)

    if len(resp_by_bin) < 4:
        return None

    count_vals = np.asarray(list(counts.values()), dtype=np.int64) if counts else np.asarray([], dtype=np.int64)
    hist: dict[int, int] = {}
    if count_vals.size:
        uniq, cnt = np.unique(count_vals, return_counts=True)
        hist = {int(u): int(c) for u, c in zip(uniq, cnt, strict=False)}

    diagnostics = {
        "occupied_bins": int(len(counts)),
        "retained_bins": int(len(resp_by_bin)),
        "retained_fraction": float(len(resp_by_bin) / max(len(counts), 1)),
        "min_bin_count_threshold": int(min_bin_count),
        "bin_count_min": int(np.min(count_vals)) if count_vals.size else 0,
        "bin_count_median": float(np.median(count_vals)) if count_vals.size else float("nan"),
        "bin_count_max": int(np.max(count_vals)) if count_vals.size else 0,
        "bin_count_distribution": json.dumps(hist, sort_keys=True),
        "eye_x_centered_min": float(np.min(x_centered[:, 0])) if x_centered.size else float("nan"),
        "eye_x_centered_max": float(np.max(x_centered[:, 0])) if x_centered.size else float("nan"),
        "eye_y_centered_min": float(np.min(x_centered[:, 1])) if x_centered.size else float("nan"),
        "eye_y_centered_max": float(np.max(x_centered[:, 1])) if x_centered.size else float("nan"),
        "eye_units_assumed": "degrees_visual_angle_assumed",
    }
    return resp_by_bin, eye_by_bin, diagnostics


def _compute_image_similarity(img_a: np.ndarray, img_b: np.ndarray) -> dict[str, float]:
    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    af = a.ravel()
    bf = b.ravel()
    pixel_corr = _pearson(af, bf)
    rms_diff = float(abs(np.std(af) - np.std(bf)))

    fa = np.abs(np.fft.rfft2(a))
    fb = np.abs(np.fft.rfft2(b))
    fourier_amp_corr = _pearson(fa.ravel(), fb.ravel())
    return {
        "pixel_correlation": pixel_corr,
        "rms_contrast_difference": rms_diff,
        "fourier_amplitude_similarity": fourier_amp_corr,
    }


def _bootstrap_ci(vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _image_bootstrap_weighted_mean(
    rows: list[dict[str, object]],
    value_key: str,
    image_ids: list[int],
    *,
    seed: int,
    n_bootstrap: int = 2000,
) -> np.ndarray:
    if not rows or not image_ids:
        return np.asarray([], dtype=np.float64)
    vals = np.asarray([float(r[value_key]) for r in rows], dtype=np.float64)
    img_i = np.asarray([int(r["image_i"]) for r in rows], dtype=np.int64)
    img_j = np.asarray([int(r["image_j"]) for r in rows], dtype=np.int64)
    base_images = np.asarray(image_ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    boots: list[float] = []

    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(base_images, size=base_images.size, replace=True)
        uniq, counts = np.unique(sampled, return_counts=True)
        count_map = {int(u): int(c) for u, c in zip(uniq, counts, strict=False)}
        weights = np.asarray(
            [float(count_map.get(int(i), 0) * count_map.get(int(j), 0)) for i, j in zip(img_i, img_j, strict=False)],
            dtype=np.float64,
        )
        keep = np.isfinite(vals) & (weights > 0)
        if not np.any(keep):
            continue
        boots.append(float(np.sum(vals[keep] * weights[keep]) / np.sum(weights[keep])))
    return np.asarray(boots, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2+3 STG residual RDM geometry with image-similarity controls")
    p.add_argument("--subject", type=str, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--dataset-configs-path", type=Path, default=Path("experiments") / "dataset_configs" / "multi_basic_240_rsvp.yaml")
    p.add_argument("--source", choices=("recorded", "twin"), required=True)
    p.add_argument("--image-set", type=str, default="high_support")
    p.add_argument("--min-samples", type=int, default=320)
    p.add_argument("--eye-bin-step", type=float, default=0.10)
    p.add_argument("--min-bin-count", type=int, default=2)
    p.add_argument("--min-common-bins", type=int, default=8)
    p.add_argument("--predict-batch-size", type=int, default=64)
    p.add_argument("--model-device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--use-cached-data", action="store_true", default=True)
    return p


def _source_rates(args: argparse.Namespace, data: dict[str, Any]) -> np.ndarray:
    if args.source == "recorded":
        return np.asarray(data["robs"], dtype=np.float64)
    twin_rates, _ = _predict_twin_rates(
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
    return np.asarray(twin_rates, dtype=np.float64)


def main() -> None:
    args = build_parser().parse_args()

    data = get_fixrsvp_data(
        subject=args.subject,
        date=args.date,
        dataset_configs_path=str(args.dataset_configs_path),
        use_cached_data=bool(args.use_cached_data),
    )
    data = harmonize_fixrsvp_arrays(data)

    rates = _source_rates(args, data)
    eyepos = np.asarray(data["eyepos"], dtype=np.float64)
    image_ids = np.asarray(data["image_ids"], dtype=np.int64)
    stim = np.asarray(data["stim"], dtype=np.float64)

    valid = np.isfinite(rates).all(axis=2) & np.isfinite(eyepos).all(axis=2) & (image_ids >= 0)
    all_img_ids = sorted(int(i) for i in np.unique(image_ids[valid]))

    image_bins: dict[int, tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], dict[str, object]]] = {}
    image_templates: dict[int, np.ndarray] = {}
    binning_diag_rows: list[dict[str, object]] = []
    for img in all_img_ids:
        built = _build_image_bin_map(
            rates,
            eyepos,
            image_ids,
            valid,
            img_id=img,
            min_samples=int(args.min_samples),
            eye_bin_step=float(args.eye_bin_step),
            min_bin_count=int(args.min_bin_count),
        )
        if built is None:
            continue
        image_bins[img] = built
        _, _, diag = built
        binning_diag_rows.append(
            {
                "session_id": f"{args.subject}_{args.date}",
                "subject": args.subject,
                "date": args.date,
                "source": args.source,
                "image_set": args.image_set,
                "analysis_representation": "eye_bin_averages",
                "image_id": int(img),
                "min_samples": int(args.min_samples),
                "eye_bin_step": float(args.eye_bin_step),
                **diag,
            }
        )

        mask = valid & (image_ids == int(img))
        st = stim[mask]
        if st.ndim >= 3:
            tpl = np.mean(st[..., 0, :, :] if st.ndim == 4 else st, axis=0)
            image_templates[img] = np.asarray(tpl, dtype=np.float64)

    usable = sorted(image_bins.keys())
    out_dir = Path(args.out_dir) / f"{args.subject}_{args.date}" / f"source_{args.source}"
    out_dir.mkdir(parents=True, exist_ok=True)

    smoothness_rows: list[dict[str, object]] = []
    for img in usable:
        resp_by_bin, eye_by_bin, _ = image_bins[img]
        keys = sorted(resp_by_bin.keys())
        resp = np.stack([resp_by_bin[k] for k in keys], axis=0)
        cen = np.stack([eye_by_bin[k] for k in keys], axis=0)
        _, feats = _pair_features(cen)
        for metric in ("euclidean_zscored", "correlation_distance"):
            d = _distance_vector(resp, metric)
            smoothness_rows.append(
                {
                    "session_id": f"{args.subject}_{args.date}",
                    "subject": args.subject,
                    "date": args.date,
                    "source": args.source,
                    "image_set": args.image_set,
                    "analysis_representation": "eye_bin_averages",
                    "image_id": int(img),
                    "n_samples": int(args.min_samples),
                    "n_bins": int(resp.shape[0]),
                    "distance_metric": metric,
                    "corr_with_radial_distance": _pearson(d, feats["radial"]),
                    "corr_with_abs_dx": _pearson(d, feats["abs_dx"]),
                    "corr_with_abs_dy": _pearson(d, feats["abs_dy"]),
                }
            )

    pair_rows: list[dict[str, object]] = []
    for img_i, img_j in itertools.combinations(usable, 2):
        resp_i, eye_i, _ = image_bins[img_i]
        resp_j, eye_j, _ = image_bins[img_j]
        common = sorted(set(resp_i.keys()) & set(resp_j.keys()))
        if len(common) < int(args.min_common_bins):
            continue

        ri = np.stack([resp_i[k] for k in common], axis=0)
        rj = np.stack([resp_j[k] for k in common], axis=0)
        ce = np.stack([eye_i[k] for k in common], axis=0)
        _, feats = _pair_features(ce)

        sim = {
            "pixel_correlation": float("nan"),
            "rms_contrast_difference": float("nan"),
            "fourier_amplitude_similarity": float("nan"),
        }
        if img_i in image_templates and img_j in image_templates:
            sim = _compute_image_similarity(image_templates[img_i], image_templates[img_j])

        for metric in ("euclidean_zscored", "correlation_distance"):
            di = _distance_vector(ri, metric)
            dj = _distance_vector(rj, metric)
            e1_i = _residualize(di, [feats["radial"]])
            e1_j = _residualize(dj, [feats["radial"]])
            e2_i = _residualize(di, [
                feats["radial"], feats["abs_dx"], feats["abs_dy"], feats["dx2"], feats["dy2"], feats["abs_dxdy"]
            ])
            e2_j = _residualize(dj, [
                feats["radial"], feats["abs_dx"], feats["abs_dy"], feats["dx2"], feats["dy2"], feats["abs_dxdy"]
            ])

            pair_rows.append(
                {
                    "session_id": f"{args.subject}_{args.date}",
                    "subject": args.subject,
                    "date": args.date,
                    "source": args.source,
                    "image_set": args.image_set,
                    "analysis_representation": "eye_bin_averages",
                    "image_i": int(img_i),
                    "image_j": int(img_j),
                    "n_samples": int(args.min_samples),
                    "n_common_bins": int(len(common)),
                    "distance_metric": metric,
                    "raw_rdm_corr": _pearson(di, dj),
                    "radial_residual_corr": _pearson(e1_i, e1_j),
                    "expanded_residual_corr": _pearson(e2_i, e2_j),
                    "pixel_correlation": sim["pixel_correlation"],
                    "rms_contrast_difference": sim["rms_contrast_difference"],
                    "fourier_amplitude_similarity": sim["fourier_amplitude_similarity"],
                }
            )

    control_rows: list[dict[str, object]] = []
    for metric in ("euclidean_zscored", "correlation_distance"):
        mrows = [r for r in pair_rows if r["distance_metric"] == metric]
        for target in ("raw_rdm_corr", "radial_residual_corr", "expanded_residual_corr"):
            for control in ("pixel_correlation", "rms_contrast_difference", "fourier_amplitude_similarity"):
                vals = [(float(r[target]), float(r[control])) for r in mrows if np.isfinite(float(r[target])) and np.isfinite(float(r[control]))]
                if len(vals) < 3:
                    continue
                y = np.asarray([v[0] for v in vals], dtype=np.float64)
                x = np.asarray([v[1] for v in vals], dtype=np.float64)
                x_centered = x - float(np.mean(x))
                X = np.stack([np.ones_like(x_centered), x_centered], axis=1)
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - (X @ beta)
                med = float(np.median(x))
                low = y[x <= med]
                high = y[x > med]
                control_rows.append(
                    {
                        "session_id": f"{args.subject}_{args.date}",
                        "subject": args.subject,
                        "date": args.date,
                        "source": args.source,
                        "image_set": args.image_set,
                        "analysis_representation": "eye_bin_averages",
                        "distance_metric": metric,
                        "target_metric": target,
                        "control_metric": control,
                        "n_pairs": int(y.size),
                        "alpha1": float(beta[1]),
                        "mean_metric": float(np.mean(y)),
                        "adjusted_mean_metric_at_centered_similarity": float(beta[0]),
                        "mean_residual_after_control": float(np.mean(resid)),
                        "mean_metric_low_similarity": float(np.mean(low)) if low.size else float("nan"),
                        "mean_metric_high_similarity": float(np.mean(high)) if high.size else float("nan"),
                    }
                )

    if not control_rows:
        control_rows.append(
            {
                "session_id": f"{args.subject}_{args.date}",
                "subject": args.subject,
                "date": args.date,
                "source": args.source,
                "image_set": args.image_set,
                "analysis_representation": "eye_bin_averages",
                "distance_metric": "not_available",
                "target_metric": "not_available",
                "control_metric": "not_available",
                "n_pairs": 0,
                "alpha1": float("nan"),
                "mean_metric": float("nan"),
                "adjusted_mean_metric_at_centered_similarity": float("nan"),
                "mean_residual_after_control": float("nan"),
                "mean_metric_low_similarity": float("nan"),
                "mean_metric_high_similarity": float("nan"),
            }
        )

    summary_rows: list[dict[str, object]] = []
    for metric in ("euclidean_zscored", "correlation_distance"):
        mrows = [r for r in pair_rows if r["distance_metric"] == metric]
        for model_name, col in (
            ("raw", "raw_rdm_corr"),
            ("radial_only", "radial_residual_corr"),
            ("expanded", "expanded_residual_corr"),
        ):
            rows_for_metric = [r for r in mrows if np.isfinite(float(r[col]))]
            vals = np.asarray([float(r[col]) for r in rows_for_metric], dtype=np.float64)
            boots = _image_bootstrap_weighted_mean(
                rows_for_metric,
                col,
                usable,
                seed=int(args.seed) + len(summary_rows) + 1,
                n_bootstrap=2000,
            )
            ci_low, ci_high = _bootstrap_ci(boots)
            mean_effect = float(np.mean(vals)) if vals.size else float("nan")

            if model_name == "expanded" and np.isfinite(ci_low) and ci_low > 0.0:
                label = "residual_shared_geometry"
            elif model_name == "radial_only" and np.isfinite(ci_low) and ci_low > 0.0:
                label = "residual_shared_geometry"
            elif model_name == "raw" and np.isfinite(ci_low) and ci_low > 0.0:
                label = "raw_only"
            else:
                label = "not_supported"

            summary_rows.append(
                {
                    "session_id": f"{args.subject}_{args.date}",
                    "subject": args.subject,
                    "date": args.date,
                    "source": args.source,
                    "image_set": args.image_set,
                    "analysis_representation": "eye_bin_averages",
                    "distance_metric": metric,
                    "model_name": model_name,
                    "bootstrap_unit": "image",
                    "n_images": int(len(usable)),
                    "n_pairs": int(vals.size),
                    "mean_effect": mean_effect,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "interpretation_label": label,
                }
            )

    _write_csv(out_dir / "stg_image_smoothness_metrics.csv", smoothness_rows)
    _write_csv(out_dir / "stg_cross_image_residual_rdm.csv", pair_rows)
    _write_csv(out_dir / "stg_image_similarity_controls.csv", control_rows)
    _write_csv(out_dir / "stg_session_rdm_summary.csv", summary_rows)
    _write_csv(out_dir / "stg_stage23_binning_diagnostics.csv", binning_diag_rows)

    metadata = {
        "session_id": f"{args.subject}_{args.date}",
        "source": args.source,
        "analysis_representation": "eye_bin_averages",
        "eye_bin_step": float(args.eye_bin_step),
        "min_bin_count": int(args.min_bin_count),
        "min_common_bins": int(args.min_common_bins),
        "bootstrap_unit": "image",
    }
    (out_dir / "stg_stage23_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(str(out_dir / "stg_image_smoothness_metrics.csv"))
    print(str(out_dir / "stg_cross_image_residual_rdm.csv"))
    print(str(out_dir / "stg_image_similarity_controls.csv"))
    print(str(out_dir / "stg_session_rdm_summary.csv"))
    print(str(out_dir / "stg_stage23_binning_diagnostics.csv"))
    print(str(out_dir / "stg_stage23_metadata.json"))


if __name__ == "__main__":
    main()
