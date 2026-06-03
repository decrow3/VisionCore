from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.shared_transformation_geometry.utils import DEFAULT_OUT_ROOT  # type: ignore
else:
    from .utils import DEFAULT_OUT_ROOT


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 0.0 or bn <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def _gram_feature(j: np.ndarray) -> tuple[np.ndarray, float]:
    jj = np.asarray(j, dtype=np.float64)
    g = jj.T @ jj  # 2x2
    tr = float(np.trace(g))
    if tr <= 0.0 or (not np.isfinite(tr)):
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float64), 0.0
    g_norm = g / tr
    feat = np.asarray([float(g_norm[0, 0]), float(g_norm[1, 1]), float(g_norm[0, 1])], dtype=np.float64)
    eig = np.linalg.eigvalsh(g_norm)
    eig = np.sort(eig)[::-1]
    frac1 = float(eig[0]) if eig.size else 0.0
    return feat, frac1


def _fit_tangent_map(rates: np.ndarray, dxdy: np.ndarray, ridge_alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(dxdy, dtype=np.float64)
    y = np.asarray(rates, dtype=np.float64)
    xc = x - np.mean(x, axis=0, keepdims=True)
    yc = y - np.mean(y, axis=0, keepdims=True)
    xtx = xc.T @ xc
    b = np.linalg.solve(xtx + (ridge_alpha * np.eye(xtx.shape[0], dtype=np.float64)), xc.T @ yc)
    bx = b[0, :].copy()
    by = b[1, :].copy()
    j = np.stack([bx, by], axis=1)
    return bx, by, j


def _random_map_with_norms(n_units: int, norm_x: float, norm_y: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bx = rng.standard_normal(n_units)
    by = rng.standard_normal(n_units)
    bx *= norm_x / (np.linalg.norm(bx) + 1e-12)
    by *= norm_y / (np.linalg.norm(by) + 1e-12)
    j = np.stack([bx, by], axis=1)
    return bx, by, j


def _bootstrap_ci(vals: np.ndarray) -> tuple[float, float]:
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _image_bootstrap_mean(rows: list[dict[str, object]], value_key: str, *, seed: int, n_bootstrap: int) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=np.float64)
    vals = np.asarray([float(r[value_key]) for r in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, vals.size, size=vals.size)
        boots.append(float(np.mean(vals[idx])))
    return np.asarray(boots, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1b twin-template tangent confirmation")
    p.add_argument("--subject", type=str, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--n-nulls", type=int, default=200)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    return p


def _load_maps(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "images" not in payload:
        raise ValueError(f"Malformed tangent map file: {path}")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    session_root = Path(args.out_dir) / f"{args.subject}_{args.date}"
    rec_path = session_root / "source_recorded" / "stg_tangent_maps.pkl"
    twin_path = session_root / "source_twin" / "stg_tangent_maps.pkl"
    if (not rec_path.exists()) or (not twin_path.exists()):
        raise FileNotFoundError(
            f"Expected both files: {rec_path} and {twin_path}. Run Stage 1 for recorded and twin first."
        )

    rec = _load_maps(rec_path)
    twin = _load_maps(twin_path)
    rec_imgs = {int(k): v for k, v in rec["images"].items()}
    twin_imgs = {int(k): v for k, v in twin["images"].items()}

    common_ids = sorted(set(rec_imgs.keys()) & set(twin_imgs.keys()))
    if len(common_ids) < 4:
        raise ValueError(f"Need >=4 common images for template confirmation, got {len(common_ids)}")

    n_thresh_rec = int(rec.get("n_samples_threshold", 0))
    n_thresh_twin = int(twin.get("n_samples_threshold", 0))
    n_thresh = max(n_thresh_rec, n_thresh_twin)

    high_support = [
        img
        for img in common_ids
        if int(twin_imgs[img].get("n_samples_used", 0)) >= n_thresh and int(rec_imgs[img].get("n_samples_used", 0)) >= n_thresh
    ]
    if len(high_support) < 4:
        high_support = common_ids

    twin_features = []
    twin_frac1 = []
    twin_feature_by_image: dict[int, np.ndarray] = {}
    twin_frac1_by_image: dict[int, float] = {}
    for img in high_support:
        j_t = np.asarray(twin_imgs[img]["j"], dtype=np.float64)
        feat_t, frac_t = _gram_feature(j_t)
        twin_features.append(feat_t)
        twin_frac1.append(frac_t)
        twin_feature_by_image[img] = feat_t
        twin_frac1_by_image[img] = frac_t

    feat_tpl = np.mean(np.stack(twin_features, axis=0), axis=0)
    feat_tpl = feat_tpl / (np.linalg.norm(feat_tpl) + 1e-12)
    frac1_tpl = float(np.mean(np.asarray(twin_frac1, dtype=np.float64)))

    match_rows: list[dict[str, object]] = []
    for img in high_support:
        r = rec_imgs[img]
        j = np.asarray(r["j"], dtype=np.float64)
        feat_r, frac1_r = _gram_feature(j)

        signed = _cos(feat_r, feat_tpl)
        subspace = float(1.0 - abs(frac1_r - frac1_tpl))

        eye_null_signed: list[float] = []
        eye_null_subspace: list[float] = []
        img_null_signed: list[float] = []
        img_null_subspace: list[float] = []
        rand_null_signed: list[float] = []
        rand_null_subspace: list[float] = []

        dxdy = np.asarray(r["dxdy"], dtype=np.float64)
        y = np.asarray(r["y"], dtype=np.float64)
        n_units = int(r.get("n_units", y.shape[1]))
        for _ in range(int(args.n_nulls)):
            perm = rng.permutation(dxdy.shape[0])
            _, _, j_s = _fit_tangent_map(y, dxdy[perm], ridge_alpha=float(args.ridge_alpha))
            feat_s, frac1_s = _gram_feature(j_s)
            eye_null_signed.append(_cos(feat_s, feat_tpl))
            eye_null_subspace.append(float(1.0 - abs(frac1_s - frac1_tpl)))

            pick = int(rng.choice(high_support))
            feat_t = twin_feature_by_image[pick]
            frac1_t = twin_frac1_by_image[pick]
            img_null_signed.append(_cos(feat_r, feat_t))
            img_null_subspace.append(float(1.0 - abs(frac1_r - frac1_t)))

            rbx, rby, rj = _random_map_with_norms(n_units, float(np.linalg.norm(r["bx"])), float(np.linalg.norm(r["by"])), rng)
            feat_rand, frac1_rand = _gram_feature(rj)
            rand_null_signed.append(_cos(feat_rand, feat_tpl))
            rand_null_subspace.append(float(1.0 - abs(frac1_rand - frac1_tpl)))

        eye_null_signed_arr = np.asarray(eye_null_signed, dtype=np.float64)
        img_null_signed_arr = np.asarray(img_null_signed, dtype=np.float64)
        rand_null_signed_arr = np.asarray(rand_null_signed, dtype=np.float64)
        eye_null_subspace_arr = np.asarray(eye_null_subspace, dtype=np.float64)
        img_null_subspace_arr = np.asarray(img_null_subspace, dtype=np.float64)
        rand_null_subspace_arr = np.asarray(rand_null_subspace, dtype=np.float64)

        match_rows.append(
            {
                "session_id": f"{args.subject}_{args.date}",
                "subject": args.subject,
                "date": args.date,
                "image_id": int(img),
                "template_feature_type": "gram_JtJ",
                "template_match_semantics": "unit_count_invariant_tangent_metric_match",
                "n_samples_used_recorded": int(r.get("n_samples_used", 0)),
                "n_samples_used_twin": int(twin_imgs[img].get("n_samples_used", 0)),
                "recorded_n_units": int(np.asarray(r["j"]).shape[0]),
                "twin_n_units": int(np.asarray(twin_imgs[img]["j"]).shape[0]),
                "signed_template_match": signed,
                "subspace_template_match": subspace,
                "eye_shuffle_null_signed_mean": float(np.mean(eye_null_signed_arr)),
                "image_label_shuffle_null_signed_mean": float(np.mean(img_null_signed_arr)),
                "random_map_null_signed_mean": float(np.mean(rand_null_signed_arr)),
                "effect_signed_minus_eye_shuffle": float(signed - np.mean(eye_null_signed_arr)),
                "effect_signed_minus_image_shuffle": float(signed - np.mean(img_null_signed_arr)),
                "effect_signed_minus_random_map": float(signed - np.mean(rand_null_signed_arr)),
                "eye_shuffle_null_subspace_mean": float(np.mean(eye_null_subspace_arr)),
                "image_label_shuffle_null_subspace_mean": float(np.mean(img_null_subspace_arr)),
                "random_map_null_subspace_mean": float(np.mean(rand_null_subspace_arr)),
                "effect_subspace_minus_eye_shuffle": float(subspace - np.mean(eye_null_subspace_arr)),
                "effect_subspace_minus_image_shuffle": float(subspace - np.mean(img_null_subspace_arr)),
                "effect_subspace_minus_random_map": float(subspace - np.mean(rand_null_subspace_arr)),
            }
        )

    keys = (
        "effect_signed_minus_eye_shuffle",
        "effect_signed_minus_image_shuffle",
        "effect_signed_minus_random_map",
        "effect_subspace_minus_eye_shuffle",
        "effect_subspace_minus_image_shuffle",
        "effect_subspace_minus_random_map",
    )
    summary: dict[str, object] = {
        "session_id": f"{args.subject}_{args.date}",
        "subject": args.subject,
        "date": args.date,
        "template_feature_type": "gram_JtJ",
        "template_match_semantics": "unit_count_invariant_tangent_metric_match",
        "n_images_common": int(len(common_ids)),
        "n_images_template": int(len(high_support)),
        "n_nulls": int(args.n_nulls),
        "bootstrap_unit": "image",
    }
    for idx, key in enumerate(keys, start=1):
        arr = np.asarray([float(r[key]) for r in match_rows], dtype=np.float64)
        boots = _image_bootstrap_mean(match_rows, key, seed=int(args.seed) + idx, n_bootstrap=int(args.bootstrap_repeats))
        ci = _bootstrap_ci(boots)
        p = float(np.mean(boots <= 0.0)) if boots.size else float("nan")
        summary[f"mean_{key}"] = float(np.mean(arr)) if arr.size else float("nan")
        summary[f"ci_low_{key}"] = float(ci[0])
        summary[f"ci_high_{key}"] = float(ci[1])
        summary[f"p_{key}_le_0"] = p

    ci_eye = float(summary["ci_low_effect_signed_minus_eye_shuffle"])
    ci_img = float(summary["ci_low_effect_signed_minus_image_shuffle"])
    ci_rand = float(summary["ci_low_effect_signed_minus_random_map"])
    summary["interpretation_label"] = (
        "template_confirmed"
        if np.isfinite(ci_eye) and np.isfinite(ci_img) and np.isfinite(ci_rand) and ci_eye > 0.0 and ci_img > 0.0 and ci_rand > 0.0
        else "not_supported"
    )
    summary["interpretation_label_semantic"] = "tangent_metric_match"

    out_match = session_root / "stg_tangent_template_match.csv"
    out_summary = session_root / "stg_tangent_template_summary.csv"
    out_meta = session_root / "stg_tangent_template_metadata.json"
    _write_csv(out_match, match_rows)
    _write_csv(out_summary, [summary])
    out_meta.write_text(
        json.dumps(
            {
                "session_id": f"{args.subject}_{args.date}",
                "high_support_images": [int(i) for i in high_support],
                "nulls": ["eye_label_shuffle", "image_label_shuffle", "random_map"],
                "template_feature_type": "gram_JtJ",
                "template_match_semantics": "unit_count_invariant_tangent_metric_match",
                "matching_space": "gram_feature_from_JtJ",
                "ridge_alpha": float(args.ridge_alpha),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(str(out_match))
    print(str(out_summary))
    print(str(out_meta))


if __name__ == "__main__":
    main()
