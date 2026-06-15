"""Incremental static-plus-motion decoding for BackImage aggregate FEM runs.

This cache-only posthoc asks whether motion summaries add image-feature
decoding signal beyond the full static response summary:

    z ~ R_static
    z ~ R_static + R_motion

It also compares the incremental gain from empirical motion against matched
OU/Brownian/rotated controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    from .run_backimage_latent_information_screen import _cross_validated_decode
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import _cross_validated_decode


DEFAULT_RUN_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_aggregate_fem_information_n128_k4_rel025-1_gabor_pyramid"
)


STATIC_SUMMARY_FOR_MOTION = {
    "temporal_pca": "temporal_pca",
    "temporal_delta_pca": "temporal_pca",
    "temporal_dct": "temporal_dct",
    "temporal_dct_delta": "temporal_dct",
    "mean": "mean",
    "delta_mean": "mean",
}


def _parse_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in _parse_list(text)]


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in _parse_list(text)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {key: np.asarray(loaded[key]) for key in loaded.files}


def _filter_latents(latents: dict[str, np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
    if not names or "all" in names:
        return {key: value for key, value in latents.items()}
    missing = sorted(set(names).difference(latents))
    if missing:
        raise ValueError(f"Requested latent arrays are missing: {missing}")
    return {name: latents[name] for name in names}


def _response_key(summary: str, family: str, scale_id: str) -> str:
    return f"{summary}__{family}__{scale_id}"


def _available_scale_ids(responses: dict[str, np.ndarray], families: list[str], summaries: list[str]) -> list[str]:
    scales: set[str] = set()
    for key in responses:
        parts = key.split("__")
        if len(parts) != 3:
            continue
        summary, family, scale_id = parts
        if summary in summaries and family in families and scale_id != "static":
            scales.add(scale_id)
    return sorted(scales, key=lambda s: (len(s), s))


def _session_bootstrap_delta(
    left: np.ndarray,
    right: np.ndarray,
    sessions: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    sessions = np.asarray(sessions)
    ok = np.isfinite(delta)
    delta = delta[ok]
    sessions = sessions[ok]
    if delta.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "n_sessions": 0}
    session_values = pd.DataFrame({"session": sessions, "delta": delta}).groupby("session")["delta"].mean().to_numpy(dtype=np.float64)
    mean = float(np.nanmean(session_values))
    if int(n_bootstrap) <= 0 or session_values.size <= 1:
        lo = hi = mean
    else:
        boot = np.empty(int(n_bootstrap), dtype=np.float64)
        for i in range(int(n_bootstrap)):
            boot[i] = float(np.nanmean(rng.choice(session_values, size=session_values.size, replace=True)))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "mean": mean,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(delta.size),
        "n_sessions": int(session_values.size),
    }


def _decode(
    X: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    *,
    k: int,
    alphas: list[float],
    fixed_alpha: float,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict[str, Any]:
    return _cross_validated_decode(
        X,
        Z,
        groups,
        k=int(k),
        alphas=alphas,
        alpha_mode="fixed",
        fixed_alpha=float(fixed_alpha),
        outer_folds=int(outer_folds),
        inner_folds=int(inner_folds),
        seed=int(seed),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--summaries", default="temporal_pca,temporal_dct,temporal_delta_pca,temporal_dct_delta,mean,delta_mean")
    parser.add_argument("--families", default="empirical,ou,brownian,rotated")
    parser.add_argument("--scale-ids", default="all")
    parser.add_argument("--latent-names", default="gabor_local_field,pyramid_local_field")
    parser.add_argument("--pca-k-list", default="4,8")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--fixed-ridge-alpha", type=float, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--decode-group-mode",
        choices=("image", "session"),
        default="image",
        help="CV grouping for decoding. Use image for the pathfinder; session is stricter by recording session.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "incremental_static_plus_motion"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    images = pd.read_csv(run_dir / "analysis_images.csv")
    sessions = images["session"].to_numpy()
    decode_groups = images["image_index"].to_numpy(dtype=int) if str(args.decode_group_mode) == "image" else sessions
    latents = _filter_latents(_load_npz(run_dir / "latent_feature_arrays.npz"), _parse_list(args.latent_names))
    responses = _load_npz(run_dir / "response_summary_arrays.npz")
    summaries = _parse_list(args.summaries)
    families = _parse_list(args.families)
    scale_ids = _parse_list(args.scale_ids)
    if not scale_ids or "all" in scale_ids:
        scale_ids = _available_scale_ids(responses, families, summaries)
    alphas = _parse_float_list(args.ridge_alphas)
    fixed_alpha = float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else float(alphas[len(alphas) // 2])
    pca_k_list = _parse_int_list(args.pca_k_list)

    decode_rows: list[dict[str, Any]] = []
    per_image: dict[tuple[str, str, str, str, str, int], np.ndarray] = {}

    for summary in summaries:
        static_summary = STATIC_SUMMARY_FOR_MOTION.get(summary)
        if static_summary is None:
            raise ValueError(f"No static summary mapping is defined for {summary!r}")
        static_key = _response_key(static_summary, "static", "static")
        if static_key not in responses:
            raise ValueError(f"Missing static response array {static_key!r}")
        X_static = responses[static_key]
        for latent_name, Z in latents.items():
            for k in pca_k_list:
                static_result = _decode(
                    X_static,
                    Z,
                    decode_groups,
                    k=k,
                    alphas=alphas,
                    fixed_alpha=fixed_alpha,
                    outer_folds=int(args.outer_folds),
                    inner_folds=int(args.inner_folds),
                    seed=int(args.seed),
                )
                static_per_key = (summary, "static_only", "static", "static", latent_name, int(k))
                per_image[static_per_key] = np.asarray(static_result["per_window_score"], dtype=np.float64)
                decode_rows.append(
                    {
                        "motion_summary": summary,
                        "static_summary": static_summary,
                        "model": "static_only",
                        "family": "static",
                        "scale_id": "static",
                        "latent": latent_name,
                        "k": int(k),
                        "mean_neg_mse": float(static_result["mean_neg_mse"]),
                        "r2": float(static_result["r2"]),
                        "chosen_alpha_median": float(static_result["chosen_alpha_median"]),
                        "target_dim": int(static_result["target_dim"]),
                        "n_images": int(X_static.shape[0]),
                        "decode_group_mode": str(args.decode_group_mode),
                        "n_decode_groups": int(np.unique(decode_groups).size),
                        "feature_dim": int(X_static.shape[1]),
                    }
                )
                for scale_id in scale_ids:
                    for family in families:
                        motion_key = _response_key(summary, family, scale_id)
                        if motion_key not in responses:
                            continue
                        X_motion = responses[motion_key]
                        X_aug = np.concatenate([X_static, X_motion], axis=1)
                        aug_result = _decode(
                            X_aug,
                            Z,
                            decode_groups,
                            k=k,
                            alphas=alphas,
                            fixed_alpha=fixed_alpha,
                            outer_folds=int(args.outer_folds),
                            inner_folds=int(args.inner_folds),
                            seed=int(args.seed),
                        )
                        key = (summary, "static_plus_motion", family, scale_id, latent_name, int(k))
                        per_image[key] = np.asarray(aug_result["per_window_score"], dtype=np.float64)
                        decode_rows.append(
                            {
                                "motion_summary": summary,
                                "static_summary": static_summary,
                                "model": "static_plus_motion",
                                "family": family,
                                "scale_id": scale_id,
                                "latent": latent_name,
                                "k": int(k),
                                "mean_neg_mse": float(aug_result["mean_neg_mse"]),
                                "r2": float(aug_result["r2"]),
                                "chosen_alpha_median": float(aug_result["chosen_alpha_median"]),
                                "target_dim": int(aug_result["target_dim"]),
                                "n_images": int(X_aug.shape[0]),
                                "decode_group_mode": str(args.decode_group_mode),
                                "n_decode_groups": int(np.unique(decode_groups).size),
                                "feature_dim": int(X_aug.shape[1]),
                            }
                        )

    gain_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for summary in summaries:
        for latent_name in latents:
            for k in pca_k_list:
                static_key = (summary, "static_only", "static", "static", latent_name, int(k))
                if static_key not in per_image:
                    continue
                for scale_id in scale_ids:
                    gain_by_family: dict[str, np.ndarray] = {}
                    for family in families:
                        aug_key = (summary, "static_plus_motion", family, scale_id, latent_name, int(k))
                        if aug_key not in per_image:
                            continue
                        gain = per_image[aug_key] - per_image[static_key]
                        gain_by_family[family] = gain
                        boot = _session_bootstrap_delta(per_image[aug_key], per_image[static_key], sessions, rng=rng, n_bootstrap=int(args.n_bootstrap))
                        gain_rows.append(
                            {
                                "motion_summary": summary,
                                "family": family,
                                "scale_id": scale_id,
                                "latent": latent_name,
                                "k": int(k),
                                "incremental_gain_neg_mse": boot["mean"],
                                "ci95_low": boot["ci_low"],
                                "ci95_high": boot["ci_high"],
                                "n_images": boot["n"],
                                "n_sessions": boot["n_sessions"],
                            }
                        )
                    if "empirical" in gain_by_family:
                        for rhs in ("ou", "brownian", "rotated"):
                            if rhs not in gain_by_family:
                                continue
                            boot = _session_bootstrap_delta(gain_by_family["empirical"], gain_by_family[rhs], sessions, rng=rng, n_bootstrap=int(args.n_bootstrap))
                            contrast_rows.append(
                                {
                                    "motion_summary": summary,
                                    "lhs_family": "empirical",
                                    "rhs_family": rhs,
                                    "scale_id": scale_id,
                                    "latent": latent_name,
                                    "k": int(k),
                                    "incremental_gain_delta_neg_mse": boot["mean"],
                                    "ci95_low": boot["ci_low"],
                                    "ci95_high": boot["ci_high"],
                                    "n_images": boot["n"],
                                    "n_sessions": boot["n_sessions"],
                                }
                            )

    _write_csv(out_dir / "incremental_decode_summary.csv", decode_rows)
    _write_csv(out_dir / "incremental_gain_vs_static.csv", gain_rows)
    _write_csv(out_dir / "incremental_gain_contrasts.csv", contrast_rows)
    _write_json(
        out_dir / "run_metadata.json",
        {
            "source_run_dir": run_dir,
            "summaries": summaries,
            "static_summary_for_motion": STATIC_SUMMARY_FOR_MOTION,
            "families": families,
            "scale_ids": scale_ids,
            "latent_names": list(latents),
            "pca_k_list": pca_k_list,
            "ridge_alpha_mode": "fixed",
            "fixed_ridge_alpha": fixed_alpha,
            "decode_group_mode": str(args.decode_group_mode),
            "n_decode_groups": int(np.unique(decode_groups).size),
            "outer_folds": int(args.outer_folds),
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
        },
    )
    report = [
        "# Incremental Static Plus Motion",
        "",
        f"Source run: `{run_dir}`",
        "",
        "Question:",
        "",
        "`z ~ R_static` versus `z ~ R_static + R_motion_summary`.",
        "",
        "Primary files:",
        "- `incremental_gain_vs_static.csv`",
        "- `incremental_gain_contrasts.csv`",
        "- `incremental_decode_summary.csv`",
    ]
    (out_dir / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote incremental summaries to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
