"""Cache-only screen of local BackImage feature target spaces.

This runner reuses a completed local-pairing response cache and swaps the
decoded target features. It is meant to answer a narrower model-selection
question than the full local analysis: which image-feature space is worth
promoting to an uncapped local information analysis?
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .summarize_backimage_local_pairing_incremental_motion import (
        build_parser as build_incremental_parser,
        run as run_incremental_decoding,
    )
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.summarize_backimage_local_pairing_incremental_motion import (
        build_parser as build_incremental_parser,
        run as run_incremental_decoding,
    )


DEFAULT_LOCAL_RUN_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_local_pairing_Iz_power_pyramid_k16_rel025_0p5_1_seed7_k64_v1"
)

DEFAULT_SCREEN_TARGETS = (
    "image_contour_axis_code,"
    "pyramid_energy_global,"
    "pyramid_orientation_energy,"
    "pyramid_scale_energy,"
    "pyramid_band_energy,"
    "pyramid_local_contrast_energy,"
    "pyramid_energy_grid_coarse4,"
    "pyramid_energy_grid,"
    "pyramid_signed_grid,"
    "pyramid_local_field"
)


def _parse_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {key: np.asarray(loaded[key]) for key in loaded.files}


def _as_2d_finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _drop_constant_columns(values: np.ndarray, *, atol: float = 1e-12) -> tuple[np.ndarray, int]:
    arr = _as_2d_finite(values)
    if arr.shape[1] == 0:
        return arr, 0
    sd = np.nanstd(arr.astype(np.float64), axis=0)
    keep = np.isfinite(sd) & (sd > float(atol))
    if not np.any(keep):
        return np.empty((arr.shape[0], 0), dtype=np.float32), int(arr.shape[1])
    return arr[:, keep].astype(np.float32, copy=False), int(np.count_nonzero(~keep))


def _add_target(
    targets: dict[str, np.ndarray],
    manifest: list[dict[str, Any]],
    *,
    name: str,
    values: np.ndarray,
    source: str,
    description: str,
) -> None:
    arr_raw = _as_2d_finite(values)
    arr, n_dropped = _drop_constant_columns(arr_raw)
    if arr.shape[1] == 0:
        manifest.append(
            {
                "latent": name,
                "source": source,
                "target_dim_raw": int(arr_raw.shape[1]),
                "target_dim": 0,
                "dropped_constant_columns": int(n_dropped),
                "included": False,
                "description": description,
            }
        )
        return
    targets[name] = arr
    manifest.append(
        {
            "latent": name,
            "source": source,
            "target_dim_raw": int(arr_raw.shape[1]),
            "target_dim": int(arr.shape[1]),
            "dropped_constant_columns": int(n_dropped),
            "included": True,
            "description": description,
        }
    )


def _reshape_pyramid_local_field(
    values: np.ndarray,
    *,
    n_scales: int,
    n_orientations: int,
    n_channels: int,
) -> np.ndarray:
    arr = _as_2d_finite(values)
    block = int(n_scales) * int(n_orientations) * int(n_channels)
    if block <= 0 or arr.shape[1] % block != 0:
        raise ValueError(
            f"Cannot reshape pyramid field with dim={arr.shape[1]} into "
            f"{n_scales} scales x {n_orientations} orientations x {n_channels} channels"
        )
    n_grid_cells = arr.shape[1] // block
    return arr.reshape(arr.shape[0], int(n_scales), int(n_orientations), int(n_channels), int(n_grid_cells))


def _coarse_grid(values: np.ndarray, *, coarse_grid: int) -> np.ndarray | None:
    arr = np.asarray(values, dtype=np.float32)
    n, n_scales, n_orientations, n_cells = arr.shape
    side = int(round(float(np.sqrt(n_cells))))
    if side * side != n_cells or int(coarse_grid) <= 0 or side % int(coarse_grid) != 0:
        return None
    factor = side // int(coarse_grid)
    grid = arr.reshape(n, n_scales, n_orientations, side, side)
    pooled = grid.reshape(n, n_scales, n_orientations, int(coarse_grid), factor, int(coarse_grid), factor)
    pooled = pooled.mean(axis=(4, 6))
    return pooled.reshape(n, -1)


def _pyramid_feature_targets(
    latents: dict[str, np.ndarray],
    *,
    pyramid_latent_name: str,
    n_scales: int,
    n_orientations: int,
    n_channels: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    targets: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    if pyramid_latent_name not in latents:
        return targets, manifest

    original = _as_2d_finite(latents[pyramid_latent_name])
    tensor = _reshape_pyramid_local_field(
        original,
        n_scales=int(n_scales),
        n_orientations=int(n_orientations),
        n_channels=int(n_channels),
    )
    real_imag = tensor[:, :, :, :2, :]
    magnitude = tensor[:, :, :, 2, :]

    _add_target(
        targets,
        manifest,
        name=pyramid_latent_name,
        values=original,
        source=pyramid_latent_name,
        description="Full cached local steerable-pyramid field: real, imaginary, and magnitude grid features.",
    )
    _add_target(
        targets,
        manifest,
        name="pyramid_signed_grid",
        values=real_imag.reshape(real_imag.shape[0], -1),
        source=pyramid_latent_name,
        description="Local signed quadrature grid only, dropping magnitude channels.",
    )
    _add_target(
        targets,
        manifest,
        name="pyramid_energy_grid",
        values=magnitude.reshape(magnitude.shape[0], -1),
        source=pyramid_latent_name,
        description="Local magnitude/energy grid only, preserving scale, orientation, and grid cell.",
    )
    _add_target(
        targets,
        manifest,
        name="pyramid_band_energy",
        values=magnitude.mean(axis=-1).reshape(magnitude.shape[0], -1),
        source=pyramid_latent_name,
        description="Mean local magnitude per scale and orientation band.",
    )
    _add_target(
        targets,
        manifest,
        name="pyramid_orientation_energy",
        values=magnitude.mean(axis=(1, 3)),
        source=pyramid_latent_name,
        description="Mean local magnitude per orientation, averaged across scales and grid cells.",
    )
    _add_target(
        targets,
        manifest,
        name="pyramid_scale_energy",
        values=magnitude.mean(axis=(2, 3)),
        source=pyramid_latent_name,
        description="Mean local magnitude per scale, averaged across orientations and grid cells.",
    )
    _add_target(
        targets,
        manifest,
        name="pyramid_energy_global",
        values=magnitude.mean(axis=(1, 2, 3)),
        source=pyramid_latent_name,
        description="Single global mean local magnitude across the pyramid field.",
    )
    _add_target(
        targets,
        manifest,
        name="pyramid_signed_band_mean",
        values=real_imag.mean(axis=-1).reshape(real_imag.shape[0], -1),
        source=pyramid_latent_name,
        description="Mean signed real/imaginary coefficient per scale and orientation band.",
    )
    _add_target(
        targets,
        manifest,
        name="pyramid_local_contrast_energy",
        values=magnitude.std(axis=-1).reshape(magnitude.shape[0], -1),
        source=pyramid_latent_name,
        description="Across-grid standard deviation of magnitude per scale and orientation band.",
    )
    for coarse_grid in (2, 4):
        pooled = _coarse_grid(magnitude, coarse_grid=coarse_grid)
        if pooled is None:
            continue
        _add_target(
            targets,
            manifest,
            name=f"pyramid_energy_grid_coarse{coarse_grid}",
            values=pooled,
            source=pyramid_latent_name,
            description=f"Magnitude grid pooled to {coarse_grid} x {coarse_grid} cells per scale/orientation.",
        )
    return targets, manifest


def _axis_code_deg(axis_deg: np.ndarray, *, weight: np.ndarray | None = None) -> np.ndarray:
    radians = np.radians(np.asarray(axis_deg, dtype=np.float64) * 2.0)
    code = np.column_stack([np.cos(radians), np.sin(radians)]).astype(np.float32)
    if weight is not None:
        code = code * np.asarray(weight, dtype=np.float32)[:, None]
    return np.nan_to_num(code, nan=0.0, posinf=0.0, neginf=0.0)


def _image_metadata_targets(
    images: pd.DataFrame,
    *,
    include_motion_diagnostic_targets: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    targets: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    n = int(images.shape[0])

    if {"image_edge_axis_deg", "image_orientation_coherence"}.issubset(images.columns):
        coherence = images["image_orientation_coherence"].to_numpy(dtype=np.float32)
        edge_axis = images["image_edge_axis_deg"].to_numpy(dtype=np.float32)
        values = np.column_stack([coherence, _axis_code_deg(edge_axis, weight=coherence)])
        _add_target(
            targets,
            manifest,
            name="image_contour_axis_code",
            values=values,
            source="analysis_images.csv",
            description="Image contour coherence plus coherence-weighted 180-degree edge-axis code.",
        )

    if "image_orientation_coherence" in images.columns:
        _add_target(
            targets,
            manifest,
            name="image_orientation_coherence",
            values=images["image_orientation_coherence"].to_numpy(dtype=np.float32),
            source="analysis_images.csv",
            description="Scalar image orientation coherence.",
        )

    if "image_edge_axis_deg" in images.columns:
        _add_target(
            targets,
            manifest,
            name="image_edge_axis_code",
            values=_axis_code_deg(images["image_edge_axis_deg"].to_numpy(dtype=np.float32)),
            source="analysis_images.csv",
            description="Unweighted 180-degree image edge-axis code.",
        )

    if not include_motion_diagnostic_targets:
        return targets, manifest

    if {"image_edge_axis_deg", "drift_orientation_deg"}.issubset(images.columns):
        edge = images["image_edge_axis_deg"].to_numpy(dtype=np.float32)
        drift = images["drift_orientation_deg"].to_numpy(dtype=np.float32)
        alignment = np.cos(np.radians(2.0 * (edge - drift))).astype(np.float32)
        columns = [alignment]
        names = ["cos2_edge_minus_drift"]
        if "image_orientation_coherence" in images.columns:
            coherence = images["image_orientation_coherence"].to_numpy(dtype=np.float32)
            columns.append(coherence * alignment)
            names.append("coherence_weighted_alignment")
        if "drift_anisotropy" in images.columns:
            columns.append(images["drift_anisotropy"].to_numpy(dtype=np.float32))
            names.append("drift_anisotropy")
        values = np.column_stack(columns) if columns else np.empty((n, 0), dtype=np.float32)
        _add_target(
            targets,
            manifest,
            name="image_motion_alignment_diagnostic",
            values=values,
            source="analysis_images.csv",
            description=(
                "Diagnostic target using image-drift alignment fields "
                f"({', '.join(names)}); not a pure image-feature target."
            ),
        )

    return targets, manifest


def build_feature_targets(
    *,
    images: pd.DataFrame,
    latents: dict[str, np.ndarray],
    pyramid_latent_name: str,
    n_scales: int,
    n_orientations: int,
    n_channels: int,
    include_motion_diagnostic_targets: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    targets: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    image_targets, image_manifest = _image_metadata_targets(
        images,
        include_motion_diagnostic_targets=bool(include_motion_diagnostic_targets),
    )
    pyramid_targets, pyramid_manifest = _pyramid_feature_targets(
        latents,
        pyramid_latent_name=str(pyramid_latent_name),
        n_scales=int(n_scales),
        n_orientations=int(n_orientations),
        n_channels=int(n_channels),
    )
    targets.update(image_targets)
    targets.update(pyramid_targets)
    manifest.extend(image_manifest)
    manifest.extend(pyramid_manifest)
    return targets, manifest


def _filter_targets(
    targets: dict[str, np.ndarray],
    manifest: list[dict[str, Any]],
    requested: list[str],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    included = {str(row["latent"]) for row in manifest if bool(row.get("included", False))}
    if not requested or "screen_default" in requested:
        requested = _parse_list(DEFAULT_SCREEN_TARGETS)
    elif "all" in requested:
        requested = sorted(included)
    missing = sorted(set(requested).difference(included))
    if missing:
        raise ValueError(f"Requested target spaces are unavailable or constant: {missing}")
    keep = set(requested)
    return {name: targets[name] for name in requested}, [row for row in manifest if str(row["latent"]) in keep]


def _incremental_argv(args: argparse.Namespace, *, latent_npz: Path, latent_names: list[str], decode_out_dir: Path) -> list[str]:
    argv = [
        "--run-dir",
        str(args.run_dir),
        "--out-dir",
        str(decode_out_dir),
        "--latent-npz",
        str(latent_npz),
        "--summaries",
        str(args.summaries),
        "--families",
        str(args.families),
        "--contrast-pairs",
        str(args.contrast_pairs),
        "--scale-ids",
        str(args.scale_ids),
        "--latent-names",
        ",".join(latent_names),
        "--pca-k-list",
        str(args.pca_k_list),
        "--ridge-alphas",
        str(args.ridge_alphas),
        "--ridge-alpha-mode",
        str(args.ridge_alpha_mode),
        "--outer-folds",
        str(args.outer_folds),
        "--inner-folds",
        str(args.inner_folds),
        "--decode-group-mode",
        str(args.decode_group_mode),
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--max-sample-keys-per-family",
        str(args.max_sample_keys_per_family),
        "--information-variance-floor",
        str(args.information_variance_floor),
        "--seed",
        str(args.seed),
    ]
    if args.fixed_ridge_alpha is not None:
        argv.extend(["--fixed-ridge-alpha", str(args.fixed_ridge_alpha)])
    if bool(args.allow_unmatched_alpha_information):
        argv.append("--allow-unmatched-alpha-information")
    return argv


def _score_column(df: pd.DataFrame) -> str:
    if "incremental_gain_delta_info_diag_bits" in df.columns and np.isfinite(
        pd.to_numeric(df["incremental_gain_delta_info_diag_bits"], errors="coerce")
    ).any():
        return "incremental_gain_delta_info_diag_bits"
    return "incremental_gain_delta_neg_mse"


def _rank_screen_outputs(out_dir: Path, decode_out_dir: Path) -> None:
    contrast_path = decode_out_dir / "incremental_gain_contrasts.csv"
    gain_path = decode_out_dir / "incremental_gain_vs_static.csv"
    if not contrast_path.exists() or contrast_path.stat().st_size == 0:
        _write_csv(out_dir / "screen_ranked_contrasts.csv", [])
        _write_csv(out_dir / "screen_best_contrast_by_target.csv", [])
        return

    contrasts = pd.read_csv(contrast_path)
    if contrasts.empty:
        _write_csv(out_dir / "screen_ranked_contrasts.csv", [])
        _write_csv(out_dir / "screen_best_contrast_by_target.csv", [])
        return
    score_col = _score_column(contrasts)
    contrasts["screen_score"] = pd.to_numeric(contrasts[score_col], errors="coerce")
    ranked = contrasts.sort_values("screen_score", ascending=False)
    ranked.to_csv(out_dir / "screen_ranked_contrasts.csv", index=False)
    best = ranked.dropna(subset=["screen_score"]).groupby("latent", as_index=False).head(1)
    best.to_csv(out_dir / "screen_best_contrast_by_target.csv", index=False)
    _plot_ranked(best, out_dir / "screen_best_contrast_by_target.png", score_col=score_col)

    matched = contrasts[
        (contrasts["lhs_family"].astype(str) == "actual_paired_empirical")
        & (contrasts["rhs_family"].astype(str) == "matched_unpaired_empirical")
    ].copy()
    if not matched.empty:
        matched["screen_score"] = pd.to_numeric(matched[score_col], errors="coerce")
        matched_ranked = matched.sort_values("screen_score", ascending=False)
        matched_ranked.to_csv(out_dir / "screen_ranked_empirical_vs_matched.csv", index=False)
        matched_best = matched_ranked.dropna(subset=["screen_score"]).groupby("latent", as_index=False).head(1)
        matched_best.to_csv(out_dir / "screen_best_empirical_vs_matched_by_target.csv", index=False)
        _plot_ranked(
            matched_best,
            out_dir / "screen_best_empirical_vs_matched_by_target.png",
            score_col=score_col,
        )

    if gain_path.exists() and gain_path.stat().st_size > 0:
        gains = pd.read_csv(gain_path)
        if not gains.empty and "incremental_gain_info_diag_bits" in gains.columns:
            gains["screen_score"] = pd.to_numeric(gains["incremental_gain_info_diag_bits"], errors="coerce")
        elif not gains.empty:
            gains["screen_score"] = pd.to_numeric(gains["incremental_gain_neg_mse"], errors="coerce")
        best_gain = gains.sort_values("screen_score", ascending=False).dropna(subset=["screen_score"])
        best_gain = best_gain.groupby(["family", "latent"], as_index=False).head(1)
        best_gain.to_csv(out_dir / "screen_best_gain_vs_static_by_family_target.csv", index=False)


def _plot_ranked(best: pd.DataFrame, path: Path, *, score_col: str) -> None:
    if best.empty:
        return
    plot_df = best.sort_values("screen_score", ascending=True)
    height = max(4.0, 0.36 * float(plot_df.shape[0]) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, height), constrained_layout=True)
    colors = ["#4C78A8" if value >= 0 else "#F58518" for value in plot_df["screen_score"].to_numpy()]
    ax.barh(plot_df["latent"].astype(str), plot_df["screen_score"].to_numpy(dtype=float), color=colors)
    ax.axvline(0.0, color="0.2", linewidth=0.8)
    ax.set_xlabel(score_col)
    ax.set_ylabel("target space")
    ax.set_title("Best local feature-space contrast per target")
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_LOCAL_RUN_DIR)
    parser.add_argument("--base-latent-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--targets", default="screen_default")
    parser.add_argument("--pyramid-latent-name", default="pyramid_local_field")
    parser.add_argument("--pyramid-scales", type=int, default=4)
    parser.add_argument("--pyramid-orientations", type=int, default=4)
    parser.add_argument("--pyramid-channels", type=int, default=3)
    parser.add_argument("--include-motion-diagnostic-targets", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--summaries", default="delta_mean")
    parser.add_argument(
        "--families",
        default=(
            "actual_paired_empirical,matched_unpaired_empirical,rotated_actual_90,"
            "ou_matched_actual,brownian_matched_actual"
        ),
    )
    parser.add_argument(
        "--contrast-pairs",
        default=(
            "actual_paired_empirical:matched_unpaired_empirical,"
            "actual_paired_empirical:rotated_actual_90,"
            "actual_paired_empirical:ou_matched_actual,"
            "actual_paired_empirical:brownian_matched_actual"
        ),
    )
    parser.add_argument("--scale-ids", default="rel_1x")
    parser.add_argument("--pca-k-list", default="1,2,4,8,16")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--ridge-alpha-mode", choices=("fixed", "nested_per_candidate"), default="fixed")
    parser.add_argument("--fixed-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--decode-group-mode", choices=("image", "source_trial", "session"), default="source_trial")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--max-sample-keys-per-family", type=int, default=8)
    parser.add_argument("--information-variance-floor", type=float, default=1e-12)
    parser.add_argument("--allow-unmatched-alpha-information", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "local_feature_space_screen_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_latent_npz = Path(args.base_latent_npz) if args.base_latent_npz is not None else run_dir / "latent_feature_arrays.npz"
    images = pd.read_csv(run_dir / "analysis_images.csv")
    latents = _load_npz(base_latent_npz)
    targets, manifest = build_feature_targets(
        images=images,
        latents=latents,
        pyramid_latent_name=str(args.pyramid_latent_name),
        n_scales=int(args.pyramid_scales),
        n_orientations=int(args.pyramid_orientations),
        n_channels=int(args.pyramid_channels),
        include_motion_diagnostic_targets=bool(args.include_motion_diagnostic_targets),
    )
    targets, manifest = _filter_targets(targets, manifest, _parse_list(args.targets))
    target_npz = out_dir / "engineered_local_feature_targets.npz"
    np.savez_compressed(target_npz, **targets)
    _write_csv(out_dir / "engineered_local_feature_targets_manifest.csv", manifest)
    _write_json(
        out_dir / "run_metadata.json",
        {
            "source_run_dir": run_dir,
            "analysis_images": run_dir / "analysis_images.csv",
            "base_latent_feature_arrays": base_latent_npz,
            "engineered_latent_feature_arrays": target_npz,
            "target_names": list(targets),
            "target_manifest": manifest,
            "pyramid_shape_contract": {
                "latent": str(args.pyramid_latent_name),
                "scales": int(args.pyramid_scales),
                "orientations": int(args.pyramid_orientations),
                "channels": ["real", "imag", "magnitude"][: int(args.pyramid_channels)],
            },
            "screening_decode": {
                "summaries": str(args.summaries),
                "families": str(args.families),
                "contrast_pairs": str(args.contrast_pairs),
                "scale_ids": str(args.scale_ids),
                "pca_k_list": str(args.pca_k_list),
                "decode_group_mode": str(args.decode_group_mode),
                "max_sample_keys_per_family": int(args.max_sample_keys_per_family),
                "n_bootstrap": int(args.n_bootstrap),
                "seed": int(args.seed),
            },
        },
    )

    decode_out_dir = out_dir / "incremental_decode"
    if not bool(args.skip_decode):
        incremental_parser = build_incremental_parser()
        incremental_args = incremental_parser.parse_args(
            _incremental_argv(args, latent_npz=target_npz, latent_names=list(targets), decode_out_dir=decode_out_dir)
        )
        run_incremental_decoding(incremental_args)
        _rank_screen_outputs(out_dir, decode_out_dir)

    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Local Feature-Space Screen",
                "",
                "This is a cache-only screen over candidate feature target spaces for the local BackImage information model.",
                "It reuses the cached static and motion-rendered response summaries and changes only the decoded target `z`.",
                "",
                "Interpretation contract:",
                "- `delta_mean` is the motion-rendered response increment, decoded as `static mean + delta_mean`.",
                "- Gain-vs-static is the improvement of that augmented decoder over the stabilized-image static mean decoder.",
                "- Contrast rows are differences between two gain-vs-static estimates, usually empirical motion minus a matched control.",
                "- A positive empirical-minus-matched contrast means empirical pose-aware motion adds more information about that target than the control.",
                "",
                "Screening caveat:",
                f"- Sampled families were capped at {int(args.max_sample_keys_per_family)} sample arrays per family when this value is positive.",
                "- Promote promising target spaces to an uncapped rerun before treating the estimate as final.",
                "",
                "Primary files:",
                "- `engineered_local_feature_targets.npz`",
                "- `engineered_local_feature_targets_manifest.csv`",
                "- `incremental_decode/incremental_gain_vs_static.csv`",
                "- `incremental_decode/incremental_gain_contrasts.csv`",
                "- `screen_ranked_contrasts.csv`",
                "- `screen_best_contrast_by_target.csv`",
                "- `screen_best_empirical_vs_matched_by_target.csv`",
                "- `screen_best_contrast_by_target.png`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote local feature-space screen to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
