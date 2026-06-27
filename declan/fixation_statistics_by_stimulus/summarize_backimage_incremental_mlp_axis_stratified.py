"""Axis-alignment stratification of the 4B MLP decoder results.

Uses the existing 384-image 4B dataset. For each image, the actual observed
eye movement trajectory has a known orientation relative to the dominant image
edge axis. This stratification asks:

    Does the MLP motion-response information gain depend on whether the
    trajectory ran along the dominant image edge (parallel) or across it
    (orthogonal)?

Axis alignment: cos²(drift_orientation_deg − image_edge_axis_deg)
    1.0  = perfectly parallel  (along edge)
    0.0  = perfectly orthogonal (across edge)
    Split at 0.5 → 'along' vs 'across'

For each (input_mode, summary, k, latent):
    • compute MLP gain = per_window_cosine(condition) − per_window_cosine(static)
    • split images by alignment: along (cos² > 0.5) vs across (cos² ≤ 0.5)
    • bootstrap within each subset
    • also compute continuous Pearson r between cos² and per-window gain

Input modes mirror the 4B script:
    static_only   — MLP(X_static) baseline
    motion_only   — MLP(X_motion)
    augmented     — MLP([X_static, X_motion])

Outputs
-------
mlp_axis_stratified_gain.csv
    Tidy table: one row per (input_mode, motion_summary, axis_group, k, latent)
per_window_cosines.npz
    Raw per-window cosines, one array per (input_mode, motion_summary, k, latent)
figures/

Usage::

    python -m declan.fixation_statistics_by_stimulus.summarize_backimage_incremental_mlp_axis_stratified \\
        --run-dir outputs/.../backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1 \\
        --window-manifest outputs/.../backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv \\
        --summaries delta_mean,mean \\
        --input-modes motion_only,augmented \\
        --pca-k-list 8,16 \\
        --mlp-device auto
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
from scipy import stats as scipy_stats

try:
    from sklearn.decomposition import PCA
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn required") from exc

try:
    from .summarize_backimage_aggregate_incremental_motion import (
        STATIC_SUMMARY_FOR_MOTION,
        _available_scale_ids,
        _response_key,
        _session_bootstrap_delta,
    )
    from .summarize_backimage_aggregate_incremental_motion_mlp import (
        _build_X,
        _mlp_cross_validated_decode,
        _pca_transform_fold,
        _per_window_cosine,
        _per_window_neg_mse,
        _split_by_groups,
    )
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion import (
        STATIC_SUMMARY_FOR_MOTION,
        _available_scale_ids,
        _response_key,
        _session_bootstrap_delta,
    )
    from declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion_mlp import (
        _build_X,
        _mlp_cross_validated_decode,
        _pca_transform_fold,
        _per_window_cosine,
        _per_window_neg_mse,
        _split_by_groups,
    )

try:
    from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_feature_embedding_reconstruction import (
        MLPConfig,
        _assign_source_folds,
        _fit_predict_mlp,
        _resolve_torch_device,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("build_panel_c_continuous_feature_embedding_reconstruction required") from exc


DEFAULT_RUN_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1"
)
DEFAULT_WINDOW_MANIFEST = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)

AXIS_COLORS = {
    "along": "#2e7d32",
    "across": "#6a1b9a",
    "all": "#555555",
}
AXIS_LABELS = {"along": "along edge\n(cos²≥0.5)", "across": "across edge\n(cos²<0.5)", "all": "all images"}


# ---------------------------------------------------------------------------
# Axis alignment loading
# ---------------------------------------------------------------------------

def _load_axis_alignment(
    analysis_images: pd.DataFrame,
    window_manifest_path: Path,
    cos2_threshold: float = 0.5,
) -> pd.DataFrame:
    """Join analysis_images with edge-axis and drift-orientation columns.

    Returns analysis_images with added columns:
        image_edge_axis_deg, drift_orientation_deg, cos2_alignment, axis_group
    """
    wm = pd.read_csv(window_manifest_path, usecols=["session", "trial_idx", "image_edge_axis_deg", "drift_orientation_deg"])

    # source_row in analysis_images is the row index into the window manifest
    wm_indexed = wm.reset_index(drop=False).rename(columns={"index": "source_row"})
    merged = analysis_images.merge(
        wm_indexed[["source_row", "image_edge_axis_deg", "drift_orientation_deg"]],
        on="source_row", how="left",
    )

    angle_diff_rad = np.deg2rad(merged["drift_orientation_deg"] - merged["image_edge_axis_deg"])
    merged["cos2_alignment"] = np.cos(angle_diff_rad) ** 2
    merged["axis_group"] = np.where(merged["cos2_alignment"] >= cos2_threshold, "along", "across")
    return merged


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _stratified_bootstrap(
    per_window_gain: np.ndarray,
    sessions: np.ndarray,
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    """Bootstrap CI for mean gain within a subset of images."""
    sub_gain = per_window_gain[mask]
    sub_sessions = sessions[mask]
    unique_sessions = np.unique(sub_sessions)
    n_sess = len(unique_sessions)
    session_means = np.array([np.nanmean(sub_gain[sub_sessions == s]) for s in unique_sessions])
    boots = [
        np.nanmean(rng.choice(session_means, size=n_sess, replace=True))
        for _ in range(n_bootstrap)
    ]
    boots = np.array(boots)
    return {
        "mean": float(np.nanmean(session_means)),
        "ci_low": float(np.nanpercentile(boots, 2.5)),
        "ci_high": float(np.nanpercentile(boots, 97.5)),
        "n": int(np.sum(mask)),
        "n_sessions": int(n_sess),
    }


def _axis_correlation(
    per_window_gain: np.ndarray,
    cos2_alignment: np.ndarray,
    sessions: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    """Session-mean Pearson r between cos² alignment and per-window gain."""
    unique_sessions = np.unique(sessions)
    session_r_vals = []
    for s in unique_sessions:
        mask = sessions == s
        if mask.sum() < 3:
            continue
        r, _ = scipy_stats.pearsonr(cos2_alignment[mask], per_window_gain[mask])
        if np.isfinite(r):
            session_r_vals.append(r)
    if not session_r_vals:
        return {"r": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_sessions": 0}
    session_r = np.array(session_r_vals)
    boots = [np.nanmean(rng.choice(session_r, size=len(session_r), replace=True)) for _ in range(n_bootstrap)]
    return {
        "r": float(np.nanmean(session_r)),
        "ci_low": float(np.nanpercentile(boots, 2.5)),
        "ci_high": float(np.nanpercentile(boots, 97.5)),
        "n_sessions": int(len(session_r)),
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _parse_list(text: str) -> list[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(p) for p in _parse_list(text)]


def _jsonable(v: Any) -> Any:
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, dict): return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_jsonable(x) for x in v]
    return v


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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8dde3", lw=0.8)


def _scale_value(scale_id: str) -> float:
    return float(str(scale_id).replace("rel_", "").replace("p", ".").replace("x", ""))


def _scale_label(v: float) -> str:
    return f"{v:g}x"


def _build_figure(
    df: pd.DataFrame,
    *,
    latent: str,
    k: int,
    summary: str,
    input_mode: str,
    out_dir: Path,
) -> None:
    sub = df[
        (df["latent"] == latent)
        & (df["k"] == k)
        & (df["motion_summary"] == summary)
        & (df["input_mode"] == input_mode)
    ].copy()
    if sub.empty:
        return
    sub["scale"] = sub["scale_id"].map(_scale_value)
    all_scales = sorted(sub["scale"].unique())
    x = list(range(len(all_scales)))
    xlabels = [_scale_label(v) for v in all_scales]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)

    # Panel 1: absolute cosines by axis group
    ax = axes[0]
    for group in ["along", "across", "all"]:
        block = sub[sub["axis_group"] == group].sort_values("scale")
        if block.empty:
            continue
        bi = [all_scales.index(s) for s in block["scale"]]
        y = block["mlp_condition_cosine"].to_numpy(dtype=float)
        lo = block["mlp_condition_ci_low"].to_numpy(dtype=float)
        hi = block["mlp_condition_ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            [x[i] for i in bi], y, yerr=np.vstack([y - lo, hi - y]),
            marker="o", markersize=4, linewidth=1.6, capsize=0,
            color=AXIS_COLORS[group], label=AXIS_LABELS[group],
        )
    # Static baseline (use 'all' group)
    static_block = sub[(sub["axis_group"] == "all")].sort_values("scale")
    if not static_block.empty:
        bs = [all_scales.index(s) for s in static_block["scale"]]
        ax.plot(
            [x[i] for i in bs],
            static_block["mlp_static_cosine"].to_numpy(dtype=float),
            color=AXIS_COLORS["all"], linewidth=1.2, linestyle="--",
            alpha=0.5, label="MLP(static)",
        )
    ax.set_title(f"MLP condition cosine by axis group")
    ax.set_xlabel("motion scale")
    ax.set_ylabel("feature cosine")
    ax.set_xticks(x, xlabels)
    ax.legend(frameon=False, fontsize=7)
    _clean_axis(ax)

    # Panel 2: gain (condition − static) by axis group
    ax = axes[1]
    ax.axhline(0, color="#222222", lw=0.8)
    for group in ["along", "across", "all"]:
        block = sub[sub["axis_group"] == group].sort_values("scale")
        if block.empty:
            continue
        bi = [all_scales.index(s) for s in block["scale"]]
        y = block["gain_cosine"].to_numpy(dtype=float)
        lo = block["gain_ci_low"].to_numpy(dtype=float)
        hi = block["gain_ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            [x[i] for i in bi], y, yerr=np.vstack([y - lo, hi - y]),
            marker="o", markersize=4, linewidth=1.6, capsize=0,
            color=AXIS_COLORS[group], label=AXIS_LABELS[group],
        )
    ax.set_title("MLP gain (condition − static) by axis group")
    ax.set_xlabel("motion scale")
    ax.set_ylabel("cosine gain over static")
    ax.set_xticks(x, xlabels)
    ax.legend(frameon=False, fontsize=7)
    _clean_axis(ax)

    # Panel 3: along − across contrast
    ax = axes[2]
    ax.axhline(0, color="#222222", lw=0.8)
    along_block = sub[sub["axis_group"] == "along"].sort_values("scale").set_index("scale")
    across_block = sub[sub["axis_group"] == "across"].sort_values("scale").set_index("scale")
    for sv in all_scales:
        if sv not in along_block.index or sv not in across_block.index:
            continue
        diff = along_block.loc[sv, "gain_cosine"] - across_block.loc[sv, "gain_cosine"]
        xi = x[all_scales.index(sv)]
        ax.bar(xi, diff, color=AXIS_COLORS["along"], alpha=0.7, width=0.4)
    ax.set_title("Along gain − Across gain\n(axis specificity of motion benefit)")
    ax.set_xlabel("motion scale")
    ax.set_ylabel("Δ cosine gain (along − across)")
    ax.set_xticks(x, xlabels)
    _clean_axis(ax)

    fig.suptitle(
        f"4B MLP axis stratification: {input_mode}({summary}) | {latent} k={k}",
        fontsize=9,
    )
    slug = f"mlp_axis_strat_{input_mode}_{summary}_{latent}_k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{slug}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{slug}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--window-manifest", type=Path, default=DEFAULT_WINDOW_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--summaries", default="delta_mean,mean")
    parser.add_argument("--input-modes", default="motion_only,augmented")
    parser.add_argument("--families", default="empirical")
    parser.add_argument("--scale-ids", default="all")
    parser.add_argument("--latent-names", default="pyramid_local_field")
    parser.add_argument("--pca-k-list", default="8,16")
    parser.add_argument("--cos2-threshold", type=float, default=0.5,
                        help="cos² threshold for along (≥) vs across (<) split.")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-learning-rate", type=float, default=3e-4)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-patience", type=int, default=30)
    parser.add_argument("--mlp-validation-fraction", type=float, default=0.2)
    parser.add_argument("--mlp-max-train-samples", type=int, default=0)
    parser.add_argument("--mlp-device", default="auto")
    return parser


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else run_dir / "incremental_mlp_axis_stratified"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    analysis_images = pd.read_csv(run_dir / "analysis_images.csv")

    # Load axis alignment
    aligned = _load_axis_alignment(
        analysis_images,
        Path(args.window_manifest),
        cos2_threshold=float(args.cos2_threshold),
    )
    cos2 = aligned["cos2_alignment"].to_numpy(dtype=np.float64)
    axis_group = aligned["axis_group"].to_numpy()
    along_mask = axis_group == "along"
    across_mask = axis_group == "across"
    sessions = aligned["session"].to_numpy()
    decode_groups = aligned["image_index"].to_numpy(dtype=int)

    n_along = int(along_mask.sum())
    n_across = int(across_mask.sum())
    print(f"Axis split: along={n_along}, across={n_across} (cos2_threshold={args.cos2_threshold})", flush=True)

    latent_npz = np.load(run_dir / "latent_feature_arrays.npz")
    response_npz = np.load(run_dir / "response_summary_arrays.npz")

    latent_names = _parse_list(args.latent_names)
    summaries = _parse_list(args.summaries)
    input_modes = _parse_list(args.input_modes)
    families = _parse_list(args.families)
    pca_k_list = _parse_int_list(args.pca_k_list)
    scale_ids = _parse_list(args.scale_ids)
    if not scale_ids or "all" in scale_ids:
        scale_ids = _available_scale_ids(response_npz, families, summaries)

    mlp_config = MLPConfig(
        hidden_dim=int(args.mlp_hidden_dim),
        layers=int(args.mlp_layers),
        dropout=float(args.mlp_dropout),
        learning_rate=float(args.mlp_learning_rate),
        weight_decay=float(args.mlp_weight_decay),
        batch_size=int(args.mlp_batch_size),
        epochs=int(args.mlp_epochs),
        patience=int(args.mlp_patience),
        validation_fraction=float(args.mlp_validation_fraction),
        max_train_samples=int(args.mlp_max_train_samples),
        device=str(args.mlp_device),
        seed=int(args.seed),
    )
    resolved_device = _resolve_torch_device(mlp_config.device)
    print(f"MLP device: {resolved_device}", flush=True)

    decode_cache: dict[tuple, dict[str, np.ndarray]] = {}
    saved_arrays: dict[str, np.ndarray] = {}

    def _cached_decode(
        key: tuple, X: np.ndarray, Z: np.ndarray, slug: str
    ) -> dict[str, np.ndarray]:
        if key not in decode_cache:
            decode_cache[key] = _mlp_cross_validated_decode(
                X, Z, decode_groups,
                k=key[-1],
                outer_folds=int(args.outer_folds),
                seed=int(args.seed),
                mlp_config=mlp_config,
                spec_slug=slug,
            )
        return decode_cache[key]

    gain_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []

    for summary in summaries:
        static_summary = STATIC_SUMMARY_FOR_MOTION[summary]
        static_key_str = _response_key(static_summary, "static", "static")
        X_static = np.asarray(response_npz[static_key_str], dtype=np.float32)

        for latent_name in latent_names:
            Z = np.asarray(latent_npz[latent_name], dtype=np.float64)

            for k in pca_k_list:
                static_cache_key = ("static", summary, latent_name, k)
                print(f"\n  Static: {summary} {latent_name} k={k}", flush=True)
                static_result = _cached_decode(static_cache_key, X_static, Z, "axis_static")
                static_cosines = static_result["cosine"]
                static_neg_mse = static_result["neg_mse"]
                saved_arrays[f"static__{summary}__{latent_name}__k{k}__cosine"] = static_cosines
                saved_arrays[f"static__{summary}__{latent_name}__k{k}__neg_mse"] = static_neg_mse

                for scale_id in scale_ids:
                    for family in families:
                        motion_key_str = _response_key(summary, family, scale_id)
                        if motion_key_str not in response_npz:
                            continue
                        X_motion = np.asarray(response_npz[motion_key_str], dtype=np.float32)

                        for mode in input_modes:
                            X_cond = _build_X(mode, X_static, X_motion)
                            cond_cache_key = (mode, summary, family, scale_id, latent_name, k)
                            print(
                                f"  MLP {mode}: {summary} {family} {scale_id} {latent_name} k={k}",
                                flush=True,
                            )
                            cond_result = _cached_decode(
                                cond_cache_key, X_cond, Z, f"{mode}_{family}",
                            )
                            cond_cosines = cond_result["cosine"]
                            cond_neg_mse = cond_result["neg_mse"]
                            npz_prefix = f"{mode}__{summary}__{family}__{scale_id}__{latent_name}__k{k}"
                            saved_arrays[f"{npz_prefix}__cosine"] = cond_cosines
                            saved_arrays[f"{npz_prefix}__neg_mse"] = cond_neg_mse

                            per_window_gain_cosine = cond_cosines - static_cosines
                            per_window_gain_neg_mse = cond_neg_mse - static_neg_mse

                            # Correlation with cos² for both metrics
                            for metric, gain_arr in [
                                ("cosine", per_window_gain_cosine),
                                ("neg_mse", per_window_gain_neg_mse),
                            ]:
                                corr = _axis_correlation(
                                    gain_arr, cos2, sessions,
                                    rng=rng, n_bootstrap=int(args.n_bootstrap),
                                )
                                corr_rows.append({
                                    "input_mode": mode,
                                    "motion_summary": summary,
                                    "family": family,
                                    "scale_id": scale_id,
                                    "latent": latent_name,
                                    "k": int(k),
                                    "metric": metric,
                                    "r_cos2_gain": corr["r"],
                                    "r_ci_low": corr["ci_low"],
                                    "r_ci_high": corr["ci_high"],
                                    "n_sessions": corr["n_sessions"],
                                })

                            for group, mask in [
                                ("along", along_mask),
                                ("across", across_mask),
                                ("all", np.ones(len(sessions), dtype=bool)),
                            ]:
                                cond_cos_boot = _stratified_bootstrap(
                                    cond_cosines, sessions, mask, rng=rng, n_bootstrap=int(args.n_bootstrap),
                                )
                                static_cos_boot = _stratified_bootstrap(
                                    static_cosines, sessions, mask, rng=rng, n_bootstrap=int(args.n_bootstrap),
                                )
                                gain_cos_boot = _stratified_bootstrap(
                                    per_window_gain_cosine, sessions, mask, rng=rng, n_bootstrap=int(args.n_bootstrap),
                                )
                                cond_mse_boot = _stratified_bootstrap(
                                    cond_neg_mse, sessions, mask, rng=rng, n_bootstrap=int(args.n_bootstrap),
                                )
                                static_mse_boot = _stratified_bootstrap(
                                    static_neg_mse, sessions, mask, rng=rng, n_bootstrap=int(args.n_bootstrap),
                                )
                                gain_mse_boot = _stratified_bootstrap(
                                    per_window_gain_neg_mse, sessions, mask, rng=rng, n_bootstrap=int(args.n_bootstrap),
                                )
                                gain_rows.append({
                                    "input_mode": mode,
                                    "motion_summary": summary,
                                    "family": family,
                                    "scale_id": scale_id,
                                    "latent": latent_name,
                                    "k": int(k),
                                    "axis_group": group,
                                    "n_images": gain_cos_boot["n"],
                                    "n_sessions": gain_cos_boot["n_sessions"],
                                    # cosine columns
                                    "mlp_static_cosine": static_cos_boot["mean"],
                                    "mlp_static_cosine_ci_low": static_cos_boot["ci_low"],
                                    "mlp_static_cosine_ci_high": static_cos_boot["ci_high"],
                                    "mlp_condition_cosine": cond_cos_boot["mean"],
                                    "mlp_condition_cosine_ci_low": cond_cos_boot["ci_low"],
                                    "mlp_condition_cosine_ci_high": cond_cos_boot["ci_high"],
                                    "gain_cosine": gain_cos_boot["mean"],
                                    "gain_cosine_ci_low": gain_cos_boot["ci_low"],
                                    "gain_cosine_ci_high": gain_cos_boot["ci_high"],
                                    # neg_mse columns (matches 4D metric)
                                    "mlp_static_neg_mse": static_mse_boot["mean"],
                                    "mlp_static_neg_mse_ci_low": static_mse_boot["ci_low"],
                                    "mlp_static_neg_mse_ci_high": static_mse_boot["ci_high"],
                                    "mlp_condition_neg_mse": cond_mse_boot["mean"],
                                    "mlp_condition_neg_mse_ci_low": cond_mse_boot["ci_low"],
                                    "mlp_condition_neg_mse_ci_high": cond_mse_boot["ci_high"],
                                    "gain_neg_mse": gain_mse_boot["mean"],
                                    "gain_neg_mse_ci_low": gain_mse_boot["ci_low"],
                                    "gain_neg_mse_ci_high": gain_mse_boot["ci_high"],
                                    # legacy aliases
                                    "mlp_static_ci_low": static_cos_boot["ci_low"],
                                    "mlp_static_ci_high": static_cos_boot["ci_high"],
                                    "mlp_condition_ci_low": cond_cos_boot["ci_low"],
                                    "mlp_condition_ci_high": cond_cos_boot["ci_high"],
                                    "gain_ci_low": gain_cos_boot["ci_low"],
                                    "gain_ci_high": gain_cos_boot["ci_high"],
                                })

    _write_csv(out_dir / "mlp_axis_stratified_gain.csv", gain_rows)
    _write_csv(out_dir / "mlp_axis_correlation.csv", corr_rows)
    np.savez_compressed(out_dir / "per_window_arrays.npz", **saved_arrays)
    _write_json(out_dir / "axis_alignment_summary.json", {
        "cos2_threshold": float(args.cos2_threshold),
        "n_along": int(n_along),
        "n_across": int(n_across),
        "n_total": int(len(sessions)),
        "cos2_mean": float(np.nanmean(cos2)),
        "cos2_std": float(np.nanstd(cos2)),
    })
    _write_json(out_dir / "run_metadata.json", {
        "run_dir": run_dir,
        "summaries": summaries,
        "input_modes": input_modes,
        "families": families,
        "scale_ids": scale_ids,
        "latent_names": latent_names,
        "pca_k_list": pca_k_list,
        "cos2_threshold": float(args.cos2_threshold),
        "mlp": mlp_config.__dict__,
    })

    df = pd.DataFrame(gain_rows)
    for latent_name in latent_names:
        for k in pca_k_list:
            for summary in summaries:
                for mode in input_modes:
                    _build_figure(
                        df, latent=latent_name, k=k,
                        summary=summary, input_mode=mode, out_dir=out_dir,
                    )

    print(f"\nWrote axis-stratified MLP results to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
