"""MLP nonlinear feature decoder adjudication grid for Figure 4B.

Runs a cross-validated MLP for each combination of input representation,
motion summary, family, scale, latent, and k — producing a tidy table
with one row per (input_mode, family, scale_id, latent, k).

Input modes
-----------
motion_only  : MLP(X_motion)           — is the motion signal a good image code?
augmented    : MLP([X_static, X_motion]) — does motion add information beyond static?

Both are compared against the shared baseline MLP(X_static).

The 'augmented' mode is the nonlinear analogue of the incremental
static-plus-motion question that the linear ridge decoder already tested.
'motion_only' is a diagnostic asking whether the motion signal can stand alone.

X_static = mean response of the digital twin to counterfactual stabilised stimuli.
X_motion = motion_summary (e.g. delta_mean or mean) of the response given a
           non-zero trajectory. delta_mean is the differential (motion-induced
           change); mean is the total time-averaged response during motion.

Usage::

    python -m declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion_mlp \\
        --run-dir outputs/.../backimage_aggregate_fem_information_n384_... \\
        --summaries delta_mean,mean \\
        --input-modes motion_only,augmented \\
        --families empirical,brownian,rotated \\
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

try:
    from sklearn.decomposition import PCA
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for this diagnostic") from exc

try:
    from .summarize_backimage_aggregate_incremental_motion import (
        STATIC_SUMMARY_FOR_MOTION,
        _available_scale_ids,
        _response_key,
        _session_bootstrap_delta,
    )
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion import (
        STATIC_SUMMARY_FOR_MOTION,
        _available_scale_ids,
        _response_key,
        _session_bootstrap_delta,
    )

try:
    from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_feature_embedding_reconstruction import (
        MLPConfig,
        _assign_source_folds,
        _fit_predict_mlp,
        _resolve_torch_device,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("build_panel_c_continuous_feature_embedding_reconstruction is required") from exc


DEFAULT_RUN_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1"
)
DEFAULT_INPUT_MODES = ("motion_only", "augmented")

COLORS = {
    "empirical": "#244f7a",
    "ou": "#d07a22",
    "brownian": "#707070",
    "rotated": "#8064a2",
    "static": "#aab0b6",
}
FAMILY_LABELS = {
    "empirical": "recorded drift",
    "ou": "OU control",
    "brownian": "random drift",
    "rotated": "rotated drift",
}
MODE_LABELS = {
    "motion_only": "MLP(motion only)",
    "augmented": "MLP(static + motion)",
}
MODE_LINESTYLE = {
    "motion_only": "-",
    "augmented": "--",
}


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def _build_X(mode: str, X_static: np.ndarray, X_motion: np.ndarray) -> np.ndarray:
    if mode == "motion_only":
        return np.asarray(X_motion, dtype=np.float32)
    if mode == "augmented":
        return np.concatenate(
            [np.asarray(X_static, dtype=np.float32), np.asarray(X_motion, dtype=np.float32)],
            axis=1,
        )
    raise ValueError(f"Unknown input mode {mode!r}; valid: motion_only, augmented")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in _parse_list(text)]


def _parse_contrast_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in _parse_list(text):
        for sep in (":", ">", "-"):
            if sep in part:
                lhs, rhs = part.split(sep, 1)
                pairs.append((lhs.strip(), rhs.strip()))
                break
        else:
            raise ValueError(f"Contrast pair must use lhs:rhs syntax; got {part!r}")
    return pairs


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

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


def _filter_latents(latents: dict[str, np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
    if not names or "all" in names:
        return dict(latents)
    missing = sorted(set(names).difference(latents))
    if missing:
        raise ValueError(f"Requested latent arrays are missing: {missing}")
    return {name: latents[name] for name in names}


# ---------------------------------------------------------------------------
# MLP decoding
# ---------------------------------------------------------------------------

def _pca_transform_fold(
    Z: np.ndarray,
    train_idx: np.ndarray,
    k: int,
) -> tuple[np.ndarray, int]:
    """Fit PCA on train rows only; transform all rows. Returns (Z_all_k, k_eff)."""
    Z = np.asarray(Z, dtype=np.float64)
    k_eff = min(int(k), Z.shape[1], max(1, len(train_idx) - 1))
    pca = PCA(n_components=k_eff, svd_solver="full")
    Z_mean = np.mean(Z[train_idx], axis=0, keepdims=True)
    pca.fit(Z[train_idx] - Z_mean)
    Z_all_k = pca.transform(Z - Z_mean)
    return Z_all_k, k_eff


def _split_by_groups(
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups)
    fold_by_group = _assign_source_folds(groups, n_folds=n_folds, seed=seed)
    n_folds_actual = max(fold_by_group.values()) + 1
    splits = []
    for fold in range(n_folds_actual):
        test_mask = np.asarray([fold_by_group.get(int(g), 0) == fold for g in groups])
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        if train_idx.size > 0 and test_idx.size > 0:
            splits.append((train_idx, test_idx))
    return splits


def _per_window_cosine(z_hat: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    pred = np.asarray(z_hat, dtype=np.float64)
    true = np.asarray(z_true, dtype=np.float64)
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    dots = np.einsum("ij,ij->i", pred, true)
    return np.where(denom > 1e-12, dots / denom, np.nan).astype(np.float64)


def _per_window_neg_mse(z_hat: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    """Per-window negative mean squared error (higher = better, matches 4D metric)."""
    pred = np.asarray(z_hat, dtype=np.float64)
    true = np.asarray(z_true, dtype=np.float64)
    return -np.mean((pred - true) ** 2, axis=1).astype(np.float64)


def _mlp_cross_validated_decode(
    X: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    *,
    k: int,
    outer_folds: int,
    seed: int,
    mlp_config: MLPConfig,
    spec_slug: str = "decode",
) -> dict[str, np.ndarray]:
    """Returns dict with per-window 'cosine' and 'neg_mse' arrays (nan for uncovered windows)."""
    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float64)
    groups = np.asarray(groups)
    n = X.shape[0]
    splits = _split_by_groups(groups, n_folds=int(outer_folds), seed=int(seed))
    per_window_cosine = np.full(n, np.nan, dtype=np.float64)
    per_window_neg_mse = np.full(n, np.nan, dtype=np.float64)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        Z_all_k, k_eff = _pca_transform_fold(Z, train_idx, k)
        Z_all_k = Z_all_k.astype(np.float32, copy=False)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_idx] = True
        z_hat, _ = _fit_predict_mlp(
            x_all=X,
            z_all=Z_all_k,
            source_rows=groups,
            train_mask=train_mask,
            x_test=X[test_idx],
            config=mlp_config,
            fold=int(fold_idx),
            spec_slug=spec_slug,
            feature_space_mode=f"pca_k{k_eff}",
        )
        per_window_cosine[test_idx] = _per_window_cosine(z_hat, Z_all_k[test_idx])
        per_window_neg_mse[test_idx] = _per_window_neg_mse(z_hat, Z_all_k[test_idx])

    return {"cosine": per_window_cosine, "neg_mse": per_window_neg_mse}


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
    families: list[str],
    input_modes: list[str],
    out_dir: Path,
) -> None:
    sub = df[
        (df["latent"] == latent) & (df["k"] == k) & (df["motion_summary"] == summary)
    ].copy()
    if sub.empty:
        return
    sub["scale"] = sub["scale_id"].map(_scale_value)
    all_scales = sorted(sub["scale"].unique())

    n_modes = len(input_modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(4.2 * n_modes, 3.2), constrained_layout=True)
    if n_modes == 1:
        axes = [axes]

    for ax, mode in zip(axes, input_modes):
        mode_sub = sub[sub["input_mode"] == mode]
        ax.axhline(0.0, color="#222222", lw=0.8)
        for family in families:
            block = mode_sub[mode_sub["family"] == family].sort_values("scale")
            if block.empty:
                continue
            x = block["scale"].to_numpy(dtype=float)
            y = block["mlp_gain_cosine"].to_numpy(dtype=float)
            lo = block["mlp_gain_ci_low"].to_numpy(dtype=float)
            hi = block["mlp_gain_ci_high"].to_numpy(dtype=float)
            ax.errorbar(
                x, y,
                yerr=np.vstack([y - lo, hi - y]),
                marker="o", markersize=3.8, linewidth=1.8, capsize=0,
                color=COLORS.get(family, "#555555"),
                label=FAMILY_LABELS.get(family, family),
            )
        ax.set_title(f"{MODE_LABELS.get(mode, mode)} − MLP(static)")
        ax.set_xlabel("motion scale")
        ax.set_ylabel("cosine gain over MLP(static)")
        ax.set_xticks(all_scales, [_scale_label(v) for v in all_scales])
        ax.legend(frameon=False, fontsize=7.0)
        _clean_axis(ax)

    fig.suptitle(
        f"4B MLP adjudication: {summary} | {latent} k={k}",
        fontsize=9.0,
    )
    slug = f"mlp_adjudication_{latent}_k{k}_{summary}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{slug}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{slug}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / run
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--summaries", default="delta_mean",
                        help="Comma-separated motion summary keys.")
    parser.add_argument("--input-modes", default=",".join(DEFAULT_INPUT_MODES),
                        help="motion_only and/or augmented.")
    parser.add_argument("--families", default="empirical,brownian,rotated")
    parser.add_argument("--contrast-pairs", default="empirical:brownian,empirical:rotated")
    parser.add_argument("--scale-ids", default="all")
    parser.add_argument("--latent-names", default="pyramid_local_field")
    parser.add_argument("--pca-k-list", default="8,16")
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
        else run_dir / "incremental_staticmean_plus_motion_mlp"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    images = pd.read_csv(run_dir / "analysis_images.csv")
    sessions = images["session"].to_numpy()
    decode_groups = images["image_index"].to_numpy(dtype=int)

    latents = _filter_latents(
        _load_npz(run_dir / "latent_feature_arrays.npz"),
        _parse_list(args.latent_names),
    )
    responses = _load_npz(run_dir / "response_summary_arrays.npz")
    summaries = _parse_list(args.summaries)
    input_modes = _parse_list(args.input_modes)
    unknown = sorted(set(input_modes) - {"motion_only", "augmented"})
    if unknown:
        raise ValueError(f"Unknown input modes: {unknown}")
    families = _parse_list(args.families)
    contrast_pairs = _parse_contrast_pairs(args.contrast_pairs)
    scale_ids = _parse_list(args.scale_ids)
    if not scale_ids or "all" in scale_ids:
        scale_ids = _available_scale_ids(responses, families, summaries)
    pca_k_list = _parse_int_list(args.pca_k_list)

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
    print(f"Input modes: {input_modes}", flush=True)

    # Cache decoded metrics per (X_key → {"cosine": arr, "neg_mse": arr})
    decode_cache: dict[tuple, dict[str, np.ndarray]] = {}

    def _cached_decode(
        cache_key: tuple, X: np.ndarray, Z: np.ndarray, slug: str
    ) -> dict[str, np.ndarray]:
        if cache_key not in decode_cache:
            decode_cache[cache_key] = _mlp_cross_validated_decode(
                X, Z, decode_groups,
                k=cache_key[-1],  # k is last element of key
                outer_folds=int(args.outer_folds),
                seed=int(args.seed),
                mlp_config=mlp_config,
                spec_slug=slug,
            )
        return decode_cache[cache_key]

    gain_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []

    for summary in summaries:
        static_summary = STATIC_SUMMARY_FOR_MOTION.get(summary)
        if static_summary is None:
            raise ValueError(f"No static summary mapping for {summary!r}")
        static_key = _response_key(static_summary, "static", "static")
        if static_key not in responses:
            raise ValueError(f"Missing static response array {static_key!r}")
        X_static = np.asarray(responses[static_key], dtype=np.float32)

        for latent_name, Z_raw in latents.items():
            Z = np.asarray(Z_raw, dtype=np.float64)

            for k in pca_k_list:
                # Baseline: MLP(X_static) — shared across all families and modes
                static_cache_key = ("static_only", summary, latent_name, k)
                print(f"  MLP static: {summary} {latent_name} k={k}", flush=True)
                static_metrics = _cached_decode(static_cache_key, X_static, Z, "static")
                static_cosines = static_metrics["cosine"]
                static_neg_mse = static_metrics["neg_mse"]
                static_cosine_boot = _session_bootstrap_delta(
                    static_cosines, np.zeros_like(static_cosines),
                    sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                )
                static_neg_mse_boot = _session_bootstrap_delta(
                    static_neg_mse, np.zeros_like(static_neg_mse),
                    sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                )

                for scale_id in scale_ids:
                    cosines_by_family_mode: dict[tuple[str, str], np.ndarray] = {}
                    neg_mse_by_family_mode: dict[tuple[str, str], np.ndarray] = {}

                    for family in families:
                        motion_key_str = _response_key(summary, family, scale_id)
                        if motion_key_str not in responses:
                            continue
                        X_motion = np.asarray(responses[motion_key_str], dtype=np.float32)

                        for mode in input_modes:
                            X_cond = _build_X(mode, X_static, X_motion)
                            cond_cache_key = (mode, summary, family, scale_id, latent_name, k)
                            print(
                                f"  MLP {mode}: {summary} {family} {scale_id} {latent_name} k={k}",
                                flush=True,
                            )
                            cond_metrics = _cached_decode(
                                cond_cache_key, X_cond, Z, f"{mode}_{family}",
                            )
                            cond_cosines = cond_metrics["cosine"]
                            cond_neg_mse = cond_metrics["neg_mse"]
                            cosines_by_family_mode[(family, mode)] = cond_cosines
                            neg_mse_by_family_mode[(family, mode)] = cond_neg_mse

                            cond_cosine_boot = _session_bootstrap_delta(
                                cond_cosines, np.zeros_like(cond_cosines),
                                sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                            )
                            gain_cosine_boot = _session_bootstrap_delta(
                                cond_cosines, static_cosines,
                                sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                            )
                            cond_neg_mse_boot = _session_bootstrap_delta(
                                cond_neg_mse, np.zeros_like(cond_neg_mse),
                                sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                            )
                            gain_neg_mse_boot = _session_bootstrap_delta(
                                cond_neg_mse, static_neg_mse,
                                sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                            )
                            gain_rows.append({
                                "motion_summary": summary,
                                "input_mode": mode,
                                "family": family,
                                "scale_id": scale_id,
                                "latent": latent_name,
                                "k": int(k),
                                # cosine columns
                                "mlp_static_cosine": static_cosine_boot["mean"],
                                "mlp_static_cosine_ci_low": static_cosine_boot["ci_low"],
                                "mlp_static_cosine_ci_high": static_cosine_boot["ci_high"],
                                "mlp_condition_cosine": cond_cosine_boot["mean"],
                                "mlp_condition_cosine_ci_low": cond_cosine_boot["ci_low"],
                                "mlp_condition_cosine_ci_high": cond_cosine_boot["ci_high"],
                                "mlp_gain_cosine": gain_cosine_boot["mean"],
                                "mlp_gain_cosine_ci_low": gain_cosine_boot["ci_low"],
                                "mlp_gain_cosine_ci_high": gain_cosine_boot["ci_high"],
                                # neg_mse columns (matches 4D metric)
                                "mlp_static_neg_mse": static_neg_mse_boot["mean"],
                                "mlp_static_neg_mse_ci_low": static_neg_mse_boot["ci_low"],
                                "mlp_static_neg_mse_ci_high": static_neg_mse_boot["ci_high"],
                                "mlp_condition_neg_mse": cond_neg_mse_boot["mean"],
                                "mlp_condition_neg_mse_ci_low": cond_neg_mse_boot["ci_low"],
                                "mlp_condition_neg_mse_ci_high": cond_neg_mse_boot["ci_high"],
                                "mlp_gain_neg_mse": gain_neg_mse_boot["mean"],
                                "mlp_gain_neg_mse_ci_low": gain_neg_mse_boot["ci_low"],
                                "mlp_gain_neg_mse_ci_high": gain_neg_mse_boot["ci_high"],
                                # legacy aliases so old analysis code reading ci_low/ci_high still works
                                "mlp_static_ci_low": static_cosine_boot["ci_low"],
                                "mlp_static_ci_high": static_cosine_boot["ci_high"],
                                "mlp_condition_ci_low": cond_cosine_boot["ci_low"],
                                "mlp_condition_ci_high": cond_cosine_boot["ci_high"],
                                "mlp_gain_ci_low": gain_cosine_boot["ci_low"],
                                "mlp_gain_ci_high": gain_cosine_boot["ci_high"],
                                "n_images": gain_cosine_boot["n"],
                                "n_sessions": gain_cosine_boot["n_sessions"],
                                "mlp_hidden_dim": int(args.mlp_hidden_dim),
                                "mlp_layers": int(args.mlp_layers),
                                "mlp_device": resolved_device,
                            })

                    # Contrasts within each mode
                    for mode in input_modes:
                        for lhs, rhs in contrast_pairs:
                            lk, rk = (lhs, mode), (rhs, mode)
                            if lk not in cosines_by_family_mode or rk not in cosines_by_family_mode:
                                continue
                            contrast_cosine_boot = _session_bootstrap_delta(
                                cosines_by_family_mode[lk], cosines_by_family_mode[rk],
                                sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                            )
                            contrast_neg_mse_boot = _session_bootstrap_delta(
                                neg_mse_by_family_mode[lk], neg_mse_by_family_mode[rk],
                                sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                            )
                            contrast_rows.append({
                                "motion_summary": summary,
                                "input_mode": mode,
                                "lhs_family": lhs,
                                "rhs_family": rhs,
                                "scale_id": scale_id,
                                "latent": latent_name,
                                "k": int(k),
                                "mlp_gain_delta_cosine": contrast_cosine_boot["mean"],
                                "mlp_gain_delta_cosine_ci_low": contrast_cosine_boot["ci_low"],
                                "mlp_gain_delta_cosine_ci_high": contrast_cosine_boot["ci_high"],
                                "mlp_gain_delta_neg_mse": contrast_neg_mse_boot["mean"],
                                "mlp_gain_delta_neg_mse_ci_low": contrast_neg_mse_boot["ci_low"],
                                "mlp_gain_delta_neg_mse_ci_high": contrast_neg_mse_boot["ci_high"],
                                # legacy alias
                                "mlp_gain_delta_ci_low": contrast_cosine_boot["ci_low"],
                                "mlp_gain_delta_ci_high": contrast_cosine_boot["ci_high"],
                                "n_images": contrast_cosine_boot["n"],
                                "n_sessions": contrast_cosine_boot["n_sessions"],
                            })

    _write_csv(out_dir / "mlp_gain_vs_static.csv", gain_rows)
    _write_csv(out_dir / "mlp_gain_contrasts.csv", contrast_rows)
    _write_json(out_dir / "run_metadata.json", {
        "source_run_dir": run_dir,
        "summaries": summaries,
        "input_modes": input_modes,
        "families": families,
        "contrast_pairs": contrast_pairs,
        "scale_ids": scale_ids,
        "latent_names": list(latents),
        "pca_k_list": pca_k_list,
        "outer_folds": int(args.outer_folds),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "mlp": mlp_config.__dict__,
        "input_mode_definitions": {
            "static_only": "MLP(X_static) — baseline, mean response to stabilised stimuli",
            "motion_only": "MLP(X_motion) — motion response alone, no static backbone",
            "augmented": "MLP([X_static, X_motion]) — nonlinear analogue of incremental linear test",
        },
    })

    df = pd.DataFrame(gain_rows)
    for latent_name in latents:
        for k in pca_k_list:
            for summary in summaries:
                _build_figure(
                    df, latent=latent_name, k=k, summary=summary,
                    families=families, input_modes=input_modes, out_dir=out_dir,
                )

    report_lines = [
        "# MLP Adjudication Grid — Nonlinear Feature Decoder",
        "",
        f"Source run: `{run_dir}`",
        "",
        "Input modes (all compared against MLP(X_static) baseline):",
        "  motion_only : MLP(X_motion)            — is the motion signal a standalone image code?",
        "  augmented   : MLP([X_static, X_motion]) — nonlinear analogue of the incremental ridge test",
        "",
        "X_static = mean response to counterfactual stabilised stimuli.",
        "X_motion = motion_summary of response given non-zero trajectory.",
        "          delta_mean is the differential; mean is the total time-averaged response.",
        "",
        "Primary files:",
        "  mlp_gain_vs_static.csv  — tidy table, one row per (input_mode, family, scale_id, latent, k)",
        "  mlp_gain_contrasts.csv  — empirical minus control within each mode",
    ]
    (out_dir / "summary_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote MLP adjudication grid to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
