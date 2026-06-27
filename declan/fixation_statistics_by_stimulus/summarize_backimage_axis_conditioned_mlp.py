"""MLP nonlinear feature decoder for Figure 4D axis-conditioned comparison.

Applies the same cross-validated MLP decoder (as in the 4B incremental analysis)
to test whether along-edge vs across-edge trajectory conditions produce responses
that better encode image features, using the nonlinear decoder.

For each (scale, latent, k):
    X_static : mean digital-twin response under zero/static trajectory
    X_along  : mean digital-twin prediction averaged over along-edge prior trajectories
    X_across : mean digital-twin prediction averaged over across-edge prior trajectories

Comparisons:
    MLP(X_along) − MLP(X_static)   — does along-edge motion help?
    MLP(X_across) − MLP(X_static)  — does across-edge motion help?
    MLP(X_along) − MLP(X_across)   — does axis alignment matter?

Input data
----------
Axis-conditioned percandidate run (n=128 images):
    response_tables/trial_*_{parallel,orthogonal}_rel*.npz
    Each NPZ contains:
        prior_lambda_counts : (n_cand, 16, 40, 756) — model predictions for prior trajectories
        zero_lambda_counts  : (n_cand, 40, 756)     — model predictions under static/zero traj
        y_obs_counts        : (40, 756)              — actual observation response
        true_candidate_index: (1,)                   — index of the true image in the candidate set

Latent feature arrays from the matched bconsistent feature-posterior run:
    feature_latent_arrays.npz  — pyramid_local_field, source_row

Usage::

    python -m declan.fixation_statistics_by_stimulus.summarize_backimage_axis_conditioned_mlp \\
        --run-dir outputs/.../backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1 \\
        --latent-run-dir outputs/.../backimage_axis_conditioned_matched_static_feature_posterior_pyramid_k8_16_n128_scales_0p5_1_2_bconsistent_v1 \\
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
    raise ImportError("scikit-learn is required") from exc

try:
    from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_feature_embedding_reconstruction import (
        MLPConfig,
        _assign_source_folds,
        _fit_predict_mlp,
        _resolve_torch_device,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("build_panel_c_continuous_feature_embedding_reconstruction is required") from exc

try:
    from .summarize_backimage_aggregate_incremental_motion import _session_bootstrap_delta
except ImportError:  # pragma: no cover
    from declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion import (
        _session_bootstrap_delta,
    )


DEFAULT_PCAN_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1"
)
DEFAULT_LATENT_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_matched_static_feature_posterior_pyramid_k8_16_n128_scales_0p5_1_2_bconsistent_v1"
)

AXIS_COLORS = {
    "along": "#2e7d32",   # green (same as 4D along-edge)
    "across": "#6a1b9a",  # purple (same as 4D across-edge)
    "static": "#aab0b6",
}
SCALE_LABELS = {0.5: "0.5x", 1.0: "1x", 2.0: "2x"}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _extract_response_arrays(
    run_dir: Path,
    observer_trials: pd.DataFrame,
) -> dict[tuple[int, float, str], np.ndarray]:
    """Load response tables and return mean response vectors.

    Returns dict keyed by (source_row, scale, condition) where condition is
    'along', 'across', or 'static'. Each value is shape (756,).
    """
    AXIS_MAP = {
        "axis_edge_parallel": "along",
        "axis_edge_orthogonal": "across",
    }
    arrays: dict[tuple[int, float, str], np.ndarray] = {}

    n_total = len(observer_trials)
    for i, (_, trial) in enumerate(observer_trials.iterrows()):
        src = int(trial["observation_source_row"])
        scale = float(trial["observation_scale"])
        axis_family = trial["prior_family"]
        condition = AXIS_MAP.get(axis_family)
        if condition is None:
            continue

        cache_path = run_dir / trial["response_cache_path"]
        d = np.load(cache_path)
        true_ci = int(d["true_candidate_index"][0])

        # Mean over 16 prior trajectories × 40 timebins
        X_axis = d["prior_lambda_counts"][true_ci].mean(axis=(0, 1))  # (756,)
        arrays[(src, scale, condition)] = X_axis.astype(np.float32)

        # Static (same for both axis conditions; overwrite is fine)
        X_static = d["zero_lambda_counts"][true_ci].mean(axis=0)  # (756,)
        arrays[(src, scale, "static")] = X_static.astype(np.float32)

        if i % 50 == 0:
            print(f"  Loading response tables: {i}/{n_total}", flush=True)

    print(f"  Loaded {n_total} response tables.", flush=True)
    return arrays


def _build_arrays_for_scale(
    response_arrays: dict[tuple[int, float, str], np.ndarray],
    source_rows: list[int],
    scale: float,
) -> dict[str, np.ndarray]:
    """Assemble (n_images, 756) arrays for each condition at a given scale."""
    out: dict[str, np.ndarray] = {}
    for condition in ("along", "across", "static"):
        rows = []
        for src in source_rows:
            key = (src, scale, condition)
            if key not in response_arrays:
                raise KeyError(f"Missing response array for {key}")
            rows.append(response_arrays[key])
        out[condition] = np.stack(rows, axis=0)
    return out


# ---------------------------------------------------------------------------
# MLP decoding (shared with 4B script)
# ---------------------------------------------------------------------------

def _pca_transform_fold(Z: np.ndarray, train_idx: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    Z = np.asarray(Z, dtype=np.float64)
    k_eff = min(int(k), Z.shape[1], max(1, len(train_idx) - 1))
    pca = PCA(n_components=k_eff, svd_solver="full")
    Z_mean = np.mean(Z[train_idx], axis=0, keepdims=True)
    pca.fit(Z[train_idx] - Z_mean)
    return pca.transform(Z - Z_mean), k_eff


def _split_by_groups(groups: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups)
    fold_by_group = _assign_source_folds(groups, n_folds=n_folds, seed=seed)
    n_actual = max(fold_by_group.values()) + 1
    splits = []
    for fold in range(n_actual):
        test_mask = np.asarray([fold_by_group.get(int(g), 0) == fold for g in groups])
        tr, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        if tr.size > 0 and te.size > 0:
            splits.append((tr, te))
    return splits


def _per_window_cosine(z_hat: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    pred, true = np.asarray(z_hat, dtype=np.float64), np.asarray(z_true, dtype=np.float64)
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    dots = np.einsum("ij,ij->i", pred, true)
    return np.where(denom > 1e-12, dots / denom, np.nan).astype(np.float64)


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
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float64)
    groups = np.asarray(groups)
    n = X.shape[0]
    per_window = np.full(n, np.nan, dtype=np.float64)
    for fold_idx, (train_idx, test_idx) in enumerate(
        _split_by_groups(groups, n_folds=outer_folds, seed=seed)
    ):
        Z_all_k, k_eff = _pca_transform_fold(Z, train_idx, k)
        Z_all_k = Z_all_k.astype(np.float32, copy=False)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_idx] = True
        z_hat, _ = _fit_predict_mlp(
            x_all=X, z_all=Z_all_k, source_rows=groups,
            train_mask=train_mask, x_test=X[test_idx],
            config=mlp_config, fold=int(fold_idx),
            spec_slug=spec_slug, feature_space_mode=f"pca_k{k_eff}",
        )
        per_window[test_idx] = _per_window_cosine(z_hat, Z_all_k[test_idx])
    return per_window


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
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


def _parse_list(text: str) -> list[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(p) for p in _parse_list(text)]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8dde3", lw=0.8)


def _build_figure(df: pd.DataFrame, *, latent: str, k: int, scales: list[float], out_dir: Path) -> None:
    sub = df[(df["latent"] == latent) & (df["k"] == k)].copy()
    if sub.empty:
        return

    comparisons = [
        ("along_vs_static", "MLP(X_along) − MLP(X_static)", AXIS_COLORS["along"]),
        ("across_vs_static", "MLP(X_across) − MLP(X_static)", AXIS_COLORS["across"]),
        ("along_vs_across", "MLP(X_along) − MLP(X_across)", "#1565c0"),
    ]

    fig, axes = plt.subplots(1, len(comparisons), figsize=(4.2 * len(comparisons), 3.2), constrained_layout=True)
    x = list(range(len(scales)))
    xlabels = [SCALE_LABELS.get(s, f"{s}x") for s in scales]

    for ax, (comp_key, comp_label, color) in zip(axes, comparisons):
        block = sub[sub["comparison"] == comp_key].copy()
        if block.empty:
            continue
        block = block.set_index("scale")
        y = [block.loc[s, "gain_cosine"] if s in block.index else np.nan for s in scales]
        lo = [block.loc[s, "gain_ci_low"] if s in block.index else np.nan for s in scales]
        hi = [block.loc[s, "gain_ci_high"] if s in block.index else np.nan for s in scales]
        y, lo, hi = np.array(y), np.array(lo), np.array(hi)
        ax.axhline(0.0, color="#222222", lw=0.8)
        ax.errorbar(
            x, y, yerr=np.vstack([y - lo, hi - y]),
            marker="o", markersize=4.5, linewidth=1.8, capsize=0, color=color,
        )
        ax.set_title(comp_label, fontsize=8)
        ax.set_xlabel("trajectory scale")
        ax.set_ylabel("cosine gain")
        ax.set_xticks(x, xlabels)
        _clean_axis(ax)

    fig.suptitle(f"4D MLP decoder: along vs across edge | {latent} k={k}", fontsize=9)
    slug = f"mlp_axis_gain_{latent}_k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{slug}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{slug}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_PCAN_DIR,
                        help="Percandidate axis-conditioned run directory.")
    parser.add_argument("--latent-run-dir", type=Path, default=DEFAULT_LATENT_DIR,
                        help="Run directory that has feature_latent_arrays.npz for the same images.")
    parser.add_argument("--out-dir", type=Path, default=None)
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
    latent_run_dir = Path(args.latent_run_dir)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else run_dir / "axis_conditioned_mlp"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    pca_k_list = _parse_int_list(args.pca_k_list)
    latent_names = _parse_list(args.latent_names)

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

    # Load observer trials to find response table paths
    observer_trials = pd.read_csv(run_dir / "observer_trials.csv")
    scales = sorted(observer_trials["observation_scale"].unique().tolist())
    source_rows = sorted(observer_trials["observation_source_row"].unique().tolist())
    print(f"Images: {len(source_rows)}, scales: {scales}", flush=True)

    # Infer sessions from the motion_catalog (for bootstrap)
    motion_cat = pd.read_csv(run_dir / "motion_catalog.csv")
    obs_rows = motion_cat[motion_cat["role"] == "observation"]
    src_to_session = (
        obs_rows.drop_duplicates("source_row")
        .set_index("source_row")["source_session"]
        .to_dict()
    )
    sessions = np.array([src_to_session.get(sr, "unknown") for sr in source_rows])

    # Load latent features Z
    latent_npz = np.load(latent_run_dir / "feature_latent_arrays.npz")
    latent_source_rows = latent_npz["source_row"].tolist()
    row_to_latent_idx = {int(r): i for i, r in enumerate(latent_source_rows)}
    latent_indices = [row_to_latent_idx[sr] for sr in source_rows]

    latents: dict[str, np.ndarray] = {}
    for name in latent_names:
        if name not in latent_npz.files:
            raise ValueError(f"Latent {name!r} not in {latent_run_dir / 'feature_latent_arrays.npz'}")
        latents[name] = np.asarray(latent_npz[name])[latent_indices]

    # Extract response arrays from all response tables
    print("Extracting response arrays from response tables...", flush=True)
    response_arrays = _extract_response_arrays(run_dir, observer_trials)

    decode_groups = np.array(source_rows, dtype=int)
    decode_cache: dict[tuple, np.ndarray] = {}

    def _cached_decode(key: tuple, X: np.ndarray, Z: np.ndarray, slug: str) -> np.ndarray:
        if key not in decode_cache:
            decode_cache[key] = _mlp_cross_validated_decode(
                X, Z, decode_groups,
                k=key[-1], outer_folds=int(args.outer_folds),
                seed=int(args.seed), mlp_config=mlp_config, spec_slug=slug,
            )
        return decode_cache[key]

    gain_rows: list[dict[str, Any]] = []

    for scale in scales:
        cond_arrays = _build_arrays_for_scale(response_arrays, source_rows, scale)
        scale_label = SCALE_LABELS.get(scale, f"{scale}x")

        for latent_name, Z in latents.items():
            Z = np.asarray(Z, dtype=np.float64)

            for k in pca_k_list:
                print(f"\n  Scale {scale_label} | {latent_name} k={k}", flush=True)

                # Static baseline
                static_key = ("static", scale, latent_name, k)
                print(f"    MLP(static)...", flush=True)
                static_cosines = _cached_decode(static_key, cond_arrays["static"], Z, "axis_static")
                static_boot = _session_bootstrap_delta(
                    static_cosines, np.zeros_like(static_cosines),
                    sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                )

                per_cond: dict[str, np.ndarray] = {"static": static_cosines}
                for condition in ("along", "across"):
                    cond_key = (condition, scale, latent_name, k)
                    print(f"    MLP({condition})...", flush=True)
                    cosines = _cached_decode(cond_key, cond_arrays[condition], Z, f"axis_{condition}")
                    per_cond[condition] = cosines

                # Compute all comparisons
                comparisons_spec = [
                    ("along_vs_static", "along", "static"),
                    ("across_vs_static", "across", "static"),
                    ("along_vs_across", "along", "across"),
                ]
                for comp_key, lhs, rhs in comparisons_spec:
                    boot = _session_bootstrap_delta(
                        per_cond[lhs], per_cond[rhs],
                        sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                    )
                    cond_boot = _session_bootstrap_delta(
                        per_cond[lhs], np.zeros_like(per_cond[lhs]),
                        sessions, rng=rng, n_bootstrap=int(args.n_bootstrap),
                    )
                    gain_rows.append({
                        "scale": scale,
                        "scale_label": scale_label,
                        "latent": latent_name,
                        "k": int(k),
                        "comparison": comp_key,
                        "lhs_condition": lhs,
                        "rhs_condition": rhs,
                        "mlp_lhs_cosine": cond_boot["mean"],
                        "mlp_lhs_ci_low": cond_boot["ci_low"],
                        "mlp_lhs_ci_high": cond_boot["ci_high"],
                        "mlp_static_cosine": static_boot["mean"],
                        "mlp_static_ci_low": static_boot["ci_low"],
                        "mlp_static_ci_high": static_boot["ci_high"],
                        "gain_cosine": boot["mean"],
                        "gain_ci_low": boot["ci_low"],
                        "gain_ci_high": boot["ci_high"],
                        "n_images": boot["n"],
                        "n_sessions": boot["n_sessions"],
                    })

    _write_csv(out_dir / "mlp_axis_gain.csv", gain_rows)
    _write_json(out_dir / "run_metadata.json", {
        "run_dir": run_dir,
        "latent_run_dir": latent_run_dir,
        "n_images": len(source_rows),
        "scales": scales,
        "latent_names": latent_names,
        "pca_k_list": pca_k_list,
        "outer_folds": int(args.outer_folds),
        "n_bootstrap": int(args.n_bootstrap),
        "comparisons": {
            "along_vs_static": "MLP(mean over along-edge prior trajectories) − MLP(zero/static)",
            "across_vs_static": "MLP(mean over across-edge prior trajectories) − MLP(zero/static)",
            "along_vs_across": "MLP(along) − MLP(across) — key 4D nonlinear test",
        },
        "mlp": mlp_config.__dict__,
    })

    df = pd.DataFrame(gain_rows)
    for latent_name in latents:
        for k in pca_k_list:
            _build_figure(df, latent=latent_name, k=k, scales=scales, out_dir=out_dir)

    print(f"Wrote axis-conditioned MLP results to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
