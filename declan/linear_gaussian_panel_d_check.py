"""Minimal linear-Gaussian check for the Figure 4D along/across result.

The promoted Figure 4D feature panel is built from the finite candidate
trajectory observer. This script keeps the same response-table inputs and
candidate feature target, but replaces the Poisson table likelihood with a
cross-validated ridge linear-Gaussian feature-to-response model:

    response_movie = intercept + feature_scores @ weights + Gaussian noise

It is intentionally a small diagnostic, not a replacement observer.
"""

from __future__ import annotations

# %% Imports and paths
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.analyze_feature_posterior import (
    _candidate_ids,
    _candidate_set_lookup,
    _candidate_window_indices,
    _fit_feature_spaces,
)
from declan.backimage_trajectory_observer.likelihood import (
    effective_count,
    entropy,
    rank_desc,
)
from declan.backimage_trajectory_observer.observer import (
    feature_recovery_metrics,
    posterior_weighted_feature,
)


BACKIMAGE_BASE = REPO_ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_RUN_DIR = BACKIMAGE_BASE / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1"
DEFAULT_FEATURE_NPZ = (
    BACKIMAGE_BASE
    / "backimage_axis_conditioned_matched_static_feature_posterior_gabor_pyramid_k4_8_uncertainty_v2"
    / "feature_latent_arrays.npz"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "declan"
    / "figure4_active_sensing_atlas"
    / "figures"
    / "panel_D"
    / "diagnostics"
    / "linear_gaussian_feature_model"
)

AXIS_FAMILIES = ("axis_edge_orthogonal", "axis_edge_parallel")
AXIS_LABELS = {
    "axis_edge_orthogonal": "across local edge",
    "axis_edge_parallel": "along local edge",
}
AXIS_COLORS = {
    "axis_edge_orthogonal": "#8064a2",
    "axis_edge_parallel": "#2f8f6a",
}


@dataclass(frozen=True)
class ResponseRecord:
    trial_id: int
    prior_family: str
    response_cache_path: str
    true_candidate_index: int
    candidate_features: np.ndarray


@dataclass(frozen=True)
class LinearGaussianFit:
    weights: np.ndarray
    residual_variance: float


# %% Small statistics helpers
def _bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    confidence: float,
) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or int(n_bootstrap) <= 0:
        return float("nan"), float("nan")
    boot = rng.choice(vals, size=(int(n_bootstrap), vals.size), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha))


def _sign_flip_p(values: np.ndarray, *, rng: np.random.Generator, n_permutations: int) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or int(n_permutations) <= 0:
        return float("nan")
    observed = abs(float(np.mean(vals)))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(n_permutations), vals.size), replace=True)
    null = np.abs((signs * vals[None, :]).mean(axis=1))
    return float((np.sum(null >= observed) + 1.0) / (int(n_permutations) + 1.0))


def _trial_fold_lookup(trial_ids: list[int], n_folds: int) -> dict[int, int]:
    if int(n_folds) < 2:
        raise ValueError("--n-folds must be at least 2 for trial-heldout evaluation")
    return {int(trial_id): int(i % int(n_folds)) for i, trial_id in enumerate(sorted(set(trial_ids)))}


# %% Linear-Gaussian model
def _fit_linear_gaussian(features: np.ndarray, responses: np.ndarray, *, ridge_alpha: float) -> LinearGaussianFit:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"features and responses must be 2D, got {x.shape=} {y.shape=}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"features/response row mismatch: {x.shape[0]} vs {y.shape[0]}")
    x_aug = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(ridge_alpha)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    residual = y - x_aug @ weights
    variance = float(np.mean(residual * residual))
    if not np.isfinite(variance) or variance <= 1e-12:
        variance = 1e-12
    return LinearGaussianFit(weights=weights, residual_variance=variance)


def _score_candidates(fit: LinearGaussianFit, candidate_features: np.ndarray, observed_response: np.ndarray) -> np.ndarray:
    x = np.asarray(candidate_features, dtype=np.float64)
    y = np.asarray(observed_response, dtype=np.float64).reshape(1, -1)
    x_aug = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
    pred = x_aug @ fit.weights
    mse = np.mean((pred - y) ** 2, axis=1)
    return -0.5 * mse / float(fit.residual_variance)


def _posterior_feature_metrics(
    *,
    scores: np.ndarray,
    candidate_features: np.ndarray,
    true_candidate_index: int,
    posterior_temperature: float,
) -> dict[str, Any]:
    z_hat, posterior = posterior_weighted_feature(
        scores,
        candidate_features,
        temperature=float(posterior_temperature),
    )
    true_idx = int(true_candidate_index)
    metrics = feature_recovery_metrics(z_hat, candidate_features[true_idx])
    top = int(np.nanargmax(scores)) if np.isfinite(scores).any() else -1
    return {
        **metrics,
        "image_correct": bool(top == true_idx),
        "score_true_rank": rank_desc(scores, true_idx),
        "posterior_true_mass": float(posterior[true_idx]),
        "posterior_entropy": entropy(posterior),
        "posterior_n_eff": effective_count(posterior),
        "score_gap": float(scores[true_idx] - np.max(np.delete(scores, true_idx)))
        if scores.size > 1
        else float("nan"),
    }


# %% Data loading
def _load_feature_scores(feature_npz: Path, latent: str, k: int) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(feature_npz, allow_pickle=True) as data:
        if str(latent) not in data.files:
            raise ValueError(f"{feature_npz} does not contain latent {latent!r}")
        arrays = {str(latent): np.asarray(data[str(latent)], dtype=np.float32)}
    spaces, qc_rows = _fit_feature_spaces(arrays, [int(k)])
    space = spaces[(str(latent), int(k))]
    return np.asarray(space["scores"], dtype=np.float64), {"space": space, "qc_rows": qc_rows}


def _load_records(
    *,
    run_dir: Path,
    feature_scores: np.ndarray,
    candidate_set_mode: str,
    scale: float,
    max_tables: int,
) -> list[ResponseRecord]:
    windows = pd.read_csv(run_dir / "selected_windows.csv")
    candidate_sets = pd.read_csv(run_dir / "candidate_sets.csv")
    manifest = pd.read_csv(run_dir / "response_cache_manifest.csv")
    manifest = manifest[
        manifest["candidate_set_mode"].astype(str).eq(str(candidate_set_mode))
        & manifest["prior_family"].astype(str).isin(AXIS_FAMILIES)
        & manifest["scale"].astype(float).eq(float(scale))
    ].copy()
    if int(max_tables) > 0:
        manifest = manifest.head(int(max_tables)).copy()
    if manifest.empty:
        raise ValueError("No matching response-cache rows found")

    candidate_lookup = _candidate_set_lookup(candidate_sets)
    source_row_to_pos = {int(row["source_row"]): int(pos) for pos, row in windows.iterrows()}
    records: list[ResponseRecord] = []
    for _, row in manifest.iterrows():
        table_path = run_dir / str(row["response_cache_path"])
        with np.load(table_path, allow_pickle=True) as table:
            n_candidates = int(table["prior_lambda_counts"].shape[0])
            true_idx = int(np.asarray(table["true_candidate_index"]).reshape(-1)[0])
            table_ids = _candidate_ids({key: table[key] for key in table.files}, n_candidates)
        candidate_indices, _source = _candidate_window_indices(
            manifest_row=row,
            candidate_ids=table_ids,
            candidate_lookup=candidate_lookup,
            source_row_to_pos=source_row_to_pos,
            n_windows=int(windows.shape[0]),
        )
        records.append(
            ResponseRecord(
                trial_id=int(row["trial_id"]),
                prior_family=str(row["prior_family"]),
                response_cache_path=str(row["response_cache_path"]),
                true_candidate_index=true_idx,
                candidate_features=feature_scores[np.asarray(candidate_indices, dtype=int)],
            )
        )
    return records


def _load_prior_for_tau(run_dir: Path, record: ResponseRecord, trajectory_index: int) -> np.ndarray:
    with np.load(run_dir / record.response_cache_path, allow_pickle=True) as table:
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float32)
    return prior[:, int(trajectory_index), :, :].reshape(prior.shape[0], -1).astype(np.float64)


def _load_true_observation_for_tau(run_dir: Path, record: ResponseRecord, trajectory_index: int) -> np.ndarray:
    responses = _load_prior_for_tau(run_dir, record, int(trajectory_index))
    return responses[int(record.true_candidate_index)]


def _load_zero_response(run_dir: Path, record: ResponseRecord) -> np.ndarray:
    with np.load(run_dir / record.response_cache_path, allow_pickle=True) as table:
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float32)
    return zero.reshape(zero.shape[0], -1).astype(np.float64)


def _infer_n_trajectories(run_dir: Path, record: ResponseRecord) -> int:
    with np.load(run_dir / record.response_cache_path, allow_pickle=True) as table:
        return int(table["prior_lambda_counts"].shape[1])


# %% Evaluation
def _evaluate(args: argparse.Namespace, records: list[ResponseRecord]) -> pd.DataFrame:
    run_dir = Path(args.run_dir)
    n_trajectories = _infer_n_trajectories(run_dir, records[0])
    fold_of = _trial_fold_lookup([record.trial_id for record in records], int(args.n_folds))
    folds = sorted(set(fold_of.values()))

    zero_rows = [
        (record.trial_id, record.candidate_features, _load_zero_response(run_dir, record))
        for record in records
    ]
    zero_fits: dict[int, LinearGaussianFit] = {}
    for fold in folds:
        train_rows = [row for row in zero_rows if fold_of[int(row[0])] != fold]
        x_train = np.vstack([row[1] for row in train_rows])
        y_train = np.vstack([row[2] for row in train_rows])
        zero_fits[fold] = _fit_linear_gaussian(
            x_train,
            y_train,
            ridge_alpha=float(args.ridge_alpha),
        )

    rows: list[dict[str, Any]] = []
    for trajectory_index in range(n_trajectories):
        for family in AXIS_FAMILIES:
            family_records = [record for record in records if record.prior_family == family]
            x_blocks: list[np.ndarray] = []
            y_blocks: list[np.ndarray] = []
            train_trial_ids: list[int] = []
            loaded_observations: list[tuple[ResponseRecord, np.ndarray]] = []
            for record in family_records:
                candidate_response = _load_prior_for_tau(run_dir, record, trajectory_index)
                x_blocks.append(record.candidate_features)
                y_blocks.append(candidate_response)
                train_trial_ids.extend([int(record.trial_id)] * int(record.candidate_features.shape[0]))
                loaded_observations.append(
                    (
                        record,
                        candidate_response[int(record.true_candidate_index)].copy(),
                    )
                )

            x_all = np.vstack(x_blocks)
            y_all = np.vstack(y_blocks)
            train_trial_ids_arr = np.asarray(train_trial_ids, dtype=int)
            for fold in folds:
                train = np.asarray([fold_of[int(trial_id)] != fold for trial_id in train_trial_ids_arr])
                axis_fit = _fit_linear_gaussian(
                    x_all[train],
                    y_all[train],
                    ridge_alpha=float(args.ridge_alpha),
                )
                zero_fit = zero_fits[fold]
                for record, obs in loaded_observations:
                    if fold_of[int(record.trial_id)] != fold:
                        continue
                    axis_scores = _score_candidates(axis_fit, record.candidate_features, obs)
                    zero_scores = _score_candidates(zero_fit, record.candidate_features, obs)
                    axis_metrics = _posterior_feature_metrics(
                        scores=axis_scores,
                        candidate_features=record.candidate_features,
                        true_candidate_index=record.true_candidate_index,
                        posterior_temperature=float(args.posterior_temperature),
                    )
                    zero_metrics = _posterior_feature_metrics(
                        scores=zero_scores,
                        candidate_features=record.candidate_features,
                        true_candidate_index=record.true_candidate_index,
                        posterior_temperature=float(args.posterior_temperature),
                    )
                    rows.append(
                        {
                            "trial_id": int(record.trial_id),
                            "fold": int(fold),
                            "trajectory_index": int(trajectory_index),
                            "prior_family": str(family),
                            "axis_label": AXIS_LABELS[str(family)],
                            "response_cache_path": str(record.response_cache_path),
                            "true_candidate_index": int(record.true_candidate_index),
                            "linear_gaussian_feature_cosine": float(axis_metrics["feature_cosine"]),
                            "zero_feature_cosine": float(zero_metrics["feature_cosine"]),
                            "linear_gaussian_gain_cosine": float(
                                axis_metrics["feature_cosine"] - zero_metrics["feature_cosine"]
                            ),
                            "linear_gaussian_feature_neg_mse": float(axis_metrics["feature_neg_mse"]),
                            "zero_feature_neg_mse": float(zero_metrics["feature_neg_mse"]),
                            "linear_gaussian_gain_neg_mse": float(
                                axis_metrics["feature_neg_mse"] - zero_metrics["feature_neg_mse"]
                            ),
                            "linear_gaussian_image_correct": bool(axis_metrics["image_correct"]),
                            "zero_image_correct": bool(zero_metrics["image_correct"]),
                            "linear_gaussian_score_true_rank": float(axis_metrics["score_true_rank"]),
                            "zero_score_true_rank": float(zero_metrics["score_true_rank"]),
                            "linear_gaussian_posterior_true_mass": float(axis_metrics["posterior_true_mass"]),
                            "zero_posterior_true_mass": float(zero_metrics["posterior_true_mass"]),
                            "linear_gaussian_posterior_n_eff": float(axis_metrics["posterior_n_eff"]),
                            "zero_posterior_n_eff": float(zero_metrics["posterior_n_eff"]),
                            "linear_gaussian_score_gap": float(axis_metrics["score_gap"]),
                            "zero_score_gap": float(zero_metrics["score_gap"]),
                            "axis_residual_variance": float(axis_fit.residual_variance),
                            "zero_residual_variance": float(zero_fit.residual_variance),
                        }
                    )
        print(f"finished trajectory {trajectory_index + 1}/{n_trajectories}", flush=True)
    return pd.DataFrame(rows)


def _summarize(trials: pd.DataFrame, *, rng: np.random.Generator, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "linear_gaussian_feature_neg_mse",
        "zero_feature_neg_mse",
        "linear_gaussian_gain_neg_mse",
        "linear_gaussian_feature_cosine",
        "zero_feature_cosine",
        "linear_gaussian_gain_cosine",
        "linear_gaussian_image_correct",
        "zero_image_correct",
        "linear_gaussian_posterior_true_mass",
    ]
    for family, group in trials.groupby("prior_family", sort=True):
        trial_means = group.groupby("trial_id")[metrics].mean(numeric_only=True)
        row: dict[str, Any] = {
            "prior_family": str(family),
            "axis_label": AXIS_LABELS.get(str(family), str(family)),
            "n_rows": int(group.shape[0]),
            "n_trials": int(group["trial_id"].nunique()),
            "n_trajectory_samples": int(group["trajectory_index"].nunique()),
        }
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            clustered = trial_means[metric].astype(float).to_numpy()
            ci_low, ci_high = _bootstrap_ci(
                clustered,
                rng=rng,
                n_bootstrap=int(args.n_bootstrap),
                confidence=float(args.confidence),
            )
            row[f"mean_{metric}"] = float(np.nanmean(values))
            row[f"median_{metric}"] = float(np.nanmedian(values))
            row[f"trial_cluster_ci_low_{metric}"] = ci_low
            row[f"trial_cluster_ci_high_{metric}"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows).sort_values("prior_family").reset_index(drop=True)


def _contrasts(trials: pd.DataFrame, *, rng: np.random.Generator, args: argparse.Namespace) -> pd.DataFrame:
    metrics = [
        ("linear_gaussian_feature_neg_mse", "linear-Gaussian feature -MSE"),
        ("linear_gaussian_gain_neg_mse", "linear-Gaussian feature gain vs zero -MSE"),
        ("linear_gaussian_feature_cosine", "linear-Gaussian feature cosine"),
        ("linear_gaussian_gain_cosine", "linear-Gaussian feature gain vs zero cosine"),
        ("linear_gaussian_image_correct", "linear-Gaussian image accuracy"),
        ("linear_gaussian_posterior_true_mass", "linear-Gaussian true posterior mass"),
    ]
    rows: list[dict[str, Any]] = []
    for metric, label in metrics:
        wide = trials.pivot_table(
            index=["trial_id", "trajectory_index"],
            columns="prior_family",
            values=metric,
            aggfunc="first",
        )
        if not set(AXIS_FAMILIES).issubset(set(wide.columns)):
            continue
        diffs = (wide["axis_edge_parallel"].astype(float) - wide["axis_edge_orthogonal"].astype(float)).dropna()
        row_values = diffs.to_numpy(dtype=np.float64)
        row_ci_low, row_ci_high = _bootstrap_ci(
            row_values,
            rng=rng,
            n_bootstrap=int(args.n_bootstrap),
            confidence=float(args.confidence),
        )
        trial_values = (
            diffs.rename("parallel_minus_orthogonal")
            .reset_index()
            .groupby("trial_id")["parallel_minus_orthogonal"]
            .mean()
            .to_numpy(dtype=np.float64)
        )
        ci_low, ci_high = _bootstrap_ci(
            trial_values,
            rng=rng,
            n_bootstrap=int(args.n_bootstrap),
            confidence=float(args.confidence),
        )
        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "mean_parallel_minus_orthogonal": float(np.nanmean(row_values)),
                "median_parallel_minus_orthogonal": float(np.nanmedian(row_values)),
                "uncertainty_unit": "trial_cluster_mean",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "sign_flip_p_two_sided": _sign_flip_p(
                    trial_values,
                    rng=rng,
                    n_permutations=int(args.n_permutations),
                ),
                "row_ci_low": row_ci_low,
                "row_ci_high": row_ci_high,
                "row_sign_flip_p_two_sided": _sign_flip_p(
                    row_values,
                    rng=rng,
                    n_permutations=int(args.n_permutations),
                ),
                "n_pairs": int(row_values.shape[0]),
                "n_trials": int(trial_values.shape[0]),
                "fraction_positive": float(np.mean(row_values > 0.0)),
                "fraction_positive_trials": float(np.mean(trial_values > 0.0)),
            }
        )
    return pd.DataFrame(rows)


# %% Plotting and output
def _plot(summary: pd.DataFrame, contrasts: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), constrained_layout=True)
    order = ["axis_edge_orthogonal", "axis_edge_parallel"]
    sub = summary.set_index("prior_family").loc[order].reset_index()
    x = np.arange(len(sub))

    def bar_metric(ax: plt.Axes, metric: str, ylabel: str, title: str) -> None:
        y = sub[f"mean_{metric}"].to_numpy(dtype=float)
        lo = sub[f"trial_cluster_ci_low_{metric}"].to_numpy(dtype=float)
        hi = sub[f"trial_cluster_ci_high_{metric}"].to_numpy(dtype=float)
        ax.bar(x, y, color=[AXIS_COLORS[fam] for fam in sub["prior_family"]], width=0.62)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="none", ecolor="#1f252b", capsize=3)
        ax.set_xticks(x, sub["axis_label"].str.replace(" ", "\n"))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0.0, color="#242a2f", lw=0.8)

    def contrast_metric(ax: plt.Axes, metric: str, ylabel: str, title: str) -> None:
        contrast = contrasts[contrasts["metric"].eq(metric)].iloc[0]
        diff = float(contrast["mean_parallel_minus_orthogonal"])
        ci_low = float(contrast["ci_low"])
        ci_high = float(contrast["ci_high"])
        ax.axhline(0.0, color="#747a80", lw=1.0)
        ax.bar([0], [diff], color=AXIS_COLORS["axis_edge_parallel"], width=0.5)
        ax.errorbar([0], [diff], yerr=[[diff - ci_low], [ci_high - diff]], fmt="none", ecolor="#1f252b", capsize=3)
        ax.set_xticks([0], ["along - across"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\np={float(contrast['sign_flip_p_two_sided']):.3g}")
        pad = max(abs(ci_low), abs(ci_high), abs(diff), 1e-3) * 1.3
        ax.set_ylim(-pad, pad)

    bar_metric(
        axes[0, 0],
        "linear_gaussian_feature_neg_mse",
        "feature recovery [-MSE]",
        "Raw feature recovery",
    )
    contrast_metric(
        axes[0, 1],
        "linear_gaussian_feature_neg_mse",
        "along - across [-MSE]",
        "Raw recovery contrast",
    )
    bar_metric(
        axes[1, 0],
        "linear_gaussian_gain_neg_mse",
        "gain vs zero eye [-MSE]",
        "Zero-baseline gain",
    )
    contrast_metric(
        axes[1, 1],
        "linear_gaussian_feature_cosine",
        "along - across cosine",
        "Cosine contrast",
    )

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d8dde3", lw=0.7, alpha=0.9)
    fig.suptitle("Figure 4D minimal linear-Gaussian check", fontsize=10)
    fig.savefig(out_dir / "linear_gaussian_panel_d_check.png", dpi=300)
    fig.savefig(out_dir / "linear_gaussian_panel_d_check.pdf")
    plt.close(fig)


def build(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_scores, feature_info = _load_feature_scores(Path(args.feature_npz), str(args.latent), int(args.k))
    records = _load_records(
        run_dir=Path(args.run_dir),
        feature_scores=feature_scores,
        candidate_set_mode=str(args.candidate_set_mode),
        scale=float(args.scale),
        max_tables=int(args.max_tables),
    )
    print(f"loaded {len(records)} response tables", flush=True)
    trials = _evaluate(args, records)
    rng = np.random.default_rng(int(args.seed))
    summary = _summarize(trials, rng=rng, args=args)
    contrasts = _contrasts(trials, rng=rng, args=args)

    trials.to_csv(out_dir / "linear_gaussian_panel_d_trials.csv", index=False)
    summary.to_csv(out_dir / "linear_gaussian_panel_d_summary.csv", index=False)
    contrasts.to_csv(out_dir / "linear_gaussian_panel_d_contrasts.csv", index=False)
    _plot(summary, contrasts, out_dir)
    readme = [
        "# Figure 4D Linear-Gaussian Feature-Model Check",
        "",
        "Minimal sanity check for the along/across contour result.",
        "",
        "Inputs are the same matched-static panel-D response tables and the same",
        f"`{args.latent}` PCA feature target. The likelihood is replaced by a",
        "trial-heldout ridge linear-Gaussian feature-to-response model.",
        "",
        "Panel-matched contrast: `axis_edge_parallel - axis_edge_orthogonal`",
        "in feature-recovery gain over the zero-eye baseline, using `-MSE`.",
        "The summary figure also shows raw feature recovery and cosine contrasts,",
        "because the linear-Gaussian zero baseline is a deliberately different",
        "model from the original Poisson observer baseline.",
        "",
        "Outputs:",
        "",
        "- `linear_gaussian_panel_d_trials.csv`",
        "- `linear_gaussian_panel_d_summary.csv`",
        "- `linear_gaussian_panel_d_contrasts.csv`",
        "- `linear_gaussian_panel_d_check.png`",
        "",
        f"Feature-space variance fraction: {float(feature_info['space']['variance_fraction']):.6f}",
        f"Ridge alpha: {float(args.ridge_alpha):.6g}",
        f"Posterior temperature: {float(args.posterior_temperature):.6g}",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--feature-npz", type=Path, default=DEFAULT_FEATURE_NPZ)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-set-mode", default="matched_static_response")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--latent", default="pyramid_local_field")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--posterior-temperature", type=float, default=0.01)
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    out_dir = build(build_parser().parse_args())
    print(out_dir)


if __name__ == "__main__":
    main()
