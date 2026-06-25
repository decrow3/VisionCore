"""Known-axis along/across feature-alignment diagnostic for Panel 4D.

This is deliberately different from the hidden-eye axis-prior observer. It
uses the saved axis-conditioned prior response tables as synthetic observed
movies: for each trial and trajectory sample, the true candidate's along- or
across-axis response becomes the observation, and the known trajectory index is
used to score the candidate images.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

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
    poisson_expected_count_loglik,
    rank_desc,
    true_margin,
)
from declan.backimage_trajectory_observer.observer import (
    feature_recovery_metrics,
    posterior_weighted_feature,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1"
)
DEFAULT_FEATURE_NPZ = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
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
    / "known_axis_feature_alignment"
)

AXIS_LABELS = {
    "axis_edge_parallel": "along contour",
    "axis_edge_orthogonal": "across contour",
}
AXIS_COLORS = {
    "axis_edge_parallel": "#2f8f6a",
    "axis_edge_orthogonal": "#8064a2",
}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _posterior_row(
    *,
    scores: np.ndarray,
    candidate_features: np.ndarray,
    true_feature: np.ndarray,
    true_idx: int,
    temperature: float,
) -> dict[str, Any]:
    z_hat, posterior = posterior_weighted_feature(scores, candidate_features, temperature=float(temperature))
    metrics = feature_recovery_metrics(z_hat, true_feature)
    top = int(np.nanargmax(posterior)) if posterior.size and np.isfinite(posterior).any() else -1
    return {
        **metrics,
        "candidate_posterior_true_mass": float(posterior[int(true_idx)]),
        "candidate_posterior_entropy": entropy(posterior),
        "candidate_posterior_N_eff": effective_count(posterior),
        "candidate_posterior_N_eff_fraction": float(effective_count(posterior) / posterior.size),
        "posterior_top_candidate_index": top,
        "posterior_top_is_true": bool(top == int(true_idx)),
        "score_true_rank": rank_desc(scores, int(true_idx)),
        "score_true_margin": true_margin(scores, int(true_idx)),
        "score_true_value": float(scores[int(true_idx)]),
    }


def _bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int, confidence: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or int(n_bootstrap) <= 0:
        return float("nan"), float("nan")
    samples = rng.choice(arr, size=(int(n_bootstrap), arr.size), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _sign_flip_p(values: np.ndarray, *, rng: np.random.Generator, n_permutations: int) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or int(n_permutations) <= 0:
        return float("nan")
    observed = abs(float(np.mean(arr)))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(n_permutations), arr.size), replace=True)
    null = np.abs((signs * arr[None, :]).mean(axis=1))
    return float((np.sum(null >= observed) + 1.0) / (int(n_permutations) + 1.0))


def _summarize(rows: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int, confidence: float) -> pd.DataFrame:
    out = []
    for family, grp in rows.groupby("prior_family", sort=True):
        feature_values = grp["known_axis_feature_cosine"].to_numpy(dtype=float)
        gain_values = grp["known_axis_minus_zero_feature_cosine"].to_numpy(dtype=float)
        neg_mse_values = grp["known_axis_feature_neg_mse"].to_numpy(dtype=float)
        gain_neg_mse_values = grp["known_axis_minus_zero_feature_neg_mse"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(feature_values, rng=rng, n_bootstrap=n_bootstrap, confidence=confidence)
        gain_ci_low, gain_ci_high = _bootstrap_ci(gain_values, rng=rng, n_bootstrap=n_bootstrap, confidence=confidence)
        neg_mse_ci_low, neg_mse_ci_high = _bootstrap_ci(
            neg_mse_values, rng=rng, n_bootstrap=n_bootstrap, confidence=confidence
        )
        gain_neg_mse_ci_low, gain_neg_mse_ci_high = _bootstrap_ci(
            gain_neg_mse_values, rng=rng, n_bootstrap=n_bootstrap, confidence=confidence
        )
        out.append(
            {
                "prior_family": family,
                "axis_label": AXIS_LABELS.get(str(family), str(family)),
                "n_rows": int(len(grp)),
                "n_trials": int(grp["trial_id"].nunique()),
                "n_trajectory_samples": int(grp["trajectory_index"].nunique()),
                "mean_known_axis_feature_cosine": float(np.mean(feature_values)),
                "median_known_axis_feature_cosine": float(np.median(feature_values)),
                "known_axis_feature_cosine_ci_low": ci_low,
                "known_axis_feature_cosine_ci_high": ci_high,
                "mean_known_axis_minus_zero_feature_cosine": float(np.mean(gain_values)),
                "known_axis_minus_zero_feature_cosine_ci_low": gain_ci_low,
                "known_axis_minus_zero_feature_cosine_ci_high": gain_ci_high,
                "mean_known_axis_feature_neg_mse": float(np.mean(neg_mse_values)),
                "known_axis_feature_neg_mse_ci_low": neg_mse_ci_low,
                "known_axis_feature_neg_mse_ci_high": neg_mse_ci_high,
                "mean_known_axis_minus_zero_feature_neg_mse": float(np.mean(gain_neg_mse_values)),
                "known_axis_minus_zero_feature_neg_mse_ci_low": gain_neg_mse_ci_low,
                "known_axis_minus_zero_feature_neg_mse_ci_high": gain_neg_mse_ci_high,
                "image_accuracy": float(grp["known_axis_image_correct"].mean()),
                "mean_true_mass": float(grp["known_axis_candidate_posterior_true_mass"].mean()),
            }
        )
    return pd.DataFrame(out).sort_values("prior_family").reset_index(drop=True)


def _contrasts(rows: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int, n_permutations: int, confidence: float) -> pd.DataFrame:
    key_cols = ["trial_id", "trajectory_index"]
    wide = rows.pivot_table(
        index=key_cols,
        columns="prior_family",
        values=[
            "known_axis_feature_cosine",
            "known_axis_minus_zero_feature_cosine",
            "known_axis_feature_neg_mse",
            "known_axis_minus_zero_feature_neg_mse",
            "known_axis_image_correct",
            "known_axis_candidate_posterior_true_mass",
        ],
        aggfunc="first",
    )
    required = {"axis_edge_parallel", "axis_edge_orthogonal"}
    out = []
    for metric, label in [
        ("known_axis_feature_cosine", "known-axis feature cosine"),
        ("known_axis_minus_zero_feature_cosine", "known-axis feature gain vs zero"),
        ("known_axis_feature_neg_mse", "known-axis feature neg-MSE"),
        ("known_axis_minus_zero_feature_neg_mse", "known-axis feature neg-MSE gain vs zero"),
        ("known_axis_image_correct", "known-axis image accuracy"),
        ("known_axis_candidate_posterior_true_mass", "known-axis true posterior mass"),
    ]:
        if not required.issubset(set(wide[metric].columns)):
            continue
        diffs = (
            wide[(metric, "axis_edge_parallel")].astype(float)
            - wide[(metric, "axis_edge_orthogonal")].astype(float)
        ).dropna()
        row_values = diffs.to_numpy(dtype=float)
        row_ci_low, row_ci_high = _bootstrap_ci(
            row_values, rng=rng, n_bootstrap=n_bootstrap, confidence=confidence
        )
        trial_values = (
            diffs.rename("parallel_minus_orthogonal")
            .reset_index()
            .groupby("trial_id")["parallel_minus_orthogonal"]
            .mean()
            .to_numpy(dtype=float)
        )
        ci_low, ci_high = _bootstrap_ci(
            trial_values, rng=rng, n_bootstrap=n_bootstrap, confidence=confidence
        )
        out.append(
            {
                "metric": metric,
                "metric_label": label,
                "mean_parallel_minus_orthogonal": float(diffs.mean()),
                "median_parallel_minus_orthogonal": float(diffs.median()),
                "uncertainty_unit": "trial_cluster_mean",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "sign_flip_p_two_sided": _sign_flip_p(
                    trial_values, rng=rng, n_permutations=n_permutations
                ),
                "row_ci_low": row_ci_low,
                "row_ci_high": row_ci_high,
                "row_sign_flip_p_two_sided": _sign_flip_p(
                    row_values, rng=rng, n_permutations=n_permutations
                ),
                "n_pairs": int(diffs.shape[0]),
                "n_trials": int(trial_values.shape[0]),
                "fraction_positive": float(np.mean(row_values > 0.0)),
                "fraction_positive_trials": float(np.mean(trial_values > 0.0)),
            }
        )
    return pd.DataFrame(out)


def _plot(summary: pd.DataFrame, contrasts: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    order = ["axis_edge_orthogonal", "axis_edge_parallel"]
    sub = summary.set_index("prior_family").loc[order].reset_index()
    x = np.arange(len(sub))
    y = sub["mean_known_axis_feature_cosine"].to_numpy(dtype=float)
    lo = sub["known_axis_feature_cosine_ci_low"].to_numpy(dtype=float)
    hi = sub["known_axis_feature_cosine_ci_high"].to_numpy(dtype=float)
    axes[0].bar(x, y, color=[AXIS_COLORS[fam] for fam in sub["prior_family"]], width=0.62)
    axes[0].errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="none", ecolor="#1f252b", capsize=3, lw=1.0)
    axes[0].set_xticks(x, sub["axis_label"].str.replace(" ", "\n"))
    axes[0].set_ylabel("feature recovery (cosine)")
    axes[0].set_title("Known rotated trace")
    axes[0].set_ylim(max(0.0, float(np.nanmin(lo)) - 0.03), min(1.0, float(np.nanmax(hi)) + 0.03))

    contrast = contrasts[contrasts["metric"].eq("known_axis_feature_cosine")].iloc[0]
    diff = float(contrast["mean_parallel_minus_orthogonal"])
    lo_d = float(contrast["ci_low"])
    hi_d = float(contrast["ci_high"])
    axes[1].axhline(0.0, color="#747a80", lw=1.0)
    axes[1].bar([0], [diff], color="#2f8f6a", width=0.5)
    axes[1].errorbar([0], [diff], yerr=[[diff - lo_d], [hi_d - diff]], fmt="none", ecolor="#1f252b", capsize=3)
    axes[1].set_xticks([0], ["along - across"])
    axes[1].set_ylabel("feature cosine difference")
    axes[1].set_title("Paired contrast")
    pad = max(abs(lo_d), abs(hi_d), abs(diff), 1e-3) * 1.3
    axes[1].set_ylim(-pad, pad)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Panel 4D known-axis feature-alignment diagnostic", fontsize=10)
    fig.savefig(out_dir / "panel_D_known_axis_feature_alignment.png", dpi=300)
    fig.savefig(out_dir / "panel_D_known_axis_feature_alignment.pdf")
    plt.close(fig)


def build(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(run_dir / "selected_windows.csv")
    candidate_sets = pd.read_csv(run_dir / "candidate_sets.csv")
    manifest = pd.read_csv(run_dir / "response_cache_manifest.csv")
    manifest = manifest[
        manifest["candidate_set_mode"].astype(str).eq(str(args.candidate_set_mode))
        & manifest["prior_family"].astype(str).isin(["axis_edge_parallel", "axis_edge_orthogonal"])
        & manifest["scale"].astype(float).eq(float(args.scale))
    ].copy()
    if int(args.max_tables) > 0:
        manifest = manifest.head(int(args.max_tables)).copy()
    if manifest.empty:
        raise ValueError("No matching response-cache rows found")

    with np.load(args.feature_npz, allow_pickle=True) as data:
        feature_arrays = {str(args.latent): np.asarray(data[str(args.latent)], dtype=np.float32)}
    feature_spaces, _qc = _fit_feature_spaces(feature_arrays, [int(args.k)])
    space = feature_spaces[(str(args.latent), int(args.k))]
    features_all = np.asarray(space["scores"], dtype=np.float64)

    candidate_lookup = _candidate_set_lookup(candidate_sets)
    source_row_to_pos = {int(row["source_row"]): int(pos) for pos, row in windows.iterrows()}

    rows: list[dict[str, Any]] = []
    for table_index, man_row in manifest.iterrows():
        table_path = run_dir / str(man_row["response_cache_path"])
        table = _load_npz(table_path)
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        true_idx = int(np.asarray(table["true_candidate_index"]).reshape(-1)[0])
        candidate_ids = _candidate_ids(table, int(prior.shape[0]))
        candidate_indices, candidate_index_source = _candidate_window_indices(
            manifest_row=man_row,
            candidate_ids=candidate_ids,
            candidate_lookup=candidate_lookup,
            source_row_to_pos=source_row_to_pos,
            n_windows=int(windows.shape[0]),
        )
        candidate_features = features_all[np.asarray(candidate_indices, dtype=int)]
        true_feature = candidate_features[true_idx]
        for trajectory_index in range(int(prior.shape[1])):
            y_obs = prior[true_idx, trajectory_index]
            known_axis_scores = poisson_expected_count_loglik(
                y_obs,
                prior[:, trajectory_index, :, :],
                eps=float(args.eps),
                likelihood_scale=float(args.likelihood_scale),
            )
            zero_scores = poisson_expected_count_loglik(
                y_obs,
                zero,
                eps=float(args.eps),
                likelihood_scale=float(args.likelihood_scale),
            )
            known_axis = _posterior_row(
                scores=known_axis_scores,
                candidate_features=candidate_features,
                true_feature=true_feature,
                true_idx=true_idx,
                temperature=float(args.posterior_temperature),
            )
            zero_row = _posterior_row(
                scores=zero_scores,
                candidate_features=candidate_features,
                true_feature=true_feature,
                true_idx=true_idx,
                temperature=float(args.posterior_temperature),
            )
            rows.append(
                {
                    "table_index": int(table_index),
                    "trial_id": int(man_row["trial_id"]),
                    "trajectory_index": int(trajectory_index),
                    "response_cache_path": str(man_row["response_cache_path"]),
                    "candidate_set_mode": str(man_row["candidate_set_mode"]),
                    "prior_family": str(man_row["prior_family"]),
                    "axis_label": AXIS_LABELS.get(str(man_row["prior_family"]), str(man_row["prior_family"])),
                    "scale": float(man_row["scale"]),
                    "n_candidates": int(prior.shape[0]),
                    "n_trajectories": int(prior.shape[1]),
                    "true_candidate_index": int(true_idx),
                    "candidate_index_source": str(candidate_index_source),
                    "latent": str(args.latent),
                    "requested_k": int(args.k),
                    "k_eff": int(space["k_eff"]),
                    "likelihood_scale": float(args.likelihood_scale),
                    "posterior_temperature": float(args.posterior_temperature),
                    "known_axis_feature_cosine": float(known_axis["feature_cosine"]),
                    "zero_feature_cosine": float(zero_row["feature_cosine"]),
                    "known_axis_minus_zero_feature_cosine": float(
                        known_axis["feature_cosine"] - zero_row["feature_cosine"]
                    ),
                    "known_axis_feature_neg_mse": float(known_axis["feature_neg_mse"]),
                    "zero_feature_neg_mse": float(zero_row["feature_neg_mse"]),
                    "known_axis_minus_zero_feature_neg_mse": float(
                        known_axis["feature_neg_mse"] - zero_row["feature_neg_mse"]
                    ),
                    "known_axis_image_correct": bool(known_axis["posterior_top_is_true"]),
                    "zero_image_correct": bool(zero_row["posterior_top_is_true"]),
                    "known_axis_candidate_posterior_true_mass": float(
                        known_axis["candidate_posterior_true_mass"]
                    ),
                    "zero_candidate_posterior_true_mass": float(zero_row["candidate_posterior_true_mass"]),
                    "known_axis_score_true_rank": float(known_axis["score_true_rank"]),
                    "zero_score_true_rank": float(zero_row["score_true_rank"]),
                }
            )

    trials = pd.DataFrame(rows)
    rng = np.random.default_rng(int(args.seed))
    summary = _summarize(trials, rng=rng, n_bootstrap=int(args.n_bootstrap), confidence=float(args.confidence))
    contrasts = _contrasts(
        trials,
        rng=rng,
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        confidence=float(args.confidence),
    )

    trials.to_csv(out_dir / "panel_D_known_axis_feature_alignment_trials.csv", index=False)
    summary.to_csv(out_dir / "panel_D_known_axis_feature_alignment_summary.csv", index=False)
    contrasts.to_csv(out_dir / "panel_D_known_axis_feature_alignment_contrasts.csv", index=False)
    _plot(summary, contrasts, out_dir)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Panel 4D Known-Axis Feature-Alignment Diagnostic",
                "",
                "This diagnostic tests the direct along/across question: for each saved",
                "axis-conditioned trajectory sample, the true candidate's rotated response",
                "movie is treated as the observation, and the same known trajectory index is",
                "used to score candidate images. This is not the hidden-eye joint decoder.",
                "",
                "Primary contrast: `axis_edge_parallel - axis_edge_orthogonal` in",
                "known-axis posterior feature cosine. The main confidence intervals",
                "and sign-flip tests are clustered by trial; row-level trajectory-sample",
                "uncertainty is retained in the contrast CSV for auditing.",
                "",
                "Outputs:",
                "",
                "- `panel_D_known_axis_feature_alignment.png`",
                "- `panel_D_known_axis_feature_alignment_summary.csv`",
                "- `panel_D_known_axis_feature_alignment_contrasts.csv`",
                "- `panel_D_known_axis_feature_alignment_trials.csv`",
                "",
            ]
        ),
        encoding="utf-8",
    )
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
    parser.add_argument("--likelihood-scale", type=float, default=1.0)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
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
