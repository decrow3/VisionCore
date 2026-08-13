#!/usr/bin/env python3
"""Checkpoint 13B: grouped rank and influence audit for the RR100 full bank.

The input is checkpoint 13A's map-validated low-frequency modulation measure.
All inferential summaries treat source trials, rather than cached windows, as
the independent condition unit.  Paired, matched-other-trial, and their signed
difference are analyzed separately.  Session-demeaned and session-disjoint
analyses test whether the result is merely a between-session effect.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs/fig4_active_sensing/rr100_fullbank_map_checkpoint_13a_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_fullbank_rank_checkpoint_13b_v1"
RESPONSIVE_THRESHOLD_HZ = 1e-4
EPS = 1e-12
SEED = 20260812

MATRIX_LABELS = {
    "paired": "Own eye trajectory",
    "matched": "Matched other-trial trajectories",
    "pairing_residual": "Own minus matched",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-split-half", type=int, default=3000)
    parser.add_argument("--n-group-splits", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def safe_corr(a: np.ndarray, b: np.ndarray, kind: str = "pearson") -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) <= EPS or np.std(b) <= EPS:
        return np.nan
    if kind == "pearson":
        return float(pearsonr(a, b).statistic)
    return float(spearmanr(a, b).statistic)


def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, float)
    scale = matrix.std(axis=0, ddof=0)
    if np.any(scale <= EPS):
        raise ValueError("Constant unit encountered during standardization")
    return (matrix - matrix.mean(axis=0, keepdims=True)) / scale


def pca_stats_from_z(z: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    covariance = z.T @ z
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0)
    loading = vectors[:, order[0]].copy()
    score = z @ loading
    simple = z.mean(axis=1)
    if safe_corr(score, simple) < 0:
        score *= -1
        loading *= -1
    fraction = float(values[0] / np.maximum(values.sum(), EPS))
    return fraction, score, loading, values


def pairwise_unit_profile_spearman(z: np.ndarray) -> float:
    corr = spearmanr(z, axis=0).statistic
    corr = np.asarray(corr, float)
    return float(np.nanmedian(corr[np.triu_indices(z.shape[1], 1)]))


def aggregate_rows(matrix: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(matrix)
    frame.insert(0, "group", labels)
    grouped = frame.groupby("group", sort=True).mean()
    return grouped.to_numpy(float), grouped.index.to_numpy(str)


def load_inputs(input_dir: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    archive = np.load(input_dir / "fullbank_rr100_effect_matrices.npz")
    metadata = pd.read_csv(input_dir / "window_effect_summary.csv").sort_values("image_index").reset_index(drop=True)
    paired = archive["paired_four_dct_rms_hz"].astype(float)
    matched = archive["matched_unpaired_four_dct_rms_hz"].astype(float)
    samples = archive["matched_unpaired_sample_four_dct_rms_hz"].astype(float)
    if paired.shape != (384, 100) or matched.shape != paired.shape or samples.shape != (384, 8, 100):
        raise ValueError(f"Unexpected input shapes: {paired.shape}, {matched.shape}, {samples.shape}")
    matrices = {"paired": paired, "matched": matched, "pairing_residual": paired - matched}
    return matrices, metadata, samples


def build_levels(
    matrices: dict[str, np.ndarray], metadata: pd.DataFrame, responsive: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    trial_labels = metadata["source_trial_group"].astype(str).to_numpy()
    session_labels_window = metadata["session"].astype(str).to_numpy()
    trial_metadata = metadata[["source_trial_group", "session", "trial_idx"]].drop_duplicates("source_trial_group")
    trial_metadata = trial_metadata.set_index("source_trial_group")
    levels: dict[str, dict[str, np.ndarray]] = {}
    aligned_trial_ids = None
    for name, matrix in matrices.items():
        x = matrix[:, responsive]
        trial, trial_ids = aggregate_rows(x, trial_labels)
        if aligned_trial_ids is None:
            aligned_trial_ids = trial_ids
        elif not np.array_equal(aligned_trial_ids, trial_ids):
            raise ValueError("Trial aggregation misalignment")
        trial_sessions = trial_metadata.loc[trial_ids, "session"].astype(str).to_numpy()
        session, session_ids = aggregate_rows(trial, trial_sessions)
        counts = pd.Series(trial_sessions).value_counts()
        eligible = np.array([counts[value] >= 2 for value in trial_sessions])
        within = trial[eligible].copy()
        within_sessions = trial_sessions[eligible]
        for session_name in np.unique(within_sessions):
            mask = within_sessions == session_name
            within[mask] -= within[mask].mean(axis=0, keepdims=True)
        levels[name] = {
            "window": x,
            "trial": trial,
            "session": session,
            "trial_within_session": within,
            "trial_ids": trial_ids,
            "trial_sessions": trial_sessions,
            "session_ids": session_ids,
            "within_session_labels": within_sessions,
            "within_trial_ids": trial_ids[eligible],
        }
    trial_table = trial_metadata.loc[aligned_trial_ids].reset_index()
    trial_table["trial_row"] = np.arange(len(trial_table))
    return levels, trial_table


def rank_summary(levels: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for matrix_name, content in levels.items():
        arrays[matrix_name] = {}
        for level_name in ("window", "trial", "trial_within_session", "session"):
            matrix = content[level_name]
            nonconstant = matrix.std(axis=0) > EPS
            z = standardize_columns(matrix[:, nonconstant])
            fraction, score, loading, eigenvalues = pca_stats_from_z(z)
            raw_values = np.linalg.eigvalsh(matrix[:, nonconstant].T @ matrix[:, nonconstant])
            raw_values = np.maximum(raw_values, 0)
            rows.append(
                {
                    "matrix": matrix_name,
                    "matrix_label": MATRIX_LABELS[matrix_name],
                    "level": level_name,
                    "n_rows": matrix.shape[0],
                    "n_units": int(nonconstant.sum()),
                    "centered_per_unit_zscored_pc1_variance_fraction": fraction,
                    "raw_uncentered_rank1_energy_fraction": float(raw_values[-1] / np.maximum(raw_values.sum(), EPS)),
                    "median_pairwise_unit_profile_spearman": pairwise_unit_profile_spearman(z),
                    "pc1_vs_population_mean_z_pearson": safe_corr(score, z.mean(axis=1)),
                }
            )
            arrays[matrix_name][f"{level_name}_z"] = z
            arrays[matrix_name][f"{level_name}_score"] = score
            arrays[matrix_name][f"{level_name}_loading"] = loading
            arrays[matrix_name][f"{level_name}_eigenvalues"] = eigenvalues
            arrays[matrix_name][f"{level_name}_unit_mask"] = nonconstant
    return pd.DataFrame(rows), arrays


def shuffled_pc1(
    matrix: np.ndarray,
    session_labels: np.ndarray,
    n_null: int,
    rng: np.random.Generator,
    within_session: bool,
) -> np.ndarray:
    z = standardize_columns(matrix)
    out = np.empty(n_null, float)
    session_indices = [np.flatnonzero(session_labels == value) for value in np.unique(session_labels)]
    for iteration in range(n_null):
        shuffled = np.empty_like(z)
        for unit in range(z.shape[1]):
            if within_session:
                for indices in session_indices:
                    shuffled[indices, unit] = z[rng.permutation(indices), unit]
            else:
                shuffled[:, unit] = z[rng.permutation(z.shape[0]), unit]
        out[iteration] = pca_stats_from_z(shuffled)[0]
    return out


def cluster_bootstrap_pc1(
    matrix: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = np.unique(labels)
    indices = {value: np.flatnonzero(labels == value) for value in groups}
    out = np.empty(n_bootstrap, float)
    for iteration in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([indices[value] for value in sampled])
        candidate = matrix[rows]
        keep = candidate.std(axis=0) > EPS
        out[iteration] = pca_stats_from_z(standardize_columns(candidate[:, keep]))[0]
    return out


def rank_uncertainty(
    levels: dict[str, dict[str, np.ndarray]],
    rank: pd.DataFrame,
    n_null: int,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    rows = []
    saved = {}
    for matrix_name, content in levels.items():
        trial = content["trial"]
        sessions = content["trial_sessions"]
        observed = float(
            rank.loc[rank["matrix"].eq(matrix_name) & rank["level"].eq("trial"), "centered_per_unit_zscored_pc1_variance_fraction"].iloc[0]
        )
        null_global = shuffled_pc1(trial, sessions, n_null, rng, within_session=False)
        null_within = shuffled_pc1(trial, sessions, n_null, rng, within_session=True)
        boot_trial = cluster_bootstrap_pc1(trial, np.arange(len(trial)).astype(str), n_bootstrap, rng)
        boot_session = cluster_bootstrap_pc1(trial, sessions, n_bootstrap, rng)
        rows.append(
            {
                "matrix": matrix_name,
                "trial_pc1_observed": observed,
                "global_trial_shuffle_null_median": float(np.median(null_global)),
                "global_trial_shuffle_null_ci99_high": float(np.quantile(null_global, 0.99)),
                "global_trial_shuffle_p_upper": float((1 + np.sum(null_global >= observed)) / (n_null + 1)),
                "within_session_trial_shuffle_null_median": float(np.median(null_within)),
                "within_session_trial_shuffle_null_ci99_high": float(np.quantile(null_within, 0.99)),
                "within_session_trial_shuffle_p_upper": float((1 + np.sum(null_within >= observed)) / (n_null + 1)),
                "trial_bootstrap_pc1_ci95_low": float(np.quantile(boot_trial, 0.025)),
                "trial_bootstrap_pc1_median": float(np.median(boot_trial)),
                "trial_bootstrap_pc1_ci95_high": float(np.quantile(boot_trial, 0.975)),
                "session_cluster_bootstrap_pc1_ci95_low": float(np.quantile(boot_session, 0.025)),
                "session_cluster_bootstrap_pc1_median": float(np.median(boot_session)),
                "session_cluster_bootstrap_pc1_ci95_high": float(np.quantile(boot_session, 0.975)),
            }
        )
        saved[f"{matrix_name}_null_global"] = null_global
        saved[f"{matrix_name}_null_within_session"] = null_within
        saved[f"{matrix_name}_bootstrap_trial"] = boot_trial
        saved[f"{matrix_name}_bootstrap_session"] = boot_session
    return pd.DataFrame(rows), saved


def split_half_scores(z: np.ndarray, n_split: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pearson = np.empty(n_split, float)
    spearman = np.empty(n_split, float)
    half = z.shape[1] // 2
    for iteration in range(n_split):
        order = rng.permutation(z.shape[1])
        a = z[:, order[:half]].mean(axis=1)
        b = z[:, order[half:]].mean(axis=1)
        pearson[iteration] = safe_corr(a, b)
        spearman[iteration] = safe_corr(a, b, "spearman")
    return pearson, spearman


def grouped_loading_splits(
    trial: np.ndarray,
    sessions: np.ndarray,
    mode: str,
    n_splits: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    groups = np.arange(len(trial)).astype(str) if mode == "trial_disjoint" else sessions
    unique = np.unique(groups)
    for iteration in range(n_splits):
        shuffled = rng.permutation(unique)
        n_test = max(2, int(round(0.2 * len(unique))))
        test_groups = set(shuffled[:n_test])
        test = np.array([value in test_groups for value in groups])
        train = ~test
        train_keep = trial[train].std(axis=0) > EPS
        test_keep = trial[test].std(axis=0) > EPS
        keep = train_keep & test_keep
        z_train = standardize_columns(trial[train][:, keep])
        z_test = standardize_columns(trial[test][:, keep])
        train_fraction, _, train_loading, _ = pca_stats_from_z(z_train)
        test_fraction, _, test_loading, _ = pca_stats_from_z(z_test)
        rows.append(
            {
                "mode": mode,
                "iteration": iteration,
                "n_train_trials": int(train.sum()),
                "n_test_trials": int(test.sum()),
                "n_test_groups": n_test,
                "n_units": int(keep.sum()),
                "train_pc1_fraction": train_fraction,
                "test_pc1_fraction": test_fraction,
                "train_test_pc1_loading_pearson": safe_corr(train_loading, test_loading),
                "train_test_pc1_loading_spearman": safe_corr(train_loading, test_loading, "spearman"),
            }
        )
    return pd.DataFrame(rows)


def validation_summary(
    levels: dict[str, dict[str, np.ndarray]],
    rank_arrays: dict[str, dict[str, np.ndarray]],
    n_split_half: int,
    n_group_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED + 1)
    summary_rows = []
    iteration_tables = []
    saved = {}
    for matrix_name, content in levels.items():
        for level_name in ("trial", "trial_within_session"):
            z = rank_arrays[matrix_name][f"{level_name}_z"]
            p, s = split_half_scores(z, n_split_half, rng)
            summary_rows.append(
                {
                    "matrix": matrix_name,
                    "validation": f"unit_split_half_{level_name}",
                    "pearson_median": float(np.median(p)),
                    "pearson_ci95_low": float(np.quantile(p, 0.025)),
                    "pearson_ci95_high": float(np.quantile(p, 0.975)),
                    "spearman_median": float(np.median(s)),
                }
            )
            saved[f"{matrix_name}_{level_name}_split_half_pearson"] = p
            saved[f"{matrix_name}_{level_name}_split_half_spearman"] = s
        for mode in ("trial_disjoint", "session_disjoint"):
            table = grouped_loading_splits(content["trial"], content["trial_sessions"], mode, n_group_splits, rng)
            table.insert(0, "matrix", matrix_name)
            iteration_tables.append(table)
            summary_rows.append(
                {
                    "matrix": matrix_name,
                    "validation": f"pc1_loading_{mode}",
                    "pearson_median": float(table["train_test_pc1_loading_pearson"].median()),
                    "pearson_ci95_low": float(table["train_test_pc1_loading_pearson"].quantile(0.025)),
                    "pearson_ci95_high": float(table["train_test_pc1_loading_pearson"].quantile(0.975)),
                    "spearman_median": float(table["train_test_pc1_loading_spearman"].median()),
                }
            )
    return pd.DataFrame(summary_rows), pd.concat(iteration_tables, ignore_index=True), saved


def influence_audit(
    levels: dict[str, dict[str, np.ndarray]], rank_arrays: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_rows = []
    trial_rows = []
    for matrix_name, content in levels.items():
        trial = content["trial"]
        trial_ids = content["trial_ids"]
        sessions = content["trial_sessions"]
        full_fraction = pca_stats_from_z(standardize_columns(trial))[0]
        full_score = rank_arrays[matrix_name]["trial_score"]
        total_energy = np.sum(full_score**2)
        for session in np.unique(sessions):
            removed = sessions == session
            candidate = trial[~removed]
            keep = candidate.std(axis=0) > EPS
            fraction = pca_stats_from_z(standardize_columns(candidate[:, keep]))[0]
            session_rows.append(
                {
                    "matrix": matrix_name,
                    "session": session,
                    "n_trials_removed": int(removed.sum()),
                    "full_trial_pc1_fraction": full_fraction,
                    "leave_session_out_pc1_fraction": fraction,
                    "change_from_full": fraction - full_fraction,
                    "fraction_of_full_pc1_score_squared_energy": float(np.sum(full_score[removed] ** 2) / total_energy),
                }
            )
        for row, trial_id in enumerate(trial_ids):
            keep_rows = np.arange(len(trial)) != row
            candidate = trial[keep_rows]
            keep_units = candidate.std(axis=0) > EPS
            fraction = pca_stats_from_z(standardize_columns(candidate[:, keep_units]))[0]
            trial_rows.append(
                {
                    "matrix": matrix_name,
                    "source_trial_group": trial_id,
                    "session": sessions[row],
                    "full_trial_pc1_fraction": full_fraction,
                    "leave_trial_out_pc1_fraction": fraction,
                    "change_from_full": fraction - full_fraction,
                    "fraction_of_full_pc1_score_squared_energy": float(full_score[row] ** 2 / total_energy),
                }
            )
    return pd.DataFrame(session_rows), pd.DataFrame(trial_rows)


def matrix_alignment(levels: dict[str, dict[str, np.ndarray]], rank_arrays: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for level_name in ("window", "trial", "trial_within_session", "session"):
        a = levels["paired"][level_name]
        b = levels["matched"][level_name]
        score_a = rank_arrays["paired"][f"{level_name}_score"]
        score_b = rank_arrays["matched"][f"{level_name}_score"]
        loading_a = rank_arrays["paired"][f"{level_name}_loading"]
        loading_b = rank_arrays["matched"][f"{level_name}_loading"]
        rows.append(
            {
                "level": level_name,
                "all_entries_pearson": safe_corr(a.ravel(), b.ravel()),
                "population_mean_pearson": safe_corr(a.mean(axis=1), b.mean(axis=1)),
                "population_mean_spearman": safe_corr(a.mean(axis=1), b.mean(axis=1), "spearman"),
                "pc1_score_pearson": safe_corr(score_a, score_b),
                "pc1_score_spearman": safe_corr(score_a, score_b, "spearman"),
                "pc1_loading_pearson": safe_corr(loading_a, loading_b),
                "pc1_loading_spearman": safe_corr(loading_a, loading_b, "spearman"),
            }
        )
    return pd.DataFrame(rows)


def matched_trace_reliability(samples: np.ndarray, responsive: np.ndarray) -> pd.DataFrame:
    from itertools import combinations

    rows = []
    for first_indices in combinations(range(8), 4):
        if 0 not in first_indices:
            continue
        second_indices = tuple(value for value in range(8) if value not in first_indices)
        first = samples[:, first_indices][:, :, responsive].mean(axis=1)
        second = samples[:, second_indices][:, :, responsive].mean(axis=1)
        rows.append(
            {
                "first_samples": ",".join(map(str, first_indices)),
                "second_samples": ",".join(map(str, second_indices)),
                "all_entries_pearson": safe_corr(first.ravel(), second.ravel()),
                "window_population_mean_pearson": safe_corr(first.mean(axis=1), second.mean(axis=1)),
                "window_population_mean_spearman": safe_corr(first.mean(axis=1), second.mean(axis=1), "spearman"),
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    rank: pd.DataFrame,
    uncertainty: pd.DataFrame,
    validation: pd.DataFrame,
    group_iterations: pd.DataFrame,
    alignment: pd.DataFrame,
    session_influence: pd.DataFrame,
    levels: dict[str, dict[str, np.ndarray]],
    rank_arrays: dict[str, dict[str, np.ndarray]],
    out: Path,
    dpi: int,
) -> None:
    colors = {"paired": "#d62828", "matched": "#457b9d", "pairing_residual": "#6a4c93"}
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 9.8), constrained_layout=True)

    ax = axes[0, 0]
    level_order = ["window", "trial", "trial_within_session", "session"]
    x = np.arange(len(level_order))
    for offset, matrix_name in zip((-0.24, 0, 0.24), MATRIX_LABELS):
        block = rank.loc[rank["matrix"].eq(matrix_name)].set_index("level").loc[level_order]
        ax.bar(x + offset, block["centered_per_unit_zscored_pc1_variance_fraction"], width=0.22, color=colors[matrix_name], label=MATRIX_LABELS[matrix_name])
    ax.set_xticks(x, ["384\nwindows", "221\ntrials", "within-session\ntrials", "29\nsessions"])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("PC1 variance fraction")
    ax.set_title("A  Shared unit ordering survives trial aggregation", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for matrix_name in MATRIX_LABELS:
        u = uncertainty.set_index("matrix").loc[matrix_name]
        center = u["trial_pc1_observed"]
        low = u["session_cluster_bootstrap_pc1_ci95_low"]
        high = u["session_cluster_bootstrap_pc1_ci95_high"]
        xpos = list(MATRIX_LABELS).index(matrix_name)
        ax.errorbar(xpos, center, yerr=[[center-low], [high-center]], fmt="o", color=colors[matrix_name], capsize=5, ms=8)
        ax.scatter(xpos, u["within_session_trial_shuffle_null_ci99_high"], marker="_", s=260, color="black")
    ax.set_xticks(range(3), ["own", "matched", "residual"])
    ax.set_ylabel("Trial-level PC1 fraction")
    ax.set_title("B  Session-cluster uncertainty and conservative null", loc="left", fontweight="bold")
    ax.text(0.02, 0.96, "dots: observed · bars: session-bootstrap 95% CI\nblack ticks: within-session shuffle 99th percentile", transform=ax.transAxes, va="top", fontsize=8)

    ax = axes[0, 2]
    positions = []
    values = []
    box_colors = []
    labels = []
    pos = 0
    for mode in ("trial_disjoint", "session_disjoint"):
        for matrix_name in MATRIX_LABELS:
            block = group_iterations.loc[group_iterations["matrix"].eq(matrix_name) & group_iterations["mode"].eq(mode), "train_test_pc1_loading_pearson"]
            values.append(block.to_numpy())
            positions.append(pos)
            box_colors.append(colors[matrix_name])
            labels.append(f"{mode.split('_')[0]}\n{matrix_name.replace('pairing_', '')}")
            pos += 1
        pos += 0.5
    bp = ax.boxplot(values, positions=positions, widths=0.65, showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("Train–held-out PC1 loading Pearson r")
    ax.set_title("C  Unit loading pattern transfers to held-out groups", loc="left", fontweight="bold")

    ax = axes[1, 0]
    for matrix_name in MATRIX_LABELS:
        for level_name, linestyle in (("trial", "-"), ("trial_within_session", "--")):
            row = validation.loc[validation["matrix"].eq(matrix_name) & validation["validation"].eq(f"unit_split_half_{level_name}")].iloc[0]
            xpos = list(MATRIX_LABELS).index(matrix_name) + (-0.08 if level_name == "trial" else 0.08)
            ax.errorbar(xpos, row["pearson_median"], yerr=[[row["pearson_median"]-row["pearson_ci95_low"]], [row["pearson_ci95_high"]-row["pearson_median"]]], fmt="o", color=colors[matrix_name], mfc=(colors[matrix_name] if level_name == "trial" else "white"), capsize=4)
    ax.set_xticks(range(3), ["own", "matched", "residual"])
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel("Correlation between random unit halves")
    ax.set_title("D  Independent unit halves recover trial ordering", loc="left", fontweight="bold")
    ax.text(0.02, 0.08, "filled: trial profiles\nopen: session means removed", transform=ax.transAxes, fontsize=8)

    ax = axes[1, 1]
    a = rank_arrays["paired"]["trial_score"]
    b = rank_arrays["matched"]["trial_score"]
    ax.scatter(b, a, s=18, alpha=0.6, color="#355070", edgecolors="none")
    rho = alignment.set_index("level").loc["trial", "pc1_score_spearman"]
    ax.set_xlabel("Matched-trace PC1 trial score")
    ax.set_ylabel("Own-trace PC1 trial score")
    ax.set_title(f"E  Own and matched traces rank trials similarly\nSpearman ρ = {rho:.2f}", loc="left", fontweight="bold")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)

    ax = axes[1, 2]
    for matrix_name in MATRIX_LABELS:
        block = session_influence.loc[session_influence["matrix"].eq(matrix_name)].sort_values("leave_session_out_pc1_fraction")
        ax.plot(np.arange(len(block)), block["leave_session_out_pc1_fraction"], marker="o", ms=3, lw=1, color=colors[matrix_name], label=MATRIX_LABELS[matrix_name])
    ax.set_xlabel("Sessions ordered separately within each curve")
    ax.set_ylabel("PC1 fraction after leaving out one session")
    ax.set_title("F  No single session determines the rank result", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.grid(color="0.93", zorder=0)
    fig.suptitle(
        "Full cached movie bank: common RR100 modulation structure generalizes beyond the original 16 pairs",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrices, metadata, samples = load_inputs(args.input_dir)
    mean_reference = 0.5 * (matrices["paired"].mean(axis=0) + matrices["matched"].mean(axis=0))
    responsive = mean_reference > RESPONSIVE_THRESHOLD_HZ
    levels, trial_table = build_levels(matrices, metadata, responsive)
    rank, rank_arrays = rank_summary(levels)
    uncertainty, uncertainty_arrays = rank_uncertainty(levels, rank, args.n_null, args.n_bootstrap)
    validation, group_iterations, validation_arrays = validation_summary(
        levels, rank_arrays, args.n_split_half, args.n_group_splits
    )
    session_influence, trial_influence = influence_audit(levels, rank_arrays)
    alignment = matrix_alignment(levels, rank_arrays)
    matched_reliability = matched_trace_reliability(samples, responsive)

    rank.to_csv(args.out_dir / "rank_summary_by_condition_level.csv", index=False)
    uncertainty.to_csv(args.out_dir / "rank_null_and_cluster_bootstrap_summary.csv", index=False)
    validation.to_csv(args.out_dir / "grouped_validation_summary.csv", index=False)
    group_iterations.to_csv(args.out_dir / "grouped_loading_split_iterations.csv", index=False)
    session_influence.to_csv(args.out_dir / "leave_one_session_out_influence.csv", index=False)
    trial_influence.to_csv(args.out_dir / "leave_one_trial_out_influence.csv", index=False)
    alignment.to_csv(args.out_dir / "paired_matched_alignment.csv", index=False)
    matched_reliability.to_csv(args.out_dir / "matched_trace_split_half_reliability.csv", index=False)
    trial_table.to_csv(args.out_dir / "source_trial_rows.csv", index=False)
    np.savez_compressed(
        args.out_dir / "rank_validation_arrays.npz",
        responsive_rr100_mask=responsive,
        **uncertainty_arrays,
        **validation_arrays,
        **{
            f"{matrix_name}_{key}": value
            for matrix_name, arrays in rank_arrays.items()
            for key, value in arrays.items()
            if isinstance(value, np.ndarray)
        },
    )
    make_figure(
        rank, uncertainty, validation, group_iterations, alignment, session_influence,
        levels, rank_arrays, args.out_dir / "fullbank_rank_and_grouped_validation", args.dpi,
    )

    rank_idx = rank.set_index(["matrix", "level"])
    uncertainty_idx = uncertainty.set_index("matrix")
    validation_idx = validation.set_index(["matrix", "validation"])
    influence_ranges = session_influence.groupby("matrix")["leave_session_out_pc1_fraction"].agg(["min", "max"])
    paired_trial = rank_idx.loc[("paired", "trial")]
    paired_within = rank_idx.loc[("paired", "trial_within_session")]
    paired_uncertainty = uncertainty_idx.loc["paired"]
    paired_split = validation_idx.loc[("paired", "unit_split_half_trial")]
    paired_split_within = validation_idx.loc[("paired", "unit_split_half_trial_within_session")]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "full cached bank grouped rank and influence audit",
        "input_checkpoint": str(args.input_dir.resolve()),
        "response_metric": "RMS in first four zero-mean temporal DCT components of FEM-minus-static response",
        "metric_scope": "validated ordering proxy for low-frequency temporal modulation; not full temporal SD",
        "n_windows": 384,
        "n_source_trials": int(trial_table.shape[0]),
        "n_sessions": int(trial_table["session"].nunique()),
        "n_rr100_units": 100,
        "n_responsive_units": int(responsive.sum()),
        "responsive_threshold_hz": RESPONSIVE_THRESHOLD_HZ,
        "primary_paired_trial_pc1_fraction": float(paired_trial["centered_per_unit_zscored_pc1_variance_fraction"]),
        "primary_paired_trial_median_pairwise_unit_spearman": float(paired_trial["median_pairwise_unit_profile_spearman"]),
        "paired_within_session_trial_pc1_fraction": float(paired_within["centered_per_unit_zscored_pc1_variance_fraction"]),
        "paired_within_session_trial_median_pairwise_unit_spearman": float(paired_within["median_pairwise_unit_profile_spearman"]),
        "paired_global_trial_shuffle_p_upper": float(paired_uncertainty["global_trial_shuffle_p_upper"]),
        "paired_within_session_trial_shuffle_p_upper": float(paired_uncertainty["within_session_trial_shuffle_p_upper"]),
        "paired_session_cluster_bootstrap_pc1_ci95": [
            float(paired_uncertainty["session_cluster_bootstrap_pc1_ci95_low"]),
            float(paired_uncertainty["session_cluster_bootstrap_pc1_ci95_high"]),
        ],
        "paired_unit_split_half_trial_score_pearson_median": float(paired_split["pearson_median"]),
        "paired_unit_split_half_within_session_score_pearson_median": float(paired_split_within["pearson_median"]),
        "paired_vs_matched_trial_pc1_score_spearman": float(alignment.set_index("level").loc["trial", "pc1_score_spearman"]),
        "paired_vs_matched_trial_pc1_loading_pearson": float(alignment.set_index("level").loc["trial", "pc1_loading_pearson"]),
        "matched_trace_half_window_population_mean_pearson_median": float(matched_reliability["window_population_mean_pearson"].median()),
        "leave_one_session_out_pc1_ranges": influence_ranges.to_dict(orient="index"),
        "scope_statement": (
            "Within this frozen RR100 model and cached BackImage bank, low-frequency FEM-modulation magnitude has a common "
            "ordering across units over 221 source trials. The ordering remains after removing session means and when "
            "entire trials or sessions are held out."
        ),
        "unsupported": [
            "generalization to another model seed or experimental neural population",
            "equivalence of the four-DCT proxy to total temporal modulation magnitude",
            "a causal retinal-power mechanism",
            "additive or multiplicative gain",
            "independence of image and trajectory effects; paired-minus-matched residual retains structure",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# RR100 full-bank rank checkpoint 13B\n\n"
        "This checkpoint follows the visual map checkpoint 13A. It treats 221 session-by-trial groups as the condition "
        "observations rather than treating 384 windows as independent. The primary statistic is centered PC1 variance "
        "after each responsive RR100 unit is standardized across source trials.\n\n"
        "Paired own-trace modulation, the mean magnitude under eight matched other-trial traces, and their signed residual "
        "are analyzed separately. Trial-shuffle nulls, source-trial and session-cluster bootstraps, random unit-half score "
        "reliability, trial-disjoint and session-disjoint PC1-loading replication, within-session demeaning, and leave-one-"
        "session-out influence checks are all saved as separate tables.\n"
    )
    print(json.dumps(manifest, indent=2))
    print("\nRANK\n", rank.to_string(index=False))
    print("\nVALIDATION\n", validation.to_string(index=False))


if __name__ == "__main__":
    main()
