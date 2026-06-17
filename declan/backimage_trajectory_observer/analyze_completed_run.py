"""Post-hoc analysis for completed BackImage trajectory-table observer runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURES = [
    ("image_patch_rms_contrast", "contrast"),
    ("image_gradient_energy", "gradient_energy"),
    ("image_edge_density", "edge_density"),
    ("image_orientation_coherence", "orientation_coherence"),
    ("image_high_freq_power_fraction", "high_freq_power"),
    ("image_power_8plus_cpd_fraction", "power_8plus_cpd"),
    ("contrast_distance_to_nearest_distractor", "nearest_distractor_contrast_distance"),
    ("structure_distance_to_nearest_distractor", "nearest_distractor_structure_distance"),
    ("static_response_distance_to_nearest_distractor", "nearest_distractor_static_response_distance"),
    ("mean_rate_distance_to_nearest_distractor", "nearest_distractor_mean_rate_distance"),
]


CONDITION_KEYS = ["candidate_set_mode", "observation_scale", "prior_family", "likelihood_scale"]


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _finite_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _finite_series(x), "y": _finite_series(y)}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _pearson(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": _finite_series(x), "y": _finite_series(y)}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].corr(frame["y"]))


def _safe_recovery_fraction(known: float, zero: float, joint: float) -> float:
    denom = known - zero
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        return float("nan")
    return float((joint - zero) / denom)


def _condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(CONDITION_KEYS, dropna=False):
        known = float(grp["known_correct_i"].mean())
        zero = float(grp["zero_correct_i"].mean())
        joint = float(grp["joint_correct_i"].mean())
        best_single = float(grp["best_single_tau_correct_i"].mean())
        rows.append(
            {
                "candidate_set_mode": key[0],
                "observation_scale": key[1],
                "prior_family": key[2],
                "likelihood_scale": key[3],
                "n_trials": int(len(grp)),
                "known_accuracy": known,
                "zero_accuracy": zero,
                "joint_accuracy": joint,
                "best_single_tau_accuracy": best_single,
                "joint_minus_zero_accuracy": joint - zero,
                "known_minus_zero_accuracy": known - zero,
                "joint_recovery_fraction_of_known_zero_gap": _safe_recovery_fraction(known, zero, joint),
                "median_N_eff_fraction": float(grp["N_eff_true_image_fraction"].median()),
                "median_nearest_tau_rank": float(grp["nearest_tau_rank"].median()),
                "median_joint_vs_best_single_tau_gap": float(grp["joint_vs_best_single_tau_gap"].median()),
                "median_joint_minus_zero_true_score": float(grp["joint_minus_zero_true_score"].median()),
                "median_joint_true_margin": float(grp["joint_true_margin"].median()),
                "median_zero_true_margin": float(grp["zero_true_margin"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(CONDITION_KEYS)


def _feature_bin_summary(df: pd.DataFrame, features: Iterable[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for col, label in features:
        if col not in df.columns:
            continue
        values = _finite_series(df[col])
        if values.notna().sum() < 8 or values.nunique(dropna=True) < 4:
            rows.append({"feature": label, "feature_column": col, "status": "insufficient_finite_values"})
            continue
        try:
            bins = pd.qcut(values, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        except ValueError:
            rows.append({"feature": label, "feature_column": col, "status": "could_not_bin"})
            continue
        tmp = df.copy()
        tmp["_feature_bin"] = bins
        tmp["_feature_value"] = values
        for key, cond in tmp.groupby(CONDITION_KEYS, dropna=False):
            for bin_name, grp in cond.groupby("_feature_bin", observed=True):
                if len(grp) == 0:
                    continue
                zero = float(grp["zero_correct_i"].mean())
                joint = float(grp["joint_correct_i"].mean())
                known = float(grp["known_correct_i"].mean())
                rows.append(
                    {
                        "feature": label,
                        "feature_column": col,
                        "status": "ok",
                        "candidate_set_mode": key[0],
                        "observation_scale": key[1],
                        "prior_family": key[2],
                        "likelihood_scale": key[3],
                        "feature_bin": str(bin_name),
                        "n_trials": int(len(grp)),
                        "feature_min": float(grp["_feature_value"].min()),
                        "feature_median": float(grp["_feature_value"].median()),
                        "feature_max": float(grp["_feature_value"].max()),
                        "known_accuracy": known,
                        "zero_accuracy": zero,
                        "joint_accuracy": joint,
                        "joint_minus_zero_accuracy": joint - zero,
                        "joint_recovery_fraction_of_known_zero_gap": _safe_recovery_fraction(known, zero, joint),
                        "median_joint_minus_zero_true_score": float(grp["joint_minus_zero_true_score"].median()),
                        "median_N_eff_fraction": float(grp["N_eff_true_image_fraction"].median()),
                    }
                )
    return pd.DataFrame(rows)


def _correlation_summary(df: pd.DataFrame, features: Iterable[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(CONDITION_KEYS, dropna=False):
        for col, label in features:
            if col not in grp.columns:
                continue
            rows.append(
                {
                    "candidate_set_mode": key[0],
                    "observation_scale": key[1],
                    "prior_family": key[2],
                    "likelihood_scale": key[3],
                    "feature": label,
                    "feature_column": col,
                    "n_finite": int(pd.DataFrame({"x": _finite_series(grp[col]), "y": grp["joint_minus_zero_true_score"]}).dropna().shape[0]),
                    "spearman_feature_vs_joint_minus_zero_score": _spearman(grp[col], grp["joint_minus_zero_true_score"]),
                    "pearson_feature_vs_joint_minus_zero_score": _pearson(grp[col], grp["joint_minus_zero_true_score"]),
                    "spearman_feature_vs_N_eff_fraction": _spearman(grp[col], grp["N_eff_true_image_fraction"]),
                    "pearson_feature_vs_N_eff_fraction": _pearson(grp[col], grp["N_eff_true_image_fraction"]),
                    "spearman_feature_vs_joint_zero_correct_delta": _spearman(grp[col], grp["joint_zero_correct_delta"]),
                    "spearman_N_eff_fraction_vs_joint_minus_zero_score": _spearman(
                        grp["N_eff_true_image_fraction"], grp["joint_minus_zero_true_score"]
                    ),
                    "pearson_N_eff_fraction_vs_joint_minus_zero_score": _pearson(
                        grp["N_eff_true_image_fraction"], grp["joint_minus_zero_true_score"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _error_case_summary(df: pd.DataFrame, features: Iterable[tuple[str, str]]) -> pd.DataFrame:
    tmp = df.copy()
    tmp["error_case"] = np.select(
        [
            (tmp["zero_correct_i"] == 0) & (tmp["joint_correct_i"] == 1),
            (tmp["zero_correct_i"] == 1) & (tmp["joint_correct_i"] == 0),
            (tmp["zero_correct_i"] == 0) & (tmp["joint_correct_i"] == 0),
            (tmp["zero_correct_i"] == 1) & (tmp["joint_correct_i"] == 1),
        ],
        ["rescued_by_joint", "lost_by_joint", "both_wrong", "both_correct"],
        default="other",
    )
    rows = []
    for key, grp in tmp.groupby(CONDITION_KEYS + ["error_case"], dropna=False):
        row = {
            "candidate_set_mode": key[0],
            "observation_scale": key[1],
            "prior_family": key[2],
            "likelihood_scale": key[3],
            "error_case": key[4],
            "n_trials": int(len(grp)),
            "median_N_eff_fraction": float(grp["N_eff_true_image_fraction"].median()),
            "median_nearest_tau_rank": float(grp["nearest_tau_rank"].median()),
            "median_joint_minus_zero_true_score": float(grp["joint_minus_zero_true_score"].median()),
        }
        for col, label in features:
            if col in grp.columns:
                vals = _finite_series(grp[col])
                row[f"median_{label}"] = float(vals.median()) if vals.notna().any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(CONDITION_KEYS + ["error_case"])


def _plot_outputs(df: pd.DataFrame, condition: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = sorted(condition["observation_scale"].unique())
    primary_mode = "matched_static_response" if "matched_static_response" in set(condition["candidate_set_mode"]) else str(condition["candidate_set_mode"].iloc[0])
    condition_plot = condition[condition["candidate_set_mode"] == primary_mode]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, like in zip(axes, sorted(condition_plot["likelihood_scale"].unique()), strict=False):
        sub = condition_plot[condition_plot["likelihood_scale"] == like]
        known_by_scale = sub.groupby("observation_scale")["known_accuracy"].mean()
        zero_by_scale = sub.groupby("observation_scale")["zero_accuracy"].mean()
        ax.plot(scales, [known_by_scale.get(s, np.nan) for s in scales], marker="o", label="known-eye", color="black")
        ax.plot(scales, [zero_by_scale.get(s, np.nan) for s in scales], marker="o", label="zero-eye", color="gray")
        for prior, color in [("empirical", "#1f77b4"), ("ou", "#d62728")]:
            vals = sub[sub["prior_family"] == prior].set_index("observation_scale")["joint_accuracy"]
            ax.plot(scales, [vals.get(s, np.nan) for s in scales], marker="o", label=f"joint {prior}", color=color)
        ax.set_title(f"likelihood scale {like:g}")
        ax.set_xlabel("motion scale")
        ax.set_ylim(0.2, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("image identity accuracy")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_scale.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("median_N_eff_fraction", "median N_eff / K"),
        ("median_nearest_tau_rank", "median nearest tau rank"),
        ("median_joint_vs_best_single_tau_gap", "median joint vs best-single-tau gap"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        for (prior, like), sub in condition_plot.groupby(["prior_family", "likelihood_scale"]):
            vals = sub.set_index("observation_scale")[metric]
            ax.plot(scales, [vals.get(s, np.nan) for s in scales], marker="o", label=f"{prior}, like {like:g}")
        ax.set_xlabel("motion scale")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "posterior_diagnostics_vs_scale.png", dpi=180)
    plt.close(fig)

    primary = df[
        (df["candidate_set_mode"] == primary_mode)
        & (df["observation_scale"] == 1.0)
        & (df["likelihood_scale"] == 1.0)
    ].copy()
    plot_features = [
        ("image_patch_rms_contrast", "contrast"),
        ("image_edge_density", "edge density"),
        ("image_orientation_coherence", "orientation coherence"),
        ("image_high_freq_power_fraction", "high frequency power"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, title) in zip(axes.ravel(), plot_features, strict=True):
        if col not in primary.columns:
            ax.set_axis_off()
            continue
        for prior, color in [("empirical", "#1f77b4"), ("ou", "#d62728")]:
            sub = primary[primary["prior_family"] == prior]
            ax.scatter(_finite_series(sub[col]), sub["joint_minus_zero_true_score"], s=22, alpha=0.65, label=prior, color=color)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("joint - zero true score")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_zero_score_gain_vs_image_features_scale1_like1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for prior, color in [("empirical", "#1f77b4"), ("ou", "#d62728")]:
        sub = primary[primary["prior_family"] == prior]
        ax.scatter(sub["N_eff_true_image_fraction"], sub["joint_minus_zero_true_score"], s=26, alpha=0.7, label=prior, color=color)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("N_eff / K")
    ax.set_ylabel("joint - zero true score")
    ax.set_title("Pose posterior concentration vs joint-zero score gain, 1.0x")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "neff_fraction_vs_joint_zero_score_gain_scale1_like1.png", dpi=180)
    plt.close(fig)


def _write_report(run_dir: Path, out_dir: Path, condition: pd.DataFrame, corr: pd.DataFrame, features: list[tuple[str, str]]) -> None:
    primary_mode = "matched_static_response" if "matched_static_response" in set(condition["candidate_set_mode"]) else str(condition["candidate_set_mode"].iloc[0])
    unavailable = []
    for col, label in features:
        if col not in corr["feature_column"].unique() or corr.loc[corr["feature_column"] == col, "n_finite"].max() < 8:
            unavailable.append(f"- `{label}` (`{col}`)")

    scale_lines = []
    for (mode, scale), grp in condition.groupby(["candidate_set_mode", "observation_scale"]):
        scale_lines.append(
            "- `{mode}`, `{scale:g}x`: zero `{zero:.3f}`, joint range `{jmin:.3f}-{jmax:.3f}`, joint-zero range `{dmin:+.3f}-{dmax:+.3f}`".format(
                mode=str(mode),
                scale=float(scale),
                zero=float(grp["zero_accuracy"].mean()),
                jmin=float(grp["joint_accuracy"].min()),
                jmax=float(grp["joint_accuracy"].max()),
                dmin=float(grp["joint_minus_zero_accuracy"].min()),
                dmax=float(grp["joint_minus_zero_accuracy"].max()),
            )
        )

    primary_corr = corr[
        (corr["candidate_set_mode"] == primary_mode)
        & (corr["observation_scale"] == 1.0)
        & (corr["likelihood_scale"] == 1.0)
    ].copy()
    primary_corr = primary_corr.sort_values("spearman_feature_vs_joint_minus_zero_score", key=lambda s: s.abs(), ascending=False)
    corr_lines = []
    for _, row in primary_corr.head(8).iterrows():
        corr_lines.append(
            "- `{feature}` / `{prior}`: rho(feature, joint-zero score) = `{rho:.3f}`, rho(feature, N_eff/K) = `{rho_neff:.3f}`".format(
                feature=row["feature"],
                prior=row["prior_family"],
                rho=float(row["spearman_feature_vs_joint_minus_zero_score"])
                if np.isfinite(row["spearman_feature_vs_joint_minus_zero_score"])
                else float("nan"),
                rho_neff=float(row["spearman_feature_vs_N_eff_fraction"])
                if np.isfinite(row["spearman_feature_vs_N_eff_fraction"])
                else float("nan"),
            )
        )

    text = [
        "# Option C Image/Condition Post-Hoc Analysis",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Scale Pattern",
        "",
        *scale_lines,
        "",
        "This is a pose-uncertainty rescue pattern, not evidence that 1.0x motion",
        "contains the most image information. The 1.0x improvement is mainly driven",
        "by zero-eye falling sharply while joint-eye remains comparatively robust.",
        "Joint accuracy does decline modestly from the 0.25x condition, so the",
        "effect is not only zero collapse, but the dominant contrast is the widened",
        "zero-to-joint gap.",
        "",
        "## Image-Structure Correlations",
        "",
        "The CSV outputs contain full per-condition feature-bin and correlation",
        "tables. The strongest absolute Spearman relationships for the primary",
        f"`{primary_mode}`, `1.0x`, likelihood-scale `1.0` slice are:",
        "",
        *(corr_lines or ["- No finite correlations available."]),
        "",
        "Interpret these raw image-feature correlations as exploratory. The run has",
        "only 64 unique windows and repeated rows across candidate modes/priors/likelihood settings.",
        "Likelihood-derived structure metrics, especially posterior concentration",
        "and trajectory separability under the true image, should carry the main",
        "mechanistic interpretation.",
        "",
        "## Unavailable Diagnostics",
        "",
        *(unavailable or ["- None."]),
        "",
        (
            "`matched_static_response` is present; static-response nearest-distractor "
            "distances are available for that candidate mode."
            if "matched_static_response" in set(condition["candidate_set_mode"])
            else "`matched_static_response` was not used in this run, so static-response "
            "nearest-distractor distances are unavailable. That remains the key "
            "candidate-control addition for the confirmatory run."
        ),
        "",
        "## Outputs",
        "",
        "- `condition_summary.csv`",
        "- `feature_bin_summary.csv`",
        "- `feature_correlation_summary.csv`",
        "- `error_case_summary.csv`",
        "- `accuracy_vs_scale.png`",
        "- `posterior_diagnostics_vs_scale.png`",
        "- `joint_zero_score_gain_vs_image_features_scale1_like1.png`",
        "- `neff_fraction_vs_joint_zero_score_gain_scale1_like1.png`",
    ]
    (out_dir / "image_condition_analysis_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def analyze_run(run_dir: Path, out_dir: Path | None = None) -> None:
    run_dir = Path(run_dir)
    out_dir = run_dir / "posthoc_image_condition_analysis" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(run_dir / "observer_trials.csv")
    windows = pd.read_csv(run_dir / "selected_windows.csv")
    candidates = pd.read_csv(run_dir / "candidate_sets.csv")

    for col in ["known_correct", "zero_correct", "joint_correct", "best_trajectory_oracle_correct"]:
        trials[f"{col}_i"] = _bool_series(trials[col]).astype(int)
    trials["best_single_tau_correct_i"] = trials["best_trajectory_oracle_correct_i"]
    trials["joint_zero_correct_delta"] = trials["joint_correct_i"] - trials["zero_correct_i"]

    numeric_cols = [
        "observation_scale",
        "likelihood_scale",
        "N_eff_true_image_fraction",
        "nearest_tau_rank",
        "joint_vs_best_dilution_gap",
        "joint_minus_zero_true_score",
        "joint_true_margin",
        "zero_true_margin",
    ]
    for col in numeric_cols:
        if col in trials.columns:
            trials[col] = _finite_series(trials[col])
    trials["joint_vs_best_single_tau_gap"] = _finite_series(trials.get("joint_vs_best_single_tau_gap", trials["joint_vs_best_dilution_gap"]))

    keep_window_cols = ["source_row"] + [col for col, _ in FEATURES if col in windows.columns]
    merged = trials.merge(
        windows[keep_window_cols].drop_duplicates("source_row"),
        left_on="observation_source_row",
        right_on="source_row",
        how="left",
        suffixes=("", "_window"),
    )
    candidate_keep = [
        "trial_id",
        "structure_distance_to_nearest_distractor",
        "contrast_distance_to_nearest_distractor",
    ]
    candidate_keep = [col for col in candidate_keep if col in candidates.columns]
    if len(candidate_keep) > 1:
        merged = merged.merge(
            candidates[candidate_keep].drop_duplicates("trial_id"),
            on="trial_id",
            how="left",
            suffixes=("", "_candidate"),
        )
        for col in ["structure_distance_to_nearest_distractor", "contrast_distance_to_nearest_distractor"]:
            alt = f"{col}_candidate"
            if alt in merged.columns:
                merged[col] = _finite_series(merged[col]).fillna(_finite_series(merged[alt]))

    condition = _condition_summary(merged)
    feature_bins = _feature_bin_summary(merged, FEATURES)
    corr = _correlation_summary(merged, FEATURES)
    errors = _error_case_summary(merged, FEATURES)

    merged.to_csv(out_dir / "trial_feature_join.csv", index=False)
    condition.to_csv(out_dir / "condition_summary.csv", index=False)
    feature_bins.to_csv(out_dir / "feature_bin_summary.csv", index=False)
    corr.to_csv(out_dir / "feature_correlation_summary.csv", index=False)
    errors.to_csv(out_dir / "error_case_summary.csv", index=False)

    try:
        _plot_outputs(merged, condition, out_dir)
    except Exception as exc:  # pragma: no cover - plotting is best-effort.
        (out_dir / "plot_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    _write_report(run_dir, out_dir, condition, corr, FEATURES)
    print(f"Wrote post-hoc analysis to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    analyze_run(args.run_dir, args.out_dir)


if __name__ == "__main__":
    main()
