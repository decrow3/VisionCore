"""Compare along- versus across-contour axis priors with the joint estimator.

This is a cache-only diagnostic for asking whether the promoted continuous
joint observer behaves like the Figure 4D along/across readout. It re-scores the
strict no-start scale-prior run in the same posterior-weighted feature-cosine
metric used for Panel C, then computes paired axis contrasts:

    axis_edge_parallel - axis_edge_orthogonal

The observed response family in this cache is empirical; the axis labels refer
to the image-conditioned trajectory prior/catalog family used by the observer.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.figure4_active_sensing_atlas.scripts.build_panel_c_continuous_joint_feature_recovery import (
    PRIMARY_LATENT,
    _load_feature_tables,
    _vectorized_mode_rows,
)
from declan.figure4_active_sensing_atlas.scripts.run_panel_c_promoted_continuous_joint_observer import (
    DEFAULT_OUT_DIR,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "declan" / "figure4_active_sensing_atlas" / "figures" / "panel_C" / "diagnostics"
CONTINUOUS_OUT_DIR = OUT_DIR / "continuous_joint"
POSTERIOR_CSV = DEFAULT_OUT_DIR / "continuous_joint_feature_posterior.csv"
TRIALS_CSV = DEFAULT_OUT_DIR / "continuous_joint_trials.csv"
N_BOOT = 10_000
RNG_SEED = 20260624

AXIS_LABELS = {
    "axis_edge_parallel": "along contour",
    "axis_edge_orthogonal": "across contour",
}
AXIS_COLORS = {
    "axis_edge_parallel": "#2f8f6a",
    "axis_edge_orthogonal": "#8a5ca8",
}
OBSERVER_LABELS = {
    "zero": "zero-eye",
    "continuous_joint": "joint estimator",
    "known": "known-eye",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9dee5", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _scale_x(scales: pd.Series) -> np.ndarray:
    return scales.astype(float).map({0.5: 0.0, 1.0: 1.0, 2.0: 2.0}).to_numpy()


def _format_scale(value: float | str) -> str:
    x = float(value)
    if x == 1.0:
        return "1x"
    return f"{x:g}x"


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(arr, size=(N_BOOT, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _sign_flip_p(values: np.ndarray, rng: np.random.Generator) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    observed = abs(float(arr.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(N_BOOT, arr.size), replace=True)
    null = np.abs((signs * arr[None, :]).mean(axis=1))
    return float((np.sum(null >= observed) + 1.0) / (N_BOOT + 1.0))


def _read_promoted_rows() -> pd.DataFrame:
    rows = pd.read_csv(POSTERIOR_CSV)
    if "likelihood_scale" in rows.columns:
        rows = rows[rows["likelihood_scale"].astype(float).eq(1.0)].copy()
    rows = rows[
        rows["prior_family"].isin(["axis_edge_parallel", "axis_edge_orthogonal"])
        & rows["observer_mode"].isin(["zero", "continuous_joint", "known"])
    ].copy()
    if rows.empty:
        raise ValueError(f"No selected posterior rows found in {POSTERIOR_CSV}")
    return rows


def _score_rows(rows: pd.DataFrame) -> pd.DataFrame:
    feature_tables = _load_feature_tables()
    feature_table = feature_tables[PRIMARY_LATENT]
    out: list[pd.DataFrame] = []
    score_column = "candidate_score_raw" if "candidate_score_raw" in rows.columns else "candidate_score"
    for (_, mode, temp), group in rows.groupby(["prior_scale", "observer_mode", "posterior_temperature"], sort=True):
        out.append(
            _vectorized_mode_rows(
                rows=group,
                latent=PRIMARY_LATENT,
                feature_table=feature_table,
                posterior_temperature=float(temp),
                score_column=score_column,
            )
        )
    scored = pd.concat(out, ignore_index=True)
    scored["axis_label"] = scored["prior_family"].map(AXIS_LABELS)
    scored["observer_label"] = scored["observer_mode"].map(OBSERVER_LABELS)
    return scored


def _summarize(scored: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["prior_scale", "prior_family", "axis_label", "observer_mode", "observer_label"]
    return (
        scored.groupby(group_cols, as_index=False)
        .agg(
            n=("feature_cosine", "size"),
            image_accuracy=("image_correct", "mean"),
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_map_feature_cosine=("map_feature_cosine", "mean"),
            mean_true_mass=("candidate_posterior_true_mass", "mean"),
            median_N_eff_fraction=("candidate_posterior_N_eff_fraction", "median"),
            posterior_temperature=("posterior_temperature", "first"),
        )
        .sort_values(["prior_scale", "observer_mode", "prior_family"])
    )


def _add_zero_gain(scored: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["trial_id", "prior_scale", "prior_family"]
    wide = scored.pivot_table(index=key_cols, columns="observer_mode", values="feature_cosine", aggfunc="first")
    if "zero" not in wide.columns:
        raise ValueError("zero observer rows are required for feature-gain calculation")
    gain_rows = []
    for observer in ["continuous_joint", "known"]:
        if observer not in wide.columns:
            continue
        block = (wide[observer] - wide["zero"]).rename("feature_gain_vs_zero").reset_index()
        block["observer_mode"] = observer
        gain_rows.append(block)
    gains = pd.concat(gain_rows, ignore_index=True)
    return scored.merge(gains, on=key_cols + ["observer_mode"], how="left")


def _paired_axis_contrasts(scored: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, object]] = []
    metric_defs = {
        "feature_cosine": "feature cosine",
        "feature_gain_vs_zero": "feature gain vs zero-eye",
        "image_correct": "image accuracy",
        "candidate_posterior_true_mass": "true posterior mass",
    }
    key_cols = ["trial_id", "prior_scale", "observer_mode"]
    for metric, label in metric_defs.items():
        if metric not in scored.columns:
            continue
        pivot = scored.pivot_table(index=key_cols, columns="prior_family", values=metric, aggfunc="first")
        required = {"axis_edge_parallel", "axis_edge_orthogonal"}
        if not required.issubset(set(pivot.columns)):
            continue
        pivot = pivot.dropna(subset=sorted(required)).reset_index()
        pivot["parallel_minus_orthogonal"] = (
            pivot["axis_edge_parallel"].astype(float) - pivot["axis_edge_orthogonal"].astype(float)
        )
        for (scale, observer), group in pivot.groupby(["prior_scale", "observer_mode"], sort=True):
            values = group["parallel_minus_orthogonal"].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean(values, rng)
            rows.append(
                {
                    "prior_scale": float(scale),
                    "observer_mode": str(observer),
                    "observer_label": OBSERVER_LABELS.get(str(observer), str(observer)),
                    "metric": metric,
                    "metric_label": label,
                    "mean_parallel_minus_orthogonal": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "sign_flip_p_two_sided": _sign_flip_p(values, rng),
                    "n_pairs": int(values.size),
                    "fraction_positive": float(np.mean(values > 0.0)),
                }
            )
        for observer, group in pivot.groupby("observer_mode", sort=True):
            values = group["parallel_minus_orthogonal"].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean(values, rng)
            rows.append(
                {
                    "prior_scale": "all",
                    "observer_mode": str(observer),
                    "observer_label": OBSERVER_LABELS.get(str(observer), str(observer)),
                    "metric": metric,
                    "metric_label": label,
                    "mean_parallel_minus_orthogonal": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "sign_flip_p_two_sided": _sign_flip_p(values, rng),
                    "n_pairs": int(values.size),
                    "fraction_positive": float(np.mean(values > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, contrasts: pd.DataFrame) -> tuple[Path, Path]:
    _configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.05), constrained_layout=True)

    ax = axes[0]
    feature = summary[summary["observer_mode"].eq("continuous_joint")].copy()
    for family in ["axis_edge_orthogonal", "axis_edge_parallel"]:
        block = feature[feature["prior_family"].eq(family)].sort_values("prior_scale")
        ax.plot(
            _scale_x(block["prior_scale"]),
            block["mean_feature_cosine"],
            marker="o",
            lw=2.0,
            color=AXIS_COLORS[family],
            label=AXIS_LABELS[family],
        )
    ax.set_title("A. joint feature recovery")
    ax.set_ylabel("mean feature cosine")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0.90, 0.98)
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)

    ax = axes[1]
    gain = contrasts[
        contrasts["observer_mode"].eq("continuous_joint")
        & contrasts["metric"].eq("feature_gain_vs_zero")
        & ~contrasts["prior_scale"].astype(str).eq("all")
    ].copy()
    x = _scale_x(gain["prior_scale"])
    y = gain["mean_parallel_minus_orthogonal"].to_numpy(dtype=float)
    yerr = np.vstack([y - gain["ci_low"].to_numpy(dtype=float), gain["ci_high"].to_numpy(dtype=float) - y])
    ax.errorbar(x, y, yerr=yerr, marker="o", lw=1.8, capsize=2.5, color="#235789")
    ax.axhline(0, color="#6b7280", lw=0.9)
    ax.set_title("B. along - across gain")
    ax.set_ylabel("feature gain difference")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    _clean_axis(ax)

    ax = axes[2]
    acc = summary[summary["observer_mode"].eq("continuous_joint")].copy()
    width = 0.34
    scales = [0.5, 1.0, 2.0]
    offsets = {"axis_edge_orthogonal": -width / 2.0, "axis_edge_parallel": width / 2.0}
    for family in ["axis_edge_orthogonal", "axis_edge_parallel"]:
        block = acc[acc["prior_family"].eq(family)].set_index("prior_scale").loc[scales].reset_index()
        ax.bar(
            _scale_x(block["prior_scale"]) + offsets[family],
            block["image_accuracy"],
            width=width,
            color=AXIS_COLORS[family],
            label=AXIS_LABELS[family],
        )
    ax.set_title("C. joint hard image ID")
    ax.set_ylabel("image accuracy")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0.50, 0.86)
    _clean_axis(ax)

    png = CONTINUOUS_OUT_DIR / "continuous_joint_axis_trace_diagnostic.png"
    pdf = CONTINUOUS_OUT_DIR / "continuous_joint_axis_trace_diagnostic.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _write_readme(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    joint_summary = summary[summary["observer_mode"].eq("continuous_joint")].copy()
    scale1 = joint_summary[joint_summary["prior_scale"].astype(float).eq(1.0)].set_index("prior_family")
    all_feature = contrasts[
        contrasts["prior_scale"].astype(str).eq("all")
        & contrasts["observer_mode"].eq("continuous_joint")
        & contrasts["metric"].eq("feature_cosine")
    ].iloc[0]
    all_gain = contrasts[
        contrasts["prior_scale"].astype(str).eq("all")
        & contrasts["observer_mode"].eq("continuous_joint")
        & contrasts["metric"].eq("feature_gain_vs_zero")
    ].iloc[0]
    lines = [
        "# Continuous Joint Along-Versus-Across Trace Diagnostic",
        "",
        "This cache-only diagnostic asks whether the promoted strict no-start joint",
        "estimator shows the same along-contour advantage as the older Figure 4D",
        "axis-prior readout.",
        "",
        "Scope note: the observed response family in this cache is empirical. The",
        "`axis_edge_parallel` and `axis_edge_orthogonal` labels refer to the",
        "image-conditioned trajectory prior/catalog family used by the observer,",
        "not to two newly rendered observed movies.",
        "",
        "At 1x for the promoted continuous joint estimator:",
        "",
        "```text",
        f"along-contour feature cosine:  {scale1.loc['axis_edge_parallel', 'mean_feature_cosine']:.4f}",
        f"across-contour feature cosine: {scale1.loc['axis_edge_orthogonal', 'mean_feature_cosine']:.4f}",
        f"along-contour image accuracy:  {scale1.loc['axis_edge_parallel', 'image_accuracy']:.4f}",
        f"across-contour image accuracy: {scale1.loc['axis_edge_orthogonal', 'image_accuracy']:.4f}",
        "```",
        "",
        "Across all scales, the paired continuous-joint contrast is:",
        "",
        "```text",
        f"feature cosine along - across:       {float(all_feature['mean_parallel_minus_orthogonal']):+.4f}",
        f"feature gain-vs-zero along - across: {float(all_gain['mean_parallel_minus_orthogonal']):+.4f}",
        "```",
        "",
        "Interpretation: this promoted continuous estimator does not reproduce a",
        "clean along-contour advantage. In feature cosine, along is slightly",
        "lower at 0.5x and slightly higher at 1x/2x, with confidence intervals",
        "crossing zero. In hard image ID, along is better at 0.5x, tied at 1x,",
        "and worse at 2x. Therefore the older 4D along-axis story should not be",
        "automatically transferred to the strict continuous joint estimator",
        "without this caveat.",
        "",
        "Outputs:",
        "",
        "- `continuous_joint_axis_trace_diagnostic.png`",
        "- `continuous_joint_axis_trace_diagnostic_summary.csv`",
        "- `continuous_joint_axis_trace_diagnostic_contrasts.csv`",
        "- `continuous_joint_axis_trace_diagnostic_trials.csv`",
    ]
    (CONTINUOUS_OUT_DIR / "continuous_joint_axis_trace_diagnostic_README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = _read_promoted_rows()
    scored = _score_rows(rows)
    scored = _add_zero_gain(scored)
    summary = _summarize(scored)
    contrasts = _paired_axis_contrasts(scored)

    CONTINUOUS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(CONTINUOUS_OUT_DIR / "continuous_joint_axis_trace_diagnostic_trials.csv", index=False)
    summary.to_csv(CONTINUOUS_OUT_DIR / "continuous_joint_axis_trace_diagnostic_summary.csv", index=False)
    contrasts.to_csv(CONTINUOUS_OUT_DIR / "continuous_joint_axis_trace_diagnostic_contrasts.csv", index=False)
    _write_readme(summary, contrasts)
    png, pdf = _plot(summary, contrasts)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    joint = summary[summary["observer_mode"].eq("continuous_joint")]
    print(
        joint[["prior_scale", "axis_label", "mean_feature_cosine", "image_accuracy", "mean_true_mass"]]
        .sort_values(["prior_scale", "axis_label"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
