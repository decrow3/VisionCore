"""Validation plots for the local BackImage feature information audit."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_SOURCE_RUN = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_local_pairing_Iz_revisit_clean_fixedmanifest_sampledK32_gabor_pyramid_rel025_0p5_1_seed7_v1"
)
DEFAULT_CORRECTED_RUN = DEFAULT_SOURCE_RUN / "local_delta_mean_info_source_trial_mean_sample_b200_20260630"

FAMILY_LABELS = {
    "actual_paired_empirical": "actual paired",
    "matched_unpaired_empirical": "matched unpaired",
}
LATENT_LABELS = {
    "gabor_local_field": "Gabor local field",
    "pyramid_local_field": "Pyramid local field",
}
COLORS = {
    "actual_paired_empirical": "#0f766e",
    "matched_unpaired_empirical": "#c2410c",
    "old": "#64748b",
    "corrected": "#111827",
    "accent": "#2563eb",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _scale_value(scale_id: str) -> float:
    return float(str(scale_id).replace("rel_", "").replace("p", ".").replace("x", ""))


def _scale_label(scale_id: str) -> str:
    return f"{_scale_value(scale_id):g}x"


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.set_axisbelow(True)


def _claim_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        df["motion_summary"].eq("delta_mean")
        & df["lhs_family"].eq("actual_paired_empirical")
        & df["rhs_family"].eq("matched_unpaired_empirical")
    ].copy()
    out["scale"] = out["scale_id"].map(_scale_value)
    return out.sort_values(["latent", "k", "scale"])


def _gain_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        df["motion_summary"].eq("delta_mean")
        & df["family"].isin(["actual_paired_empirical", "matched_unpaired_empirical"])
    ].copy()
    out["scale"] = out["scale_id"].map(_scale_value)
    return out.sort_values(["latent", "k", "family", "scale"])


def _facets(df: pd.DataFrame) -> list[tuple[str, int]]:
    keys = sorted({(str(row.latent), int(row.k)) for row in df.itertuples()}, key=lambda item: (item[0], item[1]))
    latent_order = {"gabor_local_field": 0, "pyramid_local_field": 1}
    return sorted(keys, key=lambda item: (latent_order.get(item[0], 99), item[1]))


def _save(fig: plt.Figure, out_dir: Path, name: str, pdf: PdfPages | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_claim_contrast_info(corrected_contrasts: pd.DataFrame, out_dir: Path, pdf: PdfPages | None) -> None:
    data = _claim_rows(corrected_contrasts)
    facets = _facets(data)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    axes = axes.ravel()
    yvals = np.concatenate(
        [
            data["info_diag_ci95_low"].to_numpy(dtype=float),
            data["info_diag_ci95_high"].to_numpy(dtype=float),
        ]
    )
    span = max(0.1, float(np.nanmax(np.abs(yvals))) * 1.12)
    for ax, (latent, k) in zip(axes, facets, strict=True):
        block = data[data["latent"].eq(latent) & data["k"].eq(k)].sort_values("scale")
        x = block["scale"].to_numpy(dtype=float)
        y = block["incremental_gain_delta_info_diag_bits"].to_numpy(dtype=float)
        lo = block["info_diag_ci95_low"].to_numpy(dtype=float)
        hi = block["info_diag_ci95_high"].to_numpy(dtype=float)
        ax.axhline(0.0, color="#111827", linewidth=0.8)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            marker="o",
            markersize=4.8,
            linewidth=1.8,
            capsize=3,
            color=COLORS["corrected"],
        )
        ax.set_title(f"{LATENT_LABELS.get(latent, latent)} | k={k}")
        ax.set_xticks(x, [_scale_label(scale_id) for scale_id in block["scale_id"]])
        ax.set_ylim(-span, span)
        _clean_axis(ax)
    fig.supylabel("actual - matched information gain (bits)")
    fig.supxlabel("motion scale")
    fig.suptitle("Corrected local pairing contrast: source-trial grouped, static-mean baseline", y=1.02)
    _save(fig, out_dir, "01_corrected_actual_minus_matched_info_bits", pdf)


def plot_gain_over_static_info(corrected_gains: pd.DataFrame, out_dir: Path, pdf: PdfPages | None) -> None:
    data = _gain_rows(corrected_gains)
    facets = _facets(data)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    axes = axes.ravel()
    yvals = np.concatenate(
        [
            data["info_diag_ci95_low"].to_numpy(dtype=float),
            data["info_diag_ci95_high"].to_numpy(dtype=float),
        ]
    )
    lo_lim = min(-0.1, float(np.nanmin(yvals)) * 1.08)
    hi_lim = max(0.1, float(np.nanmax(yvals)) * 1.08)
    for ax, (latent, k) in zip(axes, facets, strict=True):
        for family in ["actual_paired_empirical", "matched_unpaired_empirical"]:
            block = data[data["latent"].eq(latent) & data["k"].eq(k) & data["family"].eq(family)].sort_values("scale")
            x = block["scale"].to_numpy(dtype=float)
            y = block["incremental_gain_info_diag_bits"].to_numpy(dtype=float)
            lo = block["info_diag_ci95_low"].to_numpy(dtype=float)
            hi = block["info_diag_ci95_high"].to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                marker="o",
                markersize=4.5,
                linewidth=1.7,
                capsize=3,
                color=COLORS[family],
                label=FAMILY_LABELS[family],
            )
        ax.axhline(0.0, color="#111827", linewidth=0.8)
        ax.set_title(f"{LATENT_LABELS.get(latent, latent)} | k={k}")
        ax.set_xticks(sorted(data["scale"].unique()), [f"{v:g}x" for v in sorted(data["scale"].unique())])
        ax.set_ylim(lo_lim, hi_lim)
        _clean_axis(ax)
    axes[0].legend(frameon=False, loc="upper left")
    fig.supylabel("information gain over static (bits)")
    fig.supxlabel("motion scale")
    fig.suptitle("Motion information over stabilized baseline: actual vs matched", y=1.02)
    _save(fig, out_dir, "02_gain_over_static_info_bits", pdf)


def plot_old_vs_corrected_neg_mse(
    old_contrasts: pd.DataFrame,
    corrected_contrasts: pd.DataFrame,
    out_dir: Path,
    pdf: PdfPages | None,
) -> None:
    old = _claim_rows(old_contrasts)
    new = _claim_rows(corrected_contrasts)
    old["contract"] = "old image grouped, zero baseline"
    new["contract"] = "corrected source trial, static mean"
    data = pd.concat([old, new], ignore_index=True)
    facets = _facets(data)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (latent, k) in zip(axes, facets, strict=True):
        for offset, contract, color in [
            (-0.035, "old image grouped, zero baseline", COLORS["old"]),
            (0.035, "corrected source trial, static mean", COLORS["corrected"]),
        ]:
            block = data[data["latent"].eq(latent) & data["k"].eq(k) & data["contract"].eq(contract)].sort_values("scale")
            x = block["scale"].to_numpy(dtype=float) + offset
            y = block["incremental_gain_delta_neg_mse"].to_numpy(dtype=float)
            lo = block["ci95_low"].to_numpy(dtype=float)
            hi = block["ci95_high"].to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                marker="o",
                markersize=4.5,
                linewidth=0,
                elinewidth=1.5,
                capsize=3,
                color=color,
                label=contract,
            )
        ax.axhline(0.0, color="#111827", linewidth=0.8)
        ax.set_title(f"{LATENT_LABELS.get(latent, latent)} | k={k}")
        ax.set_xticks(sorted(data["scale"].unique()), [f"{v:g}x" for v in sorted(data["scale"].unique())])
        _clean_axis(ax)
    axes[0].legend(frameon=False, loc="upper left")
    fig.supylabel("actual - matched gain contrast (-MSE)")
    fig.supxlabel("motion scale")
    fig.suptitle("Diagnostic contract check: old positive rows collapse after correction", y=1.02)
    _save(fig, out_dir, "03_old_vs_corrected_neg_mse_contrast", pdf)


def plot_gain_scatter(corrected_gains: pd.DataFrame, out_dir: Path, pdf: PdfPages | None) -> None:
    data = _gain_rows(corrected_gains)
    actual = data[data["family"].eq("actual_paired_empirical")]
    matched = data[data["family"].eq("matched_unpaired_empirical")]
    merged = actual.merge(
        matched,
        on=["motion_summary", "static_summary", "scale_id", "latent", "k"],
        suffixes=("_actual", "_matched"),
    )
    x = merged["incremental_gain_info_diag_bits_matched"].to_numpy(dtype=float)
    y = merged["incremental_gain_info_diag_bits_actual"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lo = min(float(np.nanmin(x)), float(np.nanmin(y)), -0.1)
    hi = max(float(np.nanmax(x)), float(np.nanmax(y)), 0.1)
    pad = 0.08 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111827", linewidth=0.9, linestyle="--")
    for latent, marker in [("gabor_local_field", "o"), ("pyramid_local_field", "s")]:
        block = merged[merged["latent"].eq(latent)]
        ax.scatter(
            block["incremental_gain_info_diag_bits_matched"],
            block["incremental_gain_info_diag_bits_actual"],
            s=46,
            marker=marker,
            color=COLORS["accent"],
            edgecolor="white",
            linewidth=0.7,
            label=LATENT_LABELS.get(latent, latent),
        )
        for row in block.itertuples():
            ax.text(
                float(row.incremental_gain_info_diag_bits_matched) + 0.015,
                float(row.incremental_gain_info_diag_bits_actual) + 0.015,
                f"k={int(row.k)} {_scale_label(row.scale_id)}",
                fontsize=6.8,
                color="#334155",
            )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("matched unpaired gain over static (bits)")
    ax.set_ylabel("actual paired gain over static (bits)")
    ax.set_title("Actual vs matched gains: points near y=x mean no pairing advantage")
    ax.legend(frameon=False, loc="upper left")
    _clean_axis(ax)
    _save(fig, out_dir, "04_actual_vs_matched_gain_scatter", pdf)


def plot_grouping_qc(images: pd.DataFrame, out_dir: Path, pdf: PdfPages | None) -> None:
    groups = images["session"].astype(str) + "::trial_" + images["trial_idx"].astype(int).astype(str)
    counts = groups.value_counts()
    count_bins = counts.value_counts().sort_index()
    n_images = int(images.shape[0])
    n_groups = int(counts.shape[0])
    repeated_groups = int((counts > 1).sum())
    repeated_images = int(counts[counts > 1].sum())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    ax.bar(count_bins.index.astype(int), count_bins.to_numpy(dtype=int), color="#475569", width=0.65)
    ax.set_xlabel("windows per source trial")
    ax.set_ylabel("source-trial groups")
    ax.set_title("Source-trial repeat structure")
    ax.set_xticks(count_bins.index.astype(int))
    _clean_axis(ax)

    ax = axes[1]
    values = [n_images - repeated_images, repeated_images]
    labels = ["single-source windows", "windows in repeated groups"]
    ax.bar([0, 1], values, color=["#0f766e", "#c2410c"], width=0.58)
    ax.set_xticks([0, 1], labels, rotation=15, ha="right")
    ax.set_ylabel("windows")
    ax.set_title(f"{n_groups} source-trial groups; {repeated_groups} repeat")
    for i, value in enumerate(values):
        ax.text(i, value + 2, str(value), ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(values) * 1.22)
    _clean_axis(ax)
    fig.suptitle("Why source-trial grouped CV is required", y=1.03)
    _save(fig, out_dir, "05_source_trial_grouping_qc", pdf)


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 3:
        return float("nan")
    xr = pd.Series(x[ok]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[ok]).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(xr, yr)[0, 1])


def _window_pairing_score(window_contrasts: pd.DataFrame) -> pd.DataFrame:
    data = _claim_rows(window_contrasts)
    data["row_key"] = data["scale_id"].astype(str) + "|" + data["latent"].astype(str) + "|k" + data["k"].astype(str)
    data["z_delta"] = data.groupby("row_key")["incremental_gain_delta_neg_mse"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) if s.std(ddof=1) > 0 else np.nan)
    )
    edge = data["image_edge_axis_deg"].to_numpy(dtype=float)
    drift = data["drift_orientation_deg"].to_numpy(dtype=float)
    diff = np.abs(((edge - drift + 90.0) % 180.0) - 90.0)
    data["edge_drift_axis_diff_deg"] = diff
    data["edge_drift_parallel_cos2"] = np.cos(2.0 * np.deg2rad(diff))
    metrics = [
        "image_orientation_coherence",
        "drift_anisotropy",
        "actual_observed_rms_deg",
        "actual_path_length_deg",
        "actual_lag1_autocorr",
        "edge_drift_parallel_cos2",
        "edge_drift_axis_diff_deg",
    ]
    id_cols = ["image_index", "source_row", "session", "trial_idx", "phase", *metrics]
    return data.groupby(id_cols, dropna=False)["z_delta"].mean().reset_index(name="pairing_advantage_z")


def plot_exploratory_metric_correlations(
    window_contrasts: pd.DataFrame,
    out_dir: Path,
    pdf: PdfPages | None,
) -> None:
    score = _window_pairing_score(window_contrasts)
    metrics = [
        ("actual_observed_rms_deg", "observed RMS"),
        ("image_orientation_coherence", "image coherence"),
        ("drift_anisotropy", "drift anisotropy"),
        ("edge_drift_parallel_cos2", "edge/drift parallel"),
        ("actual_lag1_autocorr", "lag-1 autocorr"),
        ("actual_path_length_deg", "path length"),
    ]
    rows = []
    rng = np.random.default_rng(13)
    groups = (score["session"].astype(str) + "::trial_" + score["trial_idx"].astype(int).astype(str)).to_numpy()
    unique_groups = np.array(sorted(pd.unique(groups)))
    idx_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for metric, label in metrics:
        x = score[metric].to_numpy(dtype=float)
        y = score["pairing_advantage_z"].to_numpy(dtype=float)
        boots = np.empty(400, dtype=float)
        for i in range(boots.size):
            chosen = rng.choice(unique_groups, size=unique_groups.size, replace=True)
            idx = np.concatenate([idx_by_group[group] for group in chosen])
            boots[i] = _rank_corr(x[idx], y[idx])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append({"metric": label, "rho": _rank_corr(x, y), "lo": lo, "hi": hi})
    corr = pd.DataFrame(rows).sort_values("rho")
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    y = np.arange(corr.shape[0])
    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.errorbar(
        corr["rho"],
        y,
        xerr=np.vstack([corr["rho"] - corr["lo"], corr["hi"] - corr["rho"]]),
        marker="o",
        markersize=4.6,
        linewidth=0,
        elinewidth=1.5,
        capsize=3,
        color=COLORS["corrected"],
    )
    ax.set_yticks(y, corr["metric"])
    ax.set_xlabel("Spearman rho with window-level pairing advantage")
    ax.set_title("Exploratory local-trend check: weak positive descriptors, wide uncertainty")
    _clean_axis(ax)
    _save(fig, out_dir, "06_exploratory_metric_correlations", pdf)


def plot_exploratory_metric_split_heatmap(
    window_contrasts: pd.DataFrame,
    out_dir: Path,
    pdf: PdfPages | None,
) -> None:
    data = _claim_rows(window_contrasts)
    data["row_key"] = data["scale_id"].astype(str) + "|" + data["latent"].astype(str) + "|k" + data["k"].astype(str)
    edge = data["image_edge_axis_deg"].to_numpy(dtype=float)
    drift = data["drift_orientation_deg"].to_numpy(dtype=float)
    diff = np.abs(((edge - drift + 90.0) % 180.0) - 90.0)
    data["edge_drift_parallel_cos2"] = np.cos(2.0 * np.deg2rad(diff))
    metrics = [
        ("actual_observed_rms_deg", "RMS"),
        ("image_orientation_coherence", "coherence"),
        ("drift_anisotropy", "drift anis."),
        ("edge_drift_parallel_cos2", "edge/drift"),
    ]
    rows = []
    for key, sub in data.groupby("row_key"):
        scale_id, latent, k_token = key.split("|")
        label = f"{LATENT_LABELS.get(latent, latent).replace(' local field', '')} {k_token} {_scale_label(scale_id)}"
        row = {"row": label}
        for metric, metric_label in metrics:
            threshold = float(np.nanmedian(sub[metric]))
            high = sub[sub[metric] >= threshold]["incremental_gain_delta_neg_mse"].mean()
            low = sub[sub[metric] < threshold]["incremental_gain_delta_neg_mse"].mean()
            row[metric_label] = float(high - low)
        rows.append(row)
    heat = pd.DataFrame(rows).set_index("row")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    vmax = float(np.nanmax(np.abs(heat.to_numpy(dtype=float))))
    im = ax.imshow(heat.to_numpy(dtype=float), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(heat.shape[0]), heat.index)
    ax.set_xticks(np.arange(heat.shape[1]), heat.columns)
    ax.set_title("High-minus-low pairing advantage by descriptor (-MSE units)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = float(heat.iloc[i, j])
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7.0, color="#111827")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("high - low actual-minus-matched gain")
    _save(fig, out_dir, "07_exploratory_metric_split_heatmap", pdf)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--corrected-run-dir", type=Path, default=DEFAULT_CORRECTED_RUN)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> Path:
    _configure_matplotlib()
    source_run = Path(args.source_run_dir)
    corrected_run = Path(args.corrected_run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else corrected_run / "validation_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    old_contrasts = pd.read_csv(source_run / "incremental_gain_contrasts.csv")
    corrected_contrasts = pd.read_csv(corrected_run / "incremental_gain_contrasts.csv")
    corrected_gains = pd.read_csv(corrected_run / "incremental_gain_vs_static.csv")
    window_contrasts = pd.read_csv(corrected_run / "incremental_gain_contrasts_by_window.csv")
    images = pd.read_csv(source_run / "analysis_images.csv")

    with PdfPages(out_dir / "local_feature_information_validation_plots.pdf") as pdf:
        plot_claim_contrast_info(corrected_contrasts, out_dir, pdf)
        plot_gain_over_static_info(corrected_gains, out_dir, pdf)
        plot_old_vs_corrected_neg_mse(old_contrasts, corrected_contrasts, out_dir, pdf)
        plot_gain_scatter(corrected_gains, out_dir, pdf)
        plot_grouping_qc(images, out_dir, pdf)
        plot_exploratory_metric_correlations(window_contrasts, out_dir, pdf)
        plot_exploratory_metric_split_heatmap(window_contrasts, out_dir, pdf)

    print(f"Wrote validation plots to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
