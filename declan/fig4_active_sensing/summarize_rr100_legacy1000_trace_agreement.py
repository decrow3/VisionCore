#!/usr/bin/env python3
"""Summarize old-versus-corrected agreement for the 1,000 legacy traces.

This Phase-1, no-neural-model checkpoint joins the immutable legacy trace bank
to its corrected 240-Hz/global-even/dpi_pix descriptor crosswalk.  It reports
agreement and rank transitions only; it does not modify cached neural values or
claim that those values were computed with the corrected visual contract.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DESCRIPTORS = ROOT / "outputs/fig4_active_sensing/rr100_legacy1000_corrected_trace_descriptors_v1/corrected_trace_descriptors.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_legacy1000_trace_agreement_checkpoint_23_v1"

METRICS = (
    ("path_length_arcmin", "path length (arcmin)"),
    ("rms_radius_arcmin", "RMS radius (arcmin)"),
    ("max_radius_arcmin", "maximum radius (arcmin)"),
    ("median_step_arcmin", "median step (arcmin)"),
    ("mean_speed_dps", "mean speed (deg/s)"),
    ("cov_anisotropy", "covariance anisotropy"),
    ("position_power_centroid_hz", "position-power centroid (Hz)"),
    ("position_power_fraction_15plus_hz", "position power >=15 Hz"),
    ("position_power_fraction_32plus_hz", "position power >32 Hz"),
)
OLD = "legacy_cached_as120_"
NEW = "corrected_dpi_crop120_"
N_QUANTILES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", type=Path, default=DESCRIPTORS)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def rank_verdict(rho: float) -> str:
    if not np.isfinite(rho):
        return "not evaluable"
    if rho >= 0.90:
        return "high rank preservation; legacy ordering may be used only as a structural hypothesis"
    if rho >= 0.60:
        return "moderate rank preservation; use only for convergence, low-rank structure, and example selection"
    return "poor rank preservation; do not reinterpret legacy neural responses with this corrected descriptor"


def subsets(frame: pd.DataFrame) -> dict[str, pd.Series]:
    # These microsaccade labels were generated under the legacy 120-Hz
    # interpretation and deliberately remain labeled as legacy strata.
    legacy_event = frame["legacy_cached_as120_p95_speed_dps"] >= frame["legacy_cached_as120_p95_speed_dps"].quantile(0.80)
    return {
        "all_1000": pd.Series(True, index=frame.index),
        "explicit_history_valid_973": frame["explicit_history_valid"].astype(bool),
        "legacy_event_positive_200": legacy_event,
        "legacy_event_negative_800": ~legacy_event,
    }


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(stats.pearsonr(x, y).statistic if method == "pearson" else stats.spearmanr(x, y).statistic)


def numeric_agreement(frame: pd.DataFrame, mask: pd.Series, key: str) -> dict[str, object]:
    sub = frame.loc[mask, [OLD + key, NEW + key]].dropna()
    x, y = sub[OLD + key].to_numpy(float), sub[NEW + key].to_numpy(float)
    slope = intercept = np.nan
    if len(sub) >= 3 and np.nanstd(x) > 0:
        fit = stats.theilslopes(y, x)
        slope, intercept = float(fit.slope), float(fit.intercept)
    rank_delta = pd.Series(y).rank(pct=True) - pd.Series(x).rank(pct=True)
    return {
        "n": int(len(sub)),
        "pearson_r": safe_corr(x, y, "pearson"),
        "spearman_rho": safe_corr(x, y, "spearman"),
        "theil_sen_slope_new_on_old": slope,
        "theil_sen_intercept_new_on_old": intercept,
        "median_abs_difference": float(np.median(np.abs(y - x))),
        "median_rank_delta": float(np.median(rank_delta)),
        "median_abs_rank_delta": float(np.median(np.abs(rank_delta))),
    }


def orientation_agreement(frame: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    sub = frame.loc[mask, [OLD + "cov_orientation_deg", NEW + "cov_orientation_deg"]].dropna()
    old, new = sub.iloc[:, 0].to_numpy(float), sub.iloc[:, 1].to_numpy(float)
    delta = ((new - old + 90.0) % 180.0) - 90.0
    return {
        "n": int(len(sub)),
        "pearson_r": safe_corr(np.cos(np.deg2rad(2 * old)), np.cos(np.deg2rad(2 * new)), "pearson"),
        "spearman_rho": np.nan,
        "theil_sen_slope_new_on_old": np.nan,
        "theil_sen_intercept_new_on_old": np.nan,
        "median_abs_difference": float(np.median(np.abs(delta))),
        "median_rank_delta": np.nan,
        "median_abs_rank_delta": np.nan,
    }


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset_name, mask in subsets(frame).items():
        for key, label in METRICS:
            row = {"subset": subset_name, "metric": key, "metric_label": label, **numeric_agreement(frame, mask, key)}
            row["reuse_verdict"] = rank_verdict(float(row["spearman_rho"]))
            rows.append(row)
        row = {"subset": subset_name, "metric": "cov_orientation_deg", "metric_label": "covariance orientation (deg, axial)", **orientation_agreement(frame, mask)}
        row["reuse_verdict"] = "orientation agreement is descriptive only; corrected orientation claims require corrected neural responses"
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_transitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    counts: list[dict[str, object]] = []
    for key, _ in METRICS:
        sub = frame[["trace_index", "explicit_history_valid", OLD + key, NEW + key]].dropna().copy()
        sub["legacy_quantile"] = pd.qcut(sub[OLD + key].rank(method="first"), N_QUANTILES, labels=False) + 1
        sub["corrected_quantile"] = pd.qcut(sub[NEW + key].rank(method="first"), N_QUANTILES, labels=False) + 1
        sub["metric"] = key
        sub["quantile_delta"] = sub["corrected_quantile"] - sub["legacy_quantile"]
        rows.extend(sub[["trace_index", "explicit_history_valid", "metric", "legacy_quantile", "corrected_quantile", "quantile_delta"]].to_dict("records"))
        grouped = sub.groupby(["legacy_quantile", "corrected_quantile"], as_index=False).size()
        grouped["metric"] = key
        grouped.rename(columns={"size": "n_traces"}, inplace=True)
        grouped["fraction_within_legacy_quantile"] = grouped["n_traces"] / grouped.groupby("legacy_quantile")["n_traces"].transform("sum")
        counts.extend(grouped.to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(counts)


def write_crosswalk(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    path_old, path_new = OLD + "path_length_arcmin", NEW + "path_length_arcmin"
    out["legacy_path_rank"] = out[path_old].rank(pct=True)
    out["corrected_path_rank"] = out[path_new].rank(pct=True)
    out["corrected_minus_legacy_path_rank"] = out["corrected_path_rank"] - out["legacy_path_rank"]
    out["legacy_event_stratum"] = np.where(
        out["legacy_cached_as120_p95_speed_dps"] >= out["legacy_cached_as120_p95_speed_dps"].quantile(0.80),
        "legacy_event_positive_200", "legacy_event_negative_800",
    )
    out["inclusion_recommendation"] = np.where(
        out["explicit_history_valid"],
        "eligible for matched legacy bridge; legacy neural response remains structural-only",
        "exclude from explicit-history bridge; replace in corrected 1,000-trace production cohort",
    )
    return out


def plot_agreement(frame: pd.DataFrame, summary: pd.DataFrame, out: Path, dpi: int) -> None:
    valid = frame[frame["explicit_history_valid"]].copy()
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.2), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, METRICS):
        x, y = valid[OLD + key].to_numpy(float), valid[NEW + key].to_numpy(float)
        axis.scatter(x, y, s=12, alpha=0.55, color="#377eb8", edgecolors="none")
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        axis.plot([lo, hi], [lo, hi], color="0.45", lw=1, ls="--")
        row = summary[(summary["subset"] == "explicit_history_valid_973") & (summary["metric"] == key)].iloc[0]
        axis.set(title=f"{label}\nPearson r={row.pearson_r:+.2f}; Spearman rho={row.spearman_rho:+.2f}", xlabel="legacy cached", ylabel="corrected dpi_pix")
        axis.grid(alpha=0.15)
    fig.suptitle("Phase 1: legacy versus corrected trace descriptors (973 explicit-history-valid identities)", fontweight="bold")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_transitions(counts: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.0), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, METRICS):
        sub = counts[counts["metric"].eq(key)]
        matrix = sub.pivot(index="corrected_quantile", columns="legacy_quantile", values="fraction_within_legacy_quantile").reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
        image = axis.imshow(matrix.to_numpy(), origin="lower", vmin=0, vmax=1, cmap="magma")
        for r in range(5):
            for c in range(5):
                axis.text(c, r, f"{matrix.iat[r, c]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix.iat[r, c] > .45 else "black")
        axis.set(title=label, xlabel="legacy quantile", ylabel="corrected quantile", xticks=range(5), xticklabels=range(1, 6), yticks=range(5), yticklabels=range(1, 6))
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.8, label="fraction within legacy quantile")
    fig.suptitle("Phase 1: old-to-corrected quantile transitions (all 1,000 identities)", fontweight="bold")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {args.out_dir}")
    frame = pd.read_csv(args.descriptors)
    if len(frame) != 1000 or frame["trace_index"].nunique() != 1000:
        raise ValueError("Expected exactly 1,000 unique legacy trace identities")
    required = {"trace_index", "explicit_history_valid", *(OLD + key for key, _ in METRICS), *(NEW + key for key, _ in METRICS)}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Descriptor file lacks required columns: {sorted(missing)}")
    args.out_dir.mkdir(parents=True)
    summary = build_summary(frame)
    transitions, transition_counts = quantile_transitions(frame)
    crosswalk = write_crosswalk(frame)
    summary.to_csv(args.out_dir / "descriptor_agreement_summary.csv", index=False)
    transitions.to_csv(args.out_dir / "trace_quantile_transitions.csv", index=False)
    transition_counts.to_csv(args.out_dir / "trace_quantile_transition_counts.csv", index=False)
    crosswalk.to_csv(args.out_dir / "corrected_trace_crosswalk.csv", index=False)
    crosswalk.loc[~crosswalk["explicit_history_valid"]].to_csv(args.out_dir / "invalid_history_table.csv", index=False)
    plot_agreement(frame, summary, args.out_dir / "population_descriptor_agreement.png", args.dpi)
    plot_transitions(transition_counts, args.out_dir / "quantile_transition_heatmaps.png", args.dpi)
    valid_summary = summary[summary["subset"].eq("explicit_history_valid_973")].set_index("metric")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "phase_1_trace_agreement_complete",
        "scope": "1,000 legacy trace identities; no neural model rerun",
        "input_descriptor_path": str(args.descriptors.resolve()),
        "counts": {"n_trace_identities": 1000, "n_explicit_history_valid": int(frame["explicit_history_valid"].sum()), "n_explicit_history_invalid": int((~frame["explicit_history_valid"]).sum())},
        "predeclared_rank_interpretation": {"high": "Spearman rho >= 0.90", "moderate": "0.60 <= Spearman rho < 0.90", "poor": "Spearman rho < 0.60"},
        "path_rank_reuse_verdict": valid_summary.loc["path_length_arcmin", "reuse_verdict"],
        "radius_rank_reuse_verdict": valid_summary.loc["rms_radius_arcmin", "reuse_verdict"],
        "guardrail": "This is an input-identity agreement audit. The legacy 100x1000 neural cache remains legacy reconstructed-motion structural-only evidence.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
