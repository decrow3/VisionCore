"""Audit whether BackImage motion scale tracks local spatial-frequency content.

This analysis is deliberately redundant: it checks the same hypothesis with
window-level residual correlations, per-session correlations, and within-session
high-vs-low spatial-frequency dose summaries. The goal is to make a negative or
small result hard to dismiss as a metric artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review")
DEFAULT_INPUT = BASE_DIR / "backimage_image_structure_reviewed_v2_screenfiltered_yfix" / "backimage_image_fem_windows.csv"
DEFAULT_OUT_DIR = BASE_DIR / "backimage_motion_scale_spatial_frequency_audit"

MOTION_METRICS = [
    "rms_radius_deg",
    "p95_radius_deg",
    "step_mean_deg",
    "path_length_deg_s",
    "speed_mean_deg_s",
    "diffusion_constant_deg2_s",
]

SF_BAND_METRICS = [
    "image_high_freq_power_fraction",
    "image_power_0_2_cpd_fraction",
    "image_power_2_4_cpd_fraction",
    "image_power_4_8_cpd_fraction",
    "image_power_8plus_cpd_fraction",
]

SF_SLOPE_METRICS = [
    "image_power_slope_0p5_16_cpd",
    "image_amplitude_slope_0p5_16_cpd",
    "image_power_slope_deviation_from_1f",
    "image_amplitude_slope_deviation_from_1f",
    "image_abs_power_slope_deviation_from_1f",
    "image_abs_amplitude_slope_deviation_from_1f",
]

CONTROL_METRICS = [
    "samples_since_event",
    "abs_mean_radius_deg",
    "image_patch_fraction_inside_image",
    "image_patch_fraction_background",
    "image_patch_distance_to_image_border_px",
]

FOCUS_MOTION = ["rms_radius_deg", "step_mean_deg", "speed_mean_deg_s", "diffusion_constant_deg2_s"]
FOCUS_SF = [
    "image_amplitude_slope_0p5_16_cpd",
    "image_amplitude_slope_deviation_from_1f",
    "image_abs_amplitude_slope_deviation_from_1f",
    "sf_high_4plus_fraction",
    "image_high_freq_power_fraction",
    "image_power_8plus_cpd_fraction",
]


def _require_columns(df: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    xa = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64)
    ya = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    return xa[ok], ya[ok]


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 4 or np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 4 or np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1 or n_boot <= 0:
        return float(vals[0]), float(vals[0])
    idx = rng.integers(0, vals.size, size=(int(n_boot), vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _bh_q(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if not np.any(finite):
        return out
    pf = p[finite]
    order = np.argsort(pf)
    q = np.empty_like(pf)
    prev = 1.0
    n = pf.size
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = n - rank_from_end + 1
        prev = min(prev, pf[idx] * n / rank)
        q[idx] = prev
    out[np.where(finite)[0]] = np.minimum(q, 1.0)
    return out


def _add_spatial_frequency_summaries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    low = out["image_power_0_2_cpd_fraction"].astype(float) + out["image_power_2_4_cpd_fraction"].astype(float)
    high = out["image_power_4_8_cpd_fraction"].astype(float) + out["image_power_8plus_cpd_fraction"].astype(float)
    out["sf_low_0_4_fraction"] = low
    out["sf_high_4plus_fraction"] = high
    out["sf_mid_2_8_fraction"] = out["image_power_2_4_cpd_fraction"].astype(float) + out["image_power_4_8_cpd_fraction"].astype(float)
    out["sf_high_vs_low_fraction"] = high - low
    out["sf_high_low_log_ratio"] = np.log((high + 1e-6) / (low + 1e-6))
    # A bounded descriptive centroid. The 8+ bin is assigned 12 cpd only as a
    # display/summary convention, not as an exact spectral moment.
    out["sf_centroid_cpd_approx"] = (
        1.0 * out["image_power_0_2_cpd_fraction"].astype(float)
        + 3.0 * out["image_power_2_4_cpd_fraction"].astype(float)
        + 6.0 * out["image_power_4_8_cpd_fraction"].astype(float)
        + 12.0 * out["image_power_8plus_cpd_fraction"].astype(float)
    )
    return out


def _prepare_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["session", "phase", "image_feature_ok", *MOTION_METRICS, *SF_BAND_METRICS, *CONTROL_METRICS]
    _require_columns(df, path, required)
    df = df[df["image_feature_ok"].fillna(False).astype(bool)].copy()
    df = _add_spatial_frequency_summaries(df)
    cols = MOTION_METRICS + SF_BAND_METRICS + [col for col in SF_SLOPE_METRICS if col in df.columns] + [
        "sf_low_0_4_fraction",
        "sf_high_4plus_fraction",
        "sf_mid_2_8_fraction",
        "sf_high_vs_low_fraction",
        "sf_high_low_log_ratio",
        "sf_centroid_cpd_approx",
    ] + CONTROL_METRICS
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["session", "phase", *MOTION_METRICS, *SF_BAND_METRICS]).copy()
    return df


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    controls = df[CONTROL_METRICS].astype(float).copy()
    for col in CONTROL_METRICS:
        values = controls[col].to_numpy(dtype=np.float64)
        mu = np.nanmean(values)
        sd = np.nanstd(values)
        controls[col] = (values - mu) / sd if sd > 0 else 0.0
    dummies = pd.get_dummies(df[["session", "phase"]].astype(str), drop_first=True, dtype=float)
    x = pd.concat([pd.Series(1.0, index=df.index, name="intercept"), controls, dummies], axis=1)
    arr = x.to_numpy(dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _residualize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    x = _design_matrix(df)
    for col in columns:
        y = df[col].to_numpy(dtype=np.float64)
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        resid = np.full(y.shape, np.nan, dtype=np.float64)
        if np.count_nonzero(ok) > x.shape[1] + 2:
            beta, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
            resid[ok] = y[ok] - x[ok] @ beta
        out[col] = resid
    return out


def _window_correlation_summary(df: pd.DataFrame, sf_metrics: list[str]) -> pd.DataFrame:
    residuals = _residualize_columns(df, MOTION_METRICS + sf_metrics)
    rows: list[dict[str, object]] = []
    for motion in MOTION_METRICS:
        for sf in sf_metrics:
            x, y = _finite_pair(df[motion], df[sf])
            rho, sp = _spearman(x, y)
            r, pp = _pearson(x, y)
            xr, yr = _finite_pair(residuals[motion], residuals[sf])
            rrho, rsp = _spearman(xr, yr)
            rr, rpp = _pearson(xr, yr)
            rows.append(
                {
                    "motion_metric": motion,
                    "sf_metric": sf,
                    "n_windows": int(x.size),
                    "raw_spearman_rho": rho,
                    "raw_spearman_p": sp,
                    "raw_pearson_r": r,
                    "raw_pearson_p": pp,
                    "residual_spearman_rho": rrho,
                    "residual_spearman_p": rsp,
                    "residual_pearson_r": rr,
                    "residual_pearson_p": rpp,
                }
            )
    out = pd.DataFrame(rows)
    out["residual_spearman_q_bh"] = _bh_q(out["residual_spearman_p"].to_numpy(dtype=np.float64))
    out["raw_spearman_q_bh"] = _bh_q(out["raw_spearman_p"].to_numpy(dtype=np.float64))
    return out


def _session_correlation_summary(
    df: pd.DataFrame,
    sf_metrics: list[str],
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for motion in MOTION_METRICS:
        for sf in sf_metrics:
            rhos: list[float] = []
            for session, sub in df.groupby("session", dropna=False):
                x, y = _finite_pair(sub[motion], sub[sf])
                rho, p = _spearman(x, y)
                if np.isfinite(rho):
                    rhos.append(rho)
                per_rows.append(
                    {
                        "session": session,
                        "motion_metric": motion,
                        "sf_metric": sf,
                        "n_windows": int(x.size),
                        "spearman_rho": rho,
                        "spearman_p": p,
                    }
                )
            arr = np.asarray(rhos, dtype=np.float64)
            ci_low, ci_high = _bootstrap_ci(arr, rng, n_boot)
            n_pos = int(np.count_nonzero(arr > 0.0))
            sign_p = float(stats.binomtest(n_pos, n=arr.size, p=0.5).pvalue) if arr.size else float("nan")
            wilcoxon_p = float(stats.wilcoxon(arr, alternative="two-sided").pvalue) if arr.size > 0 and np.any(arr != 0) else float("nan")
            summary_rows.append(
                {
                    "motion_metric": motion,
                    "sf_metric": sf,
                    "n_sessions": int(arr.size),
                    "mean_session_spearman_rho": float(np.mean(arr)) if arr.size else float("nan"),
                    "median_session_spearman_rho": float(np.median(arr)) if arr.size else float("nan"),
                    "ci95_low_session_mean_rho": ci_low,
                    "ci95_high_session_mean_rho": ci_high,
                    "n_positive_sessions": n_pos,
                    "sign_test_p": sign_p,
                    "wilcoxon_p": wilcoxon_p,
                }
            )
    per_session = pd.DataFrame(per_rows)
    summary = pd.DataFrame(summary_rows)
    summary["sign_test_q_bh"] = _bh_q(summary["sign_test_p"].to_numpy(dtype=np.float64))
    summary["wilcoxon_q_bh"] = _bh_q(summary["wilcoxon_p"].to_numpy(dtype=np.float64))
    return per_session, summary


def _dose_response_summary(
    df: pd.DataFrame,
    sf_metrics: list[str],
    *,
    rng: np.random.Generator,
    n_boot: int,
    quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for motion in MOTION_METRICS:
        for sf in sf_metrics:
            diffs: list[float] = []
            std_diffs: list[float] = []
            for session, sub in df.groupby("session", dropna=False):
                values = pd.to_numeric(sub[sf], errors="coerce")
                motion_values = pd.to_numeric(sub[motion], errors="coerce")
                ok = np.isfinite(values) & np.isfinite(motion_values)
                values = values[ok]
                motion_values = motion_values[ok]
                if values.size < 12:
                    continue
                lo = float(values.quantile(quantile))
                hi = float(values.quantile(1.0 - quantile))
                low_motion = motion_values[values <= lo]
                high_motion = motion_values[values >= hi]
                if low_motion.size < 3 or high_motion.size < 3:
                    continue
                diff = float(high_motion.mean() - low_motion.mean())
                pooled = pd.concat([low_motion, high_motion]).std()
                standardized = diff / float(pooled) if np.isfinite(pooled) and pooled > 0 else float("nan")
                diffs.append(diff)
                if np.isfinite(standardized):
                    std_diffs.append(standardized)
                per_rows.append(
                    {
                        "session": session,
                        "motion_metric": motion,
                        "sf_metric": sf,
                        "low_quantile": quantile,
                        "high_quantile": 1.0 - quantile,
                        "n_low": int(low_motion.size),
                        "n_high": int(high_motion.size),
                        "low_mean_motion": float(low_motion.mean()),
                        "high_mean_motion": float(high_motion.mean()),
                        "high_minus_low_motion": diff,
                        "high_minus_low_standardized": standardized,
                    }
                )
            arr = np.asarray(diffs, dtype=np.float64)
            std_arr = np.asarray(std_diffs, dtype=np.float64)
            ci_low, ci_high = _bootstrap_ci(arr, rng, n_boot)
            std_ci_low, std_ci_high = _bootstrap_ci(std_arr, rng, n_boot)
            n_pos = int(np.count_nonzero(arr > 0.0))
            sign_p = float(stats.binomtest(n_pos, n=arr.size, p=0.5).pvalue) if arr.size else float("nan")
            wilcoxon_p = float(stats.wilcoxon(arr, alternative="two-sided").pvalue) if arr.size > 0 and np.any(arr != 0) else float("nan")
            summary_rows.append(
                {
                    "motion_metric": motion,
                    "sf_metric": sf,
                    "n_sessions": int(arr.size),
                    "mean_high_minus_low_motion": float(np.mean(arr)) if arr.size else float("nan"),
                    "median_high_minus_low_motion": float(np.median(arr)) if arr.size else float("nan"),
                    "ci95_low_session_mean_diff": ci_low,
                    "ci95_high_session_mean_diff": ci_high,
                    "mean_high_minus_low_standardized": float(np.mean(std_arr)) if std_arr.size else float("nan"),
                    "median_high_minus_low_standardized": float(np.median(std_arr)) if std_arr.size else float("nan"),
                    "ci95_low_session_mean_standardized": std_ci_low,
                    "ci95_high_session_mean_standardized": std_ci_high,
                    "n_positive_sessions": n_pos,
                    "sign_test_p": sign_p,
                    "wilcoxon_p": wilcoxon_p,
                }
            )
    per_session = pd.DataFrame(per_rows)
    summary = pd.DataFrame(summary_rows)
    summary["sign_test_q_bh"] = _bh_q(summary["sign_test_p"].to_numpy(dtype=np.float64))
    summary["wilcoxon_q_bh"] = _bh_q(summary["wilcoxon_p"].to_numpy(dtype=np.float64))
    return per_session, summary


def _plot_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    block = summary.pivot(index="motion_metric", columns="sf_metric", values="residual_spearman_rho").loc[MOTION_METRICS]
    fig, ax = plt.subplots(figsize=(12.5, 4.2), constrained_layout=True)
    vmax = max(0.08, float(np.nanmax(np.abs(block.to_numpy(dtype=np.float64)))))
    im = ax.imshow(block.to_numpy(dtype=np.float64), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(block.shape[1]), block.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(block.shape[0]), block.index)
    for iy in range(block.shape[0]):
        for ix in range(block.shape[1]):
            val = block.iloc[iy, ix]
            ax.text(ix, iy, f"{val:+.3f}", ha="center", va="center", fontsize=7.0)
    ax.set_title("Residual window-level Spearman correlation: motion scale vs local spatial frequency")
    fig.colorbar(im, ax=ax, label="rho after session/phase/control residualization")
    fig.savefig(out_dir / "motion_scale_spatial_frequency_residual_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_focus_dose(dose: pd.DataFrame, out_dir: Path) -> None:
    block = dose[dose["motion_metric"].isin(FOCUS_MOTION) & dose["sf_metric"].isin(FOCUS_SF)].copy()
    block["motion_metric"] = pd.Categorical(block["motion_metric"], FOCUS_MOTION, ordered=True)
    block["sf_metric"] = pd.Categorical(block["sf_metric"], FOCUS_SF, ordered=True)
    block = block.sort_values(["motion_metric", "sf_metric"])
    fig, axes = plt.subplots(len(FOCUS_MOTION), 1, figsize=(10.5, 8.2), sharex=True, constrained_layout=True)
    for ax, motion in zip(np.atleast_1d(axes), FOCUS_MOTION, strict=False):
        sub = block[block["motion_metric"] == motion].copy()
        y = np.arange(sub.shape[0])
        vals = sub["mean_high_minus_low_standardized"].to_numpy(dtype=np.float64)
        lo = sub["ci95_low_session_mean_standardized"].to_numpy(dtype=np.float64)
        hi = sub["ci95_high_session_mean_standardized"].to_numpy(dtype=np.float64)
        colors = np.where(vals >= 0, "#2f8f6a", "#bf5b4b")
        ax.barh(y, vals, color=colors, alpha=0.9)
        ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none", ecolor="black", capsize=2, linewidth=1.0)
        ax.axvline(0.0, color="0.25", linewidth=0.9)
        ax.set_yticks(y, sub["sf_metric"].astype(str))
        ax.set_title(motion)
        ax.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("Standardized session mean motion in top SF tertile minus bottom SF tertile")
    fig.savefig(out_dir / "motion_scale_spatial_frequency_high_low_dose_focus.png", dpi=180)
    plt.close(fig)


def _write_readme(out_dir: Path, input_path: Path, df: pd.DataFrame) -> None:
    text = f"""# BackImage Motion Scale / Spatial Frequency Audit

Input table:

```text
{input_path}
```

Scope:

```text
n_windows = {df.shape[0]}
n_sessions = {df['session'].nunique()}
```

Primary outputs:

- `window_residual_correlation_summary.csv`
- `per_session_spearman_correlations.csv`
- `session_spearman_summary.csv`
- `per_session_high_low_dose.csv`
- `high_low_dose_summary.csv`
- `motion_scale_spatial_frequency_residual_correlation_heatmap.png`
- `motion_scale_spatial_frequency_high_low_dose_focus.png`

Interpretation notes:

- Window residual correlations remove session, phase, and basic window/patch
  controls with fixed effects before correlating the residuals.
- Session summaries treat each recording session as the inference unit.
- High-low dose summaries compare top versus bottom within-session spatial
  frequency tertiles, then summarize those session-wise differences.
- `sf_centroid_cpd_approx` assigns the open-ended 8+ cpd bin to 12 cpd only as
  a descriptive convention.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    df = _prepare_table(Path(args.input))
    sf_metrics = SF_BAND_METRICS + [col for col in SF_SLOPE_METRICS if col in df.columns] + [
        "sf_low_0_4_fraction",
        "sf_high_4plus_fraction",
        "sf_mid_2_8_fraction",
        "sf_high_vs_low_fraction",
        "sf_high_low_log_ratio",
        "sf_centroid_cpd_approx",
    ]

    window_summary = _window_correlation_summary(df, sf_metrics)
    per_session, session_summary = _session_correlation_summary(df, sf_metrics, rng=rng, n_boot=int(args.n_bootstrap))
    per_dose, dose_summary = _dose_response_summary(
        df,
        sf_metrics,
        rng=rng,
        n_boot=int(args.n_bootstrap),
        quantile=float(args.dose_quantile),
    )

    window_summary.to_csv(out_dir / "window_residual_correlation_summary.csv", index=False)
    per_session.to_csv(out_dir / "per_session_spearman_correlations.csv", index=False)
    session_summary.to_csv(out_dir / "session_spearman_summary.csv", index=False)
    per_dose.to_csv(out_dir / "per_session_high_low_dose.csv", index=False)
    dose_summary.to_csv(out_dir / "high_low_dose_summary.csv", index=False)
    _plot_heatmap(window_summary, out_dir)
    _plot_focus_dose(dose_summary, out_dir)
    _write_readme(out_dir, Path(args.input), df)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "out_dir": str(out_dir),
                "n_windows": int(df.shape[0]),
                "n_sessions": int(df["session"].nunique()),
                "motion_metrics": MOTION_METRICS,
                "spatial_frequency_metrics": sf_metrics,
                "control_metrics": CONTROL_METRICS,
                "n_bootstrap": int(args.n_bootstrap),
                "dose_quantile": float(args.dose_quantile),
                "seed": int(args.seed),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote motion/SF audit for {df.shape[0]} windows to {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--dose-quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
