"""Cheap synthesis summaries for the BackImage twin stability metric audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_AUDIT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_twin_stability_metric_audit"
)
DEFAULT_STABILITY_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_edge_parallel_stability_screen_yfix_n256_pop256"
)
DEFAULT_OUT_DIR = DEFAULT_AUDIT_DIR / "cheap_synthesis"


def _session_bootstrap_mean(
    df: pd.DataFrame,
    value_col: str,
    *,
    session_col: str = "session_id",
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    sessions = np.asarray(sorted(df[session_col].dropna().unique()))
    sess_mean = df.groupby(session_col)[value_col].mean().reindex(sessions).to_numpy(dtype=np.float64)
    sess_mean = sess_mean[np.isfinite(sess_mean)]
    if sess_mean.size == 0:
        return {"mean_session": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_sessions": 0}
    if int(n_bootstrap) <= 0 or sess_mean.size < 2:
        return {
            "mean_session": float(np.nanmean(sess_mean)),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_sessions": int(sess_mean.size),
        }
    draws = rng.choice(sess_mean, size=(int(n_bootstrap), sess_mean.size), replace=True)
    return {
        "mean_session": float(np.nanmean(sess_mean)),
        "ci_low": float(np.nanpercentile(np.nanmean(draws, axis=1), 2.5)),
        "ci_high": float(np.nanpercentile(np.nanmean(draws, axis=1), 97.5)),
        "n_sessions": int(sess_mean.size),
    }


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 3:
        return float("nan")
    if np.nanstd(x[ok]) <= 1e-12 or np.nanstd(y[ok]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _within_session_demean(values: pd.Series, sessions: pd.Series) -> np.ndarray:
    return (values.astype(float) - values.astype(float).groupby(sessions).transform("mean")).to_numpy(dtype=np.float64)


def _bootstrap_corr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    session_col: str = "session_id",
    within_session: bool,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    sessions = np.asarray(sorted(df[session_col].dropna().unique()))
    if within_session:
        x = _within_session_demean(df[x_col], df[session_col])
        y = _within_session_demean(df[y_col], df[session_col])
        observed = _corr(x, y)
        session_arrays = []
        session_values = df[session_col].to_numpy()
        for sess in sessions:
            idx = session_values == sess
            session_arrays.append((x[idx], y[idx]))
    else:
        sess = df.groupby(session_col)[[x_col, y_col]].mean()
        sx = sess[x_col].to_numpy(dtype=np.float64)
        sy = sess[y_col].to_numpy(dtype=np.float64)
        observed = _corr(sx, sy)
    if int(n_bootstrap) <= 0 or sessions.size < 2:
        return {"r": observed, "ci_low": float("nan"), "ci_high": float("nan")}
    vals = []
    n_sessions = int(sessions.size)
    for _ in range(int(n_bootstrap)):
        draw = rng.integers(0, n_sessions, size=n_sessions)
        if within_session:
            bx = np.concatenate([session_arrays[j][0] for j in draw])
            by = np.concatenate([session_arrays[j][1] for j in draw])
            vals.append(_corr(bx, by))
        else:
            vals.append(_corr(sx[draw], sy[draw]))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "r": observed,
        "ci_low": float(np.nanpercentile(arr, 2.5)),
        "ci_high": float(np.nanpercentile(arr, 97.5)),
    }


def _metric_names(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if not col.endswith("_stability_advantage"):
            continue
        metric = col[: -len("_stability_advantage")]
        if metric in {"pixel", "twin", "edge_parallel"}:
            continue
        if col.startswith("edge_parallel_"):
            continue
        if f"{metric}_parallel_cost" in df.columns and f"{metric}_orthogonal_cost" in df.columns:
            out.append(metric)
    return sorted(out)


def _write_first_order(df: pd.DataFrame, metrics: list[str], out_dir: Path, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_specs = [("pixel", "pixel_stability_advantage")]
    if "twin_stability_advantage" in df.columns:
        base_specs.append(("original_twin_relative_screen_metric_raw_mse", "twin_stability_advantage"))
    for metric in metrics:
        base_specs.append((metric, f"{metric}_stability_advantage"))
    for metric, col in base_specs:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=["session_id", col]).copy()
        ci = _session_bootstrap_mean(sub, col, rng=rng, n_bootstrap=n_bootstrap)
        rows.append(
            {
                "metric": metric,
                "advantage_column": col,
                "mean_window": float(sub[col].mean()),
                **ci,
                "n_windows": int(sub.shape[0]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "first_order_signed_stability_advantage_session_ci.csv", index=False)
    return out


def _write_pixel_twin_corr(df: pd.DataFrame, metrics: list[str], out_dir: Path, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = []
    if "twin_stability_advantage" in df.columns:
        specs.append(("original_twin_raw_mse", "twin_stability_advantage"))
    specs.extend((metric, f"{metric}_stability_advantage") for metric in metrics)
    for metric, twin_col in specs:
        if twin_col not in df.columns:
            continue
        sub = df.dropna(subset=["session_id", "pixel_stability_advantage", twin_col]).copy()
        for level, within in (("window_within_session", True), ("session_mean", False)):
            corr = _bootstrap_corr(
                sub,
                "pixel_stability_advantage",
                twin_col,
                within_session=within,
                rng=rng,
                n_bootstrap=n_bootstrap,
            )
            rows.append(
                {
                    "metric": metric,
                    "twin_column": twin_col,
                    "level": level,
                    **corr,
                    "n_windows": int(sub.shape[0]),
                    "n_sessions": int(sub["session_id"].nunique()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "pixel_vs_twin_signed_stability_correlations.csv", index=False)
    return out


def _write_scale_summary(
    df: pd.DataFrame,
    first_order: pd.DataFrame,
    stability_dir: Path,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_path = stability_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"config": {}}
    displacement_deg = float(metadata.get("config", {}).get("displacement_deg", np.nan))
    rows = []
    if "rms_radius_deg" in df.columns:
        rms = df["rms_radius_deg"].astype(float)
        rows.append(
            {
                "quantity": "rms_radius_deg",
                "fixed_endpoint_displacement_deg": displacement_deg,
                "mean": float(np.nanmean(rms)),
                "median": float(np.nanmedian(rms)),
                "q25": float(np.nanpercentile(rms, 25)),
                "q75": float(np.nanpercentile(rms, 75)),
                "q90": float(np.nanpercentile(rms, 90)),
                "fraction_within_half_to_2x_endpoint": float(np.nanmean((rms >= 0.5 * displacement_deg) & (rms <= 2.0 * displacement_deg))) if np.isfinite(displacement_deg) else float("nan"),
                "fraction_below_endpoint": float(np.nanmean(rms <= displacement_deg)) if np.isfinite(displacement_deg) else float("nan"),
                "n_windows": int(rms.notna().sum()),
            }
        )
        bins = pd.qcut(rms, q=3, labels=["low_rms", "middle_rms", "high_rms"], duplicates="drop")
        metric_cols = [
            "pixel_stability_advantage",
            "raw_mse_stability_advantage",
            "response_norm_mse_stability_advantage",
            "per_rate_mse_stability_advantage",
            "full_cov_whitened_mse_stability_advantage",
        ]
        bin_rows = []
        for level, block in df.assign(rms_radius_bin=bins).groupby("rms_radius_bin", observed=True):
            rec: dict[str, Any] = {
                "rms_radius_bin": str(level),
                "rms_min": float(block["rms_radius_deg"].min()),
                "rms_max": float(block["rms_radius_deg"].max()),
                "rms_median": float(block["rms_radius_deg"].median()),
                "n_windows": int(block.shape[0]),
                "n_sessions": int(block["session_id"].nunique()),
            }
            for col in metric_cols:
                if col in block.columns:
                    rec[f"{col}_mean_session"] = float(block.groupby("session_id")[col].mean().mean())
            bin_rows.append(rec)
        pd.DataFrame(bin_rows).to_csv(out_dir / "stability_advantage_by_observed_rms_bin.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "observed_drift_scale_summary.csv", index=False)
    scale_note = {
        "fixed_endpoint_displacement_deg": displacement_deg,
        "interpretation": (
            "This audit has endpoint-cache twin responses at one displacement only, so it cannot prove a peak over scale. "
            "It can only ask whether that endpoint displacement lies in the observed fixation-scale range and whether "
            "signed edge-parallel stability is still positive across observed RMS-radius bins."
        ),
    }
    (out_dir / "scale_interpretation.json").write_text(json.dumps(scale_note, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out, scale_note


def _write_plots(out_dir: Path, first_order: pd.DataFrame, corr: pd.DataFrame) -> None:
    focus = first_order[first_order["metric"].isin(["pixel", "raw_mse", "response_norm_mse", "per_rate_mse", "diag_whitened_mse", "full_cov_whitened_mse"])].copy()
    focus = focus.sort_values("mean_session")
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=150)
    y = np.arange(focus.shape[0])
    ax.barh(y, focus["mean_session"], color=["#4878a8" if m == "pixel" else "#c16622" for m in focus["metric"]])
    ax.errorbar(
        focus["mean_session"],
        y,
        xerr=[focus["mean_session"] - focus["ci_low"], focus["ci_high"] - focus["mean_session"]],
        fmt="none",
        color="black",
        linewidth=0.8,
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(focus["metric"], fontsize=8)
    ax.set_xlabel("session-mean signed advantage: orthogonal - parallel")
    fig.tight_layout()
    fig.savefig(out_dir / "first_order_signed_stability_advantage_ci.png", dpi=150)
    plt.close(fig)

    cfocus = corr[(corr["level"] == "window_within_session") & corr["metric"].isin(["raw_mse", "response_norm_mse", "per_rate_mse", "diag_whitened_mse", "full_cov_whitened_mse"])].copy()
    cfocus = cfocus.sort_values("r")
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=150)
    y = np.arange(cfocus.shape[0])
    ax.barh(y, cfocus["r"], color="#6a8f5f")
    ax.errorbar(cfocus["r"], y, xerr=[cfocus["r"] - cfocus["ci_low"], cfocus["ci_high"] - cfocus["r"]], fmt="none", color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(cfocus["metric"], fontsize=8)
    ax.set_xlabel("pixel vs twin signed advantage correlation")
    fig.tight_layout()
    fig.savefig(out_dir / "pixel_vs_twin_signed_stability_correlations.png", dpi=150)
    plt.close(fig)


def _write_report(out_dir: Path, first_order: pd.DataFrame, corr: pd.DataFrame, scale: pd.DataFrame) -> None:
    def row(metric: str) -> pd.Series | None:
        hit = first_order[first_order["metric"] == metric]
        return hit.iloc[0] if not hit.empty else None

    lines = [
        "# BackImage Twin Stability Cheap Synthesis",
        "",
        "## First-Order Signed Stability",
        "",
        "Positive values mean edge-orthogonal displacement disrupts more than edge-parallel displacement.",
        "",
    ]
    for metric in ["pixel", "raw_mse", "response_norm_mse", "per_rate_mse", "diag_whitened_mse", "full_cov_whitened_mse", "other_units_raw_mse"]:
        r = row(metric)
        if r is None:
            continue
        lines.append(
            f"- `{metric}`: session mean `{r['mean_session']:+.4g}` CI "
            f"`[{r['ci_low']:+.4g}, {r['ci_high']:+.4g}]`, n `{int(r['n_windows'])}` windows."
        )
    lines.extend(["", "## Pixel-Twin Agreement", ""])
    cfocus = corr[(corr["level"] == "window_within_session") & corr["metric"].isin(["raw_mse", "response_norm_mse", "per_rate_mse", "diag_whitened_mse", "full_cov_whitened_mse"])].copy()
    for _, r in cfocus.sort_values("r", ascending=False).iterrows():
        lines.append(f"- Window within-session, `{r['metric']}`: r `{r['r']:+.3f}` CI `[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]`.")
    sfocus = corr[(corr["level"] == "session_mean") & corr["metric"].isin(["raw_mse", "response_norm_mse", "per_rate_mse", "diag_whitened_mse", "full_cov_whitened_mse"])].copy()
    lines.append("")
    for _, r in sfocus.sort_values("r", ascending=False).iterrows():
        lines.append(f"- Session mean, `{r['metric']}`: r `{r['r']:+.3f}` CI `[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]`.")
    if not scale.empty:
        s = scale.iloc[0]
        lines.extend(
            [
                "",
                "## Scale Check",
                "",
                f"- Twin endpoint displacement in this cache: `{s['fixed_endpoint_displacement_deg']:.4f}` deg.",
                f"- Observed RMS radius: median `{s['median']:.4f}` deg, IQR `[{s['q25']:.4f}, {s['q75']:.4f}]`, q90 `{s['q90']:.4f}` deg.",
                f"- Fraction of audited windows within 0.5x-2x endpoint displacement: `{s['fraction_within_half_to_2x_endpoint']:.3f}`.",
                "",
                "This one-scale endpoint cache cannot establish a scale optimum. It does say whether the tested edge-parallel displacement lies in the observed fixation-motion regime.",
            ]
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "The strongest defensible claim is first-order preservation: at the tested small displacement, edge-parallel motion tends to disrupt pixels and multiple twin signed metrics less than edge-orthogonal motion. Pixel-twin agreement should determine how much we trust the twin as a local-structure readout here.",
            "",
        ]
    )
    (out_dir / "cheap_synthesis_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    audit_dir = Path(args.audit_dir)
    df = pd.read_csv(audit_dir / "twin_stability_metric_by_window.csv")
    metrics = _metric_names(df)
    first_order = _write_first_order(df, metrics, out_dir, rng, int(args.n_bootstrap))
    corr = _write_pixel_twin_corr(df, metrics, out_dir, rng, int(args.n_bootstrap))
    scale, _ = _write_scale_summary(df, first_order, Path(args.stability_dir), out_dir)
    _write_plots(out_dir, first_order, corr)
    _write_report(out_dir, first_order, corr, scale)
    (out_dir / "summary_metadata.json").write_text(
        json.dumps(
            {
                "audit_dir": str(audit_dir),
                "stability_dir": str(args.stability_dir),
                "n_bootstrap": int(args.n_bootstrap),
                "seed": int(args.seed),
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote BackImage twin stability cheap synthesis to {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--stability-dir", type=Path, default=DEFAULT_STABILITY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
