#!/usr/bin/env python3
"""Compare BackImage drift alignment to average-edge and WTA local axes.

This is a lightweight screening diagnostic for the Figure 4D axis-estimator
question. It does not rerun model-response decoding. Instead, it asks whether
the recorded drift orientation is better aligned to the stored patch-average
orientation-energy axis or to a winner-take-all prominent local orientation
estimated from the same BackImage patch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

try:
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = REPO_ROOT / "outputs" / "fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_WINDOWS_CSV = BASE / "backimage_image_structure_reviewed_v2_screenfiltered_yfix" / "backimage_image_fem_windows.csv"
DEFAULT_OUT_DIR = BASE / "backimage_wta_orientation_behavior_diagnostic_v1"

INK = "#20262c"
MUTED = "#68727d"
GRID = "#dfe4e9"
AVERAGE_COLOR = "#244f7a"
WTA_COLOR = "#c15b44"
DELTA_COLOR = "#2f8f6a"


@dataclass(frozen=True)
class Config:
    windows_csv: str
    out_dir: str
    sessions: str
    max_windows: int
    seed: int
    n_bins: int
    energy_quantile: float
    min_peak_fraction: float
    reliable_image_coherence_min: float
    reliable_drift_anisotropy_min: float
    high_image_coherence_min: float
    min_duration_s: float
    n_bootstrap: int
    n_permutations: int


def _axis_delta_deg(a_deg: np.ndarray | float, b_deg: np.ndarray | float) -> np.ndarray:
    return 0.5 * np.degrees(np.angle(np.exp(2j * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))))


def _cos2_delta(a_deg: np.ndarray | float, b_deg: np.ndarray | float) -> np.ndarray:
    return np.cos(2.0 * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))


def _axial_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    angles = np.radians(np.asarray(angles_deg, dtype=np.float64))
    weights = np.asarray(weights, dtype=np.float64)
    if angles.size == 0 or not np.isfinite(weights).any() or float(np.nansum(weights)) <= 0:
        return float("nan")
    z = np.nansum(weights * np.exp(2j * angles))
    if not np.isfinite(z.real) or not np.isfinite(z.imag) or abs(z) <= 0:
        return float("nan")
    return float(0.5 * np.degrees(np.angle(z)))


def _crop_patch(row: pd.Series) -> tuple[np.ndarray, float, tuple[float, float]]:
    canvas, ppd, screen_shape = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    center_x = float(row.get("image_patch_center_x_px", np.nan))
    center_y = float(row.get("image_patch_center_y_px", np.nan))
    if np.isfinite(center_x) and np.isfinite(center_y):
        center = np.asarray([center_x, center_y], dtype=np.float64)
    else:
        center = gaze_deg_to_screen_px(
            np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])], dtype=np.float64),
            ppd=float(ppd),
            screen_shape=screen_shape,
        )
    radius_value = float(row.get("image_patch_radius_px", 0))
    if not np.isfinite(radius_value) or radius_value <= 0:
        rad = max(2, int(round(1.0 * float(ppd))))
    else:
        rad = int(round(radius_value))
    height, width = canvas.shape[:2]
    x0 = max(0, int(round(float(center[0]))) - rad)
    x1 = min(width, int(round(float(center[0]))) + rad + 1)
    y0 = max(0, int(round(float(center[1]))) - rad)
    y1 = min(height, int(round(float(center[1]))) + rad + 1)
    return np.asarray(canvas[y0:y1, x0:x1], dtype=np.float64), float(ppd), (float(center[0]), float(center[1]))


def _wta_axis_from_patch(
    patch: np.ndarray,
    *,
    n_bins: int,
    energy_quantile: float,
) -> dict[str, float]:
    if patch.size < 16:
        return {"wta_ok": 0.0, "wta_error": "patch_too_small"}

    gx = ndimage.sobel(patch, axis=1, mode="nearest")
    gy = ndimage.sobel(patch, axis=0, mode="nearest")
    energy = gx * gx + gy * gy
    finite = np.isfinite(energy)
    if not finite.any() or float(np.nanmax(energy)) <= 0:
        return {"wta_ok": 0.0, "wta_error": "no_gradient_energy"}

    threshold = float(np.nanquantile(energy[finite], float(energy_quantile)))
    mask = finite & (energy >= threshold) & (energy > 0)
    if np.count_nonzero(mask) < 8:
        mask = finite & (energy > 0)
    if np.count_nonzero(mask) < 8:
        return {"wta_ok": 0.0, "wta_error": "too_few_edge_pixels"}

    gradient_axis_array = np.degrees(np.arctan2(gy[mask], gx[mask]))
    edge_axis_array = ((gradient_axis_array + 90.0 + 90.0) % 180.0) - 90.0
    weights = energy[mask].astype(np.float64)
    bin_edges = np.linspace(-90.0, 90.0, int(n_bins) + 1)
    hist, _ = np.histogram(edge_axis_array, bins=bin_edges, weights=weights)
    total = float(np.sum(hist))
    if total <= 0:
        return {"wta_ok": 0.0, "wta_error": "empty_orientation_histogram"}

    peak_bin = int(np.argmax(hist))
    peak_fraction = float(hist[peak_bin] / total)
    lo, hi = float(bin_edges[peak_bin]), float(bin_edges[peak_bin + 1])
    in_peak = (edge_axis_array >= lo) & (edge_axis_array < hi)
    if peak_bin == len(hist) - 1:
        in_peak = (edge_axis_array >= lo) & (edge_axis_array <= hi)

    if np.count_nonzero(in_peak) >= 3:
        edge_axis_array_wta = _axial_mean_deg(edge_axis_array[in_peak], weights[in_peak])
    else:
        edge_axis_array_wta = (lo + hi) / 2.0

    all_axis_array_mean = _axial_mean_deg(edge_axis_array, weights)
    return {
        "wta_ok": 1.0,
        "wta_error": "",
        "wta_edge_axis_array_deg": edge_axis_array_wta,
        "wta_edge_axis_deg": -edge_axis_array_wta,
        "wta_hist_peak_bin": float(peak_bin),
        "wta_hist_peak_low_deg_array": lo,
        "wta_hist_peak_high_deg_array": hi,
        "wta_peak_fraction": peak_fraction,
        "wta_total_gradient_energy": total,
        "wta_n_edge_pixels": float(np.count_nonzero(mask)),
        "wta_energy_quantile_threshold": threshold,
        "wta_all_energy_edge_axis_array_deg": all_axis_array_mean,
        "wta_all_energy_edge_axis_deg": -all_axis_array_mean,
    }


def _load_windows(path: Path, *, sessions: str, max_windows: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = (
        df["image_feature_ok"].astype(bool)
        & np.isfinite(pd.to_numeric(df["image_edge_axis_deg"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df["drift_orientation_deg"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df["drift_edge_cos2"], errors="coerce"))
    )
    work = df.loc[keep].copy()
    requested_sessions = [s.strip() for s in sessions.split(",") if s.strip()]
    if requested_sessions:
        work = work[work["session"].astype(str).isin(requested_sessions)].copy()
    work["source_window_row"] = work.index.astype(int)
    if max_windows > 0 and work.shape[0] > max_windows:
        work = work.sample(n=int(max_windows), replace=False, random_state=int(seed)).sort_index().copy()
    return work.reset_index(drop=True)


def _augment_with_wta(work: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    work = work.copy()
    work["_wta_patch_x_round"] = pd.to_numeric(work["image_patch_center_x_px"], errors="coerce").round().astype("Int64")
    work["_wta_patch_y_round"] = pd.to_numeric(work["image_patch_center_y_px"], errors="coerce").round().astype("Int64")
    work["_wta_patch_radius_round"] = pd.to_numeric(work["image_patch_radius_px"], errors="coerce").round().astype("Int64")
    key_cols = ["session", "trial_idx", "_wta_patch_x_round", "_wta_patch_y_round", "_wta_patch_radius_round"]
    patch_work = work.drop_duplicates(key_cols).sort_values(["session", "trial_idx", "_wta_patch_x_round", "_wta_patch_y_round"])

    records: list[dict[str, Any]] = []
    n = int(patch_work.shape[0])
    last_report = time.time()
    for processed, (_, row) in enumerate(patch_work.iterrows(), start=1):
        rec = {col: row[col] for col in key_cols}
        try:
            patch, ppd, center = _crop_patch(row)
            rec["wta_patch_height_px"] = int(patch.shape[0])
            rec["wta_patch_width_px"] = int(patch.shape[1])
            rec["wta_ppd"] = float(ppd)
            rec["wta_center_x_px"] = center[0]
            rec["wta_center_y_px"] = center[1]
            rec.update(
                _wta_axis_from_patch(
                    patch,
                    n_bins=int(cfg.n_bins),
                    energy_quantile=float(cfg.energy_quantile),
                )
            )
        except Exception as exc:  # pragma: no cover - data availability failures are recorded.
            rec.update({"wta_ok": 0.0, "wta_error": str(exc)})
        records.append(rec)
        if processed == n or time.time() - last_report > 20.0:
            print(f"processed {processed}/{n} unique patches", flush=True)
            last_report = time.time()

    wta = pd.DataFrame(records)
    out = work.merge(wta, on=key_cols, how="left")
    out = out.drop(columns=[col for col in key_cols if col.startswith("_wta_")])
    for col in ["wta_edge_axis_deg", "wta_peak_fraction", "wta_average_axis_delta_deg"]:
        if col not in out:
            out[col] = np.nan
    out["wta_ok"] = out["wta_ok"].astype(float) > 0
    for col in ["image_edge_axis_deg", "wta_edge_axis_deg", "drift_orientation_deg"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["drift_average_axis_delta_deg"] = _axis_delta_deg(out["drift_orientation_deg"], out["image_edge_axis_deg"])
    out["drift_wta_axis_delta_deg"] = _axis_delta_deg(out["drift_orientation_deg"], out["wta_edge_axis_deg"])
    out["drift_average_cos2"] = _cos2_delta(out["drift_orientation_deg"], out["image_edge_axis_deg"])
    out["drift_wta_cos2"] = _cos2_delta(out["drift_orientation_deg"], out["wta_edge_axis_deg"])
    out["wta_minus_average_cos2"] = out["drift_wta_cos2"] - out["drift_average_cos2"]
    out["wta_average_axis_delta_deg"] = np.abs(_axis_delta_deg(out["wta_edge_axis_deg"], out["image_edge_axis_deg"]))
    out["wta_minus_average_abs_delta_deg"] = (
        np.abs(out["drift_wta_axis_delta_deg"]) - np.abs(out["drift_average_axis_delta_deg"])
    )
    return out


def _subset_masks(df: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    duration = pd.to_numeric(df["duration_s"] if "duration_s" in df else df.get("epoch_duration_s", np.nan), errors="coerce")
    base = (
        df["wta_ok"].astype(bool)
        & np.isfinite(df["drift_average_cos2"].astype(float))
        & np.isfinite(df["drift_wta_cos2"].astype(float))
    )
    reliable = (
        base
        & (pd.to_numeric(df["image_orientation_coherence"], errors="coerce") >= cfg.reliable_image_coherence_min)
        & (pd.to_numeric(df["anisotropy"], errors="coerce") >= cfg.reliable_drift_anisotropy_min)
        & (duration >= cfg.min_duration_s)
    )
    high_coherence = reliable & (
        pd.to_numeric(df["image_orientation_coherence"], errors="coerce") >= cfg.high_image_coherence_min
    )
    strong_wta = reliable & (pd.to_numeric(df["wta_peak_fraction"], errors="coerce") >= cfg.min_peak_fraction)
    estimator_disagreement = reliable & (pd.to_numeric(df["wta_average_axis_delta_deg"], errors="coerce") >= 10.0)
    return {
        "all_valid": base,
        "reliable": reliable,
        "high_image_coherence": high_coherence,
        "strong_wta_peak": strong_wta,
        "estimator_disagreement_ge10deg": estimator_disagreement,
    }


def _bootstrap_ci(values: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or n_bootstrap <= 0:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(int(n_bootstrap), values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _sign_flip_p(values: np.ndarray, *, n_permutations: int, rng: np.random.Generator) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    if observed <= 0:
        return 1.0
    if values.size == 1 or n_permutations <= 0:
        return float("nan")
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(n_permutations), values.size), replace=True)
    null = np.abs(signs * values).mean(axis=1)
    return float((np.count_nonzero(null >= observed) + 1.0) / (null.size + 1.0))


def _summarize(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(cfg.seed))
    summary_rows: list[dict[str, Any]] = []
    session_rows: list[pd.DataFrame] = []
    for subset, mask in _subset_masks(df, cfg).items():
        block = df.loc[mask].copy()
        if block.empty:
            summary_rows.append({"subset": subset, "n_windows": 0, "n_sessions": 0})
            continue
        by_session = (
            block.groupby("session", observed=True)
            .agg(
                n_windows=("session", "size"),
                average_cos2=("drift_average_cos2", "mean"),
                wta_cos2=("drift_wta_cos2", "mean"),
                wta_minus_average_cos2=("wta_minus_average_cos2", "mean"),
                average_abs_delta_deg=("drift_average_axis_delta_deg", lambda x: float(np.mean(np.abs(x)))),
                wta_abs_delta_deg=("drift_wta_axis_delta_deg", lambda x: float(np.mean(np.abs(x)))),
                wta_peak_fraction=("wta_peak_fraction", "mean"),
                wta_average_axis_delta_deg=("wta_average_axis_delta_deg", "mean"),
            )
            .reset_index()
        )
        by_session.insert(0, "subset", subset)
        session_rows.append(by_session)

        session_delta = by_session["wta_minus_average_cos2"].to_numpy(dtype=np.float64)
        delta_lo, delta_hi = _bootstrap_ci(session_delta, n_bootstrap=cfg.n_bootstrap, rng=rng)
        avg_lo, avg_hi = _bootstrap_ci(by_session["average_cos2"].to_numpy(dtype=np.float64), n_bootstrap=cfg.n_bootstrap, rng=rng)
        wta_lo, wta_hi = _bootstrap_ci(by_session["wta_cos2"].to_numpy(dtype=np.float64), n_bootstrap=cfg.n_bootstrap, rng=rng)
        summary_rows.append(
            {
                "subset": subset,
                "n_windows": int(block.shape[0]),
                "n_sessions": int(by_session.shape[0]),
                "window_mean_average_cos2": float(block["drift_average_cos2"].mean()),
                "window_mean_wta_cos2": float(block["drift_wta_cos2"].mean()),
                "window_mean_wta_minus_average_cos2": float(block["wta_minus_average_cos2"].mean()),
                "session_mean_average_cos2": float(by_session["average_cos2"].mean()),
                "session_mean_average_cos2_ci_low": avg_lo,
                "session_mean_average_cos2_ci_high": avg_hi,
                "session_mean_wta_cos2": float(by_session["wta_cos2"].mean()),
                "session_mean_wta_cos2_ci_low": wta_lo,
                "session_mean_wta_cos2_ci_high": wta_hi,
                "session_mean_wta_minus_average_cos2": float(np.mean(session_delta)),
                "session_mean_wta_minus_average_cos2_ci_low": delta_lo,
                "session_mean_wta_minus_average_cos2_ci_high": delta_hi,
                "session_mean_wta_minus_average_cos2_signflip_p": _sign_flip_p(
                    session_delta,
                    n_permutations=cfg.n_permutations,
                    rng=rng,
                ),
                "window_fraction_wta_better": float(np.mean(block["wta_minus_average_cos2"] > 0)),
                "session_fraction_wta_better": float(np.mean(session_delta > 0)),
                "window_mean_average_abs_delta_deg": float(np.mean(np.abs(block["drift_average_axis_delta_deg"]))),
                "window_mean_wta_abs_delta_deg": float(np.mean(np.abs(block["drift_wta_axis_delta_deg"]))),
                "window_mean_wta_average_axis_delta_deg": float(block["wta_average_axis_delta_deg"].mean()),
                "window_mean_wta_peak_fraction": float(block["wta_peak_fraction"].mean()),
            }
        )
    session_df = pd.concat(session_rows, ignore_index=True) if session_rows else pd.DataFrame()
    return pd.DataFrame(summary_rows), session_df


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_summary(window_df: pd.DataFrame, summary: pd.DataFrame, session_df: pd.DataFrame, cfg: Config, out_dir: Path) -> Path:
    _configure_matplotlib()
    fig = plt.figure(figsize=(8.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])

    subsets = ["all_valid", "reliable", "high_image_coherence", "strong_wta_peak"]
    plot_summary = summary[summary["subset"].isin(subsets)].copy()
    x = np.arange(plot_summary.shape[0], dtype=float)
    width = 0.34
    avg = plot_summary["session_mean_average_cos2"].to_numpy(dtype=float)
    wta = plot_summary["session_mean_wta_cos2"].to_numpy(dtype=float)
    ax_bar.bar(x - width / 2.0, avg, width=width, color=AVERAGE_COLOR, label="average axis")
    ax_bar.bar(x + width / 2.0, wta, width=width, color=WTA_COLOR, label="WTA axis")
    ax_bar.axhline(0, color=INK, lw=0.8)
    ax_bar.set_ylabel("drift-axis alignment (cos 2Δ)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([s.replace("_", "\n") for s in plot_summary["subset"]])
    ax_bar.set_title("Behavior alignment by axis estimator")
    ax_bar.grid(axis="y", color=GRID, lw=0.8)
    ax_bar.legend(frameon=False, loc="upper left")
    _clean_axis(ax_bar)

    reliable_sessions = session_df[session_df["subset"] == "reliable"].copy()
    ax_delta.axhline(0, color=INK, lw=0.8)
    if not reliable_sessions.empty:
        order = reliable_sessions.sort_values("wta_minus_average_cos2")
        ax_delta.barh(
            np.arange(order.shape[0]),
            order["wta_minus_average_cos2"],
            color=[DELTA_COLOR if v >= 0 else "#8d96a0" for v in order["wta_minus_average_cos2"]],
        )
        ax_delta.set_yticks(np.arange(order.shape[0]))
        ax_delta.set_yticklabels(order["session"].astype(str), fontsize=6.5)
    ax_delta.set_xlabel("WTA - average alignment")
    ax_delta.set_title("Reliable subset, session paired")
    ax_delta.grid(axis="x", color=GRID, lw=0.8)
    _clean_axis(ax_delta)

    reliable = window_df.loc[_subset_masks(window_df, cfg).get("reliable", pd.Series(False, index=window_df.index))].copy()
    if not reliable.empty:
        ax_hist.hist(
            reliable["wta_average_axis_delta_deg"].dropna(),
            bins=np.linspace(0, 90, 31),
            color="#68727d",
            alpha=0.85,
        )
    ax_hist.set_xlabel("|WTA axis - average axis| (deg)")
    ax_hist.set_ylabel("windows")
    ax_hist.set_title("Estimator disagreement")
    ax_hist.grid(axis="y", color=GRID, lw=0.8)
    _clean_axis(ax_hist)

    if not reliable.empty:
        sample = reliable
        if sample.shape[0] > 3000:
            sample = sample.sample(n=3000, random_state=0)
        ax_scatter.scatter(
            sample["drift_average_cos2"],
            sample["drift_wta_cos2"],
            s=8,
            c=sample["wta_peak_fraction"],
            cmap="viridis",
            alpha=0.42,
            linewidths=0,
        )
    lim = (-1.02, 1.02)
    ax_scatter.plot(lim, lim, color=INK, lw=0.8)
    ax_scatter.set_xlim(lim)
    ax_scatter.set_ylim(lim)
    ax_scatter.set_xlabel("average-axis alignment")
    ax_scatter.set_ylabel("WTA-axis alignment")
    ax_scatter.set_title("Window-level comparison")
    ax_scatter.grid(color=GRID, lw=0.8)
    _clean_axis(ax_scatter)

    fig.suptitle("Panel 4D WTA orientation behavior diagnostic", x=0.02, ha="left", fontsize=12, fontweight="bold", color=INK)
    path = out_dir / "panel_d_wta_behavior_diagnostic.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _write_readme(out_dir: Path, cfg: Config, summary: pd.DataFrame, elapsed_s: float) -> None:
    reliable = summary[summary["subset"] == "reliable"]
    if reliable.empty:
        read = "Reliable subset was empty."
    else:
        row = reliable.iloc[0]
        read = (
            f"Reliable subset: average-axis alignment {row['session_mean_average_cos2']:+.4f}, "
            f"WTA-axis alignment {row['session_mean_wta_cos2']:+.4f}, "
            f"WTA-average {row['session_mean_wta_minus_average_cos2']:+.4f} "
            f"[{row['session_mean_wta_minus_average_cos2_ci_low']:+.4f}, "
            f"{row['session_mean_wta_minus_average_cos2_ci_high']:+.4f}], "
            f"session sign-flip p={row['session_mean_wta_minus_average_cos2_signflip_p']:.4f}."
        )
    lines = [
        "# Panel 4D WTA Orientation Behavior Diagnostic",
        "",
        "Purpose: quick behavior-side screen for whether recorded drift is better aligned to the stored patch-average edge axis or to a winner-take-all prominent local orientation axis.",
        "",
        read,
        "",
        f"Elapsed wall time: {elapsed_s:.1f} s.",
        "",
        "![Diagnostic plot](/home/declan/VisionCore/outputs/fixation_statistics_by_stimulus_all_sessions_after_review/backimage_wta_orientation_behavior_diagnostic_v1/panel_d_wta_behavior_diagnostic.png)",
        "",
        "## Files",
        "",
        "- `window_wta_orientation_behavior.csv`: per-window WTA axis and behavior alignment.",
        "- `summary_by_subset.csv`: window/session summaries and paired WTA-average intervals.",
        "- `session_summary_by_subset.csv`: session-level means used for paired summaries.",
        "- `run_metadata.json`: exact configuration.",
        "",
        "## Notes",
        "",
        "- This is not a model-response decoding rerun.",
        "- WTA axes are image-only and estimated before any behavior comparison.",
        "- Positive `WTA - average alignment` means the recorded drift axis is closer to the WTA prominent local orientation than to the patch-average orientation-energy axis.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(asdict(cfg), indent=2, sort_keys=True),
        "```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-csv", type=Path, default=DEFAULT_WINDOWS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sessions", default="", help="Optional comma-separated session names.")
    parser.add_argument("--max-windows", type=int, default=0, help="0 means all valid windows.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=36)
    parser.add_argument("--energy-quantile", type=float, default=0.75)
    parser.add_argument("--min-peak-fraction", type=float, default=0.12)
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--high-image-coherence-min", type=float, default=0.50)
    parser.add_argument("--min-duration-s", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config(
        windows_csv=str(args.windows_csv),
        out_dir=str(args.out_dir),
        sessions=str(args.sessions),
        max_windows=int(args.max_windows),
        seed=int(args.seed),
        n_bins=int(args.n_bins),
        energy_quantile=float(args.energy_quantile),
        min_peak_fraction=float(args.min_peak_fraction),
        reliable_image_coherence_min=float(args.reliable_image_coherence_min),
        reliable_drift_anisotropy_min=float(args.reliable_drift_anisotropy_min),
        high_image_coherence_min=float(args.high_image_coherence_min),
        min_duration_s=float(args.min_duration_s),
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
    )
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    metadata_path = out_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps({"config": asdict(cfg), "started_unix_s": start}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    windows = _load_windows(Path(cfg.windows_csv), sessions=cfg.sessions, max_windows=cfg.max_windows, seed=cfg.seed)
    print(f"loaded {windows.shape[0]} valid windows", flush=True)
    window_df = _augment_with_wta(windows, cfg)
    summary, session_df = _summarize(window_df, cfg)

    window_df.to_csv(out_dir / "window_wta_orientation_behavior.csv", index=False)
    summary.to_csv(out_dir / "summary_by_subset.csv", index=False)
    session_df.to_csv(out_dir / "session_summary_by_subset.csv", index=False)
    plot_path = _plot_summary(window_df, summary, session_df, cfg, out_dir)

    elapsed = time.time() - start
    metadata_path.write_text(
        json.dumps({"config": asdict(cfg), "elapsed_s": elapsed, "plot": str(plot_path)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_readme(out_dir, cfg, summary, elapsed)
    print(summary.to_string(index=False), flush=True)
    print(plot_path, flush=True)


if __name__ == "__main__":
    main()
