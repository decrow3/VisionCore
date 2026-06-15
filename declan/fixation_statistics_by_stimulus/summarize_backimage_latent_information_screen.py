"""Summarize a BackImage latent-information screen run.

This is a no-GPU post-processing pass for
``run_backimage_latent_information_screen``.  It works from the saved
per-window decode scores so the key comparisons are paired within fixation
window, then bootstrapped over sessions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_RUN_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_latent_information_screen_canonical_pyramid_gabor_dct_n256"
)


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    left: str
    right: str
    question: str


CANDIDATE_CONTRASTS = (
    ContrastSpec("real_minus_static", "real_drift_axis", "static", "is_real_drift_informative"),
    ContrastSpec("real_minus_random", "real_drift_axis", "random_axis_mean", "is_real_drift_special_vs_random"),
    ContrastSpec("real_minus_edge", "real_drift_axis", "edge", "is_real_drift_special_vs_edge"),
    ContrastSpec("real_minus_edge_orthogonal", "real_drift_axis", "edge_orthogonal", "is_real_drift_special_vs_orthogonal"),
    ContrastSpec("real_minus_spectrum", "real_drift_axis", "spectrum", "is_real_drift_special_vs_spectrum"),
    ContrastSpec("edge_minus_orthogonal", "edge", "edge_orthogonal", "is_edge_parallel_better_than_orthogonal"),
    ContrastSpec("edge_minus_random", "edge", "random_axis_mean", "is_edge_better_than_random"),
    ContrastSpec("edge_orthogonal_minus_random", "edge_orthogonal", "random_axis_mean", "is_orthogonal_better_than_random"),
)


SCALE_GROUP_COLS = ["motion_scale_id", "motion_scale_kind", "motion_scale_value", "motion_scale_label"]


def _scale_group_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in SCALE_GROUP_COLS if col in df.columns]


def _add_clipping_columns(table: pd.DataFrame, clipping: pd.DataFrame) -> pd.DataFrame:
    if table.empty or clipping.empty:
        return table
    merge_cols = [col for col in SCALE_GROUP_COLS if col in table.columns and col in clipping.columns]
    if not merge_cols:
        return table
    keep_cols = [
        *merge_cols,
        "scale_n_windows",
        "fraction_rms_clipped_low",
        "fraction_rms_clipped_high",
        "mean_raw_rms_radius_deg",
        "mean_actual_rms_radius_deg",
        "median_actual_rms_radius_deg",
    ]
    return table.merge(clipping[[col for col in keep_cols if col in clipping.columns]], on=merge_cols, how="left")


def _scale_prefix(row: pd.Series) -> str:
    if "motion_scale_label" not in row.index:
        return ""
    return f" scale `{row['motion_scale_label']}`"


def _nonzero_scale_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "motion_scale_value" not in df.columns:
        return df
    nonzero = df[np.asarray(df["motion_scale_value"], dtype=float) > 0.0].copy()
    return nonzero if not nonzero.empty else df


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _cos2(a_deg: np.ndarray | float, b_deg: np.ndarray | float) -> np.ndarray:
    return np.cos(2.0 * np.radians(np.asarray(a_deg, dtype=np.float64) - np.asarray(b_deg, dtype=np.float64)))


def _demean_within_session(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    return (series - series.groupby(pd.Series(sessions)).transform("mean")).to_numpy(dtype=np.float64)


def _session_bootstrap_mean(
    values: np.ndarray,
    sessions: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float]:
    df = pd.DataFrame({"value": np.asarray(values, dtype=np.float64), "session": np.asarray(sessions)})
    df = df[np.isfinite(df["value"])]
    if df.empty:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_le_0": float("nan"),
            "p_ge_0": float("nan"),
            "fraction_positive_windows": float("nan"),
            "n_windows": 0,
            "n_sessions": 0,
        }
    session_mean = df.groupby("session")["value"].mean().to_numpy(dtype=np.float64)
    mean = float(np.mean(session_mean))
    out = {
        "mean": mean,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_le_0": float("nan"),
        "p_ge_0": float("nan"),
        "fraction_positive_windows": float(np.mean(df["value"].to_numpy(dtype=np.float64) > 0.0)),
        "n_windows": int(df.shape[0]),
        "n_sessions": int(session_mean.size),
    }
    if session_mean.size > 1 and int(n_bootstrap) > 0:
        draws = session_mean[rng.integers(0, session_mean.size, size=(int(n_bootstrap), session_mean.size))]
        boot = np.mean(draws, axis=1)
        out.update(
            {
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "p_le_0": float((1.0 + np.count_nonzero(boot <= 0.0)) / (boot.size + 1.0)),
                "p_ge_0": float((1.0 + np.count_nonzero(boot >= 0.0)) / (boot.size + 1.0)),
            }
        )
    return out


def _ols_increment(y: np.ndarray, predictor: np.ndarray, controls: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    predictor = np.asarray(predictor, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    ok = np.isfinite(y) & np.isfinite(predictor) & np.all(np.isfinite(controls), axis=1)
    y = y[ok]
    predictor = predictor[ok]
    controls = controls[ok]
    if y.size <= controls.shape[1] + 3:
        return {"coef": float("nan"), "incremental_r2": float("nan"), "full_r2": float("nan"), "control_r2": float("nan"), "n": int(y.size)}

    def zscore(A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=np.float64).copy()
        if A.ndim == 1:
            return (A - np.mean(A)) / (np.std(A) + 1e-12)
        for j in range(A.shape[1]):
            A[:, j] = (A[:, j] - np.mean(A[:, j])) / (np.std(A[:, j]) + 1e-12)
        return A

    y = zscore(y)
    X0 = np.column_stack([np.ones(y.size), zscore(controls)])
    X1 = np.column_stack([X0, zscore(predictor)])
    beta0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    beta1, *_ = np.linalg.lstsq(X1, y, rcond=None)

    def r2(X: np.ndarray, beta: np.ndarray) -> float:
        pred = X @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    r2_control = r2(X0, beta0)
    r2_full = r2(X1, beta1)
    return {
        "coef": float(beta1[-1]),
        "incremental_r2": float(r2_full - r2_control),
        "full_r2": float(r2_full),
        "control_r2": float(r2_control),
        "n": int(y.size),
    }


def _candidate_score_pivot(block: pd.DataFrame) -> pd.DataFrame:
    pivot = block.pivot_table(
        index=["window_row", "window_id", "session"],
        columns="candidate",
        values="decode_score_neg_mse",
        aggfunc="mean",
    )
    random_cols = [col for col in pivot.columns if str(col).startswith("random_axis_")]
    if random_cols:
        pivot["random_axis_mean"] = pivot[random_cols].mean(axis=1)
    return pivot


def _scale_clipping_summary(motion: pd.DataFrame) -> pd.DataFrame:
    if motion.empty:
        return pd.DataFrame()
    scale_cols = _scale_group_cols(motion)
    if not scale_cols or "window_row" not in motion.columns:
        return pd.DataFrame()
    source = motion[motion["candidate"].astype(str) != "static"].copy() if "candidate" in motion.columns else motion.copy()
    if source.empty:
        source = motion.copy()
    dedup = source.drop_duplicates(["window_row", *scale_cols]).copy()
    rows: list[dict[str, Any]] = []
    for key, block in dedup.groupby(scale_cols, sort=True):
        raw = block["raw_rms_radius_deg"].to_numpy(dtype=np.float64) if "raw_rms_radius_deg" in block.columns else np.full(block.shape[0], np.nan)
        actual = block["rms_radius_deg"].to_numpy(dtype=np.float64) if "rms_radius_deg" in block.columns else np.full(block.shape[0], np.nan)
        low = block["rms_clipped_low"].astype(bool).to_numpy() if "rms_clipped_low" in block.columns else np.zeros(block.shape[0], dtype=bool)
        high = block["rms_clipped_high"].astype(bool).to_numpy() if "rms_clipped_high" in block.columns else np.zeros(block.shape[0], dtype=bool)
        rows.append(
            {
                **dict(zip(scale_cols, key if isinstance(key, tuple) else (key,), strict=True)),
                "scale_n_windows": int(block.shape[0]),
                "fraction_rms_clipped_low": float(np.mean(low)) if low.size else float("nan"),
                "fraction_rms_clipped_high": float(np.mean(high)) if high.size else float("nan"),
                "mean_raw_rms_radius_deg": float(np.nanmean(raw)),
                "mean_actual_rms_radius_deg": float(np.nanmean(actual)),
                "median_actual_rms_radius_deg": float(np.nanmedian(actual)),
                "max_actual_rms_radius_deg": float(np.nanmax(actual)),
            }
        )
    return pd.DataFrame(rows)


def _paired_candidate_contrasts(per_window: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [*_scale_group_cols(per_window), "latent_name", "latent_family", "latent_scope", "observer", "pca_k"]
    for key, block in per_window.groupby(group_cols, sort=True):
        pivot = _candidate_score_pivot(block)
        for spec in CANDIDATE_CONTRASTS:
            if spec.left not in pivot.columns or spec.right not in pivot.columns:
                continue
            diff = pivot[spec.left].to_numpy(dtype=np.float64) - pivot[spec.right].to_numpy(dtype=np.float64)
            stats = _session_bootstrap_mean(
                diff,
                pivot.index.get_level_values("session").to_numpy(),
                rng=rng,
                n_bootstrap=n_bootstrap,
            )
            rows.append(
                {
                    **dict(zip(group_cols, key, strict=True)),
                    "contrast": spec.name,
                    "left_candidate": spec.left,
                    "right_candidate": spec.right,
                    "question": spec.question,
                    "mean_score_delta": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "bootstrap_p_delta_le_0": stats["p_le_0"],
                    "bootstrap_p_delta_ge_0": stats["p_ge_0"],
                    "fraction_positive_windows": stats["fraction_positive_windows"],
                    "n_windows": stats["n_windows"],
                    "n_sessions": stats["n_sessions"],
                }
            )
    return pd.DataFrame(rows)


def _pose_contrasts(per_window: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [*_scale_group_cols(per_window), "latent_name", "latent_family", "latent_scope", "candidate", "pca_k"]
    for key, block in per_window.groupby(group_cols, sort=True):
        pivot = block.pivot_table(
            index=["window_row", "window_id", "session"],
            columns="observer",
            values="decode_score_neg_mse",
            aggfunc="mean",
        )
        if "pose_aware_flat" not in pivot.columns or "pose_blind_mean" not in pivot.columns:
            continue
        diff = pivot["pose_aware_flat"].to_numpy(dtype=np.float64) - pivot["pose_blind_mean"].to_numpy(dtype=np.float64)
        stats = _session_bootstrap_mean(
            diff,
            pivot.index.get_level_values("session").to_numpy(),
            rng=rng,
            n_bootstrap=n_bootstrap,
        )
        rows.append(
            {
                **dict(zip(group_cols, key, strict=True)),
                "contrast": "pose_aware_minus_pose_blind",
                "mean_score_delta": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "bootstrap_p_delta_le_0": stats["p_le_0"],
                "bootstrap_p_delta_ge_0": stats["p_ge_0"],
                "fraction_positive_windows": stats["fraction_positive_windows"],
                "n_windows": stats["n_windows"],
                "n_sessions": stats["n_sessions"],
            }
        )
    return pd.DataFrame(rows)


def _latent_carrier_summary(decode: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    non_control = decode[(decode["candidate"] != "static") & (~decode["candidate"].str.startswith("random_axis_"))].copy()
    scale_cols = _scale_group_cols(decode)
    for group_cols in (
        [*scale_cols, "latent_name", "observer", "pca_k"],
        [*scale_cols, "latent_family", "latent_scope", "observer", "pca_k"],
    ):
        for key, block in non_control.groupby(group_cols, sort=True):
            best_static = block.sort_values("Delta_score_vs_static", ascending=False).iloc[0]
            best_random = block.sort_values("Delta_score_vs_random_axis", ascending=False).iloc[0]
            row = dict(zip(group_cols, key, strict=True))
            if "latent_family" not in row:
                row["latent_family"] = str(best_static["latent_family"])
            if "latent_scope" not in row:
                row["latent_scope"] = str(best_static["latent_scope"])
            row.update(
                {
                    "summary_level": "+".join(group_cols),
                    "best_candidate_vs_static": str(best_static["candidate"]),
                    "best_delta_score_vs_static": float(best_static["Delta_score_vs_static"]),
                    "best_R2_z_vs_static": float(best_static["R2_z"]),
                    "best_candidate_vs_random": str(best_random["candidate"]),
                    "best_delta_score_vs_random": float(best_random["Delta_score_vs_random_axis"]),
                    "best_R2_z_vs_random": float(best_random["R2_z"]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _scale_curve_summary(per_window: pd.DataFrame, *, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [*_scale_group_cols(per_window), "latent_name", "latent_family", "latent_scope", "observer", "pca_k"]
    for key, block in per_window.groupby(group_cols, sort=True):
        pivot = _candidate_score_pivot(block)
        if "static" not in pivot.columns:
            continue
        sessions = pivot.index.get_level_values("session").to_numpy()
        static = pivot["static"].to_numpy(dtype=np.float64)
        candidate_values: dict[str, np.ndarray] = {}
        for candidate in ("real_drift_axis", "edge", "edge_orthogonal", "spectrum", "random_axis_mean"):
            if candidate in pivot.columns:
                candidate_values[candidate] = pivot[candidate].to_numpy(dtype=np.float64)
        grid_cols = [col for col in pivot.columns if str(col).startswith("grid_")]
        if grid_cols:
            candidate_values["grid_best"] = pivot.loc[:, grid_cols].max(axis=1).to_numpy(dtype=np.float64)
        for candidate, values in candidate_values.items():
            delta = values - static
            stats = _session_bootstrap_mean(delta, sessions, rng=rng, n_bootstrap=n_bootstrap)
            rows.append(
                {
                    **dict(zip(group_cols, key, strict=True)),
                    "candidate": candidate,
                    "contrast": f"{candidate}_minus_static",
                    "mean_score_delta_vs_static": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "bootstrap_p_delta_le_0": stats["p_le_0"],
                    "bootstrap_p_delta_ge_0": stats["p_ge_0"],
                    "fraction_positive_windows": stats["fraction_positive_windows"],
                    "n_windows": stats["n_windows"],
                    "n_sessions": stats["n_sessions"],
                }
            )
    return pd.DataFrame(rows)


def _info_argmax_axis_summary(
    per_window: pd.DataFrame,
    windows: pd.DataFrame,
    motion: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    axis_meta = motion.pivot_table(index="window_row", columns="candidate", values="axis_deg", aggfunc="first")
    win = windows.set_index("window_row")
    rows: list[dict[str, Any]] = []
    group_cols = [*_scale_group_cols(per_window), "latent_name", "latent_family", "latent_scope", "observer", "pca_k"]
    for key, block in per_window.groupby(group_cols, sort=True):
        pivot = _candidate_score_pivot(block)
        grid_candidates = sorted(str(c) for c in pivot.columns if str(c).startswith("grid_"))
        axis_candidates = tuple(grid_candidates) if grid_candidates else ("edge", "edge_orthogonal", "spectrum")
        if not set(axis_candidates).issubset(pivot.columns):
            continue
        score_block = pivot.loc[:, list(axis_candidates)]
        best_candidate = score_block.idxmax(axis=1)
        best_score = score_block.max(axis=1).to_numpy(dtype=np.float64)
        edge_score = pivot["edge"].to_numpy(dtype=np.float64)
        real_score = pivot["real_drift_axis"].to_numpy(dtype=np.float64) if "real_drift_axis" in pivot.columns else np.full(edge_score.shape, np.nan)
        window_rows = pivot.index.get_level_values("window_row").to_numpy(dtype=int)
        sessions = pivot.index.get_level_values("session").to_numpy()
        real_axis = win.loc[window_rows, "real_drift_axis_deg"].to_numpy(dtype=np.float64)
        edge_axis = win.loc[window_rows, "edge_axis_deg"].to_numpy(dtype=np.float64)
        best_axis = np.asarray(
            [axis_meta.loc[int(w), str(c)] if int(w) in axis_meta.index and str(c) in axis_meta.columns else np.nan for w, c in zip(window_rows, best_candidate, strict=True)],
            dtype=np.float64,
        )
        best_minus_edge = best_score - edge_score
        best_minus_real = best_score - real_score
        best_cos2_real = _cos2(best_axis, real_axis)
        edge_cos2_real = _cos2(edge_axis, real_axis)
        cos2_gain_vs_edge = best_cos2_real - edge_cos2_real
        score_stats = _session_bootstrap_mean(best_minus_edge, sessions, rng=rng, n_bootstrap=n_bootstrap)
        real_stats = _session_bootstrap_mean(best_minus_real, sessions, rng=rng, n_bootstrap=n_bootstrap)
        cos_stats = _session_bootstrap_mean(cos2_gain_vs_edge, sessions, rng=rng, n_bootstrap=n_bootstrap)
        row = {
            **dict(zip(group_cols, key, strict=True)),
            "argmax_candidate_set": ",".join(axis_candidates),
            "argmax_candidate_family": "fixed_grid" if grid_candidates else "image_axes",
            "mean_best_score_minus_edge": score_stats["mean"],
            "ci_low_best_score_minus_edge": score_stats["ci_low"],
            "ci_high_best_score_minus_edge": score_stats["ci_high"],
            "p_best_score_minus_edge_le_0": score_stats["p_le_0"],
            "mean_best_score_minus_real": real_stats["mean"],
            "ci_low_best_score_minus_real": real_stats["ci_low"],
            "ci_high_best_score_minus_real": real_stats["ci_high"],
            "p_best_score_minus_real_le_0": real_stats["p_le_0"],
            "mean_best_cos2_real": float(np.nanmean(best_cos2_real)),
            "mean_edge_cos2_real": float(np.nanmean(edge_cos2_real)),
            "mean_best_minus_edge_cos2_real": cos_stats["mean"],
            "ci_low_best_minus_edge_cos2_real": cos_stats["ci_low"],
            "ci_high_best_minus_edge_cos2_real": cos_stats["ci_high"],
            "p_best_minus_edge_cos2_real_le_0": cos_stats["p_le_0"],
            "n_windows": int(pivot.shape[0]),
            "n_sessions": int(pd.Series(sessions).nunique()),
        }
        counts = best_candidate.value_counts(normalize=True)
        for rank, (cand, frac) in enumerate(counts.sort_values(ascending=False).head(5).items(), start=1):
            row[f"argmax_rank{rank}_candidate"] = str(cand)
            row[f"argmax_rank{rank}_fraction"] = float(frac)
        for cand in ("edge", "edge_orthogonal", "spectrum"):
            row[f"fraction_argmax_{cand}"] = float(counts.get(cand, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _residual_prediction_summary(per_window: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    win = windows.set_index("window_row")
    group_cols = [*_scale_group_cols(per_window), "latent_name", "latent_family", "latent_scope", "observer", "pca_k"]
    for key, block in per_window.groupby(group_cols, sort=True):
        pivot = _candidate_score_pivot(block)
        required = {"edge", "edge_orthogonal", "real_drift_axis", "random_axis_mean"}
        if not required.issubset(pivot.columns):
            continue
        window_rows = pivot.index.get_level_values("window_row").to_numpy(dtype=int)
        sessions = pivot.index.get_level_values("session").to_numpy()
        y = win.loc[window_rows, "drift_edge_cos2"].to_numpy(dtype=np.float64)
        controls = np.column_stack(
            [
                win.loc[window_rows, "image_orientation_coherence"].to_numpy(dtype=np.float64),
                win.loc[window_rows, "drift_anisotropy"].to_numpy(dtype=np.float64),
            ]
        )
        predictors = {
            "edge_minus_orthogonal_score": pivot["edge"].to_numpy(dtype=np.float64) - pivot["edge_orthogonal"].to_numpy(dtype=np.float64),
            "real_minus_random_score": pivot["real_drift_axis"].to_numpy(dtype=np.float64) - pivot["random_axis_mean"].to_numpy(dtype=np.float64),
            "real_minus_edge_score": pivot["real_drift_axis"].to_numpy(dtype=np.float64) - pivot["edge"].to_numpy(dtype=np.float64),
        }
        for pred_name, pred in predictors.items():
            result = _ols_increment(
                _demean_within_session(y, sessions),
                _demean_within_session(pred, sessions),
                np.column_stack([_demean_within_session(controls[:, j], sessions) for j in range(controls.shape[1])]),
            )
            rows.append(
                {
                    **dict(zip(group_cols, key, strict=True)),
                    "target": "within_session_drift_edge_cos2",
                    "predictor": pred_name,
                    "controls": "within_session_image_orientation_coherence+drift_anisotropy",
                    "coef": result["coef"],
                    "incremental_r2": result["incremental_r2"],
                    "full_r2": result["full_r2"],
                    "control_r2": result["control_r2"],
                    "n_windows": result["n"],
                }
            )
    return pd.DataFrame(rows)


def _write_scale_figures(out_dir: Path, contrasts: pd.DataFrame, scale_curve: pd.DataFrame, argmax_axis: pd.DataFrame) -> None:
    if "motion_scale_value" not in contrasts.columns:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def plot_lines(table: pd.DataFrame, *, value_col: str, title: str, filename: str, ylabel: str) -> None:
        if table.empty:
            return
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
        for key, block in table.groupby(["latent_name", "observer", "pca_k"], sort=True):
            block = block.sort_values("motion_scale_value")
            latent_name, observer, k = key
            label = f"{latent_name} {observer} k={int(k)}"
            ax.plot(
                block["motion_scale_value"].to_numpy(dtype=float),
                block[value_col].to_numpy(dtype=float),
                marker="o",
                linewidth=1.4,
                label=label,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Relative RMS scale")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=150)
        plt.close(fig)

    local_k = lambda df: df[(df["latent_scope"] == "local_field") & (df["pca_k"].isin([4, 8]))].copy()
    rel = contrasts[contrasts.get("motion_scale_kind", "") == "relative_observed_rms"].copy()
    plot_lines(
        local_k(rel[rel["contrast"] == "real_minus_random"]),
        value_col="mean_score_delta",
        title="Real minus random versus scale",
        filename="scale_curve_real_minus_random.png",
        ylabel="Mean paired score delta",
    )
    plot_lines(
        local_k(rel[rel["contrast"] == "edge_minus_orthogonal"]),
        value_col="mean_score_delta",
        title="Edge minus orthogonal versus scale",
        filename="scale_curve_edge_minus_orthogonal.png",
        ylabel="Mean paired score delta",
    )
    curve_rel = scale_curve[scale_curve.get("motion_scale_kind", "") == "relative_observed_rms"].copy()
    curve_rel = curve_rel[curve_rel["candidate"].isin(["real_drift_axis", "edge", "edge_orthogonal", "random_axis_mean", "grid_best"])]
    for candidate, block in curve_rel.groupby("candidate", sort=True):
        plot_lines(
            local_k(block),
            value_col="mean_score_delta_vs_static",
            title=f"{candidate} minus static versus scale",
            filename=f"scale_curve_{candidate}_minus_static.png",
            ylabel="Mean score delta vs static",
        )
    if not argmax_axis.empty and "motion_scale_value" in argmax_axis.columns:
        arg_rel = argmax_axis[argmax_axis.get("motion_scale_kind", "") == "relative_observed_rms"].copy()
        plot_lines(
            local_k(arg_rel),
            value_col="mean_best_minus_edge_cos2_real",
            title="Grid-best alignment gain versus scale",
            filename="scale_curve_grid_best_alignment_gain.png",
            ylabel="cos2(best, real) - cos2(edge, real)",
        )


def _write_figures(out_dir: Path, contrasts: pd.DataFrame, pose: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    key = contrasts[
        (contrasts["contrast"].isin(["real_minus_random", "edge_minus_orthogonal"]))
        & (contrasts["pca_k"].isin([4, 8, 16]))
    ].copy()
    for contrast, block in key.groupby("contrast"):
        labels = []
        vals = []
        colors = []
        for _, row in block.sort_values(["latent_scope", "latent_name", "observer", "pca_k"]).iterrows():
            labels.append(f"{row['latent_name']}\n{row['observer']}\nk={int(row['pca_k'])}")
            vals.append(float(row["mean_score_delta"]))
            colors.append("#4c78a8" if str(row["latent_scope"]) == "local_field" else "#f58518")
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(max(8.0, 0.28 * len(vals)), 4.0), dpi=150)
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.set_ylabel("Mean paired score delta")
        ax.set_title(contrast, loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{contrast}_paired_score_delta.png", dpi=150)
        plt.close(fig)

    pose_key = pd.DataFrame()
    if not pose.empty and "candidate" in pose.columns:
        pose_key = pose[(pose["candidate"].isin(["edge", "edge_orthogonal", "real_drift_axis", "random_axis_0"]))].copy()
    if not pose_key.empty:
        top = pose_key.sort_values("mean_score_delta", ascending=False).head(40)
        labels = [f"{r.latent_name}\n{r.candidate}\nk={int(r.pca_k)}" for r in top.itertuples()]
        fig, ax = plt.subplots(figsize=(max(8.0, 0.35 * len(labels)), 4.0), dpi=150)
        ax.bar(np.arange(len(labels)), top["mean_score_delta"].to_numpy(dtype=float), color="#54a24b")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.set_ylabel("Pose-aware minus pose-blind score")
        ax.set_title("Pose-aware advantage", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "pose_aware_minus_pose_blind_top40.png", dpi=150)
        plt.close(fig)


def _fmt_ci(row: pd.Series) -> str:
    return f"{row['mean_score_delta']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"


def _write_markdown(
    out_dir: Path,
    *,
    meta: dict[str, Any],
    contrasts: pd.DataFrame,
    clipping: pd.DataFrame,
    pose: pd.DataFrame,
    carrier: pd.DataFrame,
    scale_curve: pd.DataFrame,
    argmax_axis: pd.DataFrame,
    residual: pd.DataFrame,
    alignment: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# BackImage Latent-Information Posthoc Summary",
        "",
        f"Input run: `{out_dir}`",
        "",
        "## Run",
        "",
        f"- Windows: `{meta.get('n_windows', 'unknown')}`",
        f"- Population: `{meta.get('population_source', 'unknown')}`; units `{meta.get('actual_response_units', 'unknown')}`",
        f"- Model family: `{meta.get('model_family', 'unknown')}`",
        f"- Motion scales: `{', '.join(meta.get('motion_scale_ids', [])) if meta.get('motion_scale_ids') else 'single/default'}`",
        "",
        "## Main Read",
        "",
    ]

    analysis_contrasts = _nonzero_scale_rows(contrasts)
    analysis_pose = _nonzero_scale_rows(pose)
    analysis_carrier = _nonzero_scale_rows(carrier)
    analysis_argmax = _nonzero_scale_rows(argmax_axis)
    analysis_residual = _nonzero_scale_rows(residual)

    real = analysis_contrasts[analysis_contrasts["contrast"] == "real_minus_random"].sort_values("mean_score_delta", ascending=False)
    real_edge = analysis_contrasts[analysis_contrasts["contrast"] == "real_minus_edge"].sort_values("mean_score_delta", ascending=False)
    edge_random = analysis_contrasts[analysis_contrasts["contrast"] == "edge_minus_random"].sort_values("mean_score_delta", ascending=False)
    real_orth = analysis_contrasts[analysis_contrasts["contrast"] == "real_minus_edge_orthogonal"].sort_values("mean_score_delta", ascending=False)
    edge = analysis_contrasts[analysis_contrasts["contrast"] == "edge_minus_orthogonal"].sort_values("mean_score_delta", ascending=False)
    pose_top = (
        analysis_pose.sort_values("mean_score_delta", ascending=False)
        if not analysis_pose.empty and "mean_score_delta" in analysis_pose.columns
        else pd.DataFrame()
    )
    carrier_top = analysis_carrier[analysis_carrier["summary_level"].str.endswith("latent_name+observer+pca_k")].sort_values(
        "best_delta_score_vs_static", ascending=False
    )
    argmax_top = analysis_argmax.sort_values("mean_best_score_minus_edge", ascending=False) if not analysis_argmax.empty else analysis_argmax
    residual_top = analysis_residual.sort_values("incremental_r2", ascending=False) if not analysis_residual.empty else analysis_residual

    if not real.empty:
        row = real.iloc[0]
        lines.append(
            f"- Best real-drift-vs-random paired gain{_scale_prefix(row)}: `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}` "
            f"{_fmt_ci(row)}, p(delta<=0)=`{row['bootstrap_p_delta_le_0']:.4f}`."
        )
    if not edge.empty:
        row = edge.iloc[0]
        lines.append(
            f"- Best edge-vs-orthogonal paired gain{_scale_prefix(row)}: `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}` "
            f"{_fmt_ci(row)}, p(delta<=0)=`{row['bootstrap_p_delta_le_0']:.4f}`."
        )
        neg = edge.sort_values("mean_score_delta", ascending=True).iloc[0]
        lines.append(
            f"- Strongest orthogonal-over-edge case{_scale_prefix(neg)}: `{neg['latent_name']}` `{neg['observer']}` k=`{int(neg['pca_k'])}` "
            f"{_fmt_ci(neg)} for edge-minus-orthogonal."
        )
    if not carrier_top.empty:
        row = carrier_top.iloc[0]
        lines.append(
            f"- Strongest absolute carrier vs static{_scale_prefix(row)}: `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}`, "
            f"candidate `{row['best_candidate_vs_static']}`, delta score `{row['best_delta_score_vs_static']:+.3f}`."
        )
    if not pose_top.empty:
        row = pose_top.iloc[0]
        lines.append(
            f"- Strongest pose-aware advantage{_scale_prefix(row)}: `{row['latent_name']}` `{row['candidate']}` k=`{int(row['pca_k'])}` "
            f"{_fmt_ci(row)}, p(delta<=0)=`{row['bootstrap_p_delta_le_0']:.4f}`."
        )
    if not real_edge.empty:
        row = real_edge.iloc[0]
        lines.append(
            f"- Best real-vs-raw-edge gate{_scale_prefix(row)}: `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}` "
            f"{_fmt_ci(row)}, p(delta<=0)=`{row['bootstrap_p_delta_le_0']:.4f}`."
        )

    if not clipping.empty:
        lines.extend(["", "## Scale Clipping", ""])
        lines.append("Rows are de-duplicated per window and non-static motion scale before clipping fractions are computed.")
        for _, row in clipping.sort_values("motion_scale_value" if "motion_scale_value" in clipping.columns else "motion_scale_label").iterrows():
            label = row["motion_scale_label"] if "motion_scale_label" in row.index else row.get("motion_scale_id", "unknown")
            lines.append(
                f"- `{label}`: high-clipped `{row['fraction_rms_clipped_high']:.3f}`, "
                f"low-clipped `{row['fraction_rms_clipped_low']:.3f}`, "
                f"mean actual RMS `{row['mean_actual_rms_radius_deg']:.4f}` deg "
                f"(raw `{row['mean_raw_rms_radius_deg']:.4f}` deg), n=`{int(row['scale_n_windows'])}`."
            )

    if not scale_curve.empty and "motion_scale_value" in scale_curve.columns:
        lines.extend(["", "## Scale Geometry", ""])
        lines.append("Scale curves use paired per-window score deltas; `grid_best` is the best fixed-grid axis per window at that scale.")
        rel_curve = scale_curve[scale_curve.get("motion_scale_kind", "") == "relative_observed_rms"].copy()
        rel_curve_nonzero = _nonzero_scale_rows(rel_curve)
        for candidate in ("real_drift_axis", "edge", "edge_orthogonal", "random_axis_mean", "grid_best"):
            block = rel_curve_nonzero[
                (rel_curve_nonzero["candidate"] == candidate)
                & (rel_curve_nonzero["latent_scope"] == "local_field")
                & (rel_curve_nonzero["pca_k"].isin([4, 8]))
            ].copy()
            if block.empty:
                continue
            row = block.sort_values("mean_score_delta_vs_static", ascending=False).iloc[0]
            lines.append(
                f"- Best `{candidate}` minus static: `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}` "
                f"scale `{row['motion_scale_label']}` {row['mean_score_delta_vs_static']:+.3f} "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}], p<=0 `{row['bootstrap_p_delta_le_0']:.4f}`."
            )
        if not argmax_axis.empty and "motion_scale_value" in argmax_axis.columns:
            arg_rel = argmax_axis[argmax_axis.get("motion_scale_kind", "") == "relative_observed_rms"].copy()
            arg_rel = _nonzero_scale_rows(arg_rel)
            arg_rel = arg_rel[(arg_rel["latent_scope"] == "local_field") & (arg_rel["pca_k"].isin([4, 8]))]
            if not arg_rel.empty:
                row = arg_rel.sort_values("mean_best_minus_edge_cos2_real", ascending=False).iloc[0]
                lines.append(
                    f"- Best grid-alignment gain: `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}` "
                    f"scale `{row['motion_scale_label']}` cos2 gain `{row['mean_best_minus_edge_cos2_real']:+.3f}` "
                    f"[{row['ci_low_best_minus_edge_cos2_real']:+.3f}, {row['ci_high_best_minus_edge_cos2_real']:+.3f}]."
                )

    lines.extend(["", "## Real Drift Specificity", ""])
    for _, row in real.head(8).iterrows():
        lines.append(
            f"- `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}`{_scale_prefix(row)}: real-random {_fmt_ci(row)}, "
            f"p<=0 `{row['bootstrap_p_delta_le_0']:.4f}`."
        )

    lines.extend(["", "## Raw Edge Gates", ""])
    for label, table in (
        ("real-edge", real_edge),
        ("edge-random", edge_random),
        ("real-orthogonal", real_orth),
    ):
        if table.empty:
            continue
        lines.append(f"Top `{label}` paired score deltas:")
        for _, row in table.head(6).iterrows():
            lines.append(
                f"- `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}`{_scale_prefix(row)}: {label} {_fmt_ci(row)}, "
                f"p<=0 `{row['bootstrap_p_delta_le_0']:.4f}`."
            )

    lines.extend(["", "## Edge Interpretation", ""])
    for _, row in edge.head(8).iterrows():
        lines.append(
            f"- `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}`{_scale_prefix(row)}: edge-orth {_fmt_ci(row)}, "
            f"p<=0 `{row['bootstrap_p_delta_le_0']:.4f}`."
        )
    if not alignment.empty:
        lines.extend(["", "Alignment regression rows already in run output, top positive coefficients:", ""])
        for _, row in alignment.sort_values("within_session_coef", ascending=False).head(6).iterrows():
            lines.append(
                f"- `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}`{_scale_prefix(row)}: coef "
                f"`{row['within_session_coef']:+.4f}`, dR2 `{row['within_session_incremental_r2']:.4f}`, "
                f"pcoef `{row['within_session_shuffle_p_coef_ge']:.4f}`."
            )

    lines.extend(["", "## Information-Argmax Axis", ""])
    if not argmax_top.empty:
        family = str(argmax_top.iloc[0].get("argmax_candidate_family", "image_axes"))
        if family == "fixed_grid":
            lines.append("Argmax is over fixed-grid `grid_*` axes; edge and real drift are used only as comparison references.")
        else:
            lines.append("Argmax is over the limited image-axis candidate set `edge, edge_orthogonal, spectrum`; this run did not include a fixed-grid axis sweep.")
        for _, row in argmax_top.head(8).iterrows():
            top_bits = []
            for rank in range(1, 4):
                cand = row.get(f"argmax_rank{rank}_candidate", "")
                frac = row.get(f"argmax_rank{rank}_fraction", np.nan)
                if isinstance(cand, str) and cand and np.isfinite(float(frac)):
                    top_bits.append(f"{cand}:{float(frac):.2f}")
            lines.append(
                f"- `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}`{_scale_prefix(row)}: "
                f"best-edge score `{row['mean_best_score_minus_edge']:+.3f}` "
                f"[`{row['ci_low_best_score_minus_edge']:+.3f}`, `{row['ci_high_best_score_minus_edge']:+.3f}`], "
                f"best-edge cos2(real) `{row['mean_best_minus_edge_cos2_real']:+.3f}`; "
                f"top argmax candidates `{', '.join(top_bits)}`."
            )

    lines.extend(["", "## Residual Prediction", ""])
    lines.append("OLS is within-session demeaned; controls are image orientation coherence and drift anisotropy.")
    if not residual_top.empty:
        for _, row in residual_top.head(8).iterrows():
            lines.append(
                f"- `{row['latent_name']}` `{row['observer']}` k=`{int(row['pca_k'])}`{_scale_prefix(row)} `{row['predictor']}`: "
                f"coef `{row['coef']:+.4f}`, incremental R2 `{row['incremental_r2']:.4f}`."
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `posthoc_candidate_contrast_summary.csv`",
            "- `posthoc_scale_clipping_summary.csv`",
            "- `posthoc_pose_observer_contrast_summary.csv`",
            "- `posthoc_latent_carrier_summary.csv`",
            "- `posthoc_scale_curve_summary.csv`",
            "- `posthoc_info_argmax_axis_summary.csv`",
            "- `posthoc_residual_prediction_summary.csv`",
            "- `figures/real_minus_random_paired_score_delta.png`",
            "- `figures/edge_minus_orthogonal_paired_score_delta.png`",
            "- `figures/pose_aware_minus_pose_blind_top40.png`",
            "",
        ]
    )
    (out_dir / "posthoc_summary.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    rng = np.random.default_rng(int(args.seed))
    with (run_dir / "run_metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    per_window = pd.read_csv(run_dir / "decode_score_by_window_candidate.csv")
    decode = pd.read_csv(run_dir / "decode_summary_by_candidate.csv")
    windows = pd.read_csv(run_dir / "analysis_windows.csv")
    motion = pd.read_csv(run_dir / "candidate_motion_metadata.csv")
    alignment_path = run_dir / "alignment_strength_prediction_summary.csv"
    alignment = _read_csv_or_empty(alignment_path)

    clipping = _scale_clipping_summary(motion)
    contrasts = _paired_candidate_contrasts(per_window, rng=rng, n_bootstrap=int(args.n_bootstrap))
    pose = _pose_contrasts(per_window, rng=rng, n_bootstrap=int(args.n_bootstrap))
    carrier = _latent_carrier_summary(decode)
    scale_curve = _scale_curve_summary(per_window, rng=rng, n_bootstrap=int(args.n_bootstrap))
    argmax_axis = _info_argmax_axis_summary(per_window, windows, motion, rng=rng, n_bootstrap=int(args.n_bootstrap))
    residual = _residual_prediction_summary(per_window, windows)
    contrasts = _add_clipping_columns(contrasts, clipping)
    scale_curve = _add_clipping_columns(scale_curve, clipping)

    contrasts.to_csv(run_dir / "posthoc_candidate_contrast_summary.csv", index=False)
    clipping.to_csv(run_dir / "posthoc_scale_clipping_summary.csv", index=False)
    pose.to_csv(run_dir / "posthoc_pose_observer_contrast_summary.csv", index=False)
    carrier.to_csv(run_dir / "posthoc_latent_carrier_summary.csv", index=False)
    scale_curve.to_csv(run_dir / "posthoc_scale_curve_summary.csv", index=False)
    argmax_axis.to_csv(run_dir / "posthoc_info_argmax_axis_summary.csv", index=False)
    residual.to_csv(run_dir / "posthoc_residual_prediction_summary.csv", index=False)
    _write_figures(run_dir, contrasts, pose)
    _write_scale_figures(run_dir, contrasts, scale_curve, argmax_axis)
    _write_markdown(
        run_dir,
        meta=meta,
        contrasts=contrasts,
        clipping=clipping,
        pose=pose,
        carrier=carrier,
        scale_curve=scale_curve,
        argmax_axis=argmax_axis,
        residual=residual,
        alignment=alignment,
    )
    print(f"Wrote BackImage latent-information posthoc summary to {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    summarize(build_parser().parse_args())


if __name__ == "__main__":
    main()
