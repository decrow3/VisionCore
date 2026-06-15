"""BackImage edge-parallel versus edge-orthogonal stability screen.

This is a cheap follow-up to the fixed-grid twin objective screen.  Instead of
asking a PA/PB/Pareto objective to discover an axis, it asks whether the local
raw-edge axis has the expected functional property: same-amplitude motion along
the edge should perturb pixels, and optionally V1-twin responses, less than
motion orthogonal to the edge.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import shift as scipy_shift
from tqdm import tqdm

try:
    from .image_features import _backimage_canvas, gaze_deg_to_screen_px
    from .run_backimage_twin_drift_geometry import TwinScorer, _clip_patch, _cos2, _standardize_uint_like
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
        TwinScorer,
        _clip_patch,
        _cos2,
        _standardize_uint_like,
    )


DEFAULT_INPUT = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_edge_parallel_stability_screen"
)


@dataclass(frozen=True)
class RunConfig:
    input: str
    out_dir: str
    max_windows: int
    reliable_image_coherence_min: float
    reliable_drift_anisotropy_min: float
    min_duration_s: float
    patch_size_px: int
    min_patch_image_margin_px: float
    sample_size_px: int
    displacement_deg: float
    run_twin: bool
    twin_population_n: int
    twin_batch_size: int
    twin_hold_frames: int
    twin_tail_frames: int
    device: str
    n_shuffle_nulls: int
    n_session_bootstrap: int
    seed: int


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _axis_vector(axis_deg: float, displacement_deg: float) -> np.ndarray:
    theta = np.radians(float(axis_deg))
    return float(displacement_deg) * np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)


def _central_crop(image: np.ndarray, sample_size_px: int) -> np.ndarray:
    half = int(sample_size_px) // 2
    cy = image.shape[0] // 2
    cx = image.shape[1] // 2
    return image[cy - half : cy + half + 1, cx - half : cx + half + 1]


def _shift_patch_for_gaze_displacement(patch: np.ndarray, displacement_xy_deg: np.ndarray, ppd: float) -> np.ndarray:
    dx, dy = float(displacement_xy_deg[0]), float(displacement_xy_deg[1])
    return scipy_shift(
        patch,
        shift=(-dy * float(ppd), -dx * float(ppd)),
        order=1,
        mode="nearest",
        prefilter=False,
    )


def _pixel_axis_cost(
    patch: np.ndarray,
    *,
    axis_deg: float,
    displacement_deg: float,
    ppd: float,
    sample_size_px: int,
) -> float:
    image = _standardize_uint_like(patch)
    base = _central_crop(image, sample_size_px)
    vector = _axis_vector(axis_deg, displacement_deg)
    costs = []
    for sign in (1.0, -1.0):
        shifted = _shift_patch_for_gaze_displacement(image, sign * vector, ppd)
        shifted_crop = _central_crop(shifted, sample_size_px)
        costs.append(float(np.nanmean((shifted_crop - base) ** 2)))
    return float(np.nanmean(costs))


def _relative_advantage(parallel_cost: float, orthogonal_cost: float) -> float:
    denom = abs(float(parallel_cost)) + abs(float(orthogonal_cost)) + 1e-12
    return float((float(orthogonal_cost) - float(parallel_cost)) / denom)


def _tail_mean(response: np.ndarray, *, block_idx: int, hold_frames: int, tail_frames: int) -> np.ndarray:
    end = min((int(block_idx) + 1) * int(hold_frames), response.shape[0])
    start = max(int(block_idx) * int(hold_frames), end - int(tail_frames))
    if end <= start:
        start = max(0, end - 1)
    return np.nanmean(response[start:end], axis=0)


def _twin_endpoint_costs(
    scorer: TwinScorer,
    patch: np.ndarray,
    *,
    edge_axis_deg: float,
    displacement_deg: float,
    hold_frames: int,
    tail_frames: int,
) -> dict[str, float]:
    parallel = _axis_vector(edge_axis_deg, displacement_deg)
    orthogonal = _axis_vector(edge_axis_deg + 90.0, displacement_deg)
    endpoints = [
        np.asarray([0.0, 0.0], dtype=np.float32),
        parallel,
        -parallel,
        orthogonal,
        -orthogonal,
    ]
    trace = np.concatenate(
        [np.repeat(endpoint[None, :], int(hold_frames), axis=0) for endpoint in endpoints],
        axis=0,
    ).astype(np.float32)
    response = scorer.response(patch, trace)
    base = _tail_mean(response, block_idx=0, hold_frames=hold_frames, tail_frames=tail_frames)
    par_plus = _tail_mean(response, block_idx=1, hold_frames=hold_frames, tail_frames=tail_frames)
    par_minus = _tail_mean(response, block_idx=2, hold_frames=hold_frames, tail_frames=tail_frames)
    orth_plus = _tail_mean(response, block_idx=3, hold_frames=hold_frames, tail_frames=tail_frames)
    orth_minus = _tail_mean(response, block_idx=4, hold_frames=hold_frames, tail_frames=tail_frames)
    parallel_cost = float(0.5 * (np.nanmean((par_plus - base) ** 2) + np.nanmean((par_minus - base) ** 2)))
    orthogonal_cost = float(0.5 * (np.nanmean((orth_plus - base) ** 2) + np.nanmean((orth_minus - base) ** 2)))
    return {
        "twin_parallel_cost": parallel_cost,
        "twin_orthogonal_cost": orthogonal_cost,
        "twin_stability_advantage": float(orthogonal_cost - parallel_cost),
        "twin_relative_advantage": _relative_advantage(parallel_cost, orthogonal_cost),
    }


def _session_mean(values: np.ndarray, sessions: np.ndarray) -> float:
    df = pd.DataFrame({"value": values, "session": sessions})
    return float(df.groupby("session")["value"].mean().mean())


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or int(n_bootstrap) <= 0:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(int(n_bootstrap), values.size), replace=True)
    means = np.mean(draws, axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _stability_summary(df: pd.DataFrame, *, prefix: str, rng: np.random.Generator, n_bootstrap: int) -> dict[str, Any]:
    parallel = df[f"{prefix}_parallel_cost"].to_numpy(dtype=np.float64)
    orthogonal = df[f"{prefix}_orthogonal_cost"].to_numpy(dtype=np.float64)
    advantage = orthogonal - parallel
    sessions = df["session"].to_numpy()
    session_adv = pd.DataFrame({"session": sessions, "advantage": advantage}).groupby("session")["advantage"].mean()
    ci_lo, ci_hi = _bootstrap_ci(session_adv.to_numpy(), rng, n_bootstrap)
    return {
        "screen": prefix,
        "n_windows": int(df.shape[0]),
        "n_sessions": int(session_adv.shape[0]),
        "mean_parallel_cost_window": float(np.nanmean(parallel)),
        "mean_orthogonal_cost_window": float(np.nanmean(orthogonal)),
        "mean_advantage_window": float(np.nanmean(advantage)),
        "mean_advantage_session_mean": float(np.nanmean(session_adv)),
        "ci95_low_session_mean": ci_lo,
        "ci95_high_session_mean": ci_hi,
        "n_sessions_positive_advantage": int(np.count_nonzero(session_adv.to_numpy() > 0)),
        "fraction_windows_positive_advantage": float(np.nanmean(advantage > 0)),
    }


def _finite_design(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return y[ok], X[ok]


def _standardize_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    for j in range(X.shape[1]):
        sd = float(np.nanstd(X[:, j]))
        mean = float(np.nanmean(X[:, j]))
        if np.isfinite(sd) and sd > 1e-12:
            X[:, j] = (X[:, j] - mean) / sd
        else:
            X[:, j] = 0.0
    return X


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    y, X = _finite_design(np.asarray(y, dtype=np.float64), np.asarray(X, dtype=np.float64))
    if y.size <= X.shape[1] + 1:
        return np.full(X.shape[1] + 1, np.nan), float("nan")
    y = (y - float(np.mean(y))) / (float(np.std(y)) + 1e-12)
    X = _standardize_columns(X)
    design = np.column_stack([np.ones(y.size), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, float(r2)


def _demean_within_session(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    means = series.groupby(pd.Series(sessions)).transform("mean")
    return (series - means).to_numpy(dtype=np.float64)


def _incremental_prediction_summary(
    df: pd.DataFrame,
    *,
    predictor: str,
    controls: list[str],
    label: str,
    rng: np.random.Generator,
    n_shuffle_nulls: int,
) -> dict[str, Any]:
    y = df["drift_edge_cos2"].to_numpy(dtype=np.float64)
    sessions = df["session"].to_numpy()
    pred = df[predictor].to_numpy(dtype=np.float64)
    control_mat = np.column_stack([df[col].to_numpy(dtype=np.float64) for col in controls]) if controls else np.empty((df.shape[0], 0))

    X_control = control_mat
    _, r2_control = _ols(y, X_control)
    X_full = np.column_stack([control_mat, pred])
    beta_full, r2_full = _ols(y, X_full)

    y_dm = _demean_within_session(y, sessions)
    pred_dm = _demean_within_session(pred, sessions)
    control_dm = (
        np.column_stack([_demean_within_session(df[col].to_numpy(dtype=np.float64), sessions) for col in controls])
        if controls
        else np.empty((df.shape[0], 0))
    )
    _, r2_control_dm = _ols(y_dm, control_dm)
    beta_dm, r2_full_dm = _ols(y_dm, np.column_stack([control_dm, pred_dm]))
    observed_coef = float(beta_dm[-1])
    observed_delta_r2 = float(r2_full_dm - r2_control_dm)

    null_coef = np.empty(int(n_shuffle_nulls), dtype=np.float64)
    null_delta_r2 = np.empty(int(n_shuffle_nulls), dtype=np.float64)
    for j in range(int(n_shuffle_nulls)):
        shuffled = pred.copy()
        for session in np.unique(sessions):
            idx = np.flatnonzero(sessions == session)
            if idx.size > 1:
                shuffled[idx] = rng.permutation(shuffled[idx])
        shuffled_dm = _demean_within_session(shuffled, sessions)
        beta_null, r2_null = _ols(y_dm, np.column_stack([control_dm, shuffled_dm]))
        null_coef[j] = float(beta_null[-1])
        null_delta_r2[j] = float(r2_null - r2_control_dm)

    p_coef_ge = (1.0 + float(np.count_nonzero(null_coef >= observed_coef))) / (float(n_shuffle_nulls) + 1.0)
    p_delta_r2_ge = (1.0 + float(np.count_nonzero(null_delta_r2 >= observed_delta_r2))) / (float(n_shuffle_nulls) + 1.0)
    return {
        "model": label,
        "predictor": predictor,
        "controls": "+".join(controls) if controls else "none",
        "n_windows": int(df.shape[0]),
        "n_sessions": int(df["session"].nunique()),
        "full_standardized_coef": float(beta_full[-1]),
        "full_r2": float(r2_full),
        "control_r2": float(r2_control),
        "incremental_r2": float(r2_full - r2_control),
        "within_session_standardized_coef": observed_coef,
        "within_session_full_r2": float(r2_full_dm),
        "within_session_control_r2": float(r2_control_dm),
        "within_session_incremental_r2": observed_delta_r2,
        "within_session_shuffle_p_coef_ge": p_coef_ge,
        "within_session_shuffle_p_incremental_r2_ge": p_delta_r2_ge,
        "null_coef_mean": float(np.nanmean(null_coef)),
        "null_coef_ci95_low": float(np.nanquantile(null_coef, 0.025)),
        "null_coef_ci95_high": float(np.nanquantile(null_coef, 0.975)),
        "null_delta_r2_mean": float(np.nanmean(null_delta_r2)),
        "null_delta_r2_ci95_low": float(np.nanquantile(null_delta_r2, 0.025)),
        "null_delta_r2_ci95_high": float(np.nanquantile(null_delta_r2, 0.975)),
    }


def _plot_outputs(out_dir: Path, df: pd.DataFrame, stability_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> None:
    if stability_rows:
        sdf = pd.DataFrame(stability_rows)
        fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=150)
        x = np.arange(sdf.shape[0])
        ax.bar(x, sdf["mean_advantage_session_mean"], color=["#4878a8" if s == "pixel" else "#c16622" for s in sdf["screen"]])
        ax.errorbar(
            x,
            sdf["mean_advantage_session_mean"],
            yerr=[
                sdf["mean_advantage_session_mean"] - sdf["ci95_low_session_mean"],
                sdf["ci95_high_session_mean"] - sdf["mean_advantage_session_mean"],
            ],
            fmt="none",
            color="black",
            linewidth=0.9,
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sdf["screen"])
        ax.set_ylabel("orthogonal cost - parallel cost")
        ax.set_title("Edge-parallel stability advantage", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_stability_advantage.png", dpi=150)
        plt.close(fig)

    if not df.empty:
        fig, axes = plt.subplots(1, 2 if "twin_relative_advantage" in df.columns else 1, figsize=(8.0, 3.4), dpi=150)
        axes = np.atleast_1d(axes)
        axes[0].scatter(df["pixel_relative_advantage"], df["drift_edge_cos2"], s=10, alpha=0.45, color="#4878a8")
        axes[0].axhline(0.0, color="black", linewidth=0.7)
        axes[0].axvline(0.0, color="black", linewidth=0.7)
        axes[0].set_xlabel("pixel relative advantage")
        axes[0].set_ylabel("real drift-edge cos2")
        axes[0].set_title("Pixel", loc="left", fontsize=10)
        if "twin_relative_advantage" in df.columns:
            axes[1].scatter(df["twin_relative_advantage"], df["drift_edge_cos2"], s=10, alpha=0.45, color="#c16622")
            axes[1].axhline(0.0, color="black", linewidth=0.7)
            axes[1].axvline(0.0, color="black", linewidth=0.7)
            axes[1].set_xlabel("twin relative advantage")
            axes[1].set_title("Twin", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_alignment_strength_vs_advantage.png", dpi=150)
        plt.close(fig)


def _summary_markdown(
    *,
    out_dir: Path,
    cfg: RunConfig,
    stability_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# BackImage Edge-Parallel Stability Screen",
        "",
        f"Input: `{cfg.input}`",
        "",
        "## Run",
        "",
        f"- Windows: `{cfg.max_windows}`",
        f"- Displacement: `{cfg.displacement_deg}` deg",
        f"- Twin: `{cfg.run_twin}`; population `{cfg.twin_population_n}`; hold frames `{cfg.twin_hold_frames}`",
        f"- Shuffle nulls: `{cfg.n_shuffle_nulls}`; session bootstraps: `{cfg.n_session_bootstrap}`",
        "",
        "## Stability Advantage",
        "",
    ]
    for row in stability_rows:
        lines.append(
            f"- `{row['screen']}`: session mean advantage "
            f"`{row['mean_advantage_session_mean']:.6g}`, CI "
            f"`[{row['ci95_low_session_mean']:.6g}, {row['ci95_high_session_mean']:.6g}]`, "
            f"`{row['n_sessions_positive_advantage']}/{row['n_sessions']}` positive sessions."
        )
    lines.extend(["", "## Alignment-Strength Prediction", ""])
    for row in prediction_rows:
        lines.append(
            f"- `{row['model']}`: within-session coef `{row['within_session_standardized_coef']:.4f}`, "
            f"incremental R2 `{row['within_session_incremental_r2']:.4f}`, "
            f"shuffle p(coef>=obs) `{row['within_session_shuffle_p_coef_ge']:.4f}`, "
            f"shuffle p(dR2>=obs) `{row['within_session_shuffle_p_incremental_r2_ge']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `edge_parallel_stability_by_window.csv`",
            "- `stability_summary.csv`",
            "- `alignment_strength_prediction_summary.csv`",
            "- `fig_stability_advantage.png`",
            "- `fig_alignment_strength_vs_advantage.png`",
            "",
        ]
    )
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--sample-size-px", type=int, default=151)
    parser.add_argument("--displacement-deg", type=float, default=0.125)
    parser.add_argument("--run-twin", action="store_true")
    parser.add_argument("--twin-population-n", type=int, default=256)
    parser.add_argument("--twin-batch-size", type=int, default=24)
    parser.add_argument("--twin-hold-frames", type=int, default=40)
    parser.add_argument("--twin-tail-frames", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-shuffle-nulls", type=int, default=1000)
    parser.add_argument("--n-session-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    min_patch_image_margin_px = (
        float(args.min_patch_image_margin_px)
        if args.min_patch_image_margin_px is not None
        else float(args.patch_size_px) / 2.0
    )
    cfg = RunConfig(
        input=str(args.input),
        out_dir=str(out_dir),
        max_windows=int(args.max_windows),
        reliable_image_coherence_min=float(args.reliable_image_coherence_min),
        reliable_drift_anisotropy_min=float(args.reliable_drift_anisotropy_min),
        min_duration_s=float(args.min_duration_s),
        patch_size_px=int(args.patch_size_px),
        min_patch_image_margin_px=min_patch_image_margin_px,
        sample_size_px=int(args.sample_size_px),
        displacement_deg=float(args.displacement_deg),
        run_twin=bool(args.run_twin),
        twin_population_n=int(args.twin_population_n),
        twin_batch_size=int(args.twin_batch_size),
        twin_hold_frames=int(args.twin_hold_frames),
        twin_tail_frames=int(args.twin_tail_frames),
        device=str(args.device),
        n_shuffle_nulls=int(args.n_shuffle_nulls),
        n_session_bootstrap=int(args.n_session_bootstrap),
        seed=int(args.seed),
    )
    rng = np.random.default_rng(int(args.seed))
    df = pd.read_csv(args.input)
    required = [
        "session",
        "trial_idx",
        "mean_x_deg",
        "mean_y_deg",
        "drift_orientation_deg",
        "anisotropy",
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "image_patch_distance_to_image_border_px",
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "duration_s" not in df.columns:
        df["duration_s"] = df.get("epoch_duration_s", np.nan)
    keep = (
        np.isfinite(df["drift_orientation_deg"].astype(float))
        & np.isfinite(df["image_edge_axis_deg"].astype(float))
        & (df["anisotropy"].astype(float) >= float(args.reliable_drift_anisotropy_min))
        & (df["image_orientation_coherence"].astype(float) >= float(args.reliable_image_coherence_min))
        & (df["duration_s"].astype(float) >= float(args.min_duration_s))
        & (df["image_patch_distance_to_image_border_px"].astype(float) >= min_patch_image_margin_px)
    )
    work = df.loc[keep].copy()
    work["window_id"] = np.arange(work.shape[0], dtype=int)
    if int(args.max_windows) > 0 and work.shape[0] > int(args.max_windows):
        work = work.sample(n=int(args.max_windows), replace=False, random_state=int(args.seed)).sort_values(
            ["session", "trial_idx", "window_id"]
        )
    work = work.reset_index(drop=True)

    scorer = None
    if args.run_twin:
        scorer = TwinScorer(
            device=str(args.device),
            population_n=int(args.twin_population_n),
            batch_size=int(args.twin_batch_size),
            seed=int(args.seed),
        )

    rows: list[dict[str, Any]] = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    iterator = tqdm(list(work.iterrows()), desc="edge-parallel stability")
    for i, (_, row) in enumerate(iterator):
        try:
            canvas_key = (str(row["session"]), int(row["trial_idx"]))
            if canvas_key not in canvas_cache:
                canvas_cache[canvas_key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
            canvas, ppd, screen_shape = canvas_cache[canvas_key]
            center_px = gaze_deg_to_screen_px(
                np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
                ppd=ppd,
                screen_shape=screen_shape,
            )
            patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(args.patch_size_px))
            edge_axis = float(row["image_edge_axis_deg"])
            pixel_parallel = _pixel_axis_cost(
                patch,
                axis_deg=edge_axis,
                displacement_deg=float(args.displacement_deg),
                ppd=float(ppd),
                sample_size_px=int(args.sample_size_px),
            )
            pixel_orthogonal = _pixel_axis_cost(
                patch,
                axis_deg=edge_axis + 90.0,
                displacement_deg=float(args.displacement_deg),
                ppd=float(ppd),
                sample_size_px=int(args.sample_size_px),
            )
            out = {
                "window_row": int(i),
                "window_id": int(row["window_id"]),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "phase": str(row.get("phase", "")),
                "edge_axis_deg": edge_axis,
                "real_drift_axis_deg": float(row["drift_orientation_deg"]),
                "drift_edge_cos2": float(_cos2(float(row["drift_orientation_deg"]), edge_axis)),
                "alignment_weight": float(row["anisotropy"]) * float(row["image_orientation_coherence"]),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
                "drift_anisotropy": float(row["anisotropy"]),
                "pixel_parallel_cost": pixel_parallel,
                "pixel_orthogonal_cost": pixel_orthogonal,
                "pixel_stability_advantage": float(pixel_orthogonal - pixel_parallel),
                "pixel_relative_advantage": _relative_advantage(pixel_parallel, pixel_orthogonal),
            }
            if scorer is not None:
                out.update(
                    _twin_endpoint_costs(
                        scorer,
                        patch,
                        edge_axis_deg=edge_axis,
                        displacement_deg=float(args.displacement_deg),
                        hold_frames=int(args.twin_hold_frames),
                        tail_frames=int(args.twin_tail_frames),
                    )
                )
            rows.append(out)
        except Exception as exc:
            rows.append(
                {
                    "window_row": int(i),
                    "window_id": int(row.get("window_id", i)),
                    "session": str(row.get("session", "")),
                    "trial_idx": int(row.get("trial_idx", -1)),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    result = pd.DataFrame(rows)
    status = result["status"].fillna("ok") if "status" in result.columns else pd.Series("ok", index=result.index)
    ok = result[status != "failed"].copy()
    result.to_csv(out_dir / "edge_parallel_stability_by_window.csv", index=False)

    stability_rows: list[dict[str, Any]] = []
    if not ok.empty:
        stability_rows.append(_stability_summary(ok, prefix="pixel", rng=rng, n_bootstrap=int(args.n_session_bootstrap)))
        if "twin_parallel_cost" in ok.columns:
            stability_rows.append(_stability_summary(ok, prefix="twin", rng=rng, n_bootstrap=int(args.n_session_bootstrap)))
    _write_csv(out_dir / "stability_summary.csv", stability_rows)

    prediction_rows: list[dict[str, Any]] = []
    if not ok.empty:
        prediction_rows.append(
            _incremental_prediction_summary(
                ok,
                predictor="pixel_relative_advantage",
                controls=["image_orientation_coherence"],
                label="pixel_advantage_beyond_edge_coherence",
                rng=rng,
                n_shuffle_nulls=int(args.n_shuffle_nulls),
            )
        )
        if "twin_relative_advantage" in ok.columns:
            prediction_rows.append(
                _incremental_prediction_summary(
                    ok,
                    predictor="twin_relative_advantage",
                    controls=["image_orientation_coherence"],
                    label="twin_advantage_beyond_edge_coherence",
                    rng=rng,
                    n_shuffle_nulls=int(args.n_shuffle_nulls),
                )
            )
            prediction_rows.append(
                _incremental_prediction_summary(
                    ok,
                    predictor="twin_relative_advantage",
                    controls=["image_orientation_coherence", "pixel_relative_advantage"],
                    label="twin_advantage_beyond_edge_coherence_and_pixel_advantage",
                    rng=rng,
                    n_shuffle_nulls=int(args.n_shuffle_nulls),
                )
            )
    _write_csv(out_dir / "alignment_strength_prediction_summary.csv", prediction_rows)

    _plot_outputs(out_dir, ok, stability_rows, prediction_rows)
    _summary_markdown(out_dir=out_dir, cfg=cfg, stability_rows=stability_rows, prediction_rows=prediction_rows)
    _write_json(
        out_dir / "run_metadata.json",
        {
            "config": asdict(cfg),
            "n_input_rows": int(df.shape[0]),
            "n_reliable_rows": int(work.shape[0]),
            "n_success_rows": int(ok.shape[0]),
            "n_failed_rows": int(result.shape[0] - ok.shape[0]),
            "n_canvas_cache_entries": int(len(canvas_cache)),
            "notes": (
                "Positive stability_advantage means same-amplitude motion orthogonal to the local edge "
                "perturbed pixels or twin responses more than motion parallel to the edge. "
                "Alignment-strength models predict drift_edge_cos2 and use within-session shuffles "
                "of the stability advantage for the null."
            ),
        },
    )
    print(f"Wrote BackImage edge-parallel stability screen to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
