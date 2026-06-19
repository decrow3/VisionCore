from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm

from VisionCore.paths import VISIONCORE_ROOT

try:
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import _prepare_windows
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
        CanonicalTwinScorer,
        _align_response_to_trace,
        _static_trace,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import (
        DEFAULT_INPUT,
        _extract_patch,
    )
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import _prepare_windows
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
        CanonicalTwinScorer,
        _align_response_to_trace,
        _static_trace,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import (
        DEFAULT_INPUT,
        _extract_patch,
    )


DEFAULT_OUT_DIR = (
    VISIONCORE_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_cardinal_tangent_charts"
)
BX_COLOR = "#2f5f9f"
BY_COLOR = "#7b5ea7"


def _progress(message: str) -> None:
    print(f"[backimage-cardinal-tangents] {message}", flush=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _constant_trace(dx_deg: float, dy_deg: float, n_timepoints: int) -> np.ndarray:
    trace = np.zeros((int(n_timepoints), 2), dtype=np.float32)
    trace[:, 0] = float(dx_deg)
    trace[:, 1] = float(dy_deg)
    return trace


def _reduce_response(response: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(response, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected aligned response as time x units, got {x.shape}")
    if mode == "mean_time":
        return np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    if mode == "flatten_time":
        return x.reshape(-1).astype(np.float32, copy=False)
    if mode == "last_time":
        return x[-1].astype(np.float32, copy=False)
    raise ValueError(f"Unknown response_reduction={mode!r}")


def _fit_pca(points: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"Need a 2D point matrix with at least two rows, got {x.shape}")
    mean = np.mean(x, axis=0)
    xc = x - mean[None, :]
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    evals = (s * s) / max(x.shape[0] - 1, 1)
    k = min(int(n_components), vt.shape[0])
    return mean, vt[:k].T, evals


def _project_vectors(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float64) @ np.asarray(basis, dtype=np.float64)


def _project_points(points: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - mean[None, :]) @ np.asarray(basis, dtype=np.float64)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


def _sampled_pairwise_cosines(vectors: np.ndarray, *, max_pairs: int = 50_000, seed: int = 11) -> np.ndarray:
    v = _unit_rows(vectors)
    keep = np.isfinite(v).all(axis=1) & (np.linalg.norm(v, axis=1) > 1e-12)
    v = v[keep]
    n = int(v.shape[0])
    if n < 2:
        return np.empty(0, dtype=np.float64)
    total = n * (n - 1) // 2
    if total <= int(max_pairs):
        vals = []
        for i in range(n):
            vals.extend((v[i] @ v[i + 1 :].T).tolist())
        return np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    i = rng.integers(0, n, size=int(max_pairs))
    j = rng.integers(0, n - 1, size=int(max_pairs))
    j = j + (j >= i)
    return np.sum(v[i] * v[j], axis=1)


def _farthest_point_subset(points: np.ndarray, n_show: int, seed: int = 2) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    finite = np.all(np.isfinite(pts), axis=1)
    valid_idx = np.flatnonzero(finite)
    pts = pts[finite]
    if pts.shape[0] <= int(n_show):
        return valid_idx
    rng = np.random.default_rng(int(seed))
    start_scores = np.linalg.norm(pts - np.median(pts, axis=0, keepdims=True), axis=1)
    start_scores = start_scores + 1e-9 * rng.normal(size=pts.shape[0])
    chosen = [int(np.argmax(start_scores))]
    min_dist = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(1, int(n_show)):
        score = min_dist + 1e-9 * rng.normal(size=pts.shape[0])
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(pts - pts[nxt], axis=1))
    return valid_idx[np.asarray(chosen, dtype=np.int64)]


def _axis_limits(points: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float64)
    lo = np.nanpercentile(pts, 1, axis=0)
    hi = np.nanpercentile(pts, 99, axis=0)
    center = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    return center, span


def _plot_pc12(
    out_path: Path,
    r0_proj: np.ndarray,
    bx_proj: np.ndarray,
    by_proj: np.ndarray,
    selected: np.ndarray,
    *,
    var_fraction: np.ndarray,
) -> None:
    idx = np.asarray(selected, dtype=np.int64)
    spread = float(np.std(r0_proj[:, :2])) * 0.30
    bx_u = _unit_rows(bx_proj[:, :2])
    by_u = _unit_rows(by_proj[:, :2])
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    ax.scatter(r0_proj[:, 0], r0_proj[:, 1], s=8, color="0.82", alpha=0.55, linewidths=0, zorder=1)
    ax.quiver(
        r0_proj[idx, 0],
        r0_proj[idx, 1],
        bx_u[idx, 0] * spread,
        bx_u[idx, 1] * spread,
        color=BX_COLOR,
        scale=1.0,
        scale_units="xy",
        angles="xy",
        width=0.004,
        headwidth=4,
        headlength=5,
        alpha=0.88,
        zorder=4,
    )
    ax.quiver(
        r0_proj[idx, 0],
        r0_proj[idx, 1],
        by_u[idx, 0] * spread,
        by_u[idx, 1] * spread,
        color=BY_COLOR,
        scale=1.0,
        scale_units="xy",
        angles="xy",
        width=0.004,
        headwidth=4,
        headlength=5,
        alpha=0.88,
        zorder=4,
    )
    ax.scatter(r0_proj[idx, 0], r0_proj[idx, 1], s=18, color="white", edgecolors="0.45", linewidths=0.5, zorder=5)
    ax.set_xlabel(f"Static response PC1 ({100.0 * var_fraction[0]:.1f}% var.)")
    ax.set_ylabel(f"Static response PC2 ({100.0 * var_fraction[1]:.1f}% var.)")
    ax.set_title("BackImage cardinal local translation tangents")
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], color=BX_COLOR, lw=1.5, label=r"$b_x(I)$ horizontal"),
            Line2D([0], [0], color=BY_COLOR, lw=1.5, label=r"$b_y(I)$ vertical"),
        ],
        frameon=False,
        loc="upper right",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_pc123(
    out_path: Path,
    r0_proj: np.ndarray,
    bx_proj: np.ndarray,
    by_proj: np.ndarray,
    selected: np.ndarray,
    *,
    var_fraction: np.ndarray,
) -> None:
    idx = np.asarray(selected, dtype=np.int64)
    spread = float(np.std(r0_proj[:, :3])) * 0.32
    bx_u = _unit_rows(bx_proj[:, :3])
    by_u = _unit_rows(by_proj[:, :3])
    bx_end = r0_proj[idx, :3] + bx_u[idx] * spread
    by_end = r0_proj[idx, :3] + by_u[idx] * spread
    all_plot_points = np.concatenate([r0_proj[:, :3], bx_end, by_end], axis=0)
    center, span = _axis_limits(all_plot_points)
    views = [(18, -60), (18, 35), (55, -60), (8, -100)]
    fig = plt.figure(figsize=(13.0, 10.0), constrained_layout=True)
    for view_idx, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, view_idx + 1, projection="3d")
        ax.scatter(
            r0_proj[:, 0],
            r0_proj[:, 1],
            r0_proj[:, 2],
            s=7,
            color="0.78",
            alpha=0.42,
            linewidths=0,
            depthshade=False,
        )
        starts = r0_proj[idx, :3]
        ax.quiver(
            starts[:, 0],
            starts[:, 1],
            starts[:, 2],
            bx_u[idx, 0] * spread,
            bx_u[idx, 1] * spread,
            bx_u[idx, 2] * spread,
            length=1.0,
            normalize=False,
            arrow_length_ratio=0.22,
            linewidth=0.85,
            color=BX_COLOR,
            alpha=0.82,
        )
        ax.quiver(
            starts[:, 0],
            starts[:, 1],
            starts[:, 2],
            by_u[idx, 0] * spread,
            by_u[idx, 1] * spread,
            by_u[idx, 2] * spread,
            length=1.0,
            normalize=False,
            arrow_length_ratio=0.22,
            linewidth=0.85,
            color=BY_COLOR,
            alpha=0.82,
        )
        ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2], s=16, color="white", edgecolors="0.35", linewidths=0.5)
        ax.view_init(elev=elev, azim=azim)
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, strict=False):
            setter(float(c - 0.56 * span), float(c + 0.56 * span))
        ax.set_xlabel(f"PC1 ({100.0 * var_fraction[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100.0 * var_fraction[1]:.1f}%)")
        ax.set_zlabel(f"PC3 ({100.0 * var_fraction[2]:.1f}%)")
        ax.set_title(f"elev {elev}, azim {azim}", fontsize=9)
    fig.suptitle("BackImage cardinal local translation tangents in static-response PCA", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_interactive_pc123(
    out_path: Path,
    r0_proj: np.ndarray,
    bx_proj: np.ndarray,
    by_proj: np.ndarray,
    selected: np.ndarray,
    object_ids: np.ndarray,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False
    idx = np.asarray(selected, dtype=np.int64)
    spread = float(np.std(r0_proj[:, :3])) * 0.32
    bx_u = _unit_rows(bx_proj[:, :3])
    by_u = _unit_rows(by_proj[:, :3])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=r0_proj[:, 0],
            y=r0_proj[:, 1],
            z=r0_proj[:, 2],
            mode="markers",
            marker={"size": 2.4, "color": "rgba(120,120,120,0.45)"},
            text=[str(v) for v in object_ids.tolist()],
            hoverinfo="text",
            name="static fixation responses",
        )
    )
    for label, color, vecs in (("bx horizontal", BX_COLOR, bx_u), ("by vertical", BY_COLOR, by_u)):
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        text: list[str | None] = []
        for i in idx.tolist():
            start = r0_proj[i, :3]
            end = start + vecs[i, :3] * spread
            xs.extend([float(start[0]), float(end[0]), None])
            ys.extend([float(start[1]), float(end[1]), None])
            zs.extend([float(start[2]), float(end[2]), None])
            hover = f"{label}<br>{object_ids[i]}"
            text.extend([hover, hover, None])
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": color, "width": 4.0},
                text=text,
                hoverinfo="text",
                name=label,
            )
        )
    fig.update_layout(
        title="BackImage cardinal local translation tangents",
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3", "aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return True


def _summary_rows(
    *,
    object_ids: np.ndarray,
    source_rows: np.ndarray,
    sessions: np.ndarray,
    trial_indices: np.ndarray,
    r0: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bx_proj: np.ndarray,
    by_proj: np.ndarray,
) -> list[dict[str, object]]:
    bx_norm = np.linalg.norm(bx, axis=1)
    by_norm = np.linalg.norm(by, axis=1)
    cos = np.sum(bx * by, axis=1) / (bx_norm * by_norm + 1e-12)
    bx_proj_norm = np.linalg.norm(bx_proj[:, :3], axis=1)
    by_proj_norm = np.linalg.norm(by_proj[:, :3], axis=1)
    cos_proj = np.sum(bx_proj[:, :3] * by_proj[:, :3], axis=1) / (bx_proj_norm * by_proj_norm + 1e-12)
    rows: list[dict[str, object]] = []
    for i in range(int(r0.shape[0])):
        rows.append(
            {
                "object_index": int(i),
                "object_id": str(object_ids[i]),
                "source_row": int(source_rows[i]),
                "session": str(sessions[i]),
                "trial_idx": int(trial_indices[i]),
                "r0_mean_rate": float(np.mean(r0[i])),
                "norm_bx": float(bx_norm[i]),
                "norm_by": float(by_norm[i]),
                "norm_bx_over_by": float(bx_norm[i] / (by_norm[i] + 1e-12)),
                "cos_bx_by": float(cos[i]),
                "angle_bx_by_deg": float(np.degrees(np.arccos(np.clip(cos[i], -1.0, 1.0)))),
                "cos_bx_by_pc123": float(cos_proj[i]),
                "angle_bx_by_pc123_deg": float(np.degrees(np.arccos(np.clip(cos_proj[i], -1.0, 1.0)))),
            }
        )
    return rows


def _scalar_summary(rows: list[dict[str, object]], bx: np.ndarray, by: np.ndarray, evals: np.ndarray) -> dict[str, Any]:
    def vals(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    total = float(np.sum(evals)) + 1e-12
    cos_xx = _sampled_pairwise_cosines(bx, seed=13)
    cos_yy = _sampled_pairwise_cosines(by, seed=17)
    return {
        "n_objects": int(len(rows)),
        "n_features": int(bx.shape[1]),
        "static_response_pca_fraction_first3": [float(v / total) for v in evals[:3]],
        "norm_bx_median": float(np.nanmedian(vals("norm_bx"))),
        "norm_by_median": float(np.nanmedian(vals("norm_by"))),
        "norm_bx_over_by_median": float(np.nanmedian(vals("norm_bx_over_by"))),
        "within_object_cos_bx_by_median": float(np.nanmedian(vals("cos_bx_by"))),
        "within_object_angle_bx_by_deg_median": float(np.nanmedian(vals("angle_bx_by_deg"))),
        "within_object_angle_bx_by_deg_iqr": [
            float(v) for v in np.nanpercentile(vals("angle_bx_by_deg"), [25, 75])
        ],
        "within_object_angle_bx_by_pc123_deg_median": float(np.nanmedian(vals("angle_bx_by_pc123_deg"))),
        "cross_object_bx_bx_cos_median": float(np.nanmedian(cos_xx)) if cos_xx.size else float("nan"),
        "cross_object_by_by_cos_median": float(np.nanmedian(cos_yy)) if cos_yy.size else float("nan"),
        "cross_object_bx_bx_cos_iqr": [float(v) for v in np.nanpercentile(cos_xx, [25, 75])] if cos_xx.size else [],
        "cross_object_by_by_cos_iqr": [float(v) for v in np.nanpercentile(cos_yy, [25, 75])] if cos_yy.size else [],
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = _prepare_windows(args)
    if work.empty:
        raise ValueError("No BackImage windows survived filters")
    step_deg = float(args.step_arcmin) / 60.0
    if not np.isfinite(step_deg) or step_deg <= 0.0:
        raise ValueError("--step-arcmin must be positive and finite")
    traces = [
        _static_trace(int(args.n_timepoints)),
        _constant_trace(step_deg, 0.0, int(args.n_timepoints)),
        _constant_trace(-step_deg, 0.0, int(args.n_timepoints)),
        _constant_trace(0.0, step_deg, int(args.n_timepoints)),
        _constant_trace(0.0, -step_deg, int(args.n_timepoints)),
    ]
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.twin_batch_size))
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    object_ids: list[str] = []
    source_rows: list[int] = []
    sessions: list[str] = []
    trial_indices: list[int] = []
    r0_rows: list[np.ndarray] = []
    rxp_rows: list[np.ndarray] = []
    rxm_rows: list[np.ndarray] = []
    ryp_rows: list[np.ndarray] = []
    rym_rows: list[np.ndarray] = []
    patch_rows: list[dict[str, object]] = []
    _progress(
        f"scoring {work.shape[0]} BackImage fixation objects; step={args.step_arcmin:g} arcmin; "
        f"reduction={args.response_reduction}; device={args.device}"
    )
    for object_index, (_row_index, row) in enumerate(tqdm(list(work.iterrows()), desc="cardinal tangent objects")):
        patch, patch_meta = _extract_patch(row, canvas_cache=canvas_cache, patch_size_px=int(args.patch_size_px))
        raw = scorer.responses(patch, traces, trace_batch_size=int(args.twin_trace_batch_size))
        aligned = [_align_response_to_trace(resp, int(args.n_timepoints)) for resp in raw]
        reduced = [_reduce_response(resp, str(args.response_reduction)) for resp in aligned]
        r0_rows.append(reduced[0])
        rxp_rows.append(reduced[1])
        rxm_rows.append(reduced[2])
        ryp_rows.append(reduced[3])
        rym_rows.append(reduced[4])
        object_ids.append(f"source_row:{int(row['source_row'])}")
        source_rows.append(int(row["source_row"]))
        sessions.append(str(row["session"]))
        trial_indices.append(int(row["trial_idx"]))
        patch_rows.append(
            {
                "object_index": int(object_index),
                "object_id": object_ids[-1],
                "source_row": source_rows[-1],
                "session": sessions[-1],
                "trial_idx": trial_indices[-1],
                "mean_x_deg": float(row["mean_x_deg"]),
                "mean_y_deg": float(row["mean_y_deg"]),
                "patch_center_x_px": float(patch_meta["patch_center_x_px"]),
                "patch_center_y_px": float(patch_meta["patch_center_y_px"]),
                "patch_ppd": float(patch_meta["patch_ppd"]),
                "image_orientation_coherence": float(row.get("image_orientation_coherence", np.nan)),
                "image_edge_axis_deg": float(row.get("image_edge_axis_deg", np.nan)),
                "anisotropy": float(row.get("anisotropy", np.nan)),
            }
        )
        if (object_index + 1) == 1 or (object_index + 1) == work.shape[0] or (
            int(args.progress_every) > 0 and (object_index + 1) % int(args.progress_every) == 0
        ):
            _progress(f"objects {object_index + 1}/{work.shape[0]}")

    r0 = np.stack(r0_rows, axis=0).astype(np.float32, copy=False)
    rxp = np.stack(rxp_rows, axis=0).astype(np.float32, copy=False)
    rxm = np.stack(rxm_rows, axis=0).astype(np.float32, copy=False)
    ryp = np.stack(ryp_rows, axis=0).astype(np.float32, copy=False)
    rym = np.stack(rym_rows, axis=0).astype(np.float32, copy=False)
    bx = ((rxp - rxm) / (2.0 * step_deg)).astype(np.float32, copy=False)
    by = ((ryp - rym) / (2.0 * step_deg)).astype(np.float32, copy=False)
    finite = (
        np.isfinite(r0).all(axis=1)
        & np.isfinite(bx).all(axis=1)
        & np.isfinite(by).all(axis=1)
        & (np.linalg.norm(bx, axis=1) > float(args.tangent_norm_eps))
        & (np.linalg.norm(by, axis=1) > float(args.tangent_norm_eps))
    )
    if not np.all(finite):
        keep = np.flatnonzero(finite)
        r0, rxp, rxm, ryp, rym, bx, by = (arr[keep] for arr in (r0, rxp, rxm, ryp, rym, bx, by))
        object_ids = [object_ids[int(i)] for i in keep.tolist()]
        source_rows = [source_rows[int(i)] for i in keep.tolist()]
        sessions = [sessions[int(i)] for i in keep.tolist()]
        trial_indices = [trial_indices[int(i)] for i in keep.tolist()]
    if r0.shape[0] < 3:
        raise ValueError(f"Need at least 3 finite tangent objects for plotting, got {r0.shape[0]}")

    mean, basis, evals = _fit_pca(r0, n_components=3)
    total = float(np.sum(evals)) + 1e-12
    var_fraction = evals[:3] / total
    r0_proj = _project_points(r0, mean, basis)
    bx_proj = _project_vectors(bx, basis)
    by_proj = _project_vectors(by, basis)
    selected = _farthest_point_subset(r0_proj[:, :3], min(int(args.n_show), r0.shape[0]), seed=int(args.seed) + 2)
    object_id_arr = np.asarray(object_ids)
    source_row_arr = np.asarray(source_rows, dtype=np.int64)
    session_arr = np.asarray(sessions)
    trial_arr = np.asarray(trial_indices, dtype=np.int64)
    np.savez_compressed(
        out_dir / "backimage_cardinal_tangent_cache.npz",
        r0=r0,
        bx=bx,
        by=by,
        rx_p=rxp,
        rx_m=rxm,
        ry_p=ryp,
        ry_m=rym,
        object_ids=object_id_arr,
        source_rows=source_row_arr,
        sessions=session_arr,
        trial_indices=trial_arr,
        pca_mean=mean.astype(np.float32),
        pca_basis=basis.astype(np.float32),
        pca_eigenvalues=evals.astype(np.float32),
        r0_pc123=r0_proj.astype(np.float32),
        bx_pc123=bx_proj.astype(np.float32),
        by_pc123=by_proj.astype(np.float32),
        selected_indices=selected.astype(np.int64),
        step_arcmin=np.asarray([float(args.step_arcmin)], dtype=np.float32),
        step_deg=np.asarray([float(step_deg)], dtype=np.float32),
        response_reduction=np.asarray([str(args.response_reduction)]),
    )
    rows = _summary_rows(
        object_ids=object_id_arr,
        source_rows=source_row_arr,
        sessions=session_arr,
        trial_indices=trial_arr,
        r0=r0,
        bx=bx,
        by=by,
        bx_proj=bx_proj,
        by_proj=by_proj,
    )
    summary = _scalar_summary(rows, bx, by, evals)
    summary.update(
        {
            "input": str(args.input),
            "out_dir": str(out_dir),
            "step_arcmin": float(args.step_arcmin),
            "step_deg": float(step_deg),
            "response_reduction": str(args.response_reduction),
            "n_objects_requested": int(work.shape[0]),
            "n_objects_dropped_nonfinite_or_zero": int(work.shape[0] - r0.shape[0]),
            "selected_indices": [int(v) for v in selected.tolist()],
        }
    )
    _write_csv(out_dir / "backimage_cardinal_tangent_objects.csv", rows)
    _write_csv(out_dir / "backimage_cardinal_tangent_patch_metadata.csv", patch_rows)
    _write_json(out_dir / "backimage_cardinal_tangent_summary.json", summary)
    _write_json(out_dir / "run_config.json", vars(args))

    fig_dir = out_dir / "figures"
    _plot_pc12(
        fig_dir / "backimage_cardinal_tangent_charts_pc12.png",
        r0_proj,
        bx_proj,
        by_proj,
        selected,
        var_fraction=var_fraction,
    )
    _plot_pc123(
        fig_dir / "backimage_cardinal_tangent_charts_pc123.png",
        r0_proj,
        bx_proj,
        by_proj,
        selected,
        var_fraction=var_fraction,
    )
    interactive_path = fig_dir / "backimage_cardinal_tangent_charts_pc123.html"
    if _plot_interactive_pc123(interactive_path, r0_proj, bx_proj, by_proj, selected, object_id_arr):
        summary["interactive_pc123_html"] = str(interactive_path)
        _write_json(out_dir / "backimage_cardinal_tangent_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute and plot BackImage per-fixation cardinal x/y local translation tangents."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-manifest", type=Path, default=None)
    parser.add_argument("--max-images", "--max-objects", dest="max_images", type=int, default=64)
    parser.add_argument("--step-arcmin", type=float, default=0.25)
    parser.add_argument("--response-reduction", choices=("mean_time", "flatten_time", "last_time"), default="mean_time")
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    parser.add_argument("--n-show", type=int, default=24)
    parser.add_argument("--tangent-norm-eps", type=float, default=1e-12)
    parser.add_argument("--twin-batch-size", type=int, default=96)
    parser.add_argument("--twin-trace-batch-size", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=8)
    return parser


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
