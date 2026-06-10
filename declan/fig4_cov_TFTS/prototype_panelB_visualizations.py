#!/usr/bin/env python3
"""Prototype alternative Panel B visualizations for Figure 4 geometry.

The variants in this script use saved baseline responses r0(I), local tangent
vectors bx(I)/by(I), and finite-shift endpoint responses when present.  Any
smooth response-landscape surface is only a visual guide through sampled
baseline response states, not a generative image manifold.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


TEXT = "#202124"
BLUE = "#2f5f9f"
PURPLE = "#7b5ea7"
GRAY = "#b0b0b0"
LIGHT_GRAY = "#d8d8d8"
SURFACE_GRAY = "#aeb4bc"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
    }
)


@dataclass
class TangentPayload:
    object_ids: list[str]
    image_ids: np.ndarray
    trial_indices: np.ndarray
    time_indices: np.ndarray
    r0: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    history: np.ndarray | None
    rx_p: np.ndarray | None
    rx_m: np.ndarray | None
    ry_p: np.ndarray | None
    ry_m: np.ndarray | None
    requested_delta: float
    delta: float
    delta_model_px: float
    n_loaded: int
    n_valid: int
    dropped_object_ids: list[str]


@dataclass
class Embedding:
    mean: np.ndarray
    components: np.ndarray
    scores: np.ndarray
    variance_explained: np.ndarray


def _clean_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, color="0.88", lw=0.6, zorder=-10)


def _set_equal_2d(ax: plt.Axes, coords: np.ndarray, pad: float = 0.12) -> None:
    lo = np.nanmin(coords[:, :2], axis=0)
    hi = np.nanmax(coords[:, :2], axis=0)
    center = 0.5 * (lo + hi)
    span = float(max(hi - lo))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    half = 0.5 * span * (1.0 + pad)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal", adjustable="box")


def _set_equal_3d(ax: plt.Axes, coords: np.ndarray, pad: float = 0.08) -> None:
    lo = np.nanmin(coords[:, :3], axis=0)
    hi = np.nanmax(coords[:, :3], axis=0)
    center = 0.5 * (lo + hi)
    span = float(max(hi - lo))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    half = 0.5 * span * (1.0 + pad)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1, 1, 0.72))
    except Exception:
        pass


def _soften_3d_grid(ax: plt.Axes) -> None:
    ax.grid(True)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.16)
        axis._axinfo["grid"]["linewidth"] = 0.45
        axis._axinfo["axisline"]["color"] = (0.25, 0.25, 0.25, 0.55)
        axis._axinfo["tick"]["color"] = (0.25, 0.25, 0.25, 0.70)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.9, 0.9, 0.9, 0.20))
    ax.yaxis.pane.set_edgecolor((0.9, 0.9, 0.9, 0.20))
    ax.zaxis.pane.set_edgecolor((0.9, 0.9, 0.9, 0.20))


def _focus_3d_on_surface(ax: plt.Axes, coords: np.ndarray) -> None:
    lo = np.nanpercentile(coords[:, :3], 5, axis=0)
    hi = np.nanpercentile(coords[:, :3], 95, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = span * np.asarray([0.01, 0.01, 0.04])
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    try:
        aspect = span / np.max(span)
        ax.set_box_aspect(tuple(aspect), zoom=1.30)
    except TypeError:
        try:
            aspect = span / np.max(span)
            ax.set_box_aspect(tuple(aspect))
        except Exception:
            pass


def _hide_3d_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.set_axis_off()


def _add_pc_triad(ax: plt.Axes, coords: np.ndarray, emb: Embedding) -> None:
    lo = np.asarray([lim[0] for lim in [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]], dtype=float)
    hi = np.asarray([lim[1] for lim in [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]], dtype=float)
    span = hi - lo
    origin = lo + span * np.asarray([0.08, 0.13, 0.13])
    length = float(np.min(span[:2]) * 0.16)
    specs = [
        (np.asarray([1.0, 0.0, 0.0]), "PC1", TEXT),
        (np.asarray([0.0, 1.0, 0.0]), "PC2", TEXT),
        (np.asarray([0.0, 0.0, 1.0]), "PC3", TEXT),
    ]
    for direction, label, color in specs:
        vec = direction * length
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vec[0],
            vec[1],
            vec[2],
            color=color,
            arrow_length_ratio=0.22,
            lw=0.85,
            alpha=0.82,
        )
        end = origin + vec * 1.12
        ax.text(end[0], end[1], end[2], label, color=color, fontsize=6.2, ha="center", va="center")


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 300) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _pca(x: np.ndarray, n_components: int = 3) -> Embedding:
    mean = np.nanmean(x, axis=0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    eig = s**2
    total = float(np.sum(eig))
    variance_explained = eig / total if total > 0 else np.zeros_like(eig)
    components = vt[:n_components].copy()
    scores = xc @ components.T
    return Embedding(mean=mean, components=components, scores=scores, variance_explained=variance_explained)


def _project_points(x: np.ndarray, emb: Embedding) -> np.ndarray:
    return (x - emb.mean) @ emb.components.T


def _project_vectors(x: np.ndarray, emb: Embedding) -> np.ndarray:
    return x @ emb.components.T


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _farthest_subset(coords: np.ndarray, n: int, seed: int = 3) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    n = min(int(n), coords.shape[0])
    if n <= 0:
        return np.array([], dtype=int)
    finite = np.isfinite(coords).all(axis=1)
    candidates = np.where(finite)[0]
    if candidates.size <= n:
        return candidates

    rng = np.random.default_rng(seed)
    center = np.nanmedian(coords[candidates], axis=0)
    first = candidates[int(np.argmin(np.linalg.norm(coords[candidates] - center, axis=1)))]
    chosen = [int(first)]
    min_dist = np.linalg.norm(coords[candidates] - coords[first], axis=1)
    while len(chosen) < n:
        jitter = rng.uniform(0.0, 1e-9, size=candidates.size)
        nxt = candidates[int(np.argmax(min_dist + jitter))]
        chosen.append(int(nxt))
        min_dist = np.minimum(min_dist, np.linalg.norm(coords[candidates] - coords[nxt], axis=1))
    return np.asarray(chosen, dtype=int)


def _choose_label_position(coords: np.ndarray) -> tuple[float, float, str, str]:
    x_med = float(np.median(coords[:, 0]))
    y_med = float(np.median(coords[:, 1]))
    corners = [
        (0.04, 0.95, "left", "top", coords[:, 0] < x_med, coords[:, 1] > y_med),
        (0.96, 0.95, "right", "top", coords[:, 0] > x_med, coords[:, 1] > y_med),
        (0.04, 0.06, "left", "bottom", coords[:, 0] < x_med, coords[:, 1] < y_med),
        (0.96, 0.06, "right", "bottom", coords[:, 0] > x_med, coords[:, 1] < y_med),
    ]
    best = min(corners, key=lambda c: int(np.sum(c[4] & c[5])))
    return best[0], best[1], best[2], best[3]


def load_payload(path: Path, delta: float) -> TangentPayload:
    with path.open("rb") as handle:
        cached = pickle.load(handle)
    available = sorted(float(d) for d in cached["object_payload"].keys())
    delta_key = min(available, key=lambda d: abs(d - float(delta)))
    raw = cached["object_payload"][delta_key]
    object_ids = sorted(str(oid) for oid in raw.keys())
    required = ["r0", "bx", "by"]
    optional = ["rx_p", "rx_m", "ry_p", "ry_m"]
    for field in required:
        if field not in raw[object_ids[0]]:
            raise KeyError(f"Payload does not contain required field {field!r}.")

    arrays = {
        field: np.vstack([np.asarray(raw[oid][field], dtype=np.float64) for oid in object_ids])
        for field in required + [f for f in optional if f in raw[object_ids[0]]]
    }
    histories = None
    if "history" in raw[object_ids[0]]:
        histories = np.stack([np.asarray(raw[oid]["history"], dtype=np.float64) for oid in object_ids], axis=0)
    valid = np.ones(len(object_ids), dtype=bool)
    for arr in arrays.values():
        valid &= np.isfinite(arr).all(axis=1)
    if histories is not None:
        valid &= np.isfinite(histories.reshape(histories.shape[0], -1)).all(axis=1)

    kept_ids = [oid for oid, ok in zip(object_ids, valid) if ok]
    dropped = [oid for oid, ok in zip(object_ids, valid) if not ok]
    meta = [raw[oid] for oid in kept_ids]
    return TangentPayload(
        object_ids=kept_ids,
        image_ids=np.asarray([int(m["image_id"]) for m in meta], dtype=int),
        trial_indices=np.asarray([int(m["trial_index"]) for m in meta], dtype=int),
        time_indices=np.asarray([int(m["time_index"]) for m in meta], dtype=int),
        r0=arrays["r0"][valid],
        bx=arrays["bx"][valid],
        by=arrays["by"][valid],
        history=histories[valid] if histories is not None else None,
        rx_p=arrays.get("rx_p", None)[valid] if "rx_p" in arrays else None,
        rx_m=arrays.get("rx_m", None)[valid] if "rx_m" in arrays else None,
        ry_p=arrays.get("ry_p", None)[valid] if "ry_p" in arrays else None,
        ry_m=arrays.get("ry_m", None)[valid] if "ry_m" in arrays else None,
        requested_delta=float(delta),
        delta=float(delta_key),
        delta_model_px=float(raw[object_ids[0]].get("delta_model_px", np.nan)),
        n_loaded=len(object_ids),
        n_valid=int(np.sum(valid)),
        dropped_object_ids=dropped,
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    )


def _knn_overlap(full: np.ndarray, low: np.ndarray, k: int = 5) -> float:
    n = full.shape[0]
    k = max(1, min(int(k), n - 1))
    full_d2 = np.sum((full[:, None, :] - full[None, :, :]) ** 2, axis=2)
    low_d2 = np.sum((low[:, None, :] - low[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(full_d2, np.inf)
    np.fill_diagonal(low_d2, np.inf)
    full_nn = np.argpartition(full_d2, kth=k - 1, axis=1)[:, :k]
    low_nn = np.argpartition(low_d2, kth=k - 1, axis=1)[:, :k]
    overlap = [len(set(full_nn[i]).intersection(set(low_nn[i]))) / k for i in range(n)]
    return float(np.median(overlap))


def _pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    gram = x @ x.T
    diag = np.diag(gram)
    d2 = diag[:, None] + diag[None, :] - 2.0 * gram
    return np.maximum(d2, 0.0)


def _knn_indices_from_d2(d2: np.ndarray, k: int) -> np.ndarray:
    d = d2.copy()
    np.fill_diagonal(d, np.inf)
    k = max(1, min(int(k), d.shape[0] - 1))
    idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    row = np.arange(d.shape[0])[:, None]
    return idx[row, np.argsort(d[row, idx], axis=1)]


def _rank_matrix_from_d2(d2: np.ndarray) -> np.ndarray:
    order = np.argsort(d2, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    for i in range(order.shape[0]):
        ranks[i, order[i]] = np.arange(order.shape[1], dtype=np.int32)
    return ranks


def _trustworthiness(high_d2: np.ndarray, low_d2: np.ndarray, k: int) -> float:
    n = high_d2.shape[0]
    k = max(1, min(int(k), n - 1))
    denom = n * k * (2 * n - 3 * k - 1)
    if denom <= 0:
        return float("nan")
    high_nn = _knn_indices_from_d2(high_d2, k)
    low_nn = _knn_indices_from_d2(low_d2, k)
    high_ranks = _rank_matrix_from_d2(high_d2)
    penalty = 0.0
    for i in range(n):
        high_set = set(int(j) for j in high_nn[i])
        for j in low_nn[i]:
            if int(j) not in high_set:
                penalty += float(high_ranks[i, int(j)] - k)
    return float(1.0 - (2.0 / denom) * penalty)


def _continuity(high_d2: np.ndarray, low_d2: np.ndarray, k: int) -> float:
    return _trustworthiness(low_d2, high_d2, k)


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sorted_x = x[order]
    start = 0
    while start < x.size:
        stop = start + 1
        while stop < x.size and sorted_x[stop] == sorted_x[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    rx = _rankdata_average(np.asarray(x)[mask])
    ry = _rankdata_average(np.asarray(y)[mask])
    rx = rx - np.mean(rx)
    ry = ry - np.mean(ry)
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(np.dot(rx, ry) / denom) if denom > 0 else float("nan")


def _participation_ratio_from_spectrum(eig: np.ndarray) -> float:
    eig = np.asarray(eig, dtype=float)
    eig = eig[np.isfinite(eig) & (eig > 0)]
    if eig.size == 0:
        return float("nan")
    return float(np.sum(eig) ** 2 / (np.sum(eig**2) + 1e-12))


def _response_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xc = x - np.mean(x, axis=0, keepdims=True)
    s = np.linalg.svd(xc, full_matrices=False, compute_uv=False)
    eig = s**2
    total = float(np.sum(eig))
    var = eig / total if total > 0 else np.zeros_like(eig)
    return eig, var


def _unit_shuffle_response_null(
    r0: np.ndarray,
    n_shuffles: int,
    seed: int = 47,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, p = r0.shape
    rows: list[dict[str, float | int]] = []
    for shuffle in range(int(n_shuffles)):
        shuffled = np.empty_like(r0)
        for unit in range(p):
            shuffled[:, unit] = r0[rng.permutation(n), unit]
        eig, var = _response_spectrum(shuffled)
        rows.append(
            {
                "shuffle": shuffle,
                "pc1_3_variance_explained": float(np.sum(var[:3])),
                "pc1_5_variance_explained": float(np.sum(var[:5])),
                "pc1_10_variance_explained": float(np.sum(var[:10])),
                "participation_ratio": _participation_ratio_from_spectrum(eig),
            }
        )
    return pd.DataFrame(rows)


def _orthonormal_tangent_bases(bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    bases = []
    for x_vec, y_vec in zip(bx, by):
        mat = np.column_stack([x_vec, y_vec])
        u, _, _ = np.linalg.svd(mat, full_matrices=False)
        bases.append(u[:, :2])
    return np.stack(bases, axis=0)


def _plane_similarity(bases: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    sims = []
    for a, b in zip(np.asarray(i, dtype=int), np.asarray(j, dtype=int)):
        cross = bases[a].T @ bases[b]
        sims.append(float(np.sum(cross**2) / 2.0))
    return np.asarray(sims, dtype=float)


def compute_response_state_manifold_audit(
    payload: TangentPayload,
    emb: Embedding,
    out_dir: Path,
    neighbor_k: int = 5,
    n_shuffles: int = 200,
    seed: int = 53,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eig, var = _response_spectrum(payload.r0)
    null_df = _unit_shuffle_response_null(payload.r0, n_shuffles=n_shuffles, seed=seed)
    null_summary = {}
    for col in [
        "pc1_3_variance_explained",
        "pc1_5_variance_explained",
        "pc1_10_variance_explained",
        "participation_ratio",
    ]:
        values = null_df[col].to_numpy(dtype=float)
        null_summary[f"unit_shuffle_{col}_mean"] = float(np.mean(values))
        null_summary[f"unit_shuffle_{col}_ci_low"] = float(np.percentile(values, 2.5))
        null_summary[f"unit_shuffle_{col}_ci_high"] = float(np.percentile(values, 97.5))

    high_d2 = _pairwise_sq_dists(payload.r0)
    low_d2 = _pairwise_sq_dists(emb.scores[:, :3])
    tri = np.triu_indices(payload.r0.shape[0], k=1)
    full_dist = np.sqrt(high_d2[tri])
    low_dist = np.sqrt(low_d2[tri])

    bases = _orthonormal_tangent_bases(payload.bx, payload.by)
    nn = _knn_indices_from_d2(high_d2, neighbor_k)
    rng = np.random.default_rng(seed + 1)
    neighbor_rows: list[dict[str, float | int | str]] = []
    for i in range(payload.r0.shape[0]):
        neighbor_js = nn[i]
        forbidden = set(int(j) for j in neighbor_js)
        forbidden.add(i)
        random_pool = np.asarray([j for j in range(payload.r0.shape[0]) if j not in forbidden], dtype=int)
        if random_pool.size < neighbor_k:
            random_pool = np.asarray([j for j in range(payload.r0.shape[0]) if j != i], dtype=int)
        random_js = rng.choice(random_pool, size=neighbor_k, replace=random_pool.size < neighbor_k)
        for label, js in [("knn", neighbor_js), ("random_non_neighbor", random_js)]:
            ii = np.full(len(js), i, dtype=int)
            plane = _plane_similarity(bases, ii, js)
            bx_abs = np.abs(_cosine(payload.bx[ii], payload.bx[js]))
            by_abs = np.abs(_cosine(payload.by[ii], payload.by[js]))
            neighbor_rows.append(
                {
                    "object_index": i,
                    "comparison": label,
                    "mean_response_distance": float(np.mean(np.sqrt(high_d2[i, js]))),
                    "mean_tangent_plane_similarity": float(np.mean(plane)),
                    "mean_abs_cos_bx": float(np.mean(bx_abs)),
                    "mean_abs_cos_by": float(np.mean(by_abs)),
                }
            )
    neighbor_df = pd.DataFrame(neighbor_rows)

    pair_plane = _plane_similarity(bases, tri[0], tri[1])
    pair_bx = np.abs(_cosine(payload.bx[tri[0]], payload.bx[tri[1]]))
    pair_by = np.abs(_cosine(payload.by[tri[0]], payload.by[tri[1]]))
    quantiles = np.quantile(full_dist, np.linspace(0.0, 1.0, 6))
    quantiles[0] -= 1e-12
    quantiles[-1] += 1e-12
    bin_rows = []
    for b in range(5):
        mask = (full_dist > quantiles[b]) & (full_dist <= quantiles[b + 1])
        bin_rows.append(
            {
                "distance_bin": b + 1,
                "full_response_distance_low": float(quantiles[b]),
                "full_response_distance_high": float(quantiles[b + 1]),
                "n_pairs": int(np.sum(mask)),
                "median_tangent_plane_similarity": float(np.median(pair_plane[mask])),
                "median_abs_cos_bx": float(np.median(pair_bx[mask])),
                "median_abs_cos_by": float(np.median(pair_by[mask])),
            }
        )
    distance_bin_df = pd.DataFrame(bin_rows)

    knn_rows = neighbor_df[neighbor_df["comparison"] == "knn"]
    random_rows = neighbor_df[neighbor_df["comparison"] == "random_non_neighbor"]
    audit: dict[str, float | int] = {
        "r0_pc1_3_variance_explained": float(np.sum(var[:3])),
        "r0_pc1_5_variance_explained": float(np.sum(var[:5])),
        "r0_pc1_10_variance_explained": float(np.sum(var[:10])),
        "r0_participation_ratio": _participation_ratio_from_spectrum(eig),
        "neighbor_k": int(neighbor_k),
        "n_unit_shuffles": int(n_shuffles),
        "knn_overlap_full_vs_pc1_3": _knn_overlap(payload.r0, emb.scores[:, :3], k=neighbor_k),
        "trustworthiness_pc1_3": _trustworthiness(high_d2, low_d2, k=neighbor_k),
        "continuity_pc1_3": _continuity(high_d2, low_d2, k=neighbor_k),
        "spearman_pairwise_distance_full_vs_pc1_3": _spearmanr(full_dist, low_dist),
        "median_knn_tangent_plane_similarity": float(np.median(knn_rows["mean_tangent_plane_similarity"])),
        "median_random_tangent_plane_similarity": float(np.median(random_rows["mean_tangent_plane_similarity"])),
        "median_knn_minus_random_tangent_plane_similarity": float(
            np.median(knn_rows["mean_tangent_plane_similarity"].to_numpy() - random_rows["mean_tangent_plane_similarity"].to_numpy())
        ),
        "median_knn_abs_cos_bx": float(np.median(knn_rows["mean_abs_cos_bx"])),
        "median_random_abs_cos_bx": float(np.median(random_rows["mean_abs_cos_bx"])),
        "median_knn_abs_cos_by": float(np.median(knn_rows["mean_abs_cos_by"])),
        "median_random_abs_cos_by": float(np.median(random_rows["mean_abs_cos_by"])),
        "spearman_response_distance_vs_tangent_plane_similarity": _spearmanr(full_dist, pair_plane),
        "spearman_response_distance_vs_abs_cos_bx": _spearmanr(full_dist, pair_bx),
        "spearman_response_distance_vs_abs_cos_by": _spearmanr(full_dist, pair_by),
        **null_summary,
    }

    null_df.to_csv(out_dir / "panelB_response_state_unit_shuffle_null.csv", index=False)
    neighbor_df.to_csv(out_dir / "panelB_tangent_neighbor_smoothness.csv", index=False)
    distance_bin_df.to_csv(out_dir / "panelB_tangent_smoothness_by_distance.csv", index=False)
    pd.DataFrame([audit]).to_csv(out_dir / "panelB_response_state_manifold_audit.csv", index=False)
    (out_dir / "panelB_response_state_manifold_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit, null_df, neighbor_df, distance_bin_df


def plot_response_state_manifold_audit(
    payload: TangentPayload,
    emb: Embedding,
    manifold_audit: dict[str, float | int],
    null_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    distance_bin_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    var = emb.variance_explained
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.5))

    ax = axes[0]
    kmax = min(20, var.size)
    ax.plot(np.arange(1, kmax + 1), np.cumsum(var[:kmax]), color=TEXT, lw=1.7, label="r0")
    null_cum = np.vstack(
        [
            [row["pc1_3_variance_explained"], row["pc1_5_variance_explained"], row["pc1_10_variance_explained"]]
            for _, row in null_df.iterrows()
        ]
    )
    ax.errorbar(
        [3, 5, 10],
        np.mean(null_cum, axis=0),
        yerr=[
            np.mean(null_cum, axis=0) - np.percentile(null_cum, 2.5, axis=0),
            np.percentile(null_cum, 97.5, axis=0) - np.mean(null_cum, axis=0),
        ],
        fmt="o",
        ms=3.5,
        color="0.55",
        ecolor="0.72",
        lw=0.8,
        capsize=2,
        label="unit shuffle",
    )
    ax.set_title("Baseline response spectrum", loc="left", fontweight="bold")
    ax.set_xlabel("PCs")
    ax.set_ylabel("cumulative variance")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", handlelength=1.3)
    _clean_axes(ax, grid=True)

    ax = axes[1]
    coords = emb.scores[:, :3]
    high_d2 = _pairwise_sq_dists(payload.r0)
    low_d2 = _pairwise_sq_dists(coords)
    tri = np.triu_indices(payload.r0.shape[0], k=1)
    ax.scatter(np.sqrt(high_d2[tri]), np.sqrt(low_d2[tri]), s=5, color="0.55", alpha=0.28, linewidths=0)
    ax.set_title("Neighborhood faithfulness", loc="left", fontweight="bold")
    ax.set_xlabel("full response distance")
    ax.set_ylabel("PC1-3 distance")
    ax.text(
        0.04,
        0.96,
        f"rho={float(manifold_audit['spearman_pairwise_distance_full_vs_pc1_3']):.2f}\n"
        f"T={float(manifold_audit['trustworthiness_pc1_3']):.2f}, C={float(manifold_audit['continuity_pc1_3']):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="0.30",
    )
    _clean_axes(ax, grid=True)

    ax = axes[2]
    x = distance_bin_df["distance_bin"].to_numpy(dtype=float)
    ax.plot(x, distance_bin_df["median_tangent_plane_similarity"], "-o", color=TEXT, ms=3.5, lw=1.4, label="plane")
    ax.plot(x, distance_bin_df["median_abs_cos_bx"], "-o", color=BLUE, ms=3.2, lw=1.1, label=r"$|b_x|$")
    ax.plot(x, distance_bin_df["median_abs_cos_by"], "-o", color=PURPLE, ms=3.2, lw=1.1, label=r"$|b_y|$")
    knn = neighbor_df[neighbor_df["comparison"] == "knn"]["mean_tangent_plane_similarity"].to_numpy(dtype=float)
    random = neighbor_df[neighbor_df["comparison"] == "random_non_neighbor"]["mean_tangent_plane_similarity"].to_numpy(dtype=float)
    ax.axhline(float(np.median(random)), color="0.55", ls=":", lw=1.0, label="random plane")
    ax.set_title("Tangent smoothness", loc="left", fontweight="bold")
    ax.set_xlabel("response-distance quintile")
    ax.set_ylabel("similarity")
    ax.set_ylim(0, 1.02)
    ax.text(
        0.04,
        0.08,
        f"kNN plane={float(np.median(knn)):.2f}\nrandom={float(np.median(random)):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="0.30",
    )
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    _clean_axes(ax, grid=True)

    fig.tight_layout(w_pad=1.0)
    return _save_figure(fig, out_dir, "panelB_response_state_manifold_audit")[0]


def compute_audit(payload: TangentPayload, emb: Embedding, out_dir: Path) -> dict[str, float | int | str]:
    proj_bx = _project_vectors(payload.bx, emb)
    proj_by = _project_vectors(payload.by, emb)
    frac_bx = np.sum(proj_bx**2, axis=1) / (np.sum(payload.bx**2, axis=1) + 1e-12)
    frac_by = np.sum(proj_by**2, axis=1) / (np.sum(payload.by**2, axis=1) + 1e-12)
    cos_full = np.clip(_cosine(payload.bx, payload.by), -1.0, 1.0)
    cos_low = np.clip(_cosine(proj_bx, proj_by), -1.0, 1.0)
    angle_error = np.abs(np.degrees(np.arccos(cos_low)) - np.degrees(np.arccos(cos_full)))
    audit: dict[str, float | int | str] = {
        "requested_delta_arcmin": payload.requested_delta,
        "delta_arcmin_used": payload.delta,
        "delta_model_px": payload.delta_model_px,
        "n_loaded": payload.n_loaded,
        "n_valid": payload.n_valid,
        "n_dropped_nonfinite": len(payload.dropped_object_ids),
        "dropped_object_ids": ";".join(payload.dropped_object_ids),
        "r0_pc1_variance_explained": float(emb.variance_explained[0]),
        "r0_pc2_variance_explained": float(emb.variance_explained[1]),
        "r0_pc3_variance_explained": float(emb.variance_explained[2]),
        "r0_pc1_3_variance_explained": float(np.sum(emb.variance_explained[:3])),
        "median_visible_energy_bx_pc1_3": float(np.median(frac_bx)),
        "median_visible_energy_by_pc1_3": float(np.median(frac_by)),
        "median_knn5_overlap_full_vs_pc1_3_r0": _knn_overlap(payload.r0, emb.scores[:, :3], k=5),
        "median_angle_error_deg_cos_bx_by_full_vs_pc1_3": float(np.median(angle_error)),
        "median_abs_cos_error_bx_by_full_vs_pc1_3": float(np.median(np.abs(cos_low - cos_full))),
        "median_full_cos_bx_by": float(np.median(cos_full)),
        "median_pc1_3_cos_bx_by": float(np.median(cos_low)),
    }
    pd.DataFrame([audit]).to_csv(out_dir / "panelB_embedding_audit.csv", index=False)
    (out_dir / "panelB_embedding_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def _draw_arrow_2d(
    ax: plt.Axes,
    start: np.ndarray,
    vec: np.ndarray,
    color: str,
    length: float,
    alpha: float = 0.9,
    lw: float = 1.25,
) -> np.ndarray:
    v = np.asarray(vec[:2], dtype=float)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12 or not np.isfinite(norm):
        return start[:2].copy()
    end = start[:2] + v / norm * length
    ax.annotate(
        "",
        xy=end,
        xytext=start[:2],
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, alpha=alpha),
        zorder=5,
    )
    return end


def plot_variant_2d_charts(payload: TangentPayload, emb: Embedding, out_dir: Path) -> Path:
    coords = emb.scores[:, :2]
    bx2 = _project_vectors(payload.bx, emb)[:, :2]
    by2 = _project_vectors(payload.by, emb)[:, :2]
    idx = _farthest_subset(coords, 28, seed=11)
    arrow_len = float(max(np.std(coords[:, 0]), np.std(coords[:, 1])) * 0.22)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(coords[:, 0], coords[:, 1], s=13, color=GRAY, alpha=0.42, linewidths=0, zorder=1)
    order = idx[np.argsort(coords[idx, 1])]
    for i in order:
        start = coords[i]
        bx_end = _draw_arrow_2d(ax, start, bx2[i], BLUE, arrow_len, alpha=0.86, lw=1.25)
        by_end = _draw_arrow_2d(ax, start, by2[i], PURPLE, arrow_len, alpha=0.86, lw=1.25)
        ax.plot(
            [bx_end[0], by_end[0]],
            [bx_end[1], by_end[1]],
            color="0.70",
            lw=0.6,
            alpha=0.36,
            zorder=3,
        )
    ax.scatter(coords[idx, 0], coords[idx, 1], s=18, facecolor="white", edgecolor="0.42", lw=0.55, zorder=6)
    ax.set_title("Response PCA local charts", loc="left", fontweight="bold")
    ax.set_xlabel(f"Response PC1 ({emb.variance_explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"Response PC2 ({emb.variance_explained[1] * 100:.1f}% var.)")
    handles = [
        Line2D([0], [0], color=BLUE, lw=1.5, label=r"$b_x(I)$ horizontal"),
        Line2D([0], [0], color=PURPLE, lw=1.5, label=r"$b_y(I)$ vertical"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", handlelength=1.4)
    cos_xy = _cosine(payload.bx, payload.by)
    x, y, ha, va = _choose_label_position(coords[idx])
    inset = ax.inset_axes([0.66 if x < 0.5 else 0.06, 0.08 if y > 0.5 else 0.67, 0.28, 0.23])
    inset.hist(cos_xy, bins=np.linspace(-1, 1, 17), color="0.72", edgecolor="white", lw=0.35)
    inset.axvline(float(np.median(cos_xy)), color=TEXT, lw=0.9)
    inset.axvline(0.0, color="0.45", lw=0.6, ls=":")
    inset.set_xlim(-1, 1)
    inset.set_yticks([])
    inset.set_xticks([-1, 0, 1])
    inset.set_title(r"$\cos(b_x,b_y)$", fontsize=6.0, pad=1.0)
    inset.tick_params(labelsize=5.8, length=2, pad=1)
    for spine in ["top", "right", "left"]:
        inset.spines[spine].set_visible(False)
    ax.text(
        x,
        y,
        f"{len(idx)} well-spaced sampled histories",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=6.4,
        color="0.45",
    )
    _set_equal_2d(ax, coords)
    _clean_axes(ax, grid=True)
    return _save_figure(fig, out_dir, "panelB_variant_2d_charts")[0]


def _draw_quiver3d(
    ax: plt.Axes,
    start: np.ndarray,
    vec: np.ndarray,
    color: str,
    length: float,
    alpha: float = 0.9,
    lw: float = 1.0,
) -> np.ndarray:
    v = np.asarray(vec[:3], dtype=float)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12 or not np.isfinite(norm):
        return start[:3].copy()
    u = v / norm * length
    ax.quiver(
        start[0],
        start[1],
        start[2],
        u[0],
        u[1],
        u[2],
        color=color,
        arrow_length_ratio=0.28,
        lw=lw,
        alpha=alpha,
        zorder=5,
    )
    return start[:3] + u


def plot_variant_3d_charts(
    payload: TangentPayload,
    emb: Embedding,
    out_dir: Path,
    angles: Iterable[tuple[float, float, str]],
) -> list[Path]:
    coords = emb.scores[:, :3]
    bx3 = _project_vectors(payload.bx, emb)[:, :3]
    by3 = _project_vectors(payload.by, emb)[:, :3]
    idx = _farthest_subset(coords[:, :2], 36, seed=13)
    length = float(np.median(np.std(coords, axis=0)) * 0.34)
    paths: list[Path] = []
    for elev, azim, name in angles:
        fig = plt.figure(figsize=(4.6, 4.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=9, c=GRAY, alpha=0.24, depthshade=False)
        planes = []
        for i in idx:
            ux = _unit_rows(bx3[[i]])[0] * length * 0.72
            uy = _unit_rows(by3[[i]])[0] * length * 0.72
            c = coords[i]
            planes.append([c - ux - uy, c + ux - uy, c + ux + uy, c - ux + uy])
        poly = Poly3DCollection(
            planes,
            facecolors=(0.63, 0.65, 0.70, 0.12),
            edgecolors=(0.40, 0.40, 0.45, 0.20),
            linewidths=0.35,
            zorder=2,
        )
        ax.add_collection3d(poly)
        for i in idx:
            _draw_quiver3d(ax, coords[i], bx3[i], BLUE, length, alpha=0.82, lw=0.9)
            _draw_quiver3d(ax, coords[i], by3[i], PURPLE, length, alpha=0.82, lw=0.9)
        ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2], s=13, c="white", edgecolor="0.40", lw=0.35, depthshade=False)
        ax.set_title("3D response PCA; local shift charts at sampled histories", loc="left", fontweight="bold")
        ax.set_xlabel(f"PC1 ({emb.variance_explained[0] * 100:.1f}%)", labelpad=-1)
        ax.set_ylabel(f"PC2 ({emb.variance_explained[1] * 100:.1f}%)", labelpad=-1)
        ax.set_zlabel(f"PC3 ({emb.variance_explained[2] * 100:.1f}%)", labelpad=-1)
        ax.view_init(elev=elev, azim=azim)
        _set_equal_3d(ax, coords)
        ax.grid(True, alpha=0.18)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        paths.append(_save_figure(fig, out_dir, f"panelB_variant_3d_local_charts_{name}")[0])
    return paths


def _endpoint_coords(payload: TangentPayload, emb: Embedding) -> dict[str, np.ndarray] | None:
    if any(v is None for v in [payload.rx_p, payload.rx_m, payload.ry_p, payload.ry_m]):
        return None
    assert payload.rx_p is not None and payload.rx_m is not None and payload.ry_p is not None and payload.ry_m is not None
    return {
        "rx_p": _project_points(payload.rx_p, emb),
        "rx_m": _project_points(payload.rx_m, emb),
        "ry_p": _project_points(payload.ry_p, emb),
        "ry_m": _project_points(payload.ry_m, emb),
    }


def _scaled_endpoint_coords(
    payload: TangentPayload,
    emb: Embedding,
    target_fraction: float = 0.28,
    max_gain: float = 4.0,
) -> tuple[dict[str, np.ndarray], float] | tuple[None, float]:
    endpoints = _endpoint_coords(payload, emb)
    if endpoints is None:
        return None, 1.0
    coords = emb.scores[:, :3]
    center = coords
    disp = [
        endpoints["rx_p"][:, :3] - center,
        endpoints["rx_m"][:, :3] - center,
        endpoints["ry_p"][:, :3] - center,
        endpoints["ry_m"][:, :3] - center,
    ]
    radii = np.concatenate([np.linalg.norm(d[:, :2], axis=1) for d in disp])
    radii = radii[np.isfinite(radii) & (radii > 1e-12)]
    target = float(np.median(np.std(coords[:, :2], axis=0)) * target_fraction)
    gain = float(min(max_gain, target / float(np.median(radii)))) if radii.size and target > 0 else 1.0
    scaled = {name: center + (value[:, :3] - center) * gain for name, value in endpoints.items()}
    return scaled, max(1.0, gain)


def plot_variant_ribbon_2d(payload: TangentPayload, emb: Embedding, out_dir: Path) -> Path:
    coords = emb.scores[:, :2]
    endpoints3, endpoint_gain = _scaled_endpoint_coords(payload, emb)
    bx2 = _project_vectors(payload.bx, emb)[:, :2]
    by2 = _project_vectors(payload.by, emb)[:, :2]
    idx = _farthest_subset(coords, 42, seed=23)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(coords[:, 0], coords[:, 1], s=10, color=GRAY, alpha=0.28, linewidths=0, zorder=1)
    if endpoints3 is not None:
        for i in idx:
            verts = np.vstack(
                [
                    endpoints3["rx_p"][i, :2],
                    endpoints3["ry_p"][i, :2],
                    endpoints3["rx_m"][i, :2],
                    endpoints3["ry_m"][i, :2],
                ]
            )
            ax.add_patch(
                Polygon(
                    verts,
                    closed=True,
                    facecolor=(0.50, 0.48, 0.60, 0.18),
                    edgecolor=(0.35, 0.35, 0.42, 0.42),
                    lw=0.62,
                    zorder=2,
                )
            )
            ax.plot([endpoints3["rx_m"][i, 0], endpoints3["rx_p"][i, 0]], [endpoints3["rx_m"][i, 1], endpoints3["rx_p"][i, 1]], color=BLUE, lw=0.85, alpha=0.68, zorder=3)
            ax.plot([endpoints3["ry_m"][i, 0], endpoints3["ry_p"][i, 0]], [endpoints3["ry_m"][i, 1], endpoints3["ry_p"][i, 1]], color=PURPLE, lw=0.85, alpha=0.68, zorder=3)
            _draw_arrow_2d(ax, coords[i], endpoints3["rx_p"][i, :2] - coords[i], BLUE, np.linalg.norm(endpoints3["rx_p"][i, :2] - coords[i]), alpha=0.70, lw=0.75)
            _draw_arrow_2d(ax, coords[i], endpoints3["ry_p"][i, :2] - coords[i], PURPLE, np.linalg.norm(endpoints3["ry_p"][i, :2] - coords[i]), alpha=0.70, lw=0.75)
    else:
        length = float(max(np.std(coords[:, 0]), np.std(coords[:, 1])) * 0.13)
        for i in idx:
            ux = _unit_rows(bx2[[i]])[0] * length
            uy = _unit_rows(by2[[i]])[0] * length
            verts = np.vstack([coords[i] + ux, coords[i] + uy, coords[i] - ux, coords[i] - uy])
            ax.add_patch(Polygon(verts, closed=True, facecolor=(0.50, 0.48, 0.60, 0.13), edgecolor=(0.35, 0.35, 0.42, 0.32), lw=0.55, zorder=2))
    ax.scatter(coords[idx, 0], coords[idx, 1], s=13, facecolor="white", edgecolor="0.38", lw=0.4, zorder=5)
    ax.set_title("Finite-shift local patch bundle", loc="left", fontweight="bold")
    ax.set_xlabel(f"Response PC1 ({emb.variance_explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"Response PC2 ({emb.variance_explained[1] * 100:.1f}% var.)")
    ax.text(
        0.04,
        0.95,
        f"diamonds use saved +/- {payload.delta:g} arcmin endpoint directions; {endpoint_gain:.1f}x visual scale",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="0.43",
    )
    _set_equal_2d(ax, coords)
    _clean_axes(ax, grid=True)
    return _save_figure(fig, out_dir, "panelB_variant_ribbon_2d")[0]


def plot_variant_local_chart_atlas_2d(payload: TangentPayload, emb: Embedding, out_dir: Path) -> Path:
    """A more legible atlas version of the ribbon idea.

    The goal is not density. It is to make clear that each selected response
    state carries its own finite local chart: center r0, blue +/-x endpoint
    axis, purple +/-y endpoint axis, and the small diamond spanned by those
    four actual finite-shift endpoints after one global visual magnification.
    """
    coords = emb.scores[:, :2]
    endpoints3, endpoint_gain = _scaled_endpoint_coords(payload, emb, target_fraction=0.42, max_gain=5.5)
    bx2 = _project_vectors(payload.bx, emb)[:, :2]
    by2 = _project_vectors(payload.by, emb)[:, :2]
    idx = _farthest_subset(coords, 18, seed=41)

    fig, ax = plt.subplots(figsize=(4.6, 4.25))
    ax.scatter(coords[:, 0], coords[:, 1], s=14, color="0.84", alpha=0.54, linewidths=0, zorder=1)

    if endpoints3 is not None:
        order = idx[np.argsort(coords[idx, 1])]
        for i in order:
            c = coords[i]
            xp = endpoints3["rx_p"][i, :2]
            xm = endpoints3["rx_m"][i, :2]
            yp = endpoints3["ry_p"][i, :2]
            ym = endpoints3["ry_m"][i, :2]
            verts = np.vstack([xp, yp, xm, ym])
            ax.add_patch(
                Polygon(
                    verts,
                    closed=True,
                    facecolor=(0.50, 0.48, 0.60, 0.18),
                    edgecolor=(0.30, 0.30, 0.36, 0.42),
                    lw=0.75,
                    zorder=2,
                )
            )
            ax.plot([xm[0], xp[0]], [xm[1], xp[1]], color=BLUE, lw=1.25, alpha=0.86, zorder=4)
            ax.plot([ym[0], yp[0]], [ym[1], yp[1]], color=PURPLE, lw=1.25, alpha=0.86, zorder=4)
            ax.scatter([xp[0], xm[0]], [xp[1], xm[1]], s=10, color=BLUE, alpha=0.75, linewidths=0, zorder=5)
            ax.scatter([yp[0], ym[0]], [yp[1], ym[1]], s=10, color=PURPLE, alpha=0.75, linewidths=0, zorder=5)
            ax.scatter(c[0], c[1], s=16, facecolor="white", edgecolor="0.25", lw=0.55, zorder=6)
    else:
        length = float(max(np.std(coords[:, 0]), np.std(coords[:, 1])) * 0.24)
        for i in idx:
            c = coords[i]
            ux = _unit_rows(bx2[[i]])[0] * length
            uy = _unit_rows(by2[[i]])[0] * length
            verts = np.vstack([c + ux, c + uy, c - ux, c - uy])
            ax.add_patch(Polygon(verts, closed=True, facecolor=(0.50, 0.48, 0.60, 0.18), edgecolor=(0.30, 0.30, 0.36, 0.42), lw=0.75, zorder=2))
            ax.plot([c[0] - ux[0], c[0] + ux[0]], [c[1] - ux[1], c[1] + ux[1]], color=BLUE, lw=1.25, alpha=0.86, zorder=4)
            ax.plot([c[0] - uy[0], c[0] + uy[0]], [c[1] - uy[1], c[1] + uy[1]], color=PURPLE, lw=1.25, alpha=0.86, zorder=4)
            ax.scatter(c[0], c[1], s=16, facecolor="white", edgecolor="0.25", lw=0.55, zorder=6)

    # Mini key drawn in axis coordinates so the chart glyph has an unambiguous read.
    key = ax.inset_axes([0.62, 0.05, 0.33, 0.23])
    key.set_xlim(-1, 1)
    key.set_ylim(-1, 1)
    key.add_patch(Polygon(np.asarray([[0.85, 0], [0, 0.55], [-0.85, 0], [0, -0.55]]), closed=True, facecolor=(0.50, 0.48, 0.60, 0.18), edgecolor="0.45", lw=0.7))
    key.plot([-0.85, 0.85], [0, 0], color=BLUE, lw=1.4)
    key.plot([0, 0], [-0.55, 0.55], color=PURPLE, lw=1.4)
    key.scatter([0], [0], s=18, facecolor="white", edgecolor="0.25", lw=0.6, zorder=4)
    key.text(0, -0.86, "one local chart", ha="center", va="top", fontsize=6.3, color="0.30")
    key.set_xticks([])
    key.set_yticks([])
    for spine in key.spines.values():
        spine.set_visible(False)
    key.patch.set_alpha(0.78)

    ax.set_title("Local chart atlas over compact response states", loc="left", fontweight="bold")
    ax.set_xlabel(f"Response PC1 ({emb.variance_explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"Response PC2 ({emb.variance_explained[1] * 100:.1f}% var.)")
    ax.text(
        0.04,
        0.96,
        f"{len(idx)} selected sampled histories; endpoint directions, {endpoint_gain:.1f}x visual scale",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="0.40",
    )
    ax.text(
        0.04,
        0.06,
        "gray dots: all baseline r0 states",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="0.48",
    )
    _set_equal_2d(ax, coords)
    _clean_axes(ax, grid=True)
    return _save_figure(fig, out_dir, "panelB_variant_local_chart_atlas_2d")[0]


def plot_variant_direction_normalized_atlas_2d(payload: TangentPayload, emb: Embedding, out_dir: Path) -> Path:
    """Readable local chart atlas where each glyph has fixed display size."""
    coords = emb.scores[:, :2]
    endpoints = _endpoint_coords(payload, emb)
    bx2 = _project_vectors(payload.bx, emb)[:, :2]
    by2 = _project_vectors(payload.by, emb)[:, :2]
    idx = _farthest_subset(coords, 14, seed=43)
    half_len = float(np.median(np.std(coords[:, :2], axis=0)) * 0.26)

    fig, ax = plt.subplots(figsize=(4.6, 4.25))
    ax.scatter(coords[:, 0], coords[:, 1], s=14, color="0.84", alpha=0.52, linewidths=0, zorder=1)

    for i in idx:
        c = coords[i]
        if endpoints is not None:
            x_dir = endpoints["rx_p"][i, :2] - endpoints["rx_m"][i, :2]
            y_dir = endpoints["ry_p"][i, :2] - endpoints["ry_m"][i, :2]
        else:
            x_dir = bx2[i]
            y_dir = by2[i]
        x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12) * half_len
        y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-12) * half_len
        xp, xm = c + x_dir, c - x_dir
        yp, ym = c + y_dir, c - y_dir
        verts = np.vstack([xp, yp, xm, ym])
        ax.add_patch(
            Polygon(
                verts,
                closed=True,
                facecolor=(0.50, 0.48, 0.60, 0.16),
                edgecolor=(0.30, 0.30, 0.36, 0.36),
                lw=0.65,
                zorder=2,
            )
        )
        ax.plot([xm[0], xp[0]], [xm[1], xp[1]], color=BLUE, lw=1.35, alpha=0.92, zorder=4)
        ax.plot([ym[0], yp[0]], [ym[1], yp[1]], color=PURPLE, lw=1.35, alpha=0.92, zorder=4)
        ax.scatter(c[0], c[1], s=18, facecolor="white", edgecolor="0.24", lw=0.6, zorder=6)

    key = ax.inset_axes([0.62, 0.05, 0.33, 0.23])
    key.set_xlim(-1, 1)
    key.set_ylim(-1, 1)
    key.add_patch(Polygon(np.asarray([[0.85, 0], [0, 0.55], [-0.85, 0], [0, -0.55]]), closed=True, facecolor=(0.50, 0.48, 0.60, 0.16), edgecolor="0.45", lw=0.7))
    key.plot([-0.85, 0.85], [0, 0], color=BLUE, lw=1.4)
    key.plot([0, 0], [-0.55, 0.55], color=PURPLE, lw=1.4)
    key.scatter([0], [0], s=18, facecolor="white", edgecolor="0.25", lw=0.6, zorder=4)
    key.text(0, -0.86, "one local chart", ha="center", va="top", fontsize=6.3, color="0.30")
    key.set_xticks([])
    key.set_yticks([])
    for spine in key.spines.values():
        spine.set_visible(False)
    key.patch.set_facecolor("white")
    key.patch.set_alpha(0.90)

    ax.set_title("Direction-normalized local chart atlas", loc="left", fontweight="bold")
    ax.set_xlabel(f"Response PC1 ({emb.variance_explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"Response PC2 ({emb.variance_explained[1] * 100:.1f}% var.)")
    ax.text(
        0.04,
        0.96,
        f"{len(idx)} sampled histories; glyph lengths normalized for readability",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="0.40",
    )
    ax.text(
        0.04,
        0.06,
        "gray dots: all baseline r0 states",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="0.48",
    )
    _set_equal_2d(ax, coords)
    _clean_axes(ax, grid=True)
    return _save_figure(fig, out_dir, "panelB_variant_direction_normalized_atlas_2d")[0]


def plot_variant_ribbon_3d(
    payload: TangentPayload,
    emb: Embedding,
    out_dir: Path,
    angles: Iterable[tuple[float, float, str]],
) -> list[Path]:
    coords = emb.scores[:, :3]
    endpoints, endpoint_gain = _scaled_endpoint_coords(payload, emb)
    bx3 = _project_vectors(payload.bx, emb)[:, :3]
    by3 = _project_vectors(payload.by, emb)[:, :3]
    idx = _farthest_subset(coords[:, :2], 40, seed=29)
    length = float(np.median(np.std(coords, axis=0)) * 0.22)
    paths: list[Path] = []
    for elev, azim, name in angles:
        fig = plt.figure(figsize=(4.6, 4.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=8, c=GRAY, alpha=0.22, depthshade=False)
        polys = []
        if endpoints is not None:
            for i in idx:
                polys.append(
                    [
                        endpoints["rx_p"][i, :3],
                        endpoints["ry_p"][i, :3],
                        endpoints["rx_m"][i, :3],
                        endpoints["ry_m"][i, :3],
                    ]
                )
        poly = Poly3DCollection(
            polys,
            facecolors=(0.50, 0.48, 0.60, 0.14),
            edgecolors=(0.35, 0.35, 0.42, 0.34),
            linewidths=0.45,
        )
        ax.add_collection3d(poly)
        for i in idx:
            _draw_quiver3d(ax, coords[i], bx3[i], BLUE, length, alpha=0.62, lw=0.75)
            _draw_quiver3d(ax, coords[i], by3[i], PURPLE, length, alpha=0.62, lw=0.75)
        ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2], s=12, c="white", edgecolor="0.38", lw=0.35, depthshade=False)
        ax.set_title("3D finite-shift patch bundle", loc="left", fontweight="bold")
        ax.set_xlabel(f"PC1 ({emb.variance_explained[0] * 100:.1f}%)", labelpad=-1)
        ax.set_ylabel(f"PC2 ({emb.variance_explained[1] * 100:.1f}%)", labelpad=-1)
        ax.set_zlabel(f"PC3 ({emb.variance_explained[2] * 100:.1f}%)", labelpad=-1)
        ax.view_init(elev=elev, azim=azim)
        _set_equal_3d(ax, coords)
        ax.grid(True, alpha=0.18)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.text2D(
            0.03,
            0.94,
            f"endpoint directions, {endpoint_gain:.1f}x visual scale",
            transform=ax.transAxes,
            fontsize=6.2,
            color="0.42",
        )
        paths.append(_save_figure(fig, out_dir, f"panelB_variant_ribbon_3d_{name}")[0])
    return paths


def plot_variant_landscape(payload: TangentPayload, emb: Embedding, out_dir: Path) -> Path | None:
    try:
        from scipy.interpolate import griddata
    except Exception:
        return None

    coords = emb.scores[:, :3]
    idx = _farthest_subset(coords[:, :2], 30, seed=31)
    bx3 = _project_vectors(payload.bx, emb)[:, :3]
    by3 = _project_vectors(payload.by, emb)[:, :3]
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    gx, gy = np.meshgrid(
        np.linspace(np.percentile(x, 3), np.percentile(x, 97), 45),
        np.linspace(np.percentile(y, 3), np.percentile(y, 97), 45),
    )
    # Visual guide through sampled baseline response states only; not a
    # generative image manifold or simulated response for interpolated images.
    gz = griddata((x, y), z, (gx, gy), method="linear")
    if not np.isfinite(gz).any():
        return None

    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(gx, gy, gz, color=SURFACE_GRAY, alpha=0.20, linewidth=0, antialiased=True, shade=False)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=9, c="0.55", alpha=0.34, depthshade=False)
    length = float(np.median(np.std(coords, axis=0)) * 0.30)
    for i in idx:
        _draw_quiver3d(ax, coords[i], bx3[i], BLUE, length, alpha=0.82, lw=0.9)
        _draw_quiver3d(ax, coords[i], by3[i], PURPLE, length, alpha=0.82, lw=0.9)
    ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2], s=12, c="white", edgecolor="0.38", lw=0.35, depthshade=False)
    ax.set_title("Response landscape guide through sampled states", loc="left", fontweight="bold")
    ax.set_xlabel(f"PC1 ({emb.variance_explained[0] * 100:.1f}%)", labelpad=-1)
    ax.set_ylabel(f"PC2 ({emb.variance_explained[1] * 100:.1f}%)", labelpad=-1)
    ax.set_zlabel(f"PC3 ({emb.variance_explained[2] * 100:.1f}%)", labelpad=-1)
    ax.view_init(elev=27, azim=-54)
    _set_equal_3d(ax, coords)
    ax.grid(True, alpha=0.18)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    return _save_figure(fig, out_dir, "panelB_variant_response_landscape")[0]


def _thumbnail_from_history(history: np.ndarray, size: int = 19) -> np.ndarray:
    arr = np.asarray(history, dtype=float)
    if arr.ndim == 3:
        img = arr[-1]
    elif arr.ndim == 2:
        img = arr
    else:
        img = np.squeeze(arr)
        if img.ndim == 3:
            img = img[-1]
    if img.ndim != 2:
        img = np.zeros((size, size), dtype=float)
    y_idx = np.linspace(0, img.shape[0] - 1, size).astype(int)
    x_idx = np.linspace(0, img.shape[1] - 1, size).astype(int)
    small = img[np.ix_(y_idx, x_idx)]
    lo, hi = np.nanpercentile(small, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(small)), float(np.nanmax(small))
    if hi <= lo:
        return np.full((size, size), 0.5, dtype=float)
    return np.clip((small - lo) / (hi - lo), 0.0, 1.0)


def _add_image_plane(
    ax: plt.Axes,
    image: np.ndarray,
    center: np.ndarray,
    size: float,
    z_lift: float,
    alpha: float = 0.92,
) -> None:
    h, w = image.shape
    xs = center[0] + np.linspace(-0.5 * size, 0.5 * size, w)
    ys = center[1] + np.linspace(-0.5 * size, 0.5 * size, h)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, center[2] + z_lift)
    face = plt.cm.gray(image)
    face[..., 3] = alpha
    ax.plot_surface(
        xx,
        yy,
        zz,
        rstride=1,
        cstride=1,
        facecolors=face,
        shade=False,
        linewidth=0,
        antialiased=False,
        zorder=4,
    )
    border = [
        [xs[0], ys[0], center[2] + z_lift],
        [xs[-1], ys[0], center[2] + z_lift],
        [xs[-1], ys[-1], center[2] + z_lift],
        [xs[0], ys[-1], center[2] + z_lift],
    ]
    ax.add_collection3d(
        Poly3DCollection(
            [border],
            facecolors=(1, 1, 1, 0),
            edgecolors=(0.25, 0.25, 0.25, 0.50),
            linewidths=0.45,
            zorder=5,
        )
    )


def _tangent_plane_disk_vertices(center: np.ndarray, bx: np.ndarray, by: np.ndarray, radius: float) -> np.ndarray:
    ux = np.asarray(bx[:3], dtype=float)
    uy = np.asarray(by[:3], dtype=float)
    ux = ux / (np.linalg.norm(ux) + 1e-12)
    uy = uy - ux * float(np.dot(ux, uy))
    uy = uy / (np.linalg.norm(uy) + 1e-12)
    if not np.isfinite(uy).all() or np.linalg.norm(uy) < 1e-6:
        uy = np.asarray(by[:3], dtype=float)
        uy = uy / (np.linalg.norm(uy) + 1e-12)
    theta = np.linspace(0, 2 * np.pi, 26, endpoint=False)
    return center[None, :] + radius * (np.cos(theta)[:, None] * ux[None, :] + np.sin(theta)[:, None] * uy[None, :])


def _tangent_plane_square_vertices(center: np.ndarray, bx: np.ndarray, by: np.ndarray, radius: float) -> np.ndarray:
    ux = np.asarray(bx[:3], dtype=float)
    uy = np.asarray(by[:3], dtype=float)
    ux = ux / (np.linalg.norm(ux) + 1e-12)
    uy = uy - ux * float(np.dot(ux, uy))
    uy = uy / (np.linalg.norm(uy) + 1e-12)
    if not np.isfinite(uy).all() or np.linalg.norm(uy) < 1e-6:
        uy = np.asarray(by[:3], dtype=float)
        uy = uy / (np.linalg.norm(uy) + 1e-12)
    return np.vstack([center + ux + uy, center - ux + uy, center - ux - uy, center + ux - uy]) * radius + center * (1.0 - radius)


def _interior_farthest_subset(coords_2d: np.ndarray, n_show: int, seed: int = 0) -> np.ndarray:
    coords_2d = np.asarray(coords_2d, dtype=float)
    margins = [16, 12, 8, 5]
    selected_pool = np.arange(coords_2d.shape[0])
    for margin in margins:
        lo = np.nanpercentile(coords_2d, margin, axis=0)
        hi = np.nanpercentile(coords_2d, 100 - margin, axis=0)
        mask = np.all((coords_2d >= lo) & (coords_2d <= hi), axis=1)
        pool = np.where(mask)[0]
        if pool.size >= max(6, min(n_show, coords_2d.shape[0]) // 2):
            selected_pool = pool
            break
    local = _farthest_subset(coords_2d[selected_pool], min(n_show, selected_pool.size), seed=seed)
    return selected_pool[local]


def _interior_point_mask(coords_2d: np.ndarray, margin: float = 10.0) -> np.ndarray:
    coords_2d = np.asarray(coords_2d, dtype=float)
    lo = np.nanpercentile(coords_2d, margin, axis=0)
    hi = np.nanpercentile(coords_2d, 100.0 - margin, axis=0)
    mask = np.all((coords_2d >= lo) & (coords_2d <= hi), axis=1)
    if int(np.sum(mask)) < max(8, coords_2d.shape[0] // 3):
        lo = np.nanpercentile(coords_2d, margin * 0.6, axis=0)
        hi = np.nanpercentile(coords_2d, 100.0 - margin * 0.6, axis=0)
        mask = np.all((coords_2d >= lo) & (coords_2d <= hi), axis=1)
    return mask


def plot_variant_surface_images_charts(payload: TangentPayload, emb: Embedding, out_dir: Path) -> list[Path] | None:
    try:
        from scipy.interpolate import RBFInterpolator, griddata
        from scipy.ndimage import binary_erosion, gaussian_filter
    except Exception:
        return None

    coords = emb.scores[:, :3]
    bx3 = _project_vectors(payload.bx, emb)[:, :3]
    by3 = _project_vectors(payload.by, emb)[:, :3]
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    pc1_min = float(np.percentile(x, 5))
    pc1_max = float(np.percentile(x, 95))
    pc2_min = float(np.percentile(y, 5))
    pc2_max = float(np.percentile(y, 95))
    gx, gy = np.meshgrid(
        np.linspace(pc1_min, pc1_max, 75),
        np.linspace(pc2_min, pc2_max, 75),
    )
    # Visual guide through sampled baseline response states only; not a
    # generative image manifold or simulated response for interpolated images.
    xy = np.column_stack([x, y])
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])
    point_z = z.copy()
    try:
        rbf = RBFInterpolator(
            xy,
            z,
            kernel="thin_plate_spline",
            smoothing=0.85,
            degree=1,
        )
        gz = rbf(grid_xy).reshape(gx.shape)
        point_z = rbf(xy)
    except Exception:
        gz = griddata((x, y), z, (gx, gy), method="linear")
    if not np.isfinite(gz).any():
        return None
    hull_mask = np.isfinite(griddata((x, y), z, (gx, gy), method="linear"))
    support_mask = binary_erosion(hull_mask, iterations=3, border_value=0)
    if int(np.sum(support_mask)) < int(0.45 * np.sum(hull_mask)):
        support_mask = binary_erosion(hull_mask, iterations=2, border_value=0)
    if int(np.sum(support_mask)) == 0:
        support_mask = hull_mask
    finite_mask = np.isfinite(gz)
    nearest = griddata((x, y), z, (gx, gy), method="nearest")
    gz_filled = np.where(finite_mask, gz, nearest)
    gz_smooth = gaussian_filter(gz_filled, sigma=1.35)
    gz = np.where(support_mask, gz_smooth, np.nan)
    surface_coords = coords.copy()
    surface_coords[:, 2] = np.asarray(point_z, dtype=float)

    visible_mask = (
        (coords[:, 0] >= pc1_min)
        & (coords[:, 0] <= pc1_max)
        & (coords[:, 1] >= pc2_min)
        & (coords[:, 1] <= pc2_max)
    )
    visible_idx = np.where(visible_mask)[0]
    if visible_idx.size < 8:
        visible_idx = np.arange(coords.shape[0])
    chart_idx = visible_idx[_farthest_subset(coords[visible_idx, :2], min(11, visible_idx.size), seed=61)]
    plane_radius = float(np.median(np.std(coords[:, :2], axis=0)) * 0.155)
    planes = [_tangent_plane_square_vertices(surface_coords[i], bx3[i], by3[i], plane_radius) for i in chart_idx]
    arrow_len = plane_radius * 1.78
    view_angles = [
        (34, -70, "angle1"),
        (26, -42, "angle2"),
        (42, -98, "angle3"),
        (18, -76, "angle4"),
        (54, -58, "angle5"),
        (32, -128, "angle6"),
    ]
    paths: list[Path] = []
    for elev, azim, name in view_angles:
        fig = plt.figure(figsize=(5.8, 4.65))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            gx,
            gy,
            gz,
            color=SURFACE_GRAY,
            alpha=0.055,
            linewidth=0,
            antialiased=True,
            shade=False,
            zorder=1,
        )
        ax.plot_wireframe(
            gx,
            gy,
            gz,
            rstride=5,
            cstride=5,
            color=(0.48, 0.51, 0.56, 0.36),
            linewidth=0.55,
            zorder=2,
        )
        ax.scatter(surface_coords[visible_idx, 0], surface_coords[visible_idx, 1], surface_coords[visible_idx, 2], s=12, c="0.50", alpha=0.32, depthshade=False, zorder=3)
        ax.add_collection3d(
            Poly3DCollection(
                planes,
                facecolors=(0.50, 0.48, 0.60, 0.15),
                edgecolors=(0.30, 0.30, 0.36, 0.0),
                linewidths=0.0,
                zorder=4,
            )
        )
        for i in chart_idx:
            _draw_quiver3d(ax, surface_coords[i], bx3[i], BLUE, arrow_len, alpha=0.90, lw=1.05)
            _draw_quiver3d(ax, surface_coords[i], by3[i], PURPLE, arrow_len, alpha=0.90, lw=1.05)
        ax.scatter(surface_coords[chart_idx, 0], surface_coords[chart_idx, 1], surface_coords[chart_idx, 2], s=17, c="white", edgecolor="0.30", lw=0.45, depthshade=False, zorder=6)

        ax.set_title("Response-state surface with local tangent planes", loc="left", fontweight="bold", pad=4)
        ax.set_xlabel(f"PC1 ({emb.variance_explained[0] * 100:.1f}%)", labelpad=-2)
        ax.set_ylabel(f"PC2 ({emb.variance_explained[1] * 100:.1f}%)", labelpad=-2)
        ax.set_zlabel(f"PC3 ({emb.variance_explained[2] * 100:.1f}%)", labelpad=-2)
        ax.text2D(
            0.02,
            0.93,
            "surface: trimmed smoothed guide through supported r0 states; dots/charts are data-anchored",
            transform=ax.transAxes,
            fontsize=6.6,
            color="0.35",
        )
        ax.view_init(elev=elev, azim=azim)
        _focus_3d_on_surface(ax, surface_coords)
        _soften_3d_grid(ax)
        fig.subplots_adjust(left=-0.03, right=1.03, bottom=-0.03, top=0.92)
        paths.append(_save_figure(fig, out_dir, f"panelB_variant_surface_planes_charts_{name}")[0])

        fig = plt.figure(figsize=(5.8, 4.65))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            gx,
            gy,
            gz,
            color=SURFACE_GRAY,
            alpha=0.055,
            linewidth=0,
            antialiased=True,
            shade=False,
            zorder=1,
        )
        ax.plot_wireframe(
            gx,
            gy,
            gz,
            rstride=5,
            cstride=5,
            color=(0.48, 0.51, 0.56, 0.36),
            linewidth=0.55,
            zorder=2,
        )
        ax.scatter(surface_coords[visible_idx, 0], surface_coords[visible_idx, 1], surface_coords[visible_idx, 2], s=12, c="0.50", alpha=0.32, depthshade=False, zorder=3)
        ax.add_collection3d(
            Poly3DCollection(
                planes,
                facecolors=(0.50, 0.48, 0.60, 0.15),
                edgecolors=(0.30, 0.30, 0.36, 0.0),
                linewidths=0.0,
                zorder=4,
            )
        )
        for i in chart_idx:
            _draw_quiver3d(ax, surface_coords[i], bx3[i], BLUE, arrow_len, alpha=0.90, lw=1.05)
            _draw_quiver3d(ax, surface_coords[i], by3[i], PURPLE, arrow_len, alpha=0.90, lw=1.05)
        ax.scatter(surface_coords[chart_idx, 0], surface_coords[chart_idx, 1], surface_coords[chart_idx, 2], s=17, c="white", edgecolor="0.30", lw=0.45, depthshade=False, zorder=6)
        ax.set_title("Response-state surface with local tangent planes", loc="left", fontweight="bold", pad=4)
        ax.text2D(
            0.02,
            0.93,
            "surface: trimmed smoothed guide through supported r0 states; dots/charts are data-anchored",
            transform=ax.transAxes,
            fontsize=6.6,
            color="0.35",
        )
        ax.view_init(elev=elev, azim=azim)
        _focus_3d_on_surface(ax, surface_coords)
        _hide_3d_axes(ax)
        _add_pc_triad(ax, surface_coords, emb)
        fig.subplots_adjust(left=-0.03, right=1.03, bottom=-0.03, top=0.92)
        paths.append(_save_figure(fig, out_dir, f"panelB_variant_surface_planes_charts_{name}_triad")[0])
    return paths


def make_contact_sheet(image_paths: list[Path], out_dir: Path) -> Path:
    n = len(image_paths)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0))
    flat = np.asarray(axes).ravel()
    for ax, path in zip(flat, image_paths):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(path.stem.replace("panelB_variant_", "").replace("_", " "), fontsize=8, pad=2)
        ax.axis("off")
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle("Panel B visualization prototypes", fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(pad=0.8)
    return _save_figure(fig, out_dir, "panelB_variants_contact_sheet", dpi=220)[0]


def write_note(
    out_dir: Path,
    audit: dict[str, float | int | str],
    manifold_audit: dict[str, float | int],
    landscape_path: Path | None,
) -> Path:
    smooth_delta = float(manifold_audit["median_knn_minus_random_tangent_plane_similarity"])
    trust = float(manifold_audit["trustworthiness_pc1_3"])
    if trust >= 0.85 and smooth_delta >= 0.05:
        verdict = "The response-landscape idea has measurable support, but the finite-shift patch bundle is still the safest main-panel candidate."
    else:
        verdict = "The clearest first candidate is now the direction-normalized 2D local-chart atlas, with finite-endpoint and dense ribbon versions kept as secondary options."
    lines = [
        "# Panel B Prototype Notes",
        "",
        verdict,
        "",
        "Why:",
        "- It makes each selected sampled stimulus-history object visibly own a local chart.",
        "- The patches point in different orientations, so it does not imply one universal signed x/y population axis.",
        "- The bundle still sits inside a compact response PCA cloud, which tees up Panels C-D.",
        "- It uses saved finite-shift endpoint responses, so it carries the lowest overclaiming risk.",
        "- The direction-normalized atlas sacrifices local magnitude in the glyphs, but it makes the intended read of image-specific chart orientation much clearer.",
        f"- The surface-support audit gives PC1-3 trustworthiness={trust:.2f} and kNN-minus-random tangent-plane similarity={smooth_delta:.2f}.",
        "",
        "Other variants:",
        "- Variant 1 is the cleanest direct replacement if the final panel must remain strictly arrow-based.",
        "- Variant 2 is visually strongest for the atlas idea, but 3D camera dependence may make it less robust in print.",
        "- The surface+planes option is the loose response-landscape claim: it shows sampled response states on a compact surface with local tangent planes attached to selected points.",
        "- The finite-endpoint atlas preserves more local magnitude information but is harder to read because a few large patches dominate.",
        "- The dense ribbon view is useful as a bundle texture, but it is not the clearest standalone Panel B.",
        "- Variant 4 should remain secondary because the surface is only a visual guide through sampled baseline response states, not interpolated image responses.",
        "",
        "Audit:",
        f"- Valid local charts: {audit['n_valid']} of {audit['n_loaded']} loaded at delta={audit['delta_arcmin_used']} arcmin.",
        f"- Baseline r0 PC1-3 variance explained: {100 * float(audit['r0_pc1_3_variance_explained']):.1f}%.",
        f"- Median tangent energy visible in PC1-3: bx={100 * float(audit['median_visible_energy_bx_pc1_3']):.1f}%, by={100 * float(audit['median_visible_energy_by_pc1_3']):.1f}%.",
        f"- Median kNN5 overlap full r0 vs PC1-3: {float(audit['median_knn5_overlap_full_vs_pc1_3_r0']):.2f}.",
        f"- Median angle error for cos(bx,by) after PC1-3 projection: {float(audit['median_angle_error_deg_cos_bx_by_full_vs_pc1_3']):.1f} deg.",
        f"- Baseline r0 PC1-5 / PC1-10 variance explained: {100 * float(manifold_audit['r0_pc1_5_variance_explained']):.1f}% / {100 * float(manifold_audit['r0_pc1_10_variance_explained']):.1f}%.",
        f"- Baseline r0 participation ratio: {float(manifold_audit['r0_participation_ratio']):.1f}; unit-shuffle mean PR: {float(manifold_audit['unit_shuffle_participation_ratio_mean']):.1f}.",
        f"- PC1-3 faithfulness: trustworthiness={float(manifold_audit['trustworthiness_pc1_3']):.2f}, continuity={float(manifold_audit['continuity_pc1_3']):.2f}, distance Spearman rho={float(manifold_audit['spearman_pairwise_distance_full_vs_pc1_3']):.2f}.",
        f"- Tangent-plane smoothness: kNN median={float(manifold_audit['median_knn_tangent_plane_similarity']):.2f}, random median={float(manifold_audit['median_random_tangent_plane_similarity']):.2f}, distance-vs-plane rho={float(manifold_audit['spearman_response_distance_vs_tangent_plane_similarity']):.2f}.",
    ]
    if landscape_path is None:
        lines.append("- Response-landscape variant was skipped because scipy interpolation was unavailable or degenerate.")
    path = out_dir / "panelB_variant_recommendation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tangent-maps",
        type=Path,
        default=Path("outputs/twin_feature_tangent_structure_prod_v2/tangent_maps/twin_tangent_maps.pkl"),
        help="Path to twin_tangent_maps.pkl.",
    )
    parser.add_argument("--delta-arcmin", type=float, default=0.25)
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--n-unit-shuffles", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/fig4_cov_TFTS/panelB_visualization_prototypes"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = load_payload(args.tangent_maps, args.delta_arcmin)
    emb = _pca(payload.r0, n_components=3)
    audit = compute_audit(payload, emb, args.out_dir)
    manifold_audit, null_df, neighbor_df, distance_bin_df = compute_response_state_manifold_audit(
        payload,
        emb,
        args.out_dir,
        neighbor_k=args.neighbor_k,
        n_shuffles=args.n_unit_shuffles,
    )

    image_paths: list[Path] = []
    manifold_audit_figure = plot_response_state_manifold_audit(
        payload,
        emb,
        manifold_audit,
        null_df,
        neighbor_df,
        distance_bin_df,
        args.out_dir,
    )
    image_paths.append(plot_variant_2d_charts(payload, emb, args.out_dir))
    chart_angles = [(24, -52, "angle1"), (18, 36, "angle2"), (62, -82, "angle3")]
    image_paths.extend(plot_variant_3d_charts(payload, emb, args.out_dir, chart_angles))
    image_paths.append(plot_variant_direction_normalized_atlas_2d(payload, emb, args.out_dir))
    image_paths.append(plot_variant_local_chart_atlas_2d(payload, emb, args.out_dir))
    image_paths.append(plot_variant_ribbon_2d(payload, emb, args.out_dir))
    ribbon_angles = [(26, -54, "angle1"), (22, 38, "angle2")]
    image_paths.extend(plot_variant_ribbon_3d(payload, emb, args.out_dir, ribbon_angles))
    landscape_path = plot_variant_landscape(payload, emb, args.out_dir)
    if landscape_path is not None:
        image_paths.append(landscape_path)
    surface_plane_paths = plot_variant_surface_images_charts(payload, emb, args.out_dir)
    if surface_plane_paths is not None:
        image_paths.extend(surface_plane_paths)
    contact_sheet = make_contact_sheet(image_paths, args.out_dir)
    note_path = write_note(args.out_dir, audit, manifold_audit, landscape_path)

    manifest = {
        "tangent_maps": str(args.tangent_maps),
        "out_dir": str(args.out_dir),
        "audit_csv": str(args.out_dir / "panelB_embedding_audit.csv"),
        "audit_json": str(args.out_dir / "panelB_embedding_audit.json"),
        "response_state_manifold_audit_csv": str(args.out_dir / "panelB_response_state_manifold_audit.csv"),
        "response_state_manifold_audit_json": str(args.out_dir / "panelB_response_state_manifold_audit.json"),
        "response_state_manifold_audit_figure": str(manifold_audit_figure),
        "response_state_unit_shuffle_null_csv": str(args.out_dir / "panelB_response_state_unit_shuffle_null.csv"),
        "tangent_neighbor_smoothness_csv": str(args.out_dir / "panelB_tangent_neighbor_smoothness.csv"),
        "tangent_smoothness_by_distance_csv": str(args.out_dir / "panelB_tangent_smoothness_by_distance.csv"),
        "contact_sheet": str(contact_sheet),
        "note": str(note_path),
        "variant_pngs": [str(p) for p in image_paths],
    }
    (args.out_dir / "panelB_variant_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
