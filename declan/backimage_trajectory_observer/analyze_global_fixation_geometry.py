from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _finite_vals(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _finite_mean(values: list[float] | np.ndarray) -> float:
    vals = _finite_vals(values)
    return float(np.mean(vals)) if vals.size else float("nan")


def _finite_median(values: list[float] | np.ndarray) -> float:
    vals = _finite_vals(values)
    return float(np.median(vals)) if vals.size else float("nan")


def _finite_ci(values: list[float] | np.ndarray, q: float) -> float:
    vals = _finite_vals(values)
    return float(np.percentile(vals, q)) if vals.size else float("nan")


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _rank_at(frac: np.ndarray, threshold: float) -> int:
    vals = np.asarray(frac, dtype=np.float64)
    if vals.size == 0:
        return 0
    return int(np.searchsorted(np.cumsum(vals), float(threshold), side="left")) + 1


def _participation_ratio(evals: np.ndarray) -> float:
    vals = np.asarray(evals, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    den = float(np.sum(vals * vals))
    if vals.size == 0 or den <= 1e-12:
        return float("nan")
    return float((np.sum(vals) ** 2) / den)


def _spearman(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(keep)) < 3:
        return float("nan")
    x = x[keep]
    y = y[keep]
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= float(np.mean(rx))
    ry -= float(np.mean(ry))
    den = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(np.dot(rx, ry) / den) if den > 0 else float("nan")


def _scalar_int(table: Any, key: str, default: int = -1) -> int:
    if key not in table:
        return int(default)
    arr = np.asarray(table[key]).reshape(-1)
    if arr.size == 0:
        return int(default)
    return int(arr[0])


def _string_array(table: Any, key: str, n: int, prefix: str) -> list[str]:
    if key not in table:
        return [f"{prefix}{i}" for i in range(int(n))]
    arr = np.asarray(table[key]).reshape(-1)
    if arr.size != int(n):
        return [f"{prefix}{i}" for i in range(int(n))]
    return [str(v) for v in arr.tolist()]


def _manifest_filter(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    keep = []
    candidate_modes = set(_parse_list(args.candidate_set_modes))
    prior_families = set(_parse_list(args.prior_families))
    scales = {_safe_float(v) for v in _parse_list(args.scales)}
    axis_modes = set(_parse_list(args.axis_catalog_modes))
    for row in rows:
        if "response_cache_path" not in row or not str(row["response_cache_path"]).strip():
            continue
        if str(row.get("dry_run", "False")).lower() == "true":
            continue
        if candidate_modes and str(row.get("candidate_set_mode", "")) not in candidate_modes:
            continue
        if prior_families and str(row.get("prior_family", "")) not in prior_families:
            continue
        if scales and not any(abs(_safe_float(row.get("scale")) - s) <= 1e-9 for s in scales):
            continue
        if axis_modes and str(row.get("axis_catalog_mode", "")) not in axis_modes:
            continue
        keep.append(row)
    if int(args.max_tables) > 0:
        keep = keep[: int(args.max_tables)]
    return keep


def _load_candidate_feature_map(run_dir: Path) -> dict[str, dict[str, float | str]]:
    rows = _read_csv_rows(run_dir / "selected_windows.csv")
    out: dict[str, dict[str, float | str]] = {}
    for row in rows:
        source = str(row.get("source_row", "")).strip()
        if not source:
            continue
        cid = f"source_row:{source}"
        out[cid] = {
            "session": str(row.get("session", "")),
            "image_index": _safe_float(row.get("image_index")),
            "source_row": _safe_float(source),
            "patch_x": _safe_float(row.get("image_patch_center_x_px")),
            "patch_y": _safe_float(row.get("image_patch_center_y_px")),
            "patch_rms_contrast": _safe_float(row.get("image_patch_rms_contrast")),
            "gradient_energy": _safe_float(row.get("image_gradient_energy")),
            "edge_density": _safe_float(row.get("image_edge_density")),
            "orientation_coherence": _safe_float(row.get("image_orientation_coherence")),
            "dominant_orientation_deg": _safe_float(row.get("image_dominant_orientation_deg")),
        }
    return out


def _reshape_points(
    arr: np.ndarray,
    *,
    point_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return points plus candidate/trajectory/time indices for C,T,K,U arrays."""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 3:
        n_candidate, n_time, n_units = x.shape
        if point_mode == "trajectory_mean":
            points = np.mean(x, axis=1)
            ci = np.arange(n_candidate, dtype=np.int32)
            ti = np.full(n_candidate, -1, dtype=np.int32)
            tt = np.full(n_candidate, -1, dtype=np.int32)
        elif point_mode == "state_timepoints":
            points = x.reshape(n_candidate * n_time, n_units)
            ci = np.repeat(np.arange(n_candidate, dtype=np.int32), n_time)
            ti = np.full(points.shape[0], -1, dtype=np.int32)
            tt = np.tile(np.arange(n_time, dtype=np.int32), n_candidate)
        elif point_mode == "flatten_time":
            points = x.reshape(n_candidate, n_time * n_units)
            ci = np.arange(n_candidate, dtype=np.int32)
            ti = np.full(n_candidate, -1, dtype=np.int32)
            tt = np.full(n_candidate, -1, dtype=np.int32)
        else:
            raise ValueError(f"Unknown point_mode={point_mode}")
        return points, ci, ti, tt
    if x.ndim != 4:
        raise ValueError(f"Expected 3D or 4D response table, got {x.shape}")
    n_candidate, n_traj, n_time, n_units = x.shape
    if point_mode == "trajectory_mean":
        points = np.mean(x, axis=2).reshape(n_candidate * n_traj, n_units)
        ci = np.repeat(np.arange(n_candidate, dtype=np.int32), n_traj)
        ti = np.tile(np.arange(n_traj, dtype=np.int32), n_candidate)
        tt = np.full(points.shape[0], -1, dtype=np.int32)
    elif point_mode == "state_timepoints":
        points = x.reshape(n_candidate * n_traj * n_time, n_units)
        ci = np.repeat(np.arange(n_candidate, dtype=np.int32), n_traj * n_time)
        ti = np.tile(np.repeat(np.arange(n_traj, dtype=np.int32), n_time), n_candidate)
        tt = np.tile(np.arange(n_time, dtype=np.int32), n_candidate * n_traj)
    elif point_mode == "flatten_time":
        points = x.reshape(n_candidate * n_traj, n_time * n_units)
        ci = np.repeat(np.arange(n_candidate, dtype=np.int32), n_traj)
        ti = np.tile(np.arange(n_traj, dtype=np.int32), n_candidate)
        tt = np.full(points.shape[0], -1, dtype=np.int32)
    else:
        raise ValueError(f"Unknown point_mode={point_mode}")
    return points, ci, ti, tt


def _variant_array(table: Any, variant: str) -> np.ndarray:
    prior = np.asarray(table["prior_lambda_counts"], dtype=np.float32)
    if variant == "prior_response":
        return prior
    if variant == "motion_delta":
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float32)
        return prior - zero[:, None, :, :]
    if variant == "zero_static":
        return np.asarray(table["zero_lambda_counts"], dtype=np.float32)
    if variant == "known_response":
        return np.asarray(table["known_lambda_counts"], dtype=np.float32)
    raise ValueError(f"Unknown response variant {variant!r}")


def _pca_from_points(points: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(points, dtype=np.float64)
    mean = np.mean(x, axis=0, keepdims=True)
    xc = x - mean
    if x.shape[0] <= 1 or x.shape[1] == 0:
        return np.zeros(0), np.zeros((x.shape[1], 0)), np.zeros((x.shape[0], 0))
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    evals, vecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    vecs = vecs[:, order]
    k = max(1, min(int(n_components), vecs.shape[1]))
    scores = xc @ vecs[:, :k]
    return evals, vecs[:, :k], scores


def _spectrum_rows(evals: np.ndarray, *, variant: str, point_mode: str) -> list[dict[str, object]]:
    vals = np.asarray(evals, dtype=np.float64)
    total = float(np.sum(vals))
    rows = []
    cum = 0.0
    for idx, val in enumerate(vals):
        frac = float(val / (total + 1e-12)) if total > 0 else float("nan")
        cum += frac if np.isfinite(frac) else 0.0
        rows.append(
            {
                "response_variant": variant,
                "point_mode": point_mode,
                "pc_index": int(idx + 1),
                "eigenvalue": float(val),
                "explained_fraction": frac,
                "cumulative_fraction": float(cum),
            }
        )
    return rows


def _variance_by_group(points: np.ndarray, labels: list[str]) -> float:
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or len(labels) != x.shape[0]:
        return float("nan")
    global_mean = np.mean(x, axis=0)
    total = float(np.sum((x - global_mean[None, :]) ** 2))
    if total <= 1e-12:
        return float("nan")
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[str(label)].append(int(idx))
    between = 0.0
    for idxs in groups.values():
        block = x[np.asarray(idxs, dtype=np.int64)]
        mu = np.mean(block, axis=0)
        between += float(block.shape[0] * np.sum((mu - global_mean) ** 2))
    return float(between / total)


def _local_chart_rows(
    points: np.ndarray,
    labels: list[str],
    *,
    variant: str,
    point_mode: str,
    min_points: int,
    max_groups: int,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[str(label)].append(int(idx))
    rows: list[dict[str, object]] = []
    planes: list[np.ndarray] = []
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if int(max_groups) > 0:
        ordered = ordered[: int(max_groups)]
    for label, idxs in ordered:
        if len(idxs) < int(min_points):
            continue
        x = np.asarray(points[np.asarray(idxs, dtype=np.int64)], dtype=np.float64)
        evals, vecs, _scores = _pca_from_points(x, n_components=min(5, x.shape[1]))
        total = float(np.sum(evals))
        if total <= 1e-12:
            continue
        frac = evals / (total + 1e-12)
        if vecs.shape[1] >= 2:
            planes.append(np.asarray(vecs[:, :2], dtype=np.float64))
        rows.append(
            {
                "response_variant": variant,
                "point_mode": point_mode,
                "candidate_id": label,
                "n_points": int(len(idxs)),
                "local_rank_50": _rank_at(frac, 0.50),
                "local_rank_75": _rank_at(frac, 0.75),
                "local_rank_90": _rank_at(frac, 0.90),
                "local_participation_ratio": _participation_ratio(evals),
                "local_line_fraction": float(np.sum(frac[:1])),
                "local_plane_fraction": float(np.sum(frac[:2])),
                "local_top5_fraction": float(np.sum(frac[:5])),
            }
        )
    return rows, planes


def _plane_similarity(planes: list[np.ndarray], *, max_pairs: int, seed: int) -> np.ndarray:
    if len(planes) < 2:
        return np.zeros(0, dtype=np.float64)
    pairs = [(i, j) for i in range(len(planes)) for j in range(i + 1, len(planes))]
    rng = np.random.default_rng(int(seed))
    if int(max_pairs) > 0 and len(pairs) > int(max_pairs):
        keep = rng.choice(len(pairs), size=int(max_pairs), replace=False)
        pairs = [pairs[int(i)] for i in keep]
    vals = []
    for i, j in pairs:
        a = planes[i]
        b = planes[j]
        vals.append(float(np.sum((a.T @ b) ** 2) / min(a.shape[1], b.shape[1])))
    return np.asarray(vals, dtype=np.float64)


def _knn_same_label(
    coords: np.ndarray,
    labels: list[str],
    *,
    k: int,
    max_points: int,
    seed: int,
) -> dict[str, float]:
    x = np.asarray(coords, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] <= int(k) or len(labels) != x.shape[0]:
        return {
            "knn_same_candidate_fraction": float("nan"),
            "knn_same_candidate_chance": float("nan"),
            "knn_same_candidate_lift": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    idx = np.arange(x.shape[0])
    if int(max_points) > 0 and x.shape[0] > int(max_points):
        idx = np.sort(rng.choice(idx, size=int(max_points), replace=False))
        x = x[idx]
        labels = [labels[int(i)] for i in idx]
    norms = np.sum(x * x, axis=1)
    same_counts = []
    kk = min(int(k), x.shape[0] - 1)
    for start in range(0, x.shape[0], 512):
        stop = min(start + 512, x.shape[0])
        d2 = norms[start:stop, None] + norms[None, :] - 2.0 * (x[start:stop] @ x.T)
        for local_i, row in enumerate(d2):
            global_i = start + local_i
            row[global_i] = np.inf
            nn = np.argpartition(row, kk)[:kk]
            same_counts.append(float(np.mean([labels[int(j)] == labels[global_i] for j in nn])))
    counts = Counter(labels)
    n = float(len(labels))
    chance = float(sum((c / n) ** 2 for c in counts.values())) if n > 0 else float("nan")
    obs = _finite_mean(same_counts)
    return {
        "knn_same_candidate_fraction": obs,
        "knn_same_candidate_chance": chance,
        "knn_same_candidate_lift": float(obs / (chance + 1e-12)) if np.isfinite(obs) and np.isfinite(chance) else float("nan"),
    }


def _centroid_distance_correlation(
    points: np.ndarray,
    labels: list[str],
    feature_map: dict[str, dict[str, float | str]],
    *,
    max_pairs: int,
    seed: int,
) -> float:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label in feature_map:
            groups[str(label)].append(int(idx))
    centroids = []
    xy = []
    for label, idxs in sorted(groups.items()):
        meta = feature_map.get(label, {})
        px = _safe_float(meta.get("patch_x"))
        py = _safe_float(meta.get("patch_y"))
        if not np.isfinite(px) or not np.isfinite(py) or len(idxs) == 0:
            continue
        centroids.append(np.mean(points[np.asarray(idxs, dtype=np.int64)], axis=0))
        xy.append((px, py))
    if len(centroids) < 4:
        return float("nan")
    pairs = [(i, j) for i in range(len(centroids)) for j in range(i + 1, len(centroids))]
    rng = np.random.default_rng(int(seed))
    if int(max_pairs) > 0 and len(pairs) > int(max_pairs):
        keep = rng.choice(len(pairs), size=int(max_pairs), replace=False)
        pairs = [pairs[int(i)] for i in keep]
    response_d = []
    image_d = []
    c = np.asarray(centroids, dtype=np.float64)
    pos = np.asarray(xy, dtype=np.float64)
    for i, j in pairs:
        response_d.append(float(np.linalg.norm(c[i] - c[j])))
        image_d.append(float(np.linalg.norm(pos[i] - pos[j])))
    return _spearman(response_d, image_d)


def _shape_label(summary: dict[str, object]) -> str:
    plane = _safe_float(summary.get("global_plane_fraction"))
    top10 = _safe_float(summary.get("global_top10_fraction"))
    between = _safe_float(summary.get("between_candidate_variance_fraction"))
    local_plane = _safe_float(summary.get("local_plane_fraction_median"))
    local_line = _safe_float(summary.get("local_line_fraction_median"))
    if plane >= 0.70 and local_plane >= 0.80:
        return "global_sheet_with_local_charts"
    if local_plane >= 0.85 and top10 >= 0.75 and between >= 0.45:
        return "content_indexed_local_sheets"
    if local_line >= 0.80 and between >= 0.45:
        return "content_indexed_local_ribbons"
    if top10 >= 0.75:
        return "compact_global_cloud"
    return "distributed_high_dimensional_cloud"


def _collect_variant_points(
    run_dir: Path,
    manifest_rows: list[dict[str, str]],
    *,
    variant: str,
    point_mode: str,
    max_points: int,
    seed: int,
    dedupe_zero_static: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    rng = np.random.default_rng(int(seed))
    chunks: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []
    seen_zero: set[tuple[str, int]] = set()

    def _trim_if_needed(*, force: bool = False) -> None:
        if int(max_points) <= 0 or not chunks:
            return
        total = sum(int(chunk.shape[0]) for chunk in chunks)
        if not force and total <= int(max_points) * 2:
            return
        points_cat = np.concatenate(chunks, axis=0)
        if points_cat.shape[0] <= int(max_points):
            chunks[:] = [points_cat]
            return
        idx = np.sort(rng.choice(points_cat.shape[0], size=int(max_points), replace=False))
        chunks[:] = [points_cat[idx]]
        meta_rows[:] = [meta_rows[int(i)] for i in idx]

    for table_index, row in enumerate(manifest_rows):
        table_path = run_dir / str(row["response_cache_path"])
        if not table_path.exists():
            continue
        table = np.load(table_path, allow_pickle=True)
        arr = _variant_array(table, variant)
        points, cand_idx, traj_idx, time_idx = _reshape_points(arr, point_mode=point_mode)
        candidate_ids = _string_array(table, "candidate_ids", arr.shape[0], "candidate:")
        n_traj = arr.shape[1] if arr.ndim == 4 else 0
        trajectory_ids = _string_array(table, "prior_trajectory_ids", n_traj, "trajectory:")
        if points.size == 0:
            continue
        keep = np.ones(points.shape[0], dtype=bool)
        if bool(dedupe_zero_static) and variant == "zero_static":
            for i in range(points.shape[0]):
                cid = candidate_ids[int(cand_idx[i])]
                key = (cid, int(time_idx[i]) if int(time_idx[i]) >= 0 else -1)
                if key in seen_zero:
                    keep[i] = False
                else:
                    seen_zero.add(key)
        if not bool(np.all(keep)):
            points = points[keep]
            cand_idx = cand_idx[keep]
            traj_idx = traj_idx[keep]
            time_idx = time_idx[keep]
        if points.shape[0] == 0:
            continue
        chunks.append(points.astype(np.float32, copy=False))
        for i in range(points.shape[0]):
            ci = int(cand_idx[i])
            ti = int(traj_idx[i])
            meta_rows.append(
                {
                    "table_index": int(table_index),
                    "trial_id": int(_safe_float(row.get("trial_id"), table_index)),
                    "candidate_id": candidate_ids[ci] if 0 <= ci < len(candidate_ids) else f"candidate:{ci}",
                    "candidate_index": int(ci),
                    "trajectory_id": trajectory_ids[ti] if 0 <= ti < len(trajectory_ids) else "",
                    "trajectory_index": int(ti),
                    "time_index": int(time_idx[i]),
                    "candidate_set_mode": str(row.get("candidate_set_mode", "")),
                    "prior_family": str(row.get("prior_family", "")),
                    "scale": _safe_float(row.get("scale")),
                }
            )
        _trim_if_needed(force=False)
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32), []
    _trim_if_needed(force=True)
    points_all = np.concatenate(chunks, axis=0)
    return points_all, meta_rows


def _plot_variant(
    *,
    summary: dict[str, object],
    spectrum_rows: list[dict[str, object]],
    local_rows: list[dict[str, object]],
    points_rows: list[dict[str, object]],
    out_path: Path,
) -> None:
    if not points_rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    block_spec = [r for r in spectrum_rows if r["response_variant"] == summary["response_variant"]]
    pcs = [int(r["pc_index"]) for r in block_spec[:40]]
    cum = [float(r["cumulative_fraction"]) for r in block_spec[:40]]
    axes[0, 0].plot(pcs, cum, marker="o", ms=3)
    axes[0, 0].set_title("Global spectrum")
    axes[0, 0].set_xlabel("PC")
    axes[0, 0].set_ylabel("cumulative variance")
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].grid(alpha=0.25)

    pc1 = np.asarray([_safe_float(r.get("pc1")) for r in points_rows], dtype=np.float64)
    pc2 = np.asarray([_safe_float(r.get("pc2")) for r in points_rows], dtype=np.float64)
    time = np.asarray([_safe_float(r.get("time_index")) for r in points_rows], dtype=np.float64)
    source = np.asarray([_safe_float(str(r.get("candidate_id", "")).split(":")[-1]) for r in points_rows], dtype=np.float64)
    axes[0, 1].scatter(pc1, pc2, s=4, c=source, cmap="viridis", alpha=0.55, linewidths=0)
    axes[0, 1].set_title("PC1-PC2 by fixation id")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    axes[1, 0].scatter(pc1, pc2, s=4, c=time, cmap="magma", alpha=0.55, linewidths=0)
    axes[1, 0].set_title("PC1-PC2 by time")
    axes[1, 0].set_xlabel("PC1")
    axes[1, 0].set_ylabel("PC2")

    block_local = [r for r in local_rows if r["response_variant"] == summary["response_variant"]]
    plane = [_safe_float(r.get("local_plane_fraction")) for r in block_local]
    line = [_safe_float(r.get("local_line_fraction")) for r in block_local]
    vals = [v for v in plane if np.isfinite(v)]
    if vals:
        axes[1, 1].hist(vals, bins=np.linspace(0, 1, 21), alpha=0.65, label="plane")
    vals = [v for v in line if np.isfinite(v)]
    if vals:
        axes[1, 1].hist(vals, bins=np.linspace(0, 1, 21), alpha=0.55, label="line")
    axes[1, 1].set_title("Local chart fractions")
    axes[1, 1].set_xlabel("variance fraction")
    axes[1, 1].set_ylabel("candidate count")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        axes[1, 1].legend(handles, labels, frameon=False, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "response_cache_manifest.csv"
    manifest_all = _read_csv_rows(manifest_path)
    manifest_rows = _manifest_filter(manifest_all, args)
    if not manifest_rows:
        raise ValueError(f"No response-cache rows selected from {manifest_path}")
    feature_map = _load_candidate_feature_map(run_dir)
    variants = _parse_list(args.response_variants) or ["motion_delta"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    local_rows_all: list[dict[str, object]] = []
    points_rows_all: list[dict[str, object]] = []

    for variant_index, variant in enumerate(variants):
        points, point_meta = _collect_variant_points(
            run_dir,
            manifest_rows,
            variant=variant,
            point_mode=str(args.point_mode),
            max_points=int(args.max_points),
            seed=int(args.seed) + 1000 * variant_index,
            dedupe_zero_static=bool(args.dedupe_zero_static),
        )
        if points.shape[0] < int(args.min_points):
            summary_rows.append(
                {
                    "response_variant": variant,
                    "point_mode": str(args.point_mode),
                    "n_points": int(points.shape[0]),
                    "status": "not_run_insufficient_points",
                }
            )
            continue
        evals, _vecs, scores = _pca_from_points(points, n_components=int(args.n_pcs))
        spec = _spectrum_rows(evals, variant=variant, point_mode=str(args.point_mode))
        spectrum_rows.extend(spec)
        total = float(np.sum(evals))
        frac = evals / (total + 1e-12) if total > 0 else np.zeros_like(evals)
        labels = [str(r["candidate_id"]) for r in point_meta]
        between = _variance_by_group(points, labels)
        local_rows, planes = _local_chart_rows(
            points,
            labels,
            variant=variant,
            point_mode=str(args.point_mode),
            min_points=int(args.min_local_points),
            max_groups=int(args.max_local_groups),
        )
        local_rows_all.extend(local_rows)
        sim = _plane_similarity(planes, max_pairs=int(args.max_plane_pairs), seed=int(args.seed) + 7 + variant_index)
        neighbor_dims = max(1, min(int(args.neighbor_dims), scores.shape[1]))
        knn = _knn_same_label(
            scores[:, :neighbor_dims],
            labels,
            k=int(args.neighbor_k),
            max_points=int(args.max_neighbor_points),
            seed=int(args.seed) + 17 + variant_index,
        )
        centroid_corr = _centroid_distance_correlation(
            scores[:, :neighbor_dims],
            labels,
            feature_map,
            max_pairs=int(args.max_centroid_pairs),
            seed=int(args.seed) + 31 + variant_index,
        )
        summary = {
            "response_variant": variant,
            "point_mode": str(args.point_mode),
            "n_points": int(points.shape[0]),
            "n_dimensions": int(points.shape[1]),
            "n_manifest_rows_selected": int(len(manifest_rows)),
            "n_unique_candidates": int(len(set(labels))),
            "global_participation_ratio": _participation_ratio(evals),
            "global_rank_50": _rank_at(frac, 0.50),
            "global_rank_75": _rank_at(frac, 0.75),
            "global_rank_90": _rank_at(frac, 0.90),
            "global_rank_95": _rank_at(frac, 0.95),
            "global_line_fraction": float(np.sum(frac[:1])),
            "global_plane_fraction": float(np.sum(frac[:2])),
            "global_top3_fraction": float(np.sum(frac[:3])),
            "global_top5_fraction": float(np.sum(frac[:5])),
            "global_top10_fraction": float(np.sum(frac[:10])),
            "between_candidate_variance_fraction": between,
            "within_candidate_variance_fraction": float(1.0 - between) if np.isfinite(between) else float("nan"),
            "local_chart_count": int(len(local_rows)),
            "local_line_fraction_median": _finite_median([_safe_float(r.get("local_line_fraction")) for r in local_rows]),
            "local_plane_fraction_median": _finite_median([_safe_float(r.get("local_plane_fraction")) for r in local_rows]),
            "local_top5_fraction_median": _finite_median([_safe_float(r.get("local_top5_fraction")) for r in local_rows]),
            "local_participation_ratio_median": _finite_median([_safe_float(r.get("local_participation_ratio")) for r in local_rows]),
            "local_plane_similarity_median": _finite_median(sim),
            "local_plane_similarity_ci_low": _finite_ci(sim, 2.5),
            "local_plane_similarity_ci_high": _finite_ci(sim, 97.5),
            "centroid_response_vs_patch_position_spearman": centroid_corr,
            **knn,
            "status": "ok",
        }
        summary["shape_label"] = _shape_label(summary)
        summary_rows.append(summary)

        rng = np.random.default_rng(int(args.seed) + 4000 + variant_index)
        point_idx = np.arange(points.shape[0])
        if int(args.max_pca_rows) > 0 and point_idx.size > int(args.max_pca_rows):
            point_idx = np.sort(rng.choice(point_idx, size=int(args.max_pca_rows), replace=False))
        point_rows: list[dict[str, object]] = []
        for i in point_idx:
            meta = dict(point_meta[int(i)])
            candidate_id = str(meta.get("candidate_id", ""))
            features = feature_map.get(candidate_id, {})
            row: dict[str, object] = {
                "response_variant": variant,
                "point_mode": str(args.point_mode),
                "point_index": int(i),
                **meta,
            }
            for pc in range(min(scores.shape[1], int(args.n_pcs_to_write))):
                row[f"pc{pc + 1}"] = float(scores[int(i), pc])
            for key, value in features.items():
                row[key] = value
            point_rows.append(row)
        points_rows_all.extend(point_rows)
        _plot_variant(
            summary=summary,
            spectrum_rows=spectrum_rows,
            local_rows=local_rows_all,
            points_rows=point_rows,
            out_path=out_dir / "figures" / f"global_fixation_geometry_{variant}_{args.point_mode}.png",
        )

    _write_csv(out_dir / "global_fixation_geometry_summary.csv", summary_rows)
    _write_csv(out_dir / "global_fixation_geometry_spectrum.csv", spectrum_rows)
    _write_csv(out_dir / "global_fixation_geometry_local_charts.csv", local_rows_all)
    _write_csv(out_dir / "global_fixation_geometry_points_pca.csv", points_rows_all)
    manifest = {
        "run_dir": str(run_dir),
        "response_cache_manifest": str(manifest_path),
        "output_dir": str(out_dir),
        "point_mode": str(args.point_mode),
        "response_variants": variants,
        "n_manifest_rows_total": int(len(manifest_all)),
        "n_manifest_rows_selected": int(len(manifest_rows)),
        "filters": {
            "candidate_set_modes": _parse_list(args.candidate_set_modes),
            "prior_families": _parse_list(args.prior_families),
            "scales": _parse_list(args.scales),
            "axis_catalog_modes": _parse_list(args.axis_catalog_modes),
        },
        "max_points": int(args.max_points),
        "summary_status_counts": {
            str(status): int(sum(1 for r in summary_rows if str(r.get("status", "")) == str(status)))
            for status in sorted({str(r.get("status", "")) for r in summary_rows})
        },
    }
    _save_json(out_dir / "global_fixation_geometry_manifest.json", manifest)
    readme = [
        "# Global BackImage Fixation Geometry",
        "",
        "This cache-only probe treats each BackImage candidate fixation/window as a local image object and asks whether the union of cached response states has a recognizable global geometry.",
        "",
        f"Point mode: `{args.point_mode}`.",
        "",
        "Primary outputs:",
        "- global_fixation_geometry_summary.csv",
        "- global_fixation_geometry_spectrum.csv",
        "- global_fixation_geometry_local_charts.csv",
        "- global_fixation_geometry_points_pca.csv",
        "- figures/global_fixation_geometry_*.png",
        "",
        "Interpretation notes:",
        "- `prior_response` is the full finite response table.",
        "- `motion_delta` subtracts the static patch-center response, isolating motion-driven change.",
        "- `zero_static` is the static local image-content manifold.",
        "- `between_candidate_variance_fraction` separates content/fixation identity from within-fixation motion variation.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze global geometry of cached BackImage fixation response tables.")
    p.add_argument("--run-dir", type=Path, required=True, help="BackImage observer run directory with response_cache_manifest.csv.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=VISIONCORE_ROOT / "outputs" / "backimage_global_fixation_geometry",
    )
    p.add_argument("--candidate-set-modes", type=str, default="", help="Comma-separated manifest candidate_set_mode filter.")
    p.add_argument("--prior-families", type=str, default="", help="Comma-separated manifest prior_family filter.")
    p.add_argument("--scales", type=str, default="", help="Comma-separated manifest scale filter.")
    p.add_argument("--axis-catalog-modes", type=str, default="", help="Comma-separated axis_catalog_mode filter.")
    p.add_argument("--max-tables", type=int, default=0, help="0 means all selected tables.")
    p.add_argument(
        "--response-variants",
        type=str,
        default="motion_delta,prior_response,zero_static",
        help="Comma-separated variants: motion_delta, prior_response, zero_static, known_response.",
    )
    p.add_argument(
        "--point-mode",
        choices=("state_timepoints", "trajectory_mean", "flatten_time"),
        default="state_timepoints",
    )
    p.add_argument("--max-points", type=int, default=50000)
    p.add_argument("--min-points", type=int, default=64)
    p.add_argument("--min-local-points", type=int, default=20)
    p.add_argument("--max-local-groups", type=int, default=512)
    p.add_argument("--max-plane-pairs", type=int, default=10000)
    p.add_argument("--neighbor-dims", type=int, default=10)
    p.add_argument("--neighbor-k", type=int, default=15)
    p.add_argument("--max-neighbor-points", type=int, default=5000)
    p.add_argument("--max-centroid-pairs", type=int, default=20000)
    p.add_argument("--n-pcs", type=int, default=20)
    p.add_argument("--n-pcs-to-write", type=int, default=8)
    p.add_argument("--max-pca-rows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dedupe-zero-static", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> None:
    manifest = analyze(build_parser().parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
