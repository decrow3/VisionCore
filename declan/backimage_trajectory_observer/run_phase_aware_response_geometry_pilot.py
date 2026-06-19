from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from declan.backimage_trajectory_observer.plot_global_fixation_trajectory_lines_3d import (
    _collect_trajectories,
    _farthest_subset,
    _manifest_filter,
    _parse_list,
    _read_csv_rows,
    _safe_float,
)
from declan.backimage_trajectory_observer.plot_global_fixation_fourier_component_flow import (
    _component_group,
    _controlled_trace,
    _fft_cache_for_sources,
    _parse_bands,
    _reconstruct_prior_trace_map,
    _safe_int,
    _select_frequency_components,
)


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


def _unit_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


def _fit_pca(groups: list[np.ndarray], n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not groups:
        raise ValueError("Need at least one group")
    n_dim = int(groups[0].shape[1])
    n_total = 0
    total = np.zeros(n_dim, dtype=np.float64)
    for group in groups:
        x = np.asarray(group, dtype=np.float64)
        if x.shape[1] != n_dim:
            raise ValueError("All groups must have the same dimensionality")
        n_total += int(x.shape[0])
        total += np.sum(x, axis=0)
    mean = total / float(n_total)
    cov = np.zeros((n_dim, n_dim), dtype=np.float64)
    for group in groups:
        xc = np.asarray(group, dtype=np.float64) - mean[None, :]
        cov += xc.T @ xc
    cov /= float(max(n_total - 1, 1))
    evals, vecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    vecs = vecs[:, order]
    k = min(max(1, int(n_components)), vecs.shape[1])
    return mean.astype(np.float32), vecs[:, :k].astype(np.float32), evals.astype(np.float32)


def _project_groups(groups: list[np.ndarray], mean: np.ndarray, basis: np.ndarray) -> list[np.ndarray]:
    m = np.asarray(mean, dtype=np.float32)
    b = np.asarray(basis, dtype=np.float32)
    return [(np.asarray(group, dtype=np.float32) - m[None, :]) @ b for group in groups]


def _participation_ratio(evals: np.ndarray) -> float:
    vals = np.asarray(evals, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float("nan")
    return float((np.sum(vals) ** 2) / (np.sum(vals * vals) + 1e-12))


def _split_sources(source_rows: list[int], test_fraction: float, seed: int) -> tuple[set[int], set[int]]:
    unique = np.asarray(sorted(set(int(v) for v in source_rows)), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique)
    n_test = max(1, int(round(float(test_fraction) * unique.size)))
    n_test = min(n_test, unique.size - 1)
    test = set(int(v) for v in unique[:n_test])
    train = set(int(v) for v in unique[n_test:])
    return train, test


def _flatten_groups(
    groups: list[np.ndarray],
    source_rows: list[int],
    selected_sources: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    srcs: list[np.ndarray] = []
    for i, (group, src) in enumerate(zip(groups, source_rows, strict=False)):
        if selected_sources is not None and int(src) not in selected_sources:
            continue
        x = np.asarray(group, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
            continue
        xs.append(x)
        group_ids.append(np.full(x.shape[0], int(i), dtype=np.int64))
        srcs.append(np.full(x.shape[0], int(src), dtype=np.int64))
    if not xs:
        raise ValueError("No selected finite groups")
    return np.concatenate(xs), np.concatenate(group_ids), np.concatenate(srcs)


def _standardize(train: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, keepdims=True)
    sd = np.std(train, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return (all_x - mean) / sd, mean.squeeze(0), sd.squeeze(0)


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    return np.linalg.solve(xtx + float(alpha) * np.eye(xtx.shape[0], dtype=np.float64), x.T @ y)


def _phase_targets_from_component_group(
    component_group: np.ndarray,
    *,
    mode: str,
    base_component_group: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(component_group, dtype=np.float32)
    half = arr.shape[1] // 2
    z = arr[:, :half] + 1j * arr[:, half : 2 * half]
    z = z / (np.abs(z) + 1e-8)
    if mode == "relative_to_zero":
        if base_component_group is None:
            raise ValueError("relative_to_zero phase targets require base_component_group")
        base = np.asarray(base_component_group, dtype=np.float32)
        base_z = base[:, :half] + 1j * base[:, half : 2 * half]
        base_z = base_z / (np.abs(base_z) + 1e-8)
        z = z * np.conj(base_z[:1])
    elif mode == "relative_to_start":
        # Remove the image-specific base phase so motion_delta responses are
        # paired with displacement-induced phase advance from trace onset.
        z = z * np.conj(z[:1])
    elif mode == "step_advance":
        step_z = np.empty_like(z)
        step_z[0] = 1.0 + 0.0j
        if z.shape[0] > 1:
            step_z[1:] = z[1:] * np.conj(z[:-1])
        z = step_z
    elif mode != "absolute":
        raise ValueError(f"Unknown phase target mode {mode!r}")
    avg = np.mean(z, axis=1)
    resultant = np.abs(avg).astype(np.float32)
    return np.stack([np.real(avg), np.imag(avg)], axis=1).astype(np.float32), resultant


def _target_groups_for_band(
    fft_cache: dict[int, dict[str, Any]],
    trace_map: dict[str, np.ndarray],
    meta: list[dict[str, object]],
    source_rows: list[int],
    spec: dict[str, Any],
    *,
    normalization: str,
    phase_target_mode: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    out = []
    resultants = []
    for src, row in zip(source_rows, meta, strict=False):
        trace = trace_map[str(row["trajectory_id"])]
        comp = _component_group(
            fft_cache[int(src)],
            trace,
            list(spec["indices"]),
            list(spec["avg_power"]),
            normalization=normalization,
        )
        base_comp = None
        if phase_target_mode == "relative_to_zero":
            base_comp = _component_group(
                fft_cache[int(src)],
                np.zeros_like(trace),
                list(spec["indices"]),
                list(spec["avg_power"]),
                normalization=normalization,
            )
        target, resultant = _phase_targets_from_component_group(
            comp,
            mode=phase_target_mode,
            base_component_group=base_comp,
        )
        out.append(target)
        resultants.append(resultant)
    return out, resultants


def _mismatched_target_groups(
    target_groups: list[np.ndarray],
    source_rows: list[int],
    *,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(target_groups))
    for _ in range(32):
        perm = rng.permutation(order)
        if not any(source_rows[int(i)] == source_rows[int(j)] for i, j in zip(order, perm, strict=False)):
            return [target_groups[int(j)] for j in perm]
    return [target_groups[int(j)] for j in rng.permutation(order)]


def _mismatched_source_rows(source_rows: list[int], *, seed: int) -> list[int]:
    unique = np.asarray(sorted(set(int(v) for v in source_rows)), dtype=np.int64)
    if unique.size < 2:
        raise ValueError("Need at least two source rows for source-level mismatch control")
    rng = np.random.default_rng(int(seed))
    for _ in range(128):
        perm = rng.permutation(unique)
        if np.all(perm != unique):
            mapping = {int(src): int(dst) for src, dst in zip(unique, perm, strict=False)}
            return [mapping[int(src)] for src in source_rows]
    perm = np.roll(unique, 1)
    mapping = {int(src): int(dst) for src, dst in zip(unique, perm, strict=False)}
    return [mapping[int(src)] for src in source_rows]


def _trace_xy_groups(trace_map: dict[str, np.ndarray], meta: list[dict[str, object]]) -> list[np.ndarray]:
    return [np.asarray(trace_map[str(row["trajectory_id"])], dtype=np.float32) for row in meta]


def _trace_baseline_groups_for_mode(trace_groups: list[np.ndarray], *, mode: str) -> list[np.ndarray]:
    out = []
    for trace in trace_groups:
        arr = np.asarray(trace, dtype=np.float32)
        if mode == "absolute":
            out.append(arr)
        elif mode == "relative_to_zero":
            out.append(arr)
        elif mode == "relative_to_start":
            out.append(arr - arr[:1])
        elif mode == "step_advance":
            step = np.zeros_like(arr)
            if arr.shape[0] > 1:
                step[1:] = arr[1:] - arr[:-1]
            out.append(step)
        else:
            raise ValueError(f"Unknown phase target mode {mode!r}")
    return out


def _fit_phase_readout(
    response_groups: list[np.ndarray],
    target_groups: list[np.ndarray],
    source_rows: list[int],
    train_sources: set[int],
    test_sources: set[int],
    *,
    ridge_alpha: float,
    predictor_label: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_all, _gid, src_all = _flatten_groups(response_groups, source_rows)
    y_all, _ygid, _ysrc = _flatten_groups(target_groups, source_rows)
    if x_all.shape[0] != y_all.shape[0]:
        raise ValueError("Response and target groups flattened to different row counts")
    train_mask = np.isin(src_all, list(train_sources))
    test_mask = np.isin(src_all, list(test_sources))
    x_train_raw = x_all[train_mask].astype(np.float64)
    y_train = y_all[train_mask].astype(np.float64)
    y_test = y_all[test_mask].astype(np.float64)
    x_std_all, mean, sd = _standardize(x_train_raw, x_all.astype(np.float64))
    x_train = x_std_all[train_mask]
    x_test = x_std_all[test_mask]
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    w = _ridge_fit(x_train, y_train - y_mean, alpha=float(ridge_alpha))
    pred = x_test @ w + y_mean
    ss_res = np.sum((y_test - pred) ** 2, axis=0)
    ss_tot = np.sum((y_test - y_mean) ** 2, axis=0) + 1e-12
    r2_vec = 1.0 - ss_res / ss_tot
    cos = np.sum(_unit_rows(pred) * _unit_rows(y_test), axis=1)
    row = {
        "predictor": predictor_label,
        "n_train_points": int(np.count_nonzero(train_mask)),
        "n_test_points": int(np.count_nonzero(test_mask)),
        "cos_r2": float(r2_vec[0]),
        "sin_r2": float(r2_vec[1]),
        "mean_r2": float(np.mean(r2_vec)),
        "phase_vector_cosine_mean": float(np.mean(cos)),
        "phase_vector_cosine_median": float(np.median(cos)),
    }
    return row, w.astype(np.float32), mean.astype(np.float32), sd.astype(np.float32), x_std_all.astype(np.float32)


def _fit_trace_baseline(
    trace_groups: list[np.ndarray],
    target_groups: list[np.ndarray],
    source_rows: list[int],
    train_sources: set[int],
    test_sources: set[int],
    *,
    ridge_alpha: float,
    predictor_label: str,
) -> dict[str, object]:
    x_all, _gid, src_all = _flatten_groups(trace_groups, source_rows)
    y_all, _ygid, _ysrc = _flatten_groups(target_groups, source_rows)
    train_mask = np.isin(src_all, list(train_sources))
    test_mask = np.isin(src_all, list(test_sources))
    x_train_raw = x_all[train_mask].astype(np.float64)
    y_train = y_all[train_mask].astype(np.float64)
    y_test = y_all[test_mask].astype(np.float64)
    x_std_all, _mean, _sd = _standardize(x_train_raw, x_all.astype(np.float64))
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    w = _ridge_fit(x_std_all[train_mask], y_train - y_mean, alpha=float(ridge_alpha))
    pred = x_std_all[test_mask] @ w + y_mean
    ss_res = np.sum((y_test - pred) ** 2, axis=0)
    ss_tot = np.sum((y_test - y_mean) ** 2, axis=0) + 1e-12
    r2_vec = 1.0 - ss_res / ss_tot
    cos = np.sum(_unit_rows(pred) * _unit_rows(y_test), axis=1)
    return {
        "predictor": predictor_label,
        "n_train_points": int(np.count_nonzero(train_mask)),
        "n_test_points": int(np.count_nonzero(test_mask)),
        "cos_r2": float(r2_vec[0]),
        "sin_r2": float(r2_vec[1]),
        "mean_r2": float(np.mean(r2_vec)),
        "phase_vector_cosine_mean": float(np.mean(cos)),
        "phase_vector_cosine_median": float(np.median(cos)),
    }


def _third_axis_from_response_pc(x_std_all: np.ndarray, train_mask: np.ndarray, phase_axes: np.ndarray) -> np.ndarray:
    q, _r = np.linalg.qr(np.asarray(phase_axes, dtype=np.float64))
    train = np.asarray(x_std_all[train_mask], dtype=np.float64)
    _u, _s, vt = np.linalg.svd(train - np.mean(train, axis=0, keepdims=True), full_matrices=False)
    for axis in vt:
        v = axis - q @ (q.T @ axis)
        norm = float(np.linalg.norm(v))
        if norm > 1e-8:
            return (v / norm).astype(np.float32)
    raise ValueError("Could not find a response PC orthogonal to phase axes")


def _phase_projection_groups(
    response_groups: list[np.ndarray],
    mean: np.ndarray,
    sd: np.ndarray,
    phase_axes: np.ndarray,
    third_axis: np.ndarray,
) -> list[np.ndarray]:
    axes = np.column_stack([phase_axes, third_axis]).astype(np.float32)
    out = []
    for group in response_groups:
        z = (np.asarray(group, dtype=np.float32) - mean[None, :]) / sd[None, :]
        out.append(z @ axes)
    return out


def _linear_rotational_projection(
    fit_response_groups: list[np.ndarray],
    project_response_groups: list[np.ndarray],
    *,
    pca_components: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, object]]:
    mean, basis, evals = _fit_pca(fit_response_groups, n_components=max(3, int(pca_components)))
    fit_groups = _project_groups(fit_response_groups, mean, basis)
    project_groups = _project_groups(project_response_groups, mean, basis)
    starts = []
    stops = []
    for group in fit_groups:
        if group.shape[0] < 2:
            continue
        starts.append(group[:-1])
        stops.append(group[1:])
    x0 = np.concatenate(starts, axis=0).astype(np.float64)
    x1 = np.concatenate(stops, axis=0).astype(np.float64)
    center = np.mean(x0, axis=0, keepdims=True)
    x = x0 - center
    dx = x1 - x0
    a = _ridge_fit(x, dx, alpha=1e-3).T
    skew = 0.5 * (a - a.T)
    evals_skew, vecs = np.linalg.eig(skew.astype(np.complex128))
    order = np.argsort(np.abs(np.imag(evals_skew)))[::-1]
    v = vecs[:, order[0]]
    ax1 = np.real(v)
    ax2 = np.imag(v)
    if np.linalg.norm(ax2) < 1e-8:
        _u, _s, vt = np.linalg.svd(skew)
        ax1, ax2 = vt[0], vt[1]
    ax1 = ax1 / (np.linalg.norm(ax1) + 1e-12)
    ax2 = ax2 - ax1 * float(ax1 @ ax2)
    ax2 = ax2 / (np.linalg.norm(ax2) + 1e-12)
    pc1 = np.zeros_like(ax1)
    pc1[0] = 1.0
    ax3 = pc1 - ax1 * float(ax1 @ pc1) - ax2 * float(ax2 @ pc1)
    if np.linalg.norm(ax3) < 1e-8:
        ax3 = np.zeros_like(ax1)
        ax3[2 if ax3.size > 2 else 0] = 1.0
        ax3 = ax3 - ax1 * float(ax1 @ ax3) - ax2 * float(ax2 @ ax3)
    ax3 = ax3 / (np.linalg.norm(ax3) + 1e-12)
    rot_basis = np.column_stack([ax1, ax2, ax3]).astype(np.float32)
    projected = [(group - center.astype(np.float32)) @ rot_basis for group in project_groups]
    info = {
        "response_pca_components_for_dynamics": int(project_groups[0].shape[1]),
        "leading_skew_imag_abs": float(np.abs(np.imag(evals_skew[order[0]]))),
        "response_pca_explained_fraction_first3": [
            float(v) for v in (evals[:3] / (float(np.sum(evals)) + 1e-12)).tolist()
        ],
    }
    return projected, evals, info


def _projection_metrics(
    name: str,
    groups: list[np.ndarray],
    target_groups: list[np.ndarray] | None = None,
) -> dict[str, object]:
    radius_cvs = []
    tangential = []
    dpsi_all = []
    dtheta_all = []
    for i, group in enumerate(groups):
        xy = np.asarray(group[:, :2], dtype=np.float64)
        if xy.shape[0] < 2:
            continue
        radius = np.linalg.norm(xy, axis=1)
        if np.mean(radius) > 1e-12:
            radius_cvs.append(float(np.std(radius) / (np.mean(radius) + 1e-12)))
        vel = np.diff(xy, axis=0)
        pos = xy[:-1]
        numerator = vel[:, 0] * (-pos[:, 1]) + vel[:, 1] * pos[:, 0]
        denom = np.linalg.norm(vel, axis=1) * np.linalg.norm(pos, axis=1) + 1e-12
        tangential.extend((numerator / denom).tolist())
        psi = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
        dpsi = np.diff(psi)
        dpsi_all.extend(dpsi.tolist())
        if target_groups is not None:
            target = np.asarray(target_groups[i], dtype=np.float64)
            if target.shape[0] == xy.shape[0]:
                theta = np.unwrap(np.arctan2(target[:, 1], target[:, 0]))
                dtheta_all.extend(np.diff(theta).tolist())
    tang = np.asarray(tangential, dtype=np.float64)
    dpsi_arr = np.asarray(dpsi_all, dtype=np.float64)
    dtheta_arr = np.asarray(dtheta_all, dtype=np.float64)
    corr = float("nan")
    if dpsi_arr.size == dtheta_arr.size and dpsi_arr.size > 3:
        corr = float(np.corrcoef(dpsi_arr, dtheta_arr)[0, 1])
    return {
        "projection": name,
        "radius_cv_median": float(np.median(radius_cvs)) if radius_cvs else float("nan"),
        "radius_cv_mean": float(np.mean(radius_cvs)) if radius_cvs else float("nan"),
        "tangentiality_mean": float(np.mean(tang)) if tang.size else float("nan"),
        "tangentiality_abs_mean": float(np.mean(np.abs(tang))) if tang.size else float("nan"),
        "tangentiality_median": float(np.median(tang)) if tang.size else float("nan"),
        "angular_step_std": float(np.std(dpsi_arr)) if dpsi_arr.size else float("nan"),
        "phase_advance_corr": corr,
        "n_segments": int(tang.size),
    }


def _time_shuffle_groups(groups: list[np.ndarray], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    out = []
    for group in groups:
        order = rng.permutation(group.shape[0])
        out.append(np.asarray(group, dtype=np.float32)[order])
    return out


def _subset_groups(groups: list[np.ndarray], indices: np.ndarray) -> list[np.ndarray]:
    return [groups[int(i)] for i in np.asarray(indices, dtype=np.int64).tolist()]


def _indices_for_sources(source_rows: list[int], sources: set[int]) -> np.ndarray:
    return np.asarray([i for i, src in enumerate(source_rows) if int(src) in sources], dtype=np.int64)


def _axis_limits(coords: np.ndarray) -> tuple[np.ndarray, float]:
    lo = np.nanpercentile(coords, 1, axis=0)
    hi = np.nanpercentile(coords, 99, axis=0)
    center = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    return center, span


def _plot_3d_views(
    out_path: Path,
    groups: list[np.ndarray],
    meta: list[dict[str, object]],
    selected: np.ndarray,
    *,
    title: str,
    axis_labels: tuple[str, str, str],
    color_by: str = "time_index",
    line_alpha: float = 0.32,
    linewidth: float = 0.75,
    equal_axes: bool = True,
) -> None:
    shown = [np.asarray(groups[int(i)], dtype=np.float64)[:, :3] for i in selected]
    rows = [meta[int(i)] for i in selected]
    coords = np.concatenate(shown, axis=0)
    center, span = _axis_limits(coords)
    lo = np.nanpercentile(coords, 1, axis=0)
    hi = np.nanpercentile(coords, 99, axis=0)
    pad = 0.06 * np.maximum(hi - lo, 1e-9)
    if color_by == "time_index":
        color_values = np.concatenate([np.arange(group.shape[0], dtype=np.float64) for group in shown])
        color_label = "time index"
    elif color_by == "source_row":
        color_values = np.concatenate(
            [np.full(group.shape[0], _safe_float(row.get("source_row")), dtype=np.float64) for group, row in zip(shown, rows, strict=False)]
        )
        color_label = "source row"
    else:
        color_values = np.arange(coords.shape[0], dtype=np.float64)
        color_label = color_by
    finite = color_values[np.isfinite(color_values)]
    norm = Normalize(vmin=float(np.min(finite)), vmax=float(np.max(finite))) if finite.size else Normalize(0, 1)
    cmap = plt.get_cmap("viridis")
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    views = [(18, -60), (18, 35), (55, -60), (8, -100)]
    fig = plt.figure(figsize=(12.5, 10.0), constrained_layout=True)
    offset = 0
    for view_i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, view_i + 1, projection="3d")
        offset = 0
        for group in shown:
            colors = cmap(norm(color_values[offset : offset + group.shape[0]]))
            for t in range(group.shape[0] - 1):
                ax.plot(
                    group[t : t + 2, 0],
                    group[t : t + 2, 1],
                    group[t : t + 2, 2],
                    color=colors[t],
                    alpha=line_alpha,
                    linewidth=linewidth,
                )
            offset += group.shape[0]
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        if bool(equal_axes):
            for setter, val in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, strict=False):
                setter(float(val - 0.55 * span), float(val + 0.55 * span))
        else:
            for setter, lval, hval, pval in zip(
                (ax.set_xlim, ax.set_ylim, ax.set_zlim),
                lo,
                hi,
                pad,
                strict=False,
            ):
                setter(float(lval - pval), float(hval + pval))
        ax.set_title(f"elev {elev}, azim {azim}", fontsize=9)
    fig.suptitle(title, fontsize=13)
    cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.72, pad=0.02)
    cbar.set_label(color_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_metric_summary(out_path: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(row["projection"]) for row in rows if not str(row["projection"]).endswith("_time_shuffle")]
    tang = [float(row["tangentiality_abs_mean"]) for row in rows if not str(row["projection"]).endswith("_time_shuffle")]
    cv = [float(row["radius_cv_median"]) for row in rows if not str(row["projection"]).endswith("_time_shuffle")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x, tang, color="#0072B2")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("mean |tangentiality|")
    axes[0].set_title("Tangential flow")
    axes[1].bar(x, cv, color="#D55E00")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("median radius CV")
    axes[1].set_title("Circular radius stability")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _copy_or_plot_single_view(source: Path, dest: Path) -> None:
    # Keep requested filenames available without duplicating plotting code.
    dest.write_bytes(source.read_bytes())


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = _manifest_filter(_read_csv_rows(run_dir / "response_cache_manifest.csv"), args)
    if not manifest_rows:
        raise ValueError("No response-cache rows survived filters")
    response_groups, meta = _collect_trajectories(run_dir, manifest_rows, variant=str(args.variant))
    source_rows = [_safe_int(str(row["candidate_id"]).split(":")[-1]) for row in meta]
    train_sources, test_sources = _split_sources(source_rows, test_fraction=float(args.test_fraction), seed=int(args.seed))
    train_indices = _indices_for_sources(source_rows, train_sources)
    heldout_indices = _indices_for_sources(source_rows, test_sources)
    train_response_groups = _subset_groups(response_groups, train_indices)
    trace_map_real = _reconstruct_prior_trace_map(run_dir, manifest_rows)
    trace_map_time_perm = {
        key: _controlled_trace(val, mode="time_permute", rng=np.random.default_rng(int(args.seed) + 91 + i))
        for i, (key, val) in enumerate(sorted(trace_map_real.items()))
    }
    trace_groups = _trace_xy_groups(trace_map_real, meta)
    trace_baseline_groups = _trace_baseline_groups_for_mode(trace_groups, mode=str(args.phase_target_mode))
    fft_cache = _fft_cache_for_sources(
        run_dir,
        sorted(set(source_rows)),
        patch_size_px=int(args.patch_size_px),
        component_crop_px=int(args.component_crop_px),
        cache_path=(Path(args.fourier_cache) if args.fourier_cache is not None else None),
    )
    bands = _parse_bands(args.frequency_bands)
    selected_components = _select_frequency_components(fft_cache, bands, n_per_band=int(args.components_per_band))

    # A. Response PCA baseline.
    pca_mean, pca_basis, pca_evals = _fit_pca(train_response_groups, n_components=max(10, int(args.n_components)))
    response_pca_groups = _project_groups(response_groups, pca_mean, pca_basis)
    total_eval = float(np.sum(pca_evals)) + 1e-12
    pca_var = pca_evals / total_eval
    centroids = np.asarray([np.mean(group[:, :3], axis=0) for group in response_pca_groups], dtype=np.float64)
    selected_plot = _farthest_subset(centroids, min(int(args.max_trajectories), len(response_pca_groups)), seed=int(args.seed))
    _plot_3d_views(
        out_dir / "response_pca_3d_alt_views.png",
        response_pca_groups,
        meta,
        selected_plot,
        title=f"Neural response PCA baseline ({args.variant})",
        axis_labels=(
            f"neural PC1 ({100*pca_var[0]:.1f}%)",
            f"neural PC2 ({100*pca_var[1]:.1f}%)",
            f"neural PC3 ({100*pca_var[2]:.1f}%)",
        ),
    )
    _copy_or_plot_single_view(out_dir / "response_pca_3d_alt_views.png", out_dir / "response_pca_3d.png")

    # B. Phase-aware neural projection.
    prediction_rows: list[dict[str, object]] = []
    phase_fits: dict[str, dict[str, Any]] = {}
    for band_name, _lo, _hi in bands:
        if band_name not in selected_components:
            continue
        spec = selected_components[band_name]
        targets, target_resultants = _target_groups_for_band(
            fft_cache,
            trace_map_real,
            meta,
            source_rows,
            spec,
            normalization=str(args.phase_target_normalization),
            phase_target_mode=str(args.phase_target_mode),
        )
        row, axes, mean, sd, x_std_all = _fit_phase_readout(
            response_groups,
            targets,
            source_rows,
            train_sources,
            test_sources,
            ridge_alpha=float(args.ridge_alpha),
            predictor_label=f"neural_to_phase_{band_name}",
        )
        row["band_name"] = band_name
        row["n_complex_components"] = int(len(spec["indices"]))
        row["mean_cpd"] = float(spec["mean_cpd"])
        row["target_resultant_length_mean"] = float(np.mean(np.concatenate(target_resultants)))
        row["target_resultant_length_median"] = float(np.median(np.concatenate(target_resultants)))
        prediction_rows.append(row)
        trace_row = _fit_trace_baseline(
            trace_baseline_groups,
            targets,
            source_rows,
            train_sources,
            test_sources,
            ridge_alpha=float(args.ridge_alpha),
            predictor_label=f"dxdy_to_phase_{band_name}",
        )
        trace_row["band_name"] = band_name
        trace_row["n_complex_components"] = int(len(spec["indices"]))
        trace_row["mean_cpd"] = float(spec["mean_cpd"])
        trace_row["target_resultant_length_mean"] = row["target_resultant_length_mean"]
        trace_row["target_resultant_length_median"] = row["target_resultant_length_median"]
        prediction_rows.append(trace_row)
        shuffled_sources = _mismatched_source_rows(source_rows, seed=int(args.seed) + 119)
        shuffled, _shuffled_resultants = _target_groups_for_band(
            fft_cache,
            trace_map_real,
            meta,
            shuffled_sources,
            spec,
            normalization=str(args.phase_target_normalization),
            phase_target_mode=str(args.phase_target_mode),
        )
        shuffle_row, _sw, _sm, _ssd, _sx = _fit_phase_readout(
            response_groups,
            shuffled,
            source_rows,
            train_sources,
            test_sources,
            ridge_alpha=float(args.ridge_alpha),
            predictor_label=f"neural_to_image_shuffled_phase_{band_name}",
        )
        shuffle_row["band_name"] = band_name
        shuffle_row["target_resultant_length_mean"] = row["target_resultant_length_mean"]
        shuffle_row["target_resultant_length_median"] = row["target_resultant_length_median"]
        prediction_rows.append(shuffle_row)
        time_perm_targets, _time_resultants = _target_groups_for_band(
            fft_cache,
            trace_map_time_perm,
            meta,
            source_rows,
            spec,
            normalization=str(args.phase_target_normalization),
            phase_target_mode=str(args.phase_target_mode),
        )
        time_row, _tw, _tm, _tsd, _tx = _fit_phase_readout(
            response_groups,
            time_perm_targets,
            source_rows,
            train_sources,
            test_sources,
            ridge_alpha=float(args.ridge_alpha),
            predictor_label=f"neural_to_time_shuffled_phase_{band_name}",
        )
        time_row["band_name"] = band_name
        time_row["target_resultant_length_mean"] = row["target_resultant_length_mean"]
        time_row["target_resultant_length_median"] = row["target_resultant_length_median"]
        prediction_rows.append(time_row)
        train_mask = np.isin(
            _flatten_groups(response_groups, source_rows)[2],
            list(train_sources),
        )
        third = _third_axis_from_response_pc(x_std_all, train_mask, axes)
        projection = _phase_projection_groups(response_groups, mean, sd, axes, third)
        phase_fits[band_name] = {
            "targets": targets,
            "projection": projection,
            "axes": axes,
            "mean": mean,
            "sd": sd,
            "third_axis": third,
            "mean_r2": float(row["mean_r2"]),
            "phase_vector_cosine_mean": float(row["phase_vector_cosine_mean"]),
            "target_resultant_length_mean": row["target_resultant_length_mean"],
            "target_resultant_length_median": row["target_resultant_length_median"],
        }
    if not phase_fits:
        raise ValueError("No phase-aware fits were produced")
    available_bands = [name for name, _lo, _hi in bands if name in phase_fits]
    if str(args.phase_plot_band):
        if str(args.phase_plot_band) not in phase_fits:
            raise ValueError(f"--phase-plot-band {args.phase_plot_band!r} was not fit; available={available_bands}")
        plot_band = str(args.phase_plot_band)
    else:
        plot_band = available_bands[len(available_bands) // 2]
    plot_phase = phase_fits[plot_band]
    best_band_by_heldout_r2 = max(phase_fits, key=lambda name: float(phase_fits[name]["mean_r2"]))
    heldout_centroids = np.asarray(
        [np.mean(plot_phase["projection"][int(i)][:, :3], axis=0) for i in heldout_indices],
        dtype=np.float64,
    )
    heldout_selected_rel = _farthest_subset(
        heldout_centroids,
        min(int(args.max_trajectories), heldout_centroids.shape[0]),
        seed=int(args.seed) + 7,
    )
    heldout_selected = heldout_indices[heldout_selected_rel]
    _plot_3d_views(
        out_dir / "phase_aware_neural_projection_alt_views.png",
        plot_phase["projection"],
        meta,
        heldout_selected,
        title=f"Phase-aware neural projection, held-out sources, target {plot_band}",
        axis_labels=("neural phase axis 1", "neural phase axis 2", "orthogonal neural content axis"),
        equal_axes=False,
    )
    _copy_or_plot_single_view(
        out_dir / "phase_aware_neural_projection_alt_views.png",
        out_dir / "phase_aware_neural_projection_3d.png",
    )

    # C. Response-only rotational projection.
    rot_groups, _rot_evals, rot_info = _linear_rotational_projection(
        train_response_groups,
        response_groups,
        pca_components=int(args.rotation_pca_components),
    )
    rot_centroids = np.asarray([np.mean(group[:, :3], axis=0) for group in rot_groups], dtype=np.float64)
    rot_selected = _farthest_subset(rot_centroids, min(int(args.max_trajectories), len(rot_groups)), seed=int(args.seed) + 11)
    _plot_3d_views(
        out_dir / "response_only_rotational_projection_3d.png",
        rot_groups,
        meta,
        rot_selected,
        title="Response-only rotational projection from neural dynamics",
        axis_labels=("response-only rotation axis 1", "response-only rotation axis 2", "orthogonal neural axis"),
    )

    # Fourier positive control in component space, kept visually separate.
    pos_spec = selected_components[plot_band]
    fourier_groups = [
        _component_group(
            fft_cache[int(src)],
            trace_map_real[str(row["trajectory_id"])],
            list(pos_spec["indices"]),
            list(pos_spec["avg_power"]),
            normalization=str(args.phase_target_normalization),
        )
        for src, row in zip(source_rows, meta, strict=False)
    ]
    fourier_mean, fourier_basis, fourier_evals = _fit_pca(fourier_groups, n_components=max(3, int(args.n_components)))
    fourier_pca_groups = _project_groups(fourier_groups, fourier_mean, fourier_basis)
    fourier_var = fourier_evals / (float(np.sum(fourier_evals)) + 1e-12)
    _plot_3d_views(
        out_dir / "fourier_positive_control_3d.png",
        fourier_pca_groups,
        meta,
        selected_plot,
        title=f"Fourier positive control phase-component PCA ({plot_band})",
        axis_labels=(
            f"Fourier PC1 ({100*fourier_var[0]:.1f}%)",
            f"Fourier PC2 ({100*fourier_var[1]:.1f}%)",
            f"Fourier PC3 ({100*fourier_var[2]:.1f}%)",
        ),
    )

    heldout_response_pca = _subset_groups(response_pca_groups, heldout_indices)
    heldout_phase_projection = _subset_groups(plot_phase["projection"], heldout_indices)
    heldout_phase_targets = _subset_groups(plot_phase["targets"], heldout_indices)
    heldout_rot = _subset_groups(rot_groups, heldout_indices)
    heldout_fourier = _subset_groups(fourier_pca_groups, heldout_indices)
    metric_rows = [
        _projection_metrics("response_pca_heldout", heldout_response_pca),
        _projection_metrics(f"phase_aware_neural_{plot_band}_heldout", heldout_phase_projection, heldout_phase_targets),
        _projection_metrics("response_only_rotational_heldout", heldout_rot),
        _projection_metrics(f"fourier_positive_control_{plot_band}_heldout", heldout_fourier),
    ]
    control_rows = []
    for row_name, groups, targets in (
        ("response_pca_heldout", heldout_response_pca, None),
        (f"phase_aware_neural_{plot_band}_heldout", heldout_phase_projection, heldout_phase_targets),
        ("response_only_rotational_heldout", heldout_rot, None),
        (f"fourier_positive_control_{plot_band}_heldout", heldout_fourier, None),
    ):
        control_rows.append(_projection_metrics(f"{row_name}_real", groups, targets))
        control_rows.append(_projection_metrics(f"{row_name}_time_shuffle", _time_shuffle_groups(groups, int(args.seed) + 23), targets))
    _write_csv(out_dir / "projection_metrics.csv", metric_rows)
    _write_csv(out_dir / "heldout_phase_prediction.csv", prediction_rows)
    _write_csv(out_dir / "circularity_controls.csv", control_rows)
    _plot_metric_summary(out_dir / "metric_summary.png", metric_rows)

    readme = f"""# Phase-Aware Response Geometry Pilot

Cache used: `{run_dir}`

Response variant plotted: `{args.variant}` from the V1 digital twin response cache.
The main neural coordinates use model response vectors only. Fourier phase is used only to define held-out phase targets for selecting the first two neural axes, plus the visually separate Fourier positive control.

Train/test split: source-held-out, {len(train_sources)} train fixation sources and {len(test_sources)} held-out fixation sources.

Phase target mode: `{args.phase_target_mode}`.
Representative phase-aware plot band: `{plot_band}`. This is not selected by held-out score.
Best band by held-out mean R2, reported for exploration only: `{best_band_by_heldout_r2}`.

Axes:
- `response_pca_3d*.png`: blind PCA of neural/model response trajectories.
- `phase_aware_neural_projection*.png`: ridge readout axes from neural responses to Fourier phase targets; plotted trajectories are held-out neural responses.
- `response_only_rotational_projection_3d.png`: axes discovered from neural response dynamics only, using the leading skew-symmetric rotational plane.
- `fourier_positive_control_3d.png`: Fourier component trajectories, included only as a positive control.

Caveats:
- This is a visualization pilot, not a mechanistic claim.
- Fourier phase targets are useful interpretive labels, but circular structure in Fourier coordinates is partly guaranteed by the translation phase-shift theorem.
- For `relative_to_zero`, `relative_to_start`, and `step_advance` targets, image-specific Fourier base phase is removed. With unit phase targets, source/image shuffles can therefore be weak controls or no-ops; the useful baseline is the matched dx/dy eye-trace readout.
- Held-out phase prediction and shuffle controls should be used to judge whether the phase-aware neural axes generalize.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "variant": str(args.variant),
        "n_manifest_rows_selected": int(len(manifest_rows)),
        "n_trajectory_groups": int(len(response_groups)),
        "n_sources": int(len(set(source_rows))),
        "n_train_sources": int(len(train_sources)),
        "n_test_sources": int(len(test_sources)),
        "phase_plot_band": plot_band,
        "best_phase_band_by_heldout_r2": best_band_by_heldout_r2,
        "phase_target_mode": str(args.phase_target_mode),
        "relative_phase_caveat": (
            "relative phase targets remove image-specific Fourier base phase; with unit phase targets, "
            "source/image shuffles may be identical or nearly identical to real targets"
        ),
        "response_pca_explained_fraction_first10": [float(v) for v in pca_var[:10].tolist()],
        "response_pca_participation_ratio": _participation_ratio(pca_evals),
        "fourier_positive_control_band": plot_band,
        "rotation_info": rot_info,
        "outputs": [
            "response_pca_3d.png",
            "response_pca_3d_alt_views.png",
            "phase_aware_neural_projection_3d.png",
            "phase_aware_neural_projection_alt_views.png",
            "response_only_rotational_projection_3d.png",
            "fourier_positive_control_3d.png",
            "metric_summary.png",
            "projection_metrics.csv",
            "heldout_phase_prediction.csv",
            "circularity_controls.csv",
            "README.md",
        ],
    }
    _write_json(out_dir / "pilot_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pilot phase-aware visualization of FEM-driven neural response geometry.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-set-modes", type=str, default="")
    parser.add_argument("--prior-families", type=str, default="")
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--variant", choices=("motion_delta", "prior_response"), default="motion_delta")
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--component-crop-px", type=int, default=151)
    parser.add_argument("--fourier-cache", type=Path, default=None)
    parser.add_argument("--frequency-bands", type=str, default="2:4,4:8,8:16")
    parser.add_argument("--components-per-band", type=int, default=16)
    parser.add_argument(
        "--phase-target-normalization",
        choices=("unit_amplitude", "power_whitened", "raw"),
        default="unit_amplitude",
    )
    parser.add_argument(
        "--phase-target-mode",
        choices=("relative_to_zero", "relative_to_start", "step_advance", "absolute"),
        default="relative_to_zero",
        help=(
            "Fourier phase target frame: relative_to_zero matches prior_response-zero_static motion_delta; "
            "relative_to_start uses trace onset; step_advance uses adjacent time increments; absolute keeps image phase."
        ),
    )
    parser.add_argument(
        "--phase-plot-band",
        type=str,
        default="",
        help="Representative band to plot, e.g. '4-8cpd'. Empty uses the middle requested band.",
    )
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--rotation-pca-components", type=int, default=12)
    parser.add_argument("--max-trajectories", type=int, default=32)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
