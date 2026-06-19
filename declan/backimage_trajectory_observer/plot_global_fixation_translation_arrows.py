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
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT
from declan.backimage_trajectory_observer.analyze_global_fixation_geometry import (
    _manifest_filter,
    _read_csv_rows,
    _safe_float,
    _string_array,
)


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


def _sample_rows(arr: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if int(max_rows) > 0 and x.shape[0] > int(max_rows):
        idx = np.sort(rng.choice(x.shape[0], size=int(max_rows), replace=False))
        x = x[idx]
    return x


def _basis_points(table: Any, basis_source: str) -> np.ndarray:
    prior = np.asarray(table["prior_lambda_counts"], dtype=np.float32)
    zero = np.asarray(table["zero_lambda_counts"], dtype=np.float32)
    if basis_source == "state_union":
        return np.concatenate([prior.reshape(-1, prior.shape[-1]), zero.reshape(-1, zero.shape[-1])], axis=0)
    if basis_source == "prior_response":
        return prior.reshape(-1, prior.shape[-1])
    if basis_source == "motion_delta":
        return (prior - zero[:, None, :, :]).reshape(-1, prior.shape[-1])
    if basis_source == "zero_static":
        return zero.reshape(-1, zero.shape[-1])
    raise ValueError(f"Unknown basis_source={basis_source!r}")


def _fit_basis(
    run_dir: Path,
    manifest_rows: list[dict[str, str]],
    *,
    basis_source: str,
    max_fit_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    chunks: list[np.ndarray] = []

    def _trim(force: bool = False) -> None:
        if int(max_fit_points) <= 0 or not chunks:
            return
        total = sum(int(c.shape[0]) for c in chunks)
        if not force and total <= int(max_fit_points) * 2:
            return
        merged = np.concatenate(chunks, axis=0)
        if merged.shape[0] > int(max_fit_points):
            idx = np.sort(rng.choice(merged.shape[0], size=int(max_fit_points), replace=False))
            merged = merged[idx]
        chunks[:] = [merged]

    for row in manifest_rows:
        table_path = run_dir / str(row["response_cache_path"])
        if not table_path.exists():
            continue
        table = np.load(table_path, allow_pickle=True)
        pts = _basis_points(table, basis_source)
        chunks.append(_sample_rows(pts, max(1, int(max_fit_points) // max(len(manifest_rows), 1) * 2), rng))
        _trim(force=False)
    if not chunks:
        raise ValueError("No response tables could be loaded for PCA basis")
    _trim(force=True)
    x = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
    mean = np.mean(x, axis=0)
    xc = x - mean[None, :]
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    evals, vecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return mean, vecs[:, order[:3]], np.maximum(evals[order], 0.0)


def _project(arr: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    return (x - mean[None, :]) @ basis


def _color_values(rows: list[dict[str, object]], color_by: str) -> np.ndarray:
    if color_by == "arrow_length":
        return np.asarray([_safe_float(row.get("arrow_length_pc")) for row in rows], dtype=np.float64)
    vals = np.asarray([_safe_float(row.get(color_by)) for row in rows], dtype=np.float64)
    if np.isfinite(vals).any():
        return vals
    labels = sorted({str(row.get(color_by, "")) for row in rows})
    lookup = {label: idx for idx, label in enumerate(labels)}
    return np.asarray([lookup[str(row.get(color_by, ""))] for row in rows], dtype=np.float64)


def _candidate_ids(table: Any, n_candidates: int) -> list[str]:
    return _string_array(table, "candidate_ids", int(n_candidates), "candidate:")


def _trajectory_id(table: Any, candidate_index: int, trajectory_index: int, n_candidates: int, n_traj: int) -> str:
    if "prior_trajectory_ids" not in table:
        return f"trajectory:{int(trajectory_index)}"
    arr = np.asarray(table["prior_trajectory_ids"])
    c = int(candidate_index)
    k = int(trajectory_index)
    if arr.ndim >= 2 and arr.shape[0] > c and arr.shape[1] > k:
        return str(arr[c, k])
    flat = arr.reshape(-1)
    if flat.size == int(n_traj) and k < flat.size:
        return str(flat[k])
    flat_index = c * int(n_traj) + k
    if flat.size == int(n_candidates) * int(n_traj) and flat_index < flat.size:
        return str(flat[flat_index])
    return f"trajectory:{k}"


def _safe_int(value: object, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _screen_axis_class(output_axis_deg: float, dominance_margin: float) -> str:
    if not np.isfinite(output_axis_deg):
        return "unknown"
    theta = np.deg2rad(float(output_axis_deg))
    x_weight = abs(float(np.cos(theta)))
    y_weight = abs(float(np.sin(theta)))
    dominance = x_weight - y_weight
    if abs(dominance) < float(dominance_margin):
        return "diagonal"
    return "x_dominant" if dominance >= 0.0 else "y_dominant"


def _load_axis_lookup(run_dir: Path) -> dict[tuple[int, int, str], dict[str, object]]:
    catalog_path = run_dir / "axis_trajectory_catalog.csv"
    if not catalog_path.exists():
        catalog_path = run_dir / "motion_catalog.csv"
    if not catalog_path.exists():
        return {}
    lookup: dict[tuple[int, int, str], dict[str, object]] = {}
    for row in _read_csv_rows(catalog_path):
        if str(row.get("role", "prior")) not in {"", "prior"}:
            continue
        trajectory_id = str(row.get("trajectory_id", ""))
        trial_id = _safe_int(row.get("trial_id"), -1)
        candidate_index = _safe_int(row.get("candidate_index", row.get("axis_candidate_index")), -1)
        if trial_id < 0 or candidate_index < 0 or not trajectory_id:
            continue
        output_axis_deg = _safe_float(row.get("output_axis_deg", row.get("axis_deg")))
        theta = np.deg2rad(float(output_axis_deg)) if np.isfinite(output_axis_deg) else float("nan")
        x_weight = abs(float(np.cos(theta))) if np.isfinite(theta) else float("nan")
        y_weight = abs(float(np.sin(theta))) if np.isfinite(theta) else float("nan")
        lookup[(trial_id, candidate_index, trajectory_id)] = {
            "trajectory_family": str(row.get("family", "")),
            "axis_relation": str(row.get("axis_relation", "")),
            "axis_deg": _safe_float(row.get("axis_deg")),
            "output_axis_deg": output_axis_deg,
            "screen_axis_x_weight": x_weight,
            "screen_axis_y_weight": y_weight,
            "screen_axis_dominance": x_weight - y_weight if np.isfinite(x_weight) and np.isfinite(y_weight) else float("nan"),
            "axis_candidate_id": str(row.get("axis_candidate_id", row.get("candidate_id", ""))),
        }
    return lookup


def _annotate_axis_metadata(
    rows: list[dict[str, object]],
    axis_lookup: dict[tuple[int, int, str], dict[str, object]],
    *,
    dominance_margin: float,
) -> None:
    for row in rows:
        key = (
            _safe_int(row.get("trial_id"), -1),
            _safe_int(row.get("candidate_index"), -1),
            str(row.get("trajectory_id", "")),
        )
        meta = axis_lookup.get(key, {})
        output_axis_deg = _safe_float(meta.get("output_axis_deg"))
        row.update(
            {
                "trajectory_family": meta.get("trajectory_family", ""),
                "axis_relation": meta.get("axis_relation", ""),
                "axis_deg": _safe_float(meta.get("axis_deg")),
                "output_axis_deg": output_axis_deg,
                "screen_axis_class": _screen_axis_class(output_axis_deg, float(dominance_margin)),
                "screen_axis_x_weight": _safe_float(meta.get("screen_axis_x_weight")),
                "screen_axis_y_weight": _safe_float(meta.get("screen_axis_y_weight")),
                "screen_axis_dominance": _safe_float(meta.get("screen_axis_dominance")),
                "axis_candidate_id": meta.get("axis_candidate_id", ""),
            }
        )


def _subset_arrows(
    starts: np.ndarray,
    ends: np.ndarray,
    rows: list[dict[str, object]],
    *,
    screen_axis_class: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    idx = [i for i, row in enumerate(rows) if str(row.get("screen_axis_class", "")) == screen_axis_class]
    if not idx:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64), []
    arr_idx = np.asarray(idx, dtype=np.int64)
    return starts[arr_idx], ends[arr_idx], [rows[i] for i in idx]


def _collect_translation_arrows(
    run_dir: Path,
    manifest_rows: list[dict[str, str]],
    *,
    mean: np.ndarray,
    basis: np.ndarray,
    max_arrows: int,
    seed: int,
    arrow_gain: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    rng = np.random.default_rng(int(seed))
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    per_table = max(1, int(np.ceil(max_arrows / max(len(manifest_rows), 1))))
    for table_index, row in enumerate(manifest_rows):
        if len(rows) >= int(max_arrows):
            break
        table_path = run_dir / str(row["response_cache_path"])
        if not table_path.exists():
            continue
        table = np.load(table_path, allow_pickle=True)
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        n_candidates, n_traj, n_time, _n_units = prior.shape
        n_possible = n_candidates * n_traj * n_time
        n_pick = min(per_table, n_possible, int(max_arrows) - len(rows))
        picks = rng.choice(n_possible, size=n_pick, replace=False)
        candidate_ids = _candidate_ids(table, n_candidates)
        for pick in picks:
            rem = int(pick)
            t = rem % n_time
            rem //= n_time
            k = rem % n_traj
            c = rem // n_traj
            start = _project(zero[c, t][None, :], mean, basis)[0]
            raw_end = _project(prior[c, k, t][None, :], mean, basis)[0]
            end = start + float(arrow_gain) * (raw_end - start)
            starts.append(start)
            ends.append(end)
            rows.append(
                {
                    "arrow_type": "zero_to_translated",
                    "table_index": int(table_index),
                    "trial_id": int(_safe_float(row.get("trial_id"), table_index)),
                    "candidate_id": candidate_ids[c],
                    "candidate_index": int(c),
                    "trajectory_id": _trajectory_id(table, c, k, n_candidates, n_traj),
                    "trajectory_index": int(k),
                    "time_index": int(t),
                    "prior_family": str(row.get("prior_family", "")),
                    "scale": _safe_float(row.get("scale")),
                    "start_pc1": float(start[0]),
                    "start_pc2": float(start[1]),
                    "start_pc3": float(start[2]),
                    "end_pc1": float(end[0]),
                    "end_pc2": float(end[1]),
                    "end_pc3": float(end[2]),
                    "arrow_length_pc": float(np.linalg.norm(raw_end - start)),
                }
            )
    return np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64), rows


def _collect_step_arrows(
    run_dir: Path,
    manifest_rows: list[dict[str, str]],
    *,
    mean: np.ndarray,
    basis: np.ndarray,
    max_arrows: int,
    seed: int,
    arrow_gain: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    rng = np.random.default_rng(int(seed))
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    per_table = max(1, int(np.ceil(max_arrows / max(len(manifest_rows), 1))))
    for table_index, row in enumerate(manifest_rows):
        if len(rows) >= int(max_arrows):
            break
        table_path = run_dir / str(row["response_cache_path"])
        if not table_path.exists():
            continue
        table = np.load(table_path, allow_pickle=True)
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        n_candidates, n_traj, n_time, _n_units = prior.shape
        if n_time < 2:
            continue
        n_possible = n_candidates * n_traj * (n_time - 1)
        n_pick = min(per_table, n_possible, int(max_arrows) - len(rows))
        picks = rng.choice(n_possible, size=n_pick, replace=False)
        candidate_ids = _candidate_ids(table, n_candidates)
        for pick in picks:
            rem = int(pick)
            t = rem % (n_time - 1)
            rem //= n_time - 1
            k = rem % n_traj
            c = rem // n_traj
            start = _project(prior[c, k, t][None, :], mean, basis)[0]
            raw_end = _project(prior[c, k, t + 1][None, :], mean, basis)[0]
            end = start + float(arrow_gain) * (raw_end - start)
            starts.append(start)
            ends.append(end)
            rows.append(
                {
                    "arrow_type": "trajectory_step_t_to_tplus1",
                    "table_index": int(table_index),
                    "trial_id": int(_safe_float(row.get("trial_id"), table_index)),
                    "candidate_id": candidate_ids[c],
                    "candidate_index": int(c),
                    "trajectory_id": _trajectory_id(table, c, k, n_candidates, n_traj),
                    "trajectory_index": int(k),
                    "time_index": int(t),
                    "prior_family": str(row.get("prior_family", "")),
                    "scale": _safe_float(row.get("scale")),
                    "start_pc1": float(start[0]),
                    "start_pc2": float(start[1]),
                    "start_pc3": float(start[2]),
                    "end_pc1": float(end[0]),
                    "end_pc2": float(end[1]),
                    "end_pc3": float(end[2]),
                    "arrow_length_pc": float(np.linalg.norm(raw_end - start)),
                }
            )
    return np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64), rows


def _axis_limits(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.concatenate([starts, ends], axis=0)
    lo = np.nanpercentile(pts, 1, axis=0)
    hi = np.nanpercentile(pts, 99, axis=0)
    center = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    return center, span


def _plot_static_arrows(
    starts: np.ndarray,
    ends: np.ndarray,
    rows: list[dict[str, object]],
    *,
    title: str,
    color_by: str,
    out_path: Path,
) -> None:
    if starts.size == 0 or ends.size == 0:
        return
    colors = _color_values(rows, color_by)
    delta = ends - starts
    center, span = _axis_limits(starts, ends)
    fig = plt.figure(figsize=(13.0, 10.0), constrained_layout=True)
    views = [(18, -60), (18, 35), (55, -60), (8, -100)]
    for idx, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        sc = ax.scatter(
            starts[:, 0],
            starts[:, 1],
            starts[:, 2],
            c=colors,
            s=7,
            cmap="viridis",
            alpha=0.42,
            linewidths=0,
            depthshade=False,
        )
        ax.quiver(
            starts[:, 0],
            starts[:, 1],
            starts[:, 2],
            delta[:, 0],
            delta[:, 1],
            delta[:, 2],
            length=1.0,
            normalize=False,
            arrow_length_ratio=0.22,
            linewidth=0.5,
            color="0.12",
            alpha=0.42,
        )
        ax.view_init(elev=elev, azim=azim)
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, strict=False):
            setter(float(c - 0.56 * span), float(c + 0.56 * span))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"elev {elev}, azim {azim}", fontsize=9)
    fig.suptitle(title, fontsize=13)
    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.74, pad=0.02)
    cbar.set_label(color_by)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_interactive_arrows(
    starts: np.ndarray,
    ends: np.ndarray,
    rows: list[dict[str, object]],
    *,
    title: str,
    color_by: str,
    out_path: Path,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False
    colors = _color_values(rows, color_by)
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    hover: list[str | None] = []
    for start, end, row in zip(starts, ends, rows, strict=False):
        text = (
            f"{row.get('arrow_type')}<br>"
            f"candidate={row.get('candidate_id')}<br>"
            f"trajectory={row.get('trajectory_index')}<br>"
            f"time={row.get('time_index')}<br>"
            f"length={float(row.get('arrow_length_pc', 0.0)):.4g}"
        )
        x.extend([float(start[0]), float(end[0]), None])
        y.extend([float(start[1]), float(end[1]), None])
        z.extend([float(start[2]), float(end[2]), None])
        hover.extend([text, text, None])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=starts[:, 0],
            y=starts[:, 1],
            z=starts[:, 2],
            mode="markers",
            marker={"size": 2.8, "color": colors, "colorscale": "Viridis", "opacity": 0.62},
            text=[str(row.get("candidate_id", "")) for row in rows],
            hoverinfo="text",
            name="arrow starts",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line={"color": "rgba(20,20,20,0.45)", "width": 2.0},
            text=hover,
            hoverinfo="text",
            name="arrows",
        )
    )
    fig.update_layout(
        title=title,
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3", "aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return True


def _metric_row(
    rows: list[dict[str, object]],
    *,
    arrow_type: str,
    basis_source: str,
    screen_axis_class: str = "all",
) -> dict[str, object]:
    lengths = np.asarray([_safe_float(row.get("arrow_length_pc")) for row in rows], dtype=np.float64)
    vals = lengths[np.isfinite(lengths)]
    return {
        "arrow_type": arrow_type,
        "basis_source": basis_source,
        "screen_axis_class": screen_axis_class,
        "n_arrows": int(len(rows)),
        "arrow_length_pc_median": float(np.median(vals)) if vals.size else float("nan"),
        "arrow_length_pc_mean": float(np.mean(vals)) if vals.size else float("nan"),
        "arrow_length_pc_ci_low": float(np.percentile(vals, 2.5)) if vals.size else float("nan"),
        "arrow_length_pc_ci_high": float(np.percentile(vals, 97.5)) if vals.size else float("nan"),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    manifest_rows = _manifest_filter(_read_csv_rows(run_dir / "response_cache_manifest.csv"), args)
    if not manifest_rows:
        raise ValueError("No response-cache rows selected")
    mean, basis, evals = _fit_basis(
        run_dir,
        manifest_rows,
        basis_source=str(args.basis_source),
        max_fit_points=int(args.max_fit_points),
        seed=int(args.seed),
    )
    axis_lookup = _load_axis_lookup(run_dir)
    metric_rows: list[dict[str, object]] = []
    arrow_rows: list[dict[str, object]] = []
    outputs: list[str] = []
    for arrow_kind in [v.strip() for v in str(args.arrow_types).split(",") if v.strip()]:
        if arrow_kind == "translation":
            starts, ends, rows = _collect_translation_arrows(
                run_dir,
                manifest_rows,
                mean=mean,
                basis=basis,
                max_arrows=int(args.max_arrows),
                seed=int(args.seed) + 10,
                arrow_gain=float(args.arrow_gain),
            )
            arrow_type = "zero_to_translated"
        elif arrow_kind == "trajectory_step":
            starts, ends, rows = _collect_step_arrows(
                run_dir,
                manifest_rows,
                mean=mean,
                basis=basis,
                max_arrows=int(args.max_arrows),
                seed=int(args.seed) + 20,
                arrow_gain=float(args.arrow_gain),
            )
            arrow_type = "trajectory_step_t_to_tplus1"
        else:
            raise ValueError(f"Unknown arrow type {arrow_kind!r}")
        _annotate_axis_metadata(rows, axis_lookup, dominance_margin=float(args.axis_dominance_margin))
        arrow_rows.extend(rows)
        metric_rows.append(
            _metric_row(rows, arrow_type=arrow_type, basis_source=str(args.basis_source), screen_axis_class="all")
        )
        for color_by in [v.strip() for v in str(args.color_by).split(",") if v.strip()]:
            stem = f"{arrow_kind}_arrows_{args.basis_source}_by_{color_by}"
            title = f"{arrow_type} in {args.basis_source} PC space, colored by {color_by}"
            png_path = out_dir / f"{stem}.png"
            html_path = out_dir / f"{stem}.html"
            _plot_static_arrows(starts, ends, rows, title=title, color_by=color_by, out_path=png_path)
            outputs.extend([str(png_path), str(png_path.with_suffix(".pdf"))])
            if _plot_interactive_arrows(starts, ends, rows, title=title, color_by=color_by, out_path=html_path):
                outputs.append(str(html_path))
        if bool(args.split_screen_axes):
            for screen_axis_class in ("x_dominant", "y_dominant"):
                split_starts, split_ends, split_rows = _subset_arrows(
                    starts, ends, rows, screen_axis_class=screen_axis_class
                )
                metric_rows.append(
                    _metric_row(
                        split_rows,
                        arrow_type=arrow_type,
                        basis_source=str(args.basis_source),
                        screen_axis_class=screen_axis_class,
                    )
                )
                for color_by in [v.strip() for v in str(args.color_by).split(",") if v.strip()]:
                    stem = f"{arrow_kind}_arrows_{args.basis_source}_{screen_axis_class}_by_{color_by}"
                    label = screen_axis_class.replace("_", "-")
                    title = (
                        f"{arrow_type} in {args.basis_source} PC space, "
                        f"{label} cached trajectory axes, colored by {color_by}"
                    )
                    png_path = out_dir / f"{stem}.png"
                    html_path = out_dir / f"{stem}.html"
                    _plot_static_arrows(
                        split_starts,
                        split_ends,
                        split_rows,
                        title=title,
                        color_by=color_by,
                        out_path=png_path,
                    )
                    if split_rows:
                        outputs.extend([str(png_path), str(png_path.with_suffix(".pdf"))])
                        if _plot_interactive_arrows(
                            split_starts,
                            split_ends,
                            split_rows,
                            title=title,
                            color_by=color_by,
                            out_path=html_path,
                        ):
                            outputs.append(str(html_path))
    _write_csv(out_dir / "translation_arrow_points.csv", arrow_rows)
    _write_csv(out_dir / "translation_arrow_metrics.csv", metric_rows)
    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "n_manifest_rows_selected": int(len(manifest_rows)),
        "basis_source": str(args.basis_source),
        "basis_top3_eigenvalues": [float(v) for v in evals[:3]],
        "basis_top3_fraction": [float(v / (np.sum(evals) + 1e-12)) for v in evals[:3]],
        "arrow_types": [v.strip() for v in str(args.arrow_types).split(",") if v.strip()],
        "arrow_gain": float(args.arrow_gain),
        "max_arrows": int(args.max_arrows),
        "axis_catalog_rows_loaded": int(len(axis_lookup)),
        "split_screen_axes": bool(args.split_screen_axes),
        "axis_dominance_margin": float(args.axis_dominance_margin),
        "outputs": outputs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "translation_arrow_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot arrows showing cached BackImage responses under translation trajectories.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidate-set-modes", type=str, default="")
    p.add_argument("--prior-families", type=str, default="")
    p.add_argument("--scales", type=str, default="")
    p.add_argument("--axis-catalog-modes", type=str, default="")
    p.add_argument("--max-tables", type=int, default=0)
    p.add_argument(
        "--basis-source",
        choices=("state_union", "prior_response", "motion_delta", "zero_static"),
        default="state_union",
    )
    p.add_argument("--max-fit-points", type=int, default=50000)
    p.add_argument("--arrow-types", type=str, default="translation,trajectory_step")
    p.add_argument("--max-arrows", type=int, default=600)
    p.add_argument("--arrow-gain", type=float, default=1.0)
    p.add_argument("--color-by", type=str, default="time_index,trajectory_index,arrow_length_pc")
    p.add_argument(
        "--split-screen-axes",
        action="store_true",
        help="Also write separate plots for cached x-dominant and y-dominant translation-axis trajectories.",
    )
    p.add_argument(
        "--axis-dominance-margin",
        type=float,
        default=0.0,
        help="Minimum |abs(cos(axis))-abs(sin(axis))| required before assigning a cached trajectory to x/y.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
