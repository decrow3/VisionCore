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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _parse_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _manifest_filter(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    candidate_modes = set(_parse_list(args.candidate_set_modes))
    prior_families = set(_parse_list(args.prior_families))
    scales = set(_parse_list(args.scales))
    keep = []
    for row in rows:
        if candidate_modes and str(row.get("candidate_set_mode", "")) not in candidate_modes:
            continue
        if prior_families and str(row.get("prior_family", "")) not in prior_families:
            continue
        if scales and str(row.get("scale", "")) not in scales:
            continue
        keep.append(row)
    if int(args.max_tables) > 0:
        keep = keep[: int(args.max_tables)]
    return keep


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


def _variant_array(table: Any, variant: str) -> np.ndarray:
    prior = np.asarray(table["prior_lambda_counts"], dtype=np.float32)
    if variant == "prior_response":
        return prior
    if variant == "motion_delta":
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float32)
        return prior - zero[:, None, :, :]
    raise ValueError(f"Trajectory-line plotting expects a 4D trajectory variant, got {variant!r}")


def _string_array(table: Any, key: str, n: int, prefix: str) -> list[str]:
    if key not in table:
        return [f"{prefix}{i}" for i in range(int(n))]
    values = np.asarray(table[key]).astype(str).tolist()
    if len(values) < int(n):
        values.extend(f"{prefix}{i}" for i in range(len(values), int(n)))
    return [str(v) for v in values[: int(n)]]


def _collect_trajectories(
    run_dir: Path,
    manifest_rows: list[dict[str, str]],
    *,
    variant: str,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    groups: list[np.ndarray] = []
    meta: list[dict[str, object]] = []
    for table_index, row in enumerate(manifest_rows):
        table_path = run_dir / str(row["response_cache_path"])
        if not table_path.exists():
            continue
        table = np.load(table_path, allow_pickle=True)
        arr = _variant_array(table, variant)
        if arr.ndim != 4:
            continue
        n_candidate, n_traj, n_time, _n_units = arr.shape
        candidate_ids = _string_array(table, "candidate_ids", n_candidate, "candidate:")
        trajectory_ids = _string_array(table, "prior_trajectory_ids", n_traj, "trajectory:")
        for ci in range(n_candidate):
            for ti in range(n_traj):
                x = np.asarray(arr[ci, ti], dtype=np.float32)
                if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
                    continue
                groups.append(x)
                candidate_id = candidate_ids[ci] if ci < len(candidate_ids) else f"candidate:{ci}"
                source_row = _safe_float(str(candidate_id).split(":")[-1])
                meta.append(
                    {
                        "table_index": int(table_index),
                        "trial_id": int(_safe_float(row.get("trial_id"), table_index)),
                        "candidate_id": candidate_id,
                        "candidate_index": int(ci),
                        "trajectory_id": trajectory_ids[ti] if ti < len(trajectory_ids) else f"trajectory:{ti}",
                        "trajectory_index": int(ti),
                        "source_row": source_row,
                        "candidate_set_mode": str(row.get("candidate_set_mode", "")),
                        "prior_family": str(row.get("prior_family", "")),
                        "scale": _safe_float(row.get("scale")),
                    }
                )
    return groups, meta


def _fit_pca_streamed(groups: list[np.ndarray], n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not groups:
        raise ValueError("Need at least one trajectory group")
    n_units = int(groups[0].shape[1])
    total_n = 0
    total_sum = np.zeros(n_units, dtype=np.float64)
    for group in groups:
        if group.shape[1] != n_units:
            raise ValueError("All groups must have the same response dimension")
        total_n += int(group.shape[0])
        total_sum += np.sum(group, axis=0, dtype=np.float64)
    if total_n < 3:
        raise ValueError(f"Need at least 3 points for PCA, got {total_n}")
    mean = total_sum / float(total_n)
    cov = np.zeros((n_units, n_units), dtype=np.float64)
    for group in groups:
        xc = np.asarray(group, dtype=np.float64) - mean[None, :]
        cov += xc.T @ xc
    cov /= float(max(total_n - 1, 1))
    evals, vecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    vecs = vecs[:, order]
    k = min(max(1, int(n_components)), vecs.shape[1])
    return mean.astype(np.float32), vecs[:, :k].astype(np.float32), evals.astype(np.float32)


def _project_group(group: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(group, dtype=np.float32) - mean[None, :]) @ basis


def _scaled_groups(groups: list[np.ndarray], pc3_scale: float) -> list[np.ndarray]:
    scale = float(pc3_scale)
    out = []
    for group in groups:
        arr = np.asarray(group, dtype=np.float32).copy()
        if arr.shape[1] >= 3:
            arr[:, 2] *= scale
        out.append(arr)
    return out


def _axis_limits(coords: np.ndarray) -> tuple[np.ndarray, float]:
    lo = np.nanpercentile(coords, 1, axis=0)
    hi = np.nanpercentile(coords, 99, axis=0)
    center = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    return center, span


def _set_axis_limits(ax: Any, center: np.ndarray, span: float) -> None:
    for setter, val in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, strict=False):
        setter(float(val - 0.55 * span), float(val + 0.55 * span))


def _farthest_subset(points: np.ndarray, n_keep: int, seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(pts).all(axis=1))
    if valid.size <= int(n_keep):
        return valid
    rng = np.random.default_rng(int(seed))
    work = pts[valid]
    start_score = np.linalg.norm(work - np.median(work, axis=0, keepdims=True), axis=1)
    chosen = [int(np.argmax(start_score + 1e-9 * rng.normal(size=work.shape[0])))]
    min_dist = np.linalg.norm(work - work[chosen[0]], axis=1)
    for _ in range(1, int(n_keep)):
        nxt = int(np.argmax(min_dist + 1e-9 * rng.normal(size=work.shape[0])))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(work - work[nxt], axis=1))
    return valid[np.asarray(chosen, dtype=np.int64)]


def _color_values(groups: list[np.ndarray], meta: list[dict[str, object]], color_by: str) -> tuple[np.ndarray, str]:
    if color_by == "time_index":
        vals = np.concatenate([np.arange(group.shape[0], dtype=np.float64) for group in groups])
        return vals, "time index"
    if color_by == "trajectory_index":
        vals = np.concatenate(
            [np.full(group.shape[0], _safe_float(row.get("trajectory_index")), dtype=np.float64) for group, row in zip(groups, meta, strict=False)]
        )
        return vals, "trajectory index"
    if color_by == "source_row":
        vals = np.concatenate(
            [np.full(group.shape[0], _safe_float(row.get("source_row")), dtype=np.float64) for group, row in zip(groups, meta, strict=False)]
        )
        return vals, "source row"
    if color_by == "candidate_index":
        vals = np.concatenate(
            [np.full(group.shape[0], _safe_float(row.get("candidate_index")), dtype=np.float64) for group, row in zip(groups, meta, strict=False)]
        )
        return vals, "candidate index"
    vals = np.arange(sum(group.shape[0] for group in groups), dtype=np.float64)
    return vals, color_by


def _segment_arrays(
    groups: list[np.ndarray],
    meta: list[dict[str, object]],
    *,
    color_by: str,
    stride: int,
    normalize_arrows: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    starts: list[np.ndarray] = []
    vecs: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    step = max(1, int(stride))
    for group, row in zip(groups, meta, strict=False):
        if group.shape[0] < 2:
            continue
        start = np.asarray(group[:-1:step, :3], dtype=np.float64)
        stop = np.asarray(group[1::step, :3], dtype=np.float64)
        n = min(start.shape[0], stop.shape[0])
        if n <= 0:
            continue
        start = start[:n]
        vec = stop[:n] - start
        if bool(normalize_arrows):
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            vec = vec / (norm + 1e-12)
        starts.append(start)
        vecs.append(vec)
        if color_by == "time_index":
            vals.append(np.arange(0, n * step, step, dtype=np.float64))
        elif color_by == "trajectory_index":
            vals.append(np.full(n, _safe_float(row.get("trajectory_index")), dtype=np.float64))
        elif color_by == "source_row":
            vals.append(np.full(n, _safe_float(row.get("source_row")), dtype=np.float64))
        elif color_by == "candidate_index":
            vals.append(np.full(n, _safe_float(row.get("candidate_index")), dtype=np.float64))
        else:
            vals.append(np.arange(n, dtype=np.float64))
    if not starts:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            color_by,
        )
    labels = {
        "time_index": "time index",
        "trajectory_index": "trajectory index",
        "source_row": "source row",
        "candidate_index": "candidate index",
    }
    return np.concatenate(starts), np.concatenate(vecs), np.concatenate(vals), labels.get(color_by, color_by)


def _arrow_length(normalize_arrows: bool, span: float) -> float:
    if bool(normalize_arrows):
        return float(max(0.025 * span, 1e-6))
    return 1.0


def _static_plot(
    out_path: Path,
    projected_groups: list[np.ndarray],
    meta: list[dict[str, object]],
    selected: np.ndarray,
    *,
    color_by: str,
    var_fraction: np.ndarray,
    title: str,
    pc3_scale: float,
    plot_style: str,
    arrow_stride: int,
    normalize_arrows: bool,
) -> None:
    groups = _scaled_groups([projected_groups[int(i)] for i in selected], pc3_scale)
    rows = [meta[int(i)] for i in selected]
    coords = np.concatenate(groups, axis=0)
    center, span = _axis_limits(coords[:, :3])
    views = [(18, -60), (18, 35), (55, -60), (8, -100)]
    fig = plt.figure(figsize=(12.5, 10.0), constrained_layout=True)
    colors, color_label = _color_values(groups, rows, color_by)
    finite_colors = colors[np.isfinite(colors)]
    if finite_colors.size:
        norm = Normalize(vmin=float(np.nanmin(finite_colors)), vmax=float(np.nanmax(finite_colors)))
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("viridis")
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    starts, vecs, segment_colors, _segment_label = _segment_arrays(
        groups,
        rows,
        color_by=color_by,
        stride=int(arrow_stride),
        normalize_arrows=bool(normalize_arrows),
    )
    segment_rgba = cmap(norm(segment_colors)) if segment_colors.size else []
    for view_index, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, view_index + 1, projection="3d")
        if plot_style in {"lines", "points_lines"}:
            for group in groups:
                ax.plot(
                    group[:, 0],
                    group[:, 1],
                    group[:, 2],
                    color="0.16",
                    alpha=0.28 if plot_style == "lines" else 0.16,
                    linewidth=0.85 if plot_style == "lines" else 0.65,
                    zorder=1,
                )
        if plot_style == "arrows" and starts.size:
            ax.quiver(
                starts[:, 0],
                starts[:, 1],
                starts[:, 2],
                vecs[:, 0],
                vecs[:, 1],
                vecs[:, 2],
                length=_arrow_length(normalize_arrows, span),
                normalize=bool(normalize_arrows),
                colors=segment_rgba,
                linewidths=0.55,
                alpha=0.72,
                arrow_length_ratio=0.34,
            )
        if plot_style == "points_lines":
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colors,
                s=4,
                cmap="viridis",
                alpha=0.76,
                linewidths=0,
                depthshade=False,
                zorder=3,
            )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(f"PC1 ({100.0 * var_fraction[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100.0 * var_fraction[1]:.1f}%)")
        z_label = f"PC3 ({100.0 * var_fraction[2]:.1f}%)"
        if abs(float(pc3_scale) - 1.0) > 1e-9:
            z_label += f" x{float(pc3_scale):g}"
        ax.set_zlabel(z_label)
        ax.set_title(f"elev {elev}, azim {azim}", fontsize=9)
        _set_axis_limits(ax, center, span)
    fig.suptitle(title, fontsize=13)
    cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.72, pad=0.02)
    cbar.set_label(color_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _interactive_plot(
    out_path: Path,
    projected_groups: list[np.ndarray],
    meta: list[dict[str, object]],
    selected: np.ndarray,
    *,
    color_by: str,
    title: str,
    pc3_scale: float,
    plot_style: str,
    arrow_stride: int,
    normalize_arrows: bool,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    groups = _scaled_groups([projected_groups[int(i)] for i in selected], pc3_scale)
    rows = [meta[int(i)] for i in selected]
    coords = np.concatenate(groups, axis=0)
    colors, color_label = _color_values(groups, rows, color_by)

    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    hover: list[str] = []
    for group, row in zip(groups, rows, strict=False):
        line_x.extend([float(v) for v in group[:, 0]])
        line_y.extend([float(v) for v in group[:, 1]])
        line_z.extend([float(v) for v in group[:, 2]])
        line_x.append(None)
        line_y.append(None)
        line_z.append(None)
        for t in range(group.shape[0]):
            hover.append(
                "<br>".join(
                    [
                        f"candidate={row['candidate_id']}",
                        f"trajectory={row['trajectory_id']}",
                        f"time={t}",
                        f"source_row={row['source_row']}",
                        f"PC=({group[t, 0]:.4g}, {group[t, 1]:.4g}, {group[t, 2]:.4g})",
                    ]
                )
            )
    traces: list[Any] = []
    if plot_style in {"lines", "points_lines", "arrows"}:
        traces.append(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line={"color": "rgba(30,30,30,0.22)" if plot_style == "arrows" else "rgba(30,30,30,0.34)", "width": 2},
                hoverinfo="skip",
                name="linked trajectories",
            )
        )
    if plot_style == "arrows":
        starts, vecs, _segment_colors, _segment_label = _segment_arrays(
            groups,
            rows,
            color_by=color_by,
            stride=int(arrow_stride),
            normalize_arrows=bool(normalize_arrows),
        )
        center, span = _axis_limits(coords[:, :3])
        del center
        traces.append(
            go.Cone(
                x=starts[:, 0],
                y=starts[:, 1],
                z=starts[:, 2],
                u=vecs[:, 0],
                v=vecs[:, 1],
                w=vecs[:, 2],
                sizemode="absolute",
                sizeref=_arrow_length(normalize_arrows, span),
                anchor="tail",
                colorscale="Viridis",
                showscale=False,
                opacity=0.62,
                name="segment direction arrows",
            )
        )
    if plot_style == "points_lines":
        traces.append(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker={
                    "size": 2.2,
                    "color": colors,
                    "colorscale": "Viridis",
                    "opacity": 0.76,
                    "colorbar": {"title": color_label},
                },
                text=hover,
                hoverinfo="text",
                name="trajectory states",
            )
        )
    fig = go.Figure(data=traces)
    z_title = "PC3"
    if abs(float(pc3_scale) - 1.0) > 1e-9:
        z_title += f" x{float(pc3_scale):g}"
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "PC1",
            "yaxis_title": "PC2",
            "zaxis_title": z_title,
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return True


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    manifest_path = run_dir / "response_cache_manifest.csv"
    manifest_rows = _manifest_filter(_read_csv_rows(manifest_path), args)
    if not manifest_rows:
        raise ValueError(f"No response-cache rows selected from {manifest_path}")
    groups, meta = _collect_trajectories(run_dir, manifest_rows, variant=str(args.variant))
    if len(groups) < 2:
        raise ValueError("Need at least two trajectory groups")
    mean, basis, evals = _fit_pca_streamed(groups, n_components=max(3, int(args.n_components)))
    total = float(np.sum(evals)) + 1e-12
    var_fraction = evals[:3] / total
    projected_groups = [_project_group(group, mean, basis) for group in groups]
    centroids = np.asarray([np.mean(group[:, :3], axis=0) for group in projected_groups], dtype=np.float64)
    selected = _farthest_subset(centroids, min(int(args.max_trajectories), len(projected_groups)), seed=int(args.seed))
    color_modes = _parse_list(args.color_by) or ["time_index"]
    outputs: list[str] = []
    style_note = str(args.plot_style).replace("_", " ")
    title = f"{args.variant} PC1-PC2-PC3 {style_note} trajectories"
    for color_by in color_modes:
        stem = f"{args.variant}_pc123_trajectory_lines_by_{color_by}"
        png_path = out_dir / f"{stem}.png"
        html_path = out_dir / f"{stem}.html"
        _static_plot(
            png_path,
            projected_groups,
            meta,
            selected,
            color_by=color_by,
            var_fraction=var_fraction,
            title=f"{title}; colored by {color_by}",
            pc3_scale=float(args.pc3_scale),
            plot_style=str(args.plot_style),
            arrow_stride=int(args.arrow_stride),
            normalize_arrows=bool(args.normalize_arrows),
        )
        outputs.extend([str(png_path), str(png_path.with_suffix(".pdf"))])
        if _interactive_plot(
            html_path,
            projected_groups,
            meta,
            selected,
            color_by=color_by,
            title=f"{title}; colored by {color_by}",
            pc3_scale=float(args.pc3_scale),
            plot_style=str(args.plot_style),
            arrow_stride=int(args.arrow_stride),
            normalize_arrows=bool(args.normalize_arrows),
        ):
            outputs.append(str(html_path))
    summary = {
        "run_dir": str(run_dir),
        "response_cache_manifest": str(manifest_path),
        "output_dir": str(out_dir),
        "variant": str(args.variant),
        "filters": {
            "candidate_set_modes": _parse_list(args.candidate_set_modes),
            "prior_families": _parse_list(args.prior_families),
            "scales": _parse_list(args.scales),
        },
        "n_manifest_rows_selected": int(len(manifest_rows)),
        "n_trajectory_groups_total": int(len(groups)),
        "n_points_total": int(sum(group.shape[0] for group in groups)),
        "n_timepoints_per_complete_trajectory_median": float(np.median([group.shape[0] for group in groups])),
        "n_trajectory_groups_plotted": int(selected.size),
        "n_points_plotted": int(sum(projected_groups[int(i)].shape[0] for i in selected)),
        "plot_style": str(args.plot_style),
        "pc3_scale": float(args.pc3_scale),
        "arrow_stride": int(args.arrow_stride),
        "normalize_arrows": bool(args.normalize_arrows),
        "pca_explained_fraction_first3": [float(v) for v in var_fraction.tolist()],
        "selected_group_indices": [int(v) for v in selected.tolist()],
        "outputs": outputs,
    }
    _write_json(out_dir / "trajectory_line_plot_manifest.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot connected 3D PC trajectories from BackImage global fixation response caches."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-set-modes", type=str, default="")
    parser.add_argument("--prior-families", type=str, default="")
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--variant", choices=("motion_delta", "prior_response"), default="motion_delta")
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--max-trajectories", type=int, default=256)
    parser.add_argument("--color-by", type=str, default="time_index,trajectory_index,source_row")
    parser.add_argument("--plot-style", choices=("points_lines", "lines", "arrows"), default="points_lines")
    parser.add_argument("--pc3-scale", type=float, default=1.0)
    parser.add_argument("--arrow-stride", type=int, default=1)
    parser.add_argument("--normalize-arrows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
