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
from matplotlib.lines import Line2D

BX_COLOR = "#0072B2"
BY_COLOR = "#D55E00"


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


def _unit_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


def _axis_limits(points: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float64)
    lo = np.nanpercentile(pts, 1, axis=0)
    hi = np.nanpercentile(pts, 99, axis=0)
    center = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    return center, span


def _plot_indices(points: np.ndarray, max_arrows: int, seed: int) -> np.ndarray:
    n = int(points.shape[0])
    if int(max_arrows) <= 0 or int(max_arrows) >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=int(max_arrows), replace=False)).astype(np.int64)


def _nearest_neighbor_cosines(
    coords: np.ndarray,
    vectors: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(coords, dtype=np.float64)
    vec = _unit_rows(vectors)
    n = int(xy.shape[0])
    if n < 2:
        return np.empty(0), np.empty(0)
    kk = max(1, min(int(k), n - 1))
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    cos = np.sum(vec[:, None, :] * vec[nn], axis=2)
    return cos.reshape(-1), dist[np.arange(n)[:, None], nn].reshape(-1)


def _dense_summary(cache: dict[str, np.ndarray], *, local_k: int) -> dict[str, Any]:
    r0 = np.asarray(cache["r0"], dtype=np.float64)
    bx = np.asarray(cache["bx"], dtype=np.float64)
    by = np.asarray(cache["by"], dtype=np.float64)
    r0_pc = np.asarray(cache["r0_pc123"], dtype=np.float64)
    bx_pc = np.asarray(cache["bx_pc123"], dtype=np.float64)
    by_pc = np.asarray(cache["by_pc123"], dtype=np.float64)

    bx_norm = np.linalg.norm(bx, axis=1)
    by_norm = np.linalg.norm(by, axis=1)
    within_cos = np.sum(bx * by, axis=1) / (bx_norm * by_norm + 1e-12)
    local_bx_cos_pc12, local_dist_pc12 = _nearest_neighbor_cosines(r0_pc[:, :2], bx, k=int(local_k))
    local_by_cos_pc12, _ = _nearest_neighbor_cosines(r0_pc[:, :2], by, k=int(local_k))
    local_bx_cos_pc123, local_dist_pc123 = _nearest_neighbor_cosines(r0_pc[:, :3], bx, k=int(local_k))
    local_by_cos_pc123, _ = _nearest_neighbor_cosines(r0_pc[:, :3], by, k=int(local_k))

    def stat(vals: np.ndarray) -> dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
        return {
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    return {
        "n_objects": int(r0.shape[0]),
        "n_features": int(r0.shape[1]),
        "local_k": int(local_k),
        "within_object_bx_by_cos": stat(within_cos),
        "within_object_bx_by_angle_deg": stat(np.degrees(np.arccos(np.clip(within_cos, -1.0, 1.0)))),
        "local_pc12_bx_neighbor_cos": stat(local_bx_cos_pc12),
        "local_pc12_by_neighbor_cos": stat(local_by_cos_pc12),
        "local_pc123_bx_neighbor_cos": stat(local_bx_cos_pc123),
        "local_pc123_by_neighbor_cos": stat(local_by_cos_pc123),
        "local_pc12_neighbor_distance": stat(local_dist_pc12),
        "local_pc123_neighbor_distance": stat(local_dist_pc123),
        "pc123_bx_projected_norm": stat(np.linalg.norm(bx_pc[:, :3], axis=1)),
        "pc123_by_projected_norm": stat(np.linalg.norm(by_pc[:, :3], axis=1)),
    }


def _plot_dense_pc12(
    out_path: Path,
    cache: dict[str, np.ndarray],
    *,
    max_arrows: int,
    arrow_scale: float,
    arrow_width: float,
    arrow_alpha: float,
    dot_size: float,
    dot_alpha: float,
    x_color: str,
    y_color: str,
    show_x: bool,
    show_y: bool,
    title: str,
    seed: int,
) -> None:
    r0 = np.asarray(cache["r0_pc123"], dtype=np.float64)
    bx = np.asarray(cache["bx_pc123"], dtype=np.float64)
    by = np.asarray(cache["by_pc123"], dtype=np.float64)
    idx = _plot_indices(r0[:, :2], int(max_arrows), int(seed))
    spread = float(np.std(r0[:, :2])) * float(arrow_scale)
    bx_u = _unit_rows(bx[:, :2])
    by_u = _unit_rows(by[:, :2])
    fig, ax = plt.subplots(figsize=(7.0, 6.1), constrained_layout=True)
    ax.scatter(r0[:, 0], r0[:, 1], s=float(dot_size), color="0.55", alpha=float(dot_alpha), linewidths=0, zorder=1)
    if bool(show_x):
        ax.quiver(
            r0[idx, 0],
            r0[idx, 1],
            bx_u[idx, 0] * spread,
            bx_u[idx, 1] * spread,
            color=str(x_color),
            scale=1.0,
            scale_units="xy",
            angles="xy",
            width=float(arrow_width),
            headwidth=3.6,
            headlength=4.6,
            alpha=float(arrow_alpha),
            zorder=3,
        )
    if bool(show_y):
        ax.quiver(
            r0[idx, 0],
            r0[idx, 1],
            by_u[idx, 0] * spread,
            by_u[idx, 1] * spread,
            color=str(y_color),
            scale=1.0,
            scale_units="xy",
            angles="xy",
            width=float(arrow_width),
            headwidth=3.6,
            headlength=4.6,
            alpha=float(arrow_alpha),
            zorder=3,
        )
    ax.set_xlabel("Static response PC1")
    ax.set_ylabel("Static response PC2")
    ax.set_title(str(title))
    ax.grid(True, alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = []
    if bool(show_x):
        handles.append(Line2D([0], [0], color=str(x_color), lw=1.2, label=r"$b_x(I)$"))
    if bool(show_y):
        handles.append(Line2D([0], [0], color=str(y_color), lw=1.2, label=r"$b_y(I)$"))
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_dense_pc123(
    out_path: Path,
    cache: dict[str, np.ndarray],
    *,
    max_arrows: int,
    arrow_scale: float,
    arrow_linewidth: float,
    arrow_alpha: float,
    dot_size: float,
    dot_alpha: float,
    x_color: str,
    y_color: str,
    show_x: bool,
    show_y: bool,
    title: str,
    seed: int,
) -> None:
    r0 = np.asarray(cache["r0_pc123"], dtype=np.float64)
    bx = np.asarray(cache["bx_pc123"], dtype=np.float64)
    by = np.asarray(cache["by_pc123"], dtype=np.float64)
    idx = _plot_indices(r0[:, :3], int(max_arrows), int(seed))
    spread = float(np.std(r0[:, :3])) * float(arrow_scale)
    bx_u = _unit_rows(bx[:, :3])
    by_u = _unit_rows(by[:, :3])
    starts = r0[idx, :3]
    end_points = np.concatenate([starts, starts + bx_u[idx] * spread, starts + by_u[idx] * spread], axis=0)
    center, span = _axis_limits(np.concatenate([r0[:, :3], end_points], axis=0))
    views = [(18, -60), (18, 35), (55, -60), (8, -100)]
    fig = plt.figure(figsize=(13.0, 10.0), constrained_layout=True)
    for view_idx, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, view_idx + 1, projection="3d")
        ax.scatter(
            r0[:, 0],
            r0[:, 1],
            r0[:, 2],
            s=float(dot_size),
            color="0.55",
            alpha=float(dot_alpha),
            linewidths=0,
            depthshade=False,
        )
        arrow_sets = []
        if bool(show_x):
            arrow_sets.append((str(x_color), bx_u))
        if bool(show_y):
            arrow_sets.append((str(y_color), by_u))
        for color, vecs in arrow_sets:
            ax.quiver(
                starts[:, 0],
                starts[:, 1],
                starts[:, 2],
                vecs[idx, 0] * spread,
                vecs[idx, 1] * spread,
                vecs[idx, 2] * spread,
                length=1.0,
                normalize=False,
                arrow_length_ratio=0.18,
                linewidth=float(arrow_linewidth),
                color=color,
                alpha=float(arrow_alpha),
            )
        ax.view_init(elev=elev, azim=azim)
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, strict=False):
            setter(float(c - 0.56 * span), float(c + 0.56 * span))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"elev {elev}, azim {azim}", fontsize=9)
    fig.suptitle(str(title), fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    cache_path = Path(args.cache)
    out_dir = Path(args.out_dir)
    loaded = np.load(cache_path, allow_pickle=True)
    cache = {key: loaded[key] for key in loaded.files}
    summary = _dense_summary(cache, local_k=int(args.local_k))
    summary.update(
        {
            "cache": str(cache_path),
            "out_dir": str(out_dir),
            "max_arrows": int(args.max_arrows),
            "arrow_scale": float(args.arrow_scale),
            "arrow_width": float(args.arrow_width),
            "arrow_linewidth_3d": float(args.arrow_linewidth_3d),
            "arrow_alpha": float(args.arrow_alpha),
            "dot_size": float(args.dot_size),
            "dot_alpha": float(args.dot_alpha),
            "x_color": str(args.x_color),
            "y_color": str(args.y_color),
            "split_axis_panels": bool(args.split_axis_panels),
        }
    )
    base_title = f"Dense BackImage cardinal tangent field ({summary['n_objects']} fixation anchors)"
    _plot_dense_pc12(
        out_dir / "dense_cardinal_tangent_field_pc12.png",
        cache,
        max_arrows=int(args.max_arrows),
        arrow_scale=float(args.arrow_scale),
        arrow_width=float(args.arrow_width),
        arrow_alpha=float(args.arrow_alpha),
        dot_size=float(args.dot_size),
        dot_alpha=float(args.dot_alpha),
        x_color=str(args.x_color),
        y_color=str(args.y_color),
        show_x=True,
        show_y=True,
        title=base_title,
        seed=int(args.seed),
    )
    _plot_dense_pc123(
        out_dir / "dense_cardinal_tangent_field_pc123.png",
        cache,
        max_arrows=int(args.max_arrows),
        arrow_scale=float(args.arrow_scale),
        arrow_linewidth=float(args.arrow_linewidth_3d),
        arrow_alpha=float(args.arrow_alpha),
        dot_size=float(args.dot_size),
        dot_alpha=float(args.dot_alpha),
        x_color=str(args.x_color),
        y_color=str(args.y_color),
        show_x=True,
        show_y=True,
        title=base_title,
        seed=int(args.seed),
    )
    if bool(args.split_axis_panels):
        for axis_name, show_x, show_y in (("x_only", True, False), ("y_only", False, True)):
            axis_label = "horizontal x tangents" if show_x else "vertical y tangents"
            _plot_dense_pc12(
                out_dir / f"dense_cardinal_tangent_field_pc12_{axis_name}.png",
                cache,
                max_arrows=int(args.max_arrows),
                arrow_scale=float(args.arrow_scale),
                arrow_width=float(args.arrow_width),
                arrow_alpha=float(args.arrow_alpha),
                dot_size=float(args.dot_size),
                dot_alpha=float(args.dot_alpha),
                x_color=str(args.x_color),
                y_color=str(args.y_color),
                show_x=show_x,
                show_y=show_y,
                title=f"Dense BackImage {axis_label} ({summary['n_objects']} fixation anchors)",
                seed=int(args.seed),
            )
            _plot_dense_pc123(
                out_dir / f"dense_cardinal_tangent_field_pc123_{axis_name}.png",
                cache,
                max_arrows=int(args.max_arrows),
                arrow_scale=float(args.arrow_scale),
                arrow_linewidth=float(args.arrow_linewidth_3d),
                arrow_alpha=float(args.arrow_alpha),
                dot_size=float(args.dot_size),
                dot_alpha=float(args.dot_alpha),
                x_color=str(args.x_color),
                y_color=str(args.y_color),
                show_x=show_x,
                show_y=show_y,
                title=f"Dense BackImage {axis_label} ({summary['n_objects']} fixation anchors)",
                seed=int(args.seed),
            )
    _write_json(out_dir / "dense_cardinal_tangent_field_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot dense thin-arrow fields from a BackImage cardinal tangent cache.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-arrows", type=int, default=0, help="0 means plot every cached paired tangent.")
    parser.add_argument("--arrow-scale", type=float, default=0.18)
    parser.add_argument("--arrow-width", type=float, default=0.0020)
    parser.add_argument("--arrow-linewidth-3d", type=float, default=0.38)
    parser.add_argument("--arrow-alpha", type=float, default=0.62)
    parser.add_argument("--dot-size", type=float, default=8.0)
    parser.add_argument("--dot-alpha", type=float, default=0.45)
    parser.add_argument("--x-color", type=str, default=BX_COLOR)
    parser.add_argument("--y-color", type=str, default=BY_COLOR)
    parser.add_argument("--split-axis-panels", action="store_true")
    parser.add_argument("--local-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
