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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _point_array(rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    coords = np.asarray(
        [[_safe_float(row.get("pc1")), _safe_float(row.get("pc2")), _safe_float(row.get("pc3"))] for row in rows],
        dtype=np.float64,
    )
    fields: dict[str, np.ndarray] = {}
    for key in (
        "time_index",
        "trajectory_index",
        "source_row",
        "patch_x",
        "patch_y",
        "patch_rms_contrast",
        "gradient_energy",
        "orientation_coherence",
    ):
        fields[key] = np.asarray([_safe_float(row.get(key)) for row in rows], dtype=np.float64)
    candidate_ids = [str(row.get("candidate_id", "")) for row in rows]
    return coords, fields, candidate_ids


def _filter_rows(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get("response_variant", "")) == str(variant)]


def _downsample(
    coords: np.ndarray,
    fields: dict[str, np.ndarray],
    candidate_ids: list[str],
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    keep = np.isfinite(coords).all(axis=1)
    coords = coords[keep]
    fields = {key: val[keep] for key, val in fields.items()}
    candidate_ids = [cid for cid, ok in zip(candidate_ids, keep, strict=False) if bool(ok)]
    if int(max_points) > 0 and coords.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(coords.shape[0], size=int(max_points), replace=False))
        coords = coords[idx]
        fields = {key: val[idx] for key, val in fields.items()}
        candidate_ids = [candidate_ids[int(i)] for i in idx]
    return coords, fields, candidate_ids


def _color_values(fields: dict[str, np.ndarray], candidate_ids: list[str], color_by: str) -> np.ndarray:
    if color_by == "candidate_index":
        labels = {cid: idx for idx, cid in enumerate(sorted(set(candidate_ids)))}
        return np.asarray([labels[cid] for cid in candidate_ids], dtype=np.float64)
    if color_by in fields:
        vals = np.asarray(fields[color_by], dtype=np.float64)
        if np.isfinite(vals).any():
            return vals
    return np.arange(len(candidate_ids), dtype=np.float64)


def _set_axis_limits(ax: Any, coords: np.ndarray) -> None:
    mins = np.nanpercentile(coords, 1, axis=0)
    maxs = np.nanpercentile(coords, 99, axis=0)
    centers = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    for setter, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centers, strict=False):
        setter(float(center - 0.55 * span), float(center + 0.55 * span))


def _static_plot(
    coords: np.ndarray,
    colors: np.ndarray,
    *,
    title: str,
    color_label: str,
    out_path: Path,
) -> None:
    views = [(18, -60), (18, 35), (55, -60), (8, -100)]
    fig = plt.figure(figsize=(12.5, 10.0), constrained_layout=True)
    vals = np.asarray(colors, dtype=np.float64)
    if not np.isfinite(vals).any():
        vals = np.arange(coords.shape[0], dtype=np.float64)
    for idx, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=vals,
            s=5,
            cmap="viridis",
            alpha=0.62,
            linewidths=0,
            depthshade=False,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"elev {elev}, azim {azim}", fontsize=9)
        _set_axis_limits(ax, coords)
    fig.suptitle(title, fontsize=13)
    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.72, pad=0.02)
    cbar.set_label(color_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _interactive_plot(
    coords: np.ndarray,
    fields: dict[str, np.ndarray],
    candidate_ids: list[str],
    colors: np.ndarray,
    *,
    title: str,
    color_label: str,
    out_path: Path,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    hover = []
    for idx, cid in enumerate(candidate_ids):
        hover.append(
            "<br>".join(
                [
                    f"candidate={cid}",
                    f"time={fields['time_index'][idx]:g}",
                    f"trajectory={fields['trajectory_index'][idx]:g}",
                    f"source_row={fields['source_row'][idx]:g}",
                    f"patch=({fields['patch_x'][idx]:.1f}, {fields['patch_y'][idx]:.1f})",
                    f"PC=({coords[idx, 0]:.4g}, {coords[idx, 1]:.4g}, {coords[idx, 2]:.4g})",
                ]
            )
        )
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker={
                    "size": 2.4,
                    "color": colors,
                    "colorscale": "Viridis",
                    "opacity": 0.68,
                    "colorbar": {"title": color_label},
                },
                text=hover,
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "PC1",
            "yaxis_title": "PC2",
            "zaxis_title": "PC3",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return True


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rows_all = _read_rows(Path(args.points_csv))
    out_dir = Path(args.output_dir)
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    color_modes = [v.strip() for v in str(args.color_by).split(",") if v.strip()]
    outputs: list[str] = []
    counts: dict[str, int] = {}
    for variant in variants:
        rows = _filter_rows(rows_all, variant)
        coords, fields, candidate_ids = _point_array(rows)
        coords, fields, candidate_ids = _downsample(
            coords,
            fields,
            candidate_ids,
            max_points=int(args.max_points),
            seed=int(args.seed),
        )
        counts[variant] = int(coords.shape[0])
        if coords.shape[0] < 4:
            continue
        for color_by in color_modes:
            colors = _color_values(fields, candidate_ids, color_by)
            stem = f"{variant}_pc123_by_{color_by}"
            title = f"{variant} PC1-PC2-PC3 colored by {color_by}"
            png_path = out_dir / f"{stem}.png"
            html_path = out_dir / f"{stem}.html"
            _static_plot(coords, colors, title=title, color_label=color_by, out_path=png_path)
            outputs.append(str(png_path))
            outputs.append(str(png_path.with_suffix(".pdf")))
            if _interactive_plot(
                coords,
                fields,
                candidate_ids,
                colors,
                title=title,
                color_label=color_by,
                out_path=html_path,
            ):
                outputs.append(str(html_path))
    manifest = {
        "points_csv": str(args.points_csv),
        "output_dir": str(out_dir),
        "variants": variants,
        "color_by": color_modes,
        "max_points": int(args.max_points),
        "point_counts": counts,
        "outputs": outputs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "global_fixation_geometry_3d_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Make static and interactive 3D PC plots for global BackImage fixation geometry.")
    p.add_argument("--points-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variants", type=str, default="motion_delta")
    p.add_argument("--color-by", type=str, default="time_index,trajectory_index,source_row")
    p.add_argument("--max-points", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
