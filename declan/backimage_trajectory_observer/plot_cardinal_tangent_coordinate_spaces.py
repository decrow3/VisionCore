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
import pandas as pd
from matplotlib.lines import Line2D

from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import DEFAULT_INPUT

BX_COLOR = "#0072B2"
BY_COLOR = "#D55E00"

IMAGE_SCALAR_FEATURES = (
    "image_patch_mean",
    "image_patch_std",
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_edge_density",
    "image_orientation_coherence",
    "image_spectrum_anisotropy",
    "image_high_freq_power_fraction",
    "image_power_0_2_cpd_fraction",
    "image_power_2_4_cpd_fraction",
    "image_power_4_8_cpd_fraction",
    "image_power_8plus_cpd_fraction",
)
IMAGE_ORIENTATION_FEATURES = (
    "image_gradient_axis_deg",
    "image_edge_axis_deg",
    "image_spectrum_orientation_deg",
    "image_dominant_orientation_deg",
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


def _unit_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


def _pca(x: np.ndarray, n_components: int = 3, *, standardize: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    arr = arr - mean[None, :]
    scale = np.ones(arr.shape[1], dtype=np.float64)
    if standardize:
        scale = np.nanstd(arr, axis=0)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        arr = arr / scale[None, :]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    _, s, vt = np.linalg.svd(arr, full_matrices=False)
    evals = (s * s) / max(arr.shape[0] - 1, 1)
    k = min(int(n_components), vt.shape[0])
    return arr @ vt[:k].T, vt[:k].T, evals


def _fit_tangent_basis(bx: np.ndarray, by: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    combined = np.concatenate([np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)], axis=0)
    mean = np.mean(combined, axis=0)
    xc = combined - mean[None, :]
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    evals = (s * s) / max(combined.shape[0] - 1, 1)
    basis = vt[:3].T
    bx_score = (np.asarray(bx, dtype=np.float64) - mean[None, :]) @ basis
    by_score = (np.asarray(by, dtype=np.float64) - mean[None, :]) @ basis
    return mean, basis, evals, bx_score, by_score


def _image_feature_matrix(input_csv: Path, source_rows: np.ndarray, *, include_position: bool) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(input_csv)
    df["source_row"] = np.arange(df.shape[0], dtype=int)
    selected = df.set_index("source_row", drop=False).loc[np.asarray(source_rows, dtype=int)].reset_index(drop=True)
    cols: list[str] = []
    parts: list[np.ndarray] = []
    scalar_cols = [col for col in IMAGE_SCALAR_FEATURES if col in selected.columns]
    if include_position:
        scalar_cols.extend([col for col in ("mean_x_deg", "mean_y_deg", "image_patch_center_x_px", "image_patch_center_y_px") if col in selected.columns])
    for col in scalar_cols:
        vals = pd.to_numeric(selected[col], errors="coerce").to_numpy(dtype=np.float64)
        parts.append(vals[:, None])
        cols.append(col)
    for col in IMAGE_ORIENTATION_FEATURES:
        if col not in selected.columns:
            continue
        theta = np.deg2rad(pd.to_numeric(selected[col], errors="coerce").to_numpy(dtype=np.float64))
        parts.append(np.cos(2.0 * theta)[:, None])
        cols.append(f"{col}_cos2")
        parts.append(np.sin(2.0 * theta)[:, None])
        cols.append(f"{col}_sin2")
    if not parts:
        raise ValueError("No image feature columns were found for image-feature PCA")
    return np.concatenate(parts, axis=1), cols


def _nearest_neighbor_cosines(coords: np.ndarray, vectors: np.ndarray, *, k: int) -> np.ndarray:
    xy = np.asarray(coords, dtype=np.float64)
    vec = _unit_rows(vectors)
    n = int(xy.shape[0])
    if n < 2:
        return np.empty(0, dtype=np.float64)
    kk = max(1, min(int(k), n - 1))
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    return np.sum(vec[:, None, :] * vec[nn], axis=2).reshape(-1)


def _stat(vals: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def _plot_field(
    out_path: Path,
    anchors: np.ndarray,
    bx_glyph: np.ndarray,
    by_glyph: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    arrow_scale: float,
    arrow_width: float,
    arrow_alpha: float,
    dot_size: float,
    dot_alpha: float,
    show_x: bool = True,
    show_y: bool = True,
) -> None:
    xy = np.asarray(anchors, dtype=np.float64)[:, :2]
    bx_u = _unit_rows(np.asarray(bx_glyph, dtype=np.float64)[:, :2])
    by_u = _unit_rows(np.asarray(by_glyph, dtype=np.float64)[:, :2])
    spread = float(np.std(xy)) * float(arrow_scale)
    fig, ax = plt.subplots(figsize=(7.0, 6.1), constrained_layout=True)
    ax.scatter(xy[:, 0], xy[:, 1], s=float(dot_size), color="0.55", alpha=float(dot_alpha), linewidths=0, zorder=1)
    if show_x:
        ax.quiver(
            xy[:, 0], xy[:, 1], bx_u[:, 0] * spread, bx_u[:, 1] * spread,
            color=BX_COLOR, scale=1.0, scale_units="xy", angles="xy",
            width=float(arrow_width), headwidth=3.6, headlength=4.6, alpha=float(arrow_alpha), zorder=3,
        )
    if show_y:
        ax.quiver(
            xy[:, 0], xy[:, 1], by_u[:, 0] * spread, by_u[:, 1] * spread,
            color=BY_COLOR, scale=1.0, scale_units="xy", angles="xy",
            width=float(arrow_width), headwidth=3.6, headlength=4.6, alpha=float(arrow_alpha), zorder=3,
        )
    handles = []
    if show_x:
        handles.append(Line2D([0], [0], color=BX_COLOR, lw=1.4, label=r"$b_x(I)$"))
    if show_y:
        handles.append(Line2D([0], [0], color=BY_COLOR, lw=1.4, label=r"$b_y(I)$"))
    ax.legend(handles=handles, frameon=False, loc="upper right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_tangent_glyph(
    out_path: Path,
    bx_score: np.ndarray,
    by_score: np.ndarray,
    *,
    title: str,
    arrow_alpha: float,
    dot_size: float,
    show_x: bool = True,
    show_y: bool = True,
) -> None:
    bx = np.asarray(bx_score, dtype=np.float64)[:, :2]
    by = np.asarray(by_score, dtype=np.float64)[:, :2]
    centers = 0.5 * (bx + by)
    fig, ax = plt.subplots(figsize=(7.0, 6.1), constrained_layout=True)
    ax.scatter(centers[:, 0], centers[:, 1], s=float(dot_size), color="0.55", alpha=0.34, linewidths=0, zorder=1)
    if show_x:
        ax.quiver(
            centers[:, 0], centers[:, 1], bx[:, 0] - centers[:, 0], bx[:, 1] - centers[:, 1],
            color=BX_COLOR, scale=1.0, scale_units="xy", angles="xy",
            width=0.0014, headwidth=3.6, headlength=4.6, alpha=float(arrow_alpha), zorder=3,
        )
    if show_y:
        ax.quiver(
            centers[:, 0], centers[:, 1], by[:, 0] - centers[:, 0], by[:, 1] - centers[:, 1],
            color=BY_COLOR, scale=1.0, scale_units="xy", angles="xy",
            width=0.0014, headwidth=3.6, headlength=4.6, alpha=float(arrow_alpha), zorder=3,
        )
    handles = []
    if show_x:
        handles.append(Line2D([0], [0], color=BX_COLOR, lw=1.4, label=r"$b_x(I)$ endpoint"))
    if show_y:
        handles.append(Line2D([0], [0], color=BY_COLOR, lw=1.4, label=r"$b_y(I)$ endpoint"))
    ax.legend(handles=handles, frameon=False, loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("Tangent PC1")
    ax.set_ylabel("Tangent PC2")
    ax.grid(True, alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    cache_path = Path(args.cache)
    out_dir = Path(args.out_dir)
    z = np.load(cache_path, allow_pickle=True)
    r0 = np.asarray(z["r0"], dtype=np.float64)
    bx = np.asarray(z["bx"], dtype=np.float64)
    by = np.asarray(z["by"], dtype=np.float64)
    source_rows = np.asarray(z["source_rows"], dtype=np.int64)
    tangent_mean, tangent_basis, tangent_evals, bx_t, by_t = _fit_tangent_basis(bx, by)
    r0_tangent = (r0 - np.mean(r0, axis=0, keepdims=True)) @ tangent_basis
    image_x, image_cols = _image_feature_matrix(Path(args.input), source_rows, include_position=bool(args.include_position))
    image_pc, _image_basis, image_evals = _pca(image_x, n_components=3, standardize=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    modes = {
        "compact_response": {
            "anchors": r0_tangent,
            "bx_glyph": bx @ tangent_basis,
            "by_glyph": by @ tangent_basis,
            "xlabel": "Response projected on tangent PC1",
            "ylabel": "Response projected on tangent PC2",
            "title": "BackImage tangents over compact-tangent response coordinates",
        },
        "image_feature": {
            "anchors": image_pc,
            "bx_glyph": bx @ tangent_basis,
            "by_glyph": by @ tangent_basis,
            "xlabel": f"Image-feature PC1 ({100.0 * image_evals[0] / (np.sum(image_evals) + 1e-12):.1f}% var.)",
            "ylabel": f"Image-feature PC2 ({100.0 * image_evals[1] / (np.sum(image_evals) + 1e-12):.1f}% var.)",
            "title": "BackImage tangents over local image-feature coordinates",
        },
    }
    for mode, spec in modes.items():
        _plot_field(
            out_dir / f"{mode}_pc12_combined.png",
            spec["anchors"],
            spec["bx_glyph"],
            spec["by_glyph"],
            title=str(spec["title"]),
            xlabel=str(spec["xlabel"]),
            ylabel=str(spec["ylabel"]),
            arrow_scale=float(args.arrow_scale),
            arrow_width=float(args.arrow_width),
            arrow_alpha=float(args.arrow_alpha),
            dot_size=float(args.dot_size),
            dot_alpha=float(args.dot_alpha),
            show_x=True,
            show_y=True,
        )
        _plot_field(
            out_dir / f"{mode}_pc12_x_only.png",
            spec["anchors"],
            spec["bx_glyph"],
            spec["by_glyph"],
            title=f"{spec['title']} - x only",
            xlabel=str(spec["xlabel"]),
            ylabel=str(spec["ylabel"]),
            arrow_scale=float(args.arrow_scale),
            arrow_width=float(args.arrow_width),
            arrow_alpha=float(args.arrow_alpha),
            dot_size=float(args.dot_size),
            dot_alpha=float(args.dot_alpha),
            show_x=True,
            show_y=False,
        )
        _plot_field(
            out_dir / f"{mode}_pc12_y_only.png",
            spec["anchors"],
            spec["bx_glyph"],
            spec["by_glyph"],
            title=f"{spec['title']} - y only",
            xlabel=str(spec["xlabel"]),
            ylabel=str(spec["ylabel"]),
            arrow_scale=float(args.arrow_scale),
            arrow_width=float(args.arrow_width),
            arrow_alpha=float(args.arrow_alpha),
            dot_size=float(args.dot_size),
            dot_alpha=float(args.dot_alpha),
            show_x=False,
            show_y=True,
        )

    _plot_tangent_glyph(
        out_dir / "tangent_glyph_pc12_combined.png",
        bx_t,
        by_t,
        title="BackImage paired x/y tangents in tangent-vector PCA",
        arrow_alpha=float(args.arrow_alpha),
        dot_size=float(args.dot_size),
        show_x=True,
        show_y=True,
    )
    _plot_tangent_glyph(
        out_dir / "tangent_glyph_pc12_x_only.png",
        bx_t,
        by_t,
        title="BackImage x tangent endpoints in tangent-vector PCA",
        arrow_alpha=float(args.arrow_alpha),
        dot_size=float(args.dot_size),
        show_x=True,
        show_y=False,
    )
    _plot_tangent_glyph(
        out_dir / "tangent_glyph_pc12_y_only.png",
        bx_t,
        by_t,
        title="BackImage y tangent endpoints in tangent-vector PCA",
        arrow_alpha=float(args.arrow_alpha),
        dot_size=float(args.dot_size),
        show_x=False,
        show_y=True,
    )

    summary = {
        "cache": str(cache_path),
        "input": str(args.input),
        "out_dir": str(out_dir),
        "n_objects": int(r0.shape[0]),
        "n_features": int(r0.shape[1]),
        "image_feature_columns": image_cols,
        "include_position": bool(args.include_position),
        "tangent_pca_fraction_first3": [float(v / (np.sum(tangent_evals) + 1e-12)) for v in tangent_evals[:3]],
        "image_feature_pca_fraction_first3": [float(v / (np.sum(image_evals) + 1e-12)) for v in image_evals[:3]],
        "local_k": int(args.local_k),
        "local_compact_response_bx_cos": _stat(_nearest_neighbor_cosines(r0_tangent[:, :2], bx, k=int(args.local_k))),
        "local_compact_response_by_cos": _stat(_nearest_neighbor_cosines(r0_tangent[:, :2], by, k=int(args.local_k))),
        "local_image_feature_bx_cos": _stat(_nearest_neighbor_cosines(image_pc[:, :2], bx, k=int(args.local_k))),
        "local_image_feature_by_cos": _stat(_nearest_neighbor_cosines(image_pc[:, :2], by, k=int(args.local_k))),
        "local_tangent_glyph_center_bx_cos": _stat(_nearest_neighbor_cosines(0.5 * (bx_t[:, :2] + by_t[:, :2]), bx, k=int(args.local_k))),
        "local_tangent_glyph_center_by_cos": _stat(_nearest_neighbor_cosines(0.5 * (bx_t[:, :2] + by_t[:, :2]), by, k=int(args.local_k))),
        "note": "Image-feature plots place fixation anchors by local image-feature PCA; arrow glyph directions are response tangents projected into tangent-PC coordinates.",
    }
    _write_json(out_dir / "coordinate_space_tangent_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot BackImage cardinal tangent fields over alternate coordinate spaces.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-position", action="store_true")
    parser.add_argument("--arrow-scale", type=float, default=0.14)
    parser.add_argument("--arrow-width", type=float, default=0.00125)
    parser.add_argument("--arrow-alpha", type=float, default=0.58)
    parser.add_argument("--dot-size", type=float, default=5.0)
    parser.add_argument("--dot-alpha", type=float, default=0.32)
    parser.add_argument("--local-k", type=int, default=8)
    return parser


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
