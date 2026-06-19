from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.backimage_trajectory_observer.plot_global_fixation_trajectory_lines_3d import (
    _collect_trajectories,
    _farthest_subset,
    _interactive_plot,
    _manifest_filter,
    _parse_list,
    _read_csv_rows,
    _safe_float,
    _static_plot,
    _write_json,
)


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


def _fit_vector_pca(groups: list[np.ndarray], n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not groups:
        raise ValueError("Need trajectory groups")
    n_units = int(groups[0].shape[1])
    state_n = 0
    state_sum = np.zeros(n_units, dtype=np.float64)
    vec_n = 0
    vec_sum = np.zeros(n_units, dtype=np.float64)
    for group in groups:
        x = np.asarray(group, dtype=np.float64)
        if x.shape[1] != n_units:
            raise ValueError("All groups must share the same response dimension")
        state_n += int(x.shape[0])
        state_sum += np.sum(x, axis=0)
        d = np.diff(x, axis=0)
        vec_n += int(d.shape[0])
        vec_sum += np.sum(d, axis=0)
    if state_n < 3 or vec_n < 3:
        raise ValueError("Need at least 3 states and 3 segment vectors")
    state_mean = state_sum / float(state_n)
    vec_mean = vec_sum / float(vec_n)
    cov = np.zeros((n_units, n_units), dtype=np.float64)
    for group in groups:
        d = np.diff(np.asarray(group, dtype=np.float64), axis=0) - vec_mean[None, :]
        cov += d.T @ d
    cov /= float(max(vec_n - 1, 1))
    evals, vecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    vecs = vecs[:, order]
    k = min(max(1, int(n_components)), vecs.shape[1])
    return state_mean.astype(np.float32), vec_mean.astype(np.float32), vecs[:, :k].astype(np.float32), evals.astype(np.float32)


def _project_groups_to_basis(groups: list[np.ndarray], state_mean: np.ndarray, basis: np.ndarray) -> list[np.ndarray]:
    mean = np.asarray(state_mean, dtype=np.float32)
    b = np.asarray(basis, dtype=np.float32)
    return [(np.asarray(group, dtype=np.float32) - mean[None, :]) @ b for group in groups]


def _standardized_pca(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    centered = arr - mean[None, :]
    scale = np.nanstd(centered, axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    z = np.nan_to_num(centered / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    evals = (s * s) / max(z.shape[0] - 1, 1)
    k = min(int(n_components), vt.shape[0])
    return z @ vt[:k].T, vt[:k].T, evals, mean, scale


def _feature_table(selected_windows_csv: Path, source_rows: list[int], *, include_position: bool) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(selected_windows_csv)
    if "source_row" not in df.columns:
        raise ValueError(f"{selected_windows_csv} has no source_row column")
    lookup = df.drop_duplicates("source_row").set_index("source_row", drop=False)
    missing = sorted(set(int(v) for v in source_rows) - set(int(v) for v in lookup.index.tolist()))
    if missing:
        raise ValueError(f"{len(missing)} source rows were missing from {selected_windows_csv}: {missing[:8]}")
    selected = lookup.loc[np.asarray(source_rows, dtype=int)].reset_index(drop=True)
    cols: list[str] = []
    parts: list[np.ndarray] = []
    scalar_cols = [col for col in IMAGE_SCALAR_FEATURES if col in selected.columns]
    if include_position:
        scalar_cols.extend(
            [
                col
                for col in ("mean_x_deg", "mean_y_deg", "image_patch_center_x_px", "image_patch_center_y_px")
                if col in selected.columns
            ]
        )
    for col in scalar_cols:
        parts.append(pd.to_numeric(selected[col], errors="coerce").to_numpy(dtype=np.float64)[:, None])
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
        raise ValueError("No image feature columns were available")
    return selected, np.concatenate(parts, axis=1), cols


def _unique_source_rows(meta: list[dict[str, object]]) -> list[int]:
    rows = sorted({int(_safe_float(row.get("source_row"))) for row in meta if np.isfinite(_safe_float(row.get("source_row")))})
    if not rows:
        raise ValueError("No source_row metadata found")
    return rows


def _source_feature_scores(
    selected_windows_csv: Path,
    meta: list[dict[str, object]],
    *,
    include_position: bool,
) -> tuple[dict[int, np.ndarray], np.ndarray, list[int], list[str], np.ndarray]:
    source_rows = _unique_source_rows(meta)
    _selected, matrix, cols = _feature_table(selected_windows_csv, source_rows, include_position=include_position)
    scores, _basis, evals, _mean, _scale = _standardized_pca(matrix, n_components=3)
    return {int(src): scores[i] for i, src in enumerate(source_rows)}, scores, source_rows, cols, evals


def _time_bins(n_segments: int, n_bins: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, int(n_segments), int(n_bins) + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(int(n_bins))]


def _aggregate_feature_flow(
    groups: list[np.ndarray],
    meta: list[dict[str, object]],
    basis: np.ndarray,
    *,
    time_bin_count: int,
) -> dict[tuple[int, int], list[np.ndarray]]:
    b = np.asarray(basis, dtype=np.float64)
    buckets: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    n_segments = min(int(group.shape[0] - 1) for group in groups if group.shape[0] >= 2)
    bins = _time_bins(n_segments, int(time_bin_count))
    for group, row in zip(groups, meta, strict=False):
        source_row = int(_safe_float(row.get("source_row")))
        if not np.isfinite(source_row):
            continue
        d = np.diff(np.asarray(group, dtype=np.float64), axis=0) @ b
        unit = _unit_rows(d[:, :3])
        for bin_index, (start, stop) in enumerate(bins):
            if stop <= start:
                continue
            buckets[(source_row, bin_index)].append(unit[start:stop])
    return buckets


def _candidate_flow_rows(
    feature_scores: dict[int, np.ndarray],
    buckets: dict[tuple[int, int], list[np.ndarray]],
    *,
    time_bin_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_row, coords in sorted(feature_scores.items()):
        for bin_index in range(int(time_bin_count)):
            arrays = buckets.get((int(source_row), int(bin_index)), [])
            if not arrays:
                continue
            v = np.concatenate(arrays, axis=0)
            mean_vec = np.mean(v, axis=0)
            coherence = float(np.linalg.norm(mean_vec))
            unit = mean_vec / (coherence + 1e-12)
            rows.append(
                {
                    "source_row": int(source_row),
                    "time_bin": int(bin_index),
                    "n_segments": int(v.shape[0]),
                    "image_feature_pc1": float(coords[0]),
                    "image_feature_pc2": float(coords[1]),
                    "image_feature_pc3": float(coords[2]) if len(coords) > 2 else float("nan"),
                    "flow_tangent_pc1": float(unit[0]),
                    "flow_tangent_pc2": float(unit[1]),
                    "flow_tangent_pc3": float(unit[2]) if len(unit) > 2 else float("nan"),
                    "flow_coherence": coherence,
                }
            )
    return rows


def _neighbor_cosines(rows: list[dict[str, object]], *, k: int, n_shuffles: int, seed: int) -> dict[str, float]:
    if not rows:
        return {"median": float("nan"), "mean": float("nan")}
    rng = np.random.default_rng(int(seed))
    vals_all = []
    null_medians = []
    null_means = []
    for bin_index in sorted({int(row["time_bin"]) for row in rows}):
        block = [row for row in rows if int(row["time_bin"]) == int(bin_index)]
        if len(block) <= int(k):
            continue
        coords = np.asarray([[row["image_feature_pc1"], row["image_feature_pc2"]] for row in block], dtype=np.float64)
        vec = _unit_rows(
            np.asarray([[row["flow_tangent_pc1"], row["flow_tangent_pc2"], row["flow_tangent_pc3"]] for row in block], dtype=np.float64)
        )
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        kk = min(int(k), len(block) - 1)
        nn = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
        vals_all.extend(np.sum(vec[:, None, :] * vec[nn], axis=2).reshape(-1).tolist())
        for _ in range(int(n_shuffles)):
            shuffled = vec[rng.permutation(vec.shape[0])]
            null_vals = np.sum(shuffled[:, None, :] * shuffled[nn], axis=2).reshape(-1)
            null_medians.append(float(np.nanmedian(null_vals)))
            null_means.append(float(np.nanmean(null_vals)))
    vals = np.asarray(vals_all, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"median": float("nan"), "mean": float("nan")}
    null_medians_arr = np.asarray(null_medians, dtype=np.float64)
    null_means_arr = np.asarray(null_means, dtype=np.float64)
    median = float(np.median(vals))
    mean = float(np.mean(vals))
    return {
        "median": median,
        "mean": mean,
        "null_median_mean": float(np.nanmean(null_medians_arr)) if null_medians_arr.size else float("nan"),
        "null_mean_mean": float(np.nanmean(null_means_arr)) if null_means_arr.size else float("nan"),
        "median_lift_over_shuffle": float(median / (np.nanmean(null_medians_arr) + 1e-12)) if null_medians_arr.size else float("nan"),
        "mean_lift_over_shuffle": float(mean / (np.nanmean(null_means_arr) + 1e-12)) if null_means_arr.size else float("nan"),
    }


def _plot_image_feature_flow(
    out_path: Path,
    rows: list[dict[str, object]],
    *,
    time_bin_count: int,
    title: str,
    scale_by_coherence: bool,
) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, int(time_bin_count), figsize=(4.2 * int(time_bin_count), 4.0), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    all_xy = np.asarray([[row["image_feature_pc1"], row["image_feature_pc2"]] for row in rows], dtype=np.float64)
    center = np.nanmean(all_xy, axis=0)
    span = float(np.nanmax(np.nanpercentile(all_xy, 98, axis=0) - np.nanpercentile(all_xy, 2, axis=0)))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    arrow_len = 0.17 * span
    for ax, bin_index in zip(axes_arr, range(int(time_bin_count)), strict=False):
        block = [row for row in rows if int(row["time_bin"]) == int(bin_index)]
        xy = np.asarray([[row["image_feature_pc1"], row["image_feature_pc2"]] for row in block], dtype=np.float64)
        flow = np.asarray([[row["flow_tangent_pc1"], row["flow_tangent_pc2"]] for row in block], dtype=np.float64)
        coherence = np.asarray([row["flow_coherence"] for row in block], dtype=np.float64)
        ax.scatter(xy[:, 0], xy[:, 1], s=24, c=coherence, cmap="magma", alpha=0.82, linewidths=0, zorder=2)
        arrow_scale = np.clip(coherence, 0.0, 1.0) if bool(scale_by_coherence) else np.ones_like(coherence)
        ax.quiver(
            xy[:, 0],
            xy[:, 1],
            flow[:, 0] * arrow_len * arrow_scale,
            flow[:, 1] * arrow_len * arrow_scale,
            color="#0072B2",
            scale=1.0,
            scale_units="xy",
            angles="xy",
            width=0.0042,
            headwidth=3.6,
            headlength=4.6,
            alpha=0.86,
            zorder=3,
        )
        ax.set_xlim(float(center[0] - 0.58 * span), float(center[0] + 0.58 * span))
        ax.set_ylim(float(center[1] - 0.58 * span), float(center[1] + 0.58 * span))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"time bin {bin_index}")
        ax.set_xlabel("Image feature PC1")
        ax.grid(True, alpha=0.16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes_arr[0].set_ylabel("Image feature PC2")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=210)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    manifest_rows = _manifest_filter(_read_csv_rows(run_dir / "response_cache_manifest.csv"), args)
    if not manifest_rows:
        raise ValueError("No response-cache rows survived the filters")
    groups, meta = _collect_trajectories(run_dir, manifest_rows, variant=str(args.variant))
    if len(groups) < 2:
        raise ValueError("Need at least two trajectory groups")
    state_mean, _vector_mean, tangent_basis, tangent_evals = _fit_vector_pca(groups, n_components=max(3, int(args.n_components)))
    tangent_groups = _project_groups_to_basis(groups, state_mean, tangent_basis)
    centroids = np.asarray([np.mean(group[:, :3], axis=0) for group in tangent_groups], dtype=np.float64)
    selected = _farthest_subset(centroids, min(int(args.max_trajectories), len(tangent_groups)), seed=int(args.seed))
    tangent_total = float(np.sum(tangent_evals)) + 1e-12
    tangent_var_fraction = tangent_evals[:3] / tangent_total

    compact_dir = out_dir / "compact_tangent_trajectory_arrows"
    outputs: list[str] = []
    compact_title = f"{args.variant} trajectories in compact tangent PCs"
    for color_by in _parse_list(args.color_by):
        stem = f"{args.variant}_compact_tangent_pc123_arrows_by_{color_by}"
        png_path = compact_dir / f"{stem}.png"
        html_path = compact_dir / f"{stem}.html"
        _static_plot(
            png_path,
            tangent_groups,
            meta,
            selected,
            color_by=color_by,
            var_fraction=tangent_var_fraction,
            title=f"{compact_title}; colored by {color_by}",
            pc3_scale=float(args.pc3_scale),
            plot_style="arrows",
            arrow_stride=int(args.arrow_stride),
            normalize_arrows=bool(args.normalize_arrows),
        )
        outputs.extend([str(png_path), str(png_path.with_suffix(".pdf"))])
        if _interactive_plot(
            html_path,
            tangent_groups,
            meta,
            selected,
            color_by=color_by,
            title=f"{compact_title}; colored by {color_by}",
            pc3_scale=float(args.pc3_scale),
            plot_style="arrows",
            arrow_stride=int(args.arrow_stride),
            normalize_arrows=bool(args.normalize_arrows),
        ):
            outputs.append(str(html_path))

    feature_scores, image_scores, source_rows, feature_cols, image_evals = _source_feature_scores(
        run_dir / "selected_windows.csv",
        meta,
        include_position=bool(args.include_position),
    )
    buckets = _aggregate_feature_flow(groups, meta, tangent_basis, time_bin_count=int(args.time_bins))
    flow_rows = _candidate_flow_rows(feature_scores, buckets, time_bin_count=int(args.time_bins))
    feature_dir = out_dir / ("image_feature_flow_with_position" if args.include_position else "image_feature_flow")
    _write_csv(feature_dir / "image_feature_compact_tangent_flow_rows.csv", flow_rows)
    _plot_image_feature_flow(
        feature_dir / "image_feature_compact_tangent_flow_by_time_bin.png",
        flow_rows,
        time_bin_count=int(args.time_bins),
        title=f"{args.variant} compact-tangent flow over image-feature coordinates",
        scale_by_coherence=bool(args.scale_feature_arrows_by_coherence),
    )
    outputs.extend(
        [
            str(feature_dir / "image_feature_compact_tangent_flow_by_time_bin.png"),
            str(feature_dir / "image_feature_compact_tangent_flow_by_time_bin.pdf"),
            str(feature_dir / "image_feature_compact_tangent_flow_rows.csv"),
        ]
    )
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "variant": str(args.variant),
        "n_manifest_rows_selected": int(len(manifest_rows)),
        "n_trajectory_groups": int(len(groups)),
        "n_sources": int(len(source_rows)),
        "n_compact_tangent_trajectories_plotted": int(selected.size),
        "n_image_feature_flow_rows": int(len(flow_rows)),
        "compact_tangent_pca_fraction_first3": [float(v) for v in tangent_var_fraction.tolist()],
        "image_feature_pca_fraction_first3": [float(v) for v in (image_evals[:3] / (float(np.sum(image_evals)) + 1e-12)).tolist()],
        "image_feature_columns": feature_cols,
        "include_position": bool(args.include_position),
        "time_bins": int(args.time_bins),
        "image_feature_neighbor_flow_cosine_k8": _neighbor_cosines(flow_rows, k=8, n_shuffles=200, seed=int(args.seed) + 99),
        "outputs": outputs,
    }
    _write_json(out_dir / "trajectory_coordinate_space_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot BackImage trajectory flow in compact-tangent and image-feature coordinate spaces."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-set-modes", type=str, default="")
    parser.add_argument("--prior-families", type=str, default="")
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--variant", choices=("motion_delta", "prior_response"), default="motion_delta")
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--max-trajectories", type=int, default=32)
    parser.add_argument("--color-by", type=str, default="time_index,source_row,trajectory_index")
    parser.add_argument("--pc3-scale", type=float, default=3.0)
    parser.add_argument("--arrow-stride", type=int, default=1)
    parser.add_argument("--normalize-arrows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--time-bins", type=int, default=4)
    parser.add_argument("--include-position", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scale-feature-arrows-by-coherence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
