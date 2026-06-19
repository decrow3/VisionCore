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
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from declan.backimage_trajectory_observer.analyze_global_fixation_trajectory_flow import (
    _flatten_segments,
    _local_coherence,
    _neighbor_matrix,
    _sample_indices,
    _summarize_against_null,
    _write_csv,
)
from declan.backimage_trajectory_observer.plot_global_fixation_trajectory_lines_3d import (
    _collect_trajectories,
    _fit_pca_streamed,
    _manifest_filter,
    _parse_list,
    _project_group,
    _read_csv_rows,
    _safe_float,
    _static_plot,
    _write_json,
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


def _write_json_ready(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_row_from_meta(row: dict[str, object]) -> int:
    value = row.get("source_row")
    if value is None:
        value = str(row.get("candidate_id", "")).split(":")[-1]
    return int(_safe_float(value))


def _read_source_info(run_dir: Path, source_rows: list[int], coordinate_mode: str) -> pd.DataFrame:
    cols = [
        "source_row",
        "image_index",
        "stimulus",
        "mean_x_deg",
        "mean_y_deg",
        "image_patch_center_x_px",
        "image_patch_center_y_px",
        "image_patch_rms_contrast",
        "image_gradient_energy",
        "image_edge_density",
        "image_orientation_coherence",
    ]
    df = pd.read_csv(run_dir / "selected_windows.csv", usecols=lambda c: c in cols)
    df = df.drop_duplicates("source_row").set_index("source_row", drop=False)
    missing = sorted(set(int(v) for v in source_rows).difference(set(int(v) for v in df.index.tolist())))
    if missing:
        raise ValueError(f"{len(missing)} source rows missing from selected_windows.csv: {missing[:8]}")
    out = df.loc[np.asarray(sorted(set(int(v) for v in source_rows)), dtype=int)].reset_index(drop=True)
    if coordinate_mode == "patch_px":
        out["coord_x"] = pd.to_numeric(out["image_patch_center_x_px"], errors="coerce")
        out["coord_y"] = pd.to_numeric(out["image_patch_center_y_px"], errors="coerce")
        out["coord_unit"] = "px"
    elif coordinate_mode == "mean_deg":
        out["coord_x"] = pd.to_numeric(out["mean_x_deg"], errors="coerce")
        out["coord_y"] = pd.to_numeric(out["mean_y_deg"], errors="coerce")
        out["coord_unit"] = "deg"
    else:
        raise ValueError(f"Unknown coordinate_mode={coordinate_mode!r}")
    out = out[np.isfinite(out["coord_x"]) & np.isfinite(out["coord_y"])].copy()
    if out.empty:
        raise ValueError("No finite source coordinates")
    return out


def _densest_source_cluster(source_info: pd.DataFrame, n_sources: int) -> tuple[pd.DataFrame, dict[str, object]]:
    coords = source_info[["coord_x", "coord_y"]].to_numpy(dtype=np.float64)
    n_keep = min(int(n_sources), coords.shape[0])
    if n_keep < 2:
        raise ValueError("Need at least two local sources")
    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=n_keep)
    dist = np.asarray(dist, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    score = dist[:, -1]
    center_i = int(np.argmin(score))
    local_idx = idx[center_i]
    local = source_info.iloc[local_idx].copy()
    center = coords[center_i]
    local["distance_to_cluster_center"] = np.linalg.norm(
        local[["coord_x", "coord_y"]].to_numpy(dtype=np.float64) - center[None, :],
        axis=1,
    )
    local = local.sort_values("distance_to_cluster_center").reset_index(drop=True)
    info = {
        "cluster_center_source_row": int(source_info.iloc[center_i]["source_row"]),
        "cluster_center_coord": [float(v) for v in center.tolist()],
        "cluster_radius": float(np.max(local["distance_to_cluster_center"])),
        "cluster_median_radius": float(np.median(local["distance_to_cluster_center"])),
        "n_sources": int(local.shape[0]),
    }
    return local, info


def _source_group_indices(meta: list[dict[str, object]], sources: set[int]) -> np.ndarray:
    return np.asarray([i for i, row in enumerate(meta) if _source_row_from_meta(row) in sources], dtype=np.int64)


def _subset(values: list[Any], idx: np.ndarray) -> list[Any]:
    return [values[int(i)] for i in np.asarray(idx, dtype=np.int64).tolist()]


def _source_centroids(projected_groups: list[np.ndarray], meta: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    by_source: dict[int, list[np.ndarray]] = {}
    for group, row in zip(projected_groups, meta, strict=False):
        src = _source_row_from_meta(row)
        by_source.setdefault(src, []).append(np.mean(np.asarray(group[:, :3], dtype=np.float64), axis=0))
    for src, vals in sorted(by_source.items()):
        arr = np.mean(np.asarray(vals, dtype=np.float64), axis=0)
        rows.append({"source_row": src, "pc1": arr[0], "pc2": arr[1], "pc3": arr[2]})
    return pd.DataFrame(rows)


def _pairwise_upper(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    dist = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
    return dist[np.triu_indices(arr.shape[0], k=1)]


def _centroid_smoothness(
    source_info: pd.DataFrame,
    centroids: pd.DataFrame,
) -> dict[str, object]:
    merged = source_info.merge(centroids, on="source_row", how="inner")
    xy = merged[["coord_x", "coord_y"]].to_numpy(dtype=np.float64)
    pc = merged[["pc1", "pc2", "pc3"]].to_numpy(dtype=np.float64)
    d_space = _pairwise_upper(xy)
    d_neural = _pairwise_upper(pc)
    corr = spearmanr(d_space, d_neural).statistic if d_space.size >= 3 else float("nan")
    return {
        "n_sources_with_centroids": int(merged.shape[0]),
        "physical_pair_distance_median": float(np.median(d_space)) if d_space.size else float("nan"),
        "physical_pair_distance_max": float(np.max(d_space)) if d_space.size else float("nan"),
        "neural_centroid_pair_distance_median": float(np.median(d_neural)) if d_neural.size else float("nan"),
        "space_neural_distance_spearman": float(corr) if np.isfinite(corr) else float("nan"),
    }


def _flow_smoothness(
    projected_groups: list[np.ndarray],
    *,
    neighbor_k: int,
    query_k: int,
    max_eval_segments: int,
    n_nulls: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    segments = _flatten_segments(projected_groups)
    eval_idx = _sample_indices(segments["midpoint"].shape[0], int(max_eval_segments), rng)
    nn = _neighbor_matrix(
        segments["midpoint"],
        segments["group_id"],
        eval_idx,
        neighbor_k=int(neighbor_k),
        query_k=int(query_k),
        exclude_same_group=True,
    )
    obs = _local_coherence(segments["unit_vector"], nn)
    null_mean = []
    null_median = []
    n_segments = int(segments["midpoint"].shape[0])
    for _ in range(int(n_nulls)):
        perm = rng.permutation(n_segments)
        coh = _local_coherence(segments["unit_vector"][perm], nn)
        null_mean.append(float(np.mean(coh)))
        null_median.append(float(np.median(coh)))
    rows = [
        _summarize_against_null("local_direction_coherence_mean", float(np.mean(obs)), np.asarray(null_mean)),
        _summarize_against_null("local_direction_coherence_median", float(np.median(obs)), np.asarray(null_median)),
    ]
    return {
        "n_segments_total": int(n_segments),
        "n_eval_segments_with_neighbors": int(nn.shape[0]),
        "neighbor_k": int(neighbor_k),
        "n_nulls": int(n_nulls),
        "rows": rows,
    }


def _plot_source_map(out_path: Path, source_info: pd.DataFrame, local_sources: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    ax.scatter(source_info["coord_x"], source_info["coord_y"], s=28, color="0.72", label="available sources")
    ax.scatter(local_sources["coord_x"], local_sources["coord_y"], s=52, color="#0072B2", label="local cluster")
    for row in local_sources.itertuples(index=False):
        ax.text(float(row.coord_x), float(row.coord_y), str(int(row.source_row)), fontsize=7, alpha=0.75)
    unit = str(local_sources["coord_unit"].iloc[0]) if "coord_unit" in local_sources else ""
    ax.set_xlabel(f"fixation/window x ({unit})")
    ax.set_ylabel(f"fixation/window y ({unit})")
    ax.set_title("Selected dense local fixation neighborhood")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = _manifest_filter(_read_csv_rows(run_dir / "response_cache_manifest.csv"), args)
    if not manifest_rows:
        raise ValueError("No response-cache rows survived filters")
    groups, meta = _collect_trajectories(run_dir, manifest_rows, variant=str(args.variant))
    source_rows = [_source_row_from_meta(row) for row in meta]
    unique_sources = sorted(set(source_rows))
    source_info = _read_source_info(run_dir, unique_sources, str(args.coordinate_mode))
    local_sources, cluster_info = _densest_source_cluster(source_info, n_sources=int(args.local_sources))
    local_set = set(int(v) for v in local_sources["source_row"].tolist())
    local_idx = _source_group_indices(meta, local_set)
    if local_idx.size < 2:
        raise ValueError("Selected local cluster has fewer than two trajectory groups")
    local_groups = _subset(groups, local_idx)
    local_meta = _subset(meta, local_idx)

    global_mean, global_basis, global_evals = _fit_pca_streamed(groups, n_components=max(3, int(args.n_components)))
    global_projected = [_project_group(group, global_mean, global_basis) for group in groups]
    local_global_projected = _subset(global_projected, local_idx)
    local_mean, local_basis, local_evals = _fit_pca_streamed(local_groups, n_components=max(3, int(args.n_components)))
    local_projected = [_project_group(group, local_mean, local_basis) for group in local_groups]
    local_global_centroids = np.asarray([np.mean(group[:, :3], axis=0) for group in local_global_projected], dtype=np.float64)
    local_selected = np.arange(len(local_groups), dtype=np.int64)
    if local_selected.size > int(args.max_trajectories):
        # Select a readable spread of trajectory groups within this local patch.
        from declan.backimage_trajectory_observer.plot_global_fixation_trajectory_lines_3d import _farthest_subset

        local_selected = _farthest_subset(local_global_centroids, int(args.max_trajectories), seed=int(args.seed))

    global_var = global_evals / (float(np.sum(global_evals)) + 1e-12)
    local_var = local_evals / (float(np.sum(local_evals)) + 1e-12)
    _static_plot(
        out_dir / "local_cluster_in_global_fan_pca.png",
        local_global_projected,
        local_meta,
        local_selected,
        color_by="time_index",
        var_fraction=global_var[:3],
        title=f"Local nearby-fixation trajectories in global fan PCA ({args.variant})",
        pc3_scale=float(args.pc3_scale),
        plot_style="lines",
        arrow_stride=1,
        normalize_arrows=False,
    )
    _static_plot(
        out_dir / "local_cluster_local_pca.png",
        local_projected,
        local_meta,
        local_selected,
        color_by="time_index",
        var_fraction=local_var[:3],
        title=f"Local nearby-fixation trajectories in local PCA ({args.variant})",
        pc3_scale=float(args.pc3_scale),
        plot_style="lines",
        arrow_stride=1,
        normalize_arrows=False,
    )
    _plot_source_map(out_dir / "local_source_map.png", source_info, local_sources)

    source_centroid_global = _source_centroids(local_global_projected, local_meta)
    source_centroid_local = _source_centroids(local_projected, local_meta)
    smooth_global = _centroid_smoothness(local_sources, source_centroid_global)
    smooth_local = _centroid_smoothness(local_sources, source_centroid_local)
    flow_global = _flow_smoothness(
        local_global_projected,
        neighbor_k=int(args.neighbor_k),
        query_k=int(args.query_k),
        max_eval_segments=int(args.max_eval_segments),
        n_nulls=int(args.n_nulls),
        seed=int(args.seed),
    )
    flow_local = _flow_smoothness(
        local_projected,
        neighbor_k=int(args.neighbor_k),
        query_k=int(args.query_k),
        max_eval_segments=int(args.max_eval_segments),
        n_nulls=int(args.n_nulls),
        seed=int(args.seed) + 11,
    )
    metric_rows = []
    for row in flow_global["rows"]:
        metric_rows.append({"coordinate_space": "global_fan_pca", **row})
    for row in flow_local["rows"]:
        metric_rows.append({"coordinate_space": "local_pca", **row})
    _write_csv(out_dir / "local_flow_smoothness_metrics.csv", metric_rows)
    local_sources.to_csv(out_dir / "local_source_cluster.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "variant": str(args.variant),
        "filters": {
            "candidate_set_modes": _parse_list(args.candidate_set_modes),
            "prior_families": _parse_list(args.prior_families),
            "scales": _parse_list(args.scales),
        },
        "coordinate_mode": str(args.coordinate_mode),
        "n_manifest_rows_selected": int(len(manifest_rows)),
        "n_unique_sources_available": int(len(unique_sources)),
        "n_local_sources": int(local_sources.shape[0]),
        "n_local_trajectory_groups": int(len(local_groups)),
        "n_local_points": int(sum(group.shape[0] for group in local_groups)),
        "cluster": cluster_info,
        "global_pca_explained_first3": [float(v) for v in global_var[:3].tolist()],
        "local_pca_explained_first3": [float(v) for v in local_var[:3].tolist()],
        "centroid_smoothness_global_pca": smooth_global,
        "centroid_smoothness_local_pca": smooth_local,
        "flow_smoothness_global_pca": {
            key: value for key, value in flow_global.items() if key != "rows"
        },
        "flow_smoothness_local_pca": {
            key: value for key, value in flow_local.items() if key != "rows"
        },
        "outputs": [
            "local_source_map.png",
            "local_cluster_in_global_fan_pca.png",
            "local_cluster_local_pca.png",
            "local_source_cluster.csv",
            "local_flow_smoothness_metrics.csv",
        ],
    }
    _write_json_ready(out_dir / "local_fixation_fan_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot and quantify a dense local fixation neighborhood in BackImage fan geometry."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-set-modes", type=str, default="")
    parser.add_argument("--prior-families", type=str, default="")
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--variant", choices=("motion_delta", "prior_response"), default="motion_delta")
    parser.add_argument("--coordinate-mode", choices=("patch_px", "mean_deg"), default="patch_px")
    parser.add_argument("--local-sources", type=int, default=16)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--max-trajectories", type=int, default=192)
    parser.add_argument("--pc3-scale", type=float, default=1.0)
    parser.add_argument("--neighbor-k", type=int, default=64)
    parser.add_argument("--query-k", type=int, default=512)
    parser.add_argument("--max-eval-segments", type=int, default=20000)
    parser.add_argument("--n-nulls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
