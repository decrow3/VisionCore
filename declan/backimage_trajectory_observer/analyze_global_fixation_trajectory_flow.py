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
from scipy.spatial import cKDTree

from declan.backimage_trajectory_observer.plot_global_fixation_trajectory_lines_3d import (
    _collect_trajectories,
    _fit_pca_streamed,
    _manifest_filter,
    _parse_list,
    _project_group,
    _read_csv_rows,
    _scaled_groups,
    _write_json,
)


def _unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + float(eps))


def _finite_ci(values: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def _one_sided_p_ge(null_values: np.ndarray, observed: float) -> float:
    vals = np.asarray(null_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1 + np.sum(vals >= observed)) / (vals.size + 1))


def _z_score(observed: float, null_values: np.ndarray) -> float:
    vals = np.asarray(null_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
    if not np.isfinite(sd) or sd <= 1e-12:
        return float("nan")
    return float((observed - float(np.mean(vals))) / sd)


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


def _flatten_segments(projected_groups: list[np.ndarray]) -> dict[str, np.ndarray]:
    starts: list[np.ndarray] = []
    stops: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    time_indices: list[np.ndarray] = []
    for group_index, group in enumerate(projected_groups):
        coords = np.asarray(group[:, :3], dtype=np.float64)
        if coords.shape[0] < 2 or not np.isfinite(coords).all():
            continue
        n_seg = coords.shape[0] - 1
        starts.append(coords[:-1])
        stops.append(coords[1:])
        group_ids.append(np.full(n_seg, group_index, dtype=np.int32))
        time_indices.append(np.arange(n_seg, dtype=np.int32))
    if not starts:
        raise ValueError("No finite trajectory segments available")
    start = np.concatenate(starts, axis=0)
    stop = np.concatenate(stops, axis=0)
    vectors = stop - start
    midpoints = 0.5 * (start + stop)
    norms = np.linalg.norm(vectors, axis=1)
    keep = np.isfinite(midpoints).all(axis=1) & np.isfinite(vectors).all(axis=1) & (norms > 1e-12)
    return {
        "start": start[keep],
        "stop": stop[keep],
        "midpoint": midpoints[keep],
        "vector": vectors[keep],
        "unit_vector": _unit_rows(vectors[keep]),
        "group_id": np.concatenate(group_ids, axis=0)[keep],
        "time_index": np.concatenate(time_indices, axis=0)[keep],
    }


def _sample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(int(n), dtype=np.int64)
    if int(max_n) > 0 and idx.size > int(max_n):
        idx = np.sort(rng.choice(idx, size=int(max_n), replace=False))
    return idx


def _neighbor_matrix(
    midpoints: np.ndarray,
    group_ids: np.ndarray,
    eval_idx: np.ndarray,
    *,
    neighbor_k: int,
    query_k: int,
    exclude_same_group: bool,
) -> np.ndarray:
    k = min(max(int(query_k), int(neighbor_k) + 1), midpoints.shape[0])
    tree = cKDTree(np.asarray(midpoints, dtype=np.float64))
    _dist, idx = tree.query(midpoints[eval_idx], k=k)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx[:, None]
    rows: list[np.ndarray] = []
    for row, segment_index in zip(idx, eval_idx, strict=False):
        valid = row[(row >= 0) & (row < midpoints.shape[0]) & (row != int(segment_index))]
        if bool(exclude_same_group):
            valid = valid[group_ids[valid] != group_ids[int(segment_index)]]
        if valid.size >= int(neighbor_k):
            rows.append(valid[: int(neighbor_k)])
    if not rows:
        raise ValueError("No evaluation segments had enough neighbors")
    return np.stack(rows, axis=0)


def _local_coherence(unit_vectors: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    local = np.asarray(unit_vectors, dtype=np.float64)[np.asarray(neighbor_idx, dtype=np.int64)]
    return np.linalg.norm(np.mean(local, axis=1), axis=1)


def _linear_flow_fit(midpoints: np.ndarray, vectors: np.ndarray) -> dict[str, Any]:
    x = np.asarray(midpoints, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    center = np.mean(x, axis=0)
    xc = x - center[None, :]
    v_center = np.mean(v, axis=0)
    matrix, *_ = np.linalg.lstsq(xc, v, rcond=None)
    pred = xc @ matrix
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - v_center[None, :]) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    skew = 0.5 * (matrix - matrix.T)
    sym = 0.5 * (matrix + matrix.T)
    pred_skew = xc @ skew
    pred_sym = xc @ sym
    skew_energy = float(np.sum(pred_skew * pred_skew))
    sym_energy = float(np.sum(pred_sym * pred_sym))
    axis = np.asarray([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)
    return {
        "center": center,
        "matrix": matrix,
        "skew": skew,
        "symmetric": sym,
        "r2": r2,
        "matrix_frobenius": float(np.linalg.norm(matrix)),
        "skew_frobenius": float(np.linalg.norm(skew)),
        "symmetric_frobenius": float(np.linalg.norm(sym)),
        "skew_frobenius_fraction": float(np.linalg.norm(skew) / (np.linalg.norm(matrix) + 1e-12)),
        "skew_predicted_energy_share": float(skew_energy / (skew_energy + sym_energy + 1e-12)),
        "symmetric_predicted_energy_share": float(sym_energy / (skew_energy + sym_energy + 1e-12)),
        "rotation_axis": axis,
        "rotation_axis_unit": axis / (np.linalg.norm(axis) + 1e-12),
    }


def _angular_metrics(midpoints: np.ndarray, vectors: np.ndarray) -> dict[str, float | list[float]]:
    x = np.asarray(midpoints, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    xc = x - np.mean(x, axis=0, keepdims=True)
    unit_v = _unit_rows(v)
    angular = np.cross(xc, v)
    angular_norm = np.linalg.norm(angular, axis=1)
    keep = angular_norm > 1e-12
    if not np.any(keep):
        return {
            "angular_momentum_unit_coherence": float("nan"),
            "angular_axis_unit": [float("nan"), float("nan"), float("nan")],
            "tangential_alignment_mean": float("nan"),
            "tangential_alignment_abs_mean": float("nan"),
        }
    unit_l = angular[keep] / angular_norm[keep, None]
    mean_l = np.mean(unit_l, axis=0)
    axis = mean_l / (np.linalg.norm(mean_l) + 1e-12)
    tangent = np.cross(axis[None, :], xc)
    tangent = _unit_rows(tangent)
    cos = np.sum(unit_v * tangent, axis=1)
    cos = cos[np.isfinite(cos)]
    return {
        "angular_momentum_unit_coherence": float(np.linalg.norm(mean_l)),
        "angular_axis_unit": [float(v) for v in axis.tolist()],
        "tangential_alignment_mean": float(np.mean(cos)) if cos.size else float("nan"),
        "tangential_alignment_abs_mean": float(np.mean(np.abs(cos))) if cos.size else float("nan"),
    }


def _summarize_against_null(metric: str, observed: float, null_values: np.ndarray) -> dict[str, object]:
    vals = np.asarray(null_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    null_mean = float(np.mean(vals)) if vals.size else float("nan")
    return {
        "metric": metric,
        "observed": float(observed),
        "null_mean": null_mean,
        "null_sd": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
        "null_ci_low": _finite_ci(vals, 2.5),
        "null_ci_high": _finite_ci(vals, 97.5),
        "lift_over_null_mean": float(observed / (null_mean + 1e-12)) if np.isfinite(null_mean) else float("nan"),
        "z": _z_score(observed, vals),
        "p_ge": _one_sided_p_ge(vals, observed),
    }


def _plot_null_summaries(out_path: Path, rows: list[dict[str, object]], nulls: dict[str, np.ndarray]) -> None:
    metrics = [str(row["metric"]) for row in rows if str(row["metric"]) in nulls]
    if not metrics:
        return
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, max(2.4 * n, 3.0)), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for ax, metric in zip(axes_arr, metrics, strict=False):
        row = next(r for r in rows if str(r["metric"]) == metric)
        vals = np.asarray(nulls[metric], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=28, color="0.72", edgecolor="white")
        ax.axvline(float(row["observed"]), color="#D55E00", lw=2.0, label="observed")
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("null value")
        ax.set_ylabel("count")
        ax.legend(frameon=False, loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.seed))
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    manifest_path = run_dir / "response_cache_manifest.csv"
    manifest_rows = _manifest_filter(_read_csv_rows(manifest_path), args)
    if not manifest_rows:
        raise ValueError(f"No response-cache rows selected from {manifest_path}")

    groups, _meta = _collect_trajectories(run_dir, manifest_rows, variant=str(args.variant))
    if len(groups) < 2:
        raise ValueError("Need at least two trajectory groups")
    mean, basis, evals = _fit_pca_streamed(groups, n_components=max(3, int(args.n_components)))
    total = float(np.sum(evals)) + 1e-12
    var_fraction = evals[:3] / total
    projected_groups = [_project_group(group, mean, basis) for group in groups]
    projected_groups = _scaled_groups(projected_groups, float(args.pc3_scale))
    segments = _flatten_segments(projected_groups)

    fit_idx = _sample_indices(segments["midpoint"].shape[0], int(args.max_fit_segments), rng)
    eval_idx = _sample_indices(segments["midpoint"].shape[0], int(args.max_eval_segments), rng)
    neighbor_idx = _neighbor_matrix(
        segments["midpoint"],
        segments["group_id"],
        eval_idx,
        neighbor_k=int(args.neighbor_k),
        query_k=int(args.query_k),
        exclude_same_group=bool(args.exclude_same_trajectory),
    )
    obs_local = _local_coherence(segments["unit_vector"], neighbor_idx)
    obs_local_mean = float(np.mean(obs_local))
    obs_local_median = float(np.median(obs_local))
    fit = _linear_flow_fit(segments["midpoint"][fit_idx], segments["vector"][fit_idx])
    angular = _angular_metrics(segments["midpoint"][fit_idx], segments["vector"][fit_idx])

    null_local_mean = []
    null_local_median = []
    null_r2 = []
    null_skew_fraction = []
    null_skew_pred_share = []
    null_angular_coherence = []
    null_tangential_mean = []
    n_segments = int(segments["midpoint"].shape[0])
    for _ in range(int(args.n_nulls)):
        perm = rng.permutation(n_segments)
        null_units = segments["unit_vector"][perm]
        coh = _local_coherence(null_units, neighbor_idx)
        null_local_mean.append(float(np.mean(coh)))
        null_local_median.append(float(np.median(coh)))
        null_vectors_fit = segments["vector"][perm[fit_idx]]
        null_fit = _linear_flow_fit(segments["midpoint"][fit_idx], null_vectors_fit)
        null_ang = _angular_metrics(segments["midpoint"][fit_idx], null_vectors_fit)
        null_r2.append(float(null_fit["r2"]))
        null_skew_fraction.append(float(null_fit["skew_frobenius_fraction"]))
        null_skew_pred_share.append(float(null_fit["skew_predicted_energy_share"]))
        null_angular_coherence.append(float(null_ang["angular_momentum_unit_coherence"]))
        null_tangential_mean.append(float(null_ang["tangential_alignment_mean"]))

    nulls = {
        "local_direction_coherence_mean": np.asarray(null_local_mean),
        "local_direction_coherence_median": np.asarray(null_local_median),
        "linear_flow_r2": np.asarray(null_r2),
        "linear_skew_frobenius_fraction": np.asarray(null_skew_fraction),
        "linear_skew_predicted_energy_share": np.asarray(null_skew_pred_share),
        "angular_momentum_unit_coherence": np.asarray(null_angular_coherence),
        "tangential_alignment_mean": np.asarray(null_tangential_mean),
    }
    rows = [
        _summarize_against_null("local_direction_coherence_mean", obs_local_mean, nulls["local_direction_coherence_mean"]),
        _summarize_against_null(
            "local_direction_coherence_median", obs_local_median, nulls["local_direction_coherence_median"]
        ),
        _summarize_against_null("linear_flow_r2", float(fit["r2"]), nulls["linear_flow_r2"]),
        _summarize_against_null(
            "linear_skew_frobenius_fraction",
            float(fit["skew_frobenius_fraction"]),
            nulls["linear_skew_frobenius_fraction"],
        ),
        _summarize_against_null(
            "linear_skew_predicted_energy_share",
            float(fit["skew_predicted_energy_share"]),
            nulls["linear_skew_predicted_energy_share"],
        ),
        _summarize_against_null(
            "angular_momentum_unit_coherence",
            float(angular["angular_momentum_unit_coherence"]),
            nulls["angular_momentum_unit_coherence"],
        ),
        _summarize_against_null(
            "tangential_alignment_mean",
            float(angular["tangential_alignment_mean"]),
            nulls["tangential_alignment_mean"],
        ),
    ]

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
        "pc3_scale": float(args.pc3_scale),
        "n_manifest_rows_selected": int(len(manifest_rows)),
        "n_trajectory_groups": int(len(groups)),
        "n_segments_total": int(n_segments),
        "n_segments_fit": int(fit_idx.size),
        "n_segments_eval_requested": int(eval_idx.size),
        "n_segments_eval_with_neighbors": int(neighbor_idx.shape[0]),
        "neighbor_k": int(args.neighbor_k),
        "query_k": int(args.query_k),
        "exclude_same_trajectory": bool(args.exclude_same_trajectory),
        "n_nulls": int(args.n_nulls),
        "pca_explained_fraction_first3": [float(v) for v in var_fraction.tolist()],
        "linear_flow_matrix": np.asarray(fit["matrix"]).tolist(),
        "linear_flow_skew_matrix": np.asarray(fit["skew"]).tolist(),
        "linear_flow_symmetric_matrix": np.asarray(fit["symmetric"]).tolist(),
        "linear_rotation_axis_unit": np.asarray(fit["rotation_axis_unit"]).tolist(),
        "angular_axis_unit": angular["angular_axis_unit"],
        "observed_local_coherence_quantiles": {
            "q05": float(np.percentile(obs_local, 5)),
            "q25": float(np.percentile(obs_local, 25)),
            "q50": obs_local_median,
            "q75": float(np.percentile(obs_local, 75)),
            "q95": float(np.percentile(obs_local, 95)),
        },
        "metric_rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "trajectory_flow_metric_summary.json", summary)
    _write_csv(out_dir / "trajectory_flow_metric_rows.csv", rows)
    _plot_null_summaries(out_dir / "trajectory_flow_metric_nulls.png", rows, nulls)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantify local flow coherence and rotational structure in BackImage trajectory PC space."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-set-modes", type=str, default="")
    parser.add_argument("--prior-families", type=str, default="")
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--variant", choices=("motion_delta", "prior_response"), default="motion_delta")
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--pc3-scale", type=float, default=1.0)
    parser.add_argument("--neighbor-k", type=int, default=64)
    parser.add_argument("--query-k", type=int, default=512)
    parser.add_argument("--exclude-same-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-eval-segments", type=int, default=20000)
    parser.add_argument("--max-fit-segments", type=int, default=50000)
    parser.add_argument("--n-nulls", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
