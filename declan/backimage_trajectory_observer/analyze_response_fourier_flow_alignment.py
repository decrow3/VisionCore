from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np

from declan.backimage_trajectory_observer.plot_global_fixation_trajectory_lines_3d import (
    _collect_trajectories,
    _manifest_filter,
    _parse_list,
    _read_csv_rows,
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


def _standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, keepdims=True)
    sd = np.std(train, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return (train - mean) / sd, (test - mean) / sd, mean, sd


def _fit_pca_train(train: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(train, dtype=np.float64) - np.mean(train, axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(int(n_components), vt.shape[0])
    evals = (s * s) / max(centered.shape[0] - 1, 1)
    return vt[:k].T, evals


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    reg = float(alpha) * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(xtx + reg, x.T @ y)


def _sample_rows(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(int(n), dtype=np.int64)
    if int(max_n) > 0 and idx.size > int(max_n):
        return np.sort(rng.choice(idx, size=int(max_n), replace=False))
    return idx


def _flatten_segments(
    x_groups: list[np.ndarray],
    y_groups: list[np.ndarray],
    trace_groups: list[np.ndarray],
    source_rows: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    traces: list[np.ndarray] = []
    srcs: list[np.ndarray] = []
    for x, y, trace, source_row in zip(x_groups, y_groups, trace_groups, source_rows, strict=False):
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        tr = np.asarray(trace, dtype=np.float32)
        n = min(x_arr.shape[0], y_arr.shape[0], tr.shape[0])
        if n < 2:
            continue
        dx = np.diff(x_arr[:n], axis=0)
        dy = np.diff(y_arr[:n], axis=0)
        dt = np.diff(tr[:n], axis=0)
        keep = (
            np.isfinite(dx).all(axis=1)
            & np.isfinite(dy).all(axis=1)
            & np.isfinite(dt).all(axis=1)
            & (np.linalg.norm(dx, axis=1) > 1e-12)
            & (np.linalg.norm(dy, axis=1) > 1e-12)
        )
        if not np.any(keep):
            continue
        xs.append(dx[keep])
        ys.append(dy[keep])
        traces.append(dt[keep])
        srcs.append(np.full(int(np.count_nonzero(keep)), int(source_row), dtype=np.int64))
    if not xs:
        raise ValueError("No finite paired segments")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(traces), np.concatenate(srcs)


def _cv_source_folds(source_rows: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.asarray(sorted(set(int(v) for v in source_rows.tolist())), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique)
    folds = np.array_split(unique, min(int(n_folds), unique.size))
    out = []
    for fold_sources in folds:
        test_mask = np.isin(source_rows, fold_sources)
        train_mask = ~test_mask
        if np.count_nonzero(test_mask) and np.count_nonzero(train_mask):
            out.append((train_mask, test_mask))
    return out


def _predict_alignment(
    x: np.ndarray,
    y: np.ndarray,
    source_rows: np.ndarray,
    *,
    y_pca_components: int,
    ridge_alpha: float,
    max_train_segments: int,
    max_test_segments: int,
    n_folds: int,
    seed: int,
    label_prefix: str,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    fold_rows: list[dict[str, object]] = []
    cosines_all: list[np.ndarray] = []
    r2_num = 0.0
    r2_den = 0.0
    for fold_i, (train_mask, test_mask) in enumerate(_cv_source_folds(source_rows, n_folds, seed)):
        train_idx_full = np.flatnonzero(train_mask)
        test_idx_full = np.flatnonzero(test_mask)
        train_idx = train_idx_full[_sample_rows(train_idx_full.size, int(max_train_segments), rng)]
        test_idx = test_idx_full[_sample_rows(test_idx_full.size, int(max_test_segments), rng)]
        x_train_raw = np.asarray(x[train_idx], dtype=np.float64)
        x_test_raw = np.asarray(x[test_idx], dtype=np.float64)
        y_train_raw = np.asarray(y[train_idx], dtype=np.float64)
        y_test_raw = np.asarray(y[test_idx], dtype=np.float64)
        y_train_z, y_test_z, _ym, _ys = _standardize_train_test(y_train_raw, y_test_raw)
        basis, _evals = _fit_pca_train(y_train_z, int(y_pca_components))
        y_train = y_train_z @ basis
        y_test = y_test_z @ basis
        x_train, x_test, _xm, _xs = _standardize_train_test(x_train_raw, x_test_raw)
        w = _ridge_fit(x_train, y_train, alpha=float(ridge_alpha))
        pred = x_test @ w
        center = np.mean(y_train, axis=0, keepdims=True)
        ss_res = float(np.sum((y_test - pred) ** 2))
        ss_tot = float(np.sum((y_test - center) ** 2)) + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)
        cos = np.sum(_unit_rows(pred) * _unit_rows(y_test), axis=1)
        cos = cos[np.isfinite(cos)]
        cosines_all.append(cos)
        r2_num += ss_res
        r2_den += ss_tot
        fold_rows.append(
            {
                "predictor": label_prefix,
                "fold": int(fold_i),
                "n_train_segments": int(train_idx.size),
                "n_test_segments": int(test_idx.size),
                "response_pc_r2": r2,
                "direction_cosine_mean": float(np.mean(cos)) if cos.size else float("nan"),
                "direction_cosine_median": float(np.median(cos)) if cos.size else float("nan"),
            }
        )
    pooled = np.concatenate(cosines_all) if cosines_all else np.empty(0, dtype=np.float64)
    return {
        "predictor": label_prefix,
        "n_folds": int(len(fold_rows)),
        "pooled_response_pc_r2": float(1.0 - r2_num / (r2_den + 1e-12)) if r2_den > 0 else float("nan"),
        "pooled_direction_cosine_mean": float(np.mean(pooled)) if pooled.size else float("nan"),
        "pooled_direction_cosine_median": float(np.median(pooled)) if pooled.size else float("nan"),
        "fold_rows": fold_rows,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    manifest_rows = _manifest_filter(_read_csv_rows(run_dir / "response_cache_manifest.csv"), args)
    if not manifest_rows:
        raise ValueError("No response-cache rows survived filters")
    response_groups, meta = _collect_trajectories(run_dir, manifest_rows, variant=str(args.variant))
    trace_map_raw = _reconstruct_prior_trace_map(run_dir, manifest_rows)
    trace_rng = np.random.default_rng(int(args.seed) + 17)
    trace_map = {
        trace_id: _controlled_trace(trace, mode=str(args.trace_control), rng=trace_rng)
        for trace_id, trace in sorted(trace_map_raw.items())
    }
    source_rows = [_safe_int(str(row["candidate_id"]).split(":")[-1]) for row in meta]
    trace_groups = [trace_map[str(row["trajectory_id"])] for row in meta]
    fft_cache = _fft_cache_for_sources(
        run_dir,
        sorted(set(source_rows)),
        patch_size_px=int(args.patch_size_px),
        component_crop_px=int(args.component_crop_px),
        cache_path=(Path(args.fourier_cache) if args.fourier_cache is not None else None),
    )
    bands = _parse_bands(args.frequency_bands)
    selected = _select_frequency_components(fft_cache, bands, n_per_band=int(args.components_per_band))

    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for band_name in ["all_bands", *[name for name, _lo, _hi in bands if name in selected]]:
        spec = selected[band_name]
        fourier_groups = [
            _component_group(
                fft_cache[int(source_row)],
                trace_map[str(row["trajectory_id"])],
                list(spec["indices"]),
                list(spec["avg_power"]),
                normalization=str(args.component_normalization),
            )
            for source_row, row in zip(source_rows, meta, strict=False)
        ]
        x, y, trace_vel, src = _flatten_segments(fourier_groups, response_groups, trace_groups, source_rows)
        for label, predictor in (
            (f"fourier_{band_name}", x),
            (f"trace_velocity_{band_name}", trace_vel),
        ):
            result = _predict_alignment(
                predictor,
                y,
                src,
                y_pca_components=int(args.response_pca_components),
                ridge_alpha=float(args.ridge_alpha),
                max_train_segments=int(args.max_train_segments),
                max_test_segments=int(args.max_test_segments),
                n_folds=int(args.n_folds),
                seed=int(args.seed),
                label_prefix=label,
            )
            summary_rows.append(
                {
                    "band_name": band_name,
                    "predictor": label,
                    "trace_control": str(args.trace_control),
                    "component_normalization": str(args.component_normalization),
                    "n_complex_components": int(len(spec["indices"])),
                    "mean_cpd": float(spec["mean_cpd"]),
                    "n_segments": int(x.shape[0]),
                    "n_folds": int(result["n_folds"]),
                    "pooled_response_pc_r2": float(result["pooled_response_pc_r2"]),
                    "pooled_direction_cosine_mean": float(result["pooled_direction_cosine_mean"]),
                    "pooled_direction_cosine_median": float(result["pooled_direction_cosine_median"]),
                }
            )
            fold_rows.extend(
                {
                    "band_name": band_name,
                    **row,
                }
                for row in result["fold_rows"]
            )

    payload = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "variant": str(args.variant),
        "trace_control": str(args.trace_control),
        "component_normalization": str(args.component_normalization),
        "n_manifest_rows_selected": int(len(manifest_rows)),
        "n_trajectory_groups": int(len(meta)),
        "n_candidate_sources": int(len(set(source_rows))),
        "summary_rows": summary_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "response_fourier_flow_alignment_summary.csv", summary_rows)
    _write_csv(out_dir / "response_fourier_flow_alignment_folds.csv", fold_rows)
    _write_json(out_dir / "response_fourier_flow_alignment_summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-validate whether Fourier phase-flow vectors predict V1 response-flow vectors."
    )
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
    parser.add_argument("--frequency-bands", type=str, default="0.5:2,2:4,4:8,8:16")
    parser.add_argument("--components-per-band", type=int, default=16)
    parser.add_argument(
        "--component-normalization",
        choices=("power_whitened", "unit_amplitude", "raw"),
        default="power_whitened",
    )
    parser.add_argument("--trace-control", choices=("real", "time_permute", "reverse"), default="real")
    parser.add_argument("--response-pca-components", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-train-segments", type=int, default=60000)
    parser.add_argument("--max-test-segments", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(json.dumps(analyze(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
