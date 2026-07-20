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
import numpy as np
import pandas as pd
from tqdm import tqdm

from declan.backimage_trajectory_observer.plot_global_fixation_trajectory_lines_3d import (
    _collect_trajectories,
    _farthest_subset,
    _interactive_plot,
    _manifest_filter,
    _parse_list,
    _project_group,
    _read_csv_rows,
    _static_plot,
    _write_json,
)
from declan.backimage_trajectory_observer.analyze_global_fixation_trajectory_flow import (
    _flatten_segments,
    _local_coherence,
    _neighbor_matrix,
    _sample_indices,
    _summarize_against_null,
    _write_csv,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import (
    _child_rng,
    _extract_patch,
    _prior_trajectories,
    _session_dataset_cache,
    _trace_from_item,
    build_parser as build_observer_parser,
)
from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import _build_trace_bank


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


def _observer_args_from_metadata(run_dir: Path) -> argparse.Namespace:
    parser = build_observer_parser()
    args = parser.parse_args([])
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    for key, value in dict(metadata.get("config", {})).items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_int(value: object, default: int = -1) -> int:
    try:
        out = int(float(value))
    except Exception:
        return int(default)
    return out


def _reconstruct_prior_trace_map(run_dir: Path, manifest_rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    args = _observer_args_from_metadata(run_dir)
    work = pd.read_csv(run_dir / "selected_windows.csv")
    candidate_rows = {int(row["trial_id"]): row for row in _read_csv_dict(run_dir / "candidate_sets.csv")}
    needed: dict[tuple[int, str, float, str], set[str]] = {}
    for row in manifest_rows:
        table_path = run_dir / str(row["response_cache_path"])
        with np.load(table_path, allow_pickle=True) as table:
            ids = [str(v) for v in np.asarray(table["prior_trajectory_ids"]).astype(str).tolist()]
        key = (
            _safe_int(row.get("trial_id")),
            str(row.get("candidate_set_mode", "")),
            float(row.get("scale", "nan")),
            str(row.get("prior_family", "")),
        )
        needed.setdefault(key, set()).update(ids)

    eyepos_by_session = _session_dataset_cache(work["session"].astype(str).to_list())
    trace_bank = _build_trace_bank(
        work,
        eyepos_by_session,
        int(args.n_timepoints),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps)
            if args.microsaccade_speed_threshold_dps is not None
            else None
        ),
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
        trace_window_policy=str(getattr(args, "trace_window_policy", "center_crop_native")),
    )
    source_to_item = {int(item["source_row"]): item for item in trace_bank}
    trace_map: dict[str, np.ndarray] = {}
    for (trial_id, candidate_mode, scale, prior_family), ids in sorted(needed.items()):
        if trial_id not in candidate_rows:
            raise ValueError(f"Missing candidate_sets row for trial_id={trial_id}")
        current_source_row = int(candidate_rows[trial_id]["observation_source_row"])
        obs_item = source_to_item[current_source_row]
        obs_rng = _child_rng(int(args.seed), "observation", current_source_row, args.observation_family, float(scale))
        obs_trace, obs_meta = _trace_from_item(
            family=str(args.observation_family),
            item=obs_item,
            scale=float(scale),
            rng=obs_rng,
            max_rms_deg=float(args.max_rms_deg),
            axis_source_column=str(args.axis_source_column),
            axis_template_mode=str(args.axis_template_mode),
            axis_match_policy=str(args.axis_match_policy),
        )
        prior_rng = _child_rng(
            int(args.seed),
            "prior",
            current_source_row,
            float(scale),
            str(candidate_mode),
            str(prior_family),
            args.trajectory_prior_mode,
        )
        traces, specs, _true_index, _rejection = _prior_trajectories(
            current_source_row=current_source_row,
            observation_family=str(args.observation_family),
            observation_scale=float(scale),
            observation_trace=obs_trace,
            observation_item=obs_item,
            observation_meta=obs_meta,
            prior_family=str(prior_family),
            prior_scale=float(scale),
            trajectory_prior_mode=str(args.trajectory_prior_mode),
            n_prior_trajectories=int(args.n_prior_trajectories),
            trace_bank=trace_bank,
            args=args,
            rng=prior_rng,
        )
        for trace, spec in zip(traces, specs, strict=False):
            trace_id = str(spec["trajectory_id"])
            if trace_id in ids:
                trace_map[trace_id] = np.asarray(trace, dtype=np.float32)
        missing = sorted(ids.difference(trace_map))
        if missing:
            raise ValueError(
                f"Could not reconstruct {len(missing)} prior traces for trial={trial_id}, "
                f"candidate_mode={candidate_mode}, prior_family={prior_family}, scale={scale}: {missing[:3]}"
            )
    return trace_map


def _central_crop(image: np.ndarray, size_px: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    if int(size_px) <= 0 or int(size_px) >= min(arr.shape[:2]):
        return arr.astype(np.float32, copy=False)
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    half = int(size_px) // 2
    return arr[cy - half : cy - half + int(size_px), cx - half : cx - half + int(size_px)].astype(np.float32, copy=False)


def _zscore_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = arr - float(np.nanmean(arr))
    sd = float(np.nanstd(arr))
    if not np.isfinite(sd) or sd <= 1e-6:
        sd = 1.0
    return arr / sd


def _fft_cache_for_sources(
    run_dir: Path,
    source_rows: list[int],
    *,
    patch_size_px: int,
    component_crop_px: int,
    cache_path: Path | None,
) -> dict[int, dict[str, Any]]:
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as z:
            cached_rows = np.asarray(z["source_rows"], dtype=np.int64)
            needed = np.asarray(sorted(set(int(v) for v in source_rows)), dtype=np.int64)
            if np.array_equal(cached_rows, needed):
                fft_stack = np.asarray(z["fft_stack"], dtype=np.complex64)
                ppd = np.asarray(z["ppd"], dtype=np.float64)
                shapes = np.asarray(z["shape"], dtype=np.int64)
                return {
                    int(src): {
                        "fft": fft_stack[i],
                        "ppd": float(ppd[i]),
                        "shape": (int(shapes[i, 0]), int(shapes[i, 1])),
                    }
                    for i, src in enumerate(cached_rows.tolist())
                }
    windows = pd.read_csv(run_dir / "selected_windows.csv")
    lookup = windows.drop_duplicates("source_row").set_index("source_row", drop=False)
    missing = sorted(set(int(v) for v in source_rows).difference(set(int(v) for v in lookup.index.tolist())))
    if missing:
        raise ValueError(f"Missing {len(missing)} candidate source rows from selected_windows.csv: {missing[:8]}")
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    out: dict[int, dict[str, Any]] = {}
    for source_row in tqdm(sorted(set(int(v) for v in source_rows)), desc="fourier patch cache"):
        patch, meta = _extract_patch(
            lookup.loc[int(source_row)],
            canvas_cache=canvas_cache,
            patch_size_px=int(patch_size_px),
        )
        crop = _zscore_image(_central_crop(patch, int(component_crop_px)))
        fft = np.fft.fft2(crop, norm="ortho")
        out[int(source_row)] = {
            "fft": fft.astype(np.complex64),
            "ppd": float(meta["patch_ppd"]),
            "shape": tuple(int(v) for v in crop.shape[:2]),
        }
    if cache_path is not None:
        ordered = sorted(out)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            source_rows=np.asarray(ordered, dtype=np.int64),
            fft_stack=np.stack([out[src]["fft"] for src in ordered], axis=0).astype(np.complex64),
            ppd=np.asarray([out[src]["ppd"] for src in ordered], dtype=np.float64),
            shape=np.asarray([out[src]["shape"] for src in ordered], dtype=np.int64),
            patch_size_px=np.asarray([int(patch_size_px)], dtype=np.int64),
            component_crop_px=np.asarray([int(component_crop_px)], dtype=np.int64),
        )
    return out


def _parse_bands(text: str) -> list[tuple[str, float, float]]:
    bands = []
    for item in _parse_list(text):
        lo_s, hi_s = item.split(":", 1)
        lo = float(lo_s)
        hi = float(hi_s)
        bands.append((f"{lo:g}-{hi:g}cpd", lo, hi))
    if not bands:
        raise ValueError("--frequency-bands must contain at least one lo:hi band")
    return bands


def _frequency_grid(shape: tuple[int, int], ppd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = int(shape[0]), int(shape[1])
    fy_cpp = np.fft.fftfreq(h)
    fx_cpp = np.fft.fftfreq(w)
    fy, fx = np.meshgrid(fy_cpp, fx_cpp, indexing="ij")
    radius_cpd = np.sqrt(fx * fx + fy * fy) * float(ppd)
    half_plane = (fy > 0) | ((np.abs(fy) < 1e-12) & (fx > 0))
    return fx, fy, radius_cpd, half_plane


def _select_frequency_components(
    fft_cache: dict[int, dict[str, Any]],
    bands: list[tuple[str, float, float]],
    *,
    n_per_band: int,
) -> dict[str, dict[str, Any]]:
    first = next(iter(fft_cache.values()))
    shape = tuple(first["shape"])
    ppd = float(np.median([item["ppd"] for item in fft_cache.values()]))
    fx, fy, radius_cpd, half_plane = _frequency_grid(shape, ppd)
    avg_power = np.mean([np.abs(item["fft"]) ** 2 for item in fft_cache.values()], axis=0)
    selected: dict[str, dict[str, Any]] = {}
    all_indices: list[tuple[int, int]] = []
    for band_name, lo, hi in bands:
        mask = half_plane & (radius_cpd >= float(lo)) & (radius_cpd < float(hi))
        candidates = np.argwhere(mask)
        if candidates.size == 0:
            continue
        powers = np.asarray([avg_power[int(i), int(j)] for i, j in candidates], dtype=np.float64)
        order = np.argsort(powers)[::-1][: int(n_per_band)]
        indices = [(int(candidates[i, 0]), int(candidates[i, 1])) for i in order]
        all_indices.extend(indices)
        selected[band_name] = {
            "indices": indices,
            "mean_cpd": float(np.mean([radius_cpd[i, j] for i, j in indices])),
            "min_cpd": float(np.min([radius_cpd[i, j] for i, j in indices])),
            "max_cpd": float(np.max([radius_cpd[i, j] for i, j in indices])),
            "avg_power": [float(avg_power[i, j]) for i, j in indices],
        }
    # Preserve order while removing duplicates across adjacent bands.
    seen = set()
    unique_all = []
    for idx in all_indices:
        if idx not in seen:
            unique_all.append(idx)
            seen.add(idx)
    selected["all_bands"] = {
        "indices": unique_all,
        "mean_cpd": float(np.mean([radius_cpd[i, j] for i, j in unique_all])),
        "min_cpd": float(np.min([radius_cpd[i, j] for i, j in unique_all])),
        "max_cpd": float(np.max([radius_cpd[i, j] for i, j in unique_all])),
        "avg_power": [float(avg_power[i, j]) for i, j in unique_all],
    }
    selected["_grid"] = {"fx_cpp": fx, "fy_cpp": fy, "radius_cpd": radius_cpd}
    return selected


def _component_group(
    fft_item: dict[str, Any],
    trace_deg: np.ndarray,
    indices: list[tuple[int, int]],
    avg_power: list[float],
    *,
    normalization: str,
) -> np.ndarray:
    fft = np.asarray(fft_item["fft"], dtype=np.complex64)
    ppd = float(fft_item["ppd"])
    shape = tuple(fft_item["shape"])
    fx, fy, _radius, _half = _frequency_grid(shape, ppd)
    idx = np.asarray(indices, dtype=np.int64)
    coeff = fft[idx[:, 0], idx[:, 1]].astype(np.complex64)
    if normalization == "power_whitened":
        coeff = coeff / np.sqrt(np.asarray(avg_power, dtype=np.float32) + 1e-8)
    elif normalization == "unit_amplitude":
        coeff = coeff / (np.abs(coeff) + 1e-8)
    elif normalization != "raw":
        raise ValueError(f"Unknown component normalization {normalization!r}")
    freq_x = fx[idx[:, 0], idx[:, 1]][None, :]
    freq_y = fy[idx[:, 0], idx[:, 1]][None, :]
    shift_px = np.asarray(trace_deg, dtype=np.float64) * ppd
    phase_arg = shift_px[:, [0]] * freq_x + shift_px[:, [1]] * freq_y
    shifted = coeff[None, :] * np.exp(-2j * np.pi * phase_arg)
    return np.concatenate([np.real(shifted), np.imag(shifted)], axis=1).astype(np.float32)


def _controlled_trace(trace: np.ndarray, *, mode: str, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float32)
    if str(mode) == "real":
        return arr
    if str(mode) == "time_permute":
        return arr[rng.permutation(arr.shape[0])]
    if str(mode) == "reverse":
        return arr[::-1].copy()
    raise ValueError(f"Unknown trace_control={mode!r}")


def _fit_pca_streamed(groups: list[np.ndarray], n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_dim = int(groups[0].shape[1])
    total_n = 0
    total_sum = np.zeros(n_dim, dtype=np.float64)
    for group in groups:
        total_n += int(group.shape[0])
        total_sum += np.sum(group, axis=0, dtype=np.float64)
    mean = total_sum / float(total_n)
    cov = np.zeros((n_dim, n_dim), dtype=np.float64)
    for group in groups:
        xc = np.asarray(group, dtype=np.float64) - mean[None, :]
        cov += xc.T @ xc
    cov /= float(max(total_n - 1, 1))
    evals, vecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    vecs = vecs[:, order]
    k = min(int(n_components), vecs.shape[1])
    return mean.astype(np.float32), vecs[:, :k].astype(np.float32), evals.astype(np.float32)


def _display_scaled_groups(
    groups: list[np.ndarray],
    *,
    mode: str,
) -> tuple[list[np.ndarray], list[float]]:
    if str(mode) == "raw":
        return groups, [1.0, 1.0, 1.0]
    if str(mode) != "unit_std":
        raise ValueError(f"Unknown axis_scale_mode={mode!r}")
    coords = np.concatenate([np.asarray(group[:, :3], dtype=np.float64) for group in groups], axis=0)
    sd = np.nanstd(coords, axis=0)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    scales = 1.0 / sd
    out = []
    for group in groups:
        arr = np.asarray(group, dtype=np.float32).copy()
        arr[:, : min(3, arr.shape[1])] *= scales[: min(3, arr.shape[1])][None, :]
        out.append(arr)
    return out, [float(v) for v in scales[:3].tolist()]


def _flow_metric(projected_groups: list[np.ndarray], *, args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.seed))
    segments = _flatten_segments(projected_groups)
    eval_idx = _sample_indices(segments["midpoint"].shape[0], int(args.max_eval_segments), rng)
    neighbor_idx = _neighbor_matrix(
        segments["midpoint"],
        segments["group_id"],
        eval_idx,
        neighbor_k=int(args.neighbor_k),
        query_k=int(args.query_k),
        exclude_same_group=True,
    )
    obs = _local_coherence(segments["unit_vector"], neighbor_idx)
    null_mean = []
    null_median = []
    n_segments = int(segments["midpoint"].shape[0])
    for _ in range(int(args.n_nulls)):
        perm = rng.permutation(n_segments)
        coh = _local_coherence(segments["unit_vector"][perm], neighbor_idx)
        null_mean.append(float(np.mean(coh)))
        null_median.append(float(np.median(coh)))
    rows = [
        _summarize_against_null("local_direction_coherence_mean", float(np.mean(obs)), np.asarray(null_mean)),
        _summarize_against_null("local_direction_coherence_median", float(np.median(obs)), np.asarray(null_median)),
    ]
    return {
        "n_segments_total": int(n_segments),
        "n_segments_eval_with_neighbors": int(neighbor_idx.shape[0]),
        "neighbor_k": int(args.neighbor_k),
        "n_nulls": int(args.n_nulls),
        "metric_rows": rows,
        "observed_local_coherence_quantiles": {
            "q05": float(np.percentile(obs, 5)),
            "q25": float(np.percentile(obs, 25)),
            "q50": float(np.percentile(obs, 50)),
            "q75": float(np.percentile(obs, 75)),
            "q95": float(np.percentile(obs, 95)),
        },
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    manifest_rows = _manifest_filter(_read_csv_rows(run_dir / "response_cache_manifest.csv"), args)
    if not manifest_rows:
        raise ValueError("No response-cache rows survived filters")
    _response_groups, meta = _collect_trajectories(run_dir, manifest_rows, variant=str(args.variant))
    trace_map = _reconstruct_prior_trace_map(run_dir, manifest_rows)
    trace_rng = np.random.default_rng(int(args.seed) + 17)
    trace_map = {
        trace_id: _controlled_trace(trace, mode=str(args.trace_control), rng=trace_rng)
        for trace_id, trace in sorted(trace_map.items())
    }
    source_rows = sorted({_safe_int(str(row["candidate_id"]).split(":")[-1]) for row in meta})
    fft_cache = _fft_cache_for_sources(
        run_dir,
        source_rows,
        patch_size_px=int(args.patch_size_px),
        component_crop_px=int(args.component_crop_px),
        cache_path=(Path(args.fourier_cache) if args.fourier_cache is not None else None),
    )
    bands = _parse_bands(args.frequency_bands)
    selected = _select_frequency_components(fft_cache, bands, n_per_band=int(args.components_per_band))

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "variant": str(args.variant),
        "n_manifest_rows_selected": int(len(manifest_rows)),
        "n_trajectory_groups": int(len(meta)),
        "n_candidate_sources": int(len(source_rows)),
        "frequency_bands": [name for name, _lo, _hi in bands],
        "components_per_band": int(args.components_per_band),
        "component_normalization": str(args.component_normalization),
        "trace_control": str(args.trace_control),
        "band_summaries": {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_order = ["all_bands"] + [name for name, _lo, _hi in bands if name in selected]
    for band_name in plot_order:
        spec = selected[band_name]
        indices = list(spec["indices"])
        if not indices:
            continue
        groups: list[np.ndarray] = []
        for row in meta:
            source_row = _safe_int(str(row["candidate_id"]).split(":")[-1])
            trace = trace_map[str(row["trajectory_id"])]
            groups.append(
                _component_group(
                    fft_cache[source_row],
                    trace,
                    indices,
                    list(spec["avg_power"]),
                    normalization=str(args.component_normalization),
                )
            )
        mean, basis, evals = _fit_pca_streamed(groups, n_components=max(3, int(args.n_components)))
        projected = [_project_group(group, mean, basis) for group in groups]
        total = float(np.sum(evals)) + 1e-12
        var_fraction = evals[:3] / total
        plot_groups, display_axis_scales = _display_scaled_groups(projected, mode=str(args.axis_scale_mode))
        centroids = np.asarray([np.mean(group[:, :3], axis=0) for group in plot_groups], dtype=np.float64)
        shown = _farthest_subset(centroids, min(int(args.max_trajectories), len(projected)), seed=int(args.seed))
        band_dir = out_dir / band_name.replace(".", "p").replace("-", "_")
        scale_note = "real PC axes" if str(args.axis_scale_mode) == "raw" else "unit-std display axes"
        title = (
            f"{args.variant} eye traces in Fourier phase-component PCs; {band_name}; "
            f"{len(indices)} complex components; {scale_note}"
        )
        for color_by in _parse_list(args.color_by):
            stem = f"fourier_component_flow_{band_name}_by_{color_by}"
            _static_plot(
                band_dir / f"{stem}.png",
                plot_groups,
                meta,
                shown,
                color_by=color_by,
                var_fraction=var_fraction,
                title=title,
                pc3_scale=float(args.pc3_scale),
                plot_style="arrows",
                arrow_stride=int(args.arrow_stride),
                normalize_arrows=bool(args.normalize_arrows),
            )
            if bool(args.write_html):
                _interactive_plot(
                    band_dir / f"{stem}.html",
                    plot_groups,
                    meta,
                    shown,
                    color_by=color_by,
                    title=title,
                    pc3_scale=float(args.pc3_scale),
                    plot_style="arrows",
                    arrow_stride=int(args.arrow_stride),
                    normalize_arrows=bool(args.normalize_arrows),
                )
        metric = _flow_metric(projected, args=args)
        _write_csv(band_dir / "fourier_component_flow_metric_rows.csv", metric["metric_rows"])
        band_summary = {
            "band_name": band_name,
            "n_complex_components": int(len(indices)),
            "mean_cpd": float(spec["mean_cpd"]),
            "min_cpd": float(spec["min_cpd"]),
            "max_cpd": float(spec["max_cpd"]),
            "pca_explained_fraction_first3": [float(v) for v in var_fraction.tolist()],
            "display_axis_scale_mode": str(args.axis_scale_mode),
            "display_axis_scales_pc123": display_axis_scales,
            "pc3_scale": float(args.pc3_scale),
            "trace_control": str(args.trace_control),
            **metric,
        }
        _write_json_ready(band_dir / "fourier_component_flow_summary.json", band_summary)
        summary["band_summaries"][band_name] = band_summary
    _write_json_ready(out_dir / "fourier_component_flow_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot trajectory flow in Fourier component phase space for BackImage patches."
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
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--max-trajectories", type=int, default=32)
    parser.add_argument("--color-by", type=str, default="time_index,source_row,trajectory_index")
    parser.add_argument("--pc3-scale", type=float, default=3.0)
    parser.add_argument("--axis-scale-mode", choices=("raw", "unit_std"), default="raw")
    parser.add_argument("--arrow-stride", type=int, default=1)
    parser.add_argument("--normalize-arrows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-html", action=argparse.BooleanOptionalAction, default=True)
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
