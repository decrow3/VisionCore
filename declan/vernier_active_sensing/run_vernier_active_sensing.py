#!/usr/bin/env python3
"""Run Vernier active-sensing rendering audits and optional twin responses.

Examples
--------
Pixel/rendering smoke only:

    .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
      --skip-model --out-dir outputs/vernier_active_sensing_smoke

Small model smoke:

    .venv/bin/python -m declan.vernier_active_sensing.run_vernier_active_sensing \
      --n-traces 2 --max-frames 12 --fd-steps-arcmin 0.5 \
      --conditions static_center,real_fem,order_shuffled_positions
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .forward import compute_vernier_rates, load_model_and_readout
from .metrics import expected_counts, poisson_fisher_counts, pose_blind_diagonal_fisher, summarize_information
from .stimulus import RenderGeometry, VernierSpec, save_pixel_audit_artifacts
from .trajectories import condition_trace, load_eye_traces, subsample_traces, valid_trace


DEFAULT_OUT_DIR = Path("outputs") / "vernier_active_sensing"
DEFAULT_CONDITIONS = (
    "static_center",
    "static_repeated_phase",
    "static_phase_cloud_single",
    "static_phase_cloud_matched_positions",
    "real_fem",
    "order_shuffled_positions",
    "axis_horizontal",
    "axis_vertical",
    "scaled_real_0.5",
    "scaled_real_1.5",
)


def parse_csv_str(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_csv_float(text: str) -> list[float]:
    return [float(part) for part in parse_csv_str(text)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def summarize_condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize reliability across paired traces for each condition/readout."""
    groups: dict[tuple[str, str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("readout", "")),
            str(row.get("condition", "")),
            numeric_value(row.get("fd_step_arcmin")),
            str(row.get("inference_mode", "")),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        readout, condition, fd_step, inference_mode = key
        vals = np.asarray([numeric_value(row.get("final_fisher")) for row in groups[key]], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        thresh = np.asarray([numeric_value(row.get("final_threshold_proxy")) for row in groups[key]], dtype=np.float64)
        thresh = thresh[np.isfinite(thresh)]
        out.append(
            {
                "readout": readout,
                "condition": condition,
                "fd_step_arcmin": fd_step,
                "inference_mode": inference_mode,
                "n": int(vals.size),
                "mean_final_fisher": float(np.mean(vals)) if vals.size else float("nan"),
                "median_final_fisher": float(np.median(vals)) if vals.size else float("nan"),
                "p10_final_fisher": float(np.percentile(vals, 10)) if vals.size else float("nan"),
                "p25_final_fisher": float(np.percentile(vals, 25)) if vals.size else float("nan"),
                "mean_final_threshold_proxy": float(np.mean(thresh)) if thresh.size else float("nan"),
                "median_final_threshold_proxy": float(np.median(thresh)) if thresh.size else float("nan"),
            }
        )
    return out


def paired_contrast_rows(
    rows: list[dict[str, Any]],
    *,
    baselines: tuple[str, ...] = ("static_repeated_phase", "static_phase_cloud_matched_positions", "static_center"),
) -> list[dict[str, Any]]:
    """Trace-paired condition-vs-baseline Fisher and threshold-ratio rows."""
    table: dict[tuple[str, str, float, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        trace_raw = row.get("trace_index")
        if isinstance(trace_raw, str) and not trace_raw.isdigit():
            continue
        trace_index = int(trace_raw)
        key = (
            str(row.get("readout", "")),
            str(row.get("inference_mode", "")),
            numeric_value(row.get("fd_step_arcmin")),
            trace_index,
        )
        table.setdefault(key, {})[str(row.get("condition", ""))] = row

    out: list[dict[str, Any]] = []
    for key, by_condition in sorted(table.items()):
        readout, inference_mode, fd_step, trace_index = key
        for condition, row in by_condition.items():
            for baseline in baselines:
                if condition == baseline or baseline not in by_condition:
                    continue
                f = numeric_value(row.get("final_fisher"))
                fb = numeric_value(by_condition[baseline].get("final_fisher"))
                if not (np.isfinite(f) and np.isfinite(fb)):
                    continue
                out.append(
                    {
                        "readout": readout,
                        "inference_mode": inference_mode,
                        "fd_step_arcmin": fd_step,
                        "trace_index": trace_index,
                        "condition": condition,
                        "baseline_condition": baseline,
                        "condition_final_fisher": f,
                        "baseline_final_fisher": fb,
                        "fisher_delta": f - fb,
                        "fisher_ratio": f / fb if fb > 0 else float("nan"),
                        "threshold_ratio": np.sqrt(fb / f) if f > 0 and fb >= 0 else float("nan"),
                        "condition_beats_baseline": bool(f > fb),
                    }
                )
    return out


def summarize_contrast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("readout", "")),
            str(row.get("inference_mode", "")),
            str(row.get("condition", "")),
            str(row.get("baseline_condition", "")),
            numeric_value(row.get("fd_step_arcmin")),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        readout, inference_mode, condition, baseline, fd_step = key
        grp = groups[key]
        deltas = np.asarray([numeric_value(row.get("fisher_delta")) for row in grp], dtype=np.float64)
        ratios = np.asarray([numeric_value(row.get("threshold_ratio")) for row in grp], dtype=np.float64)
        beats = np.asarray([bool(row.get("condition_beats_baseline")) for row in grp], dtype=bool)
        out.append(
            {
                "readout": readout,
                "inference_mode": inference_mode,
                "condition": condition,
                "baseline_condition": baseline,
                "fd_step_arcmin": fd_step,
                "n": len(grp),
                "mean_fisher_delta": mean_or_nan(deltas),
                "median_fisher_delta": float(np.nanmedian(deltas)) if np.isfinite(deltas).any() else float("nan"),
                "mean_threshold_ratio": mean_or_nan(ratios),
                "median_threshold_ratio": float(np.nanmedian(ratios)) if np.isfinite(ratios).any() else float("nan"),
                "p_condition_beats_baseline": float(np.mean(beats)) if beats.size else float("nan"),
            }
        )
    return out


def build_spec(args: argparse.Namespace, offset_arcmin: float) -> VernierSpec:
    return VernierSpec(
        offset_arcmin=float(offset_arcmin),
        bar_width_arcmin=float(args.bar_width_arcmin),
        gap_arcmin=float(args.gap_arcmin),
        bar_length_arcmin=float(args.bar_length_arcmin),
        contrast=float(args.contrast),
        polarity=str(args.polarity),
    )


def run_model_responses(
    args: argparse.Namespace,
    out_dir: Path,
    geometry: RenderGeometry,
    conditions: list[str],
    fd_steps: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_set = subsample_traces(load_eye_traces(Path(args.eye_traces_path)), args.n_traces, args.seed)
    model, readout = load_model_and_readout(args.device)
    rng = np.random.default_rng(int(args.seed))
    summary_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []

    for step in fd_steps:
        plus_spec = build_spec(args, float(step))
        minus_spec = build_spec(args, -float(step))
        for condition in conditions:
            print(f"Condition={condition} fd_step={step} arcmin", flush=True)
            plus_rates: list[np.ndarray] = []
            minus_rates: list[np.ndarray] = []
            for trace_idx in range(trace_set.traces.shape[0]):
                base_trace = valid_trace(trace_set, trace_idx, max_frames=args.max_frames)
                effective_trace, trace_meta = condition_trace(
                    base_trace,
                    condition=condition,
                    trace_set=trace_set,
                    rng=rng,
                    frame_rate_hz=float(args.frame_rate_hz),
                    microsaccade_speed_threshold_dps=float(args.microsaccade_speed_threshold_dps),
                    microsaccade_pad_frames=int(args.microsaccade_pad_frames),
                )
                plus = compute_vernier_rates(
                    model,
                    readout,
                    plus_spec,
                    effective_trace,
                    inference_mode=args.inference_mode,
                    geometry=geometry,
                    batch_size=args.batch_size,
                    spatial_collapse=args.spatial_collapse,
                    device=args.device,
                )
                minus = compute_vernier_rates(
                    model,
                    readout,
                    minus_spec,
                    effective_trace,
                    inference_mode=args.inference_mode,
                    geometry=geometry,
                    batch_size=args.batch_size,
                    spatial_collapse=args.spatial_collapse,
                    device=args.device,
                )
                t = min(plus.shape[0], minus.shape[0])
                plus = plus[:t]
                minus = minus[:t]
                plus_rates.append(plus.astype(np.float32))
                minus_rates.append(minus.astype(np.float32))
                counts_plus = expected_counts(plus, args.bin_seconds)
                counts_minus = expected_counts(minus, args.bin_seconds)
                info = poisson_fisher_counts(counts_plus, counts_minus, step_arcmin=float(step), phi=args.phi)
                row = {
                    "readout": "pose_aware_diagonal_poisson",
                    "condition": condition,
                    "trace_index": trace_idx,
                    "fd_step_arcmin": float(step),
                    "inference_mode": args.inference_mode,
                    "n_timebins": int(t),
                    "n_units": int(plus.shape[1]),
                    **summarize_information(info),
                }
                summary_rows.append(row)
                inventory_rows.append(
                    {
                        "condition": condition,
                        "trace_index": trace_idx,
                        "fd_step_arcmin": float(step),
                        "n_input_frames": int(base_trace.shape[0]),
                        "n_output_timebins": int(t),
                        "trace_x_mean_deg": float(np.mean(effective_trace[:, 0])),
                        "trace_y_mean_deg": float(np.mean(effective_trace[:, 1])),
                        "trace_x_std_deg": float(np.std(effective_trace[:, 0])),
                        "trace_y_std_deg": float(np.std(effective_trace[:, 1])),
                        **trace_meta,
                    }
                )

            response_path = out_dir / "cache" / f"rates_{condition}_fd{float(step):.4f}arcmin.npz"
            response_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                response_path,
                plus=np.asarray(_pad_rates(plus_rates), dtype=np.float32),
                minus=np.asarray(_pad_rates(minus_rates), dtype=np.float32),
                lengths=np.asarray([arr.shape[0] for arr in plus_rates], dtype=np.int32),
                condition=np.asarray([condition]),
                fd_step_arcmin=np.asarray([float(step)], dtype=np.float32),
                inference_mode=np.asarray([args.inference_mode]),
            )
            if len(plus_rates) >= 2:
                pose_blind = pose_blind_diagonal_fisher(
                    plus_rates,
                    minus_rates,
                    step_arcmin=float(step),
                    bin_seconds=float(args.bin_seconds),
                    phi=float(args.phi),
                )
                summary_rows.append(
                    {
                        "readout": "pose_blind_diagonal_marginal",
                        "condition": condition,
                        "trace_index": "all",
                        "fd_step_arcmin": float(step),
                        "inference_mode": args.inference_mode,
                        "n_timebins": int(pose_blind["cumulative_fisher"].shape[0]),
                        "n_units": int(plus_rates[0].shape[1]),
                        **summarize_information(pose_blind),
                    }
                )

    return summary_rows, inventory_rows


def _pad_rates(rates: list[np.ndarray]) -> np.ndarray:
    n = len(rates)
    t = max(arr.shape[0] for arr in rates)
    u = rates[0].shape[1]
    out = np.full((n, t, u), np.nan, dtype=np.float32)
    for i, arr in enumerate(rates):
        out[i, : arr.shape[0], :] = arr
    return out


def _unpadded_rates(arr: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(arr[i, : int(lengths[i])], dtype=np.float32) for i in range(arr.shape[0])]


def _condition_from_cache_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("rates_") or "_fd" not in stem:
        raise ValueError(f"Unexpected rate cache filename: {path.name}")
    return stem[len("rates_") : stem.rindex("_fd")]


def recompute_summaries_from_cache(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute information summaries from saved finite-difference rate caches."""
    summary_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    cache_paths = sorted((out_dir / "cache").glob("rates_*_fd*arcmin.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No rate caches found under {out_dir / 'cache'}")

    for path in cache_paths:
        with np.load(path, allow_pickle=True) as npz:
            condition = str(npz["condition"][0]) if "condition" in npz else _condition_from_cache_path(path)
            fd_step = float(np.asarray(npz["fd_step_arcmin"])[0])
            inference_mode = str(npz["inference_mode"][0]) if "inference_mode" in npz else str(args.inference_mode)
            plus_rates = _unpadded_rates(np.asarray(npz["plus"], dtype=np.float32), np.asarray(npz["lengths"], dtype=np.int32))
            minus_rates = _unpadded_rates(np.asarray(npz["minus"], dtype=np.float32), np.asarray(npz["lengths"], dtype=np.int32))

        for trace_idx, (plus, minus) in enumerate(zip(plus_rates, minus_rates, strict=True)):
            t = min(plus.shape[0], minus.shape[0])
            plus = plus[:t]
            minus = minus[:t]
            info = poisson_fisher_counts(
                expected_counts(plus, args.bin_seconds),
                expected_counts(minus, args.bin_seconds),
                step_arcmin=fd_step,
                phi=args.phi,
            )
            summary_rows.append(
                {
                    "readout": "pose_aware_diagonal_poisson",
                    "condition": condition,
                    "trace_index": trace_idx,
                    "fd_step_arcmin": fd_step,
                    "inference_mode": inference_mode,
                    "n_timebins": int(t),
                    "n_units": int(plus.shape[1]),
                    **summarize_information(info),
                }
            )
            inventory_rows.append(
                {
                    "condition": condition,
                    "trace_index": trace_idx,
                    "fd_step_arcmin": fd_step,
                    "n_output_timebins": int(t),
                    "n_units": int(plus.shape[1]),
                    "source": "rate_cache",
                    "cache_path": str(path),
                }
            )

        if len(plus_rates) >= 2:
            pose_blind = pose_blind_diagonal_fisher(
                plus_rates,
                minus_rates,
                step_arcmin=fd_step,
                bin_seconds=float(args.bin_seconds),
                phi=float(args.phi),
            )
            summary_rows.append(
                {
                    "readout": "pose_blind_diagonal_count_plus_marginal",
                    "condition": condition,
                    "trace_index": "all",
                    "fd_step_arcmin": fd_step,
                    "inference_mode": inference_mode,
                    "n_timebins": int(pose_blind["cumulative_fisher"].shape[0]),
                    "n_units": int(plus_rates[0].shape[1]),
                    **summarize_information(pose_blind),
                }
            )

    condition_summary_rows = summarize_condition_rows(summary_rows)
    contrast_rows = paired_contrast_rows(summary_rows)
    contrast_summary_rows = summarize_contrast_rows(contrast_rows)
    write_csv(out_dir / "information_summary.csv", summary_rows)
    write_csv(out_dir / "cache_inventory.csv", inventory_rows)
    write_csv(out_dir / "condition_reliability_summary.csv", condition_summary_rows)
    write_csv(out_dir / "paired_baseline_contrasts.csv", contrast_rows)
    write_csv(out_dir / "paired_baseline_contrast_summary.csv", contrast_summary_rows)
    return summary_rows, condition_summary_rows, contrast_summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--eye-traces-path", type=Path, default=Path("scripts/temporal_decoding/data/eye_traces.npz"))
    parser.add_argument("--conditions", type=str, default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--fd-steps-arcmin", type=str, default="0.25,0.5")
    parser.add_argument("--bar-width-arcmin", type=float, default=2.0)
    parser.add_argument("--gap-arcmin", type=float, default=4.0)
    parser.add_argument("--bar-length-arcmin", type=float, default=12.0)
    parser.add_argument("--contrast", type=float, default=0.5)
    parser.add_argument("--polarity", type=str, default="bright", choices=("bright", "dark"))
    parser.add_argument("--n-traces", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inference-mode", type=str, default="framewise", choices=("framewise", "continuous"))
    parser.add_argument("--spatial-collapse", type=str, default="max", choices=("max", "mean"))
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--frame-rate-hz", type=float, default=120.0)
    parser.add_argument("--microsaccade-speed-threshold-dps", type=float, default=30.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--recompute-from-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "vernier_active_sensing_manifest.json"
    previous_manifest = read_json_if_exists(manifest_path)
    conditions = parse_csv_str(args.conditions)
    fd_steps = parse_csv_float(args.fd_steps_arcmin)
    geometry = RenderGeometry()
    canonical_spec = build_spec(args, 0.0)

    if args.recompute_from_cache:
        print(f"Recomputing Vernier summaries from caches in {out_dir}", flush=True)
    else:
        print(f"Writing Vernier audit to {out_dir}", flush=True)
        pixel_audit = save_pixel_audit_artifacts(
            out_dir / "render_audit",
            canonical_spec,
            fd_steps_arcmin=fd_steps,
            geometry=geometry,
            device=args.device or "cpu",
        )
        write_json(out_dir / "render_audit" / "pixel_audit.json", pixel_audit)
        write_csv(out_dir / "render_audit" / "pixel_audit_fd_rows.csv", pixel_audit["fd_rows"])

    summary_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    if args.recompute_from_cache:
        summary_rows, _condition_summary_rows, _contrast_summary_rows = recompute_summaries_from_cache(args, out_dir)
    elif not args.skip_model:
        summary_rows, inventory_rows = run_model_responses(args, out_dir, geometry, conditions, fd_steps)
        write_csv(out_dir / "information_summary.csv", summary_rows)
        write_csv(out_dir / "motion_inventory.csv", inventory_rows)
        condition_summary_rows = summarize_condition_rows(summary_rows)
        contrast_rows = paired_contrast_rows(summary_rows)
        contrast_summary_rows = summarize_contrast_rows(contrast_rows)
        write_csv(out_dir / "condition_reliability_summary.csv", condition_summary_rows)
        write_csv(out_dir / "paired_baseline_contrasts.csv", contrast_rows)
        write_csv(out_dir / "paired_baseline_contrast_summary.csv", contrast_summary_rows)

    summary_tables = [
        "information_summary.csv",
        "motion_inventory.csv",
        "cache_inventory.csv",
        "condition_reliability_summary.csv",
        "paired_baseline_contrasts.csv",
        "paired_baseline_contrast_summary.csv",
    ]
    manifest_payload = {
            "args": vars(args),
            "geometry": asdict(geometry),
            "canonical_spec": asdict(canonical_spec),
            "conditions": conditions,
            "fd_steps_arcmin": fd_steps,
            "skip_model": bool(args.skip_model),
            "recompute_from_cache": bool(args.recompute_from_cache),
            "n_information_rows": len(summary_rows),
            "n_motion_inventory_rows": len(inventory_rows),
            "summary_tables": summary_tables if (not args.skip_model or args.recompute_from_cache) else [],
            "render_audit_dir": out_dir / "render_audit",
            "rate_cache_dir": out_dir / "cache",
            "provenance": "high_res_vernier_world_render_to_retina_sampler_plus_canonical_twin_forward",
    }
    if args.recompute_from_cache and previous_manifest:
        manifest_payload["original_run_manifest"] = previous_manifest
    write_json(manifest_path, manifest_payload)
    print(f"Wrote Vernier active-sensing outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
