from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEMPORAL_DECODING_DIR = ROOT / "scripts" / "temporal_decoding"
if str(TEMPORAL_DECODING_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPORAL_DECODING_DIR))

from declan.twin_covariance_structure.eye_controls import (  # noqa: E402
    amplitude_matched_gaussian,
    occupancy_matched_shuffle,
    x_only,
    y_only,
)
from scripts.temporal_decoding.cache_eoptotype_rates import (  # noqa: E402
    _cache_path,
    _load_eye_traces,
    _load_model_and_readout,
)
from scripts.temporal_decoding.rate_computation import (  # noqa: E402
    compute_population_rates,
    compute_population_rates_hires,
    rates_to_padded_array,
)


SCRIPT_DIR = TEMPORAL_DECODING_DIR
DATA_DIR = SCRIPT_DIR / "data"
RATES_DIR = DATA_DIR / "rates"
EYE_TRACES_PATH = DATA_DIR / "eye_traces.npz"
HIRES_THRESHOLD_DEFAULT = 0.35
DEFAULT_CONTROLS = (
    "x_only",
    "y_only",
    "line_random_angle",
    "occupancy_shuffle",
    "occupancy_iid",
    "amplitude_gaussian_iso",
    "amplitude_gaussian_aniso",
)


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(float(x)) for x in str(text).split(",") if str(x).strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _project_to_fixed_axis(eye_positions: np.ndarray, theta_rad: float) -> np.ndarray:
    ep = np.asarray(eye_positions, dtype=np.float64)
    center = np.mean(ep, axis=0, keepdims=True)
    centered = ep - center
    u = np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.float64)
    coeff = centered @ u
    projected = coeff[:, None] * u[None, :]
    return projected + center


def _transform_single_trace(
    eye_positions: np.ndarray,
    control_type: str,
    rng: np.random.Generator,
    line_theta_rad: float | None = None,
) -> np.ndarray:
    ep = np.asarray(eye_positions, dtype=np.float64)
    if control_type in {"real", "real_trace", "real_2d"}:
        return ep.copy()
    if control_type == "x_only":
        return x_only(ep)
    if control_type == "y_only":
        return y_only(ep)
    if control_type == "line_random_angle":
        theta = float(line_theta_rad if line_theta_rad is not None else rng.uniform(0.0, 2.0 * np.pi))
        return _project_to_fixed_axis(ep, theta)
    if control_type == "occupancy_shuffle":
        return occupancy_matched_shuffle(ep, rng)
    if control_type == "occupancy_iid":
        idx = rng.integers(0, ep.shape[0], size=ep.shape[0])
        return ep[idx]
    if control_type == "amplitude_gaussian_iso":
        return amplitude_matched_gaussian(ep, rng, isotropic=True)
    if control_type == "amplitude_gaussian_aniso":
        return amplitude_matched_gaussian(ep, rng, isotropic=False)
    raise ValueError(f"Unsupported control_type={control_type!r}")


def _transform_trace_set(
    traces: np.ndarray,
    durations: np.ndarray,
    control_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    out = np.full_like(traces, np.nan, dtype=np.float32)
    durations_out = np.asarray(durations, dtype=int).copy()
    line_theta_rad = float(rng.uniform(0.0, 2.0 * np.pi)) if control_type == "line_random_angle" else None

    for i in range(len(durations_out)):
        T = int(durations_out[i])
        ep = traces[i, :T]
        out[i, :T] = _transform_single_trace(ep, control_type, rng, line_theta_rad=line_theta_rad).astype(np.float32)

    meta = {
        "line_theta_rad": line_theta_rad,
    }
    return out, durations_out, meta


def _flatten_positions(traces: np.ndarray, durations: np.ndarray) -> np.ndarray:
    valid = [np.asarray(traces[i, : int(durations[i])], dtype=np.float64) for i in range(len(durations))]
    return np.concatenate(valid, axis=0)


def _geometry_stats(traces: np.ndarray, durations: np.ndarray, prefix: str) -> dict[str, Any]:
    all_pos = _flatten_positions(traces, durations)
    mean = np.mean(all_pos, axis=0)
    cov = np.cov(all_pos, rowvar=False)
    centered = all_pos - mean[None, :]
    rms = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    std = np.std(all_pos, axis=0, ddof=0)
    mins = np.min(all_pos, axis=0)
    maxs = np.max(all_pos, axis=0)
    return {
        f"{prefix}_n_eye_samples": int(all_pos.shape[0]),
        f"{prefix}_eye_position_mean": mean.astype(np.float32),
        f"{prefix}_eye_position_cov": cov.astype(np.float32),
        f"{prefix}_eye_rms_radius": float(rms),
        f"{prefix}_x_mean": float(mean[0]),
        f"{prefix}_y_mean": float(mean[1]),
        f"{prefix}_x_sd": float(std[0]),
        f"{prefix}_y_sd": float(std[1]),
        f"{prefix}_cov_xy": float(cov[0, 1]),
        f"{prefix}_x_min": float(mins[0]),
        f"{prefix}_x_max": float(maxs[0]),
        f"{prefix}_y_min": float(mins[1]),
        f"{prefix}_y_max": float(maxs[1]),
    }


def _metadata_for_control(
    *,
    source_eye_condition: str,
    control_type: str,
    random_seed: int,
    source_traces: np.ndarray,
    source_durations: np.ndarray,
    control_traces: np.ndarray,
    control_durations: np.ndarray,
    temporal_order_preserved: bool,
    occupancy_samples_exactly_preserved: bool,
    amplitude_rms_matched: bool,
    selected_trace_indices: np.ndarray,
    selected_trace_indices_identical_across_conditions: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_stats = _geometry_stats(source_traces, source_durations, "original")
    control_stats = _geometry_stats(control_traces, control_durations, "control")
    exact_projection_of_real_traces = control_type in {"real", "real_trace", "real_2d", "x_only", "y_only", "line_random_angle"}
    uses_real_x_trajectory = control_type in {"real", "real_trace", "real_2d", "x_only"}
    uses_real_y_trajectory = control_type in {"real", "real_trace", "real_2d", "y_only"}
    payload: dict[str, Any] = {
        "source_eye_condition": source_eye_condition,
        "control_type": control_type,
        "random_seed": int(random_seed),
        "n_eye_samples": int(control_stats["control_n_eye_samples"]),
        "eye_position_mean": control_stats["control_eye_position_mean"],
        "eye_position_cov": control_stats["control_eye_position_cov"],
        "eye_rms_radius": float(control_stats["control_eye_rms_radius"]),
        "temporal_order_preserved": bool(temporal_order_preserved),
        "occupancy_samples_exactly_preserved": bool(occupancy_samples_exactly_preserved),
        "amplitude_rms_matched": bool(amplitude_rms_matched),
        "exact_projection_of_real_traces": bool(exact_projection_of_real_traces),
        "uses_real_x_trajectory": bool(uses_real_x_trajectory),
        "uses_real_y_trajectory": bool(uses_real_y_trajectory),
        "x_mean_preserved": bool(np.allclose(source_stats["original_x_mean"], control_stats["control_x_mean"])),
        "y_mean_preserved": bool(np.allclose(source_stats["original_y_mean"], control_stats["control_y_mean"])),
        "selected_trace_indices": np.asarray(selected_trace_indices, dtype=np.int32),
        "selected_trace_indices_identical_across_conditions": bool(selected_trace_indices_identical_across_conditions),
    }
    payload.update(source_stats)
    payload.update(control_stats)
    if extra:
        payload.update(extra)
    return payload


def _save_rates_with_metadata(result: dict[str, Any], path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rates_padded, lengths = rates_to_padded_array(result["rates"])
    payload: dict[str, Any] = {
        "rates": rates_padded,
        "lengths": lengths,
        "condition": np.array([result["condition"]]),
        "spatial_collapse": np.array([result["spatial_collapse"]]),
    }
    for k, v in (result.get("stim_params") or {}).items():
        payload[f"stim_{k}"] = np.array([v])
    for k, v in metadata.items():
        if isinstance(v, (bool, np.bool_)):
            payload[k] = np.array([bool(v)])
        elif isinstance(v, (int, np.integer, float, np.floating, str)):
            payload[k] = np.array([v])
        elif v is None:
            payload[k] = np.array([json.dumps(None)])
        else:
            payload[k] = np.asarray(v)
    np.savez_compressed(path, **payload)
    print(f"Saved rates ({rates_padded.shape}) to {path}", flush=True)


def _compute_rates_for_control(
    *,
    model,
    readout,
    rates_dir: Path,
    logmar: float,
    orientation: int,
    control_type: str,
    traces: np.ndarray,
    durations: np.ndarray,
    hires_threshold: float,
    batch_size: int,
    spatial_collapse: str,
    force: bool,
    random_seed: int,
    file_tag: str,
    source_trace_count: int,
    selected_trace_indices: np.ndarray,
    selected_trace_indices_identical_across_conditions: bool,
) -> Path:
    rng = np.random.default_rng(random_seed)
    transformed_traces, transformed_durations, transform_meta = _transform_trace_set(
        traces,
        durations,
        control_type,
        rng,
    )

    temporal_order_preserved = control_type in {"real", "real_trace", "real_2d", "x_only", "y_only", "line_random_angle"}
    occupancy_exact = control_type == "occupancy_shuffle"
    amplitude_matched = control_type in {"amplitude_gaussian_iso", "amplitude_gaussian_aniso"}
    metadata = _metadata_for_control(
        source_eye_condition="real",
        control_type=control_type,
        random_seed=random_seed,
        source_traces=traces,
        source_durations=durations,
        control_traces=transformed_traces,
        control_durations=transformed_durations,
        temporal_order_preserved=temporal_order_preserved,
        occupancy_samples_exactly_preserved=occupancy_exact,
        amplitude_rms_matched=amplitude_matched,
        selected_trace_indices=selected_trace_indices,
        selected_trace_indices_identical_across_conditions=selected_trace_indices_identical_across_conditions,
        extra={
            "source_trace_count": int(source_trace_count),
            "selected_trace_count": int(len(transformed_durations)),
            **transform_meta,
        },
    )

    use_hires = float(logmar) < float(hires_threshold)
    out_path = _cache_path(rates_dir, logmar, orientation, control_type, hires=use_hires, file_tag=file_tag)
    if out_path.exists() and not force:
        print(f"    [cached] {out_path.name}", flush=True)
        return out_path

    if use_hires:
        result = compute_population_rates_hires(
            model,
            readout,
            float(orientation),
            float(logmar),
            transformed_traces,
            transformed_durations,
            condition="real",
            batch_size=batch_size,
            spatial_collapse=spatial_collapse,
            stim_params={"logmar": float(logmar), "orientation": int(orientation)},
            verbose=False,
        )
    else:
        from scripts.temporal_decoding.stimulus import e_optotype_stack

        stim_stack = e_optotype_stack(int(orientation), float(logmar))
        result = compute_population_rates(
            model,
            readout,
            stim_stack,
            transformed_traces,
            transformed_durations,
            condition="real",
            batch_size=batch_size,
            spatial_collapse=spatial_collapse,
            stim_params={"logmar": float(logmar), "orientation": int(orientation)},
            verbose=False,
        )

    result["condition"] = control_type
    _save_rates_with_metadata(result, out_path, metadata)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate explicit A2/A5 control response caches")
    p.add_argument("--logmars", type=str, required=True)
    p.add_argument("--orientations", type=str, default="0,90,180,270")
    p.add_argument(
        "--controls",
        type=str,
        default=",".join(DEFAULT_CONTROLS),
        help="Comma-separated control cache names to generate.",
    )
    p.add_argument("--rates-dir", type=Path, default=RATES_DIR)
    p.add_argument("--eye-traces-path", type=Path, default=EYE_TRACES_PATH)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--spatial-collapse", choices=("max", "mean"), default="max")
    p.add_argument("--hires-threshold", type=float, default=HIRES_THRESHOLD_DEFAULT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-traces", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--file-tag", type=str, default="")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logmars = _parse_csv_floats(args.logmars)
    orientations = _parse_csv_ints(args.orientations)
    controls = _parse_csv_strings(args.controls)
    if not logmars:
        raise ValueError("No logmars parsed")
    if not orientations:
        raise ValueError("No orientations parsed")
    if not controls:
        raise ValueError("No controls parsed")

    rates_dir = Path(args.rates_dir)
    rates_dir.mkdir(parents=True, exist_ok=True)
    traces, durations = _load_eye_traces(Path(args.eye_traces_path))
    source_trace_count = int(len(traces))
    if args.n_traces is not None:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(traces), size=min(int(args.n_traces), len(traces)), replace=False)
        traces = traces[idx]
        durations = durations[idx]
        selected_trace_indices = np.asarray(idx, dtype=np.int32)
        print(f"Using {len(traces)} traces (subsample)", flush=True)
    else:
        selected_trace_indices = np.arange(len(traces), dtype=np.int32)
        print(f"Using {len(traces)} traces", flush=True)

    selected_trace_indices_identical_across_conditions = True

    model, readout = _load_model_and_readout(args.device)

    for logmar in logmars:
        use_hires = float(logmar) < float(args.hires_threshold)
        print(f"\n=== LogMAR {logmar:.2f} ({'hi-res' if use_hires else 'lo-res'}) ===", flush=True)
        for orientation in orientations:
            print(f"  Orientation {orientation}", flush=True)
            for control_idx, control_type in enumerate(controls):
                control_seed = int(args.seed) + (1000 * int(orientation)) + (100000 * int(round((logmar + 10.0) * 100))) + control_idx
                out_path = _compute_rates_for_control(
                    model=model,
                    readout=readout,
                    rates_dir=rates_dir,
                    logmar=float(logmar),
                    orientation=int(orientation),
                    control_type=str(control_type),
                    traces=traces,
                    durations=durations,
                    hires_threshold=float(args.hires_threshold),
                    batch_size=int(args.batch_size),
                    spatial_collapse=str(args.spatial_collapse),
                    force=bool(args.force),
                    random_seed=control_seed,
                    file_tag=str(args.file_tag),
                    source_trace_count=source_trace_count,
                    selected_trace_indices=selected_trace_indices,
                    selected_trace_indices_identical_across_conditions=selected_trace_indices_identical_across_conditions,
                )
                print(f"    [ok] {out_path.name}", flush=True)

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
