#!/usr/bin/env python3
"""Compute RR100 spatial SSI for BackImage contour-axis motion sweeps.

This runner is the BackImage counterpart of the recent Vernier RR100 unit-SSI
plots, but uses the audited BackImage axis-conditioned cache as its source of
window identities and measured traces.  The promoted condition family is not a
pure along-vs-across observer prior; it is a combined retinal trajectory:

    trace = along_scale * along_component + across_scale * across_component

where the components are computed from the saved scale-1 observed trajectory
relative to each selected window's local edge axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
    _build_trace_bank,
    _scale_to_rms,
    _session_dataset_cache,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


RR100_MOVIE_MEDOID_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
DEFAULT_AXIS_RUN_DIR = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_contour_axis_rr100_spatial_ssi_smoke"
)
DEFAULT_ACROSS_SCALES = "0,0.5,1"
STIMULUS_NORMALIZATION = "standardize_uint_like_then_minus_127_div_255"
CACHE_SCHEMA_VERSION = 2
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-run-dir", type=Path, default=DEFAULT_AXIS_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--across-scales", type=str, default=DEFAULT_ACROSS_SCALES)
    parser.add_argument("--along-scales", type=str, default="")
    parser.add_argument("--along-scale", type=float, default=1.0)
    parser.add_argument(
        "--sweep-mode",
        choices=("across", "isotropic", "grid", "pairs"),
        default="across",
        help=(
            "across: hold along-scale fixed and sweep across-scale. "
            "isotropic: use --across-scales as total motion scales with "
            "along_scale=across_scale=scale. grid: fully cross --along-scales "
            "with --across-scales. pairs: use explicit --condition-pairs."
        ),
    )
    parser.add_argument(
        "--condition-pairs",
        type=str,
        default="",
        help="For --sweep-mode pairs, comma-separated along:across scale pairs, e.g. 0:0.25,0.5:1.",
    )
    parser.add_argument("--source-trace-scale", type=float, default=1.0)
    parser.add_argument("--source-trace-prior-family", type=str, default="axis_edge_parallel")
    parser.add_argument("--axis-column", type=str, default="image_edge_axis_deg")
    parser.add_argument("--max-trials", type=int, default=4, help="0 means all eligible source windows.")
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--bin-seconds", type=float, default=None)
    parser.add_argument(
        "--primary-ssi-metric",
        choices=("mean_map", "time_resolved"),
        default="mean_map",
        help=(
            "Promoted SSI contract. mean_map computes SSI after trajectory-averaging "
            "the activation map; time_resolved keeps the legacy per-frame SSI averaged "
            "over the trajectory."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-units", type=int, default=12)
    parser.add_argument("--map-vmin-percentile", type=float, default=0.5)
    parser.add_argument("--map-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--write-zscore-plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--zscore-min-unit-std",
        type=float,
        default=1e-4,
        help="Drop units from the automatic z-scored curve plot below this absolute SSI std.",
    )
    parser.add_argument("--include-static-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write selected trial/condition inventory without loading the twin.")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one scale is required.")
    return values


def parse_condition_pairs(text: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Condition pair {token!r} must use 'along:across' syntax.")
        along_text, across_text = token.split(":", 1)
        pairs.append((float(along_text.strip()), float(across_text.strip())))
    if not pairs:
        raise ValueError("--condition-pairs must contain at least one along:across pair.")
    return pairs


def scale_token(value: float) -> str:
    text = f"{float(value):.9g}".replace("-", "m").replace(".", "p")
    return text


def safe_slug(text: object) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))
    return "_".join(part for part in slug.split("_") if part) or "unnamed"


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def identity_text(identity: dict[str, Any]) -> str:
    return json.dumps(json_ready(identity), sort_keys=True, separators=(",", ":"))


def cache_path(out_dir: Path) -> Path:
    return out_dir / "cache" / "backimage_contour_axis_rr100_spatial_ssi_cache.npz"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[axis]
    if n <= 1:
        return np.zeros_like(np.nanmean(x, axis=axis), dtype=np.float64)
    return np.nanstd(x, axis=axis, ddof=1) / math.sqrt(float(n))


def trace_rms(trace: np.ndarray) -> float:
    arr = np.asarray(trace, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(arr * arr, axis=1)))) if arr.size else 0.0


def trace_path_length(trace: np.ndarray) -> float:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def decompose_trace(trace: np.ndarray, axis_deg: float) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(trace, dtype=np.float64)
    centered = centered - np.mean(centered, axis=0, keepdims=True)
    theta = np.radians(float(axis_deg))
    along_u = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    across_u = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    along = (centered @ along_u)[:, None] * along_u[None, :]
    across = (centered @ across_u)[:, None] * across_u[None, :]
    return along.astype(np.float32), across.astype(np.float32)


def combined_axis_trace(
    source_trace: np.ndarray,
    *,
    axis_deg: float,
    along_scale: float,
    across_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    along, across = decompose_trace(source_trace, float(axis_deg))
    out = float(along_scale) * along + float(across_scale) * across
    out = out - np.mean(out, axis=0, keepdims=True)
    return out.astype(np.float32), {
        "source_trace_rms_deg": trace_rms(source_trace),
        "source_trace_path_length_deg": trace_path_length(source_trace),
        "along_component_rms_deg": trace_rms(along),
        "across_component_rms_deg": trace_rms(across),
        "output_trace_rms_deg": trace_rms(out),
        "output_trace_path_length_deg": trace_path_length(out),
        "axis_deg": float(axis_deg),
        "along_scale": float(along_scale),
        "across_scale": float(across_scale),
    }


def unit_spatial_ssi_for_movie(rate_map: np.ndarray, *, bin_seconds: float) -> dict[str, Any]:
    y = np.asarray(rate_map, dtype=np.float64)
    if y.ndim != 4:
        raise ValueError(f"Expected rate map with shape (T,N,H,W), got {y.shape}")
    if np.nanmin(y) < -1e-7:
        raise ValueError(f"rate map contains negative values; min={float(np.nanmin(y)):.6g}")
    y = np.maximum(y, 0.0)
    t_max, n_units, height, width = y.shape
    flat = y.reshape(t_max, n_units, height * width)
    rbar = np.mean(flat, axis=2)
    gain = flat / (rbar[..., None] + EPS)
    unit_bits_t = np.mean(gain * np.log2(gain + EPS), axis=2)
    unit_expected = np.sum(rbar * float(bin_seconds), axis=0)
    unit_bits = np.sum(unit_bits_t * rbar * float(bin_seconds), axis=0) / np.maximum(unit_expected, EPS)
    population_bits = float(np.sum(unit_bits_t * rbar * float(bin_seconds)) / max(float(np.sum(unit_expected)), EPS))
    return {
        "unit_bits_per_spike": unit_bits.astype(np.float32),
        "unit_expected_spikes": unit_expected.astype(np.float32),
        "unit_mean_rate": np.mean(rbar, axis=0).astype(np.float32),
        "population_bits_per_spike": population_bits,
    }


def unit_spatial_ssi_for_mean_map(mean_rate_map: np.ndarray, *, unit_weights: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(mean_rate_map, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"Expected mean rate map with shape (N,H,W), got {y.shape}")
    if np.nanmin(y) < -1e-7:
        raise ValueError(f"mean rate map contains negative values; min={float(np.nanmin(y)):.6g}")
    y = np.maximum(y, 0.0)
    n_units, height, width = y.shape
    flat = y.reshape(n_units, height * width)
    rbar = np.mean(flat, axis=1)
    gain = flat / (rbar[:, None] + EPS)
    unit_bits = np.mean(gain * np.log2(gain + EPS), axis=1)
    if unit_weights is None:
        weights = rbar
    else:
        weights = np.asarray(unit_weights, dtype=np.float64)
        if weights.shape != (n_units,):
            raise ValueError(f"Expected unit_weights shape {(n_units,)}, got {weights.shape}")
    population_bits = float(np.sum(unit_bits * weights) / max(float(np.sum(weights)), EPS))
    return {
        "unit_bits_per_spike": unit_bits.astype(np.float32),
        "unit_mean_rate": rbar.astype(np.float32),
        "population_bits_per_spike": population_bits,
    }


def condition_specs(
    scales: list[float],
    *,
    along_scale: float,
    along_scales: list[float] | None = None,
    condition_pairs: list[tuple[float, float]] | None = None,
    include_static_baseline: bool,
    sweep_mode: str,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if str(sweep_mode) == "pairs":
        if not condition_pairs:
            raise ValueError("--sweep-mode pairs requires --condition-pairs.")
        for along_value, across_value in condition_pairs:
            specs.append(
                {
                    "condition_id": f"along{scale_token(along_value)}_across{scale_token(across_value)}",
                    "condition_label": f"a{float(along_value):g}/c{float(across_value):g}",
                    "along_scale": float(along_value),
                    "across_scale": float(across_value),
                    "motion_scale": float("nan"),
                    "sweep_mode": "pairs",
                    "is_static_baseline": bool(np.isclose(float(along_value), 0.0) and np.isclose(float(across_value), 0.0)),
                    "is_across_sweep": True,
                }
            )
        return specs

    if str(sweep_mode) == "grid":
        values = list(along_scales or [])
        if not values:
            raise ValueError("--sweep-mode grid requires --along-scales.")
        for along_value in values:
            for across_value in scales:
                specs.append(
                    {
                        "condition_id": f"along{scale_token(along_value)}_across{scale_token(across_value)}",
                        "condition_label": f"a{float(along_value):g}/c{float(across_value):g}",
                        "along_scale": float(along_value),
                        "across_scale": float(across_value),
                        "motion_scale": float("nan"),
                        "sweep_mode": "grid",
                        "is_static_baseline": bool(
                            np.isclose(float(along_value), 0.0) and np.isclose(float(across_value), 0.0)
                        ),
                        "is_across_sweep": True,
                    }
                )
        return specs

    if str(sweep_mode) == "isotropic":
        values = list(scales)
        if bool(include_static_baseline) and not any(np.isclose(float(v), 0.0) for v in values):
            values = [0.0, *values]
        for scale in values:
            specs.append(
                {
                    "condition_id": f"isotropic_scale{scale_token(scale)}",
                    "condition_label": f"{float(scale):g}x",
                    "along_scale": float(scale),
                    "across_scale": float(scale),
                    "motion_scale": float(scale),
                    "sweep_mode": "isotropic",
                    "is_static_baseline": bool(np.isclose(float(scale), 0.0)),
                    "is_across_sweep": True,
                }
            )
        return specs

    if include_static_baseline:
        specs.append(
            {
                "condition_id": "static_along0_across0",
                "condition_label": "static",
                "along_scale": 0.0,
                "across_scale": 0.0,
                "motion_scale": 0.0,
                "sweep_mode": "across",
                "is_static_baseline": True,
                "is_across_sweep": False,
            }
        )
    for scale in scales:
        specs.append(
            {
                "condition_id": f"along{scale_token(along_scale)}_across{scale_token(scale)}",
                "condition_label": f"{float(scale):g}x",
                "along_scale": float(along_scale),
                "across_scale": float(scale),
                "motion_scale": float(scale),
                "sweep_mode": "across",
                "is_static_baseline": False,
                "is_across_sweep": True,
            }
        )
    return specs


def select_source_trials(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    axis_run_dir = Path(args.axis_run_dir)
    selected = pd.read_csv(axis_run_dir / "selected_windows.csv")
    candidate_sets = pd.read_csv(axis_run_dir / "candidate_sets.csv")
    manifest = pd.read_csv(axis_run_dir / "response_cache_manifest.csv")
    run_metadata = load_json(axis_run_dir / "run_metadata.json")

    source_scale = float(args.source_trace_scale)
    rows = manifest[
        np.isclose(pd.to_numeric(manifest["scale"], errors="coerce").to_numpy(dtype=np.float64), source_scale)
        & (manifest["prior_family"].astype(str) == str(args.source_trace_prior_family))
    ].copy()
    if rows.empty:
        raise ValueError(
            f"No response-manifest rows for scale={source_scale:g}, "
            f"prior_family={args.source_trace_prior_family!r}"
        )
    rows = rows.drop_duplicates("trial_id", keep="first")
    rows = rows.merge(
        candidate_sets[["trial_id", "observation_source_row", "candidate_set_mode", "candidate_ids", "candidate_indices"]],
        on="trial_id",
        how="left",
        suffixes=("", "_candidate"),
    )
    if rows["observation_source_row"].isna().any():
        raise ValueError("Some manifest trial_id values were missing from candidate_sets.csv")

    selected_by_source = selected.drop_duplicates("source_row").set_index("source_row", drop=False)
    trace_by_source: dict[int, np.ndarray] | None = None
    trace_source_contract = "response_table_observed_trajectory_xy"

    def reconstructed_trace(source_row: int) -> np.ndarray:
        nonlocal trace_by_source, trace_source_contract
        if trace_by_source is None:
            cfg = run_metadata.get("config", {})
            eyepos_by_session = _session_dataset_cache(selected["session"].astype(str).to_list())
            bank = _build_trace_bank(
                selected,
                eyepos_by_session,
                int(args.n_timepoints),
                microsaccade_speed_threshold_dps=(
                    float(cfg["microsaccade_speed_threshold_dps"])
                    if cfg.get("microsaccade_speed_threshold_dps") is not None
                    else None
                ),
                microsaccade_threshold_z=float(cfg.get("microsaccade_threshold_z", 6.0)),
                microsaccade_pad_frames=int(cfg.get("microsaccade_pad_frames", 1)),
            )
            trace_by_source = {int(item["source_row"]): np.asarray(item["trace"], dtype=np.float32) for item in bank}
            trace_source_contract = "reconstructed_trace_bank_from_selected_windows"
        if int(source_row) not in trace_by_source:
            raise ValueError(f"Could not reconstruct trace for source_row={source_row}")
        trace = np.asarray(trace_by_source[int(source_row)], dtype=np.float32)
        if np.isclose(float(args.source_trace_scale), 1.0):
            return trace
        cfg = run_metadata.get("config", {})
        target_rms = float(args.source_trace_scale) * trace_rms(trace)
        max_rms = float(cfg.get("max_rms_deg", max(target_rms, 1.0)))
        scaled, _meta = _scale_to_rms(trace, target_rms, max_rms_deg=max_rms)
        return np.asarray(scaled, dtype=np.float32)

    trial_payloads: list[dict[str, Any]] = []
    start = max(0, int(args.trial_start))
    stop = None if int(args.max_trials) <= 0 else start + int(args.max_trials)
    for _, row in rows.sort_values("trial_id").iloc[start:stop].iterrows():
        source_row = int(row["observation_source_row"])
        if source_row not in selected_by_source.index:
            raise ValueError(f"source_row={source_row} from candidate_sets.csv not found in selected_windows.csv")
        selected_row = selected_by_source.loc[source_row]
        axis_value = float(selected_row[str(args.axis_column)])
        if not np.isfinite(axis_value):
            raise ValueError(f"source_row={source_row} has non-finite {args.axis_column!r}")
        response_path = axis_run_dir / str(row["response_cache_path"])
        if not response_path.exists():
            raise FileNotFoundError(response_path)
        with np.load(response_path, allow_pickle=False) as data:
            if "observed_trajectory_xy" in data.files:
                source_trace = np.asarray(data["observed_trajectory_xy"], dtype=np.float32)
            else:
                source_trace = reconstructed_trace(source_row)
        source_trace = _align_response_to_trace(source_trace, int(args.n_timepoints))
        payload = dict(selected_row.to_dict())
        payload.update(
            {
                "trial_id": int(row["trial_id"]),
                "response_cache_path": str(row["response_cache_path"]),
                "source_trace_scale": source_scale,
                "source_trace_prior_family": str(args.source_trace_prior_family),
                "source_trace_contract": trace_source_contract,
                "source_trace": source_trace,
            }
        )
        trial_payloads.append(payload)
    if not trial_payloads:
        raise ValueError("No trials selected.")
    trial_frame = pd.DataFrame(trial_payloads)
    meta = {
        "run_metadata_config": run_metadata.get("config", {}),
        "n_available_scale1_trials": int(rows.shape[0]),
        "n_selected_trials": int(trial_frame.shape[0]),
        "source_trace_contract": trace_source_contract,
    }
    return trial_frame, meta


def cache_identity(args: argparse.Namespace, trials: pd.DataFrame, specs: list[dict[str, Any]], rr100_meta: dict[str, Any]) -> dict[str, Any]:
    trace_contracts = (
        sorted({str(v) for v in trials["source_trace_contract"].dropna().to_list()})
        if "source_trace_contract" in trials.columns
        else ["unknown"]
    )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "analysis": "backimage_contour_axis_rr100_spatial_ssi",
        "axis_run_dir": str(Path(args.axis_run_dir).expanduser().resolve()),
        "rr100": rr100_meta,
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "trace_source_contracts": trace_contracts,
        "trace_contract": "scale-1 measured BackImage trace decomposed into local edge along/across components",
        "sweep_mode": str(args.sweep_mode),
        "primary_ssi_metric": str(args.primary_ssi_metric),
        "ssi_metric_contracts": {
            "mean_map": (
                "SSI of the trajectory-averaged unit activation map; this is promoted for "
                "activation-map interpretation."
            ),
            "time_resolved": (
                "Legacy per-frame spatial SSI, rate-weighted over aligned trajectory samples; "
                "kept as a diagnostic."
            ),
        },
        "readout_contract": "average over aligned trajectory samples; no endpoint-only readout",
        "axis_column": str(args.axis_column),
        "source_trace_scale": float(args.source_trace_scale),
        "source_trace_prior_family": str(args.source_trace_prior_family),
        "n_timepoints": int(args.n_timepoints),
        "patch_size_px": int(args.patch_size_px),
        "bin_seconds": None if args.bin_seconds is None else float(args.bin_seconds),
        "trial_ids": [int(v) for v in trials["trial_id"].to_list()],
        "source_rows": [int(v) for v in trials["source_row"].to_list()],
        "condition_specs": specs,
    }


def load_cache(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            observed = str(np.asarray(data["cache_identity_json"]).ravel()[0])
            if observed != identity_text(expected_identity):
                return None
            return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}
    except Exception:
        return None


def rate_map_for_trace(scorer: CanonicalTwinScorer, patch: np.ndarray, trace: np.ndarray) -> np.ndarray:
    image = _standardize_uint_like(patch)
    trace = np.asarray(trace, dtype=np.float32)
    full_stack = np.broadcast_to(
        image[None, :, :],
        (trace.shape[0] + scorer.common.N_LAGS + 1, *image.shape),
    ).copy()
    eye = scorer.torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
    stim = scorer.common.make_counterfactual_stim(
        full_stack,
        eye,
        ppd=scorer.common.PPD,
        scale_factor=1.0,
        n_lags=scorer.common.N_LAGS,
        out_size=scorer.common.OUT_SIZE,
    )
    stim = (stim - 127.0) / 255.0
    rate_map = scorer._compute_rate_map_batched(stim)
    out = rate_map.detach().cpu().numpy().astype(np.float32, copy=False)
    del stim, rate_map
    if str(scorer.device).startswith("cuda") and scorer.torch.cuda.is_available():
        scorer.torch.cuda.empty_cache()
    return out


def compute_cache(
    args: argparse.Namespace,
    *,
    trials: pd.DataFrame,
    specs: list[dict[str, Any]],
    population_view: Any,
    bin_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}

    unit_bits_by_condition: list[list[np.ndarray]] = [[] for _ in specs]
    unit_mean_map_bits_by_condition: list[list[np.ndarray]] = [[] for _ in specs]
    unit_time_resolved_bits_by_condition: list[list[np.ndarray]] = [[] for _ in specs]
    unit_spikes_by_condition: list[list[np.ndarray]] = [[] for _ in specs]
    unit_rates_by_condition: list[list[np.ndarray]] = [[] for _ in specs]
    pop_by_condition: list[list[float]] = [[] for _ in specs]
    pop_mean_map_by_condition: list[list[float]] = [[] for _ in specs]
    pop_time_resolved_by_condition: list[list[float]] = [[] for _ in specs]
    map_sum_by_condition: list[np.ndarray | None] = [None for _ in specs]
    inventory_rows: list[dict[str, Any]] = []

    total = int(trials.shape[0]) * len(specs)
    done = 0
    for movie_idx, (_, trial) in enumerate(trials.iterrows()):
        patch, patch_meta = _extract_patch(
            trial,
            canvas_cache=canvas_cache,
            patch_size_px=int(args.patch_size_px),
        )
        source_trace = np.asarray(trial["source_trace"], dtype=np.float32)
        axis_deg = float(trial[str(args.axis_column)])
        for condition_idx, spec in enumerate(specs):
            done += 1
            if bool(spec["is_static_baseline"]):
                trace = np.zeros_like(source_trace, dtype=np.float32)
                trace_meta = {
                    "source_trace_rms_deg": trace_rms(source_trace),
                    "source_trace_path_length_deg": trace_path_length(source_trace),
                    "along_component_rms_deg": 0.0,
                    "across_component_rms_deg": 0.0,
                    "output_trace_rms_deg": 0.0,
                    "output_trace_path_length_deg": 0.0,
                    "axis_deg": axis_deg,
                    "along_scale": 0.0,
                    "across_scale": 0.0,
                }
            else:
                trace, trace_meta = combined_axis_trace(
                    source_trace,
                    axis_deg=axis_deg,
                    along_scale=float(spec["along_scale"]),
                    across_scale=float(spec["across_scale"]),
                )

            print(
                f"[backimage-contour-ssi] {done}/{total} trial={int(trial['trial_id'])} "
                f"source_row={int(trial['source_row'])} condition={spec['condition_id']}",
                flush=True,
            )
            full_map = rate_map_for_trace(scorer, patch, trace)
            full_map = _align_response_to_trace(full_map, int(args.n_timepoints))
            rr100_map = apply_population_view(full_map, population_view).astype(np.float32, copy=False)
            del full_map
            time_resolved_ssi = unit_spatial_ssi_for_movie(rr100_map, bin_seconds=float(bin_seconds))
            mean_map = np.mean(rr100_map, axis=0).astype(np.float32, copy=False)
            mean_map_ssi = unit_spatial_ssi_for_mean_map(
                mean_map,
                unit_weights=np.asarray(time_resolved_ssi["unit_expected_spikes"], dtype=np.float64),
            )
            primary_ssi = mean_map_ssi if str(args.primary_ssi_metric) == "mean_map" else time_resolved_ssi
            if map_sum_by_condition[condition_idx] is None:
                map_sum_by_condition[condition_idx] = np.zeros_like(mean_map, dtype=np.float64)
            map_sum_by_condition[condition_idx] += mean_map
            unit_bits_by_condition[condition_idx].append(np.asarray(primary_ssi["unit_bits_per_spike"], dtype=np.float32))
            unit_mean_map_bits_by_condition[condition_idx].append(
                np.asarray(mean_map_ssi["unit_bits_per_spike"], dtype=np.float32)
            )
            unit_time_resolved_bits_by_condition[condition_idx].append(
                np.asarray(time_resolved_ssi["unit_bits_per_spike"], dtype=np.float32)
            )
            unit_spikes_by_condition[condition_idx].append(
                np.asarray(time_resolved_ssi["unit_expected_spikes"], dtype=np.float32)
            )
            unit_rates_by_condition[condition_idx].append(np.asarray(time_resolved_ssi["unit_mean_rate"], dtype=np.float32))
            pop_by_condition[condition_idx].append(float(primary_ssi["population_bits_per_spike"]))
            pop_mean_map_by_condition[condition_idx].append(float(mean_map_ssi["population_bits_per_spike"]))
            pop_time_resolved_by_condition[condition_idx].append(float(time_resolved_ssi["population_bits_per_spike"]))
            inventory_rows.append(
                {
                    "movie_index": int(movie_idx),
                    "condition_index": int(condition_idx),
                    "condition_id": str(spec["condition_id"]),
                    "condition_label": str(spec["condition_label"]),
                    "trial_id": int(trial["trial_id"]),
                    "source_row": int(trial["source_row"]),
                    "session": str(trial["session"]),
                    "trial_idx": int(trial["trial_idx"]),
                    "response_cache_path": str(trial["response_cache_path"]),
                    **{k: v for k, v in patch_meta.items()},
                    **trace_meta,
                }
            )
            del rr100_map, mean_map

    n_movies = max(int(trials.shape[0]), 1)
    stats = {
        "primary_ssi_metric": np.asarray([str(args.primary_ssi_metric)]),
        "unit_bits_per_movie": np.stack([np.stack(rows, axis=0) for rows in unit_bits_by_condition], axis=0).astype(np.float32),
        "unit_mean_map_bits_per_movie": np.stack([np.stack(rows, axis=0) for rows in unit_mean_map_bits_by_condition], axis=0).astype(np.float32),
        "unit_time_resolved_bits_per_movie": np.stack([np.stack(rows, axis=0) for rows in unit_time_resolved_bits_by_condition], axis=0).astype(np.float32),
        "unit_expected_spikes_per_movie": np.stack([np.stack(rows, axis=0) for rows in unit_spikes_by_condition], axis=0).astype(np.float32),
        "unit_mean_rate_per_movie": np.stack([np.stack(rows, axis=0) for rows in unit_rates_by_condition], axis=0).astype(np.float32),
        "population_bits_per_movie": np.stack([np.asarray(rows, dtype=np.float32) for rows in pop_by_condition], axis=0),
        "population_mean_map_bits_per_movie": np.stack([np.asarray(rows, dtype=np.float32) for rows in pop_mean_map_by_condition], axis=0),
        "population_time_resolved_bits_per_movie": np.stack([np.asarray(rows, dtype=np.float32) for rows in pop_time_resolved_by_condition], axis=0),
        "mean_rate_map": np.stack([(m / float(n_movies)).astype(np.float32) for m in map_sum_by_condition], axis=0),
        "condition_id": np.asarray([str(spec["condition_id"]) for spec in specs]),
        "condition_label": np.asarray([str(spec["condition_label"]) for spec in specs]),
        "along_scale": np.asarray([float(spec["along_scale"]) for spec in specs], dtype=np.float32),
        "across_scale": np.asarray([float(spec["across_scale"]) for spec in specs], dtype=np.float32),
        "motion_scale": np.asarray([float(spec.get("motion_scale", spec["across_scale"])) for spec in specs], dtype=np.float32),
        "sweep_mode": np.asarray([str(spec.get("sweep_mode", "across")) for spec in specs]),
        "is_static_baseline": np.asarray([bool(spec["is_static_baseline"]) for spec in specs], dtype=bool),
        "is_across_sweep": np.asarray([bool(spec["is_across_sweep"]) for spec in specs], dtype=bool),
        "movie_trial_id": trials["trial_id"].to_numpy(dtype=np.int32),
        "movie_source_row": trials["source_row"].to_numpy(dtype=np.int32),
    }
    return stats, inventory_rows


def save_cache(path: Path, stats: dict[str, Any], identity: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **stats, cache_identity_json=np.asarray([identity_text(identity)]))


def summarize(stats: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    primary_ssi_metric = str(np.asarray(stats.get("primary_ssi_metric", ["time_resolved"])).astype(str).ravel()[0])
    unit_bits = np.asarray(stats["unit_bits_per_movie"], dtype=np.float64)
    unit_mean_map_bits = np.asarray(stats.get("unit_mean_map_bits_per_movie", unit_bits), dtype=np.float64)
    unit_time_resolved_bits = np.asarray(stats.get("unit_time_resolved_bits_per_movie", unit_bits), dtype=np.float64)
    unit_spikes = np.asarray(stats["unit_expected_spikes_per_movie"], dtype=np.float64)
    unit_rates = np.asarray(stats["unit_mean_rate_per_movie"], dtype=np.float64)
    pop = np.asarray(stats["population_bits_per_movie"], dtype=np.float64)
    pop_mean_map = np.asarray(stats.get("population_mean_map_bits_per_movie", pop), dtype=np.float64)
    pop_time_resolved = np.asarray(stats.get("population_time_resolved_bits_per_movie", pop), dtype=np.float64)
    condition_id = np.asarray(stats["condition_id"]).astype(str)
    condition_label = np.asarray(stats["condition_label"]).astype(str)
    along_scale = np.asarray(stats["along_scale"], dtype=np.float64)
    across_scale = np.asarray(stats["across_scale"], dtype=np.float64)
    motion_scale = np.asarray(stats.get("motion_scale", across_scale), dtype=np.float64)
    sweep_mode = np.asarray(stats.get("sweep_mode", np.asarray(["across"] * condition_id.size))).astype(str)
    is_static = np.asarray(stats["is_static_baseline"], dtype=bool)
    is_sweep = np.asarray(stats["is_across_sweep"], dtype=bool)

    bits_mean = np.nanmean(unit_bits, axis=1)
    bits_sem = sem(unit_bits, axis=1)
    mean_map_bits_mean = np.nanmean(unit_mean_map_bits, axis=1)
    mean_map_bits_sem = sem(unit_mean_map_bits, axis=1)
    time_resolved_bits_mean = np.nanmean(unit_time_resolved_bits, axis=1)
    time_resolved_bits_sem = sem(unit_time_resolved_bits, axis=1)
    rates_mean = np.nanmean(unit_rates, axis=1)
    rates_sem = sem(unit_rates, axis=1)
    pop_mean = np.nanmean(pop, axis=1)
    pop_sem = sem(pop, axis=1)
    pop_mean_map_mean = np.nanmean(pop_mean_map, axis=1)
    pop_mean_map_sem = sem(pop_mean_map, axis=1)
    pop_time_resolved_mean = np.nanmean(pop_time_resolved, axis=1)
    pop_time_resolved_sem = sem(pop_time_resolved, axis=1)

    static_idx = int(np.flatnonzero(is_static)[0]) if np.any(is_static) else -1
    across1_candidates = np.flatnonzero(is_sweep & np.isclose(along_scale, 1.0) & np.isclose(across_scale, 1.0))
    across1_idx = int(across1_candidates[0]) if across1_candidates.size else -1

    total_expected = np.sum(unit_spikes, axis=2)
    total_bits = np.sum(unit_spikes * unit_bits, axis=2)
    n_conditions, _n_movies, n_units = unit_bits.shape
    loo_pop = np.zeros((n_units, n_conditions), dtype=np.float64)
    for unit_idx in range(n_units):
        numer = total_bits - unit_spikes[:, :, unit_idx] * unit_bits[:, :, unit_idx]
        denom = np.maximum(total_expected - unit_spikes[:, :, unit_idx], EPS)
        loo_pop[unit_idx] = np.nanmean(numer / denom, axis=1)
    loo_abs_delta = loo_pop - pop_mean[None, :]
    max_abs_loo = np.nanmax(np.abs(loo_abs_delta), axis=1)
    max_abs_unit_delta = np.nanmax(
        np.abs(bits_mean - (bits_mean[static_idx][None, :] if static_idx >= 0 else np.nanmean(bits_mean, axis=0)[None, :])),
        axis=0,
    )

    condition_rows: list[dict[str, Any]] = []
    for idx in range(n_conditions):
        condition_rows.append(
            {
                "condition_index": int(idx),
                "condition_id": str(condition_id[idx]),
                "condition_label": str(condition_label[idx]),
                "is_static_baseline": bool(is_static[idx]),
                "is_across_sweep": bool(is_sweep[idx]),
                "along_scale": float(along_scale[idx]),
                "across_scale": float(across_scale[idx]),
                "motion_scale": float(motion_scale[idx]),
                "sweep_mode": str(sweep_mode[idx]),
                "primary_ssi_metric": primary_ssi_metric,
                "population_ssi_bits_per_spike_mean": float(pop_mean[idx]),
                "population_ssi_bits_per_spike_sem": float(pop_sem[idx]),
                "population_mean_map_ssi_bits_per_spike_mean": float(pop_mean_map_mean[idx]),
                "population_mean_map_ssi_bits_per_spike_sem": float(pop_mean_map_sem[idx]),
                "population_time_resolved_ssi_bits_per_spike_mean": float(pop_time_resolved_mean[idx]),
                "population_time_resolved_ssi_bits_per_spike_sem": float(pop_time_resolved_sem[idx]),
                "population_delta_vs_static": float(pop_mean[idx] - pop_mean[static_idx]) if static_idx >= 0 else float("nan"),
                "population_delta_vs_across1": float(pop_mean[idx] - pop_mean[across1_idx]) if across1_idx >= 0 else float("nan"),
                "population_mean_map_delta_vs_static": (
                    float(pop_mean_map_mean[idx] - pop_mean_map_mean[static_idx]) if static_idx >= 0 else float("nan")
                ),
                "population_time_resolved_delta_vs_static": (
                    float(pop_time_resolved_mean[idx] - pop_time_resolved_mean[static_idx])
                    if static_idx >= 0
                    else float("nan")
                ),
                "n_movies": int(pop.shape[1]),
                "n_units": int(n_units),
            }
        )

    unit_rows: list[dict[str, Any]] = []
    for idx in range(n_conditions):
        for unit_idx in range(n_units):
            unit_rows.append(
                {
                    "condition_index": int(idx),
                    "condition_id": str(condition_id[idx]),
                    "condition_label": str(condition_label[idx]),
                    "along_scale": float(along_scale[idx]),
                    "across_scale": float(across_scale[idx]),
                    "unit_index": int(unit_idx),
                    "unit_label": f"u{int(unit_idx):03d}",
                    "primary_ssi_metric": primary_ssi_metric,
                    "unit_ssi_bits_per_spike_mean": float(bits_mean[idx, unit_idx]),
                    "unit_ssi_bits_per_spike_sem": float(bits_sem[idx, unit_idx]),
                    "unit_mean_map_ssi_bits_per_spike_mean": float(mean_map_bits_mean[idx, unit_idx]),
                    "unit_mean_map_ssi_bits_per_spike_sem": float(mean_map_bits_sem[idx, unit_idx]),
                    "unit_time_resolved_ssi_bits_per_spike_mean": float(time_resolved_bits_mean[idx, unit_idx]),
                    "unit_time_resolved_ssi_bits_per_spike_sem": float(time_resolved_bits_sem[idx, unit_idx]),
                    "unit_mean_rate_mean": float(rates_mean[idx, unit_idx]),
                    "unit_mean_rate_sem": float(rates_sem[idx, unit_idx]),
                    "unit_expected_spikes_mean": float(np.nanmean(unit_spikes[idx, :, unit_idx])),
                    "unit_ssi_delta_vs_static": (
                        float(bits_mean[idx, unit_idx] - bits_mean[static_idx, unit_idx])
                        if static_idx >= 0
                        else float("nan")
                    ),
                    "unit_mean_rate_delta_vs_static": (
                        float(rates_mean[idx, unit_idx] - rates_mean[static_idx, unit_idx])
                        if static_idx >= 0
                        else float("nan")
                    ),
                    "unit_mean_map_ssi_delta_vs_static": (
                        float(mean_map_bits_mean[idx, unit_idx] - mean_map_bits_mean[static_idx, unit_idx])
                        if static_idx >= 0
                        else float("nan")
                    ),
                    "unit_time_resolved_ssi_delta_vs_static": (
                        float(time_resolved_bits_mean[idx, unit_idx] - time_resolved_bits_mean[static_idx, unit_idx])
                        if static_idx >= 0
                        else float("nan")
                    ),
                }
            )

    top_rows = [
        {
            "unit_index": int(unit_idx),
            "unit_label": f"u{int(unit_idx):03d}",
            "primary_ssi_metric": primary_ssi_metric,
            "max_abs_leave_one_out_population_ssi_delta": float(max_abs_loo[unit_idx]),
            "max_abs_unit_ssi_delta_vs_static": float(max_abs_unit_delta[unit_idx]),
            "static_unit_ssi_bits_per_spike_mean": (
                float(bits_mean[static_idx, unit_idx]) if static_idx >= 0 else float("nan")
            ),
            "static_unit_mean_map_ssi_bits_per_spike_mean": (
                float(mean_map_bits_mean[static_idx, unit_idx]) if static_idx >= 0 else float("nan")
            ),
            "static_unit_time_resolved_ssi_bits_per_spike_mean": (
                float(time_resolved_bits_mean[static_idx, unit_idx]) if static_idx >= 0 else float("nan")
            ),
            "static_unit_mean_rate_mean": (
                float(rates_mean[static_idx, unit_idx]) if static_idx >= 0 else float("nan")
            ),
        }
        for unit_idx in range(n_units)
    ]
    top_rows.sort(
        key=lambda row: (
            float(row["max_abs_leave_one_out_population_ssi_delta"]),
            float(row["max_abs_unit_ssi_delta_vs_static"]),
        ),
        reverse=True,
    )

    sweep_indices = np.flatnonzero(is_sweep)
    peak_summary: dict[str, Any] = {}
    if sweep_indices.size:
        sweep_pop = pop_mean[sweep_indices]
        best_local = int(np.nanargmax(sweep_pop))
        best_idx = int(sweep_indices[best_local])
        peak_motion_scale = float(motion_scale[best_idx])
        if not np.isfinite(peak_motion_scale):
            peak_motion_scale = float(max(abs(along_scale[best_idx]), abs(across_scale[best_idx])))
        peak_summary = {
            "primary_ssi_metric": primary_ssi_metric,
            "sweep_peak_condition_id": str(condition_id[best_idx]),
            "sweep_mode": str(sweep_mode[best_idx]),
            "sweep_peak_along_scale": float(along_scale[best_idx]),
            "sweep_peak_across_scale": float(across_scale[best_idx]),
            "sweep_peak_motion_scale": peak_motion_scale,
            "sweep_peak_population_ssi_bits_per_spike": float(pop_mean[best_idx]),
            "sweep_peak_is_below_1x": bool(peak_motion_scale < 1.0),
            "across1_condition_present": bool(across1_idx >= 0),
            "across1_population_ssi_bits_per_spike": float(pop_mean[across1_idx]) if across1_idx >= 0 else float("nan"),
        }

    diagnostics = {
        "primary_ssi_metric": primary_ssi_metric,
        "bits_mean": bits_mean,
        "bits_sem": bits_sem,
        "mean_map_bits_mean": mean_map_bits_mean,
        "time_resolved_bits_mean": time_resolved_bits_mean,
        "rates_mean": rates_mean,
        "pop_mean": pop_mean,
        "pop_sem": pop_sem,
        "pop_mean_map_mean": pop_mean_map_mean,
        "pop_time_resolved_mean": pop_time_resolved_mean,
        "loo_pop": loo_pop,
        "loo_abs_delta": loo_abs_delta,
        "top_rows": top_rows,
        "static_idx": static_idx,
        "across1_idx": across1_idx,
        "peak_summary": peak_summary,
    }
    return condition_rows, unit_rows, top_rows, diagnostics


def image_scale(images: list[np.ndarray], vmin_percentile: float, vmax_percentile: float) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(img, dtype=np.float32).ravel() for img in images])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, float(vmin_percentile)))
    vmax = float(np.nanpercentile(finite, float(vmax_percentile)))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def highlighted_unit_color_map(highlighted_units: list[int]) -> dict[int, Any]:
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(highlighted_units), 1)))
    return {int(unit_index): colors[pos] for pos, unit_index in enumerate(highlighted_units)}


def plot_summary(
    out_dir: Path,
    stats: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    top_units: int,
    dpi: int,
    map_vmin_percentile: float,
    map_vmax_percentile: float,
) -> tuple[Path, Path]:
    condition_id = np.asarray(stats["condition_id"]).astype(str)
    labels = np.asarray(stats["condition_label"]).astype(str)
    across = np.asarray(stats["across_scale"], dtype=np.float64)
    motion_scale = np.asarray(stats.get("motion_scale", across), dtype=np.float64)
    sweep_mode_values = np.asarray(stats.get("sweep_mode", np.asarray(["across"] * labels.size))).astype(str)
    is_sweep = np.asarray(stats["is_across_sweep"], dtype=bool)
    sweep_idx = np.flatnonzero(is_sweep)
    if sweep_idx.size == 0:
        raise ValueError("No across-sweep conditions to plot.")
    plot_mode = str(sweep_mode_values[sweep_idx[0]])

    primary_ssi_metric = str(diagnostics.get("primary_ssi_metric", "time_resolved"))
    metric_label = "mean-map SSI" if primary_ssi_metric == "mean_map" else "time-resolved SSI"
    metric_note = (
        "promoted metric: SSI of trajectory-averaged activation maps; legacy time-resolved SSI cached"
        if primary_ssi_metric == "mean_map"
        else "promoted metric: legacy per-frame SSI rate-weighted over trajectory; mean-map SSI cached"
    )
    bits_mean = np.asarray(diagnostics["bits_mean"], dtype=np.float64)
    pop_mean = np.asarray(diagnostics["pop_mean"], dtype=np.float64)
    mean_maps = np.asarray(stats["mean_rate_map"], dtype=np.float32)
    top_rows = diagnostics["top_rows"][: max(1, int(top_units))]
    highlighted = [int(row["unit_index"]) for row in top_rows]
    colors = highlighted_unit_color_map(highlighted)

    n_rows = len(highlighted)
    n_cols = int(sweep_idx.size)
    fig_height = max(8.0, 4.6 + 0.55 * n_rows)
    fig = plt.figure(figsize=(12.6, fig_height))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, max(1.2, 0.13 * n_rows * n_cols)],
        hspace=0.34,
        wspace=0.18,
    )
    ax_all = fig.add_subplot(gs[0, 0])
    ax_high = fig.add_subplot(gs[0, 1])
    if plot_mode == "isotropic":
        x = motion_scale[sweep_idx]
        xlabel = "isotropic measured-trace motion scale"
        xtick_labels: list[str] | None = None
    elif plot_mode in {"grid", "pairs"}:
        x = np.arange(sweep_idx.size, dtype=np.float64)
        xlabel = "condition pair (along/across scale)"
        xtick_labels = [str(labels[idx]) for idx in sweep_idx]
    else:
        x = across[sweep_idx]
        xlabel = "across-contour motion scale, along=1"
        xtick_labels = None

    for unit_idx in range(bits_mean.shape[1]):
        ax_all.plot(x, bits_mean[sweep_idx, unit_idx], color="#a8a8a8", linewidth=0.65, alpha=0.33, zorder=1)
    for unit_idx in highlighted:
        ax_all.plot(
            x,
            bits_mean[sweep_idx, unit_idx],
            marker="o",
            linewidth=1.3,
            markersize=3.2,
            color=colors[int(unit_idx)],
            label=f"u{int(unit_idx):03d}",
            zorder=3,
        )
    ax_all.plot(x, pop_mean[sweep_idx], color="black", marker="o", linewidth=2.0, markersize=4.2, label="population", zorder=4)
    ax_all.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax_all.set_title("All RR100 units; largest leave-one-out influences highlighted", fontsize=10)
    ax_all.set_xlabel(xlabel)
    ax_all.set_ylabel(f"{metric_label} (bits/spike)")
    if xtick_labels is not None:
        ax_all.set_xticks(x)
        ax_all.set_xticklabels(xtick_labels, rotation=55, ha="right", fontsize=6.2)
    ax_all.grid(True, color="#e6e6e6", linewidth=0.7)
    ax_all.legend(loc="best", fontsize=6.6, ncol=3, frameon=False)

    for unit_idx in highlighted:
        ax_high.plot(
            x,
            bits_mean[sweep_idx, unit_idx],
            marker="o",
            linewidth=1.55,
            markersize=3.6,
            color=colors[int(unit_idx)],
            label=f"u{int(unit_idx):03d}",
        )
    ax_high.plot(x, pop_mean[sweep_idx], color="black", marker="o", linewidth=2.2, markersize=4.4, label="population")
    ax_high.axvline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    ax_high.set_title("Highlighted units only", fontsize=10)
    ax_high.set_xlabel(xlabel)
    ax_high.set_ylabel(f"{metric_label} (bits/spike)")
    if xtick_labels is not None:
        ax_high.set_xticks(x)
        ax_high.set_xticklabels(xtick_labels, rotation=55, ha="right", fontsize=6.2)
    ax_high.grid(True, color="#e6e6e6", linewidth=0.7)
    ax_high.legend(loc="best", fontsize=6.6, ncol=3, frameon=False)

    map_gs = gs[1, :].subgridspec(
        nrows=n_rows + 1,
        ncols=n_cols + 1,
        width_ratios=[0.85, *([1.0] * n_cols)],
        height_ratios=[0.23, *([1.0] * n_rows)],
        hspace=0.03,
        wspace=0.04,
    )
    header_ax = fig.add_subplot(map_gs[0, 0])
    header_ax.axis("off")
    header_ax.text(0.98, 0.2, "unit", ha="right", va="center", fontsize=6.5, color="#555555")
    for col, cond_idx in enumerate(sweep_idx, start=1):
        ax = fig.add_subplot(map_gs[0, col])
        ax.axis("off")
        ax.text(0.5, 0.2, str(labels[cond_idx]), ha="center", va="center", fontsize=6.5, color="#555555")

    shown_images = [mean_maps[int(cond_idx), int(unit_idx)] for unit_idx in highlighted for cond_idx in sweep_idx]
    global_vmin, global_vmax = image_scale(shown_images, map_vmin_percentile, map_vmax_percentile)
    del global_vmin, global_vmax
    for row_idx, unit_idx in enumerate(highlighted, start=1):
        label_ax = fig.add_subplot(map_gs[row_idx, 0])
        label_ax.axis("off")
        label_ax.plot([0.08, 0.86], [0.5, 0.5], color=colors[int(unit_idx)], linewidth=2.2, solid_capstyle="round")
        label_ax.text(
            0.9,
            0.5,
            f"u{int(unit_idx):03d}",
            ha="right",
            va="center",
            color=colors[int(unit_idx)],
            fontsize=7,
            fontweight="bold",
        )
        row_maps = [mean_maps[int(cond_idx), int(unit_idx)] for cond_idx in sweep_idx]
        vmin, vmax = image_scale(row_maps, map_vmin_percentile, map_vmax_percentile)
        for col_idx, cond_idx in enumerate(sweep_idx, start=1):
            ax = fig.add_subplot(map_gs[row_idx, col_idx])
            ax.imshow(mean_maps[int(cond_idx), int(unit_idx)], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor(colors[int(unit_idx)])
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"BackImage RR100 contour-axis spatial SSI: {plot_mode} sweep\n"
        f"{metric_note}; activation maps use monotonic grayscale per unit row",
        fontsize=11.5,
        y=0.995,
    )
    png_path = out_dir / "backimage_contour_axis_rr100_spatial_ssi_absolute_with_activation_maps.png"
    pdf_path = out_dir / "backimage_contour_axis_rr100_spatial_ssi_absolute_with_activation_maps.pdf"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def dry_run_inventory(trials: pd.DataFrame, specs: list[dict[str, Any]], axis_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for movie_idx, (_, trial) in enumerate(trials.iterrows()):
        source_trace = np.asarray(trial["source_trace"], dtype=np.float32)
        axis_deg = float(trial[str(axis_column)])
        for condition_idx, spec in enumerate(specs):
            if bool(spec["is_static_baseline"]):
                trace_meta = {
                    "source_trace_rms_deg": trace_rms(source_trace),
                    "source_trace_path_length_deg": trace_path_length(source_trace),
                    "along_component_rms_deg": 0.0,
                    "across_component_rms_deg": 0.0,
                    "output_trace_rms_deg": 0.0,
                    "output_trace_path_length_deg": 0.0,
                    "axis_deg": axis_deg,
                    "along_scale": 0.0,
                    "across_scale": 0.0,
                }
            else:
                _trace, trace_meta = combined_axis_trace(
                    source_trace,
                    axis_deg=axis_deg,
                    along_scale=float(spec["along_scale"]),
                    across_scale=float(spec["across_scale"]),
                )
            rows.append(
                {
                    "movie_index": int(movie_idx),
                    "condition_index": int(condition_idx),
                    "condition_id": str(spec["condition_id"]),
                    "condition_label": str(spec["condition_label"]),
                    "trial_id": int(trial["trial_id"]),
                    "source_row": int(trial["source_row"]),
                    "session": str(trial["session"]),
                    "trial_idx": int(trial["trial_idx"]),
                    "response_cache_path": str(trial["response_cache_path"]),
                    **trace_meta,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    across_scales = parse_float_list(str(args.across_scales))
    along_scales = parse_float_list(str(args.along_scales)) if str(args.along_scales).strip() else []
    condition_pairs = parse_condition_pairs(str(args.condition_pairs)) if str(args.sweep_mode) == "pairs" else None
    specs = condition_specs(
        across_scales,
        along_scale=float(args.along_scale),
        along_scales=along_scales,
        condition_pairs=condition_pairs,
        include_static_baseline=bool(args.include_static_baseline),
        sweep_mode=str(args.sweep_mode),
    )
    trials, source_meta = select_source_trials(args)
    run_config = source_meta.get("run_metadata_config", {})
    bin_seconds = float(args.bin_seconds) if args.bin_seconds is not None else float(run_config.get("bin_seconds", 1.0 / 120.0))

    rr100 = load_population_view(version_name=str(args.rr100_version))
    rr100_meta = {
        "version": rr100.name,
        "input_channels": int(rr100.input_channels),
        "n_units": int(rr100.n_units),
        "membership_shape": list(np.asarray(rr100.membership).shape),
        "pooling_mode": str(rr100.meta.get("pooling_mode", "")),
    }
    identity = cache_identity(args, trials, specs, rr100_meta)
    write_json(
        out_dir / "run_metadata.json",
        {
            "identity": identity,
            "source_meta": source_meta,
            "bin_seconds_effective": float(bin_seconds),
            "dry_run": bool(args.dry_run),
        },
    )

    if bool(args.dry_run):
        rows = dry_run_inventory(trials, specs, str(args.axis_column))
        write_csv_rows(out_dir / "movie_condition_inventory.csv", rows)
        print(f"Dry-run wrote inventory for {len(trials)} trials x {len(specs)} conditions to {out_dir}", flush=True)
        return

    path = cache_path(out_dir)
    stats = None if bool(args.force) else load_cache(path, identity)
    inventory_rows: list[dict[str, Any]] = []
    if stats is None:
        stats, inventory_rows = compute_cache(
            args,
            trials=trials,
            specs=specs,
            population_view=rr100,
            bin_seconds=float(bin_seconds),
        )
        save_cache(path, stats, identity)
        write_csv_rows(out_dir / "movie_condition_inventory.csv", inventory_rows)
        print(f"Saved cache: {path}", flush=True)
    else:
        print(f"Loaded cache: {path}", flush=True)
        inv_path = out_dir / "movie_condition_inventory.csv"
        if inv_path.exists():
            inventory_rows = list(csv.DictReader(inv_path.open(newline="", encoding="utf-8")))

    condition_rows, unit_rows, top_rows, diagnostics = summarize(stats)
    write_csv_rows(out_dir / "condition_summary.csv", condition_rows)
    write_csv_rows(out_dir / "unit_ssi_table.csv", unit_rows)
    write_csv_rows(out_dir / "highlighted_units.csv", top_rows[: max(1, int(args.top_units))])
    summary_payload: dict[str, Any] = {
        "primary_ssi_metric": str(diagnostics["primary_ssi_metric"]),
        "condition_summary_csv": out_dir / "condition_summary.csv",
        "unit_ssi_table_csv": out_dir / "unit_ssi_table.csv",
        "highlighted_units_csv": out_dir / "highlighted_units.csv",
        "movie_condition_inventory_csv": out_dir / "movie_condition_inventory.csv",
        "cache_npz": path,
        "peak_summary": diagnostics["peak_summary"],
    }
    png_path, pdf_path = plot_summary(
        out_dir,
        stats,
        diagnostics,
        top_units=int(args.top_units),
        dpi=int(args.dpi),
        map_vmin_percentile=float(args.map_vmin_percentile),
        map_vmax_percentile=float(args.map_vmax_percentile),
    )
    print(f"Wrote plot: {png_path}", flush=True)
    print(f"Wrote plot: {pdf_path}", flush=True)
    summary_payload["absolute_plot_png"] = png_path
    summary_payload["absolute_plot_pdf"] = pdf_path
    if bool(args.write_zscore_plot):
        from declan.active_sensing_movie_information.plot_backimage_contour_axis_rr100_zscore_curves import (
            plot_zscore_curves,
        )

        zscore_result = plot_zscore_curves(
            out_dir,
            out_dir=out_dir,
            min_unit_std=float(args.zscore_min_unit_std),
            top_units=int(args.top_units),
            dpi=int(args.dpi),
        )
        print(f"Wrote z-score plot: {zscore_result['png']}", flush=True)
        print(f"Wrote z-score plot: {zscore_result['pdf']}", flush=True)
        summary_payload["zscore_plot_png"] = zscore_result["png"]
        summary_payload["zscore_plot_pdf"] = zscore_result["pdf"]
        summary_payload["zscore_curve_csv"] = zscore_result["csv"]
        summary_payload["zscore_retained_units"] = int(zscore_result["retained_units"])
        summary_payload["zscore_total_units"] = int(zscore_result["total_units"])
        summary_payload["zscore_scale_values"] = np.asarray(zscore_result["scale_values"], dtype=float)
        summary_payload["zscore_population_curve"] = np.asarray(zscore_result["population_z"], dtype=float)
        summary_payload["zscore_mean_unit_curve"] = np.asarray(zscore_result["mean_z"], dtype=float)
    write_json(out_dir / "summary.json", summary_payload)
    if diagnostics["peak_summary"]:
        peak = diagnostics["peak_summary"]
        print(
            "Sweep peak: "
            f"metric={str(peak['primary_ssi_metric'])}, "
            f"across={float(peak['sweep_peak_across_scale']):g}, "
            f"motion_scale={float(peak['sweep_peak_motion_scale']):g}, "
            f"population SSI={float(peak['sweep_peak_population_ssi_bits_per_spike']):.6g}, "
            f"below_1x={bool(peak['sweep_peak_is_below_1x'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
