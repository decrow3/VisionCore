#!/usr/bin/env python3
"""Compute SSI from endpoint-history Vernier terminal activation maps.

This is the SSI companion to ``run_endpoint_history_last_frame_readout``.  It
uses the same endpoint-aligned histories,

    endpoint_trace[t] = trace_tail[t] - trace_tail[-1],

but keeps the full spatial readout map from the final model frame.  SSI is
computed from that single terminal map, matching the "last-frame only" contract
used by the endpoint-history readout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.temporal_decoding.rate_computation import compute_trial_rates
from scripts.temporal_decoding.stimulus_hires import N_LAGS as MODEL_HISTORY_FRAMES

from declan.vernier_active_sensing.forward import (
    PKL_PATH,
    STIMULUS_NORMALIZATION,
    build_vernier_movie,
    load_model_and_readout,
)
from declan.vernier_active_sensing.plot_activation_maps_with_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
    apply_population,
    collapse_activation_map,
    draw_gallery,
    draw_single_map,
    image_scale,
    population_specs,
    safe_slug,
    ssi_single_frame,
    total_rate,
    write_csv_rows,
    write_json,
)
from declan.vernier_active_sensing.run_endpoint_history_last_frame_readout import (
    DEFAULT_CONDITIONS,
    ENDPOINT_ALIGNMENT,
    ENDPOINT_CONDITIONS,
    HISTORY_WINDOW,
    build_endpoint_trace,
    endpoint_condition_rng,
    endpoint_condition_seed_index,
)
from declan.vernier_active_sensing.run_vernier_active_sensing import build_spec, parse_csv_float, parse_csv_str
from declan.vernier_active_sensing.stimulus import RenderGeometry
from declan.vernier_active_sensing.trajectories import load_eye_traces, valid_trace


DEFAULT_OUT_DIR = Path("outputs") / "vernier_endpoint_history_last_frame_tutorial" / "ssi_last_frame_maps"
DEFAULT_POPULATIONS = ("full756", "rr100_medoid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--eye-traces-path", type=Path, default=Path("scripts/temporal_decoding/data/eye_traces.npz"))
    parser.add_argument("--conditions", type=str, default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--populations", type=str, default=",".join(DEFAULT_POPULATIONS))
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--history-frames", type=int, default=int(MODEL_HISTORY_FRAMES))
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--map-kind-for-ssi", choices=("zero", "plus", "minus"), default="zero")
    parser.add_argument("--collapse", choices=("mean", "sum", "max", "ssi_density"), default="mean")
    parser.add_argument("--vmin-percentile", type=float, default=0.5)
    parser.add_argument("--vmax-percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--frame-rate-hz", type=float, default=120.0)
    parser.add_argument("--bar-width-arcmin", type=float, default=2.0)
    parser.add_argument("--gap-arcmin", type=float, default=4.0)
    parser.add_argument("--bar-length-arcmin", type=float, default=12.0)
    parser.add_argument("--contrast", type=float, default=0.5)
    parser.add_argument("--polarity", type=str, default="bright", choices=("bright", "dark"))
    parser.add_argument("--stimulus-orientation-deg", type=float, default=0.0)
    parser.add_argument("--render-resolution-factors", type=str, default="1")
    parser.add_argument("--phi", type=float, default=1.0)
    return parser.parse_args()


def _validate_conditions(conditions: list[str]) -> None:
    invalid = [condition for condition in conditions if condition not in ENDPOINT_CONDITIONS]
    if invalid:
        valid = ", ".join(ENDPOINT_CONDITIONS)
        raise ValueError(f"Unsupported endpoint-history conditions {invalid}; valid={valid}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def _class_name(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _cache_identity(
    args: argparse.Namespace,
    *,
    condition: str,
    geometry: RenderGeometry,
    model: Any,
    readout: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "condition": str(condition),
        "condition_seed_index": endpoint_condition_seed_index(str(condition)),
        "source_condition": str(ENDPOINT_CONDITIONS[str(condition)]["source_condition"]),
        "seed": int(args.seed),
        "trace_index": int(args.trace_index),
        "eye_traces_path": str(Path(args.eye_traces_path).expanduser().resolve()),
        "history_frames": int(args.history_frames),
        "model_history_frames": int(MODEL_HISTORY_FRAMES),
        "terminal_frames": 1,
        "fd_step_arcmin": float(args.fd_step_arcmin),
        "frame_rate_hz": float(args.frame_rate_hz),
        "stimulus": {
            "bar_width_arcmin": float(args.bar_width_arcmin),
            "gap_arcmin": float(args.gap_arcmin),
            "bar_length_arcmin": float(args.bar_length_arcmin),
            "contrast": float(args.contrast),
            "polarity": str(args.polarity),
            "stimulus_orientation_deg": float(args.stimulus_orientation_deg),
            "render_resolution_factors": [float(v) for v in args.render_resolution_factors],
        },
        "geometry": asdict(geometry),
        "history_window": HISTORY_WINDOW,
        "endpoint_alignment": ENDPOINT_ALIGNMENT,
        "readout_time_contract": "terminal_response_only",
        "model_wrapper_class": _class_name(model),
        "model_core_class": _class_name(model.model) if hasattr(model, "model") else "",
        "readout_class": _class_name(readout),
        "readout_source_pkl": str(PKL_PATH.expanduser().resolve()),
        "map_contract": "full756 terminal spatial maps; population view applied downstream",
    }


def _identity_text(identity: dict[str, Any]) -> str:
    return json.dumps(_jsonable(identity), sort_keys=True, separators=(",", ":"))


def _cache_identity_matches(data: Any, expected: dict[str, Any]) -> bool:
    if "cache_identity_json" not in data:
        return False
    cached = str(np.asarray(data["cache_identity_json"]).ravel()[0])
    return cached == _identity_text(expected)


def _warn_if_history_exceeds_model_lags(args: argparse.Namespace) -> None:
    if int(args.history_frames) <= int(MODEL_HISTORY_FRAMES):
        return
    warnings.warn(
        (
            f"--history-frames={int(args.history_frames)} exceeds the model lag window "
            f"({int(MODEL_HISTORY_FRAMES)}); terminal SSI maps can only depend on the final lag window."
        ),
        RuntimeWarning,
        stacklevel=2,
    )


def _map_key(map_kind: str) -> str:
    return {
        "zero": "final_spatial_zero",
        "plus": "final_spatial_plus",
        "minus": "final_spatial_minus",
    }[str(map_kind)]


def _condition_label(condition: str) -> str:
    return str(ENDPOINT_CONDITIONS[str(condition)]["label"])


def _compute_terminal_spatial_map(
    model: Any,
    readout: Any,
    spec: Any,
    endpoint_trace: np.ndarray,
    *,
    geometry: RenderGeometry,
    device: str,
    batch_size: int,
    history_frames: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    stim = build_vernier_movie(
        spec,
        endpoint_trace,
        geometry=geometry,
        n_lags=int(MODEL_HISTORY_FRAMES),
        device=device,
    )
    if int(stim.shape[0]) != int(history_frames):
        raise RuntimeError(f"Expected {history_frames} lag windows, got {int(stim.shape[0])}")
    spatial_movie = compute_trial_rates(
        model,
        readout,
        stim,
        batch_size=int(batch_size),
        return_spatial=True,
    ).astype(np.float32)
    return np.asarray(spatial_movie[-1], dtype=np.float32), tuple(int(v) for v in spatial_movie.shape)


def _condition_cache_path(out_dir: Path, condition: str, trace_index: int, history_frames: int, fd_step: float) -> Path:
    return (
        out_dir
        / "terminal_map_cache"
        / (
            f"endpoint_last_frame_full_map_{safe_slug(condition)}"
            f"_trace{int(trace_index)}_frames{int(history_frames)}"
            f"_fd{float(fd_step):.4f}arcmin.npz"
        )
    )


def _load_or_compute_condition_maps(
    args: argparse.Namespace,
    *,
    condition: str,
    trace_set: Any,
    model: Any,
    readout: Any,
    geometry: RenderGeometry,
    device: str,
) -> tuple[Path, dict[str, np.ndarray], dict[str, Any]]:
    cache_path = _condition_cache_path(
        Path(args.out_dir),
        condition,
        int(args.trace_index),
        int(args.history_frames),
        float(args.fd_step_arcmin),
    )
    expected_identity = _cache_identity(
        args,
        condition=condition,
        geometry=geometry,
        model=model,
        readout=readout,
    )
    if cache_path.exists() and not bool(args.force_recompute):
        with np.load(cache_path, allow_pickle=True) as data:
            if _cache_identity_matches(data, expected_identity):
                maps = {
                    "zero": np.asarray(data["final_spatial_zero"], dtype=np.float32),
                    "plus": np.asarray(data["final_spatial_plus"], dtype=np.float32),
                    "minus": np.asarray(data["final_spatial_minus"], dtype=np.float32),
                }
                meta = {
                    "zero_movie_shape": tuple(int(v) for v in np.asarray(data["zero_movie_shape"]).tolist()),
                    "plus_movie_shape": tuple(int(v) for v in np.asarray(data["plus_movie_shape"]).tolist()),
                    "minus_movie_shape": tuple(int(v) for v in np.asarray(data["minus_movie_shape"]).tolist()),
                    "endpoint_norm_deg": float(np.asarray(data["endpoint_norm_deg"]).ravel()[0]),
                    "history_rms_deg": float(np.asarray(data["history_rms_deg"]).ravel()[0]),
                    "history_path_length_deg": float(np.asarray(data["history_path_length_deg"]).ravel()[0]),
                    "history_max_radius_deg": float(np.asarray(data["history_max_radius_deg"]).ravel()[0]),
                    "source_condition": str(data["source_condition"][0]) if "source_condition" in data else "",
                    "cache_identity_json": str(np.asarray(data["cache_identity_json"]).ravel()[0]),
                }
                print(f"Loaded endpoint terminal maps for {condition}: {maps['zero'].shape}", flush=True)
                return cache_path, maps, meta
        print(f"Endpoint terminal map cache metadata mismatch; recomputing {cache_path}", flush=True)

    base_trace = valid_trace(trace_set, int(args.trace_index))
    rng = endpoint_condition_rng(int(args.seed), condition, int(args.trace_index))
    trace_args = SimpleNamespace(history_frames=int(args.history_frames), frame_rate_hz=float(args.frame_rate_hz))
    endpoint_trace, trace_meta = build_endpoint_trace(
        base_trace,
        condition=condition,
        trace_set=trace_set,
        rng=rng,
        args=trace_args,
    )
    if int(endpoint_trace.shape[0]) != int(args.history_frames):
        raise RuntimeError(f"Expected {args.history_frames} endpoint frames for {condition}, got {endpoint_trace.shape[0]}")

    zero_spec = build_spec(args, 0.0)
    plus_spec = build_spec(args, +float(args.fd_step_arcmin))
    minus_spec = build_spec(args, -float(args.fd_step_arcmin))
    zero, zero_shape = _compute_terminal_spatial_map(
        model,
        readout,
        zero_spec,
        endpoint_trace,
        geometry=geometry,
        device=device,
        batch_size=int(args.batch_size),
        history_frames=int(args.history_frames),
    )
    plus, plus_shape = _compute_terminal_spatial_map(
        model,
        readout,
        plus_spec,
        endpoint_trace,
        geometry=geometry,
        device=device,
        batch_size=int(args.batch_size),
        history_frames=int(args.history_frames),
    )
    minus, minus_shape = _compute_terminal_spatial_map(
        model,
        readout,
        minus_spec,
        endpoint_trace,
        geometry=geometry,
        device=device,
        batch_size=int(args.batch_size),
        history_frames=int(args.history_frames),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        final_spatial_zero=zero,
        final_spatial_plus=plus,
        final_spatial_minus=minus,
        endpoint_trace=endpoint_trace,
        condition=np.asarray([condition]),
        condition_label=np.asarray([_condition_label(condition)]),
        source_condition=np.asarray([ENDPOINT_CONDITIONS[condition]["source_condition"]]),
        trace_index=np.asarray([int(args.trace_index)], dtype=np.int32),
        history_frames=np.asarray([int(args.history_frames)], dtype=np.int32),
        model_history_frames=np.asarray([int(MODEL_HISTORY_FRAMES)], dtype=np.int32),
        readout_time_bin=np.asarray([int(args.history_frames) - 1], dtype=np.int32),
        terminal_frames=np.asarray([1], dtype=np.int32),
        fd_step_arcmin=np.asarray([float(args.fd_step_arcmin)], dtype=np.float32),
        history_window=np.asarray([HISTORY_WINDOW]),
        endpoint_alignment=np.asarray([ENDPOINT_ALIGNMENT]),
        readout_time_contract=np.asarray(["terminal_response_only"]),
        stimulus_normalization=np.asarray([STIMULUS_NORMALIZATION]),
        zero_movie_shape=np.asarray(zero_shape, dtype=np.int32),
        plus_movie_shape=np.asarray(plus_shape, dtype=np.int32),
        minus_movie_shape=np.asarray(minus_shape, dtype=np.int32),
        endpoint_norm_deg=np.asarray([float(trace_meta["endpoint_norm_deg"])], dtype=np.float32),
        history_rms_deg=np.asarray([float(trace_meta["history_rms_deg"])], dtype=np.float32),
        history_path_length_deg=np.asarray([float(trace_meta["history_path_length_deg"])], dtype=np.float32),
        history_max_radius_deg=np.asarray([float(trace_meta["history_max_radius_deg"])], dtype=np.float32),
        seed=np.asarray([int(args.seed)], dtype=np.int64),
        condition_seed_index=np.asarray([endpoint_condition_seed_index(condition)], dtype=np.int32),
        eye_traces_path=np.asarray([str(Path(args.eye_traces_path).expanduser().resolve())]),
        cache_identity_json=np.asarray([_identity_text(expected_identity)]),
    )
    print(f"Computed endpoint terminal maps for {condition}: {zero.shape}", flush=True)
    meta = {
        "zero_movie_shape": zero_shape,
        "plus_movie_shape": plus_shape,
        "minus_movie_shape": minus_shape,
        "endpoint_norm_deg": float(trace_meta["endpoint_norm_deg"]),
        "history_rms_deg": float(trace_meta["history_rms_deg"]),
        "history_path_length_deg": float(trace_meta["history_path_length_deg"]),
        "history_max_radius_deg": float(trace_meta["history_max_radius_deg"]),
        "source_condition": str(ENDPOINT_CONDITIONS[condition]["source_condition"]),
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "cache_identity_json": _identity_text(expected_identity),
    }
    return cache_path, {"zero": zero, "plus": plus, "minus": minus}, meta


def _add_full_reference_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return rows
    full = df[df["population_key"] == "full756"][
        [
            "condition",
            "last_frame_ssi_bits_per_spike",
            "last_frame_total_rate",
            "last_frame_ssi_bits_frame_proxy",
        ]
    ].rename(
        columns={
            "last_frame_ssi_bits_per_spike": "full_last_frame_ssi_bits_per_spike",
            "last_frame_total_rate": "full_last_frame_total_rate",
            "last_frame_ssi_bits_frame_proxy": "full_last_frame_ssi_bits_frame_proxy",
        }
    )
    df = df.merge(full, on="condition", how="left")
    for num, den, out in [
        (
            "last_frame_ssi_bits_per_spike",
            "full_last_frame_ssi_bits_per_spike",
            "last_frame_ssi_bits_per_spike_vs_full",
        ),
        ("last_frame_total_rate", "full_last_frame_total_rate", "last_frame_total_rate_vs_full"),
        (
            "last_frame_ssi_bits_frame_proxy",
            "full_last_frame_ssi_bits_frame_proxy",
            "last_frame_ssi_bits_frame_proxy_vs_full",
        ),
    ]:
        df[out] = df[num] / df[den].replace(0.0, np.nan)
    return df.to_dict(orient="records")


def _draw_ssi_bars(summary_rows: list[dict[str, Any]], conditions: list[str], out_dir: Path, dpi: int) -> Path | None:
    if not summary_rows:
        return None
    df = pd.DataFrame(summary_rows)
    pop_order = [pop for pop in ("full756", "rr100_medoid") if pop in set(df["population_key"])]
    x = np.arange(len(conditions), dtype=float)
    width = min(0.36, 0.76 / max(len(pop_order), 1))
    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, 1.35 * len(conditions)), 4.2), dpi=int(dpi), constrained_layout=True)
    for idx, pop in enumerate(pop_order):
        sub = df[df["population_key"] == pop].set_index("condition")
        values = [float(sub.loc[c, "last_frame_ssi_bits_per_spike"]) if c in sub.index else np.nan for c in conditions]
        labels = sub["population_label"].dropna().unique()
        label = str(labels[0]) if len(labels) else pop
        axes[0].bar(x + (idx - (len(pop_order) - 1) / 2.0) * width, values, width=width, label=label)
    axes[0].set_ylabel("terminal-map SSI (bits/spike)")
    axes[0].set_title("Last-frame SSI")
    axes[0].legend(frameon=False)

    rr = df[df["population_key"] == "rr100_medoid"].set_index("condition")
    if not rr.empty and "last_frame_ssi_bits_per_spike_vs_full" in rr.columns:
        values = [float(rr.loc[c, "last_frame_ssi_bits_per_spike_vs_full"]) if c in rr.index else np.nan for c in conditions]
        axes[1].bar(x, values, color="#4c78a8")
        axes[1].axhline(1.0, color="#777777", linewidth=0.8)
    axes[1].set_ylabel("RR100 SSI / full 756")
    axes[1].set_title("Population retention")

    labels = [_condition_label(c) for c in conditions]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Endpoint-history Vernier SSI from terminal frame", y=1.03)
    path = out_dir / "vernier_endpoint_history_last_frame_ssi_bars.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    args.render_resolution_factors = parse_csv_float(args.render_resolution_factors)
    _warn_if_history_exceeds_model_lags(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = parse_csv_str(args.conditions)
    _validate_conditions(conditions)
    populations = population_specs(parse_csv_str(args.populations), str(args.rr100_version))
    geometry = RenderGeometry()
    trace_set = load_eye_traces(Path(args.eye_traces_path))
    model, readout = load_model_and_readout(args.device)
    device = str(next(model.model.parameters()).device)

    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    gallery_items_by_pop: dict[str, list[dict[str, Any]]] = {str(pop["key"]): [] for pop in populations}
    condition_sources: list[dict[str, Any]] = []

    for condition in conditions:
        cache_path, maps, trace_meta = _load_or_compute_condition_maps(
            args,
            condition=condition,
            trace_set=trace_set,
            model=model,
            readout=readout,
            geometry=geometry,
            device=device,
        )
        condition_sources.append(
            {
                "condition": condition,
                "condition_label": _condition_label(condition),
                "source_condition": ENDPOINT_CONDITIONS[condition]["source_condition"],
                "cache_npz": cache_path,
                **trace_meta,
            }
        )
        full_map = maps[str(args.map_kind_for_ssi)]
        for pop in populations:
            pop_map = apply_population(full_map, pop["view"])
            ssi = ssi_single_frame(pop_map)
            rate = total_rate(pop_map)
            image = collapse_activation_map(pop_map, str(args.collapse))
            row = {
                "condition": condition,
                "condition_label": _condition_label(condition),
                "source_condition": ENDPOINT_CONDITIONS[condition]["source_condition"],
                "population_key": pop["key"],
                "population_label": pop["label"],
                "population_version": pop["version"],
                "n_units": int(pop_map.shape[0]),
                "height": int(pop_map.shape[1]),
                "width": int(pop_map.shape[2]),
                "trace_index": int(args.trace_index),
                "history_frames": int(args.history_frames),
                "model_history_frames": int(MODEL_HISTORY_FRAMES),
                "readout_time_bin": int(args.history_frames) - 1,
                "terminal_frames": 1,
                "fd_step_arcmin": float(args.fd_step_arcmin),
                "map_kind": str(args.map_kind_for_ssi),
                "history_window": HISTORY_WINDOW,
                "endpoint_alignment": ENDPOINT_ALIGNMENT,
                "readout_time_contract": "terminal_response_only",
                "endpoint_norm_deg": float(trace_meta["endpoint_norm_deg"]),
                "history_rms_deg": float(trace_meta["history_rms_deg"]),
                "history_path_length_deg": float(trace_meta["history_path_length_deg"]),
                "history_max_radius_deg": float(trace_meta["history_max_radius_deg"]),
                "last_frame_ssi_bits_per_spike": float(ssi["population_bits_per_spike"]),
                "last_frame_total_rate": float(rate),
                "last_frame_ssi_bits_frame_proxy": float(ssi["population_bits_per_spike"] * rate),
                "cache_npz": cache_path,
            }
            summary_rows.append(row)
            gallery_items_by_pop[str(pop["key"])].append(
                {
                    "condition": condition,
                    "condition_label": _condition_label(condition),
                    "population_key": pop["key"],
                    "population_label": pop["label"],
                    "population_version": pop["version"],
                    "rate_map": pop_map,
                    "image": image,
                    "ssi_bits_per_spike": float(ssi["population_bits_per_spike"]),
                    "total_rate": float(rate),
                    "source_npz": cache_path,
                }
            )

    summary_rows = _add_full_reference_ratios(summary_rows)
    summary_csv = out_dir / "vernier_endpoint_history_last_frame_ssi_summary.csv"
    write_csv_rows(summary_csv, summary_rows)

    for pop in populations:
        pop_key = str(pop["key"])
        pop_items = gallery_items_by_pop[pop_key]
        if not pop_items:
            continue
        vmin, vmax = image_scale(
            [item["image"] for item in pop_items],
            float(args.vmin_percentile),
            float(args.vmax_percentile),
        )
        gallery_path = out_dir / f"vernier_endpoint_last_frame_activation_gallery_{pop_key}_{args.map_kind_for_ssi}_{args.collapse}.png"
        draw_gallery(
            pop_items,
            population_label=str(pop["label"]),
            collapse=str(args.collapse),
            vmin=vmin,
            vmax=vmax,
            path=gallery_path,
            dpi=int(args.dpi),
            suptitle=f"Vernier endpoint-history terminal-frame activation maps: {pop['label']}",
        )
        for item in pop_items:
            image_path = (
                out_dir
                / "individual_maps"
                / f"endpoint_last_frame_activation_map_{pop_key}_{safe_slug(item['condition'])}_{args.map_kind_for_ssi}_{args.collapse}.png"
            )
            draw_single_map(
                item["image"],
                title=str(item["condition_label"]).replace("\n", " "),
                subtitle=f"SSI {item['ssi_bits_per_spike']:.5f} bits/spike",
                vmin=vmin,
                vmax=vmax,
                path=image_path,
                dpi=int(args.dpi),
            )
            manifest_rows.append(
                {
                    "condition": item["condition"],
                    "condition_label": str(item["condition_label"]).replace("\n", " "),
                    "population_key": item["population_key"],
                    "population_label": item["population_label"],
                    "population_version": item["population_version"],
                    "map_kind": args.map_kind_for_ssi,
                    "collapse": args.collapse,
                    "ssi_bits_per_spike": item["ssi_bits_per_spike"],
                    "total_rate": item["total_rate"],
                    "image_png": image_path,
                    "gallery_png": gallery_path,
                    "source_npz": item["source_npz"],
                }
            )

    bars_path = _draw_ssi_bars(summary_rows, conditions, out_dir, int(args.dpi))
    manifest_csv = out_dir / "vernier_endpoint_history_last_frame_activation_map_manifest.csv"
    write_csv_rows(manifest_csv, manifest_rows)
    write_json(
        out_dir / "vernier_endpoint_history_last_frame_ssi_manifest.json",
        {
            "analysis": "vernier_endpoint_history_last_frame_ssi",
            "out_dir": out_dir,
            "summary_csv": summary_csv,
            "activation_map_manifest_csv": manifest_csv,
            "conditions": conditions,
            "populations": parse_csv_str(args.populations),
            "trace_index": int(args.trace_index),
            "history_frames": int(args.history_frames),
            "terminal_frames": 1,
            "map_kind_for_ssi": str(args.map_kind_for_ssi),
            "history_window": HISTORY_WINDOW,
            "endpoint_alignment": "tau_endpoint[t] = tau_tail[t] - tau_tail[-1]",
            "readout_time_contract": "SSI computed from the terminal spatial response map only",
            "stimulus_normalization": STIMULUS_NORMALIZATION,
            "condition_sources": condition_sources,
            "bars_png": bars_path,
            "args": vars(args),
        },
    )

    display_cols = [
        "condition",
        "population_key",
        "last_frame_ssi_bits_per_spike",
        "last_frame_ssi_bits_per_spike_vs_full",
        "last_frame_total_rate",
        "endpoint_norm_deg",
    ]
    print("\nEndpoint-history last-frame SSI:")
    print(pd.DataFrame(summary_rows)[display_cols].to_string(index=False))
    print(f"\nWrote SSI summary: {summary_csv}")
    print(f"Wrote activation-map manifest: {manifest_csv}")
    if bars_path is not None:
        print(f"Wrote SSI bar plot: {bars_path}")


if __name__ == "__main__":
    main()
