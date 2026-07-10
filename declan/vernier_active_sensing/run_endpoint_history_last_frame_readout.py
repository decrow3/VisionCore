#!/usr/bin/env python3
"""Run Vernier endpoint-history, last-frame-only readout diagnostics.

This mirrors the endpoint-history feature-readout contract used in the newest
Figure 4 feature model:

    endpoint_trace[t] = trace_tail[t] - trace_tail[-1]

All conditions therefore have the same final retinal endpoint within the final
32-frame history window, but the model sees different histories.  The readout
uses only the terminal response frame.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from declan.redundancy_resolved_v1_population import PopulationView, load_population_view
from declan.vernier_active_sensing.forward import (
    STIMULUS_NORMALIZATION,
    compute_vernier_rates,
    load_model_and_readout,
)
from declan.vernier_active_sensing.metrics import expected_counts, poisson_fisher_counts, pose_blind_diagonal_fisher
from declan.vernier_active_sensing.run_vernier_active_sensing import (
    build_spec,
    json_ready,
    parse_csv_float,
    parse_csv_str,
    summarize_condition_rows,
    summarize_contrast_rows,
    summarize_information,
    write_csv,
    write_json,
)
from declan.vernier_active_sensing.stimulus import RenderGeometry, save_pixel_audit_artifacts
from declan.vernier_active_sensing.trajectories import (
    condition_trace,
    load_eye_traces,
    subsample_traces,
    valid_trace,
)
from scripts.temporal_decoding.stimulus_hires import N_LAGS as MODEL_HISTORY_FRAMES


DEFAULT_OUT_DIR = Path("outputs") / "vernier_endpoint_history_last_frame_tutorial"
RR100_MOVIE_MEDOID_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)


ENDPOINT_CONDITIONS: dict[str, dict[str, str]] = {
    "static_endpoint_history": {
        "source_condition": "static_center",
        "label": "static endpoint",
        "interpretation": "All history frames are fixed at the shared endpoint.",
    },
    "real_fem_endpoint_history": {
        "source_condition": "real_fem",
        "label": "real FEM endpoint history",
        "interpretation": "Recorded FEM history after subtracting its terminal eye position.",
    },
    "order_shuffled_endpoint_history": {
        "source_condition": "order_shuffled_positions",
        "label": "order-shuffled endpoint history",
        "interpretation": "Recorded positions are order-shuffled, then terminal-position aligned.",
    },
    "phase_cloud_endpoint_history": {
        "source_condition": "static_phase_cloud_matched_positions",
        "label": "phase-cloud endpoint history",
        "interpretation": (
            "Same-trace position-cloud control, terminal-position aligned; this uses the same "
            "position-permutation family as the order-shuffled control."
        ),
    },
    "horizontal_endpoint_history": {
        "source_condition": "axis_horizontal",
        "label": "horizontal endpoint history",
        "interpretation": "Across-contour component history, terminal-position aligned.",
    },
    "vertical_endpoint_history": {
        "source_condition": "axis_vertical",
        "label": "vertical endpoint history",
        "interpretation": "Along-contour component history, terminal-position aligned.",
    },
}

DEFAULT_CONDITIONS = (
    "static_endpoint_history",
    "real_fem_endpoint_history",
    "order_shuffled_endpoint_history",
    "phase_cloud_endpoint_history",
    "horizontal_endpoint_history",
    "vertical_endpoint_history",
)

ENDPOINT_CONDITION_SEED_INDEX = {condition: idx for idx, condition in enumerate(DEFAULT_CONDITIONS)}
HISTORY_WINDOW = "last_n_valid_trace_frames"
ENDPOINT_ALIGNMENT = "last_history_window_minus_final_position"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--eye-traces-path", type=Path, default=Path("scripts/temporal_decoding/data/eye_traces.npz"))
    parser.add_argument("--conditions", type=str, default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--fd-steps-arcmin", type=str, default="0.25")
    parser.add_argument("--history-frames", type=int, default=32)
    parser.add_argument("--terminal-frames", type=int, default=1)
    parser.add_argument("--n-traces", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inference-mode", type=str, default="framewise", choices=("framewise", "continuous"))
    parser.add_argument("--spatial-collapse", type=str, default="max", choices=("max", "mean"))
    parser.add_argument("--population", choices=("full756", "rr100_medoid"), default="rr100_medoid")
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--frame-rate-hz", type=float, default=120.0)
    parser.add_argument("--bar-width-arcmin", type=float, default=2.0)
    parser.add_argument("--gap-arcmin", type=float, default=4.0)
    parser.add_argument("--bar-length-arcmin", type=float, default=12.0)
    parser.add_argument("--contrast", type=float, default=0.5)
    parser.add_argument("--polarity", type=str, default="bright", choices=("bright", "dark"))
    parser.add_argument("--stimulus-orientation-deg", type=float, default=0.0)
    parser.add_argument("--render-resolution-factors", type=str, default="1")
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def _validate_conditions(conditions: list[str]) -> None:
    invalid = [condition for condition in conditions if condition not in ENDPOINT_CONDITIONS]
    if invalid:
        valid = ", ".join(ENDPOINT_CONDITIONS)
        raise ValueError(f"Unsupported endpoint-history conditions {invalid}; valid={valid}")


def endpoint_condition_seed_index(condition: str) -> int:
    """Canonical seed slot for endpoint controls, independent of CLI ordering."""
    condition = str(condition)
    if condition not in ENDPOINT_CONDITION_SEED_INDEX:
        valid = ", ".join(ENDPOINT_CONDITION_SEED_INDEX)
        raise ValueError(f"Unsupported endpoint-history condition {condition!r}; valid={valid}")
    return int(ENDPOINT_CONDITION_SEED_INDEX[condition])


def endpoint_condition_rng(seed: int, condition: str, trace_index: int) -> np.random.Generator:
    return np.random.default_rng(
        int(seed) + 1009 * endpoint_condition_seed_index(str(condition)) + 9176 * int(trace_index)
    )


def warn_if_framewise_history_exceeds_model_lags(args: argparse.Namespace) -> None:
    if str(getattr(args, "inference_mode", "framewise")) != "framewise":
        return
    if int(args.history_frames) <= int(MODEL_HISTORY_FRAMES):
        return
    warnings.warn(
        (
            f"--history-frames={int(args.history_frames)} exceeds the model lag window "
            f"({int(MODEL_HISTORY_FRAMES)}) in framewise mode; terminal-frame scores "
            "can only depend on the final lag window."
        ),
        RuntimeWarning,
        stacklevel=2,
    )


def endpoint_aligned_trace(trace: np.ndarray) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"trace must be shaped (time, 2), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("trace must include at least one frame")
    return (arr - arr[-1:, :]).astype(np.float32, copy=False)


def terminal_history_window(trace: np.ndarray, history_frames: int) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"trace must be shaped (time, 2), got {arr.shape}")
    n_frames = int(history_frames)
    if n_frames < 1:
        raise ValueError(f"history_frames must be positive, got {history_frames}")
    if arr.shape[0] < n_frames:
        raise ValueError(f"trace has {arr.shape[0]} frames, fewer than requested history_frames={n_frames}")
    return arr[-n_frames:].astype(np.float32, copy=False)


def _trace_metrics(trace: np.ndarray) -> dict[str, float]:
    arr = np.asarray(trace, dtype=np.float64)
    steps = np.diff(arr, axis=0) if arr.shape[0] > 1 else np.zeros((0, 2), dtype=np.float64)
    radius = np.linalg.norm(arr, axis=1)
    return {
        "endpoint_x_deg": float(arr[-1, 0]),
        "endpoint_y_deg": float(arr[-1, 1]),
        "endpoint_norm_deg": float(radius[-1]),
        "history_rms_deg": float(np.sqrt(np.mean(np.sum(arr * arr, axis=1)))),
        "history_path_length_deg": float(np.sum(np.linalg.norm(steps, axis=1))),
        "history_max_radius_deg": float(np.max(radius)),
    }


def build_endpoint_trace(
    base_trace: np.ndarray,
    *,
    condition: str,
    trace_set: Any,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = ENDPOINT_CONDITIONS[str(condition)]
    effective, meta = condition_trace(
        base_trace,
        condition=str(spec["source_condition"]),
        trace_set=trace_set,
        rng=rng,
        frame_rate_hz=float(args.frame_rate_hz),
    )
    history = terminal_history_window(effective, int(args.history_frames))
    endpoint = endpoint_aligned_trace(history)
    out_meta = {
        "condition_family": "endpoint_history",
        "source_condition": str(spec["source_condition"]),
        "history_window": HISTORY_WINDOW,
        "endpoint_alignment": ENDPOINT_ALIGNMENT,
        "readout_time_contract": "terminal_response_only",
        "condition_label": str(spec["label"]),
        "condition_interpretation": str(spec["interpretation"]),
    }
    out_meta.update(meta)
    out_meta.update(_trace_metrics(endpoint))
    return endpoint, out_meta


def _population_view(args: argparse.Namespace) -> PopulationView | None:
    if str(args.population) == "full756":
        return None
    return load_population_view(version_name=str(args.rr100_version))


def _apply_population_rates(rates: np.ndarray, view: PopulationView | None) -> np.ndarray:
    arr = np.asarray(rates, dtype=np.float32)
    if view is None or view.membership is None:
        return arr
    membership = np.asarray(view.membership, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != membership.shape[1]:
        raise ValueError(f"rates must be (time, {membership.shape[1]}), got {arr.shape}")
    return (arr @ membership.T).astype(np.float32, copy=False)


def _terminal_rates(rates: np.ndarray, terminal_frames: int) -> np.ndarray:
    arr = np.asarray(rates, dtype=np.float32)
    frames = int(terminal_frames)
    if frames <= 0:
        raise ValueError("--terminal-frames must be positive")
    if arr.ndim != 2 or arr.shape[0] < frames:
        raise ValueError(f"Cannot take {frames} terminal frames from rates shaped {arr.shape}")
    return arr[-frames:].astype(np.float32, copy=False)


def _pad_rate_rows(rates: list[np.ndarray]) -> np.ndarray:
    n = len(rates)
    t = max(arr.shape[0] for arr in rates)
    u = rates[0].shape[1]
    out = np.full((n, t, u), np.nan, dtype=np.float32)
    for idx, arr in enumerate(rates):
        out[idx, : arr.shape[0], :] = arr
    return out


def run_endpoint_readout(
    args: argparse.Namespace,
    out_dir: Path,
    geometry: RenderGeometry,
    conditions: list[str],
    fd_steps: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_set = subsample_traces(load_eye_traces(Path(args.eye_traces_path)), int(args.n_traces), int(args.seed))
    model, readout = load_model_and_readout(args.device)
    view = _population_view(args)
    population_label = "full 756" if view is None else f"RR100 movie-medoid ({view.n_units} reps)"
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for step in fd_steps:
        plus_spec = build_spec(args, float(step))
        minus_spec = build_spec(args, -float(step))
        for condition in conditions:
            print(f"Endpoint-history condition={condition} fd_step={step} arcmin", flush=True)
            plus_terminal: list[np.ndarray] = []
            minus_terminal: list[np.ndarray] = []
            for trace_idx in range(trace_set.traces.shape[0]):
                base_trace = valid_trace(trace_set, trace_idx)
                rng = endpoint_condition_rng(int(args.seed), condition, trace_idx)
                endpoint_trace, trace_meta = build_endpoint_trace(
                    base_trace,
                    condition=condition,
                    trace_set=trace_set,
                    rng=rng,
                    args=args,
                )
                plus = compute_vernier_rates(
                    model,
                    readout,
                    plus_spec,
                    endpoint_trace,
                    inference_mode=str(args.inference_mode),
                    geometry=geometry,
                    batch_size=int(args.batch_size),
                    spatial_collapse=str(args.spatial_collapse),
                    device=args.device,
                )
                minus = compute_vernier_rates(
                    model,
                    readout,
                    minus_spec,
                    endpoint_trace,
                    inference_mode=str(args.inference_mode),
                    geometry=geometry,
                    batch_size=int(args.batch_size),
                    spatial_collapse=str(args.spatial_collapse),
                    device=args.device,
                )
                plus = _terminal_rates(_apply_population_rates(plus, view), int(args.terminal_frames))
                minus = _terminal_rates(_apply_population_rates(minus, view), int(args.terminal_frames))
                plus_terminal.append(plus)
                minus_terminal.append(minus)
                info = poisson_fisher_counts(
                    expected_counts(plus, float(args.bin_seconds)),
                    expected_counts(minus, float(args.bin_seconds)),
                    step_arcmin=float(step),
                    phi=float(args.phi),
                )
                summary_rows.append(
                    {
                        "readout": "endpoint_known_history_terminal_frame_poisson",
                        "condition": condition,
                        "condition_label": ENDPOINT_CONDITIONS[condition]["label"],
                        "trace_index": int(trace_idx),
                        "fd_step_arcmin": float(step),
                        "inference_mode": str(args.inference_mode),
                        "population": str(args.population),
                        "population_label": population_label,
                        "n_timebins": int(plus.shape[0]),
                        "n_units": int(plus.shape[1]),
                        "history_frames": int(args.history_frames),
                        "terminal_frames": int(args.terminal_frames),
                        "readout_time_contract": "terminal_response_only",
                        "history_window": HISTORY_WINDOW,
                        "endpoint_alignment": ENDPOINT_ALIGNMENT,
                        **summarize_information(info),
                    }
                )
                trace_rows.append(
                    {
                        "condition": condition,
                        "condition_label": ENDPOINT_CONDITIONS[condition]["label"],
                        "trace_index": int(trace_idx),
                        "fd_step_arcmin": float(step),
                        "n_input_frames": int(base_trace.shape[0]),
                        "history_frames": int(endpoint_trace.shape[0]),
                        "terminal_frames": int(args.terminal_frames),
                        **trace_meta,
                    }
                )

            if len(plus_terminal) > 1:
                pooled = pose_blind_diagonal_fisher(
                    plus_terminal,
                    minus_terminal,
                    step_arcmin=float(step),
                    bin_seconds=float(args.bin_seconds),
                    phi=float(args.phi),
                )
                summary_rows.append(
                    {
                        "readout": "endpoint_history_marginal_terminal_frame_poisson",
                        "condition": condition,
                        "condition_label": ENDPOINT_CONDITIONS[condition]["label"],
                        "trace_index": "all",
                        "fd_step_arcmin": float(step),
                        "inference_mode": str(args.inference_mode),
                        "population": str(args.population),
                        "population_label": population_label,
                        "n_timebins": int(pooled["cumulative_fisher"].shape[0]),
                        "n_units": int(plus_terminal[0].shape[1]),
                        "history_frames": int(args.history_frames),
                        "terminal_frames": int(args.terminal_frames),
                        "readout_time_contract": "terminal_response_only",
                        "history_window": HISTORY_WINDOW,
                        "endpoint_alignment": ENDPOINT_ALIGNMENT,
                        **summarize_information(pooled),
                    }
                )

            cache_path = out_dir / "cache" / f"endpoint_terminal_rates_{condition}_fd{float(step):.4f}arcmin.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                plus=_pad_rate_rows(plus_terminal),
                minus=_pad_rate_rows(minus_terminal),
                lengths=np.asarray([arr.shape[0] for arr in plus_terminal], dtype=np.int32),
                condition=np.asarray([condition]),
                condition_label=np.asarray([ENDPOINT_CONDITIONS[condition]["label"]]),
                fd_step_arcmin=np.asarray([float(step)], dtype=np.float32),
                population=np.asarray([str(args.population)]),
                history_frames=np.asarray([int(args.history_frames)], dtype=np.int32),
                terminal_frames=np.asarray([int(args.terminal_frames)], dtype=np.int32),
                history_window=np.asarray([HISTORY_WINDOW]),
                endpoint_alignment=np.asarray([ENDPOINT_ALIGNMENT]),
                readout_time_contract=np.asarray(["terminal_response_only"]),
                stimulus_normalization=np.asarray([STIMULUS_NORMALIZATION]),
                seed=np.asarray([int(args.seed)], dtype=np.int64),
                condition_seed_index=np.asarray([endpoint_condition_seed_index(condition)], dtype=np.int32),
                eye_traces_path=np.asarray([str(Path(args.eye_traces_path).expanduser().resolve())]),
            )
    return summary_rows, trace_rows


def _condition_baselines(condition: str) -> list[str]:
    baselines = ["static_endpoint_history"]
    if condition == "real_fem_endpoint_history":
        baselines.extend(["phase_cloud_endpoint_history", "order_shuffled_endpoint_history"])
    return [baseline for baseline in baselines if baseline != condition]


def paired_baseline_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: dict[tuple[str, float, Any], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("readout")) != "endpoint_known_history_terminal_frame_poisson":
            continue
        key = (str(row.get("readout")), float(row.get("fd_step_arcmin")), row.get("trace_index"))
        table.setdefault(key, {})[str(row.get("condition"))] = row
    out: list[dict[str, Any]] = []
    for (readout, fd_step, trace_index), by_condition in sorted(table.items(), key=lambda item: str(item[0])):
        for condition, row in by_condition.items():
            for baseline in _condition_baselines(condition):
                if baseline not in by_condition:
                    continue
                f = float(row.get("final_fisher", np.nan))
                fb = float(by_condition[baseline].get("final_fisher", np.nan))
                if not (np.isfinite(f) and np.isfinite(fb)):
                    continue
                out.append(
                    {
                        "readout": readout,
                        "inference_mode": str(row.get("inference_mode", "")),
                        "fd_step_arcmin": fd_step,
                        "trace_index": trace_index,
                        "condition": condition,
                        "condition_label": ENDPOINT_CONDITIONS[condition]["label"],
                        "baseline_condition": baseline,
                        "baseline_condition_label": ENDPOINT_CONDITIONS[baseline]["label"],
                        "condition_final_fisher": f,
                        "baseline_final_fisher": fb,
                        "fisher_delta": f - fb,
                        "fisher_ratio": f / fb if fb > 0 else float("nan"),
                        "threshold_ratio": np.sqrt(fb / f) if f > 0 and fb >= 0 else float("nan"),
                        "condition_beats_baseline": bool(f > fb),
                    }
                )
    return out


def plot_summary(out_dir: Path, condition_rows: list[dict[str, Any]]) -> Path | None:
    selected = [
        row
        for row in condition_rows
        if str(row.get("readout")) in {
            "endpoint_known_history_terminal_frame_poisson",
            "endpoint_history_marginal_terminal_frame_poisson",
        }
    ]
    if not selected:
        return None
    order = [condition for condition in DEFAULT_CONDITIONS if any(row["condition"] == condition for row in selected)]
    readouts = list(dict.fromkeys(str(row["readout"]) for row in selected))
    labels = {
        "endpoint_known_history_terminal_frame_poisson": "known history",
        "endpoint_history_marginal_terminal_frame_poisson": "history marginal",
    }
    x = np.arange(len(order), dtype=float)
    width = min(0.36, 0.72 / max(len(readouts), 1))
    fig, ax = plt.subplots(figsize=(max(7.5, 1.1 * len(order)), 4.2), dpi=150, constrained_layout=True)
    for idx, readout in enumerate(readouts):
        rows_by_cond = {str(row["condition"]): row for row in selected if str(row["readout"]) == readout}
        values = [float(rows_by_cond.get(condition, {}).get("mean_final_fisher", np.nan)) for condition in order]
        offset = (idx - (len(readouts) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=labels.get(readout, readout))
    ax.set_xticks(x)
    ax.set_xticklabels([ENDPOINT_CONDITIONS[c]["label"] for c in order], rotation=28, ha="right")
    ax.set_ylabel("terminal-frame Fisher")
    ax.set_title("Vernier endpoint-history readout: same endpoint, last frame only")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    path = out_dir / "endpoint_history_last_frame_fisher.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    args.render_resolution_factors = parse_csv_float(args.render_resolution_factors)
    warn_if_framewise_history_exceeds_model_lags(args)
    conditions = parse_csv_str(args.conditions)
    fd_steps = parse_csv_float(args.fd_steps_arcmin)
    _validate_conditions(conditions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry = RenderGeometry()
    canonical_spec = build_spec(args, 0.0)

    pixel_audit = save_pixel_audit_artifacts(
        out_dir / "render_audit",
        canonical_spec,
        fd_steps_arcmin=fd_steps,
        geometry=geometry,
        device=args.device or "cpu",
        resolution_factors=args.render_resolution_factors,
    )
    write_json(out_dir / "render_audit" / "pixel_audit.json", pixel_audit)
    write_csv(out_dir / "render_audit" / "pixel_audit_fd_rows.csv", pixel_audit["fd_rows"])

    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    contrast_summary_rows: list[dict[str, Any]] = []
    figure_path: Path | None = None
    if not bool(args.skip_model):
        summary_rows, trace_rows = run_endpoint_readout(args, out_dir, geometry, conditions, fd_steps)
        condition_rows = summarize_condition_rows(summary_rows)
        contrast_rows = paired_baseline_contrasts(summary_rows)
        contrast_summary_rows = summarize_contrast_rows(contrast_rows)
        write_csv(out_dir / "endpoint_history_last_frame_trials.csv", summary_rows)
        write_csv(out_dir / "endpoint_history_trace_metrics.csv", trace_rows)
        write_csv(out_dir / "endpoint_history_last_frame_summary.csv", condition_rows)
        write_csv(out_dir / "endpoint_history_last_frame_contrasts.csv", contrast_rows)
        write_csv(out_dir / "endpoint_history_last_frame_contrast_summary.csv", contrast_summary_rows)
        figure_path = plot_summary(out_dir, condition_rows)

    write_json(
        out_dir / "vernier_endpoint_history_last_frame_manifest.json",
        {
            "analysis": "vernier_endpoint_history_last_frame_readout",
            "assay": {
                "history_window": HISTORY_WINDOW,
                "endpoint_alignment": "tau_endpoint[t] = tau_tail[t] - tau_tail[-1]",
                "readout_contract": "decode only terminal response frame/window",
                "target_contract": "Vernier finite-difference offset at shared final retinal endpoint",
                "stimulus_normalization": STIMULUS_NORMALIZATION,
                "history_frames": int(args.history_frames),
                "terminal_frames": int(args.terminal_frames),
                "conditions": conditions,
                "fd_steps_arcmin": fd_steps,
                "population": str(args.population),
                "skip_model": bool(args.skip_model),
            },
            "conditions": [
                {
                    "condition": condition,
                    **ENDPOINT_CONDITIONS[condition],
                }
                for condition in conditions
            ],
            "geometry": asdict(geometry),
            "canonical_spec": asdict(canonical_spec),
            "args": vars(args),
            "outputs": {
                "trials": out_dir / "endpoint_history_last_frame_trials.csv",
                "trace_metrics": out_dir / "endpoint_history_trace_metrics.csv",
                "summary": out_dir / "endpoint_history_last_frame_summary.csv",
                "contrasts": out_dir / "endpoint_history_last_frame_contrasts.csv",
                "contrast_summary": out_dir / "endpoint_history_last_frame_contrast_summary.csv",
                "figure": figure_path,
                "rate_cache_dir": out_dir / "cache",
                "render_audit_dir": out_dir / "render_audit",
            },
            "provenance": "endpoint-aligned Vernier histories plus canonical twin terminal-frame readout",
        },
    )
    print(f"Wrote Vernier endpoint-history last-frame outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
