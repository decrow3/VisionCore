#!/usr/bin/env python3
"""Native-readout smoke test for the zero-gaze separable RR100 SF/TF probe.

This deliberately evaluates a small condition set.  It verifies the 33-frame
input embedding, session-native adapter/readout mapping, unused-behavior
contract, signed temporal drift, gray baseline, and unit response traces before
the production Cartesian grid is launched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from DataYatesV1.exp.gratings import GratingsTrial
from DataYatesV1.utils.io import get_session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.run_rr100_original_sequence_dense_sf_native_readout import (
    RR100_VERSION,
    load_rr100_rows,
)
from declan.run_rr100_zero_gaze_separable_sf_tf_input_checkpoint import (
    FRAME_RATE_HZ,
    HISTORY_FRAMES,
    derive_zero_gaze_roi,
    make_renderer_extended_movie,
)

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.utils import get_model_and_dataset_configs  # noqa: E402


DEFAULT_OUT = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_native_smoke_v1"
)
DT = 1.0 / FRAME_RATE_HZ


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--population-version", default=RR100_VERSION)
    parser.add_argument("--sessions", default="Allen_2022-02-16")
    parser.add_argument("--spatial-cpds", default="2,8")
    parser.add_argument("--signed-temporal-hz", default="0,4,-4,16,-16,45.254834")
    parser.add_argument("--orientation-deg", default="90")
    parser.add_argument("--static-phases", type=int, default=2)
    parser.add_argument("--dynamic-phases", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=0.6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {requested}, but CUDA is unavailable")
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def embed_native_history(movie_uint8: np.ndarray) -> torch.Tensor:
    movie = torch.from_numpy(np.asarray(movie_uint8)).to(torch.float32)
    movie = (movie - 127.0) / 255.0
    if len(movie) < HISTORY_FRAMES:
        raise ValueError("Movie is shorter than the 33-frame native history")
    current = torch.arange(HISTORY_FRAMES - 1, len(movie), dtype=torch.long)
    lags = torch.arange(HISTORY_FRAMES, dtype=torch.long)
    indices = current[:, None] - lags[None, :]
    # Native CombinedEmbeddedDataset returns B x C x lag x H x W with
    # lag order current, t-1, ..., t-32.
    return movie[indices].unsqueeze(1)


def condition_table(args: argparse.Namespace) -> pd.DataFrame:
    sfs = parse_float_list(args.spatial_cpds)
    signed_tfs = parse_float_list(args.signed_temporal_hz)
    orientations = parse_float_list(args.orientation_deg)
    rows: list[dict[str, Any]] = [
        {
            "condition_id": 0,
            "condition_kind": "gray_blank",
            "spatial_cpd": 0.0,
            "signed_temporal_hz": 0.0,
            "orientation_deg": 0.0,
            "phase_index": 0,
            "phase_rad": 0.0,
        }
    ]
    condition_id = 1
    for orientation in orientations:
        for sf in sfs:
            for tf in signed_tfs:
                n_phases = args.static_phases if np.isclose(tf, 0.0) else args.dynamic_phases
                for phase_index in range(max(1, int(n_phases))):
                    rows.append(
                        {
                            "condition_id": condition_id,
                            "condition_kind": "static_grating" if np.isclose(tf, 0.0) else "drifting_grating",
                            "spatial_cpd": sf,
                            "signed_temporal_hz": tf,
                            "orientation_deg": orientation,
                            "phase_index": phase_index,
                            "phase_rad": float(2.0 * math.pi * phase_index / max(1, int(n_phases))),
                        }
                    )
                    condition_id += 1
    return pd.DataFrame(rows)


def build_movie(
    row: pd.Series,
    trial: GratingsTrial,
    roi: np.ndarray,
    n_valid_frames: int,
) -> np.ndarray:
    total_frames = n_valid_frames + HISTORY_FRAMES - 1
    if row["condition_kind"] == "gray_blank":
        return np.full((total_frames, 51, 51), int(round(float(trial.bkgnd))), dtype=np.uint8)
    # Generate enough preceding carrier frames that the first evaluated sample
    # already has a complete phase-consistent history.
    movie = make_renderer_extended_movie(
        trial,
        roi,
        spatial_cpd=float(row["spatial_cpd"]),
        signed_temporal_hz=float(row["signed_temporal_hz"]),
        orientation_deg=float(row["orientation_deg"]),
        phase_rad=float(row["phase_rad"]) + 2.0 * math.pi * float(row["signed_temporal_hz"]) * (HISTORY_FRAMES - 1) / FRAME_RATE_HZ,
        n_frames=total_frames,
    )
    return movie


def predict_native(
    model,
    stimulus: torch.Tensor,
    dataset_idx: int,
    source_indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
    behavior_mode: str = "none",
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(stimulus), batch_size):
            stim = stimulus[start : start + batch_size].to(device)
            behavior = None
            if behavior_mode == "zeros":
                behavior = torch.zeros((len(stim), 42), dtype=stim.dtype, device=device)
            output = model.model(stim, int(dataset_idx), behavior)
            chunks.append(output[:, source_indices].detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def f1_amplitude(rate_hz: np.ndarray, signed_tf: float) -> np.ndarray:
    if np.isclose(signed_tf, 0.0):
        return np.full(rate_hz.shape[1], np.nan, dtype=np.float64)
    t = np.arange(rate_hz.shape[0], dtype=np.float64) / FRAME_RATE_HZ
    omega = 2.0 * math.pi * abs(float(signed_tf)) * t
    design = np.column_stack([np.sin(omega), np.cos(omega), np.ones_like(omega)])
    coefficients, *_ = np.linalg.lstsq(design, rate_hz.astype(np.float64), rcond=None)
    return np.sqrt(coefficients[0] ** 2 + coefficients[1] ** 2)


def summarize_condition(
    row: pd.Series,
    response_counts: np.ndarray,
    rr100_indices: np.ndarray,
    session: str,
    elapsed_s: float,
) -> list[dict[str, Any]]:
    rate = response_counts / DT
    amplitude = f1_amplitude(rate, float(row["signed_temporal_hz"]))
    records: list[dict[str, Any]] = []
    for col, rr100_index in enumerate(rr100_indices):
        records.append(
            {
                "session": session,
                "rr100_index": int(rr100_index),
                **{k: row[k] for k in row.index},
                "n_valid_response_frames": int(len(rate)),
                "mean_rate_hz": float(np.mean(rate[:, col])),
                "median_rate_hz": float(np.median(rate[:, col])),
                "std_rate_hz": float(np.std(rate[:, col])),
                "minimum_rate_hz": float(np.min(rate[:, col])),
                "maximum_rate_hz": float(np.max(rate[:, col])),
                "f1_amplitude_hz": float(amplitude[col]),
                "condition_evaluation_seconds": float(elapsed_s),
            }
        )
    return records


def choose_examples(summary: pd.DataFrame) -> pd.DataFrame:
    dynamic = summary[summary["condition_kind"].eq("drifting_grating")].copy()
    blank = summary[summary["condition_kind"].eq("gray_blank")][["rr100_index", "mean_rate_hz"]].rename(
        columns={"mean_rate_hz": "blank_rate_hz"}
    )
    by_unit = (
        dynamic.groupby("rr100_index")
        .agg(
            dynamic_rate_range_hz=("mean_rate_hz", lambda x: float(np.max(x) - np.min(x))),
            maximum_f1_hz=("f1_amplitude_hz", "max"),
            mean_dynamic_rate_hz=("mean_rate_hz", "mean"),
        )
        .reset_index()
        .merge(blank, on="rr100_index", how="left")
    )
    # Direction asymmetry is matched over |TF|, SF, orientation, and phase.
    signed = dynamic.copy()
    signed["abs_tf"] = signed["signed_temporal_hz"].abs()
    positive = signed[signed["signed_temporal_hz"] > 0].groupby(["rr100_index", "spatial_cpd", "orientation_deg", "phase_index", "abs_tf"])["mean_rate_hz"].mean()
    negative = signed[signed["signed_temporal_hz"] < 0].groupby(["rr100_index", "spatial_cpd", "orientation_deg", "phase_index", "abs_tf"])["mean_rate_hz"].mean()
    direction = positive.to_frame("positive").join(negative.to_frame("negative"), how="inner")
    direction["absolute_direction_difference_hz"] = np.abs(direction["positive"] - direction["negative"])
    direction_score = direction.groupby("rr100_index")["absolute_direction_difference_hz"].max().rename("maximum_direction_difference_hz")
    by_unit = by_unit.merge(direction_score, on="rr100_index", how="left").fillna({"maximum_direction_difference_hz": 0.0})
    roles = [
        ("strong_dynamic_modulation", "dynamic_rate_range_hz", False),
        ("largest_direction_asymmetry", "maximum_direction_difference_hz", False),
        ("weak_response_control", "dynamic_rate_range_hz", True),
    ]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for role, metric, ascending in roles:
        available = by_unit[~by_unit["rr100_index"].isin(used)].sort_values(metric, ascending=ascending)
        if available.empty:
            continue
        row = available.iloc[0].to_dict()
        row.update(
            {
                "selection_role": role,
                "selection_metric": metric,
                "selection_value": float(row[metric]),
                "selection_rule": f"{'minimum' if ascending else 'maximum'} {metric} among not-yet-selected smoke units",
            }
        )
        selected.append(row)
        used.add(int(row["rr100_index"]))
    return pd.DataFrame(selected)


def plot_smoke(summary: pd.DataFrame, traces: dict[tuple[str, int], np.ndarray], selected: pd.DataFrame, out_path: Path, dpi: int) -> None:
    units = selected["rr100_index"].astype(int).tolist()
    fig, axes = plt.subplots(len(units), 2, figsize=(11.8, 3.2 * len(units)), squeeze=False)
    for row_idx, unit in enumerate(units):
        role = str(selected.loc[selected["rr100_index"].eq(unit), "selection_role"].iloc[0])
        sub = summary[(summary["rr100_index"].eq(unit)) & (summary["condition_kind"].ne("gray_blank"))].copy()
        heat = sub.groupby(["spatial_cpd", "signed_temporal_hz"])["mean_rate_hz"].mean().unstack("signed_temporal_hz").sort_index()
        ax = axes[row_idx, 0]
        image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(np.arange(len(heat.columns)), [f"{v:+g}" for v in heat.columns])
        ax.set_yticks(np.arange(len(heat.index)), [f"{v:g}" for v in heat.index])
        ax.set(xlabel="signed TF (Hz)", ylabel="SF (cpd)", title=f"RR100 {unit}: {role}\nmean native rate (Hz)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax = axes[row_idx, 1]
        candidates = sub.sort_values("mean_rate_hz")
        condition_ids = [int(candidates.iloc[0]["condition_id"]), int(candidates.iloc[-1]["condition_id"])]
        for condition_id, color, label_prefix in zip(condition_ids, ["#0072B2", "#D55E00"], ["lowest", "highest"]):
            meta = sub[sub["condition_id"].eq(condition_id)].iloc[0]
            trace = traces[(str(meta["session"]), condition_id)][:, int(np.flatnonzero(summary[summary["session"].eq(meta["session"])]["rr100_index"].drop_duplicates().to_numpy(dtype=int) == unit)[0])] / DT
            t = np.arange(len(trace)) / FRAME_RATE_HZ
            ax.plot(t, trace, color=color, lw=1.5, label=f"{label_prefix}: {meta['spatial_cpd']:g} cpd, {meta['signed_temporal_hz']:+g} Hz")
        ax.set(xlabel="valid response time (s)", ylabel="native fitted rate (Hz)", title="Raw response traces")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, color="0.9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Native RR100 zero-gaze SF/TF smoke test", x=0.02, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))
    requested_sessions = [v.strip() for v in args.sessions.split(",") if v.strip()]
    mapping_rows, population_meta, spec_json, spec_npz = load_rr100_rows(str(args.population_version))
    mapping = pd.DataFrame(
        [
            {
                "rr100_index": int(row["rep_idx"]),
                "session": str(row["selected_session"]),
                "source_unit_index": int(row["selected_source_unit_index"]),
                "canonical_channel": int(row["selected_channel"]),
            }
            for row in mapping_rows
        ]
    )
    missing_sessions = sorted(set(requested_sessions) - set(mapping["session"]))
    if missing_sessions:
        raise ValueError(f"Requested sessions do not contribute RR100 units: {missing_sessions}")
    conditions = condition_table(args)
    conditions.to_csv(args.out_dir / "smoke_condition_table.csv", index=False)
    mapping[mapping["session"].isin(requested_sessions)].to_csv(args.out_dir / "smoke_rr100_unit_mapping.csv", index=False)

    load_start = time.perf_counter()
    model, dataset_configs = get_model_and_dataset_configs(mode="standard")
    model = model.to(device)
    model.model.eval()
    model_load_seconds = time.perf_counter() - load_start
    config_by_session = {str(config["session"]): config for config in dataset_configs}
    summary_rows: list[dict[str, Any]] = []
    trace_cache: dict[tuple[str, int], np.ndarray] = {}
    behavior_audit_rows: list[dict[str, Any]] = []
    session_runtime_rows: list[dict[str, Any]] = []
    n_valid_frames = int(round(float(args.duration_s) * FRAME_RATE_HZ))

    for session in requested_sessions:
        session_start = time.perf_counter()
        subject, date = session.split("_", maxsplit=1)
        sess = get_session(subject, date)
        dset = sess.get_dataset("gratings", strict=True)
        roi, roi_row = derive_zero_gaze_roi(dset, session)
        trial_index = int(np.asarray(dset["trial_inds"])[0])
        trial = GratingsTrial(sess.exp["D"][trial_index], sess.exp["S"])
        unit_map = mapping[mapping["session"].eq(session)].sort_values("rr100_index")
        rr100_indices = unit_map["rr100_index"].to_numpy(dtype=np.int64)
        source_indices = unit_map["source_unit_index"].to_numpy(dtype=np.int64)
        dataset_idx = list(model.names).index(session)
        print(f"{session}: {len(rr100_indices)} RR100 units, {len(conditions)} conditions on {device}", flush=True)
        for condition_number, condition in conditions.iterrows():
            movie = build_movie(condition, trial, roi, n_valid_frames)
            stimulus = embed_native_history(movie)
            start = time.perf_counter()
            response = predict_native(
                model,
                stimulus,
                dataset_idx,
                source_indices,
                device=device,
                batch_size=int(args.batch_size),
            )
            elapsed = time.perf_counter() - start
            if condition_number == 0:
                response_zeros = predict_native(
                    model,
                    stimulus[: min(len(stimulus), int(args.batch_size))],
                    dataset_idx,
                    source_indices,
                    device=device,
                    batch_size=int(args.batch_size),
                    behavior_mode="zeros",
                )
                difference = np.abs(response[: len(response_zeros)] - response_zeros)
                behavior_audit_rows.append(
                    {
                        "session": session,
                        "n_samples": len(response_zeros),
                        "maximum_abs_count_difference_none_vs_zero_behavior": float(np.max(difference)),
                        "all_exact": bool(not np.any(difference)),
                    }
                )
            trace_cache[(session, int(condition["condition_id"]))] = response
            summary_rows.extend(summarize_condition(condition, response, rr100_indices, session, elapsed))
            print(
                f"  [{condition_number + 1}/{len(conditions)}] sf={condition['spatial_cpd']:g} "
                f"tf={condition['signed_temporal_hz']:+g} phase={condition['phase_index']} "
                f"{elapsed:.2f}s",
                flush=True,
            )
        session_runtime_rows.append(
            {
                "session": session,
                "n_rr100_units": len(rr100_indices),
                "n_conditions": len(conditions),
                "n_valid_frames_per_condition": n_valid_frames,
                "wall_seconds": time.perf_counter() - session_start,
            }
        )
        del dset, sess, trial

    summary = pd.DataFrame(summary_rows)
    blank = summary[summary["condition_kind"].eq("gray_blank")][["session", "rr100_index", "mean_rate_hz"]].rename(
        columns={"mean_rate_hz": "blank_rate_hz"}
    )
    summary = summary.merge(blank, on=["session", "rr100_index"], how="left", validate="many_to_one")
    summary["mean_rate_above_blank_hz"] = summary["mean_rate_hz"] - summary["blank_rate_hz"]
    selected = choose_examples(summary)
    summary.to_csv(args.out_dir / "native_smoke_condition_unit_summary.csv", index=False)
    selected.to_csv(args.out_dir / "selected_units.csv", index=False)
    behavior_audit = pd.DataFrame(behavior_audit_rows)
    behavior_audit.to_csv(args.out_dir / "unused_behavior_audit.csv", index=False)
    runtimes = pd.DataFrame(session_runtime_rows)
    runtimes.to_csv(args.out_dir / "session_runtime_summary.csv", index=False)
    traces_path = args.out_dir / "native_smoke_response_traces.npz"
    np.savez_compressed(
        traces_path,
        **{f"{session}__condition_{condition_id:04d}": values for (session, condition_id), values in trace_cache.items()},
    )
    figure = args.out_dir / "native_rr100_zero_gaze_sf_tf_smoke.png"
    plot_smoke(summary, trace_cache, selected, figure, int(args.dpi))
    if not bool(behavior_audit["all_exact"].all()):
        raise AssertionError("Vision-only model predictions changed when zero behavior was supplied")
    manifest = {
        "analysis": "rr100_zero_gaze_separable_sf_tf_native_smoke",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "smoke_not_population_inference",
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "model_load_seconds": model_load_seconds,
        "population_version": str(args.population_version),
        "population_spec_json": str(spec_json.resolve()),
        "population_spec_npz": str(spec_npz.resolve()),
        "n_sessions": len(requested_sessions),
        "n_rr100_units": int(summary["rr100_index"].nunique()),
        "n_conditions": len(conditions),
        "n_valid_frames_per_condition": n_valid_frames,
        "input_history_frames": HISTORY_FRAMES,
        "input_lag_order": "current,t-1,...,t-32",
        "stimulus_normalization": "(uint8-127)/255",
        "native_readout_contract": "session-specific adapter and source unit readout selected by fixed RR100 movie-medoid mapping",
        "behavior_contract": "model modulator is none; None versus 42 zeros verified exactly",
        "all_behavior_audits_exact": bool(behavior_audit["all_exact"].all()),
        "total_session_wall_seconds": float(runtimes["wall_seconds"].sum()),
        "mean_condition_evaluation_seconds": float(summary.drop_duplicates(["session", "condition_id"])["condition_evaluation_seconds"].mean()),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "artifacts": {
            "figure": figure.name,
            "condition_table": "smoke_condition_table.csv",
            "unit_mapping": "smoke_rr100_unit_mapping.csv",
            "unit_summary": "native_smoke_condition_unit_summary.csv",
            "selected_units": "selected_units.csv",
            "unused_behavior_audit": "unused_behavior_audit.csv",
            "session_runtime": "session_runtime_summary.csv",
            "response_traces": traces_path.name,
        },
    }
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
