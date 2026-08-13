#!/usr/bin/env python3
"""Corrected BackImage SSI map-first smoke test for validated SF halves.

This targeted checkpoint scores two strong-contour images crossed with four
algorithmically selected corrected traces.  It uses the checkpoint-19 visual
contract: shifter-corrected ``dpi_pix``, global-even 240->120 Hz sampling,
session RF-crop offset, and retinal-translation sign.  Exact saved model frames
remain the renderer validation reference; crossed movies use the validated
large-field counterfactual renderer required for spatial response maps.

No microsaccade labels or population-level inferential claims are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import (
    corrected_crop_xy_deg,
    load_dset,
    model_aligned_indices,
)
from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
    CanonicalTwinScorer,
    _align_response_to_trace,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / (
    "outputs/active_sensing_movie_information/temporal_remapping/"
    "backimage_rr100_retiming_medium_figure4pool_n16_t32_fullgrid_cuda0_v1"
)
AUDIT19 = ROOT / "outputs/fig4_active_sensing/rr100_eye_trace_conditioning_nyquist_audit_checkpoint_19_v1"
HALF_ASSIGNMENTS = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_halves_recorded_validated_r0p5_v1/"
    "sf_half_recorded_validated_unit_assignments.csv"
)
MAPPING = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1/rr100_unit_mapping.csv"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_ssi_map_first_smoke_checkpoint_20_v2"
FRAME_RATE_HZ = 120.0
DT = 1.0 / FRAME_RATE_HZ
IMAGE_INDICES = (3, 6)  # the two strong-contour images in the existing 16-image pool
TRACE_INDICES = (1, 13, 30, 31)  # min, median-near, upper-tail, maximum corrected path
COLORS = {"sf_low_half": "#0072B2", "sf_high_half": "#D55E00"}
EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=210)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "sha256": digest.hexdigest()}


def corrected_patch(source: pd.Series, dset, canvas_cache: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    target_indices = model_aligned_indices(int(source["global_start"]), int(source["global_stop"]))
    crop_target = corrected_crop_xy_deg(dset)[target_indices]
    roi_src = np.asarray(dset.metadata["roi_src"], dtype=float)
    roi_center_yx = (roi_src[:, 0] + roi_src[:, 1] - 1.0) / 2.0
    roi_offset_xy_deg = np.asarray([roi_center_yx[1], -roi_center_yx[0]], dtype=float) / float(dset.metadata["ppd"])
    center = crop_target.mean(axis=0) + roi_offset_xy_deg
    adjusted = source.copy()
    adjusted["mean_x_deg"] = float(center[0])
    adjusted["mean_y_deg"] = float(center[1])
    patch, meta = _extract_patch(adjusted, canvas_cache=canvas_cache, patch_size_px=540)
    return np.asarray(patch, dtype=np.float32), meta, target_indices


def corrected_trace(source: pd.Series, dset) -> tuple[np.ndarray, np.ndarray]:
    indices = model_aligned_indices(int(source["global_start"]), int(source["global_stop"]))
    trace = corrected_crop_xy_deg(dset)[indices]
    trace = trace - trace.mean(axis=0, keepdims=True)
    return trace.astype(np.float32), indices


def build_native_stim(scorer: CanonicalTwinScorer, patch: np.ndarray, trace: np.ndarray, ppd: float):
    image = _standardize_uint_like(patch)
    full_stack = np.broadcast_to(
        image[None, :, :], (trace.shape[0] + int(scorer.common.N_LAGS) + 1, *image.shape)
    ).copy()
    eye = scorer.torch.from_numpy(_trace_xy_to_twin_helper_order(-np.asarray(trace, dtype=np.float32)))
    stim = scorer.common.make_counterfactual_stim(
        full_stack,
        eye,
        ppd=float(ppd),
        scale_factor=1.0,
        n_lags=int(scorer.common.N_LAGS),
        out_size=scorer.common.OUT_SIZE,
    )
    return (stim - 127.0) / 255.0


def score_condition(scorer, view, patch: np.ndarray, trace: np.ndarray, ppd: float) -> np.ndarray:
    stim = build_native_stim(scorer, patch, trace, ppd)
    full = scorer._compute_rate_map_batched(stim)
    rr100 = apply_population_view(full, view).clamp_min(0.0)
    result = rr100.detach().cpu().numpy().astype(np.float32, copy=False)
    result = _align_response_to_trace(result, n_timepoints=int(trace.shape[0]))
    del stim, full, rr100
    if scorer.torch.cuda.is_available():
        scorer.torch.cuda.empty_cache()
    return result


def ssi_metrics(rate_maps: np.ndarray) -> dict[str, np.ndarray]:
    maps = np.asarray(rate_maps, dtype=np.float64)
    flat = maps.reshape(maps.shape[0], maps.shape[1], -1)
    mean_rate = flat.mean(axis=2)
    gain = flat / np.maximum(mean_rate[..., None], EPS)
    instantaneous = np.mean(gain * np.log2(np.maximum(gain, EPS)), axis=2)
    expected = mean_rate * DT
    numerator = instantaneous * expected
    movie_ssi = numerator.sum(axis=0) / np.maximum(expected.sum(axis=0), EPS)
    return {
        "instantaneous_ssi": instantaneous,
        "mean_rate": mean_rate,
        "expected_spikes": expected.sum(axis=0),
        "information_numerator": numerator.sum(axis=0),
        "movie_ssi": movie_ssi,
    }


def select_unit_roles(assignments: pd.DataFrame) -> pd.DataFrame:
    valid = assignments[assignments["sf_outer_third"].isin(COLORS)].copy()
    roles = []
    for group, label in (("sf_low_half", "low_sf_representative"), ("sf_high_half", "high_sf_representative")):
        sub = valid[valid["sf_outer_third"].eq(group)].sort_values("preferred_sf_cpd")
        median = float(sub["preferred_sf_cpd"].median())
        representative = sub.loc[(sub["preferred_sf_cpd"] - median).abs().idxmin()]
        extreme = sub.iloc[0] if group == "sf_low_half" else sub.iloc[-1]
        for role, row in ((label, representative), (f"{group}_extreme_control", extreme)):
            record = row.to_dict()
            record["selection_role"] = role
            record["selection_criterion"] = (
                "closest to within-half median preferred SF" if "representative" in role
                else "outermost preferred SF within recorded-validated half"
            )
            record["selection_is_algorithmic"] = True
            roles.append(record)
    return pd.DataFrame(roles).drop_duplicates("rr100_index").reset_index(drop=True)


def plot_input_checkpoint(
    image_records: list[dict], trace_records: list[dict], out_path: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), constrained_layout=True)
    for col, rec in enumerate(trace_records):
        trace = rec["trace"] * 60.0
        axes[0, col].plot(trace[:, 0], trace[:, 1], color="#333333", lw=1.4)
        axes[0, col].scatter(trace[0, 0], trace[0, 1], color="#009E73", s=24, zorder=3)
        axes[0, col].scatter(trace[-1, 0], trace[-1, 1], color="#D55E00", marker="X", s=28, zorder=3)
        axes[0, col].set_aspect("equal", adjustable="datalim")
        axes[0, col].set_title(
            f"{rec['role']} · trace {rec['trace_index']}\n"
            f"corrected path {rec['path_length_arcmin']:.1f} arcmin"
        )
        axes[0, col].set_xlabel("horizontal (arcmin)")
        if col == 0:
            axes[0, col].set_ylabel("vertical (arcmin)")
    for col in range(4):
        rec = image_records[col % len(image_records)]
        patch = rec["patch"]
        center = np.asarray(patch.shape) // 2
        half = 70
        view = patch[center[0]-half:center[0]+half, center[1]-half:center[1]+half]
        axes[1, col].imshow(view, cmap="gray", vmin=np.percentile(view, 1), vmax=np.percentile(view, 99))
        axes[1, col].set_title(
            f"image {rec['image_index']} · source {rec['source_row']}\n"
            f"RF-centered patch · coherence {rec['coherence']:.2f}",
            fontsize=10,
        )
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
    fig.suptitle(
        "Checkpoint 20 inputs — corrected visual-crop contract\n"
        "global-even 240→120 Hz · dpi_pix · RF offset · retinal sign; no legacy microsaccade labels",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_map_checkpoint(
    results: dict[tuple[int, int], np.ndarray], baselines: dict[int, np.ndarray],
    roles: pd.DataFrame, image_records: list[dict], trace_records: list[dict], out_path: Path, dpi: int,
) -> None:
    units = roles["rr100_index"].astype(int).tolist()
    image_index = int(image_records[0]["image_index"])
    trace_choices = [int(trace_records[0]["trace_index"]), int(trace_records[-1]["trace_index"])]
    fig, axes = plt.subplots(len(units), 4, figsize=(11.8, 2.45 * len(units)), constrained_layout=True)
    for row, unit in enumerate(units):
        role = roles.iloc[row]
        panels = []
        for trace_index in trace_choices:
            moving = results[(image_index, trace_index)][:, unit]
            static = baselines[image_index][:, unit]
            delta = moving - static
            peak_index = int(np.argmax(np.mean(np.abs(delta), axis=(1, 2))))
            for time_label, time_index in (("first aligned", 0), ("peak |map Δ|", peak_index)):
                panels.append((trace_index, time_label, time_index, delta[time_index]))
        values = np.concatenate([panel[3].ravel() for panel in panels])
        limit = max(float(np.percentile(np.abs(values), 99)), 1e-5)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = None
        for col, (trace_index, time_label, time_index, delta) in enumerate(panels):
            image = axes[row, col].imshow(delta, cmap="RdBu_r", norm=norm, interpolation="nearest")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            axes[row, col].set_title(
                f"trace {trace_index}; {time_label} (frame {time_index})\n"
                f"map Δ mean {np.mean(delta):+.3f} Hz", fontsize=8
            )
        axes[row, 0].set_ylabel(
            f"u{unit:03d}\n{role.sf_outer_third.replace('sf_', '').replace('_', ' ')}\n"
            f"pref {role.preferred_sf_cpd:.2f} cpd", fontsize=8
        )
        fig.colorbar(image, ax=axes[row], shrink=0.70, pad=0.006, label="moving − stabilized rate (Hz)")
    fig.suptitle(
        "Corrected crossed movies: concrete RR100 response-map differences\n"
        "one strong-contour image; minimum and maximum corrected-path traces",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    # Preserve completed checkpoints, while allowing a failed partial smoke run
    # to be resumed in this explicitly versioned directory during debugging.
    if (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed checkpoint already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((SOURCE_RUN / "summary.json").read_text())
    source_rows = load_source_rows(Path(summary["source_csv"]))
    images = pd.read_csv(SOURCE_RUN / "image_feature_table.csv")
    traces_old = pd.read_csv(SOURCE_RUN / "trace_feature_table.csv")
    trace_metrics = pd.read_csv(AUDIT19 / "trace_conditioning_metrics.csv")
    trace_metrics = trace_metrics[trace_metrics["trace_contract"].eq("visual_even_decimated_corrected_crop")]
    assignments = pd.read_csv(HALF_ASSIGNMENTS)
    roles = select_unit_roles(assignments)
    roles.to_csv(args.out_dir / "selected_unit_roles.csv", index=False)

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(MAPPING).sort_values("rr100_index")
    if not np.array_equal(np.argmax(view.membership, axis=1), mapping["canonical_channel"].to_numpy(int)):
        raise ValueError("RR100 mapping mismatch")

    dset_cache = {}
    canvas_cache = {}
    image_records = []
    for image_index in IMAGE_INDICES:
        row = images.loc[images["image_index"].eq(image_index)].iloc[0]
        source = source_row_by_id(source_rows, int(row["source_row"]))
        dset = load_dset(str(source["session"]), dset_cache)
        patch, meta, indices = corrected_patch(source, dset, canvas_cache)
        image_records.append({
            "image_index": int(image_index), "source_row": int(row["source_row"]),
            "session": str(source["session"]), "patch": patch, "patch_ppd": float(meta["patch_ppd"]),
            "coherence": float(row["image_orientation_coherence"]), "target_indices": indices,
        })

    trace_records = []
    sorted_paths = trace_metrics.sort_values("path_length_arcmin")
    role_names = {TRACE_INDICES[0]: "minimum", TRACE_INDICES[1]: "median-near", TRACE_INDICES[2]: "upper-tail", TRACE_INDICES[3]: "maximum"}
    for trace_index in TRACE_INDICES:
        old = traces_old.loc[traces_old["trace_index"].eq(trace_index)].iloc[0]
        source = source_row_by_id(source_rows, int(old["trace_source_row"]))
        dset = load_dset(str(source["session"]), dset_cache)
        trace, indices = corrected_trace(source, dset)
        metric = trace_metrics.loc[trace_metrics["trace_index"].eq(trace_index)].iloc[0]
        trace_records.append({
            "trace_index": int(trace_index), "source_row": int(source["source_row"]),
            "session": str(source["session"]), "trace": trace, "target_indices": indices,
            "path_length_arcmin": float(metric["path_length_arcmin"]), "role": role_names[trace_index],
        })
    pd.DataFrame([{k: v for k, v in row.items() if k not in ("trace", "target_indices")} for row in trace_records]).to_csv(
        args.out_dir / "selected_trace_roles.csv", index=False
    )
    pd.DataFrame([{k: v for k, v in row.items() if k not in ("patch", "target_indices")} for row in image_records]).to_csv(
        args.out_dir / "selected_image_roles.csv", index=False
    )
    plot_input_checkpoint(image_records, trace_records, args.out_dir / "checkpoint_20a_corrected_inputs.png", args.dpi)

    scorer = CanonicalTwinScorer(device=args.device, batch_size=args.batch_size, empty_cache_every_batch=True)
    baselines = {}
    results = {}
    metric_rows = []
    for image in image_records:
        image_index = int(image["image_index"])
        zero = np.zeros((32, 2), dtype=np.float32)
        print(f"scoring image {image_index} stabilized", flush=True)
        baseline = score_condition(scorer, view, image["patch"], zero, image["patch_ppd"])
        baselines[image_index] = baseline
        base_metric = ssi_metrics(baseline)
        for trace in trace_records:
            trace_index = int(trace["trace_index"])
            print(f"scoring image {image_index} x corrected trace {trace_index}", flush=True)
            moving = score_condition(scorer, view, image["patch"], trace["trace"], image["patch_ppd"])
            results[(image_index, trace_index)] = moving
            moving_metric = ssi_metrics(moving)
            for unit in range(100):
                metric_rows.append({
                    "image_index": image_index, "trace_index": trace_index,
                    "corrected_path_length_arcmin": trace["path_length_arcmin"], "unit_index": unit,
                    "moving_ssi_bits_per_spike": float(moving_metric["movie_ssi"][unit]),
                    "stabilized_ssi_bits_per_spike": float(base_metric["movie_ssi"][unit]),
                    "ssi_delta_bits_per_spike": float(moving_metric["movie_ssi"][unit] - base_metric["movie_ssi"][unit]),
                    "moving_expected_spikes": float(moving_metric["expected_spikes"][unit]),
                    "stabilized_expected_spikes": float(base_metric["expected_spikes"][unit]),
                })
    metrics = pd.DataFrame(metric_rows).merge(
        assignments[["rr100_index", "sf_outer_third", "preferred_sf_cpd", "recorded_sf_curve_r_full_support"]],
        left_on="unit_index", right_on="rr100_index", how="left", validate="many_to_one",
    )
    metrics.to_csv(args.out_dir / "corrected_smoke_unit_ssi.csv", index=False)

    pop_rows = []
    for (image_index, trace_index, group), sub in metrics[metrics["sf_outer_third"].isin(COLORS)].groupby(
        ["image_index", "trace_index", "sf_outer_third"], sort=False
    ):
        moving = float(np.sum(sub["moving_ssi_bits_per_spike"] * sub["moving_expected_spikes"]) / np.sum(sub["moving_expected_spikes"]))
        static = float(np.sum(sub["stabilized_ssi_bits_per_spike"] * sub["stabilized_expected_spikes"]) / np.sum(sub["stabilized_expected_spikes"]))
        pop_rows.append({
            "image_index": image_index, "trace_index": trace_index, "sf_half": group,
            "corrected_path_length_arcmin": float(sub["corrected_path_length_arcmin"].iloc[0]),
            "n_units": int(len(sub)), "population_moving_ssi": moving,
            "population_stabilized_ssi": static, "population_delta_bits_per_spike": moving - static,
            "equal_unit_mean_delta": float(sub["ssi_delta_bits_per_spike"].mean()),
        })
    population = pd.DataFrame(pop_rows)
    population.to_csv(args.out_dir / "corrected_smoke_population_ssi.csv", index=False)
    plot_map_checkpoint(
        results, baselines, roles, image_records, trace_records,
        args.out_dir / "checkpoint_20b_corrected_response_difference_maps.png", args.dpi,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), sharey=True, constrained_layout=True)
    for ax, image_index in zip(axes, IMAGE_INDICES):
        for group in COLORS:
            sub = population[(population["image_index"].eq(image_index)) & population["sf_half"].eq(group)].sort_values("corrected_path_length_arcmin")
            ax.plot(sub["corrected_path_length_arcmin"], sub["population_delta_bits_per_spike"], marker="o", lw=2, color=COLORS[group], label=group.replace("sf_", "").replace("_", " "))
        ax.axhline(0, color="0.35", ls=":", lw=1)
        ax.set_title(f"strong-contour image {image_index}")
        ax.set_xlabel("corrected path length (arcmin)")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("moving − stabilized SSI (bits/spike)")
    axes[0].legend(frameon=False)
    fig.suptitle("Targeted corrected SSI smoke test — descriptive only (2 images × 4 traces)", fontweight="bold")
    fig.savefig(args.out_dir / "checkpoint_20c_corrected_validated_half_smoke_curves.png", dpi=args.dpi)
    plt.close(fig)

    np.savez_compressed(
        args.out_dir / "corrected_smoke_selected_response_maps.npz",
        image_indices=np.asarray(IMAGE_INDICES), trace_indices=np.asarray(TRACE_INDICES),
        selected_unit_indices=roles["rr100_index"].to_numpy(int),
        **{f"baseline_image_{key:02d}": value[:, roles["rr100_index"].to_numpy(int)] for key, value in baselines.items()},
        **{f"moving_image_{key[0]:02d}_trace_{key[1]:02d}": value[:, roles["rr100_index"].to_numpy(int)] for key, value in results.items()},
    )
    validation = pd.read_csv(AUDIT19 / "renderer_vs_exact_model_input_validation.csv")
    selected_validation = validation[validation["image_index"].isin(IMAGE_INDICES)].copy()
    selected_validation.to_csv(args.out_dir / "exact_pair_renderer_validation_for_selected_images.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "corrected_map_first_smoke_complete_stop_before_population_scaleup",
        "scope": "2 strong-contour images x 4 corrected traces; 100 RR100 units scored; 61 recorded-validated units summarized",
        "visual_contract": (
            "dpi_pix crop trajectory; global-even raw indices at 120 Hz; session roi_src offset; "
            "mean-centered trace; retinal-translation sign"
        ),
        "response_length_contract": "renderer T+1 output normalized to aligned frames 1:T+1, yielding exactly 32 response frames",
        "exact_pair_role": "checkpoint-19 exact saved stim validates the corrected renderer; crossed SSI uses its large-field equivalent",
        "inference_guardrail": "descriptive smoke test only; no microsaccade labels, bootstrap intervals, or population claim",
        "selected_images": list(IMAGE_INDICES), "selected_traces": list(TRACE_INDICES),
        "validated_half_counts": assignments[assignments["sf_outer_third"].isin(COLORS)]["sf_outer_third"].value_counts().to_dict(),
        "sources": {
            "source_summary": file_identity(SOURCE_RUN / "summary.json"),
            "audit19": file_identity(AUDIT19 / "audit_summary.json"),
            "half_assignments": file_identity(HALF_ASSIGNMENTS),
            "mapping": file_identity(MAPPING),
        },
        "selected_exact_renderer_validation": {
            "median_pixel_r": float(selected_validation["pixel_r"].median()),
            "minimum_pixel_r": float(selected_validation["pixel_r"].min()),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(population.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
