#!/usr/bin/env python3
"""Production corrected BackImage SSI bank for recorded-validated SF halves.

Scores the complete preselected 16-image x 32-trace crossed bank using the
checkpoint-19 renderer contract.  The T+1 native helper response is aligned by
dropping its first frame.  Population curves use paired spatial-information
differences, with crossed image/unit bootstrap intervals; path-length slope
inference additionally resamples traces.
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
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.fig4_active_sensing.audit_rr100_eye_trace_conditioning_and_nyquist_power import load_dset
from declan.fig4_active_sensing.run_rr100_corrected_ssi_map_first_smoke import (
    AUDIT19,
    COLORS,
    DT,
    HALF_ASSIGNMENTS,
    MAPPING,
    SOURCE_RUN,
    corrected_patch,
    corrected_trace,
    file_identity,
)
from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import RR100_MOVIE_MEDOID_VERSION
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
)
from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_ssi_validated_halves_fullbank_checkpoint_21_v1"
EPS = 1e-10
GROUPS = ("sf_low_half", "sf_high_half")
GROUP_LABELS = {"sf_low_half": "low-SF half", "sf_high_half": "high-SF half"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--frame-batch-size", type=int, default=16)
    p.add_argument("--trace-batch-size", type=int, default=4)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def score_traces(
    scorer: CanonicalTwinScorer,
    view: object,
    patch: np.ndarray,
    traces: list[np.ndarray],
    *,
    ppd: float,
    frame_batch_size: int,
    trace_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return information numerator, expected spikes, and mean rate per trace/unit."""
    image = _standardize_uint_like(patch)
    n_traces = len(traces)
    n_units = int(view.n_units)
    numer = np.zeros((n_traces, n_units), dtype=np.float64)
    expected = np.zeros((n_traces, n_units), dtype=np.float64)
    rate_sum = np.zeros((n_traces, n_units), dtype=np.float64)
    frame_count = np.zeros(n_traces, dtype=np.int64)
    device = next(scorer.ctx.model.model.parameters()).device
    scorer.ctx.model.model.eval()
    scorer.ctx.readout.eval()
    with scorer.torch.no_grad():
        for start in range(0, n_traces, int(trace_batch_size)):
            chunk = traces[start : start + int(trace_batch_size)]
            stims = []
            ids = []
            for local, trace in enumerate(chunk):
                arr = np.asarray(trace, dtype=np.float32)
                stack = np.broadcast_to(
                    image[None], (arr.shape[0] + int(scorer.common.N_LAGS) + 1, *image.shape)
                ).copy()
                eye = scorer.torch.from_numpy(_trace_xy_to_twin_helper_order(-arr))
                stim = scorer.common.make_counterfactual_stim(
                    stack, eye, ppd=float(ppd), scale_factor=1.0,
                    n_lags=int(scorer.common.N_LAGS), out_size=scorer.common.OUT_SIZE,
                )
                requested = int(arr.shape[0])
                if int(stim.shape[0]) == requested + 1:
                    stim = stim[1 : 1 + requested]
                elif int(stim.shape[0]) != requested:
                    raise ValueError(f"Expected T or T+1 frames; got {stim.shape[0]} for T={requested}")
                stims.append((stim - 127.0) / 255.0)
                ids.extend([start + local] * requested)
            stim_all = scorer.torch.cat(stims, dim=0)
            ids_arr = np.asarray(ids, dtype=np.int64)
            for frame_start in range(0, int(stim_all.shape[0]), int(frame_batch_size)):
                frame_stop = min(frame_start + int(frame_batch_size), int(stim_all.shape[0]))
                x = stim_all[frame_start:frame_stop].to(device)
                full = scorer.compute_rate_map(scorer.ctx.model, scorer.ctx.readout, x)
                rr100 = apply_population_view(full, view).clamp_min(0.0).to(scorer.torch.float64)
                flat = rr100.reshape(rr100.shape[0], rr100.shape[1], -1)
                rbar = flat.mean(dim=2)
                gain = flat / (rbar[..., None] + EPS)
                bits = (gain * scorer.torch.log2(gain + EPS)).mean(dim=2)
                r_np = rbar.detach().cpu().numpy()
                b_np = bits.detach().cpu().numpy()
                frame_ids = ids_arr[frame_start:frame_stop]
                for trace_id in np.unique(frame_ids):
                    take = frame_ids == trace_id
                    exp = r_np[take] * DT
                    expected[trace_id] += exp.sum(axis=0)
                    numer[trace_id] += (b_np[take] * exp).sum(axis=0)
                    rate_sum[trace_id] += r_np[take].sum(axis=0)
                    frame_count[trace_id] += int(take.sum())
                del x, full, rr100, flat, rbar, gain, bits
            del stim_all, stims
            if scorer.torch.cuda.is_available():
                scorer.torch.cuda.empty_cache()
    if not np.all(frame_count == 32):
        raise ValueError(f"Aligned frame-count mismatch: {frame_count.tolist()}")
    return numer.astype(np.float32), expected.astype(np.float32), (rate_sum / frame_count[:, None]).astype(np.float32)


def aggregate_curve(
    moving_numer: np.ndarray, moving_expected: np.ndarray,
    base_numer: np.ndarray, base_expected: np.ndarray,
    image_indices: np.ndarray, unit_indices: np.ndarray,
) -> tuple[np.ndarray, float]:
    mn = moving_numer[image_indices][:, :, unit_indices].sum(axis=(0, 2))
    me = moving_expected[image_indices][:, :, unit_indices].sum(axis=(0, 2))
    bn = base_numer[image_indices][:, unit_indices].sum()
    be = base_expected[image_indices][:, unit_indices].sum()
    baseline = float(bn / max(be, EPS))
    return mn / np.maximum(me, EPS) - baseline, baseline


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) <= 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def percentile_summary(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.percentile(samples, q, axis=0) for q in (2.5, 50.0, 97.5))


def bootstrap_population(
    moving_numer: np.ndarray, moving_expected: np.ndarray,
    base_numer: np.ndarray, base_expected: np.ndarray,
    paths: np.ndarray, group_units: dict[str, np.ndarray],
    *, n_bootstrap: int, seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_images, n_traces, _ = moving_numer.shape
    boot_curves = {g: np.empty((n_bootstrap, n_traces), dtype=np.float32) for g in GROUPS}
    boot_slopes = {g: np.empty(n_bootstrap, dtype=np.float32) for g in GROUPS}
    contrast = np.empty(n_bootstrap, dtype=np.float32)
    for b in range(n_bootstrap):
        ii = rng.integers(0, n_images, size=n_images)
        tt = rng.integers(0, n_traces, size=n_traces)
        for group in GROUPS:
            units = group_units[group]
            uu = rng.choice(units, size=len(units), replace=True)
            curve, _ = aggregate_curve(moving_numer, moving_expected, base_numer, base_expected, ii, uu)
            boot_curves[group][b] = curve
            boot_slopes[group][b] = slope(paths[tt], curve[tt])
        contrast[b] = boot_slopes["sf_low_half"][b] - boot_slopes["sf_high_half"][b]
    rows = []
    for group in GROUPS:
        lo, med, hi = percentile_summary(boot_slopes[group])
        rows.append({
            "quantity": "path_slope", "sf_half": group,
            "estimate_bits_per_spike_per_arcmin": np.nan,
            "bootstrap_median": float(med), "ci_low": float(lo), "ci_high": float(hi),
            "bootstrap_probability_gt_zero": float(np.mean(boot_slopes[group] > 0)),
            "bootstrap_resampling": "images+within-half units+traces",
        })
    lo, med, hi = percentile_summary(contrast)
    rows.append({
        "quantity": "low_minus_high_path_slope", "sf_half": "low_minus_high",
        "estimate_bits_per_spike_per_arcmin": np.nan,
        "bootstrap_median": float(med), "ci_low": float(lo), "ci_high": float(hi),
        "bootstrap_probability_gt_zero": float(np.mean(contrast > 0)),
        "bootstrap_resampling": "paired images+traces; units within each half",
    })
    return boot_curves, pd.DataFrame(rows)


def plot_summary(
    paths: np.ndarray, point_curves: dict[str, np.ndarray], boot_curves: dict[str, np.ndarray],
    image_slopes: pd.DataFrame, images: pd.DataFrame, trend: pd.DataFrame, out_png: Path, out_pdf: Path,
) -> None:
    order = np.argsort(paths)
    x = paths[order]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    for group in GROUPS:
        lo, _, hi = percentile_summary(boot_curves[group])
        ax.fill_between(x, lo[order], hi[order], color=COLORS[group], alpha=0.18, linewidth=0)
        ax.plot(x, point_curves[group][order], color=COLORS[group], lw=2, label=GROUP_LABELS[group])
    ax.axhline(0, color="0.35", ls=":", lw=1)
    ax.set(title="A  Full corrected bank", xlabel="corrected path length (arcmin)", ylabel="moving − stabilized SSI (bits/spike)")
    ax.legend(frameon=False); ax.grid(alpha=0.16)

    ax = axes[0, 1]
    contrast = point_curves["sf_low_half"] - point_curves["sf_high_half"]
    boot_contrast = boot_curves["sf_low_half"] - boot_curves["sf_high_half"]
    lo, _, hi = percentile_summary(boot_contrast)
    ax.fill_between(x, lo[order], hi[order], color="#6A3D9A", alpha=0.2, linewidth=0)
    ax.plot(x, contrast[order], color="#6A3D9A", lw=2)
    ax.axhline(0, color="0.35", ls=":", lw=1)
    ax.set(title="B  Low-SF minus high-SF enhancement", xlabel="corrected path length (arcmin)", ylabel="difference of SSI differences")
    ax.grid(alpha=0.16)

    ax = axes[1, 0]
    rng = np.random.default_rng(8)
    for pos, group in enumerate(GROUPS):
        vals = image_slopes.loc[image_slopes.sf_half.eq(group), "path_slope"].to_numpy()
        ax.scatter(pos + rng.normal(0, 0.035, len(vals)), vals, color=COLORS[group], alpha=0.7, s=24)
        ax.plot([pos - 0.18, pos + 0.18], [np.median(vals)] * 2, color="black", lw=2)
    ax.axhline(0, color="0.35", ls=":", lw=1)
    ax.set_xticks([0, 1], ["low-SF half", "high-SF half"])
    ax.set(title="C  Image-wise path slopes", ylabel="SSI slope (bits/spike/arcmin)")
    ax.grid(axis="y", alpha=0.16)

    ax = axes[1, 1]
    merged = image_slopes.merge(images[["image_index", "image_orientation_coherence", "image_contour_strong"]], on="image_index")
    for group in GROUPS:
        sub = merged[merged.sf_half.eq(group)]
        ax.scatter(sub.image_orientation_coherence, sub.path_slope, color=COLORS[group], s=32, alpha=0.8, label=GROUP_LABELS[group])
    strong = merged[merged.image_contour_strong.astype(bool)]
    ax.scatter(strong.image_orientation_coherence, strong.path_slope, facecolors="none", edgecolors="black", s=76, lw=1.2, label="strong-contour image")
    ax.axhline(0, color="0.35", ls=":", lw=1)
    ax.set(title="D  Is the slope contour-dependent?", xlabel="image orientation coherence", ylabel="image-wise SSI slope")
    ax.grid(alpha=0.16)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
    fig.suptitle(
        "Corrected BackImage spatial SSI — recorded-validated SF halves\n"
        "16 images × 32 corrected traces; crossed bootstrap intervals",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if (args.out_dir / "manifest.json").exists():
        raise FileExistsError(f"Completed checkpoint already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((SOURCE_RUN / "summary.json").read_text())
    source_rows = load_source_rows(Path(summary["source_csv"]))
    images = pd.read_csv(SOURCE_RUN / "image_feature_table.csv").sort_values("image_index")
    old_traces = pd.read_csv(SOURCE_RUN / "trace_feature_table.csv").sort_values("trace_index")
    corrected_metrics = pd.read_csv(AUDIT19 / "trace_conditioning_metrics.csv")
    corrected_metrics = corrected_metrics[corrected_metrics.trace_contract.eq("visual_even_decimated_corrected_crop")].sort_values("trace_index")
    assignments = pd.read_csv(HALF_ASSIGNMENTS)
    valid = assignments[assignments.sf_outer_third.isin(GROUPS)].copy()
    group_units = {g: valid.loc[valid.sf_outer_third.eq(g), "rr100_index"].to_numpy(int) for g in GROUPS}
    if {g: len(u) for g, u in group_units.items()} != {"sf_low_half": 31, "sf_high_half": 30}:
        raise ValueError("Validated-half membership changed")

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapping = pd.read_csv(MAPPING).sort_values("rr100_index")
    if not np.array_equal(np.argmax(view.membership, axis=1), mapping.canonical_channel.to_numpy(int)):
        raise ValueError("RR100 mapping mismatch")
    dset_cache, canvas_cache = {}, {}
    image_records = []
    for _, row in images.iterrows():
        source = source_row_by_id(source_rows, int(row.source_row))
        dset = load_dset(str(source.session), dset_cache)
        patch, meta, indices = corrected_patch(source, dset, canvas_cache)
        image_records.append({"image_index": int(row.image_index), "patch": patch, "ppd": float(meta["patch_ppd"]), "indices": indices})
    trace_records = []
    for _, row in old_traces.iterrows():
        source = source_row_by_id(source_rows, int(row.trace_source_row))
        dset = load_dset(str(source.session), dset_cache)
        trace, indices = corrected_trace(source, dset)
        metric = corrected_metrics.loc[corrected_metrics.trace_index.eq(int(row.trace_index))].iloc[0]
        trace_records.append({"trace_index": int(row.trace_index), "trace": trace, "indices": indices, "path": float(metric.path_length_arcmin)})
    paths = np.asarray([r["path"] for r in trace_records], dtype=np.float64)
    scorer = CanonicalTwinScorer(device=args.device, batch_size=args.frame_batch_size, empty_cache_every_batch=True)
    I, T, U = len(image_records), len(trace_records), 100
    moving_numer = np.zeros((I, T, U), dtype=np.float32)
    moving_expected = np.zeros_like(moving_numer)
    moving_rate = np.zeros_like(moving_numer)
    base_numer = np.zeros((I, U), dtype=np.float32)
    base_expected = np.zeros_like(base_numer)
    base_rate = np.zeros_like(base_numer)
    moving_traces = [r["trace"] for r in trace_records]
    for i, image in enumerate(image_records):
        print(f"image {i + 1:02d}/{I:02d}: stabilized", flush=True)
        n, e, r = score_traces(
            scorer, view, image["patch"], [np.zeros((32, 2), dtype=np.float32)], ppd=image["ppd"],
            frame_batch_size=args.frame_batch_size, trace_batch_size=1,
        )
        base_numer[i], base_expected[i], base_rate[i] = n[0], e[0], r[0]
        print(f"image {i + 1:02d}/{I:02d}: 32 corrected traces", flush=True)
        n, e, r = score_traces(
            scorer, view, image["patch"], moving_traces, ppd=image["ppd"],
            frame_batch_size=args.frame_batch_size, trace_batch_size=args.trace_batch_size,
        )
        moving_numer[i], moving_expected[i], moving_rate[i] = n, e, r
        np.savez_compressed(
            args.out_dir / "corrected_fullbank_accumulator_partial.npz",
            completed_images=np.asarray(i + 1), moving_numer=moving_numer, moving_expected=moving_expected,
            moving_mean_rate_hz=moving_rate, stabilized_numer=base_numer,
            stabilized_expected=base_expected, stabilized_mean_rate_hz=base_rate, path_length_arcmin=paths,
        )

    cache_path = args.out_dir / "corrected_fullbank_unit_sufficient_statistics.npz"
    np.savez_compressed(
        cache_path, moving_information_numerator=moving_numer, moving_expected_spikes=moving_expected,
        moving_mean_rate_hz=moving_rate, stabilized_information_numerator=base_numer,
        stabilized_expected_spikes=base_expected, stabilized_mean_rate_hz=base_rate,
        image_indices=images.image_index.to_numpy(int), trace_indices=old_traces.trace_index.to_numpy(int),
        rr100_indices=np.arange(U), path_length_arcmin=paths,
    )

    unit_rows = []
    for i in range(I):
        for t in range(T):
            for u in range(U):
                unit_rows.append({
                    "image_index": i, "trace_index": t, "rr100_index": u,
                    "corrected_path_length_arcmin": paths[t],
                    "moving_ssi_bits_per_spike": moving_numer[i, t, u] / max(moving_expected[i, t, u], EPS),
                    "stabilized_ssi_bits_per_spike": base_numer[i, u] / max(base_expected[i, u], EPS),
                    "ssi_delta_bits_per_spike": moving_numer[i, t, u] / max(moving_expected[i, t, u], EPS) - base_numer[i, u] / max(base_expected[i, u], EPS),
                    "moving_expected_spikes": moving_expected[i, t, u], "stabilized_expected_spikes": base_expected[i, u],
                    "moving_mean_rate_hz": moving_rate[i, t, u], "stabilized_mean_rate_hz": base_rate[i, u],
                })
    unit_table = pd.DataFrame(unit_rows).merge(
        assignments[["rr100_index", "sf_outer_third", "preferred_sf_cpd", "recorded_sf_curve_r_full_support"]],
        on="rr100_index", how="left", validate="many_to_one",
    )
    unit_table.to_csv(args.out_dir / "corrected_fullbank_unit_ssi.csv", index=False)

    point_curves, curve_rows = {}, []
    all_images = np.arange(I)
    for group in GROUPS:
        curve, baseline = aggregate_curve(moving_numer, moving_expected, base_numer, base_expected, all_images, group_units[group])
        point_curves[group] = curve
        for t in range(T):
            curve_rows.append({"trace_index": t, "corrected_path_length_arcmin": paths[t], "sf_half": group, "delta_ssi_bits_per_spike": curve[t], "stabilized_ssi_bits_per_spike": baseline})
    boot_curves, trend = bootstrap_population(
        moving_numer, moving_expected, base_numer, base_expected, paths, group_units,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    curve_table = pd.DataFrame(curve_rows)
    for group in GROUPS:
        lo, med, hi = percentile_summary(boot_curves[group])
        mask = curve_table.sf_half.eq(group)
        curve_table.loc[mask, "bootstrap_median"] = med
        curve_table.loc[mask, "ci_low"] = lo
        curve_table.loc[mask, "ci_high"] = hi
        estimate = slope(paths, point_curves[group])
        trend.loc[(trend.quantity.eq("path_slope")) & trend.sf_half.eq(group), "estimate_bits_per_spike_per_arcmin"] = estimate
    contrast_estimate = slope(paths, point_curves["sf_low_half"]) - slope(paths, point_curves["sf_high_half"])
    trend.loc[trend.quantity.eq("low_minus_high_path_slope"), "estimate_bits_per_spike_per_arcmin"] = contrast_estimate
    curve_table.to_csv(args.out_dir / "corrected_fullbank_validated_half_curves.csv", index=False)
    trend.to_csv(args.out_dir / "corrected_fullbank_path_slope_bootstrap.csv", index=False)

    image_slope_rows = []
    for i in range(I):
        for group in GROUPS:
            curve, _ = aggregate_curve(moving_numer, moving_expected, base_numer, base_expected, np.asarray([i]), group_units[group])
            image_slope_rows.append({"image_index": i, "sf_half": group, "path_slope": slope(paths, curve)})
    image_slopes = pd.DataFrame(image_slope_rows)
    image_slopes.to_csv(args.out_dir / "corrected_fullbank_imagewise_path_slopes.csv", index=False)
    plot_summary(
        paths, point_curves, boot_curves, image_slopes, images, trend,
        args.out_dir / "checkpoint_21_corrected_validated_halves_fullbank.png",
        args.out_dir / "checkpoint_21_corrected_validated_halves_fullbank.pdf",
    )
    validation = pd.read_csv(AUDIT19 / "renderer_vs_exact_model_input_validation.csv")
    validation.to_csv(args.out_dir / "exact_pair_renderer_validation_all_selected_images.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "corrected_fullbank_population_checkpoint_complete",
        "scope": f"{I} images x {T} corrected traces x {U} RR100 units; 61 recorded-validated units summarized",
        "visual_contract": "dpi_pix; global-even 240->120 Hz; RF offset; mean-centered corrected trace; retinal sign",
        "response_alignment": "native helper T+1: drop first response, retain exactly T=32",
        "ssi_contract": "instantaneous spatial-map SSI weighted by expected spikes; paired moving-minus-stabilized difference",
        "baseline": "zero retinal-translation trace at the same corrected RF-centered image patch",
        "bootstrap": {"n": args.n_bootstrap, "seed": args.seed, "curve": "crossed images+within-half units", "slope": "crossed images+within-half units+traces"},
        "validated_half_counts": {g: len(group_units[g]) for g in GROUPS},
        "array_sha256": {"moving_numer": sha256_array(moving_numer), "moving_expected": sha256_array(moving_expected), "base_numer": sha256_array(base_numer), "base_expected": sha256_array(base_expected)},
        "sources": {"source_summary": file_identity(SOURCE_RUN / "summary.json"), "audit19": file_identity(AUDIT19 / "audit_summary.json"), "assignments": file_identity(HALF_ASSIGNMENTS), "mapping": file_identity(MAPPING)},
        "outputs": {"cache": str(cache_path.resolve()), "figure_png": str((args.out_dir / "checkpoint_21_corrected_validated_halves_fullbank.png").resolve())},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(trend.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
