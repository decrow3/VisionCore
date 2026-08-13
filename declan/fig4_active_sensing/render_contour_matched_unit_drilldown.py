#!/usr/bin/env python3
"""Targeted map drill-down for checkpoint-3 contour-matched example units."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.plot_temporal_power_shift_activation_maps import (
    instantaneous_bits_for_maps,
    selected_rr100_maps_for_trace,
    unit_movie_ssi,
)
from declan.active_sensing_movie_information.run_backimage_contour_axis_rr100_spatial_ssi import (
    RR100_MOVIE_MEDOID_VERSION,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.redundancy_resolved_v1_population import load_population_view


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
DEFAULT_CHECKPOINT = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/"
    "checkpoint_03_contour_matched_sf_quartiles"
)
DEFAULT_OUT = DEFAULT_CHECKPOINT / "targeted_unit_drilldown_v1"
EXAMPLES = (
    (54, "largest_negative_q3"),
    (5, "median_change_q3"),
    (46, "median_change_q4_control"),
)
GROUP_COLORS = {"sf_q3": "#2FB47C", "sf_q4": "#BDDF26"}
CONDITION_COLORS = {"stabilized": "#777777", "q01": "#3B4CC0", "q06": "#B40426"}
CONDITION_LABELS = {
    "stabilized": "stabilized",
    "q01": "short-path trace",
    "q06": "long-path trace",
}
FRAME_RATE_HZ = 120.0
BIN_SECONDS = 1.0 / FRAME_RATE_HZ


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--units", type=str, default="", help="Optional comma-separated subset of 54,5,46.")
    parser.add_argument("--summarize-existing", action="store_true", help="Combine completed per-unit subdirectories without loading the model.")
    parser.add_argument("--reuse-map-cache", action="store_true", help="Relabel plots from an existing per-unit activation-map cache without loading the model.")
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path.resolve()), "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest(),
    }


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_trace_bins(movie: pd.DataFrame) -> pd.DataFrame:
    out = movie.copy()
    trace_paths = out.drop_duplicates("trace_index").set_index("trace_index")["rendered_path_length_arcmin"]
    bins = pd.qcut(trace_paths, 6, labels=[f"q{i:02d}" for i in range(1, 7)]).astype(str)
    out["trace_path_length_bin"] = bins.reindex(out["trace_index"]).to_numpy()
    return out


def parse_image_indices(value: Any) -> list[int]:
    return [int(part) for part in str(value).split() if part.strip()]


def select_concrete_examples(
    movie: pd.DataFrame,
    ssi: np.ndarray,
    selection: pd.DataFrame,
    changes: pd.DataFrame,
    examples: tuple[tuple[int, str], ...] = EXAMPLES,
) -> pd.DataFrame:
    rows = []
    selection = selection.set_index("unit_index", drop=False)
    changes = changes.set_index("unit_index", drop=False)
    for unit, role in examples:
        sel = selection.loc[int(unit)]
        change = changes.loc[int(unit)]
        matched_images = parse_image_indices(sel["matched_image_indices"])
        unit_values = np.asarray(ssi[:, int(unit)], dtype=float)
        image_rows = []
        for image_index in matched_images:
            image_mask = movie["image_index"].astype(int).eq(int(image_index)).to_numpy()
            low = unit_values[image_mask & movie["trace_path_length_bin"].eq("q01").to_numpy()]
            high = unit_values[image_mask & movie["trace_path_length_bin"].eq("q06").to_numpy()]
            image_rows.append(
                {
                    "image_index": int(image_index),
                    "q01_mean": float(np.nanmean(low)),
                    "q06_mean": float(np.nanmean(high)),
                    "q06_minus_q01": float(np.nanmean(high) - np.nanmean(low)),
                }
            )
        image_summary = pd.DataFrame(image_rows)
        target_change = float(change["last_minus_first_ssi"])
        image_summary["distance_to_unit_aggregate_change"] = (
            image_summary["q06_minus_q01"] - target_change
        ).abs()
        image_choice = image_summary.sort_values(
            ["distance_to_unit_aggregate_change", "image_index"], kind="mergesort"
        ).iloc[0]
        image_index = int(image_choice["image_index"])
        chosen = {
            "unit_index": int(unit),
            "unit_label": f"u{int(unit):03d}",
            "selection_role": role,
            "sf_quartile": str(sel["sf_group"]),
            "preferred_sf_cpd": float(change["preferred_sf_cpd"]),
            "preferred_tf_hz": float(change["preferred_tf_hz"]),
            "preferred_orientation_deg": float(sel["preferred_orientation_deg"]),
            "orientation_selectivity_index": float(sel["prior_orientation_selectivity_index"]),
            "n_matched_images": int(sel["n_matched_images"]),
            "matched_image_index": image_index,
            "image_selection_rule": "matched image whose q06-minus-q01 mean is closest to the unit aggregate q06-minus-q01 change",
            "unit_aggregate_q06_minus_q01_ssi": target_change,
            "selected_image_q06_minus_q01_ssi": float(image_choice["q06_minus_q01"]),
        }
        for path_bin in ("q01", "q06"):
            mask = (
                movie["image_index"].astype(int).eq(image_index)
                & movie["trace_path_length_bin"].astype(str).eq(path_bin)
            )
            candidates = movie.loc[mask, ["trace_index", "rendered_path_length_arcmin"]].copy()
            candidates["matrix_ssi"] = unit_values[candidates.index.to_numpy(int)]
            target = float(candidates["matrix_ssi"].mean())
            candidates["distance_to_image_bin_mean"] = (candidates["matrix_ssi"] - target).abs()
            trace_choice = candidates.sort_values(
                ["distance_to_image_bin_mean", "trace_index"], kind="mergesort"
            ).iloc[0]
            chosen[f"{path_bin}_trace_index"] = int(trace_choice["trace_index"])
            chosen[f"{path_bin}_trace_path_arcmin"] = float(trace_choice["rendered_path_length_arcmin"])
            chosen[f"{path_bin}_image_bin_mean_ssi"] = target
            chosen[f"{path_bin}_chosen_trace_ssi"] = float(trace_choice["matrix_ssi"])
            chosen[f"{path_bin}_trace_selection_rule"] = "trace whose SSI is closest to the selected image x path-bin mean"
        rows.append(chosen)
    return pd.DataFrame(rows)


def trace_speed(trace: np.ndarray) -> np.ndarray:
    xy = np.asarray(trace, dtype=float)
    step = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[[0]]), axis=1)
    return step * FRAME_RATE_HZ


def render_example(
    row: pd.Series,
    patch: np.ndarray,
    patch_meta: dict[str, Any],
    traces: dict[str, np.ndarray],
    maps: dict[str, np.ndarray],
    metrics: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
    dpi: int,
) -> None:
    frames = [0, 10, 20, 30, 39]
    unit = int(row["unit_index"])
    sf_group = str(row["sf_quartile"])
    group_color = GROUP_COLORS[sf_group]
    patch_arr = np.asarray(patch, dtype=float)
    fig = plt.figure(figsize=(13.2, 9.5))
    gs = fig.add_gridspec(5, 7, width_ratios=[1.45, 1, 1, 1, 1, 1, 0.08], hspace=0.18, wspace=0.08)
    ax_patch = fig.add_subplot(gs[0, 0])
    lo, hi = np.nanpercentile(patch_arr, [1, 99])
    ax_patch.imshow(patch_arr, cmap="gray", vmin=lo, vmax=hi)
    ppd = float(patch_meta["patch_ppd"])
    for condition, linestyle in (("q01", "-"), ("q06", "--")):
        xy = traces[condition] - np.mean(traces[condition], axis=0, keepdims=True)
        px = patch_arr.shape[1] / 2 + xy[:, 0] * ppd
        py = patch_arr.shape[0] / 2 - xy[:, 1] * ppd
        ax_patch.plot(
            px, py, color=CONDITION_COLORS[condition], ls=linestyle, lw=1.5,
            label=CONDITION_LABELS[condition],
        )
        ax_patch.scatter(px[0], py[0], s=18, color=CONDITION_COLORS[condition])
    ax_patch.set_title(f"matched image {int(row['matched_image_index'])} + paths", fontsize=9)
    ax_patch.set_xticks([]); ax_patch.set_yticks([])
    ax_patch.legend(frameon=False, fontsize=7)

    ax_speed = fig.add_subplot(gs[0, 1:6])
    t_ms = np.arange(40) * 1000.0 / FRAME_RATE_HZ
    for condition in ("q01", "q06"):
        speed = trace_speed(traces[condition])
        ax_speed.plot(
            t_ms, speed, color=CONDITION_COLORS[condition], lw=1.6,
            label=f"{CONDITION_LABELS[condition]} speed",
        )
    for frame in frames:
        ax_speed.axvline(t_ms[frame], color="0.75", lw=0.7)
    ax_speed.set(xlabel="time (ms)", ylabel="speed (deg/s)")
    ax_speed.set_title(
        f"{row['unit_label']} | {row['selection_role']} | {sf_group.upper()} | "
        f"SF {row['preferred_sf_cpd']:.3g} cpd, TF {row['preferred_tf_hz']:.3g} Hz",
        fontsize=10, color=group_color, fontweight="bold",
    )
    ax_speed.legend(frameon=False, fontsize=7)
    ax_speed.spines[["top", "right"]].set_visible(False)

    all_activation = np.concatenate([maps[key].ravel() for key in ("stabilized", "q01", "q06")])
    act_min, act_max = np.nanpercentile(all_activation, [1, 99.5])
    diff = maps["q06"] - maps["q01"]
    diff_max = max(float(np.nanpercentile(np.abs(diff), 99)), 1e-8)
    map_rows = [
        ("stabilized", "stabilized", maps["stabilized"], "cividis", act_min, act_max),
        ("q01", "short-path trace", maps["q01"], "cividis", act_min, act_max),
        ("q06", "long-path trace", maps["q06"], "cividis", act_min, act_max),
        (None, "long minus short", diff, "PuOr_r", -diff_max, diff_max),
    ]
    for ridx, (metric_key, display_label, stack, cmap, vmin, vmax) in enumerate(map_rows, start=1):
        label_ax = fig.add_subplot(gs[ridx, 0]); label_ax.axis("off")
        if metric_key in metrics:
            m = metrics[metric_key]
            text = (
                f"{display_label}\nSSI {float(m['unit_bits_per_spike'][0]):.4f}\n"
                f"rate {float(m['unit_mean_rate'][0]):.3f}\nspikes {float(m['unit_expected_spikes'][0]):.3f}"
            )
        else:
            text = display_label
        label_ax.text(1, 0.5, text, ha="right", va="center", fontsize=8.2, fontweight="bold" if metric_key is not None else None)
        first = None
        for cidx, frame in enumerate(frames, start=1):
            ax = fig.add_subplot(gs[ridx, cidx])
            first = ax.imshow(stack[frame, 0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ridx == 1:
                ax.set_title(f"frame {frame}", fontsize=8)
        cax = fig.add_subplot(gs[ridx, 6])
        fig.colorbar(first, cax=cax)
        cax.tick_params(labelsize=6)
    fig.suptitle(
        "Targeted visualization render: instantaneous RR100 activation maps\n"
        "shared activation scale within unit; symmetric long-minus-short scale",
        fontsize=12, fontweight="bold",
    )
    save_figure(fig, out_dir, f"{row['unit_label']}_representative_short_long_path_maps", dpi)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True, constrained_layout=True)
    for condition in ("stabilized", "q01", "q06"):
        color = CONDITION_COLORS[condition]
        rate = metrics[condition]["frame_mean_rate"][:, 0]
        bits = metrics[condition]["frame_bits_per_spike"][:, 0]
        axes[0].plot(t_ms, rate, color=color, lw=1.7, label=CONDITION_LABELS[condition])
        axes[1].plot(t_ms, bits, color=color, lw=1.7, label=CONDITION_LABELS[condition])
    axes[0].set_ylabel("mean map rate")
    axes[1].set_ylabel("instantaneous SSI")
    axes[1].set_xlabel("time (ms)")
    for ax in axes:
        ax.grid(True, color="0.9", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{row['unit_label']} map-derived rate and SSI timecourses", color=group_color, fontweight="bold")
    save_figure(fig, out_dir, f"{row['unit_label']}_representative_short_long_path_timecourses", dpi)


def summarize_existing(out_dir: Path, dpi: int) -> None:
    selections = []
    metrics = []
    validations = []
    for unit, _role in EXAMPLES:
        subdir = out_dir / f"u{unit:03d}"
        selections.append(pd.read_csv(subdir / "targeted_example_selection.csv"))
        metrics.append(pd.read_csv(subdir / "targeted_render_metrics.csv"))
        validations.append(pd.read_csv(subdir / "cached_vs_rerendered_ssi_validation.csv"))
    selection = pd.concat(selections, ignore_index=True)
    metric = pd.concat(metrics, ignore_index=True)
    validation = pd.concat(validations, ignore_index=True)
    selection.to_csv(out_dir / "combined_targeted_example_selection.csv", index=False)
    metric.to_csv(out_dir / "combined_targeted_render_metrics.csv", index=False)
    validation.to_csv(out_dir / "combined_cached_vs_rerendered_ssi_validation.csv", index=False)

    conditions = ("stabilized", "q01", "q06")
    condition_labels = CONDITION_LABELS
    metric_specs = [
        ("rendered_ssi_bits_per_spike", "movie SSI (bits/spike)", "A. Spatial selectivity"),
        ("rendered_mean_rate", "mean map rate", "B. Activation rate"),
        ("rendered_expected_spikes", "expected spikes", "C. Spike support"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3))
    unit_order = [54, 5, 46]
    x = np.arange(len(unit_order))
    for ax, (column, ylabel, title) in zip(axes, metric_specs, strict=True):
        for condition in conditions:
            values = []
            for unit in unit_order:
                row = metric[metric["unit_index"].eq(unit) & metric["condition"].eq(condition)].iloc[0]
                values.append(float(row[column]))
            ax.plot(
                x, values, marker="o", ms=5, lw=1.8,
                color=CONDITION_COLORS[condition], label=condition_labels[condition],
            )
        ax.set_xticks(x, ["u054\nQ3 outlier", "u005\nQ3 median", "u046\nQ4 control"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="0.92", lw=0.7)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Contour-matched targeted unit drill-down", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)
    save_figure(fig, out_dir, "targeted_unit_metric_comparison", dpi)

    lookup = metric.set_index(["unit_index", "condition"])
    lines = []
    for unit in unit_order:
        low = lookup.loc[(unit, "q01")]
        high = lookup.loc[(unit, "q06")]
        lines.append(
            f"- u{unit:03d}: SSI {low.rendered_ssi_bits_per_spike:.4f} -> {high.rendered_ssi_bits_per_spike:.4f}; "
            f"rate {low.rendered_mean_rate:.3f} -> {high.rendered_mean_rate:.3f}; "
            f"expected spikes {low.rendered_expected_spikes:.3f} -> {high.rendered_expected_spikes:.3f}."
        )
    readme = f"""# Targeted contour-matched unit drill-down

These are targeted visualization renders on algorithmically selected concrete
movies, not a new population run. Each unit uses a contour-matched image whose
short-to-long-path SSI change is closest to that unit's aggregate change; within
each endpoint bin, the chosen trace is closest to the image-bin mean.

## Exact rerendered movie metrics

{chr(10).join(lines)}

u054 shows sustained lower instantaneous SSI on the long-path trace together
with modest rate suppression. u005 is a dissociation: rate and spike support
increase while movie SSI changes little. u046 is the Q4 control and increases
in rate, spike support, and SSI. These examples establish heterogeneity and do
not by themselves estimate population prevalence.

All nine stabilized/short/long rerenders reproduce their cached SSI values to
within {validation['absolute_difference'].max():.2e} bits/spike.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "combined_targeted_unit_drilldown_complete",
        "render_type": "three targeted visualization renders, not a production population rerun",
        "unit_subdirectories": [f"u{unit:03d}" for unit in unit_order],
        "max_cached_vs_rerendered_ssi_difference": float(validation["absolute_difference"].max()),
        "outputs": {
            "comparison_figure": "targeted_unit_metric_comparison.{png,pdf}",
            "selection": "combined_targeted_example_selection.csv",
            "metrics": "combined_targeted_render_metrics.csv",
            "validation": "combined_cached_vs_rerendered_ssi_validation.csv",
        },
        "not_run": "No page-13 orthogonal summary or broader population rerun was performed.",
    }
    (out_dir / "combined_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(readme)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_existing:
        summarize_existing(Path(args.out_dir), int(args.dpi))
        return
    matrix_dir = Path(args.matrix_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    movie = add_trace_bins(pd.read_csv(matrix_dir / "movie_feature_table.csv"))
    images = pd.read_csv(matrix_dir / "image_feature_table.csv")
    trace_xy = np.load(matrix_dir / "trace_xy.npy", mmap_mode="r")
    ssi = np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r")
    selection = pd.read_csv(checkpoint_dir / "contour_matched_unit_selection.csv")
    changes = pd.read_csv(checkpoint_dir / "contour_matched_unit_changes.csv")
    requested = [int(part.strip()) for part in str(args.units).split(",") if part.strip()]
    if requested:
        role_by_unit = dict(EXAMPLES)
        unknown = sorted(set(requested).difference(role_by_unit))
        if unknown:
            raise ValueError(f"Unsupported requested units: {unknown}; choices are {sorted(role_by_unit)}")
        examples = tuple((unit, role_by_unit[unit]) for unit in requested)
    else:
        examples = EXAMPLES
    concrete = select_concrete_examples(movie, ssi, selection, changes, examples=examples)
    concrete.to_csv(args.out_dir / "targeted_example_selection.csv", index=False)

    population_view = None
    scorer = None
    if not args.reuse_map_cache:
        population_view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
        scorer = CanonicalTwinScorer(
            device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True
        )
    metric_rows = []
    validation_rows = []
    cache_payload: dict[str, np.ndarray] = {}
    for row in concrete.itertuples(index=False):
        image_row = images[images["image_index"].astype(int).eq(int(row.matched_image_index))].iloc[0]
        patch, patch_meta = _extract_patch(image_row, canvas_cache={}, patch_size_px=540)
        cache_path = args.out_dir / "targeted_activation_map_cache.npz"
        cached = np.load(cache_path) if args.reuse_map_cache else None
        traces = {
            condition: (
                np.asarray(cached[f"u{int(row.unit_index):03d}_{condition}_trace"], dtype=np.float32)
                if cached is not None else
                (np.zeros((40, 2), dtype=np.float32) if condition == "stabilized" else
                 np.asarray(trace_xy[int(getattr(row, f"{condition}_trace_index"))], dtype=np.float32))
            )
            for condition in ("stabilized", "q01", "q06")
        }
        maps = {}
        metrics = {}
        for condition in ("stabilized", "q01", "q06"):
            if cached is not None:
                print(f"Reusing cached maps for {row.unit_label} condition={condition}", flush=True)
                maps[condition] = np.asarray(
                    cached[f"u{int(row.unit_index):03d}_{condition}_maps"], dtype=np.float32
                )
            else:
                print(f"Rendering {row.unit_label} image={row.matched_image_index} condition={condition}", flush=True)
                maps[condition] = selected_rr100_maps_for_trace(
                    scorer, population_view, np.asarray(patch), traces[condition], [int(row.unit_index)], n_timepoints=40
                )
            metrics[condition] = unit_movie_ssi(maps[condition], bin_seconds=BIN_SECONDS)
            metric_rows.append(
                {
                    "unit_index": int(row.unit_index), "unit_label": str(row.unit_label),
                    "selection_role": str(row.selection_role), "sf_quartile": str(row.sf_quartile),
                    "image_index": int(row.matched_image_index), "condition": condition,
                    "trace_index": -1 if condition == "stabilized" else int(getattr(row, f"{condition}_trace_index")),
                    "rendered_ssi_bits_per_spike": float(metrics[condition]["unit_bits_per_spike"][0]),
                    "rendered_mean_rate": float(metrics[condition]["unit_mean_rate"][0]),
                    "rendered_expected_spikes": float(metrics[condition]["unit_expected_spikes"][0]),
                }
            )
        image_index = int(row.matched_image_index)
        expected_values = {
            "stabilized": float(np.load(matrix_dir / "stabilized_ssi_by_image.npy", mmap_mode="r")[image_index, int(row.unit_index)]),
        }
        for condition in ("q01", "q06"):
            mask = movie["image_index"].astype(int).eq(image_index) & movie["trace_index"].astype(int).eq(int(getattr(row, f"{condition}_trace_index")))
            expected_values[condition] = float(np.asarray(ssi[mask.to_numpy(), int(row.unit_index)])[0])
        for condition, expected_value in expected_values.items():
            observed = float(metrics[condition]["unit_bits_per_spike"][0])
            validation_rows.append(
                {
                    "unit_index": int(row.unit_index), "condition": condition,
                    "cached_ssi": expected_value, "rerendered_ssi": observed,
                    "absolute_difference": abs(observed - expected_value),
                }
            )
        render_example(pd.Series(row._asdict()), np.asarray(patch), patch_meta, traces, maps, metrics, args.out_dir, int(args.dpi))
        for condition in ("stabilized", "q01", "q06"):
            cache_payload[f"u{int(row.unit_index):03d}_{condition}_maps"] = maps[condition].astype(np.float32)
            cache_payload[f"u{int(row.unit_index):03d}_{condition}_trace"] = traces[condition].astype(np.float32)
    pd.DataFrame(metric_rows).to_csv(args.out_dir / "targeted_render_metrics.csv", index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(args.out_dir / "cached_vs_rerendered_ssi_validation.csv", index=False)
    np.savez_compressed(args.out_dir / "targeted_activation_map_cache.npz", **cache_payload)
    max_difference = float(validation["absolute_difference"].max())
    if max_difference >= 1e-5:
        raise ValueError(f"Targeted rerender did not reproduce cached SSI: max difference={max_difference}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "targeted_unit_drilldown_complete",
        "render_type": "targeted visualization render, not a production population rerun",
        "selection": concrete[["unit_label", "selection_role"]].to_dict(orient="records"),
        "concrete_movie_rule": "per unit, select a contour-matched image closest to aggregate q06-minus-q01 change, then select q01/q06 traces closest to that image-bin mean SSI",
        "baseline": "counterfactual stabilized zero-motion movie, identical to saved stabilized baseline contract",
        "matrix_dir": str(matrix_dir.resolve()),
        "sources": {
            "ssi_matrix": file_identity(matrix_dir / "ssi_matrix.npy"),
            "trace_xy": file_identity(matrix_dir / "trace_xy.npy"),
            "checkpoint_selection": file_identity(checkpoint_dir / "contour_matched_unit_selection.csv"),
            "checkpoint_changes": file_identity(checkpoint_dir / "contour_matched_unit_changes.csv"),
        },
        "device": str(args.device),
        "max_cached_vs_rerendered_ssi_difference": max_difference,
        "outputs": {
            "selection": "targeted_example_selection.csv",
            "metrics": "targeted_render_metrics.csv",
            "validation": "cached_vs_rerendered_ssi_validation.csv",
            "map_cache": "targeted_activation_map_cache.npz",
            "per_unit_maps": "u*_representative_short_long_path_maps.{png,pdf}",
            "per_unit_timecourses": "u*_representative_short_long_path_timecourses.{png,pdf}",
        },
        "not_run": "No page-13 orthogonal summary or broader population rerun was performed.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(concrete.to_string(index=False))
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
