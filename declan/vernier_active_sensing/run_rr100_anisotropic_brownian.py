"""Run RR100 Vernier responses for matched-total anisotropic Brownian traces.

This is a focused follow-up to the Vernier walkthrough. It compares Brownian
motion clouds with the same total axis variance but different anisotropy:

    isotropic:        D_across = 1.0, D_along = 1.0
    across-elongated: D_across = r,   D_along = 2 - r
    along-elongated:  D_across = 2-r, D_along = r

For each condition, it computes RR100 movie-medoid spatial rate maps, collapses
them with max pooling for pose-aware Fisher, and computes SSI from the spatial
maps. It saves compact rate/SSI caches rather than large spatial-map caches.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view
from declan.vernier_active_sensing.forward import build_vernier_movie, load_model_and_readout
from declan.vernier_active_sensing.metrics import expected_counts, poisson_fisher_counts
from declan.vernier_active_sensing.stimulus import VernierSpec
from scripts.temporal_decoding.rate_computation import compute_trial_rates


RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
BACKIMAGE_FIXATION_WINDOWS_PATH = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/notebook_vernier_walkthrough/rr100_anisotropic_brownian_long")
    parser.add_argument("--n-traces", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--fd-step-arcmin", type=float, default=0.25)
    parser.add_argument("--elongation", type=float, default=1.5, help="D value on the elongated axis; total D is held at 2.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Recompute cached compact rate/SSI arrays.")
    return parser.parse_args()


def canonical_vernier_spec(offset_arcmin: float = 0.0) -> VernierSpec:
    return VernierSpec(
        offset_arcmin=float(offset_arcmin),
        bar_width_arcmin=2.0,
        gap_arcmin=4.0,
        bar_length_arcmin=12.0,
        contrast=0.5,
        polarity="bright",
    )


def estimate_base_axis_std_deg() -> float:
    fallback = 0.03
    if not BACKIMAGE_FIXATION_WINDOWS_PATH.exists():
        return fallback
    try:
        windows = pd.read_csv(BACKIMAGE_FIXATION_WINDOWS_PATH)
        cols = windows[["cov_xx_deg2", "cov_yy_deg2"]].apply(pd.to_numeric, errors="coerce")
        base_var = 0.5 * (cols["cov_xx_deg2"] + cols["cov_yy_deg2"])
        base_var = base_var[np.isfinite(base_var) & (base_var > 0)]
        if len(base_var):
            return float(np.sqrt(np.nanmedian(base_var)))
    except Exception as exc:
        print(f"Could not read backimage fixation scale, using fallback {fallback}: {exc}", flush=True)
    return fallback


def centered_brownian_trace(n_frames: int, rng: np.random.Generator) -> np.ndarray:
    increments = rng.normal(size=(int(n_frames), 2)).astype(np.float32)
    trace = np.cumsum(increments, axis=0)
    trace -= np.mean(trace, axis=0, keepdims=True)
    return trace.astype(np.float32)


def scale_axis_to_std(values: np.ndarray, target_std: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - float(np.mean(values))
    current = float(np.std(centered))
    if current <= 1e-12 or float(target_std) <= 0.0:
        return np.zeros_like(centered, dtype=np.float32)
    return (centered * (float(target_std) / current)).astype(np.float32)


def anisotropic_trace(unit_trace: np.ndarray, *, d_across: float, d_along: float, base_std_deg: float) -> np.ndarray:
    out = np.zeros_like(unit_trace, dtype=np.float32)
    out[:, 0] = scale_axis_to_std(unit_trace[:, 0], float(base_std_deg) * math.sqrt(float(d_across)))
    out[:, 1] = scale_axis_to_std(unit_trace[:, 1], float(base_std_deg) * math.sqrt(float(d_along)))
    return out.astype(np.float32)


def ssi_single_frame(rate_maps: np.ndarray, eps: float = 1e-8) -> float:
    y = np.asarray(rate_maps, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"Expected (unit, H, W), got {y.shape}")
    flat = y.reshape(y.shape[0], -1)
    rbar = flat.mean(axis=1)
    gain = flat / (rbar[:, None] + eps)
    unit_bits = np.mean(gain * np.log2(gain + eps), axis=1)
    weights = rbar / max(float(rbar.sum()), eps)
    return float(np.sum(weights * unit_bits))


def ssi_timecourse(rate_movie: np.ndarray) -> np.ndarray:
    return np.asarray([ssi_single_frame(rate_movie[t]) for t in range(rate_movie.shape[0])], dtype=np.float32)


def collapse_max(rate_movie: np.ndarray) -> np.ndarray:
    return np.asarray(rate_movie, dtype=np.float32).max(axis=(2, 3))


def condition_specs(elongation: float) -> list[dict[str, Any]]:
    e = float(elongation)
    if e <= 1.0 or e >= 2.0:
        raise ValueError("elongation must be between 1 and 2 so total D=2 remains positive.")
    return [
        {"condition": "brownian_iso_1x", "label": "isotropic", "D_across": 1.0, "D_along": 1.0, "color": "#777777"},
        {"condition": "brownian_across_elongated", "label": "across-elongated", "D_across": e, "D_along": 2.0 - e, "color": "#2ca02c"},
        {"condition": "brownian_along_elongated", "label": "along-elongated", "D_across": 2.0 - e, "D_along": e, "color": "#ff7f0e"},
    ]


def build_trace_bank(args: argparse.Namespace, specs: list[dict[str, Any]], base_std_deg: float) -> dict[str, list[np.ndarray]]:
    rng = np.random.default_rng(int(args.seed))
    unit_traces = [centered_brownian_trace(int(args.max_frames), rng) for _ in range(int(args.n_traces))]
    return {
        row["condition"]: [
            anisotropic_trace(unit_trace, d_across=row["D_across"], d_along=row["D_along"], base_std_deg=base_std_deg)
            for unit_trace in unit_traces
        ]
        for row in specs
    }


def summarize_trace_bank(
    specs: list[dict[str, Any]],
    traces_by_condition: dict[str, list[np.ndarray]],
    *,
    base_std_deg: float,
) -> pd.DataFrame:
    """Audit realized axis variances for the finite Brownian traces."""
    rows: list[dict[str, Any]] = []
    base_var = max(float(base_std_deg) ** 2, 1e-12)
    for row in specs:
        condition = str(row["condition"])
        arr = np.asarray(traces_by_condition[condition], dtype=np.float64)
        centered = arr - np.mean(arr, axis=1, keepdims=True)
        steps = np.diff(arr, axis=1)
        pos_var_x = np.var(centered[:, :, 0], axis=1)
        pos_var_y = np.var(centered[:, :, 1], axis=1)
        step_var_x = np.var(steps[:, :, 0], axis=1)
        step_var_y = np.var(steps[:, :, 1], axis=1)
        rows.append(
            {
                "condition": condition,
                "label": row["label"],
                "target_D_across": float(row["D_across"]),
                "target_D_along": float(row["D_along"]),
                "base_std_arcmin": float(base_std_deg) * 60.0,
                "position_std_across_arcmin": float(np.mean(np.sqrt(pos_var_x))) * 60.0,
                "position_std_along_arcmin": float(np.mean(np.sqrt(pos_var_y))) * 60.0,
                "position_var_across_rel": float(np.mean(pos_var_x) / base_var),
                "position_var_along_rel": float(np.mean(pos_var_y) / base_var),
                "position_var_total_rel": float(np.mean(pos_var_x + pos_var_y) / base_var),
                "step_std_across_arcmin": float(np.mean(np.sqrt(step_var_x))) * 60.0,
                "step_std_along_arcmin": float(np.mean(np.sqrt(step_var_y))) * 60.0,
                "step_var_across": float(np.mean(step_var_x)),
                "step_var_along": float(np.mean(step_var_y)),
            }
        )
    audit = pd.DataFrame(rows)
    iso = audit[audit["condition"].eq("brownian_iso_1x")]
    if not iso.empty:
        iso_row = iso.iloc[0]
        audit["step_var_across_rel_to_iso"] = audit["step_var_across"] / max(float(iso_row["step_var_across"]), 1e-12)
        audit["step_var_along_rel_to_iso"] = audit["step_var_along"] / max(float(iso_row["step_var_along"]), 1e-12)
    return audit


def compute_compact_cache(args: argparse.Namespace, specs: list[dict[str, Any]], traces_by_condition: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    device_arg = None if str(args.device).lower() == "auto" else str(args.device)
    print("Loading model/readout...", flush=True)
    model, readout = load_model_and_readout(device=device_arg)
    device = str(next(model.model.parameters()).device)
    print(f"Model device: {device}", flush=True)

    view = load_population_view(version_name=RR100_VERSION)
    spec0 = canonical_vernier_spec(0.0)
    out: dict[str, Any] = {}

    for row in specs:
        condition = str(row["condition"])
        print(f"Condition: {condition}", flush=True)
        plus_rates: list[np.ndarray] = []
        minus_rates: list[np.ndarray] = []
        ssi_curves: list[np.ndarray] = []
        for trace_idx, trace in enumerate(traces_by_condition[condition]):
            per_sign_maps: dict[str, np.ndarray] = {}
            for sign, offset in (("plus", +float(args.fd_step_arcmin)), ("minus", -float(args.fd_step_arcmin))):
                stim = build_vernier_movie(spec0.with_offset(offset), trace, device=device)
                full_spatial = compute_trial_rates(
                    model,
                    readout,
                    stim,
                    batch_size=int(args.batch_size),
                    return_spatial=True,
                ).astype(np.float32)
                rr_spatial = apply_population_view(full_spatial, view).astype(np.float32)
                per_sign_maps[sign] = rr_spatial
                del stim, full_spatial
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            t = min(per_sign_maps["plus"].shape[0], per_sign_maps["minus"].shape[0])
            plus_map = per_sign_maps["plus"][:t]
            minus_map = per_sign_maps["minus"][:t]
            plus_rates.append(collapse_max(plus_map))
            minus_rates.append(collapse_max(minus_map))
            ssi_curves.append(ssi_timecourse(0.5 * (plus_map + minus_map)))
            print(f"  trace {trace_idx}: T={t}", flush=True)
            del per_sign_maps, plus_map, minus_map
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out[f"plus_rates__{condition}"] = np.asarray(plus_rates, dtype=np.float32)
        out[f"minus_rates__{condition}"] = np.asarray(minus_rates, dtype=np.float32)
        out[f"ssi_curves__{condition}"] = np.asarray(ssi_curves, dtype=np.float32)

    out["condition_names"] = np.asarray([row["condition"] for row in specs])
    out["condition_labels"] = np.asarray([row["label"] for row in specs])
    out["D_across"] = np.asarray([row["D_across"] for row in specs], dtype=np.float32)
    out["D_along"] = np.asarray([row["D_along"] for row in specs], dtype=np.float32)
    return out


def summarize_cache(args: argparse.Namespace, specs: list[dict[str, Any]], cache: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for row in specs:
        condition = str(row["condition"])
        plus = np.asarray(cache[f"plus_rates__{condition}"], dtype=np.float32)
        minus = np.asarray(cache[f"minus_rates__{condition}"], dtype=np.float32)
        ssi_curves = np.asarray(cache[f"ssi_curves__{condition}"], dtype=np.float32)
        fisher_finals: list[float] = []
        ssi_trace_means: list[float] = []
        for trace_idx in range(plus.shape[0]):
            t = min(plus.shape[1], minus.shape[1])
            info = poisson_fisher_counts(
                expected_counts(plus[trace_idx, :t], 1.0 / 120.0),
                expected_counts(minus[trace_idx, :t], 1.0 / 120.0),
                step_arcmin=float(args.fd_step_arcmin),
            )
            fisher_final = float(info.cumulative_fisher[-1])
            ssi_mean = float(np.nanmean(ssi_curves[trace_idx, :t]))
            fisher_finals.append(fisher_final)
            ssi_trace_means.append(ssi_mean)
            trace_rows.append(
                {
                    "condition": condition,
                    "label": row["label"],
                    "trace_index": trace_idx,
                    "pose_aware_fisher": fisher_final,
                    "ssi_bits_per_spike": ssi_mean,
                }
            )
        fisher_arr = np.asarray(fisher_finals, dtype=np.float64)
        ssi_arr = np.asarray(ssi_trace_means, dtype=np.float64)
        summary_rows.append(
            {
                "condition": condition,
                "label": row["label"],
                "D_across": row["D_across"],
                "D_along": row["D_along"],
                "D_total": row["D_across"] + row["D_along"],
                "n_traces": int(plus.shape[0]),
                "n_frames": int(plus.shape[1]),
                "fd_step_arcmin": float(args.fd_step_arcmin),
                "pose_aware_fisher_mean": float(np.nanmean(fisher_arr)),
                "pose_aware_fisher_sem": float(np.nanstd(fisher_arr, ddof=1) / max(math.sqrt(fisher_arr.size), 1.0)),
                "ssi_bits_per_spike_mean": float(np.nanmean(ssi_arr)),
                "ssi_bits_per_spike_sem": float(np.nanstd(ssi_arr, ddof=1) / max(math.sqrt(ssi_arr.size), 1.0)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    iso = summary[summary["condition"].eq("brownian_iso_1x")].iloc[0]
    summary["pose_aware_fisher_vs_iso"] = summary["pose_aware_fisher_mean"] / float(iso["pose_aware_fisher_mean"])
    summary["ssi_bits_per_spike_vs_iso"] = summary["ssi_bits_per_spike_mean"] / float(iso["ssi_bits_per_spike_mean"])
    return summary, pd.DataFrame(trace_rows)


def write_plots(args: argparse.Namespace, specs: list[dict[str, Any]], summary: pd.DataFrame, trace_table: pd.DataFrame) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iso = summary[summary["condition"].eq("brownian_iso_1x")].iloc[0]
    plot_df = summary[summary["condition"].isin(["brownian_across_elongated", "brownian_along_elongated"])].copy()
    colors = {row["condition"]: row["color"] for row in specs}

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), dpi=220, constrained_layout=True)
    metric_specs = [
        ("pose_aware_fisher_mean", "pose_aware_fisher_sem", "Pose-aware Fisher", "final Fisher"),
        ("ssi_bits_per_spike_mean", "ssi_bits_per_spike_sem", "SSI", "mean bits/spike"),
    ]
    for ax, (value_col, sem_col, title, ylabel) in zip(axes, metric_specs, strict=True):
        x = np.arange(len(plot_df))
        vals = plot_df[value_col].to_numpy(dtype=float)
        sem = plot_df[sem_col].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            yerr=sem,
            capsize=3,
            color=[colors[c] for c in plot_df["condition"]],
            alpha=0.9,
            width=0.62,
        )
        iso_val = float(iso[value_col])
        ax.axhline(iso_val, color="#555555", linestyle="--", linewidth=1.0, alpha=0.75, label="isotropic")
        for xi, val in zip(x, vals, strict=True):
            ratio = val / iso_val if iso_val > 0 else np.nan
            ax.text(xi, val + max(np.nanmax(vals), iso_val) * 0.035, f"{ratio:.2f}x", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["label"].tolist(), rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ymax = max(float(np.nanmax(vals + sem)), iso_val)
        ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "RR100 movie-medoid Vernier: matched-total anisotropic Brownian motion\n"
        f"D_across + D_along = 2; fd={float(args.fd_step_arcmin):g} arcmin; "
        f"{int(args.n_traces)} traces x {int(args.max_frames)} frames",
        y=1.07,
        fontsize=10,
    )
    fig.savefig(out_dir / "rr100_anisotropic_brownian_poseaware_ssi_bars.png", bbox_inches="tight")

    fig2, axes2 = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=200, constrained_layout=True)
    for ax, metric, ylabel, title in [
        (axes2[0], "pose_aware_fisher", "trace final Fisher", "Trace-level pose-aware Fisher"),
        (axes2[1], "ssi_bits_per_spike", "trace mean SSI", "Trace-level SSI"),
    ]:
        for row_idx, row in enumerate(specs):
            condition = row["condition"]
            vals = trace_table[trace_table["condition"].eq(condition)][metric].to_numpy(dtype=float)
            x = np.full(vals.shape, row_idx, dtype=float)
            ax.scatter(x, vals, s=18, alpha=0.55, color=row["color"], edgecolor="none")
            ax.errorbar(
                [row_idx],
                [float(np.nanmean(vals))],
                yerr=[float(np.nanstd(vals, ddof=1) / max(math.sqrt(vals.size), 1.0))],
                fmt="o",
                color="black",
                capsize=3,
                markersize=4,
            )
        ax.set_xticks(np.arange(len(specs)))
        ax.set_xticklabels([row["label"] for row in specs], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig2.suptitle("RR100 anisotropic Brownian trace-level spread", y=1.06, fontsize=10)
    fig2.savefig(out_dir / "rr100_anisotropic_brownian_trace_level_spread.png", bbox_inches="tight")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = condition_specs(float(args.elongation))
    compact_cache_path = args.out_dir / (
        f"rr100_anisotropic_brownian_compact_rates_n{int(args.n_traces)}_"
        f"t{int(args.max_frames)}_fd{float(args.fd_step_arcmin):.4f}_"
        f"e{float(args.elongation):.2f}_seed{int(args.seed)}.npz"
    )
    base_std_deg = estimate_base_axis_std_deg()
    traces_by_condition = build_trace_bank(args, specs, base_std_deg)
    trace_audit = summarize_trace_bank(specs, traces_by_condition, base_std_deg=base_std_deg)
    trace_audit_path = args.out_dir / "rr100_anisotropic_brownian_trace_audit.csv"
    trace_audit.to_csv(trace_audit_path, index=False)
    print(f"Base axis std: {base_std_deg * 60:.3f} arcmin", flush=True)
    print(f"Output dir: {args.out_dir}", flush=True)
    print(f"Compact cache: {compact_cache_path}", flush=True)
    print(f"Saved trace audit: {trace_audit_path}", flush=True)

    if compact_cache_path.exists() and not args.force:
        print("Loading compact cache.", flush=True)
        loaded = np.load(compact_cache_path, allow_pickle=True)
        cache = {key: loaded[key] for key in loaded.files}
    else:
        cache = compute_compact_cache(args, specs, traces_by_condition)
        cache["base_std_deg"] = np.asarray([base_std_deg], dtype=np.float32)
        np.savez_compressed(compact_cache_path, **cache)
        print(f"Saved compact cache: {compact_cache_path}", flush=True)

    summary, trace_table = summarize_cache(args, specs, cache)
    summary_path = args.out_dir / "rr100_anisotropic_brownian_poseaware_ssi_summary.csv"
    trace_path = args.out_dir / "rr100_anisotropic_brownian_poseaware_ssi_trace_table.csv"
    summary.to_csv(summary_path, index=False)
    trace_table.to_csv(trace_path, index=False)
    write_plots(args, specs, summary, trace_table)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"), flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved trace table: {trace_path}", flush=True)
    print(f"Saved plot: {args.out_dir / 'rr100_anisotropic_brownian_poseaware_ssi_bars.png'}", flush=True)


if __name__ == "__main__":
    main()
