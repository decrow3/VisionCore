#!/usr/bin/env python3
"""Probe RR100 grating speed tuning with repeated SF/TF speed diagonals."""

from __future__ import annotations

import argparse
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
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.active_sensing_movie_information.plot_backimage_rr100_instantaneous_unit_maps import (
    DEFAULT_OUT_DIR as DEFAULT_BACKIMAGE_UNIT_MAP_DIR,
    RR100_MOVIE_MEDOID_VERSION,
    STIMULUS_NORMALIZATION,
    orientation_axis_180,
    parse_float_list,
    parse_int_list,
    write_json,
)
from declan.active_sensing_movie_information.run_backimage_rr100_frequency_tuning_probe import (
    compute_rr100_movie_maps,
    identity_text,
    load_source_summary_and_plot_units,
    make_windowed_drifting_grating_movie,
    scalar_readout_traces,
    sinusoid_amplitude,
    stimulus_sampling_summary,
)
from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import CanonicalTwinScorer
from declan.redundancy_resolved_v1_population import load_population_view


DEFAULT_OUT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_speed_controlled_grating_probe_v1"
)
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_BACKIMAGE_UNIT_MAP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--units", type=str, default="")
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument("--orientation-deg", type=str, default="0,45,90,135")
    parser.add_argument(
        "--cycle-valid-spatial-cpds",
        type=str,
        default="0.4,0.8,1.6,3.2,6.4,12.8",
        help="Main SFs, intended to be at or above one cycle across the grating window.",
    )
    parser.add_argument(
        "--subcycle-control-spatial-cpds",
        type=str,
        default="0.05,0.1,0.2",
        help="Separate low-SF control family. Empty string disables.",
    )
    parser.add_argument("--speeds-dps", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--min-temporal-hz", type=float, default=0.2)
    parser.add_argument("--max-temporal-hz", type=float, default=54.0)
    parser.add_argument(
        "--include-subcycle-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include sub-cycle SFs as a separate control family rather than mixing them with the main speed tuning.",
    )
    parser.add_argument(
        "--scalar-readout",
        choices=("center_pixel", "spatial_mean"),
        default="center_pixel",
    )
    parser.add_argument("--n-phases", type=int, default=2)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--frame-rate-hz", type=float, default=120.0)
    parser.add_argument("--n-lags", type=int, default=32)
    parser.add_argument("--discard-frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=101)
    parser.add_argument("--ppd", type=float, default=37.50476617)
    parser.add_argument("--contrast", type=float, default=0.8)
    parser.add_argument("--window-sigma-frac", type=float, default=0.28)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Optional SF/TF pair limit for smoke tests. Zero runs the full speed-controlled table.",
    )
    return parser.parse_args()


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


def phase_schedule(n_phases: int) -> list[tuple[int, float, str]]:
    n = max(1, int(n_phases))
    return [(idx, float(2.0 * math.pi * idx / n), "dynamic_uniform_grid") for idx in range(n)]


def finalize_pair_table(table: pd.DataFrame) -> pd.DataFrame:
    table = table.reset_index(drop=True).copy()
    table = table.drop(columns=["pair_id", "n_spatial_cpds_for_speed_family"], errors="ignore")
    table.insert(0, "pair_id", np.arange(table.shape[0], dtype=int))
    support = table.groupby(["speed_family", "speed_dps"], sort=True)["spatial_cpd"].nunique()
    table["n_spatial_cpds_for_speed_family"] = [
        int(support.loc[(row.speed_family, row.speed_dps)]) for row in table.itertuples()
    ]
    return table


def build_speed_pair_table(args: argparse.Namespace, sampling: dict[str, float]) -> pd.DataFrame:
    speeds = parse_float_list(str(args.speeds_dps))
    main_sfs = parse_float_list(str(args.cycle_valid_spatial_cpds))
    sub_sfs = (
        parse_float_list(str(args.subcycle_control_spatial_cpds))
        if str(args.subcycle_control_spatial_cpds).strip()
        else []
    )
    one_cycle = float(sampling["one_cycle_across_window_cpd"])
    rows: list[dict[str, Any]] = []

    def add_family(family: str, sfs: list[float], include: bool) -> None:
        if not include:
            return
        for speed in speeds:
            for sf in sfs:
                tf = float(speed) * float(sf)
                if tf < float(args.min_temporal_hz) - 1e-12:
                    continue
                if tf > float(args.max_temporal_hz) + 1e-12:
                    continue
                rows.append(
                    {
                        "speed_family": family,
                        "speed_dps": float(speed),
                        "log2_speed_dps": float(np.log2(float(speed))),
                        "spatial_cpd": float(sf),
                        "temporal_hz": float(tf),
                        "log2_spatial_cpd": float(np.log2(float(sf))),
                        "log2_temporal_hz": float(np.log2(float(tf))),
                        "cycles_across_window": float(sf) / max(one_cycle, EPS),
                        "is_cycle_valid_sf": bool(float(sf) >= one_cycle),
                    }
                )

    add_family("cycle_valid", main_sfs, True)
    add_family("subcycle_control", sub_sfs, bool(args.include_subcycle_controls))
    if not rows:
        raise ValueError("No valid speed-controlled SF/TF pairs after temporal-frequency filtering.")
    table = pd.DataFrame(rows).sort_values(["speed_family", "speed_dps", "spatial_cpd"]).reset_index(drop=True)
    return finalize_pair_table(table)


def compute_probe_rows(
    args: argparse.Namespace,
    *,
    pair_table: pd.DataFrame,
    orientation_summary: pd.DataFrame,
) -> list[dict[str, Any]]:
    orientation_degrees = [orientation_axis_180(v) for v in parse_float_list(str(args.orientation_deg))]
    phases = phase_schedule(int(args.n_phases))
    n_valid_frames = max(int(round(float(args.duration_s) * float(args.frame_rate_hz))), int(args.n_lags) + 8)
    discard_frames = min(int(args.discard_frames), max(n_valid_frames - 8, 0))
    view = load_population_view(version_name=str(args.rr100_version))
    scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.batch_size), empty_cache_every_batch=True)
    summary_by_unit = {int(row["unit_index"]): row for _, row in orientation_summary.iterrows()}
    rr100_units = list(range(int(view.n_units)))

    rows: list[dict[str, Any]] = []
    total = len(orientation_degrees) * int(pair_table.shape[0]) * len(phases)
    done = 0
    for orientation_deg in orientation_degrees:
        for pair in pair_table.itertuples(index=False):
            for phase_idx, phase_rad, phase_policy in phases:
                movie = make_windowed_drifting_grating_movie(
                    image_size=int(args.image_size),
                    orientation_deg=float(orientation_deg),
                    spatial_cpd=float(pair.spatial_cpd),
                    temporal_hz=float(pair.temporal_hz),
                    phase_rad=float(phase_rad),
                    n_valid_frames=n_valid_frames,
                    n_lags=int(args.n_lags),
                    frame_rate_hz=float(args.frame_rate_hz),
                    ppd=float(args.ppd),
                    contrast=float(args.contrast),
                    window_sigma_frac=float(args.window_sigma_frac),
                )
                rr100 = compute_rr100_movie_maps(scorer, view, movie, n_lags=int(args.n_lags))
                scalar_all, center_y, center_x = scalar_readout_traces(rr100, str(args.scalar_readout))
                scalar_all = np.asarray(scalar_all, dtype=np.float64)
                analysis = scalar_all[int(discard_frames) :]
                mean_rate = np.mean(analysis, axis=0)
                peak_rate = np.max(analysis, axis=0)
                rate_std = np.std(analysis, axis=0)
                done += 1
                for unit in rr100_units:
                    prior_row = summary_by_unit.get(int(unit))
                    prior_pref = float("nan") if prior_row is None else float(prior_row.get("preferred_orientation_deg", float("nan")))
                    prior_osi = float("nan") if prior_row is None else float(prior_row.get("orientation_selectivity_index", float("nan")))
                    scalar = scalar_all[:, int(unit)]
                    amp = sinusoid_amplitude(
                        scalar,
                        temporal_hz=float(pair.temporal_hz),
                        frame_rate_hz=float(args.frame_rate_hz),
                        discard_frames=discard_frames,
                    )
                    rows.append(
                        {
                            "unit_index": int(unit),
                            "unit_label": f"u{int(unit):03d}",
                            "pair_id": int(pair.pair_id),
                            "speed_family": str(pair.speed_family),
                            "speed_dps": float(pair.speed_dps),
                            "log2_speed_dps": float(pair.log2_speed_dps),
                            "spatial_cpd": float(pair.spatial_cpd),
                            "temporal_hz": float(pair.temporal_hz),
                            "cycles_across_window": float(pair.cycles_across_window),
                            "is_cycle_valid_sf": bool(pair.is_cycle_valid_sf),
                            "n_spatial_cpds_for_speed_family": int(pair.n_spatial_cpds_for_speed_family),
                            "probe_orientation_deg": float(orientation_deg),
                            "prior_preferred_orientation_deg": prior_pref,
                            "prior_orientation_selectivity_index": prior_osi,
                            "phase_index": int(phase_idx),
                            "phase_rad": float(phase_rad),
                            "phase_policy": str(phase_policy),
                            "n_valid_frames": int(n_valid_frames),
                            "discard_frames": int(discard_frames),
                            "frame_rate_hz": float(args.frame_rate_hz),
                            "image_size_px": int(args.image_size),
                            "ppd": float(args.ppd),
                            "contrast": float(args.contrast),
                            "window_sigma_frac": float(args.window_sigma_frac),
                            "scalar_readout": str(args.scalar_readout),
                            "center_y": center_y,
                            "center_x": center_x,
                            "mean_rate": float(mean_rate[int(unit)]),
                            "peak_rate": float(peak_rate[int(unit)]),
                            "rate_std": float(rate_std[int(unit)]),
                            "response_amp": float(amp["response_amp"]),
                            "response_amp_sq": float(amp["response_amp_sq"]),
                            "response_amp_per_contrast": float(amp["response_amp"] / max(float(args.contrast), EPS))
                            if np.isfinite(float(amp["response_amp"]))
                            else float("nan"),
                            "response_amp_sq_per_contrast_sq": float(
                                amp["response_amp_sq"] / max(float(args.contrast) ** 2, EPS)
                            )
                            if np.isfinite(float(amp["response_amp_sq"]))
                            else float("nan"),
                            "n_analysis_frames": int(amp["n_analysis_frames"]),
                            "probe_contract": (
                                "speed-controlled repeated-diagonal drifting gratings; speed = temporal_hz / spatial_cpd; "
                                "speed_family separates cycle-valid SFs from sub-cycle flicker/ramp controls"
                            ),
                        }
                    )
                watched = [17, 18, 26]
                watched_text = ", ".join(
                    f"u{unit:03d} mean={float(mean_rate[unit]):.4g}" for unit in watched if unit < mean_rate.shape[0]
                )
                print(
                    f"[{done}/{total}] family={pair.speed_family} speed={float(pair.speed_dps):g} deg/s "
                    f"sf={float(pair.spatial_cpd):g} cpd tf={float(pair.temporal_hz):g} Hz "
                    f"ori={float(orientation_deg):g} phase={phase_idx}; {watched_text}",
                    flush=True,
                )
                del rr100, movie, scalar_all, analysis
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probe = pd.DataFrame(rows)
    if probe.empty:
        return probe, pd.DataFrame(), pd.DataFrame()
    for col in [
        "unit_index",
        "pair_id",
        "speed_dps",
        "log2_speed_dps",
        "spatial_cpd",
        "temporal_hz",
        "probe_orientation_deg",
        "mean_rate",
        "response_amp_sq",
        "response_amp",
    ]:
        probe[col] = pd.to_numeric(probe[col], errors="coerce")
    keys = [
        "unit_index",
        "unit_label",
        "pair_id",
        "speed_family",
        "speed_dps",
        "log2_speed_dps",
        "spatial_cpd",
        "temporal_hz",
        "cycles_across_window",
        "is_cycle_valid_sf",
        "n_spatial_cpds_for_speed_family",
        "probe_orientation_deg",
        "scalar_readout",
    ]
    grouped_rows: list[dict[str, Any]] = []
    for key_values, sub in probe.groupby(keys, sort=True):
        rec = dict(zip(keys, key_values, strict=True))
        amp_sq = sub["response_amp_sq"].to_numpy(dtype=float)
        amp = np.sqrt(np.nanmean(amp_sq)) if np.isfinite(amp_sq).any() else float("nan")
        rec.update(
            {
                "n_phases": int(sub.shape[0]),
                "phase_policies": ",".join(sorted(set(str(v) for v in sub["phase_policy"].to_list()))),
                "prior_preferred_orientation_deg": float(np.nanmean(sub["prior_preferred_orientation_deg"].to_numpy(dtype=float))),
                "prior_orientation_selectivity_index": float(
                    np.nanmean(sub["prior_orientation_selectivity_index"].to_numpy(dtype=float))
                ),
                "mean_rate": float(np.nanmean(sub["mean_rate"].to_numpy(dtype=float))),
                "sem_mean_rate": float(np.nanstd(sub["mean_rate"].to_numpy(dtype=float), ddof=1) / math.sqrt(sub.shape[0]))
                if sub.shape[0] > 1
                else 0.0,
                "response_amp_rms": float(amp),
                "response_amp_rms_per_contrast": float(amp / max(float(sub["contrast"].iloc[0]), EPS))
                if np.isfinite(amp)
                else float("nan"),
                "response_amp_sq_mean": float(np.nanmean(amp_sq)) if np.isfinite(amp_sq).any() else float("nan"),
                "probe_contract": str(sub["probe_contract"].iloc[0]),
            }
        )
        grouped_rows.append(rec)
    grouped = pd.DataFrame(grouped_rows)

    speed_rows: list[dict[str, Any]] = []
    for key_values, sub in grouped.groupby(["unit_index", "unit_label", "speed_family", "speed_dps"], sort=True):
        unit, unit_label, family, speed = key_values
        amp = pd.to_numeric(sub["response_amp_rms"], errors="coerce").to_numpy(dtype=float)
        speed_rows.append(
            {
                "unit_index": int(unit),
                "unit_label": str(unit_label),
                "speed_family": str(family),
                "speed_dps": float(speed),
                "log2_speed_dps": float(np.log2(float(speed))),
                "response_amp_rms_mean": float(np.nanmean(amp)),
                "response_amp_rms_median": float(np.nanmedian(amp)),
                "n_orientation_pair_rows": int(sub.shape[0]),
                "n_spatial_cpds": int(sub["spatial_cpd"].nunique()),
                "n_temporal_hz": int(sub["temporal_hz"].nunique()),
                "n_pairs": int(sub["pair_id"].nunique()),
            }
        )
    speed_curves = pd.DataFrame(speed_rows)

    summary_rows: list[dict[str, Any]] = []
    for key_values, sub in speed_curves.groupby(["unit_index", "unit_label", "speed_family"], sort=True):
        unit, unit_label, family = key_values
        sub = sub.sort_values("speed_dps")
        amp = sub["response_amp_rms_mean"].to_numpy(dtype=float)
        speed = sub["speed_dps"].to_numpy(dtype=float)
        finite = np.isfinite(amp) & np.isfinite(speed) & (speed > 0)
        if finite.sum() == 0 or np.nansum(amp[finite]) <= 0:
            continue
        amp_f = amp[finite]
        speed_f = speed[finite]
        log_speed_f = np.log2(speed_f)
        weights = amp_f / np.nansum(amp_f)
        z_std = float(np.nanstd(amp_f))
        z = (amp_f - float(np.nanmean(amp_f))) / z_std if z_std > 1e-12 else np.full_like(amp_f, np.nan)
        z_slope = float(np.polyfit(log_speed_f[np.isfinite(z)], z[np.isfinite(z)], 1)[0]) if np.isfinite(z).sum() >= 2 else float("nan")
        peak_idx = int(np.nanargmax(amp_f))
        summary_rows.append(
            {
                "unit_index": int(unit),
                "unit_label": str(unit_label),
                "speed_family": str(family),
                "peak_speed_dps": float(speed_f[peak_idx]),
                "log2_peak_speed_dps": float(log_speed_f[peak_idx]),
                "amp_weighted_speed_dps": float(np.nansum(weights * speed_f)),
                "amp_weighted_log2_speed_dps": float(np.nansum(weights * log_speed_f)),
                "speed_curve_log_slope_z": z_slope,
                "speed_curve_dynamic_range": float(np.nanmax(amp_f) - np.nanmin(amp_f)),
                "speed_curve_mean_amp": float(np.nanmean(amp_f)),
                "n_speed_bins": int(finite.sum()),
            }
        )
    speed_summary = pd.DataFrame(summary_rows)
    return grouped, speed_curves, speed_summary


def plot_speed_summary(
    out_dir: Path,
    pair_table: pd.DataFrame,
    speed_curves: pd.DataFrame,
    speed_summary: pd.DataFrame,
    *,
    plot_units: list[int],
    dpi: int,
) -> tuple[Path, Path]:
    families = [family for family in ["cycle_valid", "subcycle_control"] if family in set(pair_table["speed_family"])]
    fig, axes = plt.subplots(len(families), 3, figsize=(13.5, 3.8 * len(families)), squeeze=False)
    for row_idx, family in enumerate(families):
        family_pairs = pair_table[pair_table["speed_family"].eq(family)].copy()
        family_curves = speed_curves[speed_curves["speed_family"].eq(family)].copy()
        family_summary = speed_summary[speed_summary["speed_family"].eq(family)].copy()
        ax = axes[row_idx, 0]
        support = (
            family_pairs.groupby("speed_dps", sort=True)
            .agg(n_pairs=("pair_id", "size"), min_tf=("temporal_hz", "min"), max_tf=("temporal_hz", "max"))
            .reset_index()
        )
        ax.bar(support["speed_dps"], support["n_pairs"], width=0.18 * support["speed_dps"], color="#4c78a8")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("speed (deg/s)")
        ax.set_ylabel("SF/TF pairs")
        ax.set_title(f"{family}: diagonal support")
        ax.grid(True, axis="y", color="0.9")

        ax = axes[row_idx, 1]
        chosen = [unit for unit in plot_units if unit in set(family_curves["unit_index"])]
        if not chosen:
            chosen = [int(v) for v in family_summary.sort_values("speed_curve_dynamic_range", ascending=False)["unit_index"].head(4)]
        for unit in chosen[:6]:
            sub = family_curves[family_curves["unit_index"].eq(int(unit))].sort_values("speed_dps")
            amp = sub["response_amp_rms_mean"].to_numpy(dtype=float)
            if amp.size == 0:
                continue
            z = (amp - np.nanmean(amp)) / max(float(np.nanstd(amp)), EPS)
            ax.plot(sub["speed_dps"], z, marker="o", linewidth=1.4, label=f"u{int(unit):03d}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("speed (deg/s)")
        ax.set_ylabel("within-unit speed z")
        ax.set_title("example unit speed curves")
        ax.grid(True, color="0.9")
        ax.legend(frameon=False, fontsize=8, ncol=2)

        ax = axes[row_idx, 2]
        ax.scatter(
            family_summary["amp_weighted_speed_dps"],
            family_summary["speed_curve_dynamic_range"],
            s=28,
            alpha=0.75,
            color="#f58518",
            edgecolor="white",
            linewidth=0.35,
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("amp-weighted speed (deg/s)")
        ax.set_ylabel("speed curve dynamic range")
        ax.set_title("all-unit speed preferences")
        ax.grid(True, color="0.9")
    fig.suptitle("BackImage RR100 speed-controlled grating probe", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = out_dir / "backimage_rr100_speed_controlled_grating_probe_summary.png"
    pdf = out_dir / "backimage_rr100_speed_controlled_grating_probe_summary.pdf"
    fig.savefig(png, dpi=int(dpi))
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_units, orientation_summary = load_source_summary_and_plot_units(Path(args.source_dir), parse_int_list(str(args.units)))
    orientation_degrees = [orientation_axis_180(v) for v in parse_float_list(str(args.orientation_deg))]
    sampling = stimulus_sampling_summary(
        ppd=float(args.ppd),
        image_size=int(args.image_size),
        frame_rate_hz=float(args.frame_rate_hz),
    )
    pair_table = build_speed_pair_table(args, sampling)
    if int(args.max_pairs) > 0:
        pair_table = finalize_pair_table(pair_table.iloc[: int(args.max_pairs)])
    over_temporal = pair_table[pair_table["temporal_hz"] > float(sampling["temporal_nyquist_hz"])]
    if not over_temporal.empty:
        raise ValueError("Speed pair table contains temporal frequencies above Nyquist.")
    pair_table.to_csv(out_dir / "speed_controlled_pair_table.csv", index=False)
    identity = {
        "analysis": "backimage_rr100_speed_controlled_grating_probe",
        "source_dir": Path(args.source_dir).resolve(),
        "rr100_version": str(args.rr100_version),
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "computed_units": "all_rr100_units",
        "plot_units": plot_units,
        "orientation_degrees": orientation_degrees,
        "cycle_valid_spatial_cpds": parse_float_list(str(args.cycle_valid_spatial_cpds)),
        "subcycle_control_spatial_cpds": parse_float_list(str(args.subcycle_control_spatial_cpds))
        if str(args.subcycle_control_spatial_cpds).strip()
        else [],
        "include_subcycle_controls": bool(args.include_subcycle_controls),
        "speeds_dps": parse_float_list(str(args.speeds_dps)),
        "min_temporal_hz": float(args.min_temporal_hz),
        "max_temporal_hz": float(args.max_temporal_hz),
        "max_pairs": int(args.max_pairs),
        "pair_table_rows": int(pair_table.shape[0]),
        "scalar_readout": str(args.scalar_readout),
        "n_phases": int(args.n_phases),
        "duration_s": float(args.duration_s),
        "frame_rate_hz": float(args.frame_rate_hz),
        "n_lags": int(args.n_lags),
        "discard_frames": int(args.discard_frames),
        "image_size": int(args.image_size),
        "ppd": float(args.ppd),
        "contrast": float(args.contrast),
        "window_sigma_frac": float(args.window_sigma_frac),
        "stimulus_sampling": sampling,
        "probe_contract": (
            "Explicit repeated SF/TF speed diagonals. Main family uses SFs above one cycle per grating window; "
            "sub-cycle SFs are written as a separate control family when enabled."
        ),
    }
    write_json(out_dir / "speed_controlled_request_identity.json", identity)
    if bool(args.dry_run):
        print(json.dumps(json_ready(identity), indent=2, sort_keys=True))
        print(pair_table.to_string(index=False))
        return

    manifest_path = out_dir / "speed_controlled_manifest.json"
    probe_csv = out_dir / "speed_controlled_probe_rows.csv"
    grouped_csv = out_dir / "speed_controlled_grouped.csv"
    speed_curves_csv = out_dir / "speed_controlled_speed_curves.csv"
    speed_summary_csv = out_dir / "speed_controlled_unit_speed_summary.csv"
    use_cache = False
    if all(path.exists() for path in [manifest_path, probe_csv, grouped_csv, speed_curves_csv, speed_summary_csv]) and not bool(args.force):
        try:
            observed = json.loads(manifest_path.read_text(encoding="utf-8")).get("identity_text", "")
            use_cache = str(observed) == identity_text(identity)
        except Exception:
            use_cache = False
    if use_cache:
        probe_rows = pd.read_csv(probe_csv).to_dict("records")
        grouped = pd.read_csv(grouped_csv)
        speed_curves = pd.read_csv(speed_curves_csv)
        speed_summary = pd.read_csv(speed_summary_csv)
        print(f"Loaded cached speed-controlled probe rows from {probe_csv}", flush=True)
    else:
        probe_rows = compute_probe_rows(args, pair_table=pair_table, orientation_summary=orientation_summary)
        grouped, speed_curves, speed_summary = aggregate_rows(probe_rows)
        pd.DataFrame(probe_rows).to_csv(probe_csv, index=False)
        grouped.to_csv(grouped_csv, index=False)
        speed_curves.to_csv(speed_curves_csv, index=False)
        speed_summary.to_csv(speed_summary_csv, index=False)
    png, pdf = plot_speed_summary(out_dir, pair_table, speed_curves, speed_summary, plot_units=plot_units, dpi=int(args.dpi))
    write_json(
        manifest_path,
        {
            "identity": identity,
            "identity_text": identity_text(identity),
            "n_probe_rows": len(probe_rows),
            "n_grouped_rows": int(grouped.shape[0]),
            "n_speed_curve_rows": int(speed_curves.shape[0]),
            "n_speed_summary_rows": int(speed_summary.shape[0]),
            "outputs": {
                "pair_table_csv": out_dir / "speed_controlled_pair_table.csv",
                "probe_rows_csv": probe_csv,
                "grouped_csv": grouped_csv,
                "speed_curves_csv": speed_curves_csv,
                "speed_summary_csv": speed_summary_csv,
                "summary_png": png,
                "summary_pdf": pdf,
            },
        },
    )
    print(f"Wrote speed-controlled grating probe outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
