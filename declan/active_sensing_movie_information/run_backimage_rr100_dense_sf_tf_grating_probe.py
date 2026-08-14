#!/usr/bin/env python3
"""Run a denser Cartesian SF/TF grating probe for RR100 units."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
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
    "backimage_rr100_dense_sf_tf_grating_probe_v1"
)
EPS = 1e-12


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
        default="0.4,0.565685,0.8,1.131371,1.6,2.262742,3.2,4.525483,6.4,9.050967,12.8,16",
        help="Main SF grid, roughly half-octave spaced and above one cycle across the 101 px window.",
    )
    parser.add_argument(
        "--subcycle-control-spatial-cpds",
        type=str,
        default="0.05,0.1,0.2",
        help="Separate low-SF flicker/ramp control family. Empty string disables.",
    )
    parser.add_argument(
        "--temporal-hz",
        type=str,
        default="0.4,0.565685,0.8,1.131371,1.6,2.262742,3.2,4.525483,6.4,9.050967,12.8,18.101934,25.6,36.203867,51.2",
    )
    parser.add_argument("--max-temporal-hz", type=float, default=54.0)
    parser.add_argument(
        "--include-subcycle-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include sub-cycle SFs as a separate control family.",
    )
    parser.add_argument("--scalar-readout", choices=("center_pixel", "spatial_mean"), default="center_pixel")
    parser.add_argument("--n-phases", type=int, default=2)
    parser.add_argument(
        "--temporal-direction-signs",
        type=str,
        default="1",
        help="Comma-separated temporal direction signs. Use '-1,1' for direction-folded F0 surfaces.",
    )
    parser.add_argument(
        "--include-blank-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Score one matched mean-gray movie and save signed and nonnegative "
            "F0 (phase-averaged mean rate minus blank) for every condition."
        ),
    )
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
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional pair limit for smoke tests. Zero runs all pairs.")
    parser.add_argument(
        "--pair-shard-count",
        type=int,
        default=1,
        help="Number of balanced checkerboard SF/TF shards.",
    )
    parser.add_argument(
        "--pair-shard-index",
        type=int,
        default=0,
        help="Zero-based checkerboard shard to score.",
    )
    parser.add_argument(
        "--stop-after-new-pairs",
        type=int,
        default=0,
        help="Testing/queue limit: stop cleanly after this many newly completed pair shards.",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Aggregate and plot already completed pair shards without loading the neural model.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
    table = table.drop(
        columns=[
            "pair_id",
            "n_spatial_cpds_for_family",
            "n_temporal_hz_for_family",
            "n_temporal_hz_for_spatial_cpd",
            "n_spatial_cpds_for_temporal_hz",
        ],
        errors="ignore",
    )
    table.insert(0, "pair_id", np.arange(table.shape[0], dtype=int))
    family_sf = table.groupby("speed_family", sort=True)["spatial_cpd"].nunique()
    family_tf = table.groupby("speed_family", sort=True)["temporal_hz"].nunique()
    sf_support = table.groupby(["speed_family", "spatial_cpd"], sort=True)["temporal_hz"].nunique()
    tf_support = table.groupby(["speed_family", "temporal_hz"], sort=True)["spatial_cpd"].nunique()
    table["n_spatial_cpds_for_family"] = [int(family_sf.loc[row.speed_family]) for row in table.itertuples()]
    table["n_temporal_hz_for_family"] = [int(family_tf.loc[row.speed_family]) for row in table.itertuples()]
    table["n_temporal_hz_for_spatial_cpd"] = [
        int(sf_support.loc[(row.speed_family, row.spatial_cpd)]) for row in table.itertuples()
    ]
    table["n_spatial_cpds_for_temporal_hz"] = [
        int(tf_support.loc[(row.speed_family, row.temporal_hz)]) for row in table.itertuples()
    ]
    table["sf_grid_index"] = table.groupby("speed_family", sort=False)["spatial_cpd"].transform(
        lambda values: values.map(
            {value: index for index, value in enumerate(sorted(pd.unique(values)))}
        )
    ).astype(int)
    table["tf_grid_index"] = table.groupby("speed_family", sort=False)["temporal_hz"].transform(
        lambda values: values.map(
            {value: index for index, value in enumerate(sorted(pd.unique(values)))}
        )
    ).astype(int)
    return table


def build_pair_table(args: argparse.Namespace, sampling: dict[str, float]) -> pd.DataFrame:
    main_sfs = parse_float_list(str(args.cycle_valid_spatial_cpds))
    sub_sfs = (
        parse_float_list(str(args.subcycle_control_spatial_cpds))
        if str(args.subcycle_control_spatial_cpds).strip()
        else []
    )
    temporal_hz = parse_float_list(str(args.temporal_hz))
    one_cycle = float(sampling["one_cycle_across_window_cpd"])
    rows: list[dict[str, Any]] = []

    def add_family(family: str, sfs: list[float], include: bool) -> None:
        if not include:
            return
        for sf in sfs:
            for tf in temporal_hz:
                if float(tf) > float(args.max_temporal_hz) + 1e-12:
                    continue
                rows.append(
                    {
                        "speed_family": family,
                        "spatial_cpd": float(sf),
                        "temporal_hz": float(tf),
                        "speed_dps": float(tf) / max(float(sf), EPS),
                        "log2_speed_dps": float(np.log2(float(tf) / max(float(sf), EPS))),
                        "log2_spatial_cpd": float(np.log2(float(sf))),
                        "log2_temporal_hz": float(np.log2(float(tf))),
                        "cycles_across_window": float(sf) / max(one_cycle, EPS),
                        "is_cycle_valid_sf": bool(float(sf) >= one_cycle),
                        "is_extended_tf_core": bool(32.0 < float(tf) <= 56.0),
                        "is_nyquist_edge_control": bool(np.isclose(float(tf), 0.5 * float(args.frame_rate_hz))),
                    }
                )

    add_family("cycle_valid", main_sfs, True)
    add_family("subcycle_control", sub_sfs, bool(args.include_subcycle_controls))
    if not rows:
        raise ValueError("No valid SF/TF pairs after filtering.")
    table = pd.DataFrame(rows).sort_values(["speed_family", "spatial_cpd", "temporal_hz"]).reset_index(drop=True)
    return finalize_pair_table(table)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".csv") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_pair_shard(
    frame: pd.DataFrame,
    *,
    pair_id: int,
    n_units: int,
    n_conditions: int,
) -> None:
    expected_rows = int(n_units) * int(n_conditions)
    if len(frame) != expected_rows:
        raise ValueError(f"Pair {pair_id} shard has {len(frame)} rows; expected {expected_rows}")
    if set(pd.to_numeric(frame["pair_id"], errors="raise").astype(int)) != {int(pair_id)}:
        raise ValueError(f"Pair shard identity mismatch for pair {pair_id}")
    if frame["unit_index"].nunique() != int(n_units):
        raise ValueError(f"Pair {pair_id} shard does not contain all {n_units} units")
    condition_keys = ["probe_orientation_deg", "temporal_direction_sign", "phase_index"]
    if frame[condition_keys].drop_duplicates().shape[0] != int(n_conditions):
        raise ValueError(f"Pair {pair_id} shard does not contain all {n_conditions} stimulus conditions")
    counts = frame.groupby(condition_keys, sort=False)["unit_index"].nunique()
    if not counts.eq(int(n_units)).all():
        raise ValueError(f"Pair {pair_id} shard has incomplete unit rows within a condition")


def collect_pair_shards(
    out_dir: Path,
    pair_table: pd.DataFrame,
    *,
    n_units: int,
    n_conditions: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    completed: list[int] = []
    pair_dir = out_dir / "pair_shards"
    for pair in pair_table.itertuples(index=False):
        path = pair_dir / f"pair_{int(pair.pair_id):03d}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        validate_pair_shard(
            frame,
            pair_id=int(pair.pair_id),
            n_units=int(n_units),
            n_conditions=int(n_conditions),
        )
        rows.extend(frame.to_dict("records"))
        completed.append(int(pair.pair_id))
    return rows, completed


def write_partial_progress(
    out_dir: Path,
    pair_table: pd.DataFrame,
    completed_pair_ids: list[int],
    *,
    n_units: int,
    n_conditions_per_pair: int,
) -> None:
    requested = pair_table["pair_id"].astype(int).tolist()
    completed_set = set(int(value) for value in completed_pair_ids)
    completed_table = pair_table[pair_table["pair_id"].astype(int).isin(completed_set)].copy()
    write_json(
        out_dir / "dense_sf_tf_partial_progress.json",
        {
            "status": "requested_pair_set_complete" if len(completed_set) == len(requested) else "partial_resumable",
            "requested_pair_ids": requested,
            "completed_pair_ids": sorted(completed_set),
            "n_requested_pairs": len(requested),
            "n_completed_pairs": len(completed_set),
            "completion_fraction": len(completed_set) / max(len(requested), 1),
            "completed_spatial_cpds": sorted(completed_table["spatial_cpd"].unique().tolist()),
            "completed_temporal_hz": sorted(completed_table["temporal_hz"].unique().tolist()),
            "n_units_per_pair": int(n_units),
            "n_conditions_per_pair": int(n_conditions_per_pair),
            "atomic_unit": (
                "one SFxTF pair containing every requested orientation, temporal direction, "
                "phase, and RR100 unit"
            ),
        },
    )


def compute_probe_rows(
    args: argparse.Namespace,
    *,
    pair_table: pd.DataFrame,
    orientation_summary: pd.DataFrame,
    out_dir: Path,
) -> list[dict[str, Any]]:
    orientation_degrees = [orientation_axis_180(v) for v in parse_float_list(str(args.orientation_deg))]
    phases = phase_schedule(int(args.n_phases))
    direction_signs = parse_int_list(str(args.temporal_direction_signs))
    if not direction_signs or any(int(sign) not in (-1, 1) for sign in direction_signs):
        raise ValueError("--temporal-direction-signs must contain only -1 and/or 1")
    direction_signs = sorted(set(int(sign) for sign in direction_signs))
    n_conditions_per_pair = len(orientation_degrees) * len(direction_signs) * len(phases)
    n_valid_frames = max(int(round(float(args.duration_s) * float(args.frame_rate_hz))), int(args.n_lags) + 8)
    discard_frames = min(int(args.discard_frames), max(n_valid_frames - 8, 0))
    view = load_population_view(version_name=str(args.rr100_version))
    n_units = int(view.n_units)
    if n_units != 100:
        raise ValueError(f"Expected RR100 population, found {n_units} units")
    summary_by_unit = {int(row["unit_index"]): row for _, row in orientation_summary.iterrows()}
    rr100_units = list(range(n_units))
    pair_dir = out_dir / "pair_shards"
    pair_dir.mkdir(parents=True, exist_ok=True)

    existing_rows, completed_pair_ids = collect_pair_shards(
        out_dir,
        pair_table,
        n_units=n_units,
        n_conditions=n_conditions_per_pair,
    )
    if bool(args.force) and not bool(args.assemble_only):
        existing_rows = []
        completed_pair_ids = []
    write_partial_progress(
        out_dir,
        pair_table,
        completed_pair_ids,
        n_units=n_units,
        n_conditions_per_pair=n_conditions_per_pair,
    )
    if bool(args.assemble_only):
        if not existing_rows:
            raise FileNotFoundError(f"No completed pair shards found in {pair_dir}")
        return existing_rows

    scorer = CanonicalTwinScorer(
        device=str(args.device),
        batch_size=int(args.batch_size),
        empty_cache_every_batch=True,
    )
    blank_mean_rate = np.full(n_units, np.nan, dtype=np.float64)
    if bool(args.include_blank_reference):
        blank_movie = np.full(
            (n_valid_frames + int(args.n_lags) - 1, int(args.image_size), int(args.image_size)),
            127.5,
            dtype=np.float32,
        )
        blank_rr100 = compute_rr100_movie_maps(scorer, view, blank_movie, n_lags=int(args.n_lags))
        blank_scalar, _blank_center_y, _blank_center_x = scalar_readout_traces(
            blank_rr100, str(args.scalar_readout)
        )
        # The canonical shared readout returns expected counts per 1/frame_rate
        # bin, whereas every public column below is explicitly labelled in Hz.
        # Keep this conversion at the scalar-readout boundary so blank, F0, and
        # modulation-amplitude quantities all share the same physical units.
        blank_scalar = np.asarray(blank_scalar, dtype=np.float64) * float(args.frame_rate_hz)
        blank_mean_rate = np.mean(
            blank_scalar[int(discard_frames) :], axis=0
        )
        del blank_movie, blank_rr100, blank_scalar

    all_rows = list(existing_rows)
    completed_set = set(completed_pair_ids)
    total_conditions = n_conditions_per_pair * int(pair_table.shape[0])
    done_conditions = n_conditions_per_pair * len(completed_set)
    newly_completed = 0
    for pair in pair_table.itertuples(index=False):
        pair_id = int(pair.pair_id)
        if pair_id in completed_set and not bool(args.force):
            print(f"resume: pair {pair_id} already complete", flush=True)
            continue
        pair_rows: list[dict[str, Any]] = []
        for orientation_deg in orientation_degrees:
            for direction_sign in direction_signs:
                for phase_idx, phase_rad, phase_policy in phases:
                    movie = make_windowed_drifting_grating_movie(
                        image_size=int(args.image_size),
                        orientation_deg=float(orientation_deg),
                        spatial_cpd=float(pair.spatial_cpd),
                        temporal_hz=float(direction_sign) * float(pair.temporal_hz),
                        phase_rad=float(phase_rad),
                        n_valid_frames=n_valid_frames,
                        n_lags=int(args.n_lags),
                        frame_rate_hz=float(args.frame_rate_hz),
                        ppd=float(args.ppd),
                        contrast=float(args.contrast),
                        window_sigma_frac=float(args.window_sigma_frac),
                    )
                    rr100 = compute_rr100_movie_maps(scorer, view, movie, n_lags=int(args.n_lags))
                    scalar_all, center_y, center_x = scalar_readout_traces(
                        rr100, str(args.scalar_readout)
                    )
                    scalar_all = (
                        np.asarray(scalar_all, dtype=np.float64) * float(args.frame_rate_hz)
                    )
                    analysis = scalar_all[int(discard_frames) :]
                    mean_rate = np.mean(analysis, axis=0)
                    peak_rate = np.max(analysis, axis=0)
                    rate_std = np.std(analysis, axis=0)
                    done_conditions += 1
                    for unit in rr100_units:
                        prior_row = summary_by_unit.get(int(unit))
                        prior_pref = (
                            float("nan")
                            if prior_row is None
                            else float(prior_row.get("preferred_orientation_deg", float("nan")))
                        )
                        prior_osi = (
                            float("nan")
                            if prior_row is None
                            else float(prior_row.get("orientation_selectivity_index", float("nan")))
                        )
                        scalar = scalar_all[:, int(unit)]
                        amp = sinusoid_amplitude(
                            scalar,
                            temporal_hz=float(pair.temporal_hz),
                            frame_rate_hz=float(args.frame_rate_hz),
                            discard_frames=discard_frames,
                        )
                        blank_rate = float(blank_mean_rate[int(unit)])
                        signed_f0 = (
                            float(mean_rate[int(unit)] - blank_rate)
                            if np.isfinite(blank_rate)
                            else float("nan")
                        )
                        pair_rows.append(
                            {
                                "unit_index": int(unit),
                                "unit_label": f"u{int(unit):03d}",
                                "pair_id": pair_id,
                                "speed_family": str(pair.speed_family),
                                "speed_dps": float(pair.speed_dps),
                                "log2_speed_dps": float(pair.log2_speed_dps),
                                "spatial_cpd": float(pair.spatial_cpd),
                                "temporal_hz": float(pair.temporal_hz),
                                "temporal_direction_sign": int(direction_sign),
                                "signed_temporal_hz": float(direction_sign) * float(pair.temporal_hz),
                                "log2_spatial_cpd": float(pair.log2_spatial_cpd),
                                "log2_temporal_hz": float(pair.log2_temporal_hz),
                                "cycles_across_window": float(pair.cycles_across_window),
                                "is_cycle_valid_sf": bool(pair.is_cycle_valid_sf),
                                "is_extended_tf_core": bool(pair.is_extended_tf_core),
                                "is_nyquist_edge_control": bool(pair.is_nyquist_edge_control),
                                "n_spatial_cpds_for_family": int(pair.n_spatial_cpds_for_family),
                                "n_temporal_hz_for_family": int(pair.n_temporal_hz_for_family),
                                "n_temporal_hz_for_spatial_cpd": int(pair.n_temporal_hz_for_spatial_cpd),
                                "n_spatial_cpds_for_temporal_hz": int(pair.n_spatial_cpds_for_temporal_hz),
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
                                "blank_mean_rate": blank_rate,
                                "signed_f0_hz": signed_f0,
                                "positive_f0_hz": max(signed_f0, 0.0)
                                if np.isfinite(signed_f0)
                                else float("nan"),
                                "peak_rate": float(peak_rate[int(unit)]),
                                "rate_std": float(rate_std[int(unit)]),
                                "response_amp": float(amp["response_amp"]),
                                "response_amp_sq": float(amp["response_amp_sq"]),
                                "response_amp_per_contrast": float(
                                    amp["response_amp"] / max(float(args.contrast), EPS)
                                )
                                if np.isfinite(float(amp["response_amp"]))
                                else float("nan"),
                                "response_amp_sq_per_contrast_sq": float(
                                    amp["response_amp_sq"] / max(float(args.contrast) ** 2, EPS)
                                )
                                if np.isfinite(float(amp["response_amp_sq"]))
                                else float("nan"),
                                "n_analysis_frames": int(amp["n_analysis_frames"]),
                                "probe_contract": (
                                    "dense Cartesian SF/TF drifting-grating grid; speed_dps is derived as "
                                    "temporal_hz/spatial_cpd; one atomic pair shard contains every requested "
                                    "orientation, direction, phase, and RR100 unit"
                                ),
                            }
                        )
                    watched = [17, 18, 26]
                    watched_text = ", ".join(
                        f"u{unit:03d} mean={float(mean_rate[unit]):.4g}"
                        for unit in watched
                        if unit < mean_rate.shape[0]
                    )
                    print(
                        f"[{done_conditions}/{total_conditions}] pair={pair_id} "
                        f"sf={float(pair.spatial_cpd):g} cpd "
                        f"tf={float(direction_sign) * float(pair.temporal_hz):g} Hz "
                        f"ori={float(orientation_deg):g} phase={phase_idx}; {watched_text}",
                        flush=True,
                    )
                    del rr100, movie, scalar_all, analysis

        pair_frame = pd.DataFrame(pair_rows)
        validate_pair_shard(
            pair_frame,
            pair_id=pair_id,
            n_units=n_units,
            n_conditions=n_conditions_per_pair,
        )
        atomic_csv(pair_frame, pair_dir / f"pair_{pair_id:03d}.csv")
        all_rows.extend(pair_rows)
        completed_set.add(pair_id)
        newly_completed += 1
        write_partial_progress(
            out_dir,
            pair_table,
            sorted(completed_set),
            n_units=n_units,
            n_conditions_per_pair=n_conditions_per_pair,
        )
        if int(args.stop_after_new_pairs) > 0 and newly_completed >= int(args.stop_after_new_pairs):
            print(f"Stopping cleanly after {newly_completed} newly completed pair shards", flush=True)
            break
    return all_rows


def aggregate_rows(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    probe = pd.DataFrame(rows)
    if probe.empty:
        return probe, pd.DataFrame()
    numeric_cols = [
        "unit_index",
        "pair_id",
        "speed_dps",
        "log2_speed_dps",
        "spatial_cpd",
        "temporal_hz",
        "temporal_direction_sign",
        "signed_temporal_hz",
        "log2_spatial_cpd",
        "log2_temporal_hz",
        "probe_orientation_deg",
        "mean_rate",
        "blank_mean_rate",
        "signed_f0_hz",
        "positive_f0_hz",
        "response_amp_sq",
        "response_amp",
    ]
    for col in numeric_cols:
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
        "log2_spatial_cpd",
        "log2_temporal_hz",
        "cycles_across_window",
        "is_cycle_valid_sf",
        "is_extended_tf_core",
        "is_nyquist_edge_control",
        "n_spatial_cpds_for_family",
        "n_temporal_hz_for_family",
        "n_temporal_hz_for_spatial_cpd",
        "n_spatial_cpds_for_temporal_hz",
        "probe_orientation_deg",
        "temporal_direction_sign",
        "scalar_readout",
    ]
    grouped_rows: list[dict[str, Any]] = []
    for key_values, sub in probe.groupby(keys, sort=True):
        rec = dict(zip(keys, key_values, strict=True))
        amp_sq = sub["response_amp_sq"].to_numpy(dtype=float)
        amp = np.sqrt(np.nanmean(amp_sq)) if np.isfinite(amp_sq).any() else float("nan")
        signed_f0 = pd.to_numeric(sub.get("signed_f0_hz"), errors="coerce").to_numpy(dtype=float)
        positive_f0 = pd.to_numeric(sub.get("positive_f0_hz"), errors="coerce").to_numpy(dtype=float)
        blank_rate = pd.to_numeric(sub.get("blank_mean_rate"), errors="coerce").to_numpy(dtype=float)
        signed_f0_mean = float(np.nanmean(signed_f0)) if np.isfinite(signed_f0).any() else float("nan")
        signed_f0_sd = float(np.nanstd(signed_f0, ddof=1)) if np.count_nonzero(np.isfinite(signed_f0)) > 1 else 0.0
        rec.update(
            {
                "n_phases": int(sub.shape[0]),
                "phase_policies": ",".join(sorted(set(str(v) for v in sub["phase_policy"].to_list()))),
                "prior_preferred_orientation_deg": float(np.nanmean(sub["prior_preferred_orientation_deg"].to_numpy(dtype=float))),
                "prior_orientation_selectivity_index": float(
                    np.nanmean(sub["prior_orientation_selectivity_index"].to_numpy(dtype=float))
                ),
                "mean_rate": float(np.nanmean(sub["mean_rate"].to_numpy(dtype=float))),
                "blank_mean_rate": float(np.nanmean(blank_rate)) if np.isfinite(blank_rate).any() else float("nan"),
                "signed_f0_hz": signed_f0_mean,
                "positive_f0_hz": float(np.nanmean(positive_f0)) if np.isfinite(positive_f0).any() else float("nan"),
                "phase_signed_f0_sd_hz": signed_f0_sd,
                "phase_signed_f0_range_hz": float(np.nanmax(signed_f0) - np.nanmin(signed_f0))
                if np.isfinite(signed_f0).any()
                else float("nan"),
                "phase_signed_f0_cv_abs": signed_f0_sd / max(abs(signed_f0_mean), EPS)
                if np.isfinite(signed_f0_mean)
                else float("nan"),
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

    surface_rows: list[dict[str, Any]] = []
    surface_keys = [
        "unit_index",
        "unit_label",
        "pair_id",
        "speed_family",
        "speed_dps",
        "log2_speed_dps",
        "spatial_cpd",
        "temporal_hz",
        "log2_spatial_cpd",
        "log2_temporal_hz",
        "cycles_across_window",
        "is_cycle_valid_sf",
        "is_extended_tf_core",
        "is_nyquist_edge_control",
    ]
    for key_values, sub in grouped.groupby(surface_keys, sort=True):
        rec = dict(zip(surface_keys, key_values, strict=True))
        amp = pd.to_numeric(sub["response_amp_rms"], errors="coerce").to_numpy(dtype=float)
        rec.update(
            {
                "response_amp_rms_mean": float(np.nanmean(amp)),
                "response_amp_rms_median": float(np.nanmedian(amp)),
                "mean_rate": float(np.nanmean(sub["mean_rate"].to_numpy(dtype=float))),
                "blank_mean_rate": float(np.nanmean(sub["blank_mean_rate"].to_numpy(dtype=float)))
                if "blank_mean_rate" in sub
                else float("nan"),
                "signed_f0_hz_mean": float(np.nanmean(sub["signed_f0_hz"].to_numpy(dtype=float)))
                if "signed_f0_hz" in sub
                else float("nan"),
                "positive_f0_hz_mean": float(np.nanmean(sub["positive_f0_hz"].to_numpy(dtype=float)))
                if "positive_f0_hz" in sub
                else float("nan"),
                "phase_signed_f0_sd_hz_mean": float(np.nanmean(sub["phase_signed_f0_sd_hz"].to_numpy(dtype=float)))
                if "phase_signed_f0_sd_hz" in sub
                else float("nan"),
                "n_orientation_direction_rows": int(sub.shape[0]),
                "n_orientation_rows": int(sub["probe_orientation_deg"].nunique()),
                "n_orientations": int(sub["probe_orientation_deg"].nunique()),
                "n_direction_rows": int(sub["temporal_direction_sign"].nunique())
                if "temporal_direction_sign" in sub
                else 1,
                "n_directions": int(sub["temporal_direction_sign"].nunique())
                if "temporal_direction_sign" in sub
                else 1,
            }
        )
        surface_rows.append(rec)
    surface = pd.DataFrame(surface_rows)
    return grouped, surface


def plot_summary(out_dir: Path, pair_table: pd.DataFrame, surface: pd.DataFrame, plot_units: list[int], *, dpi: int) -> tuple[Path, Path]:
    png = out_dir / "backimage_rr100_dense_sf_tf_grating_probe_summary.png"
    pdf = out_dir / "backimage_rr100_dense_sf_tf_grating_probe_summary.pdf"
    families = [family for family in ["cycle_valid", "subcycle_control"] if family in set(pair_table["speed_family"])]
    fig, axes = plt.subplots(len(families), 3, figsize=(15.2, 4.3 * len(families)), squeeze=False)
    for row, family in enumerate(families):
        family_pairs = pair_table[pair_table["speed_family"].eq(family)].copy()
        ax = axes[row, 0]
        grid = family_pairs.pivot_table(
            index="temporal_hz",
            columns="spatial_cpd",
            values="pair_id",
            aggfunc="count",
            fill_value=0,
        )
        im = ax.imshow(grid.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1)
        del im
        ax.set_title(f"{family}: sampled SF/TF pairs")
        ax.set_xlabel("SF (cpd)")
        ax.set_ylabel("TF (Hz)")
        ax.set_xticks(np.arange(len(grid.columns)))
        ax.set_xticklabels([f"{float(v):g}" for v in grid.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(grid.index)))
        ax.set_yticklabels([f"{float(v):g}" for v in grid.index], fontsize=7)

        ax = axes[row, 1]
        fam_surf = surface[surface["speed_family"].eq(family)]
        chosen = [unit for unit in plot_units if unit in set(fam_surf["unit_index"])]
        if not chosen:
            chosen = [int(v) for v in fam_surf.groupby("unit_index")["response_amp_rms_mean"].max().sort_values(ascending=False).head(4).index]
        tuning_column = (
            "positive_f0_hz_mean"
            if "positive_f0_hz_mean" in fam_surf and fam_surf["positive_f0_hz_mean"].notna().any()
            else "response_amp_rms_mean"
        )
        for unit in chosen[:5]:
            sub = fam_surf[fam_surf["unit_index"].eq(int(unit))].sort_values("temporal_hz")
            # Marginalize over SF to make a compact diagnostic curve.
            marginal = sub.groupby("temporal_hz", sort=True)[tuning_column].mean().reset_index()
            vals = marginal[tuning_column].to_numpy(dtype=float)
            z = (vals - np.nanmean(vals)) / max(float(np.nanstd(vals)), EPS)
            ax.plot(marginal["temporal_hz"], z, marker="o", linewidth=1.3, label=f"u{int(unit):03d}")
        ax.set_xscale("log", base=2)
        ax.set_title("example F0 TF marginals" if tuning_column == "positive_f0_hz_mean" else "example F1 TF marginals")
        ax.set_xlabel("TF (Hz)")
        ax.set_ylabel("within-unit z")
        ax.grid(True, color="0.9")
        ax.legend(frameon=False, fontsize=7, ncol=2)

        ax = axes[row, 2]
        sf_count = int(family_pairs["spatial_cpd"].nunique())
        tf_count = int(family_pairs["temporal_hz"].nunique())
        ax.text(
            0.02,
            0.92,
            f"{sf_count} SFs x {tf_count} TFs = {family_pairs.shape[0]} pairs\n"
            f"{family_pairs['speed_dps'].min():.3g}-{family_pairs['speed_dps'].max():.3g} deg/s derived speeds\n"
            f"{surface[surface['speed_family'].eq(family)]['unit_index'].nunique()} units",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
        )
        ax.scatter(
            family_pairs["spatial_cpd"],
            family_pairs["temporal_hz"],
            c=np.log2(family_pairs["speed_dps"]),
            cmap="viridis",
            s=18,
            alpha=0.85,
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("SF (cpd)")
        ax.set_ylabel("TF (Hz)")
        ax.set_title("sampled derived speeds")
        ax.grid(True, color="0.9")
    fig.suptitle("BackImage RR100 dense Cartesian SF/TF grating probe", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
    pair_table = build_pair_table(args, sampling)
    shard_count = int(args.pair_shard_count)
    shard_index = int(args.pair_shard_index)
    if shard_count < 1:
        raise ValueError("--pair-shard-count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--pair-shard-index must be in [0, pair-shard-count)")
    pair_assignment = (
        (pair_table["sf_grid_index"] + pair_table["tf_grid_index"]) % shard_count
    ).astype(int)
    pair_shard_rule = "(sf_grid_index + tf_grid_index) modulo pair_shard_count"
    if shard_count == 2:
        # Pure parity checkerboards are disconnected bipartite designs. Flip a
        # 2x2 block: row/column degrees stay balanced, but both halves become
        # connected and can support a provisional separable SF x TF fit.
        connectivity_swap = pair_table["sf_grid_index"].lt(2) & pair_table["tf_grid_index"].lt(2)
        pair_assignment.loc[connectivity_swap] = 1 - pair_assignment.loc[connectivity_swap]
        pair_shard_rule += "; flip the sf_grid_index<2 x tf_grid_index<2 block for connectivity"
    pair_table = pair_table[pair_assignment.eq(shard_index)].copy().reset_index(drop=True)
    if int(args.max_pairs) > 0:
        pair_table = pair_table.iloc[: int(args.max_pairs)].copy().reset_index(drop=True)
    if pair_table.empty:
        raise ValueError("The requested checkerboard shard contains no SF/TF pairs")
    over_temporal = pair_table[pair_table["temporal_hz"] > float(sampling["temporal_nyquist_hz"])]
    if not over_temporal.empty:
        raise ValueError("Pair table contains temporal frequencies above Nyquist.")
    pair_table.to_csv(out_dir / "dense_sf_tf_pair_table.csv", index=False)
    identity = {
        "analysis": "backimage_rr100_dense_sf_tf_grating_probe",
        "source_dir": Path(args.source_dir).resolve(),
        "rr100_version": str(args.rr100_version),
        "stimulus_normalization": STIMULUS_NORMALIZATION,
        "model_output_conversion": (
            "canonical shared rate-map expected counts/bin multiplied by frame_rate_hz before "
            "saving mean_rate, blank_mean_rate, signed_f0_hz, or response amplitudes"
        ),
        "computed_units": "all_rr100_units",
        "plot_units": plot_units,
        "orientation_degrees": orientation_degrees,
        "cycle_valid_spatial_cpds": parse_float_list(str(args.cycle_valid_spatial_cpds)),
        "subcycle_control_spatial_cpds": parse_float_list(str(args.subcycle_control_spatial_cpds))
        if str(args.subcycle_control_spatial_cpds).strip()
        else [],
        "include_subcycle_controls": bool(args.include_subcycle_controls),
        "temporal_hz": parse_float_list(str(args.temporal_hz)),
        "max_temporal_hz": float(args.max_temporal_hz),
        "max_pairs": int(args.max_pairs),
        "pair_shard_count": shard_count,
        "pair_shard_index": shard_index,
        "pair_shard_rule": pair_shard_rule,
        "requested_global_pair_ids": pair_table["pair_id"].astype(int).tolist(),
        "pair_table_rows": int(pair_table.shape[0]),
        "scalar_readout": str(args.scalar_readout),
        "n_phases": int(args.n_phases),
        "temporal_direction_signs": sorted(set(parse_int_list(str(args.temporal_direction_signs)))),
        "direction_folding": len(set(parse_int_list(str(args.temporal_direction_signs)))) > 1,
        "include_blank_reference": bool(args.include_blank_reference),
        "f0_contract": (
            "phase-averaged mean rate minus one matched mean-gray blank, clipped at zero for positive_f0_hz"
            if bool(args.include_blank_reference)
            else "phase-averaged mean rate without a separately scored blank"
        ),
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
            "Dense Cartesian SF/TF grid for fitting 2D frequency-tuning surfaces. "
            "Cycle-valid SFs are separated from sub-cycle flicker/ramp controls; derived speeds are recorded but not used to define the grid."
        ),
    }
    write_json(out_dir / "dense_sf_tf_request_identity.json", identity)
    if bool(args.dry_run):
        print(json.dumps(json_ready(identity), indent=2, sort_keys=True))
        print(pair_table.to_string(index=False))
        return

    manifest_path = out_dir / "dense_sf_tf_manifest.json"
    probe_csv = out_dir / "dense_sf_tf_probe_rows.csv"
    grouped_csv = out_dir / "dense_sf_tf_grouped.csv"
    surface_csv = out_dir / "dense_sf_tf_unit_surface.csv"
    use_cache = False
    if all(path.exists() for path in [manifest_path, probe_csv, grouped_csv, surface_csv]) and not bool(args.force):
        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            observed = cached_manifest.get("identity_text", "")
            use_cache = (
                str(observed) == identity_text(identity)
                and cached_manifest.get("status") == "requested_pair_set_complete"
            )
        except Exception:
            use_cache = False
    if use_cache:
        probe_rows = pd.read_csv(probe_csv).to_dict("records")
        grouped = pd.read_csv(grouped_csv)
        surface = pd.read_csv(surface_csv)
        print(f"Loaded cached dense SF/TF probe rows from {probe_csv}", flush=True)
    else:
        probe_rows = compute_probe_rows(
            args,
            pair_table=pair_table,
            orientation_summary=orientation_summary,
            out_dir=out_dir,
        )
        grouped, surface = aggregate_rows(probe_rows)
        atomic_csv(pd.DataFrame(probe_rows), probe_csv)
        atomic_csv(grouped, grouped_csv)
        atomic_csv(surface, surface_csv)
    completed_pair_ids = sorted(pd.DataFrame(probe_rows)["pair_id"].astype(int).unique().tolist())
    requested_pair_ids = pair_table["pair_id"].astype(int).tolist()
    complete = set(completed_pair_ids) == set(requested_pair_ids)
    completed_pair_table = pair_table[pair_table["pair_id"].isin(completed_pair_ids)].copy()
    png, pdf = plot_summary(
        out_dir,
        completed_pair_table,
        surface,
        plot_units=plot_units,
        dpi=int(args.dpi),
    )
    write_json(
        manifest_path,
        {
            "status": "requested_pair_set_complete" if complete else "partial_resumable_analyzable",
            "identity": identity,
            "identity_text": identity_text(identity),
            "n_requested_pairs": len(requested_pair_ids),
            "n_completed_pairs": len(completed_pair_ids),
            "requested_pair_ids": requested_pair_ids,
            "completed_pair_ids": completed_pair_ids,
            "completion_fraction": len(completed_pair_ids) / max(len(requested_pair_ids), 1),
            "completed_spatial_cpds": sorted(completed_pair_table["spatial_cpd"].unique().tolist()),
            "completed_temporal_hz": sorted(completed_pair_table["temporal_hz"].unique().tolist()),
            "n_probe_rows": len(probe_rows),
            "n_grouped_rows": int(grouped.shape[0]),
            "n_surface_rows": int(surface.shape[0]),
            "outputs": {
                "pair_table_csv": out_dir / "dense_sf_tf_pair_table.csv",
                "probe_rows_csv": probe_csv,
                "grouped_csv": grouped_csv,
                "surface_csv": surface_csv,
                "summary_png": png,
                "summary_pdf": pdf,
            },
        },
    )
    print(f"Wrote dense SF/TF grating probe outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
