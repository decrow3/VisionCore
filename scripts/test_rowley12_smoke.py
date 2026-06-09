#!/usr/bin/env python3
"""Read-only smoke test for regenerated Rowley/Luke v12-style inputs.

This script intentionally does not write figures or caches.  It checks that the
session YAMLs point at existing processed_declan datasets, loads fixrsvp, aligns
trials, and reports the same coarse Pool A/B gates used by test_rowley12:

  dots RF SNR >= 5, total spikes > 200, split-half PSTH r^2 >= 0.025,
  and NaN fraction <= 0.20.

Optional JSON output is opened with mode "x" so an existing file is never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DataYatesV1 import DictDataset  # noqa: E402
from DataRowleyV1V2.dots_calibration.training import (  # noqa: E402
    bin_dots_to_stimulus,
    calculate_rf_snr,
)
from DataRowleyV1V2.utils.rf import calc_sta  # noqa: E402


SESSIONS_ROOT = ROOT / "experiments" / "dataset_configs" / "sessions"
DEFAULT_MULTI = ROOT / "experiments" / "dataset_configs" / "multi_Luke_processed_declan_120_rowley.yaml"

SNR_THRESHOLD = 5.0
TOTAL_SPIKES_THRESHOLD = 200
MIN_RELIABILITY = 0.025
MAX_UNIT_NAN_FRAC = 0.20
VALID_TIME_BINS = 240
MIN_FIX_DUR_BINS = 20
DOTS_ROI_DEG = np.array([[-5, 5], [-5, 5]], dtype=np.float32)
DOTS_DXY_DEG = 0.2
DOTS_STA_LAGS = np.arange(2, 8)


def _to_numpy(x: Any) -> np.ndarray:
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def _resolve_session_yaml(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path.resolve()
    if not path.name.endswith(".yaml"):
        path = path.with_suffix(".yaml")
    candidate = SESSIONS_ROOT / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve session YAML: {name_or_path}")


def _session_yamls_from_config(path: Path, limit: int | None = None) -> list[Path]:
    cfg = _load_yaml(path)
    sessions = cfg.get("sessions")
    if not sessions:
        raise ValueError(f"Config has no sessions list: {path}")
    out = [_resolve_session_yaml(str(session)) for session in sessions]
    return out[:limit] if limit is not None else out


def _as_bool_1d(x: np.ndarray, n_expected: int | None = None) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    if n_expected is not None and len(x) != n_expected:
        raise ValueError(f"Expected length {n_expected}, got {len(x)}")
    return (x > 0.5) if x.dtype != bool else x


def _get_optional(dset: DictDataset, key: str, default: Any) -> np.ndarray:
    if key not in dset.keys():
        return np.asarray(default)
    return _to_numpy(dset[key])


def _serial_to_trial_robs(
    robs: np.ndarray,
    dfs: np.ndarray,
    trial_inds: np.ndarray,
    time_inds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trials = np.unique(trial_inds)
    n_time = min(int(np.nanmax(time_inds)) + 1, VALID_TIME_BINS)
    n_units = robs.shape[1]
    robs_trials = np.full((len(trials), n_time, n_units), np.nan, dtype=np.float32)
    dfs_trials = np.zeros((len(trials), n_time), dtype=bool)
    dur_trials = np.zeros(len(trials), dtype=np.int64)

    trial_to_row = {int(trial): row for row, trial in enumerate(trials)}
    valid = (time_inds >= 0) & (time_inds < n_time)
    for src_idx in np.flatnonzero(valid):
        row = trial_to_row[int(trial_inds[src_idx])]
        col = int(time_inds[src_idx])
        robs_trials[row, col, :] = robs[src_idx, :]
        dfs_trials[row, col] = bool(dfs[src_idx])

    dur_trials = dfs_trials.sum(axis=1)
    return robs_trials, dfs_trials, dur_trials


def _compute_split_half_reliability(
    robs_trials: np.ndarray,
    n_splits: int,
    seed: int = 42,
    min_valid_bins: int = 10,
    min_trials_per_half: int = 2,
) -> np.ndarray:
    n_trials, _, n_units = robs_trials.shape
    rng = np.random.default_rng(seed)
    r2_accum = np.zeros(n_units, dtype=np.float64)
    r2_count = np.zeros(n_units, dtype=np.int64)
    if n_trials < 2:
        return r2_accum

    for _ in range(n_splits):
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        if half < min_trials_per_half:
            break
        idx_a = perm[:half]
        idx_b = perm[half:2 * half]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            psth_a = np.nanmean(robs_trials[idx_a], axis=0)
            psth_b = np.nanmean(robs_trials[idx_b], axis=0)

        counts_a = np.sum(np.isfinite(robs_trials[idx_a]), axis=0)
        counts_b = np.sum(np.isfinite(robs_trials[idx_b]), axis=0)

        for unit_idx in range(n_units):
            a = psth_a[:, unit_idx]
            b = psth_b[:, unit_idx]
            finite = (
                np.isfinite(a)
                & np.isfinite(b)
                & (counts_a[:, unit_idx] >= min_trials_per_half)
                & (counts_b[:, unit_idx] >= min_trials_per_half)
            )
            if finite.sum() < min_valid_bins:
                continue
            if np.std(a[finite]) <= 0 or np.std(b[finite]) <= 0:
                continue
            r2_accum[unit_idx] += np.corrcoef(a[finite], b[finite])[0, 1] ** 2
            r2_count[unit_idx] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.divide(r2_accum, r2_count, out=np.zeros_like(r2_accum), where=r2_count > 0)


def _session_root_from_dataset_dir(dataset_dir: Path) -> Path:
    for candidate in [dataset_dir, *dataset_dir.parents]:
        if (candidate / "dpi_calibration").exists():
            return candidate
    return dataset_dir.parents[1]


def _nearest_resample_bool(sample_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    sample_times = np.asarray(sample_times, dtype=np.float64)
    values = _as_bool_1d(values, len(sample_times))
    target_times = np.asarray(target_times, dtype=np.float64)
    right_idx = np.searchsorted(sample_times, target_times, side="left")
    right_idx = np.clip(right_idx, 0, len(sample_times) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(sample_times) - 1)
    choose_left = (
        np.abs(target_times - sample_times[left_idx])
        <= np.abs(sample_times[right_idx] - target_times)
    )
    return values[np.where(choose_left, left_idx, right_idx)]


def _interp_xy(sample_times: np.ndarray, xy: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    sample_times = np.asarray(sample_times, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float32)
    target_times = np.asarray(target_times, dtype=np.float64)
    return np.column_stack(
        [
            np.interp(target_times, sample_times, xy[:, 0]),
            np.interp(target_times, sample_times, xy[:, 1]),
        ]
    ).astype(np.float32)


def _compute_dots_snr_in_memory(
    session_root: Path,
    primary_eye: str,
    target_cids: np.ndarray,
) -> tuple[np.ndarray, str]:
    dots_path = session_root / "dpi_calibration" / "dots_binned_data.dset"
    eye_dir = session_root / "dpi_calibration" / f"{primary_eye}_eye"
    csv_path = eye_dir / "calibrated_dpi.csv"
    params_path = eye_dir / "calibration_params.npz"
    required = [dots_path, csv_path, params_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        return np.full(len(target_cids), np.nan, dtype=np.float32), (
            "missing compute inputs: " + ", ".join(str(path) for path in missing)
        )

    dots_dset = DictDataset.load(dots_path)
    dots_cids = np.asarray(dots_dset.metadata.get("cids", np.arange(_to_numpy(dots_dset["robs"]).shape[1])))
    dots_index = {int(cid): idx for idx, cid in enumerate(dots_cids.tolist())}
    matched = np.array([dots_index.get(int(cid), -1) for cid in target_cids], dtype=int)
    found_mask = matched >= 0
    if not found_mask.any():
        return np.full(len(target_cids), np.nan, dtype=np.float32), (
            f"no matching dots cids in {dots_path}"
        )

    params = np.load(params_path, allow_pickle=True)
    ppd = float(np.asarray(params["ppd"]).reshape(-1)[0])
    dpi_df = pd.read_csv(csv_path, usecols=["t_ephys", "i", "j", "valid"])
    sample_times = dpi_df["t_ephys"].to_numpy(dtype=np.float64)
    gaze_pix = dpi_df[["i", "j"]].to_numpy(dtype=np.float32)
    gaze_valid = dpi_df["valid"].to_numpy()
    valid_samples = (
        np.isfinite(sample_times)
        & _as_bool_1d(gaze_valid, len(sample_times))
        & np.all(np.isfinite(gaze_pix), axis=1)
    )
    if valid_samples.sum() < 2:
        return np.full(len(target_cids), np.nan, dtype=np.float32), (
            f"too few valid gaze samples in {csv_path}"
        )

    t_bins = _to_numpy(dots_dset["t_bins"]).astype(np.float64)
    dots_pix = _to_numpy(dots_dset["dots_pix"]).astype(np.float32)
    robs = _to_numpy(dots_dset["robs"]).astype(np.float32)

    gaze_interp = _interp_xy(sample_times[valid_samples], gaze_pix[valid_samples], t_bins)
    gaze_valid_interp = _nearest_resample_bool(sample_times, gaze_valid, t_bins)

    roi_pix = np.flipud(DOTS_ROI_DEG * ppd)
    dxy_pix = DOTS_DXY_DEG * ppd
    i_edges = np.arange(roi_pix[0, 0], roi_pix[0, 1] + dxy_pix, dxy_pix)
    j_edges = np.arange(roi_pix[1, 0], roi_pix[1, 1] + dxy_pix, dxy_pix)
    stim = bin_dots_to_stimulus(dots_pix, gaze_interp, i_edges, j_edges)[gaze_valid_interp]
    robs_valid = robs[gaze_valid_interp]
    if stim.shape[0] == 0:
        return np.full(len(target_cids), np.nan, dtype=np.float32), "no valid dots frames after gaze mask"

    matched_found = matched[found_mask]
    robs_matched = robs_valid[:, matched_found]
    stas = calc_sta(
        stim[..., None],
        robs_matched,
        DOTS_STA_LAGS,
        reverse_correlate=False,
        progress=False,
    ).squeeze().cpu().numpy()
    if stas.ndim == 3:
        stas = stas[None, ...]
    max_snr_subset, _, _ = calculate_rf_snr(stas, DOTS_DXY_DEG)

    max_snr = np.full(len(target_cids), np.nan, dtype=np.float32)
    max_snr[found_mask] = max_snr_subset.astype(np.float32, copy=False)
    return max_snr, f"computed in memory from {dots_path}"


def _load_dots_snr(
    dataset_dir: Path,
    primary_eye: str,
    target_cids: np.ndarray,
    compute_missing: bool,
) -> tuple[np.ndarray, str]:
    n_units = len(target_cids)
    session_root = _session_root_from_dataset_dir(dataset_dir)
    snr_path = session_root / "dpi_calibration" / f"{primary_eye}_eye" / "dots_rf_snr.npz"
    if not snr_path.exists():
        if compute_missing:
            return _compute_dots_snr_in_memory(session_root, primary_eye, target_cids)
        return np.full(n_units, np.nan, dtype=np.float32), f"missing: {snr_path}"

    data = np.load(snr_path, allow_pickle=True)
    max_snr = np.asarray(data["max_snr"], dtype=np.float32)
    if len(max_snr) != n_units:
        if compute_missing:
            return _compute_dots_snr_in_memory(session_root, primary_eye, target_cids)
        return np.full(n_units, np.nan, dtype=np.float32), (
            f"shape mismatch: {snr_path} has {len(max_snr)} values for {n_units} fixrsvp units"
        )
    return max_snr, str(snr_path)


def _cluster_ids_from_metadata(dset: DictDataset, n_units: int) -> np.ndarray:
    for key in ("cluster_ids", "cids", "all_cids"):
        value = getattr(dset, "metadata", {}).get(key)
        if value is not None and len(value) == n_units:
            return np.asarray(value)
    return np.arange(n_units)


def smoke_session(yaml_path: Path, n_splits: int, compute_dots_snr: bool) -> dict[str, Any]:
    cfg = _load_yaml(yaml_path)
    session = str(cfg.get("session", yaml_path.stem.replace("_V1", "")))
    primary_eye = str(cfg.get("eye", "unknown"))
    dataset_dir = Path(cfg["directory"])
    fix_path = dataset_dir / "fixrsvp.dset"
    gaborium_path = dataset_dir / "gaborium.dset"
    backimage_path = dataset_dir / "backimage.dset"
    sta_ste_path = dataset_dir / "gaborium_sta_ste.npy"

    result: dict[str, Any] = {
        "session": session,
        "yaml": str(yaml_path),
        "primary_eye": primary_eye,
        "dataset_dir": str(dataset_dir),
        "under_processed_declan": "/processed_declan/" in str(dataset_dir),
        "exists": {
            "fixrsvp": fix_path.exists(),
            "gaborium": gaborium_path.exists(),
            "backimage": backimage_path.exists(),
            "gaborium_sta_ste": sta_ste_path.exists(),
        },
    }
    if not fix_path.exists():
        result["error"] = f"missing fixrsvp: {fix_path}"
        return result

    dset = DictDataset.load(fix_path)
    robs = _to_numpy(dset["robs"]).astype(np.float32)
    n_bins, n_units = robs.shape
    cids = _cluster_ids_from_metadata(dset, n_units)
    result["n_bins"] = int(n_bins)
    result["n_units"] = int(n_units)
    result["cid_minmax"] = [int(np.nanmin(cids)), int(np.nanmax(cids))] if len(cids) else None
    result["keys"] = sorted(str(k) for k in dset.keys())

    dfs = _get_optional(dset, "dfs", np.ones(n_bins, dtype=bool))
    if dfs.ndim == 2:
        dfs = np.any(dfs > 0.5, axis=1)
    dfs = _as_bool_1d(dfs, n_bins)

    trial_inds = _to_numpy(dset["trial_inds"]).astype(int)
    time_inds = _to_numpy(dset["psth_inds"]).astype(int)
    robs_trials, dfs_trials, dur_trials = _serial_to_trial_robs(robs, dfs, trial_inds, time_inds)
    good_trials = dur_trials > MIN_FIX_DUR_BINS
    robs_good = robs_trials[good_trials]
    dfs_good = dfs_trials[good_trials]

    max_snr, snr_source = _load_dots_snr(dataset_dir, primary_eye, cids, compute_dots_snr)
    visual_ok = np.isfinite(max_snr) & (max_snr >= SNR_THRESHOLD)
    spikes_per_unit = np.nansum(np.where(dfs_good[:, :, None], robs_good, np.nan), axis=(0, 1))
    spikes_ok = spikes_per_unit > TOTAL_SPIKES_THRESHOLD
    reliability = _compute_split_half_reliability(robs_good, n_splits=n_splits)
    reliability_ok = reliability >= MIN_RELIABILITY

    valid_counts = max(int(dfs_good.sum()), 1)
    nan_frac = ((np.isnan(robs_good) & dfs_good[:, :, None]).sum(axis=(0, 1)) / valid_counts)
    nan_ok = nan_frac <= MAX_UNIT_NAN_FRAC

    pool_a = visual_ok & spikes_ok & reliability_ok
    pool_b = pool_a & nan_ok

    result.update(
        {
            "n_trials": int(len(dur_trials)),
            "n_good_trials": int(good_trials.sum()),
            "dots_snr_source": snr_source,
            "dots_snr_finite": int(np.isfinite(max_snr).sum()),
            "visual_snr_ge_5": int(visual_ok.sum()),
            "spikes_gt_200": int(spikes_ok.sum()),
            "reliability_ge_0p025": int(reliability_ok.sum()),
            "nan_frac_le_0p20": int(nan_ok.sum()),
            "pool_a": int(pool_a.sum()),
            "pool_b": int(pool_b.sum()),
            "pool_b_cids": [int(x) for x in cids[pool_b].tolist()],
            "median_reliability_visual_spike": float(np.nanmedian(reliability[visual_ok & spikes_ok]))
            if np.any(visual_ok & spikes_ok)
            else None,
            "median_nan_frac_pool_a": float(np.nanmedian(nan_frac[pool_a])) if np.any(pool_a) else None,
        }
    )
    return result


def print_result(result: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print(f"{result['session']} | {result['primary_eye']} | {Path(result['yaml']).name}")
    print("=" * 88)
    print(f"dataset: {result['dataset_dir']}")
    print(f"under processed_declan: {result['under_processed_declan']}")
    print(f"files: {result['exists']}")
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    print(f"robs: {result['n_bins']} bins x {result['n_units']} units")
    print(f"trials: {result['n_good_trials']} good / {result['n_trials']} total")
    print(f"dots SNR: {result['dots_snr_source']}")
    print("waterfall:")
    print(f"  dots RF SNR >= {SNR_THRESHOLD:g}:      {result['visual_snr_ge_5']} / {result['n_units']}")
    print(f"  total spikes > {TOTAL_SPIKES_THRESHOLD}:     {result['spikes_gt_200']} / {result['n_units']}")
    print(f"  reliability >= {MIN_RELIABILITY:g}:  {result['reliability_ge_0p025']} / {result['n_units']}")
    print(f"  NaN fraction <= {MAX_UNIT_NAN_FRAC:g}:   {result['nan_frac_le_0p20']} / {result['n_units']}")
    print(f"  Pool A:                 {result['pool_a']} / {result['n_units']}")
    print(f"  Pool B:                 {result['pool_b']} / {result['n_units']}")
    print(f"  Pool B cids:            {result['pool_b_cids']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_MULTI,
                        help="Multi-session YAML. Ignored when --session-yaml is provided.")
    parser.add_argument("--session-yaml", action="append",
                        help="Session YAML path/name. Can be passed more than once.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of sessions from --config.")
    parser.add_argument("--splits", type=int, default=5,
                        help="Split-half reliability draws. Keep small for smoke tests.")
    parser.add_argument("--compute-dots-snr", action="store_true",
                        help="Compute missing dots RF SNR in memory only; never writes dots_rf_snr.npz.")
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional JSON output path. Must not already exist.")
    args = parser.parse_args()

    if args.session_yaml:
        yaml_paths = [_resolve_session_yaml(path) for path in args.session_yaml]
    else:
        yaml_paths = _session_yamls_from_config(args.config, limit=args.limit)

    print(f"Read-only Rowley v12 smoke test")
    print(f"repo: {ROOT}")
    print(f"sessions: {len(yaml_paths)}")
    print(f"reliability splits: {args.splits}")

    results = []
    for yaml_path in yaml_paths:
        result = smoke_session(yaml_path, n_splits=args.splits, compute_dots_snr=args.compute_dots_snr)
        results.append(result)
        print_result(result)

    ok = [r for r in results if "error" not in r and r.get("pool_b", 0) > 0]
    errors = [r for r in results if "error" in r]
    print("\n" + "-" * 88)
    print(f"summary: {len(ok)} / {len(results)} sessions have Pool B > 0; errors={len(errors)}")
    if ok:
        total_pool_b = sum(int(r["pool_b"]) for r in ok)
        print(f"total Pool B units across passing sessions: {total_pool_b}")

    if args.json is not None:
        payload = {"results": results}
        with args.json.open("x") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote JSON: {args.json}")

    return 1 if errors else 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    raise SystemExit(main())
