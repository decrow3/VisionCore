#!/usr/bin/env python3
"""
Diagnostics for why Luke/Rowley units fall out of figure 2.

The script intentionally reuses the figure-2 loading helpers so the Rowley
CID mapping, dots RF gate, missingness gate, firing-rate gate, and split-half
PSTH R2 gate match the main figure code.

Outputs:
  gate_unit_table.tsv
  gate_summary.tsv
  reliability_curve.tsv
  timing_lag_summary.tsv
  timing_lag_unit_best.tsv
  reliability_curve.png
  timing_lag_summary.png
  gate_waterfall.png
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.config_loader import load_dataset_configs
from models.data import prepare_data
from VisionCore.covariance import align_fixrsvp_trials
from VisionCore.paths import VISIONCORE_ROOT
from ryan.fig2.compute_fig2_data import (
    DATASET_CONFIGS_PATH,
    DT,
    FIG2_DATA_TYPES,
    MIN_PSTH_R2,
    MIN_RATE_HZ,
    ROWLEY_DOTS_MIN_SPIKES,
    ROWLEY_DOTS_SNR_THRESH,
    ROWLEY_MAX_NAN_FRAC,
    _apply_rowley_inclusion,
    _is_rowley_config,
    _load_rowley_dots_quality_for_cids,
    _prepare_rowley_unit_mapping,
    _resolve_rowley_dataset_directory,
    _rowley_initial_cid_pool,
    _split_half_psth_r2,
)


DEFAULT_TRIAL_COUNTS = (20, 30, 40, 60, 80, 120, 160, 200, 256)
DEFAULT_LAGS = tuple(range(-6, 7))


@dataclass
class SessionPayload:
    session: str
    subject: str
    mode: str
    robs: np.ndarray
    cids: np.ndarray
    rate_hz: np.ndarray
    psth_r2: np.ndarray
    plotted: np.ndarray
    n_trials_total: int
    n_trials_good: int
    gate_rows: pd.DataFrame


def _parse_csv_values(text, cast=str):
    if text is None or str(text).strip() == "":
        return []
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def _loaded_cids(cfg, neuron_mask, n_neurons_total):
    return _loaded_ids(cfg, "cids", neuron_mask, n_neurons_total)


def _loaded_ids(cfg, key, neuron_mask, n_neurons_total, fallback=None):
    values = np.asarray(cfg.get(key, []), dtype=int)
    neuron_mask = np.asarray(neuron_mask, dtype=int)
    if values.size == n_neurons_total:
        return values[neuron_mask]
    if values.size > np.max(neuron_mask, initial=-1):
        return values[neuron_mask]
    if values.size == neuron_mask.size:
        return values.copy()
    if fallback is not None:
        fallback = np.asarray(fallback, dtype=int)
        if fallback.size == neuron_mask.size:
            return fallback
    return neuron_mask.copy()


def _rate_hz(robs):
    n_spikes = np.nansum(robs, axis=(0, 1))
    n_valid = np.sum(np.isfinite(robs), axis=(0, 1))
    return np.divide(
        n_spikes,
        np.maximum(n_valid, 1),
        out=np.full(robs.shape[2], np.nan, dtype=float),
        where=n_valid > 0,
    ) / DT


def _rowley_sets(cfg):
    return {
        "yaml_cids": set(map(int, cfg.get("cids", []) or [])),
        "yaml_visual": set(map(int, cfg.get("visual", []) or [])),
        "qccontam": set(map(int, cfg.get("qccontam", []) or [])),
        "sortercontam": set(map(int, cfg.get("sortercontam", []) or [])),
    }


def _build_gate_rows(
    base_cfg,
    cfg,
    mode,
    robs,
    eyepos,
    neuron_mask,
    n_neurons_total,
    rate_hz,
    psth_r2,
    final_keep=None,
):
    subject = cfg["session"].split("_")[0]
    cids = _loaded_cids(cfg, neuron_mask, n_neurons_total)
    if len(cids) != robs.shape[2]:
        cids = np.asarray(cfg.get("cids", np.arange(robs.shape[2])), dtype=int)
        if len(cids) == len(neuron_mask):
            cids = cids[np.asarray(neuron_mask, dtype=int)]
        if len(cids) != robs.shape[2]:
            cids = np.arange(robs.shape[2], dtype=int)

    rows = pd.DataFrame(
        {
            "session": cfg["session"],
            "subject": subject,
            "mode": mode,
            "cid": cids.astype(int),
            "rate_hz": rate_hz,
            "psth_r2": psth_r2,
            "rate_ok": np.isfinite(rate_hz) & (rate_hz >= MIN_RATE_HZ),
            "reliability_ok": np.isfinite(psth_r2) & (psth_r2 > MIN_PSTH_R2),
        }
    )
    rows["plotted_ok"] = rows["rate_ok"] & rows["reliability_ok"]
    rows["final_pool"] = True if final_keep is None else np.asarray(final_keep, dtype=bool)

    if _is_rowley_config(base_cfg):
        sets = _rowley_sets(base_cfg)
        for key, values in sets.items():
            rows[f"in_{key}"] = rows["cid"].map(lambda x, s=values: int(x) in s)

        dots_ids = _loaded_ids(
            cfg,
            "_rowley_dots_cids",
            neuron_mask,
            n_neurons_total,
            fallback=rows["cid"].to_numpy(),
        )
        try:
            dots_snr, dots_spikes = _load_rowley_dots_quality_for_cids(cfg, dots_ids)
        except Exception as exc:
            dots_snr = np.full(len(rows), np.nan)
            dots_spikes = np.full(len(rows), np.nan)
            rows["dots_quality_error"] = repr(exc)
        rows["dots_cid"] = dots_ids.astype(int)
        rows["dots_snr"] = dots_snr
        rows["dots_spikes"] = dots_spikes
        rows["dots_visual_ok"] = (
            np.isfinite(dots_snr)
            & (dots_snr >= ROWLEY_DOTS_SNR_THRESH)
            & np.isfinite(dots_spikes)
            & (dots_spikes >= ROWLEY_DOTS_MIN_SPIKES)
        )

        eye_valid = np.isfinite(np.sum(eyepos, axis=2))
        n_valid_bins = max(float(eye_valid.sum()), 1.0)
        nan_frac = (np.isnan(robs) & eye_valid[:, :, None]).sum(axis=(0, 1)) / n_valid_bins
        rows["nan_frac"] = nan_frac
        rows["nan_ok"] = nan_frac <= ROWLEY_MAX_NAN_FRAC

    return rows


def _prepare_cfg_for_mode(base_cfg, mode):
    cfg = dict(base_cfg)
    cfg["types"] = list(FIG2_DATA_TYPES)
    is_rowley = _is_rowley_config(cfg)
    if not is_rowley:
        return cfg

    cfg["directory"], cfg["eye"] = _resolve_rowley_dataset_directory(cfg)
    if mode == "yaml_cids":
        return cfg
    if mode != "fig2_dots":
        raise ValueError(f"Unknown Rowley mode {mode!r}")

    rowley_cids, rowley_pool_source = _rowley_initial_cid_pool(cfg)
    if rowley_cids.size == 0:
        raise RuntimeError("Rowley config has no qccontam/sortercontam/cids")
    cfg["_rowley_pool_source"] = rowley_pool_source
    mapped = _prepare_rowley_unit_mapping(cfg, rowley_cids)
    if mapped is None:
        raise RuntimeError("Could not map Rowley CIDs into fixrsvp dataset")
    rowley_cids, rowley_dots_cids = mapped
    cfg["cids"] = rowley_cids.tolist()
    cfg["_rowley_dots_cids"] = rowley_dots_cids.tolist()
    return cfg


def load_payload(base_cfg, mode, fixation_radius, n_splits, quiet=False):
    cfg = _prepare_cfg_for_mode(base_cfg, mode)
    subject = cfg["session"].split("_")[0]

    stream = io.StringIO() if quiet else None
    out_ctx = contextlib.redirect_stdout(stream) if quiet else contextlib.nullcontext()
    err_ctx = contextlib.redirect_stderr(stream) if quiet else contextlib.nullcontext()
    with out_ctx, err_ctx:
        train_data, _, cfg_loaded = prepare_data(cfg, strict=False)
    cfg = cfg_loaded

    dset_idx = train_data.get_dataset_index("fixrsvp")
    dset = train_data.dsets[dset_idx]

    align_kwargs = dict(valid_time_bins=120, min_fix_dur=20, min_total_spikes=0, min_neurons=1)
    if _is_rowley_config(cfg):
        align_kwargs.update(
            fixation_radius=fixation_radius,
            fixation_center="median_valid",
            require_dpi_valid=True,
        )

    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(dset, **align_kwargs)
    if robs is None:
        raise RuntimeError(f"align_fixrsvp_trials returned no data: {meta}")

    pre_rate = _rate_hz(robs)
    pre_r2 = _split_half_psth_r2(robs, n_splits, seed=42)
    pre_gate_rows = _build_gate_rows(
        base_cfg,
        cfg,
        mode,
        robs,
        eyepos,
        neuron_mask,
        meta.get("n_neurons_total", robs.shape[2]),
        pre_rate,
        pre_r2,
        final_keep=np.ones(robs.shape[2], dtype=bool),
    )

    if _is_rowley_config(cfg) and mode == "fig2_dots":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if quiet:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    out = _apply_rowley_inclusion(cfg, robs, eyepos, valid_mask, neuron_mask, meta)
            else:
                out = _apply_rowley_inclusion(cfg, robs, eyepos, valid_mask, neuron_mask, meta)
        robs, eyepos, valid_mask, neuron_mask, meta = out
        if robs is None:
            raise RuntimeError("Rowley fig2 dots inclusion left too few units")

    rate = _rate_hz(robs)
    psth_r2 = _split_half_psth_r2(robs, n_splits, seed=42)
    plotted = (
        np.isfinite(rate)
        & (rate >= MIN_RATE_HZ)
        & np.isfinite(psth_r2)
        & (psth_r2 > MIN_PSTH_R2)
    )
    cids = _loaded_cids(cfg, neuron_mask, meta.get("n_neurons_total", robs.shape[2]))
    if len(cids) != robs.shape[2]:
        cids = np.arange(robs.shape[2], dtype=int)

    gate_rows = _build_gate_rows(
        base_cfg,
        cfg,
        mode,
        robs,
        eyepos,
        neuron_mask,
        meta.get("n_neurons_total", robs.shape[2]),
        rate,
        psth_r2,
        final_keep=np.ones(robs.shape[2], dtype=bool),
    )
    gate_rows["stage"] = "final_pool"
    pre_gate_rows["stage"] = "pre_rowley_inclusion"
    gate_rows = pd.concat([pre_gate_rows, gate_rows], ignore_index=True, sort=False)

    return SessionPayload(
        session=cfg["session"],
        subject=subject,
        mode=mode,
        robs=robs,
        cids=np.asarray(cids, dtype=int),
        rate_hz=rate,
        psth_r2=psth_r2,
        plotted=plotted,
        n_trials_total=int(meta.get("n_trials_total", robs.shape[0])),
        n_trials_good=int(meta.get("n_trials_good", robs.shape[0])),
        gate_rows=gate_rows,
    )


def reliability_curve_rows(payload, trial_counts, n_subsamples, n_splits, seed):
    rows = []
    n_trials = payload.robs.shape[0]
    rng = np.random.default_rng(seed)
    valid_counts = sorted({int(n) for n in trial_counts if 4 <= int(n) <= n_trials})
    if n_trials >= 4 and n_trials not in valid_counts:
        valid_counts.append(n_trials)
    for n in valid_counts:
        repeats = 1 if n == n_trials else n_subsamples
        for rep in range(repeats):
            if n == n_trials:
                idx = np.arange(n_trials)
            else:
                idx = rng.choice(n_trials, size=n, replace=False)
            rel = _split_half_psth_r2(payload.robs[idx], n_splits, seed=seed + rep + n)
            finite = np.isfinite(rel)
            rows.append(
                {
                    "session": payload.session,
                    "subject": payload.subject,
                    "mode": payload.mode,
                    "n_trials_total": n_trials,
                    "n_trials_sample": n,
                    "repeat": rep,
                    "n_units": payload.robs.shape[2],
                    "median_r2": float(np.nanmedian(rel)) if finite.any() else np.nan,
                    "q25_r2": float(np.nanquantile(rel, 0.25)) if finite.any() else np.nan,
                    "q75_r2": float(np.nanquantile(rel, 0.75)) if finite.any() else np.nan,
                    "frac_rel_pass": float(np.mean(rel[finite] > MIN_PSTH_R2)) if finite.any() else np.nan,
                    "n_rel_pass": int(np.sum(finite & (rel > MIN_PSTH_R2))),
                }
            )
    return rows


def _lagged_corr_unit(a, b, count_a, count_b, lag, min_valid_bins=10, min_trials_per_half=2):
    if lag < 0:
        aa = a[:lag]
        bb = b[-lag:]
        ca = count_a[:lag]
        cb = count_b[-lag:]
    elif lag > 0:
        aa = a[lag:]
        bb = b[:-lag]
        ca = count_a[lag:]
        cb = count_b[:-lag]
    else:
        aa = a
        bb = b
        ca = count_a
        cb = count_b
    ok = (
        np.isfinite(aa)
        & np.isfinite(bb)
        & (ca >= min_trials_per_half)
        & (cb >= min_trials_per_half)
    )
    if ok.sum() < min_valid_bins:
        return np.nan
    if np.nanstd(aa[ok]) <= 0 or np.nanstd(bb[ok]) <= 0:
        return np.nan
    r = np.corrcoef(aa[ok], bb[ok])[0, 1]
    return float(r * r) if np.isfinite(r) else np.nan


def lagged_split_half_r2(robs, lags, n_splits, seed=42, min_trials_per_half=2):
    n_trials, _, n_units = robs.shape
    lags = np.asarray(lags, dtype=int)
    out = np.zeros((n_units, len(lags)), dtype=float)
    counts = np.zeros((n_units, len(lags)), dtype=int)
    rng = np.random.default_rng(seed)
    if n_trials < 2 * min_trials_per_half:
        out[:] = np.nan
        return out

    for split in range(n_splits):
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        if half < min_trials_per_half:
            break
        idx_a = perm[:half]
        idx_b = perm[half:2 * half]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            psth_a = np.nanmean(robs[idx_a], axis=0)
            psth_b = np.nanmean(robs[idx_b], axis=0)
        count_a = np.sum(np.isfinite(robs[idx_a]), axis=0)
        count_b = np.sum(np.isfinite(robs[idx_b]), axis=0)
        for unit in range(n_units):
            for li, lag in enumerate(lags):
                r2 = _lagged_corr_unit(
                    psth_a[:, unit],
                    psth_b[:, unit],
                    count_a[:, unit],
                    count_b[:, unit],
                    int(lag),
                    min_trials_per_half=min_trials_per_half,
                )
                if np.isfinite(r2):
                    out[unit, li] += r2
                    counts[unit, li] += 1
    return np.divide(out, counts, out=np.full_like(out, np.nan), where=counts > 0)


def timing_lag_rows(payload, lags, n_splits, seed):
    r2 = lagged_split_half_r2(payload.robs, lags, n_splits, seed=seed)
    rows = []
    unit_rows = []
    for li, lag in enumerate(lags):
        vals = r2[:, li]
        finite = np.isfinite(vals)
        rows.append(
            {
                "session": payload.session,
                "subject": payload.subject,
                "mode": payload.mode,
                "lag_bins": int(lag),
                "lag_ms": float(lag / 120 * 1000),
                "n_units": payload.robs.shape[2],
                "median_r2": float(np.nanmedian(vals)) if finite.any() else np.nan,
                "frac_rel_pass": float(np.mean(vals[finite] > MIN_PSTH_R2)) if finite.any() else np.nan,
                "n_rel_pass": int(np.sum(finite & (vals > MIN_PSTH_R2))),
            }
        )
    for unit, cid in enumerate(payload.cids):
        vals = r2[unit]
        if np.isfinite(vals).any():
            best_i = int(np.nanargmax(vals))
            best_lag = int(lags[best_i])
            best_r2 = float(vals[best_i])
            zero_i = int(np.where(np.asarray(lags) == 0)[0][0]) if 0 in set(lags) else None
            zero_r2 = float(vals[zero_i]) if zero_i is not None and np.isfinite(vals[zero_i]) else np.nan
        else:
            best_lag = np.nan
            best_r2 = np.nan
            zero_r2 = np.nan
        unit_rows.append(
            {
                "session": payload.session,
                "subject": payload.subject,
                "mode": payload.mode,
                "cid": int(cid),
                "best_lag_bins": best_lag,
                "best_lag_ms": float(best_lag / 120 * 1000) if np.isfinite(best_lag) else np.nan,
                "best_r2": best_r2,
                "zero_lag_r2": zero_r2,
                "delta_best_minus_zero": best_r2 - zero_r2 if np.isfinite(best_r2) and np.isfinite(zero_r2) else np.nan,
            }
        )
    return rows, unit_rows


def gate_summary(unit_table):
    rows = []
    final = unit_table[unit_table["stage"] == "final_pool"].copy()
    for (subject, session, mode), sub in final.groupby(["subject", "session", "mode"], dropna=False):
        row = {
            "subject": subject,
            "session": session,
            "mode": mode,
            "n_final_pool": int(len(sub)),
            "n_rate_ok": int(sub.get("rate_ok", pd.Series(dtype=bool)).fillna(False).sum()),
            "n_reliability_ok": int(sub.get("reliability_ok", pd.Series(dtype=bool)).fillna(False).sum()),
            "n_plotted_ok": int(sub.get("plotted_ok", pd.Series(dtype=bool)).fillna(False).sum()),
        }
        for col in [
            "in_yaml_cids",
            "in_yaml_visual",
            "in_qccontam",
            "in_sortercontam",
            "dots_visual_ok",
            "nan_ok",
        ]:
            if col in sub:
                row[f"n_{col}"] = int(sub[col].fillna(False).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(out_dir, reliability_df, timing_df, gate_df):
    out_dir.mkdir(parents=True, exist_ok=True)

    if not reliability_df.empty:
        agg = (
            reliability_df.groupby(["subject", "mode", "n_trials_sample"], dropna=False)
            .agg(frac_rel_pass=("frac_rel_pass", "median"), median_r2=("median_r2", "median"))
            .reset_index()
        )
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for (subject, mode), sub in agg.groupby(["subject", "mode"], dropna=False):
            label = f"{subject}:{mode}"
            sub = sub.sort_values("n_trials_sample")
            axs[0].plot(sub["n_trials_sample"], sub["frac_rel_pass"], marker="o", label=label)
            axs[1].plot(sub["n_trials_sample"], sub["median_r2"], marker="o", label=label)
        axs[0].axhline(0, color="0.8", lw=1)
        axs[0].set_ylabel("fraction units R2 > gate")
        axs[1].axhline(MIN_PSTH_R2, color="0.4", ls=":", lw=1)
        axs[1].set_ylabel("median split-half R2")
        for ax in axs:
            ax.set_xlabel("sampled trials")
            ax.grid(alpha=0.25)
        axs[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "reliability_curve.png", dpi=180)
        plt.close(fig)

    if not timing_df.empty:
        agg = (
            timing_df.groupby(["subject", "mode", "lag_bins"], dropna=False)
            .agg(frac_rel_pass=("frac_rel_pass", "median"), median_r2=("median_r2", "median"))
            .reset_index()
        )
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for (subject, mode), sub in agg.groupby(["subject", "mode"], dropna=False):
            label = f"{subject}:{mode}"
            sub = sub.sort_values("lag_bins")
            axs[0].plot(sub["lag_bins"], sub["frac_rel_pass"], marker="o", label=label)
            axs[1].plot(sub["lag_bins"], sub["median_r2"], marker="o", label=label)
        axs[0].set_ylabel("fraction units R2 > gate")
        axs[1].axhline(MIN_PSTH_R2, color="0.4", ls=":", lw=1)
        axs[1].set_ylabel("median lagged split-half R2")
        for ax in axs:
            ax.axvline(0, color="0.4", ls=":", lw=1)
            ax.set_xlabel("half-PSTH lag (bins at 120 Hz)")
            ax.grid(alpha=0.25)
        axs[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "timing_lag_summary.png", dpi=180)
        plt.close(fig)

    if not gate_df.empty:
        summary = gate_summary(gate_df)
        if not summary.empty:
            cols = [c for c in ["n_final_pool", "n_dots_visual_ok", "n_nan_ok", "n_reliability_ok", "n_rate_ok", "n_plotted_ok"] if c in summary]
            agg = summary.groupby(["subject", "mode"], dropna=False)[cols].sum().reset_index()
            x = np.arange(len(agg))
            fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(agg)), 4))
            width = 0.12
            for i, col in enumerate(cols):
                label = col[2:] if col.startswith("n_") else col
                ax.bar(x + (i - len(cols) / 2) * width, agg[col], width=width, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{r.subject}:{r.mode}" for r in agg.itertuples()], rotation=30, ha="right")
            ax.set_ylabel("unit count")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "gate_waterfall.png", dpi=180)
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="Luke", help="Comma-separated subject list.")
    parser.add_argument("--rowley-modes", default="fig2_dots,yaml_cids", help="Comma-separated Rowley modes.")
    parser.add_argument("--session-filter", default="", help="Substring filter applied to session name.")
    parser.add_argument("--max-sessions-per-subject", type=int, default=0, help="0 means all.")
    parser.add_argument("--fixation-radius", type=float, default=1.5)
    parser.add_argument("--trial-counts", default=",".join(map(str, DEFAULT_TRIAL_COUNTS)))
    parser.add_argument("--n-subsamples", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--lags", default=",".join(map(str, DEFAULT_LAGS)))
    parser.add_argument("--quiet-load", action="store_true")
    parser.add_argument("--out-dir", default=str(VISIONCORE_ROOT / "outputs" / "fig2_luke_inclusion_diagnostics"))
    args = parser.parse_args(argv)

    subjects = set(_parse_csv_values(args.subjects, str))
    rowley_modes = _parse_csv_values(args.rowley_modes, str)
    trial_counts = _parse_csv_values(args.trial_counts, int)
    lags = _parse_csv_values(args.lags, int)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [dict(c) for c in load_dataset_configs(str(DATASET_CONFIGS_PATH))]
    selected = []
    seen_by_subject = {}
    for cfg in configs:
        subject = str(cfg.get("session", "")).split("_")[0]
        if subject not in subjects:
            continue
        if args.session_filter and args.session_filter not in str(cfg.get("session", "")):
            continue
        if args.max_sessions_per_subject > 0:
            seen = seen_by_subject.get(subject, 0)
            if seen >= args.max_sessions_per_subject:
                continue
            seen_by_subject[subject] = seen + 1
        selected.append(cfg)

    payloads = []
    failures = []
    for cfg in selected:
        modes = rowley_modes if _is_rowley_config(cfg) else ["fig2"]
        for mode in modes:
            label = f"{cfg['session']}:{mode}"
            print(f"\n--- loading {label} ---", flush=True)
            try:
                payload = load_payload(
                    cfg,
                    mode,
                    fixation_radius=args.fixation_radius,
                    n_splits=args.n_splits,
                    quiet=args.quiet_load,
                )
                payloads.append(payload)
                print(
                    f"  trials={payload.robs.shape[0]} units={payload.robs.shape[2]} "
                    f"rel_pass={int(np.sum(payload.psth_r2 > MIN_PSTH_R2))} "
                    f"plotted={int(payload.plotted.sum())}",
                    flush=True,
                )
            except Exception as exc:
                failures.append({"session": cfg.get("session"), "mode": mode, "error": repr(exc)})
                print(f"  failed: {exc}", flush=True)

    gate_df = pd.concat([p.gate_rows for p in payloads], ignore_index=True, sort=False) if payloads else pd.DataFrame()
    gate_summary_df = gate_summary(gate_df) if not gate_df.empty else pd.DataFrame()

    reliability_rows = []
    timing_rows = []
    timing_unit_rows = []
    for pi, payload in enumerate(payloads):
        reliability_rows.extend(
            reliability_curve_rows(
                payload,
                trial_counts=trial_counts,
                n_subsamples=args.n_subsamples,
                n_splits=args.n_splits,
                seed=1000 + pi,
            )
        )
        rows, unit_rows = timing_lag_rows(
            payload,
            lags=lags,
            n_splits=args.n_splits,
            seed=2000 + pi,
        )
        timing_rows.extend(rows)
        timing_unit_rows.extend(unit_rows)

    reliability_df = pd.DataFrame(reliability_rows)
    timing_df = pd.DataFrame(timing_rows)
    timing_unit_df = pd.DataFrame(timing_unit_rows)
    failures_df = pd.DataFrame(failures)

    gate_df.to_csv(out_dir / "gate_unit_table.tsv", sep="\t", index=False)
    gate_summary_df.to_csv(out_dir / "gate_summary.tsv", sep="\t", index=False)
    reliability_df.to_csv(out_dir / "reliability_curve.tsv", sep="\t", index=False)
    timing_df.to_csv(out_dir / "timing_lag_summary.tsv", sep="\t", index=False)
    timing_unit_df.to_csv(out_dir / "timing_lag_unit_best.tsv", sep="\t", index=False)
    failures_df.to_csv(out_dir / "failures.tsv", sep="\t", index=False)
    save_plots(out_dir, reliability_df, timing_df, gate_df)

    print(f"\nWrote diagnostics to {out_dir}")
    if not gate_summary_df.empty:
        print("\nGate summary:")
        print(gate_summary_df.to_string(index=False))
    if failures:
        print("\nFailures:")
        print(failures_df.to_string(index=False))


if __name__ == "__main__":
    main()
