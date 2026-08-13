#!/usr/bin/env python3
"""Audit high-temporal-frequency power in the corrected Figure-4 retinal movies.

This checkpoint distinguishes power genuinely present in the exact 120-Hz
retinal inputs from coverage and Fourier-windowing artifacts.  It also asks
whether omitted (>32 Hz) power changes the supported-total versus
SFxTF-weighted predictor comparison.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy import signal, stats

from declan.active_sensing_movie_information.plot_rr100_kuang_input_power_checkpoint import (
    FRAME_RATE_HZ,
    SF_FIT_MAX_CPD,
    SF_FIT_MIN_CPD,
    TF_FIT_MAX_HZ,
    TF_FIT_MIN_HZ,
    radialize_power,
    spectral_decomposition,
)
from declan.active_sensing_movie_information.plot_temporal_power_shift_map_first import source_row_by_id
from declan.active_sensing_movie_information.run_backimage_real_trace_ssi_matrix_pilot import load_source_rows
from declan.fig4_active_sensing.analyze_rr100_corrected_figure4_cache import (
    MODELS,
    RUN,
    TRACE_CACHE,
    crossed_cv,
    quality_mask,
    render_with_common,
)
from declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer import _extract_patch
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _load_twin_common


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig4_active_sensing/rr100_corrected_retinal_high_tf_audit_checkpoint_17_v1"
OLD16 = ROOT / "outputs/fig4_active_sensing/rr100_all16_spectral_explainability_checkpoint_11_v1"
CORRECTED16 = ROOT / "outputs/fig4_active_sensing/rr100_corrected_figure4_cache_checkpoint_16_v1"
EPS = 1e-15


def alternate_decomposition(movie: np.ndarray, *, ppd: float, temporal: str, spatial: str) -> dict[str, np.ndarray]:
    """Same decomposition as the primary code with explicit window controls."""
    arr = np.asarray(movie, dtype=np.float64)
    residual = arr - arr.mean(axis=0, keepdims=True)
    if temporal == "hann":
        tw = np.hanning(arr.shape[0])
    elif temporal == "rectangular":
        tw = np.ones(arr.shape[0])
    else:
        raise ValueError(temporal)
    if spatial == "hann":
        sw = np.outer(np.hanning(arr.shape[1]), np.hanning(arr.shape[2]))
    elif spatial == "rectangular":
        sw = np.ones(arr.shape[1:])
    else:
        raise ValueError(spatial)
    weighted = residual * tw[:, None, None] * sw[None, :, :]
    spatial_fft = np.fft.fftshift(np.fft.fft2(weighted, axes=(1, 2)), axes=(1, 2))
    full_fft = np.fft.fft(spatial_fft, axis=0)
    tf = np.fft.fftfreq(arr.shape[0], d=1.0 / FRAME_RATE_HZ)
    keep = tf >= 0
    fy = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], d=1.0 / ppd))
    fx = np.fft.fftshift(np.fft.fftfreq(arr.shape[2], d=1.0 / ppd))
    return {
        "dynamic_power_tf_y_x": np.abs(full_fft[keep]) ** 2,
        "temporal_frequency_hz": tf[keep],
        "radial_sf_cpd": np.sqrt(fx[None, :] ** 2 + fy[:, None] ** 2),
    }


def power_metrics(decomp: dict[str, np.ndarray]) -> dict[str, float]:
    p = np.asarray(decomp["dynamic_power_tf_y_x"], float)
    tf = np.asarray(decomp["temporal_frequency_hz"], float)
    sf = np.asarray(decomp["radial_sf_cpd"], float)
    pos = tf > 0
    tf_ok = pos & (tf >= TF_FIT_MIN_HZ) & (tf <= TF_FIT_MAX_HZ)
    high_tf = tf > TF_FIT_MAX_HZ
    sf_ok = (sf >= SF_FIT_MIN_CPD) & (sf <= SF_FIT_MAX_CPD)
    total = float(p[pos].sum())
    joint = float(p[tf_ok][:, sf_ok].sum())
    tf_supported = float(p[tf_ok].sum())
    sf_supported = float(p[pos][:, sf_ok].sum())
    high = float(p[high_tf].sum())
    high_in_sf = float(p[high_tf][:, sf_ok].sum())
    return {
        "total_positive_tf_power": total,
        "joint_supported_power": joint,
        "tf_supported_all_sf_power": tf_supported,
        "sf_supported_all_tf_power": sf_supported,
        "above_32_all_sf_power": high,
        "above_32_in_fitted_sf_power": high_in_sf,
        "fraction_joint_supported": joint / max(total, EPS),
        "fraction_above_32_all_sf": high / max(total, EPS),
        "fraction_above_32_in_fitted_sf": high_in_sf / max(total, EPS),
        "fraction_of_above_32_in_fitted_sf": high_in_sf / max(high, EPS),
    }


def reconstruct_full_corrected_power() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    cache = OUT / "corrected_full_radial_power.npz"
    table_path = OUT / "corrected_full_power_metrics.csv"
    if cache.exists() and table_path.exists():
        z = np.load(cache)
        return pd.read_csv(table_path), z["annular_power"], z["sf"], z["tf"]

    summary = json.loads((RUN / "summary.json").read_text())
    observations = pd.read_csv(RUN / "retiming_population_observations.csv")
    images = pd.read_csv(RUN / "image_feature_table.csv").sort_values("image_index")
    traces = np.load(TRACE_CACHE)["trace"].astype(np.float32)
    source_rows = load_source_rows(Path(summary["source_csv"]))
    ppd = float(observations["patch_patch_ppd"].iloc[0])
    common = _load_twin_common()

    rows: list[dict[str, float | int | str]] = []
    radial_maps: list[np.ndarray] = []
    sf_grid = tf_grid = None
    for image in images.itertuples():
        source = source_row_by_id(source_rows, int(image.source_row))
        patch, _ = _extract_patch(source, canvas_cache={}, patch_size_px=540)
        for trace_index, trace in enumerate(traces):
            movie = render_with_common(np.asarray(patch, np.float32), trace, ppd=ppd, common=common)
            primary = spectral_decomposition(movie, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
            rectangular_t = alternate_decomposition(movie, ppd=ppd, temporal="rectangular", spatial="hann")
            rectangular_s = alternate_decomposition(movie, ppd=ppd, temporal="hann", spatial="rectangular")
            radial = radialize_power(primary, ppd=ppd, frame_size=movie.shape[-1])
            annular = radial["dynamic_radial_power"] * radial["spatial_mode_count"][:, None]
            if sf_grid is None:
                sf_grid = radial["sf_centers_cpd"].astype(float)
                tf_grid = primary["temporal_frequency_hz"].astype(float)
            radial_maps.append(annular.astype(np.float64))
            step_speed = np.linalg.norm(np.diff(trace.astype(float), axis=0), axis=1) * FRAME_RATE_HZ
            base = {
                "image_index": int(image.image_index),
                "trace_index": int(trace_index),
                "mean_speed_dps": float(step_speed.mean()),
                "rms_speed_dps": float(np.sqrt(np.mean(step_speed**2))),
                "p95_speed_dps": float(np.percentile(step_speed, 95)),
                "velocity_autocorr_lag1": float(np.corrcoef(np.diff(trace[:, 0]), np.roll(np.diff(trace[:, 0]), 1))[0, 1])
                if np.std(np.diff(trace[:, 0])) > 0 else np.nan,
            }
            for label, decomp in (
                ("hann_time_hann_space", primary),
                ("rect_time_hann_space", rectangular_t),
                ("hann_time_rect_space", rectangular_s),
            ):
                rows.append({**base, "analysis_window": label, **power_metrics(decomp)})
            del movie, primary, rectangular_t, rectangular_s, radial, annular
        gc.collect()
        print(f"full-spectrum audit: image {int(image.image_index) + 1}/16", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(table_path, index=False)
    np.savez_compressed(cache, annular_power=np.stack(radial_maps), sf=sf_grid, tf=tf_grid)
    return table, np.stack(radial_maps), np.asarray(sf_grid), np.asarray(tf_grid)


def cv_r2(pred: np.ndarray, base: np.ndarray, y: np.ndarray) -> float:
    return 1.0 - float(np.sum((y - pred) ** 2)) / max(float(np.sum((y - base) ** 2)), EPS)


def corrected_predictor_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics.loc[metrics.analysis_window.eq("hann_time_hann_space")].sort_values(["image_index", "trace_index"])
    if len(primary) != 512:
        raise ValueError("Corrected audit is not the complete 16 x 32 grid")
    old = np.load(CORRECTED16 / "corrected_cache_effect_and_power_arrays.npz")
    supported = np.sqrt(np.maximum(old["supported_power"].sum(axis=(1, 2)), 0)).reshape(16, 32)
    variants = {
        "supported_SF_and_TF": supported,
        "all_TF_within_fitted_SF": np.sqrt(np.maximum(primary.sf_supported_all_tf_power.to_numpy(), 0)).reshape(16, 32),
        "all_positive_TF_and_SF": np.sqrt(np.maximum(primary.total_positive_tf_power.to_numpy(), 0)).reshape(16, 32),
    }
    models = pd.read_csv(MODELS).sort_values("rr100_index").reset_index(drop=True)
    q = quality_mask(models).to_numpy(bool)
    rows = []
    for outcome, cube in (
        ("mean_rate_delta_hz", old["mean_rate_delta_hz"]),
        ("ssi_delta_bits_per_spike", old["ssi_delta_bits_per_spike"]),
    ):
        for unit in range(100):
            for name, x in variants.items():
                pred, baseline, slopes = crossed_cv(x, cube[:, :, unit], nonnegative=True)
                rows.append({
                    "outcome": outcome,
                    "rr100_index": unit,
                    "quality_cohort": bool(q[unit]),
                    "predictor": name,
                    "crossed_cv_r2": cv_r2(pred, baseline, cube[:, :, unit]),
                    "positive_slope_folds": int(np.sum(slopes > 0)),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "corrected_full_vs_supported_predictor_cv.csv", index=False)
    return out


def loo_predict(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full(y.shape, np.nan)
    base = np.full(y.shape, np.nan)
    for test in range(y.size):
        train = np.arange(y.size) != test
        xm, ym = float(x[train].mean()), float(y[train].mean())
        denom = float(np.sum((x[train] - xm) ** 2))
        slope = float(np.sum((x[train] - xm) * (y[train] - ym)) / max(denom, EPS))
        slope = max(0.0, slope)
        pred[test] = ym - slope * xm + slope * x[test]
        base[test] = ym
    return pred, base


def old16_predictor_comparison() -> pd.DataFrame:
    audit = pd.read_csv(OLD16 / "all16_original_pair_stimulus_audit.csv").sort_values("image_index")
    supported_table = pd.read_csv(OLD16 / "all16_original_pair_power_summary.csv").sort_values("image_index")
    response = pd.read_csv(OLD16 / "all16_original_pair_response_metrics_all_rr100.csv")
    models = pd.read_csv(MODELS).sort_values("rr100_index").reset_index(drop=True)
    variants = {
        "supported_SF_and_TF": np.sqrt(supported_table.total_supported_dynamic_power.to_numpy()),
        "all_TF_within_fitted_SF": np.sqrt(audit.total_positive_tf_dynamic_power.to_numpy() * audit.fraction_dynamic_power_in_sf_fitted_support.to_numpy()),
        "all_positive_TF_and_SF": np.sqrt(audit.total_positive_tf_dynamic_power.to_numpy()),
    }
    rows = []
    for unit, group in response.groupby("rr100_index"):
        y = group.sort_values("image_index").fem_delta_temporal_sd_hz.to_numpy(float)
        for name, x in variants.items():
            pred, base = loo_predict(x, y)
            rows.append({
                "rr100_index": int(unit),
                "quality_cohort": bool(quality_mask(models).iloc[int(unit)]),
                "predictor": name,
                "loo_cv_r2": cv_r2(pred, base, y),
                "oof_pearson_r": float(stats.pearsonr(y, pred).statistic),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "old16_full_vs_supported_predictor_cv.csv", index=False)
    return out


def trace_spectral_table() -> pd.DataFrame:
    traces = np.load(TRACE_CACHE)["trace"].astype(float)
    rows = []
    for trace_index, trace in enumerate(traces):
        freq, px = signal.welch(trace[:, 0], fs=FRAME_RATE_HZ, nperseg=trace.shape[0], detrend="constant")
        _, py = signal.welch(trace[:, 1], fs=FRAME_RATE_HZ, nperseg=trace.shape[0], detrend="constant")
        p = px + py
        speed = np.linalg.norm(np.diff(trace, axis=0), axis=1) * FRAME_RATE_HZ
        rows.append({
            "trace_index": trace_index,
            "position_fraction_above_32_hz": float(p[freq > 32].sum() / max(p[freq > 0].sum(), EPS)),
            "median_step_arcmin": float(np.median(speed / FRAME_RATE_HZ * 60)),
            "mean_speed_dps": float(speed.mean()),
            "rms_speed_dps": float(np.sqrt(np.mean(speed**2))),
            "fraction_steps_above_8_dps": float(np.mean(speed > 8)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "corrected_trace_spectral_quality.csv", index=False)
    return out


def smoothing_controls(metrics: pd.DataFrame, trace_spectra: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = OUT / "trace_smoothing_and_constant_velocity_controls.csv"
    selection_path = OUT / "trace_smoothing_control_selection.csv"
    if cache.exists() and selection_path.exists():
        return pd.read_csv(cache), pd.read_csv(selection_path)
    primary = metrics.loc[metrics.analysis_window.eq("hann_time_hann_space")]
    image_means = primary.groupby("image_index").fraction_above_32_all_sf.mean().sort_values()
    trace_order = trace_spectra.sort_values("position_fraction_above_32_hz")
    image_ranks = np.unique(np.round(np.linspace(0, len(image_means) - 1, 4)).astype(int))
    trace_ranks = np.unique(np.round(np.linspace(0, len(trace_order) - 1, 8)).astype(int))
    selected_images = image_means.index.to_numpy(int)[image_ranks]
    selected_traces = trace_order.trace_index.to_numpy(int)[trace_ranks]
    selection = pd.concat([
        pd.DataFrame({"selection_axis": "image", "selected_index": selected_images,
                      "criterion_value": image_means.loc[selected_images].to_numpy(),
                      "criterion": "quartiles of mean exact-movie fraction above 32 Hz"}),
        pd.DataFrame({"selection_axis": "trace", "selected_index": selected_traces,
                      "criterion_value": trace_spectra.set_index("trace_index").loc[selected_traces, "position_fraction_above_32_hz"].to_numpy(),
                      "criterion": "octiles of eye-position fraction above 32 Hz"}),
    ], ignore_index=True)
    selection.to_csv(selection_path, index=False)

    summary = json.loads((RUN / "summary.json").read_text())
    observations = pd.read_csv(RUN / "retiming_population_observations.csv")
    images = pd.read_csv(RUN / "image_feature_table.csv").set_index("image_index")
    traces = np.load(TRACE_CACHE)["trace"].astype(np.float32)
    source_rows = load_source_rows(Path(summary["source_csv"]))
    ppd = float(observations["patch_patch_ppd"].iloc[0])
    common = _load_twin_common()
    sos = signal.butter(4, 30.0, btype="lowpass", fs=FRAME_RATE_HZ, output="sos")
    rows = []
    for image_index in selected_images:
        source = source_row_by_id(source_rows, int(images.loc[image_index, "source_row"]))
        patch, _ = _extract_patch(source, canvas_cache={}, patch_size_px=540)
        patch = np.asarray(patch, np.float32)
        for trace_index in selected_traces:
            raw = traces[trace_index].astype(float)
            smooth = signal.sosfiltfilt(sos, raw, axis=0)
            raw_rms = np.sqrt(np.mean(np.sum(np.diff(raw, axis=0) ** 2, axis=1)))
            smooth_rms = np.sqrt(np.mean(np.sum(np.diff(smooth, axis=0) ** 2, axis=1)))
            matched = smooth * raw_rms / max(smooth_rms, EPS)
            raw_metric = primary.loc[
                primary.image_index.eq(image_index) & primary.trace_index.eq(trace_index),
                "fraction_above_32_all_sf",
            ].iloc[0]
            rows.append({"control_family": "paired_trace", "image_index": image_index,
                         "trace_index": trace_index, "variant": "raw_exact", "speed_dps": raw_rms * FRAME_RATE_HZ,
                         "fraction_above_32_all_sf": float(raw_metric)})
            for label, control in (("lowpass_30hz", smooth), ("lowpass_30hz_speed_matched", matched)):
                movie = render_with_common(patch, control.astype(np.float32), ppd=ppd, common=common)
                decomp = spectral_decomposition(movie, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
                rows.append({"control_family": "paired_trace", "image_index": image_index,
                             "trace_index": trace_index, "variant": label,
                             "speed_dps": float(np.sqrt(np.mean(np.sum(np.diff(control, axis=0) ** 2, axis=1))) * FRAME_RATE_HZ),
                             "fraction_above_32_all_sf": power_metrics(decomp)["fraction_above_32_all_sf"]})
        time = np.arange(32) / FRAME_RATE_HZ
        for speed in (1.5, 4.0, 6.0):
            for direction, vec in (("horizontal", np.array([1.0, 0.0])), ("vertical", np.array([0.0, 1.0]))):
                trace = (time - time.mean())[:, None] * speed * vec[None, :]
                movie = render_with_common(patch, trace.astype(np.float32), ppd=ppd, common=common)
                decomp = spectral_decomposition(movie, ppd=ppd, frame_rate_hz=FRAME_RATE_HZ)
                rows.append({"control_family": "constant_velocity", "image_index": image_index,
                             "trace_index": -1, "variant": direction, "speed_dps": speed,
                             "fraction_above_32_all_sf": power_metrics(decomp)["fraction_above_32_all_sf"]})
        print(f"smoothing control image {image_index}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out, selection


def make_figure(metrics: pd.DataFrame, radial_power: np.ndarray, sf: np.ndarray, tf: np.ndarray,
                corrected_cv: pd.DataFrame, old16_cv: pd.DataFrame) -> None:
    primary = metrics.loc[metrics.analysis_window.eq("hann_time_hann_space")].sort_values(["image_index", "trace_index"])
    mean_map = np.nanmean(radial_power, axis=0)
    mean_map = mean_map / np.nansum(mean_map)
    positive = tf > 0
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.5), constrained_layout=True)

    vals = mean_map[:, positive]
    mesh = ax[0, 0].pcolormesh(sf, tf[positive], vals.T, shading="nearest", cmap="turbo",
                               norm=LogNorm(vmin=max(np.nanpercentile(vals[vals > 0], 5), 1e-12), vmax=np.nanmax(vals)))
    ax[0, 0].axhline(32, color="white", ls="--", lw=1.5)
    ax[0, 0].axvline(SF_FIT_MAX_CPD, color="white", ls=":", lw=1.5)
    ax[0, 0].set(xscale="log", xlabel="spatial frequency (cycles/degree)", ylabel="temporal frequency (Hz)",
                 title="A  Mean spectrum of the 512 exact retinal movies")
    fig.colorbar(mesh, ax=ax[0, 0], label="fraction of positive-TF power")

    ax[0, 1].hist(primary.fraction_above_32_all_sf, bins=np.linspace(0, 1, 21), alpha=.75, label=">32 Hz")
    ax[0, 1].hist(primary.fraction_joint_supported, bins=np.linspace(0, 1, 21), alpha=.65, label="joint fitted support")
    ax[0, 1].axvline(primary.fraction_above_32_all_sf.median(), color="C0", ls="--")
    ax[0, 1].axvline(primary.fraction_joint_supported.median(), color="C1", ls="--")
    ax[0, 1].set(xlabel="fraction of positive-TF power", ylabel="movie count",
                 title="B  Coverage varies across image × trace movies")
    ax[0, 1].legend(frameon=False)

    per_trace = primary.groupby("trace_index").fraction_above_32_all_sf.mean().reset_index()
    trace_spectra = trace_spectral_table()
    per_trace = per_trace.merge(trace_spectra, on="trace_index")
    rho = stats.spearmanr(per_trace.position_fraction_above_32_hz, per_trace.fraction_above_32_all_sf).statistic
    ax[0, 2].scatter(per_trace.position_fraction_above_32_hz, per_trace.fraction_above_32_all_sf,
                     c=per_trace.trace_index, cmap="viridis")
    ax[0, 2].set(xlabel="eye-position power above 32 Hz", ylabel="retinal-movie power above 32 Hz",
                 title=f"C  Frame-scale gaze power transfers to the movie (ρ={rho:.2f})")

    windows = metrics.groupby("analysis_window").fraction_above_32_all_sf.agg(["median", "mean"]).reset_index()
    labels = ["Hann t / Hann xy", "Rect t / Hann xy", "Hann t / Rect xy"]
    order = ["hann_time_hann_space", "rect_time_hann_space", "hann_time_rect_space"]
    med = [float(windows.loc[windows.analysis_window.eq(k), "median"].iloc[0]) for k in order]
    ax[1, 0].bar(labels, med, color=["#0072B2", "#E69F00", "#009E73"])
    ax[1, 0].set(ylim=(0, 1), ylabel="median fraction above 32 Hz",
                 title="D  The result is not created by one Fourier window")
    ax[1, 0].tick_params(axis="x", rotation=15)

    supported = np.sqrt(primary.joint_supported_power.to_numpy())
    full = np.sqrt(primary.total_positive_tf_power.to_numpy())
    r = stats.spearmanr(supported, full).statistic
    ax[1, 1].scatter(supported, full, s=11, alpha=.45)
    ax[1, 1].set(xscale="log", yscale="log", xlabel="joint-supported power amplitude",
                 ylabel="full positive-TF power amplitude",
                 title=f"E  Supported power is only a proxy for full power (ρ={r:.2f})")

    q1 = corrected_cv.loc[corrected_cv.quality_cohort & corrected_cv.outcome.eq("mean_rate_delta_hz")]
    q2 = old16_cv.loc[old16_cv.quality_cohort]
    pred_order = ["supported_SF_and_TF", "all_TF_within_fitted_SF", "all_positive_TF_and_SF"]
    x = np.arange(3)
    corr_med = [q1.loc[q1.predictor.eq(k), "crossed_cv_r2"].median() for k in pred_order]
    old_med = [q2.loc[q2.predictor.eq(k), "loo_cv_r2"].median() for k in pred_order]
    ax[1, 2].plot(x, old_med, "o-", label="legacy 16-pair modulation magnitude")
    ax[1, 2].plot(x, corr_med, "o-", label="corrected 512-pair mean-rate change")
    ax[1, 2].axhline(0, color="0.5", lw=1)
    ax[1, 2].set_xticks(x, ["measured\nSF+TF", "all TF,\nmeasured SF", "all TF\nand SF"])
    ax[1, 2].set(ylabel="median held-out R²", title="F  Does omitted power rescue the global predictor?")
    ax[1, 2].legend(frameon=False, fontsize=8)

    fig.suptitle("Checkpoint 17 — Is >32-Hz power real in the corrected retinal cache?", fontsize=16, weight="bold")
    fig.savefig(OUT / "checkpoint_17_high_tf_power_and_support_audit.png", dpi=180)
    fig.savefig(OUT / "checkpoint_17_high_tf_power_and_support_audit.pdf")
    plt.close(fig)


def make_trace_control_figure(metrics: pd.DataFrame, radial_power: np.ndarray, sf: np.ndarray, tf: np.ndarray,
                              trace_spectra: pd.DataFrame, controls: pd.DataFrame) -> None:
    traces = np.load(TRACE_CACHE)["trace"].astype(float)
    primary = metrics.loc[metrics.analysis_window.eq("hann_time_hann_space")]
    cube = radial_power.reshape(16, 32, radial_power.shape[1], radial_power.shape[2])
    examples = [14, 8]
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    time_ms = np.arange(32) / FRAME_RATE_HZ * 1000
    for row, trace_index in enumerate(examples):
        trace = traces[trace_index]
        ax[row, 0].plot(time_ms, trace[:, 0] * 60, label="horizontal")
        ax[row, 0].plot(time_ms, trace[:, 1] * 60, label="vertical")
        frac = trace_spectra.set_index("trace_index").loc[trace_index, "position_fraction_above_32_hz"]
        role = "rare smooth-drift control" if trace_index == 14 else "typical frame-alternating trace"
        ax[row, 0].set(xlabel="time (ms)", ylabel="centered gaze (arcmin)",
                       title=f"{'A' if row == 0 else 'D'}  Trace {trace_index}: {role}\nposition power >32 Hz = {frac:.1%}")
        ax[row, 0].legend(frameon=False, fontsize=8)
        f, px = signal.welch(trace[:, 0], fs=FRAME_RATE_HZ, nperseg=32, detrend="constant")
        _, py = signal.welch(trace[:, 1], fs=FRAME_RATE_HZ, nperseg=32, detrend="constant")
        p = px + py
        ax[row, 1].plot(f[f > 0], p[f > 0] / p[f > 0].sum(), marker="o")
        ax[row, 1].axvline(32, color="0.4", ls="--")
        ax[row, 1].set(xlabel="eye-position temporal frequency (Hz)", ylabel="fraction per bin",
                       title=f"{'B' if row == 0 else 'E'}  Trace {trace_index} position spectrum")
        mean_map = cube[:, trace_index].mean(axis=0)
        mean_map /= mean_map.sum()
        positive = tf > 0
        vals = mean_map[:, positive]
        mesh = ax[row, 2].pcolormesh(sf, tf[positive], vals.T, shading="nearest", cmap="turbo",
                                    norm=LogNorm(vmin=max(np.percentile(vals[vals > 0], 5), 1e-12), vmax=vals.max()))
        ax[row, 2].axhline(32, color="white", ls="--")
        movie_frac = primary.loc[primary.trace_index.eq(trace_index), "fraction_above_32_all_sf"].mean()
        ax[row, 2].set(xscale="log", xlabel="spatial frequency (cycles/degree)", ylabel="temporal frequency (Hz)",
                       title=f"{'C' if row == 0 else 'F'}  Mean retinal spectrum\npower >32 Hz = {movie_frac:.1%}")
        fig.colorbar(mesh, ax=ax[row, 2], label="fraction of positive-TF power")
    fig.suptitle("Checkpoint 17b — The high-TF cache is dominated by frame-scale gaze alternation", fontsize=16, weight="bold")
    fig.savefig(OUT / "checkpoint_17b_trace_quality_examples.png", dpi=180)
    fig.savefig(OUT / "checkpoint_17b_trace_quality_examples.pdf")
    plt.close(fig)

    paired = controls.loc[controls.control_family.eq("paired_trace")]
    const = controls.loc[controls.control_family.eq("constant_velocity")]
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    order = ["raw_exact", "lowpass_30hz", "lowpass_30hz_speed_matched"]
    labels = ["raw exact", "30-Hz low-pass", "low-pass,\nRMS-speed matched"]
    wide = paired.pivot_table(index=["image_index", "trace_index"], columns="variant",
                              values="fraction_above_32_all_sf")
    for _, row in wide.iterrows():
        ax[0].plot(np.arange(3), row[order], color="0.75", alpha=.4, lw=.8)
    med = [wide[k].median() for k in order]
    ax[0].plot(np.arange(3), med, "o-", color="#D55E00", lw=2.5, label="median")
    ax[0].set_xticks(np.arange(3), labels)
    ax[0].set(ylim=(0, 1), ylabel="fraction of retinal power above 32 Hz",
              title="A  Removing >30-Hz gaze components")
    ax[0].legend(frameon=False)
    agg = const.groupby("speed_dps").fraction_above_32_all_sf.agg(["mean", "sem"]).reset_index()
    ax[1].errorbar(agg.speed_dps, agg["mean"], yerr=agg["sem"], marker="o", capsize=4)
    ax[1].set(ylim=(0, 1), xlabel="constant retinal-image speed (degrees/s)",
              ylabel="fraction of retinal power above 32 Hz",
              title="B  Smooth constant-velocity controls")
    fig.suptitle("Checkpoint 17c — High-TF power depends on both trace bandwidth and speed", fontsize=15, weight="bold")
    fig.savefig(OUT / "checkpoint_17c_smoothing_and_speed_controls.png", dpi=180)
    fig.savefig(OUT / "checkpoint_17c_smoothing_and_speed_controls.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, radial_power, sf, tf = reconstruct_full_corrected_power()
    corrected_cv = corrected_predictor_comparison(metrics)
    old16_cv = old16_predictor_comparison()
    make_figure(metrics, radial_power, sf, tf, corrected_cv, old16_cv)
    trace_spectra = trace_spectral_table()
    controls, _selection = smoothing_controls(metrics, trace_spectra)
    make_trace_control_figure(metrics, radial_power, sf, tf, trace_spectra, controls)
    primary = metrics.loc[metrics.analysis_window.eq("hann_time_hann_space")]
    models = pd.read_csv(MODELS).sort_values("rr100_index").reset_index(drop=True)
    qn = int(quality_mask(models).sum())
    summary = {
        "cache": str(RUN.resolve()),
        "n_movies": 512,
        "n_frames": 32,
        "frame_rate_hz": 120.0,
        "positive_tf_bins_hz": tf[tf > 0].tolist(),
        "measured_tf_support_hz": [TF_FIT_MIN_HZ, TF_FIT_MAX_HZ],
        "quality_units": qn,
        "median_fraction_above_32_hz": float(primary.fraction_above_32_all_sf.median()),
        "median_fraction_joint_supported": float(primary.fraction_joint_supported.median()),
        "median_fraction_above_32_within_fitted_sf": float(primary.fraction_above_32_in_fitted_sf.median()),
        "spearman_supported_vs_full_amplitude": float(stats.spearmanr(
            np.sqrt(primary.joint_supported_power), np.sqrt(primary.total_positive_tf_power)).statistic),
        "median_trace_position_fraction_above_32_hz": float(trace_spectra.position_fraction_above_32_hz.median()),
        "median_trace_step_arcmin": float(trace_spectra.median_step_arcmin.median()),
        "pooled_fraction_trace_steps_above_8_dps": float(np.mean(
            np.linalg.norm(np.diff(np.load(TRACE_CACHE)["trace"].astype(float), axis=1), axis=2) * FRAME_RATE_HZ > 8
        )),
        "spearman_trace_position_high_tf_vs_retinal_high_tf": float(stats.spearmanr(
            trace_spectra.sort_values("trace_index").position_fraction_above_32_hz,
            primary.groupby("trace_index").fraction_above_32_all_sf.mean().sort_index(),
        ).statistic),
        "smoothing_control_medians": controls.loc[controls.control_family.eq("paired_trace")].groupby("variant").fraction_above_32_all_sf.median().to_dict(),
        "constant_velocity_control_means": controls.loc[controls.control_family.eq("constant_velocity")].groupby("speed_dps").fraction_above_32_all_sf.mean().to_dict(),
        "window_control_medians": metrics.groupby("analysis_window").fraction_above_32_all_sf.median().to_dict(),
        "corrected_mean_rate_median_cv_r2": corrected_cv.loc[
            corrected_cv.quality_cohort & corrected_cv.outcome.eq("mean_rate_delta_hz")
        ].groupby("predictor").crossed_cv_r2.median().to_dict(),
        "corrected_ssi_median_cv_r2": corrected_cv.loc[
            corrected_cv.quality_cohort & corrected_cv.outcome.eq("ssi_delta_bits_per_spike")
        ].groupby("predictor").crossed_cv_r2.median().to_dict(),
        "legacy16_modulation_median_loo_cv_r2": old16_cv.loc[old16_cv.quality_cohort].groupby("predictor").loo_cv_r2.median().to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
