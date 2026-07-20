"""Compare BackImage SSI modulation for low- and high-SF-tuned RR100 units."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


DEFAULT_TUNING_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1"
)
DEFAULT_SSI_CSV = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_latest_v1/displayed_movie_instantaneous_ssi_all_units.csv"
)
DEFAULT_OUT_DIR = DEFAULT_TUNING_DIR / "sf_group_ssi_modulation"
VALUE_COL = "displayed_movie_time_resolved_ssi_bits_per_spike"
STATIC_LOG_GAUSSIAN_METRIC = "static_log_gaussian_nearest"
DYNAMIC_LOG_GAUSSIAN_METRIC = "dynamic_log_gaussian_marginal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--ssi-csv", type=Path, default=DEFAULT_SSI_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sf-metric",
        choices=(
            "dynamic_amp_weighted",
            "static_rate_weighted",
            "dynamic_peak",
            "static_peak",
            STATIC_LOG_GAUSSIAN_METRIC,
            DYNAMIC_LOG_GAUSSIAN_METRIC,
        ),
        default="dynamic_amp_weighted",
    )
    parser.add_argument("--tertile-n", type=int, default=None, help="Units per tail. Default is floor(n_units / 3).")
    parser.add_argument(
        "--low-sf-max-cpd",
        type=float,
        default=None,
        help="Optional threshold mode: label units with sf_split_metric <= this value as low_sf.",
    )
    parser.add_argument(
        "--high-sf-min-cpd",
        type=float,
        default=None,
        help="Optional threshold mode: label units with sf_split_metric >= this value as high_sf.",
    )
    parser.add_argument("--zscore-min-std", type=float, default=1e-8)
    parser.add_argument(
        "--sf-fit-n-bootstrap",
        type=int,
        default=0,
        help="Optional bootstrap resamples for the static nearest-orientation log-Gaussian SF fit QC.",
    )
    parser.add_argument("--sf-fit-seed", type=int, default=23)
    parser.add_argument(
        "--sf-fit-min-r2",
        type=float,
        default=0.35,
        help="Diagnostic reliability threshold for the static log-Gaussian SF fit.",
    )
    parser.add_argument(
        "--sf-fit-max-ci-width-octaves",
        type=float,
        default=3.0,
        help="Diagnostic reliability threshold for bootstrap CI width in octaves.",
    )
    parser.add_argument(
        "--sf-fit-uncertain-policy",
        choices=("keep", "sf_uncertain"),
        default="keep",
        help=(
            "For static_log_gaussian_nearest only: keep all fitted units in low/middle/high groups, "
            "or move failed/low-quality fits to sf_uncertain."
        ),
    )
    return parser.parse_args()


def sem(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def weighted_log2_sf(grouped: pd.DataFrame, *, dynamic: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if dynamic:
        use = grouped[pd.to_numeric(grouped["temporal_hz"], errors="coerce") > 0].copy()
        weight_col = "response_amp_rms"
        metric_name = "dynamic_amp_weighted_sf_cpd"
    else:
        use = grouped[pd.to_numeric(grouped["temporal_hz"], errors="coerce") == 0].copy()
        weight_col = "mean_rate"
        metric_name = "static_rate_weighted_sf_cpd"

    use["spatial_cpd"] = pd.to_numeric(use["spatial_cpd"], errors="coerce")
    use[weight_col] = pd.to_numeric(use[weight_col], errors="coerce").clip(lower=0.0)
    use = use[use["spatial_cpd"] > 0].copy()
    use["log2_sf"] = np.log2(use["spatial_cpd"].to_numpy(dtype=float))

    for (unit_index, unit_label), sub in use.groupby(["unit_index", "unit_label"], sort=True):
        weights = sub[weight_col].to_numpy(dtype=float)
        logs = sub["log2_sf"].to_numpy(dtype=float)
        ok = np.isfinite(weights) & np.isfinite(logs)
        total = float(np.nansum(weights[ok]))
        if total > 0:
            log_pref = float(np.nansum(weights[ok] * logs[ok]) / total)
            sf_pref = float(2.0**log_pref)
        else:
            log_pref = float("nan")
            sf_pref = float("nan")
        rows.append(
            {
                "unit_index": int(unit_index),
                "unit_label": str(unit_label),
                f"{metric_name}_log2": log_pref,
                metric_name: sf_pref,
                f"{metric_name}_weight_sum": total,
            }
        )
    return pd.DataFrame(rows)


def angle_180_distance(a: float, b: float) -> float:
    if not np.isfinite(float(a)) or not np.isfinite(float(b)):
        return float("nan")
    delta = abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)
    return float(delta)


def log_gaussian_curve(log2_sf: np.ndarray, baseline: float, amplitude: float, mu: float, sigma_oct: float) -> np.ndarray:
    sigma = max(float(sigma_oct), 1e-6)
    return float(baseline) + float(amplitude) * np.exp(-0.5 * ((np.asarray(log2_sf, dtype=float) - float(mu)) / sigma) ** 2)


def fit_static_log_gaussian_curve(spatial_cpd: np.ndarray, responses: np.ndarray) -> dict[str, float | bool | str]:
    sf = np.asarray(spatial_cpd, dtype=float)
    y = np.asarray(responses, dtype=float)
    ok = np.isfinite(sf) & np.isfinite(y) & (sf > 0)
    sf = sf[ok]
    y = y[ok]
    order = np.argsort(sf)
    sf = sf[order]
    y = y[order]
    if sf.size < 4:
        return {
            "fit_ok": False,
            "fit_failure_reason": "fewer_than_four_sf_points",
            "preferred_sf_cpd": float("nan"),
            "preferred_log2_sf": float("nan"),
            "baseline": float("nan"),
            "amplitude": float("nan"),
            "sigma_octaves": float("nan"),
            "fwhm_octaves": float("nan"),
            "r2": float("nan"),
            "observed_peak_sf_cpd": float("nan"),
            "observed_peak_is_boundary": False,
            "fit_peak_is_boundary": False,
        }

    x = np.log2(sf)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    spread = float(y_max - y_min)
    peak_idx = int(np.nanargmax(y))
    observed_peak_sf = float(sf[peak_idx])
    observed_peak_is_boundary = bool(peak_idx == 0 or peak_idx == sf.size - 1)
    if spread <= max(1e-9, 1e-6 * abs(float(np.nanmean(y)))):
        return {
            "fit_ok": False,
            "fit_failure_reason": "flat_static_response_curve",
            "preferred_sf_cpd": observed_peak_sf,
            "preferred_log2_sf": float(x[peak_idx]),
            "baseline": y_min,
            "amplitude": 0.0,
            "sigma_octaves": float("nan"),
            "fwhm_octaves": float("nan"),
            "r2": float("nan"),
            "observed_peak_sf_cpd": observed_peak_sf,
            "observed_peak_is_boundary": observed_peak_is_boundary,
            "fit_peak_is_boundary": observed_peak_is_boundary,
        }

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    baseline_low = min(0.0, y_min - 2.0 * spread)
    baseline_high = y_max
    amp_high = max(4.0 * spread, y_max - baseline_low, 1e-6)
    bounds = (
        np.asarray([baseline_low, 0.0, x_min, 0.25], dtype=float),
        np.asarray([baseline_high, amp_high, x_max, max(5.0, x_max - x_min)], dtype=float),
    )

    def residual(params: np.ndarray) -> np.ndarray:
        return log_gaussian_curve(x, params[0], params[1], params[2], params[3]) - y

    candidates: list[tuple[float, np.ndarray]] = []
    for sigma0 in (0.75, 1.25, 2.0, 3.0):
        p0 = np.asarray([y_min, max(spread, 1e-6), float(x[peak_idx]), float(sigma0)], dtype=float)
        p0 = np.minimum(np.maximum(p0, bounds[0] + 1e-9), bounds[1] - 1e-9)
        try:
            fit = least_squares(residual, p0, bounds=bounds, max_nfev=3000)
        except Exception:
            continue
        if not fit.success or not np.all(np.isfinite(fit.x)):
            continue
        sse = float(np.nansum(np.square(residual(fit.x))))
        candidates.append((sse, fit.x))

    if not candidates:
        return {
            "fit_ok": False,
            "fit_failure_reason": "optimizer_failed",
            "preferred_sf_cpd": observed_peak_sf,
            "preferred_log2_sf": float(x[peak_idx]),
            "baseline": float("nan"),
            "amplitude": float("nan"),
            "sigma_octaves": float("nan"),
            "fwhm_octaves": float("nan"),
            "r2": float("nan"),
            "observed_peak_sf_cpd": observed_peak_sf,
            "observed_peak_is_boundary": observed_peak_is_boundary,
            "fit_peak_is_boundary": observed_peak_is_boundary,
        }

    sse, params = min(candidates, key=lambda item: item[0])
    sst = float(np.nansum(np.square(y - float(np.nanmean(y)))))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    baseline, amplitude, mu, sigma_oct = [float(v) for v in params]
    boundary_margin_octaves = 0.25
    fit_peak_is_boundary = bool(mu <= x_min + boundary_margin_octaves or mu >= x_max - boundary_margin_octaves)
    return {
        "fit_ok": True,
        "fit_failure_reason": "",
        "preferred_sf_cpd": float(2.0**mu),
        "preferred_log2_sf": mu,
        "baseline": baseline,
        "amplitude": amplitude,
        "sigma_octaves": sigma_oct,
        "fwhm_octaves": float(2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_oct),
        "r2": r2,
        "observed_peak_sf_cpd": observed_peak_sf,
        "observed_peak_is_boundary": observed_peak_is_boundary,
        "fit_peak_is_boundary": fit_peak_is_boundary,
    }


def bootstrap_static_log_gaussian_fit(
    probe_rows: pd.DataFrame,
    *,
    unit_index: int,
    nearest_orientation_deg: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    if probe_rows.empty or int(n_bootstrap) <= 0:
        return {
            "bootstrap_successes": 0,
            "preferred_sf_ci_low_cpd": float("nan"),
            "preferred_sf_ci_high_cpd": float("nan"),
            "preferred_sf_ci_width_octaves": float("nan"),
        }
    probe = probe_rows.copy()
    probe["unit_index"] = pd.to_numeric(probe["unit_index"], errors="coerce")
    probe["temporal_hz"] = pd.to_numeric(probe["temporal_hz"], errors="coerce")
    probe["probe_orientation_deg"] = pd.to_numeric(probe["probe_orientation_deg"], errors="coerce")
    sub = probe_rows[
        np.isclose(probe["unit_index"].to_numpy(dtype=float), float(unit_index))
        & np.isclose(probe["temporal_hz"].to_numpy(dtype=float), 0.0)
        & np.isclose(
            probe["probe_orientation_deg"].to_numpy(dtype=float),
            float(nearest_orientation_deg),
        )
    ].copy()
    if sub.empty:
        return {
            "bootstrap_successes": 0,
            "preferred_sf_ci_low_cpd": float("nan"),
            "preferred_sf_ci_high_cpd": float("nan"),
            "preferred_sf_ci_width_octaves": float("nan"),
        }
    sub["spatial_cpd"] = pd.to_numeric(sub["spatial_cpd"], errors="coerce")
    sub["mean_rate"] = pd.to_numeric(sub["mean_rate"], errors="coerce")
    by_sf = [
        (float(sf), sf_sub["mean_rate"].dropna().to_numpy(dtype=float))
        for sf, sf_sub in sub.groupby("spatial_cpd", sort=True)
        if np.isfinite(float(sf)) and sf_sub["mean_rate"].dropna().shape[0] > 0
    ]
    if len(by_sf) < 4:
        return {
            "bootstrap_successes": 0,
            "preferred_sf_ci_low_cpd": float("nan"),
            "preferred_sf_ci_high_cpd": float("nan"),
            "preferred_sf_ci_width_octaves": float("nan"),
        }
    rng = np.random.default_rng(int(seed) + int(unit_index))
    boot_prefs: list[float] = []
    sf_values = np.asarray([sf for sf, _ in by_sf], dtype=float)
    for _ in range(int(n_bootstrap)):
        y_values = []
        for _, vals in by_sf:
            sample = rng.choice(vals, size=vals.size, replace=True)
            y_values.append(float(np.nanmean(sample)))
        fit = fit_static_log_gaussian_curve(sf_values, np.asarray(y_values, dtype=float))
        pref = float(fit["preferred_sf_cpd"])
        if bool(fit["fit_ok"]) and np.isfinite(pref) and pref > 0:
            boot_prefs.append(pref)
    if len(boot_prefs) < max(10, int(0.25 * int(n_bootstrap))):
        return {
            "bootstrap_successes": int(len(boot_prefs)),
            "preferred_sf_ci_low_cpd": float("nan"),
            "preferred_sf_ci_high_cpd": float("nan"),
            "preferred_sf_ci_width_octaves": float("nan"),
        }
    prefs = np.asarray(boot_prefs, dtype=float)
    lo = float(np.nanpercentile(prefs, 2.5))
    hi = float(np.nanpercentile(prefs, 97.5))
    width = float(np.log2(hi) - np.log2(lo)) if lo > 0 and hi > 0 else float("nan")
    return {
        "bootstrap_successes": int(len(boot_prefs)),
        "preferred_sf_ci_low_cpd": lo,
        "preferred_sf_ci_high_cpd": hi,
        "preferred_sf_ci_width_octaves": width,
    }


def static_log_gaussian_nearest_orientation_fits(
    tuning_dir: Path,
    summary: pd.DataFrame,
    grouped: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    min_r2: float,
    max_ci_width_octaves: float,
) -> pd.DataFrame:
    use = grouped[np.isclose(pd.to_numeric(grouped["temporal_hz"], errors="coerce").to_numpy(dtype=float), 0.0)].copy()
    use["unit_index"] = pd.to_numeric(use["unit_index"], errors="coerce").astype("Int64")
    use["probe_orientation_deg"] = pd.to_numeric(use["probe_orientation_deg"], errors="coerce")
    use["spatial_cpd"] = pd.to_numeric(use["spatial_cpd"], errors="coerce")
    use["mean_rate"] = pd.to_numeric(use["mean_rate"], errors="coerce")
    probe_path = Path(tuning_dir) / "frequency_tuning_probe_rows.csv"
    probe_rows = pd.read_csv(probe_path) if probe_path.exists() and int(n_bootstrap) > 0 else pd.DataFrame()

    summary_numeric = summary.copy()
    summary_numeric["unit_index"] = pd.to_numeric(summary_numeric["unit_index"], errors="coerce").astype("Int64")
    summary_numeric["prior_preferred_orientation_deg"] = pd.to_numeric(
        summary_numeric["prior_preferred_orientation_deg"], errors="coerce"
    )
    rows: list[dict[str, object]] = []
    for _, unit_row in summary_numeric.sort_values("unit_index").iterrows():
        unit_index = int(unit_row["unit_index"])
        unit_label = str(unit_row.get("unit_label", f"u{unit_index:03d}"))
        unit_static = use[use["unit_index"] == unit_index].copy()
        orientations = sorted(float(v) for v in unit_static["probe_orientation_deg"].dropna().unique())
        prior_ori = float(unit_row["prior_preferred_orientation_deg"])
        if not orientations:
            continue
        if np.isfinite(prior_ori):
            nearest_ori = min(orientations, key=lambda ori: angle_180_distance(ori, prior_ori))
            orientation_delta = angle_180_distance(nearest_ori, prior_ori)
            orientation_selection_mode = "nearest_prior_preferred_orientation"
        else:
            ori_means = unit_static.groupby("probe_orientation_deg", sort=True)["mean_rate"].mean()
            nearest_ori = float(ori_means.sort_values(ascending=False).index[0])
            orientation_delta = float("nan")
            orientation_selection_mode = "fallback_peak_static_orientation"
        curve = (
            unit_static[np.isclose(unit_static["probe_orientation_deg"].to_numpy(dtype=float), nearest_ori)]
            .groupby("spatial_cpd", sort=True)["mean_rate"]
            .mean()
            .reset_index()
        )
        fit = fit_static_log_gaussian_curve(
            curve["spatial_cpd"].to_numpy(dtype=float),
            curve["mean_rate"].to_numpy(dtype=float),
        )
        boot = bootstrap_static_log_gaussian_fit(
            probe_rows,
            unit_index=unit_index,
            nearest_orientation_deg=nearest_ori,
            n_bootstrap=int(n_bootstrap),
            seed=int(seed),
        )
        fit_r2 = float(fit["r2"])
        ci_width = float(boot["preferred_sf_ci_width_octaves"])
        is_reliable = (
            bool(fit["fit_ok"])
            and np.isfinite(fit_r2)
            and fit_r2 >= float(min_r2)
            and not bool(fit["fit_peak_is_boundary"])
            and not bool(fit["observed_peak_is_boundary"])
            and (not np.isfinite(ci_width) or ci_width <= float(max_ci_width_octaves))
        )
        rows.append(
            {
                "unit_index": unit_index,
                "unit_label": unit_label,
                "static_log_gaussian_nearest_orientation_deg": float(nearest_ori),
                "static_log_gaussian_orientation_delta_from_prior_deg": orientation_delta,
                "static_log_gaussian_orientation_selection_mode": orientation_selection_mode,
                "static_log_gaussian_nearest_sf_cpd": float(fit["preferred_sf_cpd"]),
                "static_log_gaussian_nearest_log2_sf": float(fit["preferred_log2_sf"]),
                "static_log_gaussian_baseline": float(fit["baseline"]),
                "static_log_gaussian_amplitude": float(fit["amplitude"]),
                "static_log_gaussian_sigma_octaves": float(fit["sigma_octaves"]),
                "static_log_gaussian_fwhm_octaves": float(fit["fwhm_octaves"]),
                "static_log_gaussian_r2": fit_r2,
                "static_log_gaussian_fit_ok": bool(fit["fit_ok"]),
                "static_log_gaussian_fit_failure_reason": str(fit["fit_failure_reason"]),
                "static_log_gaussian_observed_peak_sf_cpd": float(fit["observed_peak_sf_cpd"]),
                "static_log_gaussian_observed_peak_is_boundary": bool(fit["observed_peak_is_boundary"]),
                "static_log_gaussian_fit_peak_is_boundary": bool(fit["fit_peak_is_boundary"]),
                "static_log_gaussian_bootstrap_successes": int(boot["bootstrap_successes"]),
                "static_log_gaussian_sf_ci_low_cpd": float(boot["preferred_sf_ci_low_cpd"]),
                "static_log_gaussian_sf_ci_high_cpd": float(boot["preferred_sf_ci_high_cpd"]),
                "static_log_gaussian_sf_ci_width_octaves": ci_width,
                "static_log_gaussian_is_reliable": bool(is_reliable),
                "static_log_gaussian_reliability_contract": (
                    "static TF=0 response; SF tuning fit as baseline + amplitude * Gaussian(log2 SF); "
                    "orientation is nearest the prior preferred orientation; reliability requires fit_ok, "
                    f"R2 >= {float(min_r2):g}, non-boundary observed/fitted peak, and bootstrap CI width <= "
                    f"{float(max_ci_width_octaves):g} octaves when available"
                ),
            }
        )
    return pd.DataFrame(rows)


def dynamic_log_gaussian_marginal_fits(tuning_dir: Path, grouped: pd.DataFrame) -> pd.DataFrame:
    """Fit dynamic amplitude as a log-Gaussian SF curve after marginalizing orientation and TF."""
    use = grouped[pd.to_numeric(grouped["temporal_hz"], errors="coerce") > 0].copy()
    use["unit_index"] = pd.to_numeric(use["unit_index"], errors="coerce").astype("Int64")
    use["spatial_cpd"] = pd.to_numeric(use["spatial_cpd"], errors="coerce")
    use["response_amp_rms"] = pd.to_numeric(use["response_amp_rms"], errors="coerce").clip(lower=0.0)
    use = use[np.isfinite(use["spatial_cpd"].to_numpy(dtype=float)) & (use["spatial_cpd"].to_numpy(dtype=float) > 0)].copy()

    sampling = dynamic_sf_sampling_note(tuning_dir, use)
    rows: list[dict[str, object]] = []
    for (unit_index, unit_label), sub in use.groupby(["unit_index", "unit_label"], sort=True):
        curve = sub.groupby("spatial_cpd", sort=True)["response_amp_rms"].mean().reset_index()
        fit = fit_static_log_gaussian_curve(
            curve["spatial_cpd"].to_numpy(dtype=float),
            curve["response_amp_rms"].to_numpy(dtype=float),
        )
        amp_by_sf = sub.groupby("spatial_cpd", sort=True)["response_amp_rms"].sum()
        total_amp = float(amp_by_sf.sum())
        fov_deg = float(sampling["dynamic_sf_probe_fov_deg"])
        low_subcycle_sf_amp_share = float(
            amp_by_sf.loc[[sf for sf in amp_by_sf.index if np.isfinite(fov_deg) and float(sf) * fov_deg < 1.0]].sum()
            / total_amp
        ) if total_amp > 0 else float("nan")
        rows.append(
            {
                "unit_index": int(unit_index),
                "unit_label": str(unit_label),
                "dynamic_log_gaussian_marginal_sf_cpd": float(fit["preferred_sf_cpd"]),
                "dynamic_log_gaussian_marginal_log2_sf": float(fit["preferred_log2_sf"]),
                "dynamic_log_gaussian_marginal_baseline": float(fit["baseline"]),
                "dynamic_log_gaussian_marginal_amplitude": float(fit["amplitude"]),
                "dynamic_log_gaussian_marginal_sigma_octaves": float(fit["sigma_octaves"]),
                "dynamic_log_gaussian_marginal_fwhm_octaves": float(fit["fwhm_octaves"]),
                "dynamic_log_gaussian_marginal_r2": float(fit["r2"]),
                "dynamic_log_gaussian_marginal_fit_ok": bool(fit["fit_ok"]),
                "dynamic_log_gaussian_marginal_fit_failure_reason": str(fit["fit_failure_reason"]),
                "dynamic_log_gaussian_marginal_observed_peak_sf_cpd": float(fit["observed_peak_sf_cpd"]),
                "dynamic_log_gaussian_marginal_observed_peak_is_boundary": bool(fit["observed_peak_is_boundary"]),
                "dynamic_log_gaussian_marginal_fit_peak_is_boundary": bool(fit["fit_peak_is_boundary"]),
                "dynamic_log_gaussian_marginal_low_subcycle_amp_share": low_subcycle_sf_amp_share,
                "dynamic_log_gaussian_marginal_contract": (
                    "dynamic TF>0 response_amp_rms averaged across orientation and temporal frequency at each SF; "
                    "SF preference is the peak of baseline + amplitude * Gaussian(log2 SF)."
                ),
                **sampling,
            }
        )
    return pd.DataFrame(rows)


def dynamic_sf_sampling_note(tuning_dir: Path, grouped: pd.DataFrame) -> dict[str, float | str]:
    identity_path = Path(tuning_dir) / "frequency_tuning_request_identity.json"
    fov_deg = float("nan")
    one_cycle_cpd = float("nan")
    if identity_path.exists():
        try:
            import json

            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            sampling = identity.get("stimulus_sampling", {})
            fov_deg = float(sampling.get("fov_deg", float("nan")))
            one_cycle_cpd = float(sampling.get("one_cycle_across_window_cpd", float("nan")))
        except Exception:
            fov_deg = float("nan")
            one_cycle_cpd = float("nan")
    sf_values = pd.to_numeric(grouped.get("spatial_cpd", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    if not np.isfinite(fov_deg) or fov_deg <= 0:
        fov_deg = float("nan")
    cycles = sf_values * fov_deg if np.isfinite(fov_deg) else np.full_like(sf_values, np.nan, dtype=float)
    min_cycles = float(np.nanmin(cycles)) if cycles.size and np.isfinite(cycles).any() else float("nan")
    max_subcycle_sf = float(np.nanmax(sf_values[cycles < 1.0])) if cycles.size and np.any(cycles < 1.0) else float("nan")
    return {
        "dynamic_sf_probe_fov_deg": fov_deg,
        "dynamic_sf_probe_one_cycle_cpd": one_cycle_cpd,
        "dynamic_sf_probe_min_cycles_across_window": min_cycles,
        "dynamic_sf_probe_max_subcycle_cpd": max_subcycle_sf,
        "dynamic_sf_probe_sampling_note": (
            "SF bins below one_cycle_cpd are sub-cycle across the 101 px grating window and can behave like "
            "global luminance/ramp or flicker conditions in the dynamic probe."
        ),
    }


def label_sf_groups(units: pd.DataFrame, *, low_sf_max_cpd: float | None, high_sf_min_cpd: float | None) -> pd.DataFrame:
    units = units.copy()
    n_units = int(len(units))
    counts = units["sf_group"].value_counts().to_dict()
    use_thresholds = low_sf_max_cpd is not None or high_sf_min_cpd is not None
    if use_thresholds:
        low_thr = float(low_sf_max_cpd)
        high_thr = float(high_sf_min_cpd)
        definition = f"threshold: low_sf <= {low_thr:g} cpd; high_sf >= {high_thr:g} cpd; middle otherwise"
        labels = {
            "low_sf": f"low SF <= {low_thr:g} cpd (n={int(counts.get('low_sf', 0))})",
            "middle_sf": f"middle SF (n={int(counts.get('middle_sf', 0))})",
            "high_sf": f"high SF >= {high_thr:g} cpd (n={int(counts.get('high_sf', 0))})",
            "sf_uncertain": f"SF uncertain (n={int(counts.get('sf_uncertain', 0))})",
        }
    else:
        n_tail = int(counts.get("low_sf", 0))
        definition = f"tertile tails: n_tail={n_tail}"
        labels = {
            "low_sf": f"low SF bottom third (n={n_tail})",
            "middle_sf": f"middle SF (n={int(counts.get('middle_sf', n_units - 2 * n_tail))})",
            "high_sf": f"high SF top third (n={int(counts.get('high_sf', n_tail))})",
            "sf_uncertain": f"SF uncertain (n={int(counts.get('sf_uncertain', 0))})",
        }
    units["sf_group_definition"] = definition
    units["sf_group_label"] = units["sf_group"].map(labels).fillna(units["sf_group"])
    return units


def build_sf_groups(
    tuning_dir: Path,
    sf_metric: str,
    tertile_n: int | None,
    *,
    low_sf_max_cpd: float | None = None,
    high_sf_min_cpd: float | None = None,
    sf_fit_n_bootstrap: int = 0,
    sf_fit_seed: int = 23,
    sf_fit_min_r2: float = 0.35,
    sf_fit_max_ci_width_octaves: float = 3.0,
    sf_fit_uncertain_policy: str = "keep",
) -> pd.DataFrame:
    summary = pd.read_csv(tuning_dir / "frequency_tuning_summary.csv")
    grouped = pd.read_csv(tuning_dir / "frequency_tuning_grouped.csv")
    dynamic_weighted = weighted_log2_sf(grouped, dynamic=True)
    static_weighted = weighted_log2_sf(grouped, dynamic=False)
    units = summary.merge(dynamic_weighted, on=["unit_index", "unit_label"], how="left", validate="one_to_one")
    units = units.merge(static_weighted, on=["unit_index", "unit_label"], how="left", validate="one_to_one")
    if sf_metric == DYNAMIC_LOG_GAUSSIAN_METRIC:
        fitted = dynamic_log_gaussian_marginal_fits(tuning_dir, grouped)
        units = units.merge(fitted, on=["unit_index", "unit_label"], how="left", validate="one_to_one")
    if sf_metric == STATIC_LOG_GAUSSIAN_METRIC:
        fitted = static_log_gaussian_nearest_orientation_fits(
            tuning_dir,
            summary,
            grouped,
            n_bootstrap=int(sf_fit_n_bootstrap),
            seed=int(sf_fit_seed),
            min_r2=float(sf_fit_min_r2),
            max_ci_width_octaves=float(sf_fit_max_ci_width_octaves),
        )
        units = units.merge(fitted, on=["unit_index", "unit_label"], how="left", validate="one_to_one")

    metric_col = {
        "dynamic_amp_weighted": "dynamic_amp_weighted_sf_cpd",
        "static_rate_weighted": "static_rate_weighted_sf_cpd",
        "dynamic_peak": "dynamic_peak_spatial_cpd_by_amp",
        "static_peak": "static_peak_spatial_cpd_by_mean_rate",
        STATIC_LOG_GAUSSIAN_METRIC: "static_log_gaussian_nearest_sf_cpd",
        DYNAMIC_LOG_GAUSSIAN_METRIC: "dynamic_log_gaussian_marginal_sf_cpd",
    }[sf_metric]
    units["sf_split_metric"] = pd.to_numeric(units[metric_col], errors="coerce")
    units = units[np.isfinite(units["sf_split_metric"])].copy()
    units = units.sort_values(["sf_split_metric", "unit_index"], ascending=[True, True]).reset_index(drop=True)

    n_units = int(len(units))
    units["sf_rank_low_to_high"] = np.arange(1, n_units + 1)
    units["sf_group"] = "middle_sf"

    use_thresholds = low_sf_max_cpd is not None or high_sf_min_cpd is not None
    if use_thresholds:
        if low_sf_max_cpd is None or high_sf_min_cpd is None:
            raise ValueError("Threshold mode requires both --low-sf-max-cpd and --high-sf-min-cpd.")
        low_thr = float(low_sf_max_cpd)
        high_thr = float(high_sf_min_cpd)
        if not np.isfinite(low_thr) or not np.isfinite(high_thr) or low_thr >= high_thr:
            raise ValueError(f"Invalid SF thresholds low={low_thr}, high={high_thr}.")
        units.loc[units["sf_split_metric"].to_numpy(dtype=float) <= low_thr, "sf_group"] = "low_sf"
        units.loc[units["sf_split_metric"].to_numpy(dtype=float) >= high_thr, "sf_group"] = "high_sf"
    else:
        n_tail = int(tertile_n) if tertile_n is not None else n_units // 3
        if n_tail <= 0 or 2 * n_tail > n_units:
            raise ValueError(f"Invalid tertile-n {n_tail} for {n_units} units.")
        units.loc[units.index < n_tail, "sf_group"] = "low_sf"
        units.loc[units.index >= n_units - n_tail, "sf_group"] = "high_sf"
    if sf_metric == STATIC_LOG_GAUSSIAN_METRIC and str(sf_fit_uncertain_policy) == "sf_uncertain":
        reliable = units.get("static_log_gaussian_is_reliable", pd.Series(False, index=units.index)).astype(bool)
        units.loc[~reliable, "sf_group"] = "sf_uncertain"
    units = label_sf_groups(units, low_sf_max_cpd=low_sf_max_cpd, high_sf_min_cpd=high_sf_min_cpd)
    units["sf_split_metric_name"] = sf_metric
    units["sf_split_metric_column"] = metric_col
    return units


def add_curve_metrics(ssi: pd.DataFrame, units: pd.DataFrame, zscore_min_std: float) -> pd.DataFrame:
    merge_cols = [
        "unit_index",
        "unit_label",
        "sf_group",
        "sf_group_label",
        "sf_rank_low_to_high",
        "sf_split_metric",
        "sf_split_metric_name",
        "dynamic_amp_weighted_sf_cpd",
        "static_rate_weighted_sf_cpd",
        "dynamic_peak_spatial_cpd_by_amp",
        "static_peak_spatial_cpd_by_mean_rate",
        "dynamic_peak_temporal_hz_by_amp",
        "prior_preferred_orientation_deg",
        "prior_orientation_selectivity_index",
    ]
    for col in [
        "static_log_gaussian_nearest_sf_cpd",
        "static_log_gaussian_nearest_orientation_deg",
        "static_log_gaussian_r2",
        "static_log_gaussian_fit_ok",
        "static_log_gaussian_observed_peak_is_boundary",
        "static_log_gaussian_fit_peak_is_boundary",
        "static_log_gaussian_sf_ci_width_octaves",
        "static_log_gaussian_is_reliable",
        "dynamic_log_gaussian_marginal_sf_cpd",
        "dynamic_log_gaussian_marginal_r2",
        "dynamic_log_gaussian_marginal_fit_ok",
        "dynamic_log_gaussian_marginal_observed_peak_sf_cpd",
        "dynamic_log_gaussian_marginal_observed_peak_is_boundary",
        "dynamic_log_gaussian_marginal_fit_peak_is_boundary",
        "dynamic_log_gaussian_marginal_low_subcycle_amp_share",
        "dynamic_sf_probe_one_cycle_cpd",
        "dynamic_sf_probe_min_cycles_across_window",
        "dynamic_sf_probe_max_subcycle_cpd",
    ]:
        if col in units.columns:
            merge_cols.append(col)
    curves = ssi.merge(
        units[merge_cols],
        on=["unit_index", "unit_label"],
        how="inner",
        validate="many_to_one",
    ).copy()
    curves[VALUE_COL] = pd.to_numeric(curves[VALUE_COL], errors="coerce")
    curves["display_scale"] = pd.to_numeric(curves["display_scale"], errors="coerce")

    reference = curves[np.isclose(curves["display_scale"], 1.0)].copy()
    reference = reference[["unit_index", "axis_mode", VALUE_COL]].rename(columns={VALUE_COL: "ssi_at_scale_1"})
    curves = curves.merge(reference, on=["unit_index", "axis_mode"], how="left", validate="many_to_one")
    curves["ssi_delta_vs_1x"] = curves[VALUE_COL] - curves["ssi_at_scale_1"]

    stats = curves.groupby(["unit_index", "axis_mode"])[VALUE_COL].agg(["mean", "std"]).reset_index()
    stats = stats.rename(columns={"mean": "ssi_unit_axis_mean", "std": "ssi_unit_axis_std"})
    curves = curves.merge(stats, on=["unit_index", "axis_mode"], how="left", validate="many_to_one")
    curves["ssi_zscore_axis_mode"] = np.where(
        curves["ssi_unit_axis_std"].to_numpy(dtype=float) > float(zscore_min_std),
        (curves[VALUE_COL] - curves["ssi_unit_axis_mean"]) / curves["ssi_unit_axis_std"],
        np.nan,
    )
    curves["ssi_zscore_contract"] = "per-unit z-score across display scales within each axis_mode"
    return curves


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (sf_group, axis_mode, display_scale), sub in curves.groupby(["sf_group", "axis_mode", "display_scale"], sort=True):
        for value_col in [VALUE_COL, "ssi_delta_vs_1x", "ssi_zscore_axis_mode", "displayed_movie_mean_rate"]:
            values = pd.to_numeric(sub[value_col], errors="coerce")
            rows.append(
                {
                    "sf_group": sf_group,
                    "sf_group_label": str(sub["sf_group_label"].iloc[0]),
                    "axis_mode": axis_mode,
                    "display_scale": float(display_scale),
                    "value_name": value_col,
                    "n_units": int(sub["unit_index"].nunique()),
                    "mean": float(np.nanmean(values)),
                    "sem": sem(values),
                    "median": float(np.nanmedian(values)),
                    "q25": float(np.nanpercentile(values, 25)),
                    "q75": float(np.nanpercentile(values, 75)),
                }
            )
    return pd.DataFrame(rows)


def endpoint_summary(curves: pd.DataFrame) -> pd.DataFrame:
    endpoints = curves[curves["display_scale"].isin([1.0, 3.0])].copy()
    pivot = endpoints.pivot_table(
        index=["unit_index", "unit_label", "sf_group", "sf_group_label", "axis_mode"],
        columns="display_scale",
        values=VALUE_COL,
        aggfunc="first",
    ).reset_index()
    pivot["delta_3_minus_1"] = pivot[3.0] - pivot[1.0]
    rows: list[dict[str, object]] = []
    for (sf_group, axis_mode), sub in pivot.groupby(["sf_group", "axis_mode"], sort=True):
        rows.append(
            {
                "sf_group": sf_group,
                "sf_group_label": str(sub["sf_group_label"].iloc[0]),
                "axis_mode": axis_mode,
                "n_units": int(sub["unit_index"].nunique()),
                "mean_delta_3_minus_1": float(np.nanmean(sub["delta_3_minus_1"])),
                "sem_delta_3_minus_1": sem(sub["delta_3_minus_1"]),
                "median_delta_3_minus_1": float(np.nanmedian(sub["delta_3_minus_1"])),
            }
        )

    out = pd.DataFrame(rows)
    diff_rows: list[dict[str, object]] = []
    for axis_mode, sub in pivot.groupby("axis_mode", sort=True):
        low = sub[sub["sf_group"] == "low_sf"]["delta_3_minus_1"].to_numpy(dtype=float)
        high = sub[sub["sf_group"] == "high_sf"]["delta_3_minus_1"].to_numpy(dtype=float)
        diff_rows.append(
            {
                "axis_mode": axis_mode,
                "comparison": "high_sf_minus_low_sf",
                "mean_delta_diff": float(np.nanmean(high) - np.nanmean(low)),
                "median_delta_diff": float(np.nanmedian(high) - np.nanmedian(low)),
                "n_low": int(np.isfinite(low).sum()),
                "n_high": int(np.isfinite(high).sum()),
            }
        )
    return out.merge(pd.DataFrame(diff_rows), on="axis_mode", how="left")


def group_style(sf_group: str) -> tuple[str, float, int]:
    if sf_group == "low_sf":
        return "#2673a6", 1.0, 3
    if sf_group == "high_sf":
        return "#c74343", 1.0, 3
    if sf_group == "sf_uncertain":
        return "0.28", 0.28, 1
    return "0.62", 0.45, 2


def sf_group_order(frame: pd.DataFrame | None = None) -> list[str]:
    order = ["low_sf", "middle_sf", "high_sf", "sf_uncertain"]
    if frame is None or "sf_group" not in frame.columns:
        return order[:3]
    present = set(frame["sf_group"].dropna().astype(str).to_list())
    return [group for group in order if group in present]


def group_definition_text(units: pd.DataFrame) -> str:
    if "sf_group_definition" not in units.columns or units.empty:
        return "unknown SF group definition"
    return str(units["sf_group_definition"].dropna().iloc[0])


def plot_curves(summary: pd.DataFrame, curves: pd.DataFrame, units: pd.DataFrame, out_dir: Path, sf_metric: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    axis_titles = {
        "across_sweep": "scale across; along=1",
        "along_sweep": "scale along; across=1",
    }
    value_panels = [
        (VALUE_COL, "raw SSI bits/spike", "mean SSI"),
        ("ssi_delta_vs_1x", "delta from 1x", "SSI - SSI at 1x"),
        ("ssi_zscore_axis_mode", "within-unit z-score", "SSI z-score"),
    ]
    group_order = sf_group_order(units)

    for row_i, axis_mode in enumerate(["across_sweep", "along_sweep"]):
        for col_i, (value_name, title, ylabel) in enumerate(value_panels):
            ax = axes[row_i, col_i]
            for sf_group in group_order:
                sub = summary[
                    (summary["axis_mode"].astype(str) == axis_mode)
                    & (summary["value_name"].astype(str) == value_name)
                    & (summary["sf_group"].astype(str) == sf_group)
                ].sort_values("display_scale")
                if sub.empty:
                    continue
                color, alpha, zorder = group_style(sf_group)
                label = str(sub["sf_group_label"].iloc[0])
                x = sub["display_scale"].to_numpy(dtype=float)
                y = sub["mean"].to_numpy(dtype=float)
                err = sub["sem"].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", color=color, alpha=alpha, lw=2.2, label=label, zorder=zorder)
                ax.fill_between(x, y - err, y + err, color=color, alpha=0.12 * alpha, zorder=zorder)
            ax.axvline(1.0, ls=":", color="0.6", lw=1.0)
            if value_name in {"ssi_delta_vs_1x", "ssi_zscore_axis_mode"}:
                ax.axhline(0.0, ls="--", color="0.7", lw=0.9)
            ax.set_title(f"{axis_titles[axis_mode]}\n{title}")
            ax.set_xlabel("motion scale")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="0.9", linewidth=0.8)
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "BackImage RR100 SSI modulation by SF-tuning group\n"
        f"SF split metric: {sf_metric}; {group_definition_text(units)}; SSI is averaged over instantaneous maps",
        fontsize=14,
    )
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_ssi_modulation_curves.png", dpi=180)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_ssi_modulation_curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    ax = axes[0]
    for sf_group in group_order:
        sub = units[units["sf_group"] == sf_group]
        color, alpha, zorder = group_style(sf_group)
        ax.scatter(
            sub["sf_rank_low_to_high"],
            sub["sf_split_metric"],
            color=color,
            alpha=alpha,
            s=35,
            label=str(sub["sf_group_label"].iloc[0]) if not sub.empty else sf_group,
            zorder=zorder,
        )
    ax.set_yscale("log")
    ax.set_title("SF split ranking")
    ax.set_xlabel("unit rank, low to high SF")
    ax.set_ylabel("SF split metric (cpd)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    vals = sorted(units["dynamic_peak_spatial_cpd_by_amp"].dropna().unique())
    x = np.arange(len(vals))
    width = min(0.22, 0.75 / max(len(group_order), 1))
    offsets = np.linspace(-0.5 * width * (len(group_order) - 1), 0.5 * width * (len(group_order) - 1), len(group_order))
    for offset, sf_group in zip(offsets, group_order, strict=True):
        sub = units[units["sf_group"] == sf_group]
        color, alpha, _ = group_style(sf_group)
        counts = sub["dynamic_peak_spatial_cpd_by_amp"].value_counts().reindex(vals, fill_value=0)
        ax.bar(x + offset, counts.to_numpy(), width=width, color=color, alpha=max(alpha, 0.5), label=sf_group)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in vals], rotation=35, ha="right")
    ax.set_title("Dynamic peak SF bins")
    ax.set_xlabel("cpd")
    ax.set_ylabel("unit count")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    endpoint = endpoint_summary(curves)
    shown = endpoint[endpoint["sf_group"].isin(["low_sf", "high_sf"])].copy()
    xlabels = ["across_sweep", "along_sweep"]
    xpos = np.arange(len(xlabels))
    width = 0.32
    for offset, sf_group in [(-width / 2, "low_sf"), (width / 2, "high_sf")]:
        sub = shown[shown["sf_group"] == sf_group].set_index("axis_mode").reindex(xlabels)
        color, alpha, _ = group_style(sf_group)
        ax.bar(
            xpos + offset,
            sub["mean_delta_3_minus_1"].to_numpy(dtype=float),
            yerr=sub["sem_delta_3_minus_1"].to_numpy(dtype=float),
            width=width,
            color=color,
            alpha=alpha,
            capsize=3,
            label=str(sub["sf_group_label"].dropna().iloc[0]) if sub["sf_group_label"].notna().any() else sf_group,
        )
    ax.axhline(0.0, ls="--", color="0.7", lw=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["across\nalong=1", "along\nacross=1"])
    ax.set_title("Endpoint modulation")
    ax.set_ylabel("mean SSI(3x) - SSI(1x)")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"SF group definition and endpoint SSI modulation\n{group_definition_text(units)}", fontsize=14)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_definition_and_endpoint_deltas.png", dpi=180)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_definition_and_endpoint_deltas.pdf")
    plt.close(fig)


def endpoint_unit_deltas(curves: pd.DataFrame) -> pd.DataFrame:
    endpoints = curves[curves["display_scale"].isin([1.0, 3.0])].copy()
    pivot = endpoints.pivot_table(
        index=["unit_index", "unit_label", "sf_group", "sf_group_label", "axis_mode"],
        columns="display_scale",
        values=VALUE_COL,
        aggfunc="first",
    ).reset_index()
    pivot["delta_3_minus_1"] = pivot[3.0] - pivot[1.0]
    return pivot


def contribution_summary(curves: pd.DataFrame) -> pd.DataFrame:
    key = ["unit_index", "unit_label", "sf_group", "sf_group_label", "axis_mode"]
    endpoints = curves[curves["display_scale"].isin([1.0, 3.0])].copy()
    one = endpoints[endpoints["display_scale"].eq(1.0)][
        key + [VALUE_COL, "displayed_movie_expected_spikes_arbitrary_dt"]
    ].rename(
        columns={
            VALUE_COL: "ssi_1x",
            "displayed_movie_expected_spikes_arbitrary_dt": "expected_spikes_1x",
        }
    )
    three = endpoints[endpoints["display_scale"].eq(3.0)][
        key + [VALUE_COL, "displayed_movie_expected_spikes_arbitrary_dt"]
    ].rename(
        columns={
            VALUE_COL: "ssi_3x",
            "displayed_movie_expected_spikes_arbitrary_dt": "expected_spikes_3x",
        }
    )
    unit = one.merge(three, on=key, how="inner", validate="one_to_one")
    unit["ssi_delta_3_minus_1"] = unit["ssi_3x"] - unit["ssi_1x"]
    unit["expected_spikes_mean_endpoint"] = (unit["expected_spikes_1x"] + unit["expected_spikes_3x"]) / 2.0
    unit["spike_weighted_ssi_delta_numerator"] = (
        unit["ssi_delta_3_minus_1"] * unit["expected_spikes_mean_endpoint"]
    )
    unit["information_numerator_1x"] = unit["ssi_1x"] * unit["expected_spikes_1x"]
    unit["information_numerator_3x"] = unit["ssi_3x"] * unit["expected_spikes_3x"]
    unit["information_numerator_delta_3_minus_1"] = (
        unit["information_numerator_3x"] - unit["information_numerator_1x"]
    )

    rows: list[dict[str, object]] = []
    for axis_mode, axis_sub in unit.groupby("axis_mode", sort=True):
        net_equal_unit_delta = float(axis_sub["ssi_delta_3_minus_1"].sum())
        positive_equal_unit_delta = float(axis_sub.loc[axis_sub["ssi_delta_3_minus_1"] > 0, "ssi_delta_3_minus_1"].sum())
        net_spike_weighted_delta = float(axis_sub["spike_weighted_ssi_delta_numerator"].sum())
        net_information_numerator_delta = float(axis_sub["information_numerator_delta_3_minus_1"].sum())
        for sf_group, group_sub in axis_sub.groupby("sf_group", sort=True):
            group_equal_delta = float(group_sub["ssi_delta_3_minus_1"].sum())
            group_positive_delta = float(
                group_sub.loc[group_sub["ssi_delta_3_minus_1"] > 0, "ssi_delta_3_minus_1"].sum()
            )
            group_spike_weighted_delta = float(group_sub["spike_weighted_ssi_delta_numerator"].sum())
            group_information_delta = float(group_sub["information_numerator_delta_3_minus_1"].sum())
            rows.append(
                {
                    "axis_mode": axis_mode,
                    "sf_group": sf_group,
                    "sf_group_label": str(group_sub["sf_group_label"].iloc[0]),
                    "n_units": int(group_sub["unit_index"].nunique()),
                    "mean_delta_3_minus_1": float(np.nanmean(group_sub["ssi_delta_3_minus_1"])),
                    "median_delta_3_minus_1": float(np.nanmedian(group_sub["ssi_delta_3_minus_1"])),
                    "sum_equal_unit_delta_3_minus_1": group_equal_delta,
                    "share_of_net_equal_unit_delta": (
                        group_equal_delta / net_equal_unit_delta if net_equal_unit_delta else float("nan")
                    ),
                    "sum_positive_equal_unit_delta_3_minus_1": group_positive_delta,
                    "share_of_positive_equal_unit_delta": (
                        group_positive_delta / positive_equal_unit_delta
                        if positive_equal_unit_delta
                        else float("nan")
                    ),
                    "spike_weighted_ssi_delta_numerator": group_spike_weighted_delta,
                    "share_of_net_spike_weighted_ssi_delta": (
                        group_spike_weighted_delta / net_spike_weighted_delta
                        if net_spike_weighted_delta
                        else float("nan")
                    ),
                    "information_numerator_delta_3_minus_1": group_information_delta,
                    "share_of_information_numerator_delta": (
                        group_information_delta / net_information_numerator_delta
                        if net_information_numerator_delta
                        else float("nan")
                    ),
                    "net_equal_unit_delta_all_units": net_equal_unit_delta,
                    "positive_equal_unit_delta_all_units": positive_equal_unit_delta,
                    "net_spike_weighted_ssi_delta_all_units": net_spike_weighted_delta,
                    "net_information_numerator_delta_all_units": net_information_numerator_delta,
                }
            )
    return pd.DataFrame(rows)


def sf_group_definition_summary(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sf_group, sub in units.groupby("sf_group", sort=True):
        metric = pd.to_numeric(sub["sf_split_metric"], errors="coerce")
        rows.append(
            {
                "sf_group": sf_group,
                "sf_group_label": str(sub["sf_group_label"].iloc[0]),
                "sf_group_definition": group_definition_text(units),
                "sf_split_metric_name": str(sub["sf_split_metric_name"].iloc[0]),
                "sf_split_metric_column": str(sub["sf_split_metric_column"].iloc[0]),
                "n_units": int(sub["unit_index"].nunique()),
                "rank_min_low_to_high": int(np.nanmin(pd.to_numeric(sub["sf_rank_low_to_high"], errors="coerce"))),
                "rank_max_low_to_high": int(np.nanmax(pd.to_numeric(sub["sf_rank_low_to_high"], errors="coerce"))),
                "sf_split_metric_min": float(np.nanmin(metric)),
                "sf_split_metric_max": float(np.nanmax(metric)),
                "sf_split_metric_median": float(np.nanmedian(metric)),
            }
        )
    return pd.DataFrame(rows)


def plot_unit_level_views(summary: pd.DataFrame, curves: pd.DataFrame, units: pd.DataFrame, out_dir: Path, sf_metric: str) -> None:
    axis_titles = {
        "across_sweep": "scale across; along=1",
        "along_sweep": "scale along; across=1",
    }
    value_panels = [
        (VALUE_COL, "raw SSI bits/spike", "SSI"),
        ("ssi_delta_vs_1x", "delta from 1x", "SSI - SSI at 1x"),
        ("ssi_zscore_axis_mode", "within-unit z-score", "SSI z-score"),
    ]
    group_order = sf_group_order(units)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    for row_i, axis_mode in enumerate(["across_sweep", "along_sweep"]):
        for col_i, (value_name, title, ylabel) in enumerate(value_panels):
            ax = axes[row_i, col_i]
            for sf_group in group_order:
                group_curves = curves[
                    (curves["axis_mode"].astype(str) == axis_mode)
                    & (curves["sf_group"].astype(str) == sf_group)
                ]
                color, alpha, zorder = group_style(sf_group)
                label = None
                for unit_index, unit_sub in group_curves.groupby("unit_index", sort=False):
                    unit_sub = unit_sub.sort_values("display_scale")
                    ax.plot(
                        unit_sub["display_scale"].to_numpy(dtype=float),
                        pd.to_numeric(unit_sub[value_name], errors="coerce").to_numpy(dtype=float),
                        color=color,
                        alpha=0.22 if sf_group != "middle_sf" else 0.14,
                        lw=0.85,
                        zorder=zorder,
                    )
                    label = str(unit_sub["sf_group_label"].iloc[0])
                mean_sub = summary[
                    (summary["axis_mode"].astype(str) == axis_mode)
                    & (summary["value_name"].astype(str) == value_name)
                    & (summary["sf_group"].astype(str) == sf_group)
                ].sort_values("display_scale")
                if not mean_sub.empty:
                    ax.plot(
                        mean_sub["display_scale"].to_numpy(dtype=float),
                        mean_sub["mean"].to_numpy(dtype=float),
                        color=color,
                        alpha=max(alpha, 0.8),
                        lw=2.7,
                        marker="o",
                        ms=4,
                        label=label,
                        zorder=zorder + 4,
                    )
            ax.axvline(1.0, ls=":", color="0.6", lw=1.0)
            if value_name in {"ssi_delta_vs_1x", "ssi_zscore_axis_mode"}:
                ax.axhline(0.0, ls="--", color="0.72", lw=0.9)
            ax.set_title(f"{axis_titles[axis_mode]}\n{title}")
            ax.set_xlabel("motion scale")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="0.9", linewidth=0.8)
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "BackImage RR100 SF groups: every unit color-coded by group\n"
        f"SF split metric: {sf_metric}; {group_definition_text(units)}; thick lines are group means",
        fontsize=14,
    )
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_all_unit_colorcoded_curves.png", dpi=180)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_all_unit_colorcoded_curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(len(group_order), 2, figsize=(11.5, 3.35 * len(group_order)), sharey=True, constrained_layout=True)
    if len(group_order) == 1:
        axes = np.asarray([axes])
    finite_delta = pd.to_numeric(curves["ssi_delta_vs_1x"], errors="coerce").to_numpy(dtype=float)
    finite_delta = finite_delta[np.isfinite(finite_delta)]
    ylim = None
    if finite_delta.size:
        pad = max(0.004, 0.08 * (float(np.nanmax(finite_delta)) - float(np.nanmin(finite_delta))))
        ylim = (float(np.nanmin(finite_delta) - pad), float(np.nanmax(finite_delta) + pad))
    for row_i, sf_group in enumerate(group_order):
        color, alpha, zorder = group_style(sf_group)
        for col_i, axis_mode in enumerate(["across_sweep", "along_sweep"]):
            ax = axes[row_i, col_i]
            sub = curves[
                (curves["axis_mode"].astype(str) == axis_mode)
                & (curves["sf_group"].astype(str) == sf_group)
            ]
            for unit_index, unit_sub in sub.groupby("unit_index", sort=False):
                unit_sub = unit_sub.sort_values("display_scale")
                ax.plot(
                    unit_sub["display_scale"].to_numpy(dtype=float),
                    unit_sub["ssi_delta_vs_1x"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.32 if sf_group != "middle_sf" else 0.22,
                    lw=1.0,
                    zorder=zorder,
                )
            mean_sub = summary[
                (summary["axis_mode"].astype(str) == axis_mode)
                & (summary["value_name"].astype(str) == "ssi_delta_vs_1x")
                & (summary["sf_group"].astype(str) == sf_group)
            ].sort_values("display_scale")
            if not mean_sub.empty:
                ax.plot(
                    mean_sub["display_scale"].to_numpy(dtype=float),
                    mean_sub["mean"].to_numpy(dtype=float),
                    color="black",
                    lw=2.4,
                    marker="o",
                    ms=4,
                    label="group mean",
                    zorder=10,
                )
            ax.axvline(1.0, ls=":", color="0.6", lw=1.0)
            ax.axhline(0.0, ls="--", color="0.72", lw=0.9)
            if ylim is not None:
                ax.set_ylim(*ylim)
            group_label = str(sub["sf_group_label"].iloc[0]) if not sub.empty else sf_group
            ax.set_title(f"{group_label}\n{axis_titles[axis_mode]}")
            ax.set_xlabel("motion scale")
            if col_i == 0:
                ax.set_ylabel("SSI - SSI at 1x")
            ax.grid(True, color="0.9", linewidth=0.8)
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "BackImage RR100 SF groups: unit-level SSI modulation within each group\n"
        f"{group_definition_text(units)}; each thin line is one unit; panels share the y-axis",
        fontsize=14,
    )
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_faceted_unit_delta_curves.png", dpi=180)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_faceted_unit_delta_curves.pdf")
    plt.close(fig)

    unit_deltas = endpoint_unit_deltas(curves)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True, constrained_layout=True)
    for ax, axis_mode in zip(axes, ["across_sweep", "along_sweep"], strict=True):
        data = []
        labels = []
        colors = []
        for sf_group in group_order:
            sub = unit_deltas[
                (unit_deltas["axis_mode"].astype(str) == axis_mode)
                & (unit_deltas["sf_group"].astype(str) == sf_group)
            ]
            data.append(sub["delta_3_minus_1"].to_numpy(dtype=float))
            labels.append(str(sub["sf_group_label"].iloc[0]) if not sub.empty else sf_group)
            colors.append(group_style(sf_group)[0])
        finite_entries: list[tuple[int, np.ndarray, str]] = []
        for i, (values, color) in enumerate(zip(data, colors, strict=True), start=1):
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                ax.text(i, 0.0, "no units", rotation=90, ha="center", va="center", fontsize=8, color=color, alpha=0.75)
                continue
            finite_entries.append((i, finite, color))
        if finite_entries:
            parts = ax.violinplot(
                [values for _, values, _ in finite_entries],
                positions=[i for i, _, _ in finite_entries],
                showmeans=True,
                showextrema=False,
            )
            for body, (_, _, color) in zip(parts["bodies"], finite_entries, strict=True):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.22)
            parts["cmeans"].set_color("black")
        for i, finite, color in finite_entries:
            jitter = np.linspace(-0.11, 0.11, finite.size)
            ax.scatter(np.full(finite.size, i) + jitter, finite, s=24, color=color, alpha=0.72, edgecolor="white", linewidth=0.35)
        ax.axhline(0.0, ls="--", color="0.72", lw=0.9)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(axis_titles[axis_mode])
        ax.set_ylabel("SSI(3x) - SSI(1x)")
        ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    fig.suptitle(f"BackImage RR100 SF groups: endpoint delta distribution by unit\n{group_definition_text(units)}", fontsize=14)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_endpoint_delta_distributions.png", dpi=180)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_endpoint_delta_distributions.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    units = build_sf_groups(
        args.tuning_dir,
        args.sf_metric,
        args.tertile_n,
        low_sf_max_cpd=args.low_sf_max_cpd,
        high_sf_min_cpd=args.high_sf_min_cpd,
        sf_fit_n_bootstrap=args.sf_fit_n_bootstrap,
        sf_fit_seed=args.sf_fit_seed,
        sf_fit_min_r2=args.sf_fit_min_r2,
        sf_fit_max_ci_width_octaves=args.sf_fit_max_ci_width_octaves,
        sf_fit_uncertain_policy=args.sf_fit_uncertain_policy,
    )
    ssi = pd.read_csv(args.ssi_csv)
    curves = add_curve_metrics(ssi, units, zscore_min_std=float(args.zscore_min_std))
    summary = summarize_curves(curves)
    endpoints = endpoint_summary(curves)
    contributions = contribution_summary(curves)
    definitions = sf_group_definition_summary(units)

    units.to_csv(args.out_dir / f"{args.sf_metric}_sf_tuning_unit_groups.csv", index=False)
    curves.to_csv(args.out_dir / f"{args.sf_metric}_sf_group_ssi_curves_long.csv", index=False)
    summary.to_csv(args.out_dir / f"{args.sf_metric}_sf_group_ssi_summary.csv", index=False)
    endpoints.to_csv(args.out_dir / f"{args.sf_metric}_sf_group_endpoint_delta_summary.csv", index=False)
    contributions.to_csv(args.out_dir / f"{args.sf_metric}_sf_group_3x_increase_contribution_summary.csv", index=False)
    definitions.to_csv(args.out_dir / f"{args.sf_metric}_sf_group_definition_summary.csv", index=False)
    plot_curves(summary, curves, units, args.out_dir, args.sf_metric)
    plot_unit_level_views(summary, curves, units, args.out_dir, args.sf_metric)

    print(f"Wrote SF-group SSI modulation outputs to {args.out_dir}")
    print(group_definition_text(units))
    print(units["sf_group"].value_counts().to_string())
    print(endpoints.to_string(index=False))


if __name__ == "__main__":
    main()
