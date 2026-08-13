#!/usr/bin/env python3
"""Fit parametric joint-F0 factors and validate SF shape on recorded gratings."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
F0_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
RECORDED_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1"
OUT_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_joint_f0_parametric_recorded_validation_v1"
COLORS = {"recorded": "#111111", "parametric": "#009E73", "sampled": "#0072B2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-dir", type=Path, default=F0_DIR)
    parser.add_argument("--recorded-dir", type=Path, default=RECORDED_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--strong-recorded-modulation-fraction", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def log_gaussian(frequency: np.ndarray, baseline: float, amplitude: float, center_log2: float, sigma_oct: float) -> np.ndarray:
    x = np.log2(np.asarray(frequency, dtype=float))
    return baseline + amplitude * np.exp(-0.5 * ((x - center_log2) / sigma_oct) ** 2)


def fit_log_gaussian(frequency: np.ndarray, values: np.ndarray) -> dict[str, object]:
    frequency = np.asarray(frequency, dtype=float)
    values = np.asarray(values, dtype=float)
    x = np.log2(frequency)
    if not np.isfinite(values).all() or np.ptp(values) <= 1e-10:
        return {"fit_ok": False}
    lower = np.asarray([0.0, 0.0, x.min() - 2.0, 0.08])
    upper = np.asarray([1.5, 3.0, x.max() + 2.0, 5.0])
    baseline0 = max(float(values.min()), 0.0)
    amplitude0 = max(float(values.max() - baseline0), 0.05)
    centers = np.unique(np.asarray([x[np.argmax(values)], x.min(), np.median(x), x.max()], dtype=float))
    sigmas = (0.25, 0.5, 1.0, 2.0, 3.5)
    best = None
    for center in centers:
        for sigma in sigmas:
            start = np.asarray([baseline0, amplitude0, center, sigma], dtype=float)
            try:
                result = least_squares(
                    lambda params: log_gaussian(frequency, *params) - values,
                    start, bounds=(lower, upper), max_nfev=20_000,
                )
            except Exception:
                continue
            sse = float(np.sum(result.fun**2))
            if best is None or sse < best[0]:
                best = (sse, result)
    if best is None:
        return {"fit_ok": False}
    sse, result = best
    baseline, amplitude, center_log2, sigma = (float(v) for v in result.x)
    predicted = log_gaussian(frequency, baseline, amplitude, center_log2, sigma)
    centered = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - sse / centered if centered > 1e-15 else np.nan
    dense_frequency = np.geomspace(frequency.min(), frequency.max(), 2001)
    dense_prediction = log_gaussian(dense_frequency, baseline, amplitude, center_log2, sigma)
    preferred_within_support = float(dense_frequency[np.argmax(dense_prediction)])
    sampled_preferred = float(frequency[np.argmax(predicted)])
    return {
        "fit_ok": bool(result.success),
        "baseline": baseline,
        "amplitude": amplitude,
        "center_log2": center_log2,
        "center_frequency": float(2.0**center_log2),
        "sigma_octaves": sigma,
        "fwhm_octaves": float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma),
        "fit_r2": float(r2),
        "fit_rmse": float(np.sqrt(np.mean((predicted - values) ** 2))),
        "preferred_within_support": preferred_within_support,
        "sampled_preferred": sampled_preferred,
        "center_outside_sampled_support": bool(2.0**center_log2 < frequency.min() or 2.0**center_log2 > frequency.max()),
        "sampled_peak_at_edge": bool(np.isclose(sampled_preferred, frequency.min()) or np.isclose(sampled_preferred, frequency.max())),
        "predicted_at_samples": predicted,
    }


def fit_all_units(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = {
        "factors": args.f0_dir / "f0_separable_factor_points.csv",
        "f0_summary": args.f0_dir / "f0_separable_fit_unit_summary.csv",
        "surface_points": args.f0_dir / "f0_surface_fit_and_residual_points.csv",
        "f0_manifest": args.f0_dir / "manifest.json",
        "recorded_maps": args.recorded_dir / "rr100_grating_tuning_maps_long.csv",
        "recorded_metrics": args.recorded_dir / "rr100_grating_tuning_metrics.csv",
        "recorded_manifest": args.recorded_dir / "rr100_recorded_twin_gratings_manifest.json",
    }
    factors = pd.read_csv(paths["factors"])
    f0_summary = pd.read_csv(paths["f0_summary"]).set_index("rr100_index")
    surface = pd.read_csv(paths["surface_points"])
    rows: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for unit in range(100):
        base = f0_summary.loc[unit].to_dict()
        row: dict[str, object] = {"rr100_index": unit, **base}
        responsive = bool(base["responsive_positive_f0_flag"])
        axis_fits: dict[str, dict[str, object]] = {}
        for axis, prefix in (("spatial_frequency", "sf"), ("temporal_frequency", "tf")):
            frame = factors.loc[(factors["rr100_index"].eq(unit)) & factors["axis"].eq(axis)].sort_values("frequency")
            frequency = frame["frequency"].to_numpy(dtype=float)
            values = frame["normalized_factor"].to_numpy(dtype=float)
            fit = fit_log_gaussian(frequency, values) if responsive else {"fit_ok": False}
            axis_fits[prefix] = fit
            row[f"{prefix}_parametric_fit_ok"] = bool(fit.get("fit_ok", False))
            for key in (
                "baseline", "amplitude", "center_log2", "center_frequency", "sigma_octaves", "fwhm_octaves",
                "fit_r2", "fit_rmse", "preferred_within_support", "sampled_preferred",
                "center_outside_sampled_support", "sampled_peak_at_edge",
            ):
                row[f"{prefix}_{key}"] = fit.get(key, np.nan)
            predicted = fit.get("predicted_at_samples", np.full(len(frequency), np.nan))
            for freq, observed, prediction in zip(frequency, values, predicted):
                curves.append({
                    "rr100_index": unit, "axis": axis, "frequency": float(freq),
                    "observed_normalized_factor": float(observed),
                    "parametric_prediction": float(prediction),
                    "responsive_positive_f0_flag": responsive,
                })
        if responsive and axis_fits["sf"].get("fit_ok") and axis_fits["tf"].get("fit_ok"):
            points = surface.loc[surface["rr100_index"].eq(unit)].copy()
            sf_values = np.sort(points["spatial_cpd"].unique())
            tf_values = np.sort(points["temporal_hz"].unique())
            sf_prediction = log_gaussian(sf_values, *[
                axis_fits["sf"][k] for k in ("baseline", "amplitude", "center_log2", "sigma_octaves")
            ])
            tf_prediction = log_gaussian(tf_values, *[
                axis_fits["tf"][k] for k in ("baseline", "amplitude", "center_log2", "sigma_octaves")
            ])
            sf_prediction /= max(float(sf_prediction.max()), 1e-15)
            tf_prediction /= max(float(tf_prediction.max()), 1e-15)
            prediction = float(base["rank1_gain_f0_hz"]) * np.outer(sf_prediction, tf_prediction)
            observed = points.pivot(index="spatial_cpd", columns="temporal_hz", values="observed_positive_f0_hz").sort_index().sort_index(axis=1).to_numpy()
            residual = observed - prediction
            centered = float(np.sum((observed - observed.mean()) ** 2))
            row["parametric_surface_centered_r2"] = 1.0 - float(np.sum(residual**2)) / centered if centered > 1e-15 else np.nan
            row["parametric_surface_rmse_hz"] = float(np.sqrt(np.mean(residual**2)))
        else:
            row["parametric_surface_centered_r2"] = np.nan
            row["parametric_surface_rmse_hz"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(curves), paths


def orientation_distance(values: np.ndarray, target: float) -> np.ndarray:
    delta = np.abs(np.asarray(values, dtype=float) - float(target)) % 180.0
    return np.minimum(delta, 180.0 - delta)


def score_shape(prediction: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if len(prediction) < 3 or np.ptp(prediction) <= 1e-12 or np.ptp(observed) <= 1e-12:
        return {"pearson_r": np.nan, "normalized_rmse": np.nan}
    pred_norm = (prediction - prediction.min()) / np.ptp(prediction)
    obs_norm = (observed - observed.min()) / np.ptp(observed)
    return {
        "pearson_r": float(pearsonr(prediction, observed).statistic),
        "normalized_rmse": float(np.sqrt(np.mean((pred_norm - obs_norm) ** 2))),
    }


def validate_recorded(
    args: argparse.Namespace, fit_table: pd.DataFrame, factors: pd.DataFrame, paths: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    maps = pd.read_csv(paths["recorded_maps"])
    maps = maps.loc[maps["source"].eq("recorded")].copy()
    maps["sf_cpd"] = maps["spatial_frequency_cpd"].round(6)
    # Retain every positive SF in the recorded experiment.  The joint-F0
    # factorization was fit only through 11.3137 cpd, so 16 cpd is an explicit
    # out-of-fit-support prediction rather than a reason to discard real data.
    maps = maps.loc[maps["sf_cpd"] > 0].copy()
    metrics = pd.read_csv(paths["recorded_metrics"])[["rr100_index", "real_peak_ori", "real_peak_sf", "sf_curve_correlation"]]
    factor_sf = factors.loc[factors["axis"].eq("spatial_frequency")].copy()
    factor_lookup = factor_sf.set_index(["rr100_index", "frequency"])["observed_normalized_factor"].to_dict()
    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for _, fit in fit_table.iterrows():
        unit = int(fit["rr100_index"])
        unit_maps = maps.loc[maps["rr100_index"].eq(unit)].copy()
        orientations = np.sort(unit_maps["orientation_deg"].unique())
        predicted_orientation = float(fit["preferred_orientation_deg"])
        distances = orientation_distance(orientations, predicted_orientation)
        nearest = orientations[np.isclose(distances, distances.min())]
        predicted_ori_curve = unit_maps.loc[unit_maps["orientation_deg"].isin(nearest)].groupby("sf_cpd", as_index=False)["rate_hz"].mean()
        metric = metrics.loc[metrics["rr100_index"].eq(unit)].iloc[0]
        real_peak_ori = float(metric["real_peak_ori"])
        real_peak_curve = unit_maps.loc[np.isclose(unit_maps["orientation_deg"], real_peak_ori)].sort_values("sf_cpd")
        primary = predicted_ori_curve.sort_values("sf_cpd")
        sf = primary["sf_cpd"].to_numpy(dtype=float)
        observed = primary["rate_hz"].to_numpy(dtype=float)
        in_fit_support = sf <= 11.3137086
        param_ok = bool(fit["sf_parametric_fit_ok"])
        if param_ok:
            prediction = log_gaussian(sf, fit["sf_baseline"], fit["sf_amplitude"], fit["sf_center_log2"], fit["sf_sigma_octaves"])
        else:
            prediction = np.full(len(sf), np.nan)
        sampled_prediction = np.asarray([factor_lookup.get((unit, float(value)), np.nan) for value in sf], dtype=float)
        param_score = score_shape(prediction, observed)
        in_support_score = score_shape(prediction[in_fit_support], observed[in_fit_support])
        sampled_finite = np.isfinite(sampled_prediction)
        sampled_score = score_shape(sampled_prediction[sampled_finite], observed[sampled_finite])
        peak_prediction = float(sf[np.nanargmax(prediction)]) if np.isfinite(prediction).any() else np.nan
        peak_observed = float(sf[np.nanargmax(observed)]) if np.isfinite(observed).any() else np.nan
        sf_in = sf[in_fit_support]
        pred_in = prediction[in_fit_support]
        obs_in = observed[in_fit_support]
        peak_prediction_in = float(sf_in[np.nanargmax(pred_in)]) if np.isfinite(pred_in).any() else np.nan
        peak_observed_in = float(sf_in[np.nanargmax(obs_in)]) if np.isfinite(obs_in).any() else np.nan

        sf_sens = real_peak_curve["sf_cpd"].to_numpy(dtype=float)
        observed_sens = real_peak_curve["rate_hz"].to_numpy(dtype=float)
        prediction_sens = (
            log_gaussian(sf_sens, fit["sf_baseline"], fit["sf_amplitude"], fit["sf_center_log2"], fit["sf_sigma_octaves"])
            if param_ok else np.full(len(sf_sens), np.nan)
        )
        sensitivity_score = score_shape(prediction_sens, observed_sens)
        modulation = float(np.ptp(observed) / max(abs(float(np.mean(observed))), 1e-12))
        row = {
            "rr100_index": unit,
            "session": fit["session"],
            "responsive_positive_f0_flag": bool(fit["responsive_positive_f0_flag"]),
            "predicted_preferred_orientation_deg": predicted_orientation,
            "nearest_recorded_orientations_deg": ",".join(f"{v:g}" for v in nearest),
            "recorded_peak_orientation_deg": real_peak_ori,
            "recorded_positive_sf_support_cpd": ",".join(f"{v:g}" for v in sf),
            "n_recorded_positive_sf_points": len(sf),
            "in_fit_support_sf_cpd": ",".join(f"{v:g}" for v in sf_in),
            "n_in_fit_support_sf_points": len(sf_in),
            "includes_16_cpd_extrapolation": bool(np.any(sf > 11.3137086)),
            "recorded_curve_modulation_fraction": modulation,
            "recorded_curve_pearson_r_parametric": param_score["pearson_r"],
            "recorded_curve_normalized_rmse_parametric": param_score["normalized_rmse"],
            "recorded_curve_pearson_r_parametric_in_fit_support": in_support_score["pearson_r"],
            "recorded_curve_normalized_rmse_parametric_in_fit_support": in_support_score["normalized_rmse"],
            "recorded_curve_pearson_r_sampled_factor": sampled_score["pearson_r"],
            "recorded_curve_normalized_rmse_sampled_factor": sampled_score["normalized_rmse"],
            "recorded_peak_orientation_curve_r_parametric_sensitivity": sensitivity_score["pearson_r"],
            "predicted_peak_sf_on_recorded_support": peak_prediction,
            "recorded_peak_sf_on_predicted_orientation_recorded_support": peak_observed,
            "peak_exact_on_recorded_support": bool(np.isclose(peak_prediction, peak_observed)) if np.isfinite(peak_prediction) else False,
            "peak_absolute_difference_octaves": float(abs(np.log2(peak_prediction / peak_observed))) if peak_prediction > 0 and peak_observed > 0 else np.nan,
            "predicted_peak_sf_in_fit_support": peak_prediction_in,
            "recorded_peak_sf_in_fit_support": peak_observed_in,
            "peak_exact_in_fit_support": bool(np.isclose(peak_prediction_in, peak_observed_in)) if np.isfinite(peak_prediction_in) else False,
            "existing_heldout_twin_vs_recorded_sf_curve_r": float(metric["sf_curve_correlation"]),
        }
        rows.append(row)
        pred_norm = (prediction - np.nanmin(prediction)) / max(float(np.nanmax(prediction) - np.nanmin(prediction)), 1e-15) if np.isfinite(prediction).any() else prediction
        sampled_norm = (sampled_prediction - np.nanmin(sampled_prediction)) / max(float(np.nanmax(sampled_prediction) - np.nanmin(sampled_prediction)), 1e-15) if np.isfinite(sampled_prediction).any() else sampled_prediction
        obs_norm = (observed - observed.min()) / max(float(np.ptp(observed)), 1e-15)
        for spatial, obs, obs_n, pred, pred_n, samp, samp_n in zip(sf, observed, obs_norm, prediction, pred_norm, sampled_prediction, sampled_norm):
            curve_rows.append({
                "rr100_index": unit, "sf_cpd": float(spatial), "recorded_rate_hz": float(obs),
                "recorded_range_normalized": float(obs_n), "parametric_factor_prediction": float(pred),
                "parametric_range_normalized": float(pred_n), "sampled_factor_prediction": float(samp),
                "sampled_factor_range_normalized": float(samp_n),
            })
    validation = pd.DataFrame(rows).merge(fit_table, on=["rr100_index", "session", "responsive_positive_f0_flag"], how="left", validate="one_to_one")
    return validation, pd.DataFrame(curve_rows)


def summarize(validation: pd.DataFrame, threshold: float) -> pd.DataFrame:
    responsive = validation.loc[validation["responsive_positive_f0_flag"] & validation["sf_parametric_fit_ok"]].copy()
    strong = responsive.loc[responsive["recorded_curve_modulation_fraction"] >= threshold].copy()
    rows = []
    for label, d in (("all_f0_responsive", responsive), ("recorded_modulation_ge_threshold", strong)):
        r = d["recorded_curve_pearson_r_parametric"].dropna()
        sampled = d["recorded_curve_pearson_r_sampled_factor"].dropna()
        rows.append({
            "subset": label, "recorded_modulation_threshold": threshold, "n_units": len(d),
            "median_sf_factor_fit_r2": float(d["sf_fit_r2"].median()),
            "median_tf_factor_fit_r2": float(d["tf_fit_r2"].median()),
            "median_parametric_surface_r2": float(d["parametric_surface_centered_r2"].median()),
            "median_recorded_curve_r_parametric": float(r.median()),
            "fraction_recorded_curve_r_parametric_gt_0": float((r > 0).mean()),
            "fraction_recorded_curve_r_parametric_gt_0p5": float((r > 0.5).mean()),
            "median_recorded_curve_r_sampled_factor": float(sampled.median()),
            "median_recorded_curve_r_parametric_in_fit_support": float(d["recorded_curve_pearson_r_parametric_in_fit_support"].median()),
            "exact_recorded_support_peak_fraction": float(d["peak_exact_on_recorded_support"].mean()),
            "exact_in_fit_support_peak_fraction": float(d["peak_exact_in_fit_support"].mean()),
            "within_one_octave_peak_fraction": float((d["peak_absolute_difference_octaves"] <= 1.0).mean()),
            "median_existing_heldout_twin_vs_recorded_curve_r": float(d["existing_heldout_twin_vs_recorded_sf_curve_r"].median()),
        })
    return pd.DataFrame(rows)


def select_examples(validation: pd.DataFrame, threshold: float) -> pd.DataFrame:
    d = validation.loc[
        validation["responsive_positive_f0_flag"] & validation["sf_parametric_fit_ok"]
        & (validation["recorded_curve_modulation_fraction"] >= threshold)
    ].copy()
    roles: list[tuple[str, int, str, float]] = []

    def add(role: str, pool: pd.DataFrame, criterion: str, maximize: bool) -> None:
        used = {u for _, u, _, _ in roles}; p = pool.loc[~pool["rr100_index"].isin(used) & pool[criterion].notna()]
        row = p.loc[p[criterion].idxmax() if maximize else p[criterion].idxmin()]
        roles.append((role, int(row["rr100_index"]), criterion, float(row[criterion])))

    add("best recorded prediction", d, "recorded_curve_pearson_r_parametric", True)
    add("worst recorded prediction", d, "recorded_curve_pearson_r_parametric", False)
    good_fit_bad_real = d.loc[d["sf_fit_r2"] >= 0.9]
    add("good parametric fit, poor real transfer", good_fit_bad_real, "recorded_curve_pearson_r_parametric", False)
    median = float(d["recorded_curve_pearson_r_parametric"].median())
    d["distance_to_median_r"] = abs(d["recorded_curve_pearson_r_parametric"] - median)
    add("typical transfer", d, "distance_to_median_r", False)
    selected = pd.DataFrame(roles, columns=["selection_role", "rr100_index", "criterion", "criterion_value"])
    return selected.merge(validation, on="rr100_index", how="left", validate="one_to_one")


def plot_summary(validation: pd.DataFrame, summary: pd.DataFrame, out: Path, dpi: int) -> None:
    d = validation.loc[validation["responsive_positive_f0_flag"] & validation["sf_parametric_fit_ok"]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 9.5), constrained_layout=True)
    axes[0, 0].hist(d["sf_fit_r2"], bins=np.linspace(-1, 1, 21), alpha=.7, color="#009E73", label="SF factor")
    axes[0, 0].hist(d["tf_fit_r2"], bins=np.linspace(-1, 1, 21), alpha=.55, color="#7A5195", label="TF factor")
    axes[0, 0].set(xlabel="log-Gaussian factor-fit R²", ylabel="units")
    axes[0, 0].set_title("A  Parametric fit to sampled factors", loc="left", fontweight="bold"); axes[0, 0].legend(frameon=False)

    bins=np.linspace(-1,1,17)
    axes[0, 1].hist(d["recorded_curve_pearson_r_parametric"], bins=bins, alpha=.75, color=COLORS["parametric"], label="parametric SF fit")
    axes[0, 1].hist(d["recorded_curve_pearson_r_sampled_factor"], bins=bins, histtype="step", lw=2, color=COLORS["sampled"], label="sampled SF factor (fit support)")
    axes[0, 1].axvline(d["recorded_curve_pearson_r_parametric"].median(), color="black", ls="--")
    axes[0, 1].set(xlabel="Pearson r with recorded SF curve", ylabel="units")
    axes[0, 1].set_title("B  Transfer to held-out recorded gratings", loc="left", fontweight="bold"); axes[0, 1].legend(frameon=False)

    support=np.asarray([1.,2.,4.,8.,16.]); trans=pd.crosstab(d["recorded_peak_sf_on_predicted_orientation_recorded_support"],d["predicted_peak_sf_on_recorded_support"]).reindex(index=support,columns=support,fill_value=0)
    im=axes[1,0].imshow(trans,origin="lower",cmap="Greens")
    axes[1,0].set_xticks(range(5),["1","2","4","8","16"]); axes[1,0].set_yticks(range(5),["1","2","4","8","16"])
    axes[1,0].set(xlabel="parametric predicted peak SF",ylabel="recorded peak SF")
    axes[1,0].set_title("C  Full recorded-support peak prediction",loc="left",fontweight="bold")
    for i in range(5):
        for j in range(5):
            n=int(trans.iloc[i,j]);
            if n: axes[1,0].text(j,i,str(n),ha="center",va="center",color="white" if n>trans.to_numpy().max()*.55 else "black")
    fig.colorbar(im,ax=axes[1,0],label="unit count",shrink=.82)

    axes[1,1].scatter(d["sf_fit_r2"],d["recorded_curve_pearson_r_parametric"],c=d["recorded_curve_modulation_fraction"],cmap="viridis",s=38,alpha=.8,edgecolor="white",linewidth=.3)
    axes[1,1].axhline(0,color="0.5",lw=.8); axes[1,1].set(xlabel="SF factor parametric-fit R²",ylabel="recorded SF curve r")
    rho=spearmanr(d["sf_fit_r2"],d["recorded_curve_pearson_r_parametric"],nan_policy="omit").statistic
    axes[1,1].text(.03,.97,f"Spearman rho={rho:.2f}",transform=axes[1,1].transAxes,va="top")
    axes[1,1].set_title("D  Internal fit does not guarantee real transfer",loc="left",fontweight="bold")
    primary=summary.iloc[0]
    fig.suptitle(
        f"RR100 parametric joint-F0 fits and independent recorded-SF validation\n"
        f"n={int(primary['n_units'])}; median recorded curve r={primary['median_recorded_curve_r_parametric']:.2f}; shape-only, all recorded positive SFs",
        fontsize=14,
    )
    fig.savefig(out,dpi=dpi,bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)


def plot_examples(selected: pd.DataFrame, curves: pd.DataFrame, out: Path, dpi: int) -> None:
    fig,axes=plt.subplots(len(selected),1,figsize=(9.5,2.7*len(selected)),constrained_layout=True)
    dense=np.geomspace(1,16,501)
    for ax,(_,row) in zip(axes,selected.iterrows()):
        unit=int(row.rr100_index); c=curves[curves.rr100_index.eq(unit)].sort_values('sf_cpd')
        ax.plot(c.sf_cpd,c.recorded_range_normalized,'o-',color=COLORS['recorded'],lw=2,label='held-out recorded')
        ax.plot(c.sf_cpd,c.sampled_factor_range_normalized,'s--',color=COLORS['sampled'],lw=1.5,label='sampled joint-F0 factor')
        pred=log_gaussian(dense,row.sf_baseline,row.sf_amplitude,row.sf_center_log2,row.sf_sigma_octaves)
        pred=(pred-pred.min())/max(float(np.ptp(pred)),1e-15)
        ax.plot(dense,pred,color=COLORS['parametric'],lw=2,label='parametric log-Gaussian')
        ax.set_xscale('log',base=2); ax.set_ylim(-.08,1.08); ax.grid(axis='y',color='.9')
        ax.set_ylabel('range-normalized\nSF response')
        ax.set_title(f"{row.selection_role} — RR100 {unit}; factor R²={row.sf_fit_r2:.2f}; recorded r={row.recorded_curve_pearson_r_parametric:.2f}; support {row.recorded_positive_sf_support_cpd}",loc='left',fontsize=10)
    axes[0].legend(frameon=False,ncol=3,fontsize=8); axes[-1].set_xticks([1,2,4,8,16],["1","2","4","8","16"]); axes[-1].set_xlabel('spatial frequency (cpd; 16 is outside fit support)')
    fig.suptitle('Parametric joint-F0 SF predictions: positive examples and transfer failures',fontsize=13)
    fig.savefig(out,dpi=dpi,bbox_inches='tight'); fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight'); plt.close(fig)


def main() -> None:
    args=parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    fit_table,factor_curves,paths=fit_all_units(args)
    validation,recorded_curves=validate_recorded(args,fit_table,factor_curves,paths)
    summary=summarize(validation,float(args.strong_recorded_modulation_fraction))
    selected=select_examples(validation,float(args.strong_recorded_modulation_fraction))
    fit_table.to_csv(args.out_dir/'rr100_joint_f0_parametric_fit_summary.csv',index=False)
    factor_curves.to_csv(args.out_dir/'rr100_joint_f0_parametric_factor_points.csv',index=False)
    validation.to_csv(args.out_dir/'rr100_parametric_recorded_validation_by_unit.csv',index=False)
    recorded_curves.to_csv(args.out_dir/'rr100_parametric_recorded_validation_curve_points.csv',index=False)
    summary.to_csv(args.out_dir/'rr100_parametric_recorded_validation_population_summary.csv',index=False)
    selected.to_csv(args.out_dir/'selected_unit_examples.csv',index=False)
    plot_summary(validation,summary,args.out_dir/'rr100_parametric_recorded_validation_summary.png',int(args.dpi))
    plot_examples(selected,recorded_curves,args.out_dir/'rr100_parametric_recorded_validation_selected_units.png',int(args.dpi))
    manifest={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'analysis':'parametric log-Gaussian fits to joint dynamic-F0 factors and independent recorded-grating SF validation',
        'fit_contract':'baseline plus log-Gaussian amplitude separately fit to sampled nonnegative SF and TF factors; product with F0 rank-one gain defines parametric surface',
        'validation_contract':'shape-only prediction at every positive SF in the recorded experiment; recorded curves averaged over the two orientations nearest the synthetic F0-preferred orientation; no recorded response used to fit parametric tuning',
        'support_caveat':'11 Allen sessions contribute 2,4,8,16 cpd; 4 Logan sessions contribute 1,2,4,8 cpd. Thus every unit has four positive recorded SFs. The parametric factor was fit on synthetic 1-11.313708 cpd, making Allen 16 cpd an explicitly labeled extrapolation. In-fit-support metrics are retained alongside full-real-support metrics.',
        'inputs':{key:file_identity(path) for key,path in paths.items()},
        'population_summary':summary.to_dict(orient='records'),
    }
    with (args.out_dir/'manifest.json').open('w',encoding='utf-8') as handle: json.dump(manifest,handle,indent=2)
    print(summary.to_string(index=False)); print('\nSelected examples:\n'+selected[['selection_role','rr100_index','sf_fit_r2','recorded_curve_pearson_r_parametric','recorded_curve_modulation_fraction']].to_string(index=False)); print(f"\nWrote {args.out_dir.resolve()}")


if __name__=='__main__': main()
