#!/usr/bin/env python3
"""Audit the 45.25-Hz edge control before extending RR100 SF-by-TF maps.

The primary parametric models were fit on 0.5--32 Hz.  This script keeps the
preferred orientation fixed from that primary fit, constructs the measured
positive-F0 surface at 45.25 Hz, tests the primary model's leave-edge-out
prediction, and reports how much exact-movie FEM power the extension recovers.
"""

from __future__ import annotations

import argparse
import hashlib
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
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_native_production_v1"
MODELS = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1"
POWER = ROOT / "outputs/fig4_active_sensing/rr100_kuang_input_power_checkpoint_01_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_tf45_edge_support_audit_v1"
TF_CORE_MAX = 32.0
TF_EDGE = float(32.0 * np.sqrt(2.0))
SF_MAX = float(8.0 * np.sqrt(2.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, default=PRODUCTION)
    parser.add_argument("--model-dir", type=Path, default=MODELS)
    parser.add_argument("--power-dir", type=Path, default=POWER)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def log_gaussian(frequency: np.ndarray, baseline: np.ndarray, amplitude: np.ndarray,
                 center_log2: np.ndarray, sigma_octaves: np.ndarray) -> np.ndarray:
    return baseline + amplitude * np.exp(
        -0.5 * ((np.log2(np.asarray(frequency, dtype=float)) - center_log2) / sigma_octaves) ** 2
    )


def prepare_folded_surface(summary_path: Path, models: pd.DataFrame) -> pd.DataFrame:
    data = pd.read_csv(summary_path)
    dynamic = data.loc[data["condition_kind"].eq("drifting_grating")].copy()
    phase = dynamic.groupby(
        ["rr100_index", "session", "orientation_deg", "spatial_cpd", "signed_temporal_hz"],
        as_index=False,
    ).agg(
        signed_f0_hz=("mean_rate_above_blank_hz", "mean"),
        n_carrier_phases=("phase_index", "nunique"),
    )
    phase["temporal_hz"] = phase["signed_temporal_hz"].abs()
    folded = phase.groupby(
        ["rr100_index", "session", "orientation_deg", "spatial_cpd", "temporal_hz"],
        as_index=False,
    ).agg(
        signed_f0_hz=("signed_f0_hz", "mean"),
        direction_difference_f0_hz=("signed_f0_hz", lambda values: float(values.max() - values.min())),
        n_signed_directions=("signed_temporal_hz", "nunique"),
        minimum_carrier_phases=("n_carrier_phases", "min"),
    )
    folded["positive_f0_hz"] = folded["signed_f0_hz"].clip(lower=0.0)
    selected = folded.merge(
        models[["rr100_index", "preferred_orientation_deg", "model_valid"]],
        on="rr100_index",
        validate="many_to_one",
    )
    selected = selected.loc[
        np.isclose(selected["orientation_deg"], selected["preferred_orientation_deg"])
        & (selected["spatial_cpd"] <= SF_MAX + 1e-9)
        & (selected["temporal_hz"] <= TF_EDGE + 1e-9)
    ].copy()
    return selected


def add_primary_model_prediction(edge: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    out = edge.merge(models, on="rr100_index", suffixes=("", "_model"), validate="many_to_one")
    sf_raw = log_gaussian(
        out["spatial_cpd"].to_numpy(float), out["sf_baseline"].to_numpy(float),
        out["sf_amplitude"].to_numpy(float), out["sf_center_log2_cpd"].to_numpy(float),
        out["sf_sigma_octaves"].to_numpy(float),
    )
    tf_raw = log_gaussian(
        out["temporal_hz"].to_numpy(float), out["tf_baseline"].to_numpy(float),
        out["tf_amplitude"].to_numpy(float), out["tf_center_log2_hz"].to_numpy(float),
        out["tf_sigma_octaves"].to_numpy(float),
    )
    out["core_model_extrapolated_positive_f0_hz"] = (
        out["joint_rank1_gain_f0_hz"].to_numpy(float)
        * sf_raw / out["sf_normalization_on_fit_support"].to_numpy(float)
        * tf_raw / out["tf_normalization_on_fit_support"].to_numpy(float)
    )
    return out


def unit_metrics(selected: pd.DataFrame, predicted_edge: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    core32 = selected.loc[np.isclose(selected["temporal_hz"], TF_CORE_MAX)].set_index(
        ["rr100_index", "spatial_cpd"]
    )
    edge45 = predicted_edge.set_index(["rr100_index", "spatial_cpd"])
    rows: list[dict[str, object]] = []
    for unit in models.loc[models["model_valid"].astype(bool), "rr100_index"].astype(int):
        a = core32.loc[unit].sort_index()
        b = edge45.loc[unit].sort_index()
        observed = b["positive_f0_hz"].to_numpy(float)
        prediction = b["core_model_extrapolated_positive_f0_hz"].to_numpy(float)
        at32 = a["positive_f0_hz"].to_numpy(float)
        shape_r = (
            float(np.corrcoef(observed, prediction)[0, 1])
            if np.std(observed) > 1e-12 and np.std(prediction) > 1e-12 else np.nan
        )
        sf_shape_32_45_r = (
            float(np.corrcoef(at32, observed)[0, 1])
            if np.std(at32) > 1e-12 and np.std(observed) > 1e-12 else np.nan
        )
        scale = max(float(np.max(observed)), 0.5)
        rows.append({
            "rr100_index": int(unit),
            "observed_mean_positive_f0_32_hz": float(np.mean(at32)),
            "observed_mean_positive_f0_45p25_hz": float(np.mean(observed)),
            "observed_45_to_32_mean_ratio": float(np.mean(observed) / max(np.mean(at32), 1e-12)),
            "fraction_sf_points_45_above_32": float(np.mean(observed > at32)),
            "sf_shape_r_32_vs_45": sf_shape_32_45_r,
            "edge_prediction_shape_r": shape_r,
            "edge_prediction_rmse_hz": float(np.sqrt(np.mean((observed - prediction) ** 2))),
            "edge_prediction_nrmse_by_observed_peak": float(np.sqrt(np.mean((observed - prediction) ** 2)) / scale),
            "edge_prediction_mean_positive_f0_hz": float(np.mean(prediction)),
            "edge_prediction_to_observed_mean_ratio": float(np.mean(prediction) / max(np.mean(observed), 1e-12)),
            "edge_minimum_carrier_phases": int(b["minimum_carrier_phases"].min()),
            "edge_n_signed_directions": int(b["n_signed_directions"].min()),
        })
    return pd.DataFrame(rows).merge(
        models[["rr100_index", "tf_sampled_preferred_hz", "preferred_tf_hz", "tf_fit_r2"]],
        on="rr100_index", validate="one_to_one",
    )


def power_extension(power_path: Path) -> pd.DataFrame:
    data = pd.read_csv(power_path)
    data = data.loc[data["condition"].eq("real_fem")].copy()
    data["weighted_dynamic_power"] = data["dynamic_power"] * data["spatial_mode_count"]
    total = float(data["weighted_dynamic_power"].sum())
    rows = []
    for label, tf_max in (("primary_0p5_to_32_hz", TF_CORE_MAX), ("edge_extended_0p5_to_45p25_hz", TF_EDGE)):
        mask = (
            (data["sf_bin_center_cpd"] >= 1.0)
            & (data["sf_bin_center_cpd"] <= SF_MAX)
            & (data["temporal_frequency_hz"] >= 0.5)
            & (data["temporal_frequency_hz"] <= tf_max)
        )
        power = float(data.loc[mask, "weighted_dynamic_power"].sum())
        rows.append({"support": label, "tf_max_hz": tf_max, "dynamic_power": power,
                     "fraction_of_total_positive_tf_dynamic_power": power / total})
    result = pd.DataFrame(rows)
    core = float(result.iloc[0]["fraction_of_total_positive_tf_dynamic_power"])
    extended = float(result.iloc[1]["fraction_of_total_positive_tf_dynamic_power"])
    result["absolute_fraction_gain_over_primary"] = extended - core
    result["relative_gain_over_primary"] = (extended - core) / core
    return result


def make_figure(metrics: pd.DataFrame, edge: pd.DataFrame, power: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    paired = metrics.loc[metrics["observed_mean_positive_f0_32_hz"] > 0.1]
    ax.scatter(paired["observed_mean_positive_f0_32_hz"], paired["observed_mean_positive_f0_45p25_hz"],
               s=24, alpha=0.65, color="#0072B2", edgecolor="none")
    hi = float(max(paired["observed_mean_positive_f0_32_hz"].max(), paired["observed_mean_positive_f0_45p25_hz"].max()))
    ax.plot([0, hi], [0, hi], "--", color="0.4", lw=1)
    ax.set(xlabel="mean positive F0 at 32 Hz", ylabel="mean positive F0 at 45.25 Hz",
           title="A  Measured edge response usually turns down")
    ax.text(0.03, 0.96, f"median 45/32 = {paired['observed_45_to_32_mean_ratio'].median():.2f}",
            transform=ax.transAxes, va="top")

    ax = axes[0, 1]
    observed = edge["positive_f0_hz"].to_numpy(float)
    prediction = edge["core_model_extrapolated_positive_f0_hz"].to_numpy(float)
    ax.scatter(observed, prediction, s=11, alpha=0.35, color="#009E73", edgecolor="none")
    hi = float(max(observed.max(), prediction.max()))
    ax.plot([0, hi], [0, hi], "--", color="0.4", lw=1)
    pooled_r = float(np.corrcoef(observed, prediction)[0, 1])
    pooled_rmse = float(np.sqrt(np.mean((observed - prediction) ** 2)))
    ax.set(xlabel="measured positive F0 at 45.25 Hz", ylabel="predicted from 0.5–32 Hz fit",
           title="B  Leave-edge-out extrapolation")
    ax.text(0.03, 0.96, f"pooled r = {pooled_r:.2f}\nRMSE = {pooled_rmse:.2f} Hz",
            transform=ax.transAxes, va="top")

    ax = axes[1, 0]
    values = metrics["edge_prediction_shape_r"].dropna()
    ax.hist(values, bins=np.linspace(-1, 1, 21), color="#56B4E9", edgecolor="white")
    ax.axvline(values.median(), color="#111111", lw=1.5)
    ax.set(xlabel="within-unit SF-shape r at 45.25 Hz", ylabel="RR100 units",
           title="C  Unit-level extrapolation is heterogeneous")
    ax.text(0.03, 0.96, f"median r = {values.median():.2f}\nn = {len(values)}",
            transform=ax.transAxes, va="top")

    ax = axes[1, 1]
    colors = ["#0072B2", "#D55E00"]
    fractions = 100.0 * power["fraction_of_total_positive_tf_dynamic_power"].to_numpy(float)
    bars = ax.bar([0, 1], fractions, color=colors, width=0.68)
    ax.set_xticks([0, 1], ["core\n≤32 Hz", "edge-assisted\n≤45.25 Hz"])
    ax.set(ylabel="FEM dynamic power in 1–11.31 cpd support (%)",
           title="D  Power recovered by qualified extension")
    for bar, value in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}%", ha="center")
    ax.set_ylim(0, max(50, float(fractions.max() + 8)))

    fig.suptitle("RR100 45.25-Hz edge-control audit", fontsize=15)
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.production_dir / "native_condition_unit_summary.csv"
    model_path = args.model_dir / "rr100_sf_tf_parametric_models.csv"
    power_path = args.power_dir / "checkpoint_01_sf_tf_power_long.csv"
    models = pd.read_csv(model_path)
    selected = prepare_folded_surface(summary_path, models)
    edge = selected.loc[np.isclose(selected["temporal_hz"], TF_EDGE) & selected["model_valid"].astype(bool)].copy()
    if edge.groupby("rr100_index").size().nunique() != 1 or edge.groupby("rr100_index").size().iloc[0] != 8:
        raise ValueError("Expected eight SF edge points for every valid RR100 unit")
    predicted_edge = add_primary_model_prediction(edge, models)
    metrics = unit_metrics(selected, predicted_edge, models)
    support = power_extension(power_path)

    predicted_edge.to_csv(args.out_dir / "tf45_edge_points.csv", index=False)
    metrics.to_csv(args.out_dir / "tf45_edge_unit_audit.csv", index=False)
    support.to_csv(args.out_dir / "fem_power_support_extension.csv", index=False)
    make_figure(metrics, predicted_edge, support, args.out_dir / "tf45_edge_support_audit", args.dpi)

    informative = metrics.loc[metrics["observed_mean_positive_f0_32_hz"] > 0.1]
    observed = predicted_edge["positive_f0_hz"].to_numpy(float)
    prediction = predicted_edge["core_model_extrapolated_positive_f0_hz"].to_numpy(float)
    edge_peak_units = metrics.loc[np.isclose(metrics["tf_sampled_preferred_hz"], TF_CORE_MAX)]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "RR100 TF=45.25-Hz edge support audit",
        "definitions": {
            "response": "phase-averaged, direction-folded positive F0 = max(mean rate above blank, 0)",
            "orientation": "fixed to the preferred orientation selected using primary 0.5--32-Hz support",
            "edge_prediction": "leave-edge-out extrapolation of the existing primary parametric SFxTF model",
            "power_fraction": "radially binned exact-movie dynamic power weighted by spatial Fourier mode count",
        },
        "sampling_caveats": {
            "movie_frame_rate_hz": 120.0,
            "edge_frames_per_cycle": 120.0 / TF_EDGE,
            "edge_carrier_phases": int(predicted_edge["minimum_carrier_phases"].min()),
            "edge_signed_directions": int(predicted_edge["n_signed_directions"].min()),
            "interpretation": "qualified edge-assisted extension, not equivalent to densely sampled primary support",
        },
        "summary": {
            "valid_units": int(len(metrics)),
            "informative_units_with_mean_32hz_above_0p1": int(len(informative)),
            "median_observed_45_to_32_ratio_informative": float(informative["observed_45_to_32_mean_ratio"].median()),
            "fraction_units_mean_45_above_mean_32": float(np.mean(metrics["observed_mean_positive_f0_45p25_hz"] > metrics["observed_mean_positive_f0_32_hz"])),
            "core_sampled_peak_at_32hz_units": int(len(edge_peak_units)),
            "median_45_to_32_ratio_for_core_32hz_peak_units": float(edge_peak_units["observed_45_to_32_mean_ratio"].median()),
            "pooled_edge_prediction_r": float(np.corrcoef(observed, prediction)[0, 1]),
            "pooled_edge_prediction_rmse_hz": float(np.sqrt(np.mean((observed - prediction) ** 2))),
            "median_unit_edge_prediction_shape_r": float(metrics["edge_prediction_shape_r"].median()),
            "median_unit_edge_prediction_nrmse": float(metrics["edge_prediction_nrmse_by_observed_peak"].median()),
            "spearman_observed_vs_predicted_edge_mean": float(spearmanr(
                metrics["observed_mean_positive_f0_45p25_hz"], metrics["edge_prediction_mean_positive_f0_hz"]
            ).statistic),
            "core_power_fraction": float(support.iloc[0]["fraction_of_total_positive_tf_dynamic_power"]),
            "edge_extended_power_fraction": float(support.iloc[1]["fraction_of_total_positive_tf_dynamic_power"]),
            "absolute_power_fraction_gain": float(support.iloc[1]["absolute_fraction_gain_over_primary"]),
            "relative_power_gain_over_primary": float(support.iloc[1]["relative_gain_over_primary"]),
        },
        "inputs": {"condition_summary": file_identity(summary_path), "models": file_identity(model_path),
                   "power_map": file_identity(power_path)},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# RR100 45.25-Hz edge-support audit\n\n"
        "This checkpoint asks whether the existing 0.5–32 Hz positive-F0 parametric model can be "
        "used up to the measured 45.25-Hz edge control. The edge was not used to choose orientation or "
        "fit the primary model. It is therefore a leave-edge-out check. Because 45.25 Hz has one carrier "
        "phase and only 2.65 frames/cycle at 120 Hz, all downstream use must retain an `edge_assisted` label.\n"
    )
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
