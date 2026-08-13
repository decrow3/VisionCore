#!/usr/bin/env python3
"""Make the first unit-level Kuang-style SF-by-TF overlap checkpoint.

This is deliberately a mechanistic proxy, not a firing-rate prediction:

    overlap density(sf, tf) = normalized positive-F0 sensitivity(sf, tf)^2
                              * exact-movie FEM dynamic power(sf, tf)

The fixed-eye tuning is measured with drifting gratings at true zero gaze.  The
FEM power comes from the exact 51x51 retinal movie selected at checkpoint 01.
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1"
F0_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
POWER_DIR = ROOT / "outputs/fig4_active_sensing/rr100_kuang_input_power_checkpoint_01_v1"
EDGE_DIR = ROOT / "outputs/fig4_active_sensing/rr100_tf45_edge_support_audit_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_kuang_unit_overlap_checkpoint_02_v1"
SF_MIN, SF_MAX = 1.0, float(8.0 * np.sqrt(2.0))
TF_MIN, TF_MAX = 0.5, 32.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--f0-dir", type=Path, default=F0_DIR)
    parser.add_argument("--power-dir", type=Path, default=POWER_DIR)
    parser.add_argument("--edge-dir", type=Path, default=EDGE_DIR)
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
    return {"path": str(resolved), "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def log_gaussian(frequency: np.ndarray, baseline: float, amplitude: float,
                 center_log2: float, sigma_octaves: float) -> np.ndarray:
    return baseline + amplitude * np.exp(
        -0.5 * ((np.log2(np.asarray(frequency, dtype=float)) - center_log2) / sigma_octaves) ** 2
    )


def edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    middle = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (middle[0] - values[0]), middle, values[-1] + (values[-1] - middle[-1])]


def surface(model: pd.Series, sf: np.ndarray, tf: np.ndarray) -> np.ndarray:
    sf_factor = log_gaussian(
        sf, model["sf_baseline"], model["sf_amplitude"],
        model["sf_center_log2_cpd"], model["sf_sigma_octaves"],
    ) / float(model["sf_normalization_on_fit_support"])
    tf_factor = log_gaussian(
        tf, model["tf_baseline"], model["tf_amplitude"],
        model["tf_center_log2_hz"], model["tf_sigma_octaves"],
    ) / float(model["tf_normalization_on_fit_support"])
    return np.maximum(np.outer(sf_factor, tf_factor), 0.0)


def load_power(power_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    npz_path = power_dir / "checkpoint_01_retinal_movies_and_power.npz"
    long_path = power_dir / "checkpoint_01_sf_tf_power_long.csv"
    selection_path = power_dir / "checkpoint_01_image_selection.csv"
    archive = np.load(npz_path)
    sf_all = archive["sf_bin_centers_cpd"].astype(float)
    tf_all = archive["temporal_frequency_hz"].astype(float)
    radial_mean_power = archive["real_fem_dynamic_radial_power"].astype(float)
    long = pd.read_csv(long_path)
    mode_count = (
        long.loc[long["condition"].eq("real_fem")]
        .groupby("sf_bin_center_cpd")["spatial_mode_count"].first().reindex(sf_all).to_numpy(float)
    )
    annular_power = radial_mean_power * mode_count[:, None]
    sf_mask = (sf_all >= SF_MIN) & (sf_all <= SF_MAX)
    tf_mask = (tf_all >= TF_MIN) & (tf_all <= TF_MAX)
    selection = pd.read_csv(selection_path).iloc[0].to_dict()
    return sf_all[sf_mask], tf_all[tf_mask], annular_power[np.ix_(sf_mask, tf_mask)], selection


def score_units(models: pd.DataFrame, sf: np.ndarray, tf: np.ndarray, power: np.ndarray) -> pd.DataFrame:
    rows = []
    log_sf = np.log2(sf)[:, None]
    log_tf = np.log2(tf)[None, :]
    for _, model in models.loc[models["model_valid"].astype(bool)].iterrows():
        gain = surface(model, sf, tf)
        overlap = power * gain**2
        total = float(overlap.sum())
        rows.append({
            "rr100_index": int(model["rr100_index"]),
            "normalized_power_overlap": float(total / power.sum()),
            "overlap_sf_centroid_cpd": float(2.0 ** ((overlap * log_sf).sum() / total)) if total > 0 else np.nan,
            "overlap_tf_centroid_hz": float(2.0 ** ((overlap * log_tf).sum() / total)) if total > 0 else np.nan,
            "gain_scaled_overlap_arbitrary": float(model["joint_rank1_gain_f0_hz"] ** 2 * total),
        })
    return models.merge(pd.DataFrame(rows), on="rr100_index", how="left", validate="one_to_one")


def select_examples(scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    quality = scored.loc[
        scored["model_valid"].astype(bool)
        & (scored["sf_fit_r2"] >= 0.70)
        & (scored["tf_fit_r2"] >= 0.70)
        & (scored["joint_parametric_surface_r2"] >= 0.50)
    ].copy()
    low_cut = float(quality["preferred_sf_cpd"].quantile(1.0 / 3.0))
    high_cut = float(quality["preferred_sf_cpd"].quantile(2.0 / 3.0))
    median_gain = float(quality["joint_rank1_gain_f0_hz"].median())
    definitions = [
        ("low-SF positive overlap", quality["preferred_sf_cpd"] <= low_cut, "max"),
        ("middle-SF positive overlap", (quality["preferred_sf_cpd"] > low_cut) & (quality["preferred_sf_cpd"] <= high_cut), "max"),
        ("high-SF positive overlap", quality["preferred_sf_cpd"] > high_cut, "max"),
        ("strong-unit spectral mismatch control", quality["joint_rank1_gain_f0_hz"] >= median_gain, "min"),
    ]
    choices = []
    excluded: set[int] = set()
    for role, mask, direction in definitions:
        candidates = quality.loc[mask & ~quality["rr100_index"].isin(excluded)].copy()
        candidates = candidates.sort_values(
            ["normalized_power_overlap", "joint_parametric_surface_r2", "rr100_index"],
            ascending=[direction == "min", False, True],
        )
        choice = candidates.iloc[0].copy()
        choice["selection_role"] = role
        choice["selection_pool_n"] = int(len(candidates))
        choice["selection_extremum"] = "lowest normalized overlap" if direction == "min" else "highest normalized overlap"
        choices.append(choice)
        excluded.add(int(choice["rr100_index"]))
    selected = pd.DataFrame(choices)
    selected.insert(0, "display_order", np.arange(1, len(selected) + 1))
    return selected, {"quality_pool_n": int(len(quality)), "low_sf_tertile_cut_cpd": low_cut,
                      "high_sf_tertile_cut_cpd": high_cut, "quality_pool_median_gain_f0_hz": median_gain}


def sampled_prediction(model: pd.Series, sf: np.ndarray, tf: np.ndarray) -> np.ndarray:
    return float(model["joint_rank1_gain_f0_hz"]) * surface(model, sf, tf)


def setup_axes(ax: plt.Axes, show_y: bool = True) -> None:
    sf_ticks = np.asarray([1, 2, 4, 8], dtype=float)
    tf_ticks = np.asarray([0.5, 1, 2, 4, 8, 16, 32], dtype=float)
    ax.set_xticks(np.log2(sf_ticks), [f"{value:g}" for value in sf_ticks])
    ax.set_yticks(np.log2(tf_ticks), [f"{value:g}" for value in tf_ticks] if show_y else [])
    ax.set_xlabel("SF (cpd)")
    if show_y:
        ax.set_ylabel("TF (Hz)")
    ax.tick_params(length=2, labelsize=7)


def make_figure(selected: pd.DataFrame, observed_points: pd.DataFrame, sf: np.ndarray, tf: np.ndarray,
                power: np.ndarray, out_path: Path, dpi: int) -> pd.DataFrame:
    nrows = len(selected)
    fig, axes = plt.subplots(nrows, 5, figsize=(14.2, 2.58 * nrows), constrained_layout=True)
    power_db = 10.0 * np.log10(np.maximum(power / power.max(), 1e-6))
    sf_edge_power, tf_edge_power = edges(np.log2(sf)), edges(np.log2(tf))
    values_long = []
    ims = {}
    for row_index, (_, model) in enumerate(selected.iterrows()):
        unit = int(model["rr100_index"])
        points = observed_points.loc[observed_points["rr100_index"].eq(unit)].copy()
        obs = points.pivot(index="spatial_cpd", columns="temporal_hz", values="observed_positive_f0_hz").sort_index().sort_index(axis=1)
        obs_sf = obs.index.to_numpy(float)
        obs_tf = obs.columns.to_numpy(float)
        prediction_sampled = sampled_prediction(model, obs_sf, obs_tf)
        observed = obs.to_numpy(float)
        normalization = max(float(observed.max()), 1e-12)
        residual = (observed - prediction_sampled) / normalization
        dense_gain = surface(model, sf, tf)
        overlap = power * dense_gain**2
        overlap_db = 10.0 * np.log10(np.maximum(overlap / power.max(), 1e-6))

        ax = axes[row_index, 0]
        ims["response"] = ax.pcolormesh(edges(np.log2(obs_sf)), edges(np.log2(obs_tf)),
                                         (observed / normalization).T, shading="flat", cmap="magma", vmin=0, vmax=1)
        ax = axes[row_index, 1]
        ax.pcolormesh(edges(np.log2(obs_sf)), edges(np.log2(obs_tf)),
                      (prediction_sampled / normalization).T, shading="flat", cmap="magma", vmin=0, vmax=1)
        ax = axes[row_index, 2]
        ims["residual"] = ax.pcolormesh(edges(np.log2(obs_sf)), edges(np.log2(obs_tf)), residual.T,
                                         shading="flat", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
        ax = axes[row_index, 3]
        ims["power"] = ax.pcolormesh(sf_edge_power, tf_edge_power, power_db.T,
                                      shading="flat", cmap="viridis", vmin=-50, vmax=0)
        ax = axes[row_index, 4]
        ax.pcolormesh(sf_edge_power, tf_edge_power, overlap_db.T,
                      shading="flat", cmap="viridis", vmin=-50, vmax=0)

        for column, axis in enumerate(axes[row_index]):
            setup_axes(axis, show_y=(column == 0))
        axes[row_index, 0].set_ylabel(
            f"{model['selection_role']}\nRR100 {unit}\nTF (Hz)", fontsize=7.5
        )
        axes[row_index, 4].text(
            0.03, 0.97,
            f"overlap={model['normalized_power_overlap']:.3f}\ncentroid={model['overlap_sf_centroid_cpd']:.2f} cpd, {model['overlap_tf_centroid_hz']:.1f} Hz",
            transform=axes[row_index, 4].transAxes, va="top", color="white", fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.42, "edgecolor": "none", "pad": 2},
        )
        for i, sf_value in enumerate(sf):
            for j, tf_value in enumerate(tf):
                values_long.append({
                    "rr100_index": unit, "selection_role": model["selection_role"],
                    "spatial_cpd": float(sf_value), "temporal_hz": float(tf_value),
                    "normalized_parametric_sensitivity": float(dense_gain[i, j]),
                    "annular_fem_dynamic_power": float(power[i, j]),
                    "normalized_spectral_overlap_density": float(overlap[i, j]),
                })

    headers = ["Measured fixed-eye\npositive F0", "Parametric fixed-eye\nSF×TF tuning",
               "Measured − fitted\n(fraction of peak)", "Exact-movie FEM\ndynamic power", "Predicted spectral overlap\nG² × power"]
    for axis, title in zip(axes[0], headers):
        axis.set_title(title, fontsize=9)
    cbar = fig.colorbar(ims["response"], ax=axes[:, :2], shrink=0.7, pad=0.01)
    cbar.set_label("fraction of measured unit peak", fontsize=8)
    cbar = fig.colorbar(ims["residual"], ax=axes[:, 2], shrink=0.7, pad=0.01)
    cbar.set_label("signed residual", fontsize=8)
    cbar = fig.colorbar(ims["power"], ax=axes[:, 3:], shrink=0.7, pad=0.01)
    cbar.set_label("dB relative to maximum FEM annular power", fontsize=8)
    fig.suptitle(
        "RR100 unit-first active-sensing mechanism: fixed-eye tuning × FEM power\n"
        "Primary support only (1–11.31 cpd, 0.5–32 Hz); overlap is a spectral proxy, not predicted firing rate",
        fontsize=13,
    )
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return pd.DataFrame(values_long)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    model_path = args.model_dir / "rr100_sf_tf_parametric_models.csv"
    points_path = args.f0_dir / "f0_surface_fit_and_residual_points.csv"
    edge_manifest_path = args.edge_dir / "manifest.json"
    models = pd.read_csv(model_path)
    observed_points = pd.read_csv(points_path)
    sf, tf, power, image_selection = load_power(args.power_dir)
    scored = score_units(models, sf, tf, power)
    selected, thresholds = select_examples(scored)
    edge_units = pd.read_csv(args.edge_dir / "tf45_edge_unit_audit.csv")
    selected = selected.merge(
        edge_units[["rr100_index", "observed_45_to_32_mean_ratio", "edge_prediction_shape_r",
                    "edge_prediction_nrmse_by_observed_peak"]],
        on="rr100_index", how="left", validate="one_to_one",
    )
    selected.to_csv(args.out_dir / "selected_unit_roles.csv", index=False)
    scored.to_csv(args.out_dir / "rr100_spectral_overlap_scores.csv", index=False)
    values = make_figure(selected, observed_points, sf, tf, power,
                         args.out_dir / "checkpoint_02_unit_tuning_power_overlap", args.dpi)
    values.to_csv(args.out_dir / "selected_unit_overlap_maps_long.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "unit-first RR100 fixed-eye tuning by exact-movie FEM power",
        "primary_support": {"sf_cpd": [SF_MIN, SF_MAX], "tf_hz": [TF_MIN, TF_MAX]},
        "proxy_definition": "normalized parametric positive-F0 sensitivity squared times annular FEM dynamic power",
        "interpretation": "spectral overlap density; not a calibrated firing-rate or information prediction",
        "power_definition": "radial mean Fourier power times number of spatial Fourier modes in each annulus",
        "orientation_limitation": "retinal power is radialized; orientation-specific image power is not yet matched to unit orientation",
        "selection": {
            "quality_gate": "valid model; SF R2 >= 0.70; TF R2 >= 0.70; joint surface R2 >= 0.50",
            "roles_defined_before_selection": [
                "highest normalized overlap in low-SF preference tertile",
                "highest normalized overlap in middle-SF preference tertile",
                "highest normalized overlap in high-SF preference tertile",
                "lowest normalized overlap among units with at least median response gain",
            ],
            **thresholds,
            "selected_rr100_indices": selected["rr100_index"].astype(int).tolist(),
        },
        "edge_policy": {
            "primary_maps_stop_at_hz": 32.0,
            "45p25_hz_use": "sensitivity/coverage audit only; not silently included in primary maps",
            "audit_manifest": file_identity(edge_manifest_path),
        },
        "image_selection": image_selection,
        "inputs": {
            "models": file_identity(model_path),
            "observed_f0_points": file_identity(points_path),
            "power_npz": file_identity(args.power_dir / "checkpoint_01_retinal_movies_and_power.npz"),
            "power_long": file_identity(args.power_dir / "checkpoint_01_sf_tf_power_long.csv"),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Checkpoint 02: unit-level tuning × FEM power\n\n"
        "Each row traces a selected RR100 unit from the measured fixed-eye positive-F0 surface, through the "
        "separable parametric approximation and residual, to the exact-movie FEM power map and their spectral "
        "overlap. The overlap uses normalized sensitivity squared, matching a linear-filter power calculation. "
        "It is a hypothesis-generating proxy, not a calibrated response prediction. The primary map stops at "
        "32 Hz; the measured 45.25-Hz edge is retained only in the separate support audit.\n"
    )
    print(selected[["display_order", "selection_role", "rr100_index", "preferred_sf_cpd", "preferred_tf_hz",
                    "normalized_power_overlap", "joint_parametric_surface_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
