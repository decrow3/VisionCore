#!/usr/bin/env python3
"""Fit and audit separable SF/TF factors for the native RR100 zero-gaze sweep.

The primary response is the dynamic F1 amplitude.  Carrier phase is averaged
first.  Three complete surfaces are then fit per unit: preferred-orientation
signed TF, preferred-orientation |TF|, and orientation-marginal |TF|.  A raw
rank-one SVD supplies the least-squares multiplicative factors and a directly
inspectable residual; absolute response strength is retained beside every
normalized factor so nearly silent units cannot look meaningfully tuned.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_native_production_v1"
)
DEFAULT_OUT = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_factorization_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--responsive-max-f1-hz", type=float, default=0.1)
    parser.add_argument("--minimum-example-max-f1-hz", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def complete_matrix(
    frame: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
) -> pd.DataFrame:
    matrix = frame.pivot(index=index, columns=columns, values=values).sort_index().sort_index(axis=1)
    if matrix.isna().any().any():
        missing = int(matrix.isna().sum().sum())
        raise ValueError(f"Incomplete {index} x {columns} surface: {missing} missing cells")
    return matrix


def halfmax_metrics(frequencies: np.ndarray, factor: np.ndarray) -> dict[str, float | bool]:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    factor = np.asarray(factor, dtype=np.float64)
    selected = factor >= 0.5 * float(np.max(factor))
    if not np.any(selected):
        return {
            "halfmax_min": float("nan"),
            "halfmax_max": float("nan"),
            "halfmax_bandwidth_octaves": float("nan"),
            "halfmax_touches_low_edge": False,
            "halfmax_touches_high_edge": False,
        }
    chosen = frequencies[selected]
    return {
        "halfmax_min": float(np.min(chosen)),
        "halfmax_max": float(np.max(chosen)),
        "halfmax_bandwidth_octaves": float(np.log2(np.max(chosen) / np.min(chosen))),
        "halfmax_touches_low_edge": bool(selected[0]),
        "halfmax_touches_high_edge": bool(selected[-1]),
    }


def rank_one_fit(
    matrix: pd.DataFrame,
    *,
    rr100_index: int,
    session: str,
    surface_definition: str,
    preferred_orientation_deg: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    observed = matrix.to_numpy(dtype=np.float64)
    left, singular, right_t = np.linalg.svd(observed, full_matrices=False)
    sf_raw = left[:, 0].copy()
    tf_raw = right_t[0].copy()
    if float(np.sum(sf_raw)) < 0:
        sf_raw *= -1.0
        tf_raw *= -1.0
    # A nonnegative response surface has nonnegative leading vectors up to
    # numerical sign.  Clip only machine-scale negative leakage.
    sf_raw = np.maximum(sf_raw, 0.0)
    tf_raw = np.maximum(tf_raw, 0.0)
    reconstruction = float(singular[0]) * np.outer(sf_raw, tf_raw)
    residual = observed - reconstruction
    total_energy = float(np.sum(observed**2))
    residual_energy = float(np.sum(residual**2))
    centered_energy = float(np.sum((observed - float(np.mean(observed))) ** 2))
    response_range = float(np.max(observed) - np.min(observed))
    response_max = float(np.max(observed))
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    sf_norm = sf_raw / max(float(np.max(sf_raw)), 1e-15)
    tf_norm = tf_raw / max(float(np.max(tf_raw)), 1e-15)
    gain = float(singular[0] * np.max(sf_raw) * np.max(tf_raw))
    sf_values = matrix.index.to_numpy(dtype=np.float64)
    tf_values = matrix.columns.to_numpy(dtype=np.float64)
    peak_sf = float(sf_values[int(np.argmax(sf_norm))])
    peak_tf = float(tf_values[int(np.argmax(tf_norm))])
    tf_is_signed = "signed_tf" in surface_definition
    sf_halfmax = halfmax_metrics(sf_values, sf_norm)
    tf_halfmax = (
        {
            "halfmax_min": float("nan"),
            "halfmax_max": float("nan"),
            "halfmax_bandwidth_octaves": float("nan"),
            "halfmax_touches_low_edge": False,
            "halfmax_touches_high_edge": False,
        }
        if tf_is_signed
        else halfmax_metrics(tf_values, tf_norm)
    )
    weighted_sf = float(2.0 ** np.average(np.log2(sf_values), weights=np.maximum(sf_norm, 1e-15)))
    weighted_tf = (
        float("nan")
        if tf_is_signed
        else float(2.0 ** np.average(np.log2(tf_values), weights=np.maximum(tf_norm, 1e-15)))
    )
    summary = {
        "rr100_index": int(rr100_index),
        "session": session,
        "surface_definition": surface_definition,
        "preferred_orientation_deg": float(preferred_orientation_deg),
        "n_spatial_frequencies": int(len(sf_values)),
        "n_temporal_frequencies": int(len(tf_values)),
        "surface_mean_f1_hz": float(np.mean(observed)),
        "surface_maximum_f1_hz": response_max,
        "surface_range_f1_hz": response_range,
        "rank1_gain_f1_hz": gain,
        "rank1_energy_fraction": float(1.0 - residual_energy / max(total_energy, 1e-30)),
        "rank1_centered_r2": float(1.0 - residual_energy / centered_energy) if centered_energy > 0 else float("nan"),
        "rank1_relative_rmse": float(np.sqrt(residual_energy / max(total_energy, 1e-30))),
        "interaction_residual_rms_hz": residual_rms,
        "interaction_rms_over_surface_range": residual_rms / max(response_range, 1e-15),
        "interaction_rms_over_surface_maximum": residual_rms / max(response_max, 1e-15),
        "preferred_sf_cpd_sampled": peak_sf,
        "preferred_tf_hz_sampled": peak_tf,
        "weighted_geometric_sf_cpd": weighted_sf,
        "weighted_geometric_tf_hz": weighted_tf,
        "sf_peak_at_fit_edge": bool(np.isclose(peak_sf, sf_values[0]) or np.isclose(peak_sf, sf_values[-1])),
        "tf_peak_at_fit_edge": bool(
            np.isclose(abs(peak_tf), float(np.min(np.abs(tf_values))))
            or np.isclose(abs(peak_tf), float(np.max(np.abs(tf_values))))
        ),
        "sf_halfmax_min_cpd": sf_halfmax["halfmax_min"],
        "sf_halfmax_max_cpd": sf_halfmax["halfmax_max"],
        "sf_halfmax_bandwidth_octaves": sf_halfmax["halfmax_bandwidth_octaves"],
        "sf_halfmax_touches_low_edge": sf_halfmax["halfmax_touches_low_edge"],
        "sf_halfmax_touches_high_edge": sf_halfmax["halfmax_touches_high_edge"],
        "tf_halfmax_min_hz": tf_halfmax["halfmax_min"],
        "tf_halfmax_max_hz": tf_halfmax["halfmax_max"],
        "tf_halfmax_bandwidth_octaves": tf_halfmax["halfmax_bandwidth_octaves"],
        "tf_halfmax_touches_low_edge": tf_halfmax["halfmax_touches_low_edge"],
        "tf_halfmax_touches_high_edge": tf_halfmax["halfmax_touches_high_edge"],
    }
    factor_rows = []
    for value, raw, normalized in zip(sf_values, sf_raw, sf_norm):
        factor_rows.append(
            {
                "rr100_index": int(rr100_index),
                "session": session,
                "surface_definition": surface_definition,
                "preferred_orientation_deg": float(preferred_orientation_deg),
                "axis": "spatial_frequency",
                "frequency": float(value),
                "raw_singular_vector": float(raw),
                "normalized_factor": float(normalized),
                "rank1_gain_f1_hz": gain,
            }
        )
    for value, raw, normalized in zip(tf_values, tf_raw, tf_norm):
        factor_rows.append(
            {
                "rr100_index": int(rr100_index),
                "session": session,
                "surface_definition": surface_definition,
                "preferred_orientation_deg": float(preferred_orientation_deg),
                "axis": "signed_temporal_frequency" if tf_is_signed else "temporal_frequency_magnitude",
                "frequency": float(value),
                "raw_singular_vector": float(raw),
                "normalized_factor": float(normalized),
                "rank1_gain_f1_hz": gain,
            }
        )
    point_rows = []
    for sf_index, sf in enumerate(sf_values):
        for tf_index, tf in enumerate(tf_values):
            point_rows.append(
                {
                    "rr100_index": int(rr100_index),
                    "session": session,
                    "surface_definition": surface_definition,
                    "preferred_orientation_deg": float(preferred_orientation_deg),
                    "spatial_cpd": float(sf),
                    "temporal_hz": float(tf),
                    "observed_f1_hz": float(observed[sf_index, tf_index]),
                    "rank1_reconstructed_f1_hz": float(reconstruction[sf_index, tf_index]),
                    "interaction_residual_f1_hz": float(residual[sf_index, tf_index]),
                }
            )
    return summary, pd.DataFrame(factor_rows), pd.DataFrame(point_rows)


def direction_metrics(preferred_signed: pd.DataFrame) -> pd.DataFrame:
    work = preferred_signed.copy()
    work["tf_magnitude"] = work["signed_temporal_hz"].abs()
    positive = work[work["signed_temporal_hz"] > 0].pivot_table(
        index=["rr100_index", "session", "spatial_cpd"],
        columns="tf_magnitude",
        values="f1_amplitude_hz",
    )
    negative = work[work["signed_temporal_hz"] < 0].pivot_table(
        index=["rr100_index", "session", "spatial_cpd"],
        columns="tf_magnitude",
        values="f1_amplitude_hz",
    )
    rows = []
    for key in positive.index:
        if key not in negative.index:
            continue
        pos = positive.loc[key].to_numpy(dtype=np.float64)
        neg = negative.loc[key].to_numpy(dtype=np.float64)
        diff = pos - neg
        rows.append(
            {
                "rr100_index": int(key[0]),
                "session": str(key[1]),
                "spatial_cpd": float(key[2]),
                "direction_rms_difference_hz": float(np.sqrt(np.mean(diff**2))),
                "maximum_direction_difference_hz": float(np.max(np.abs(diff))),
                "direction_signed_sum_difference_hz": float(np.sum(diff)),
                "direction_total_f1_hz": float(np.sum(pos + neg)),
            }
        )
    per_sf = pd.DataFrame(rows)
    unit_rows = []
    for (unit, session), sub in per_sf.groupby(["rr100_index", "session"]):
        signed_sum = float(sub["direction_signed_sum_difference_hz"].sum())
        total = float(sub["direction_total_f1_hz"].sum())
        unit_rows.append(
            {
                "rr100_index": int(unit),
                "session": session,
                "direction_rms_difference_hz": float(np.sqrt(np.mean(sub["direction_rms_difference_hz"] ** 2))),
                "maximum_direction_difference_hz": float(sub["maximum_direction_difference_hz"].max()),
                "direction_signed_bias": signed_sum / max(total, 1e-15),
            }
        )
    return pd.DataFrame(unit_rows)


def choose_examples(summary: pd.DataFrame, minimum_max_f1: float) -> pd.DataFrame:
    primary = summary[summary["surface_definition"].eq("preferred_orientation_abs_tf")].copy()
    primary["response_separability_score"] = (
        primary["surface_range_f1_hz"] * primary["rank1_centered_r2"].clip(lower=0.0)
    )
    substantial = primary[primary["surface_maximum_f1_hz"] >= float(minimum_max_f1)]
    highest_available_sf = float(substantial["preferred_sf_cpd_sampled"].max())
    role_specs: list[tuple[str, pd.DataFrame, str, bool, str]] = [
        (
            "strong_separable_response",
            substantial[substantial["rank1_centered_r2"] >= 0.9],
            "response_separability_score",
            False,
            "largest absolute tuning range weighted by centered rank-one fit quality among R2>=0.9 units",
        ),
        (
            "largest_relative_interaction",
            substantial,
            "interaction_rms_over_surface_range",
            False,
            "largest rank-one residual RMS divided by raw surface range among units with max F1>=1 Hz",
        ),
        (
            "largest_direction_asymmetry",
            substantial,
            "maximum_direction_difference_hz",
            False,
            "largest matched +TF versus -TF F1 difference on the preferred-orientation surface",
        ),
        (
            "low_sf_preference_control",
            substantial[substantial["preferred_sf_cpd_sampled"] <= 2.0],
            "surface_maximum_f1_hz",
            False,
            "strongest remaining unit with sampled preferred SF<=2 cpd",
        ),
        (
            "highest_available_sf_preference_control",
            substantial[np.isclose(substantial["preferred_sf_cpd_sampled"], highest_available_sf)],
            "surface_maximum_f1_hz",
            False,
            f"strongest remaining unit at the highest sampled preferred SF present among substantive responders ({highest_available_sf:g} cpd); no >=8 cpd preferred unit was available",
        ),
        (
            "weak_response_control",
            primary,
            "surface_maximum_f1_hz",
            True,
            "smallest maximum F1 over the primary preferred-orientation surface",
        ),
    ]
    used: set[int] = set()
    rows = []
    for role, candidates, criterion, ascending, rule in role_specs:
        candidates = candidates[np.isfinite(candidates[criterion])].sort_values(criterion, ascending=ascending)
        available = candidates[~candidates["rr100_index"].isin(used)]
        if available.empty:
            continue
        row = available.iloc[0].to_dict()
        used.add(int(row["rr100_index"]))
        row.update(
            {
                "selection_role": role,
                "selection_criterion": criterion,
                "selection_value": float(row[criterion]),
                "selection_rule": rule,
                "selection_method": "predefined_algorithmic_role",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center_zero: bool = False,
) -> None:
    values = matrix.to_numpy(dtype=float)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if center_zero and vmax and vmax > 0 else None
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        norm=norm,
    )
    ax.set_yticks(np.arange(len(matrix.index)), [f"{v:g}" for v in matrix.index], fontsize=6)
    tick_positions = np.unique(np.linspace(0, len(matrix.columns) - 1, min(7, len(matrix.columns))).round().astype(int))
    ax.set_xticks(tick_positions, [f"{matrix.columns[i]:g}" for i in tick_positions], rotation=45, ha="right", fontsize=6)
    ax.set(xlabel="TF (Hz)", ylabel="SF (cpd)", title=title)
    ax.title.set_fontsize(8)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)


def plot_selected(
    points: pd.DataFrame,
    factors: pd.DataFrame,
    selected: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    n_rows = len(selected)
    fig, axes = plt.subplots(
        n_rows,
        6,
        figsize=(21.5, 3.55 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    for row_index, selected_row in selected.reset_index(drop=True).iterrows():
        unit = int(selected_row["rr100_index"])
        role = str(selected_row["selection_role"])
        signed = points[(points["rr100_index"].eq(unit)) & (points["surface_definition"].eq("preferred_orientation_signed_tf"))]
        folded = points[(points["rr100_index"].eq(unit)) & (points["surface_definition"].eq("preferred_orientation_abs_tf"))]
        observed_signed = complete_matrix(signed, index="spatial_cpd", columns="temporal_hz", values="observed_f1_hz")
        observed = complete_matrix(folded, index="spatial_cpd", columns="temporal_hz", values="observed_f1_hz")
        reconstructed = complete_matrix(folded, index="spatial_cpd", columns="temporal_hz", values="rank1_reconstructed_f1_hz")
        residual = complete_matrix(folded, index="spatial_cpd", columns="temporal_hz", values="interaction_residual_f1_hz")
        shared_max = float(max(observed_signed.to_numpy().max(), observed.to_numpy().max(), reconstructed.to_numpy().max()))
        residual_max = float(np.max(np.abs(residual.to_numpy())))
        metadata_title = (
            f"RR100 {unit}: {role}\n"
            f"ori {selected_row['preferred_orientation_deg']:g}°; max {selected_row['surface_maximum_f1_hz']:.3g} Hz; "
            f"centered R² {selected_row['rank1_centered_r2']:.3f}; residual/range {selected_row['interaction_rms_over_surface_range']:.3f}\n"
            "observed signed TF F1"
        )
        heatmap(axes[row_index, 0], observed_signed, title=metadata_title, cmap="viridis", vmin=0, vmax=shared_max)
        heatmap(axes[row_index, 1], observed, title="observed |TF| F1", cmap="viridis", vmin=0, vmax=shared_max)
        heatmap(axes[row_index, 2], reconstructed, title="rank-one reconstruction", cmap="viridis", vmin=0, vmax=shared_max)
        heatmap(
            axes[row_index, 3],
            residual,
            title="observed - rank one",
            cmap="RdBu_r",
            vmin=-residual_max,
            vmax=residual_max,
            center_zero=True,
        )
        unit_factors = factors[
            (factors["rr100_index"].eq(unit))
            & (factors["surface_definition"].eq("preferred_orientation_abs_tf"))
        ]
        sf_factor = unit_factors[unit_factors["axis"].eq("spatial_frequency")].sort_values("frequency")
        tf_factor = unit_factors[unit_factors["axis"].eq("temporal_frequency_magnitude")].sort_values("frequency")
        axes[row_index, 4].plot(sf_factor["frequency"], sf_factor["normalized_factor"], "o-", color="#0072B2", lw=1.5, ms=3)
        axes[row_index, 4].set(xscale="log", xlabel="SF (cpd)", ylabel="normalized factor", ylim=(-0.03, 1.05), title="separable SF factor")
        axes[row_index, 5].plot(tf_factor["frequency"], tf_factor["normalized_factor"], "o-", color="#D55E00", lw=1.5, ms=3)
        axes[row_index, 5].set(xscale="log", xlabel="|TF| (Hz)", ylabel="normalized factor", ylim=(-0.03, 1.05), title="separable TF factor")
        for ax in axes[row_index, 4:]:
            ax.grid(True, color="0.9", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.title.set_fontsize(8)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_response_audit(summary: pd.DataFrame, selected: pd.DataFrame, threshold: float, out_path: Path, dpi: int) -> None:
    primary = summary[summary["surface_definition"].eq("preferred_orientation_abs_tf")].copy()
    responsive = primary["surface_maximum_f1_hz"] >= threshold
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].scatter(primary.loc[responsive, "surface_maximum_f1_hz"], primary.loc[responsive, "rank1_centered_r2"], s=25, color="#0072B2", alpha=0.75, label="response flag passed")
    axes[0].scatter(primary.loc[~responsive, "surface_maximum_f1_hz"], primary.loc[~responsive, "rank1_centered_r2"], s=30, color="#CC3311", alpha=0.85, label="weak control")
    axes[0].axvline(threshold, color="0.25", ls="--", lw=1)
    axes[0].set(xscale="log", xlabel="maximum F1 (Hz)", ylabel="centered rank-one R²", title="Strength versus fit quality")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(primary.loc[responsive, "rank1_centered_r2"], bins=np.linspace(0.65, 1.0, 15), color="#56B4E9", edgecolor="white")
    axes[1].set(xlabel="centered rank-one R²", ylabel="responsive units", title="Fit-quality audit (not group inference)")
    axes[2].scatter(primary["surface_maximum_f1_hz"], primary["maximum_direction_difference_hz"], c=primary["interaction_rms_over_surface_range"], cmap="magma", s=28, alpha=0.8)
    axes[2].set_xscale("log")
    axes[2].set_yscale("symlog", linthresh=1e-4)
    axes[2].set(
        xlabel="maximum F1 (Hz)",
        ylabel="maximum +TF/-TF difference (Hz)",
        title="Direction asymmetry versus strength",
    )
    for _, row in selected.iterrows():
        match = primary[primary["rr100_index"].eq(int(row["rr100_index"]))].iloc[0]
        axes[0].annotate(str(int(row["rr100_index"])), (match["surface_maximum_f1_hz"], match["rank1_centered_r2"]), fontsize=7)
        axes[2].annotate(str(int(row["rr100_index"])), (match["surface_maximum_f1_hz"], match["maximum_direction_difference_hz"]), fontsize=7)
    for ax in axes:
        ax.grid(True, color="0.92", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Response-strength and separability audit", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_csv = args.source_dir / "native_condition_unit_summary.csv"
    source_manifest = args.source_dir / "analysis_manifest.json"
    data = pd.read_csv(source_csv)
    if data["rr100_index"].nunique() != 100:
        raise ValueError(f"Expected 100 RR units, found {data['rr100_index'].nunique()}")

    dynamic = data[data["condition_kind"].eq("drifting_grating") & data["primary_fit_support"].astype(bool)].copy()
    phase_collapsed = (
        dynamic.groupby(
            ["rr100_index", "session", "orientation_deg", "spatial_cpd", "signed_temporal_hz"],
            as_index=False,
        )
        .agg(
            f1_amplitude_hz=("f1_amplitude_hz", "mean"),
            f1_phase_std_hz=("f1_amplitude_hz", "std"),
            mean_rate_above_blank_hz=("mean_rate_above_blank_hz", "mean"),
            n_carrier_phases=("phase_index", "nunique"),
        )
    )
    orientation_scores = (
        phase_collapsed.groupby(["rr100_index", "session", "orientation_deg"], as_index=False)
        .agg(mean_primary_f1_hz=("f1_amplitude_hz", "mean"), maximum_primary_f1_hz=("f1_amplitude_hz", "max"))
        .sort_values(["rr100_index", "mean_primary_f1_hz", "orientation_deg"], ascending=[True, False, True])
    )
    preferred_orientation = orientation_scores.drop_duplicates("rr100_index").rename(
        columns={
            "orientation_deg": "preferred_orientation_deg",
            "mean_primary_f1_hz": "preferred_orientation_mean_f1_hz",
            "maximum_primary_f1_hz": "preferred_orientation_maximum_f1_hz",
        }
    )
    orientation_scores = orientation_scores.merge(
        preferred_orientation[["rr100_index", "preferred_orientation_deg"]], on="rr100_index", how="left"
    )
    preferred_signed = phase_collapsed.merge(
        preferred_orientation[["rr100_index", "preferred_orientation_deg"]], on="rr100_index", how="left"
    )
    preferred_signed = preferred_signed[
        np.isclose(preferred_signed["orientation_deg"], preferred_signed["preferred_orientation_deg"])
    ].copy()
    preferred_abs = preferred_signed.assign(temporal_hz=preferred_signed["signed_temporal_hz"].abs()).groupby(
        ["rr100_index", "session", "preferred_orientation_deg", "spatial_cpd", "temporal_hz"], as_index=False
    )["f1_amplitude_hz"].mean()
    marginal_abs = phase_collapsed.assign(temporal_hz=phase_collapsed["signed_temporal_hz"].abs()).groupby(
        ["rr100_index", "session", "spatial_cpd", "temporal_hz"], as_index=False
    )["f1_amplitude_hz"].mean()
    marginal_abs = marginal_abs.merge(
        preferred_orientation[["rr100_index", "preferred_orientation_deg"]], on="rr100_index", how="left"
    )

    fit_summaries = []
    factor_frames = []
    point_frames = []
    for _, pref_row in preferred_orientation.iterrows():
        unit = int(pref_row["rr100_index"])
        session = str(pref_row["session"])
        orientation = float(pref_row["preferred_orientation_deg"])
        definitions = [
            (
                "preferred_orientation_signed_tf",
                preferred_signed[preferred_signed["rr100_index"].eq(unit)].rename(columns={"signed_temporal_hz": "temporal_hz"}),
            ),
            ("preferred_orientation_abs_tf", preferred_abs[preferred_abs["rr100_index"].eq(unit)]),
            ("orientation_marginal_abs_tf", marginal_abs[marginal_abs["rr100_index"].eq(unit)]),
        ]
        for definition, frame in definitions:
            matrix = complete_matrix(frame, index="spatial_cpd", columns="temporal_hz", values="f1_amplitude_hz")
            summary, factors, points = rank_one_fit(
                matrix,
                rr100_index=unit,
                session=session,
                surface_definition=definition,
                preferred_orientation_deg=orientation,
            )
            fit_summaries.append(summary)
            factor_frames.append(factors)
            point_frames.append(points)

    fit_summary = pd.DataFrame(fit_summaries)
    factors = pd.concat(factor_frames, ignore_index=True)
    surface_points = pd.concat(point_frames, ignore_index=True)
    direction = direction_metrics(preferred_signed)
    fit_summary = fit_summary.merge(direction, on=["rr100_index", "session"], how="left", validate="many_to_one")
    fit_summary["responsive_max_f1_flag"] = fit_summary["surface_maximum_f1_hz"] >= float(args.responsive_max_f1_hz)

    static = data[data["condition_kind"].eq("static_grating") & ~data["spatial_edge_control"].astype(bool)].copy()
    static_control = static.groupby(
        ["rr100_index", "session", "orientation_deg", "spatial_cpd"], as_index=False
    ).agg(
        static_mean_rate_above_blank_hz=("mean_rate_above_blank_hz", "mean"),
        static_phase_std_rate_hz=("mean_rate_above_blank_hz", "std"),
        n_static_phases=("phase_index", "nunique"),
    )
    dynamic_mean_control = phase_collapsed[
        [
            "rr100_index",
            "session",
            "orientation_deg",
            "spatial_cpd",
            "signed_temporal_hz",
            "mean_rate_above_blank_hz",
            "n_carrier_phases",
        ]
    ].copy()

    selected = choose_examples(fit_summary, float(args.minimum_example_max_f1_hz))
    selected_points = surface_points[surface_points["rr100_index"].isin(selected["rr100_index"])].copy()
    selected_factors = factors[factors["rr100_index"].isin(selected["rr100_index"])].copy()

    orientation_scores.to_csv(args.out_dir / "orientation_preference_scores.csv", index=False)
    preferred_orientation.to_csv(args.out_dir / "preferred_orientation_by_unit.csv", index=False)
    phase_collapsed.to_csv(args.out_dir / "phase_collapsed_dynamic_points.csv", index=False)
    fit_summary.to_csv(args.out_dir / "separable_fit_unit_summary.csv", index=False)
    factors.to_csv(args.out_dir / "separable_factor_points.csv", index=False)
    surface_points.to_csv(args.out_dir / "separable_surface_and_residual_points.csv", index=False)
    static_control.to_csv(args.out_dir / "static_phase_averaged_dc_control.csv", index=False)
    dynamic_mean_control.to_csv(args.out_dir / "dynamic_mean_rate_control.csv", index=False)
    selected.to_csv(args.out_dir / "selected_units.csv", index=False)
    selected_points.to_csv(args.out_dir / "selected_unit_surface_points.csv", index=False)
    selected_factors.to_csv(args.out_dir / "selected_unit_factor_points.csv", index=False)

    selected_figure = args.out_dir / "selected_unit_separable_sf_tf_surfaces.png"
    plot_selected(surface_points, factors, selected, selected_figure, int(args.dpi))
    audit_figure = args.out_dir / "response_strength_separability_audit.png"
    plot_response_audit(fit_summary, selected, float(args.responsive_max_f1_hz), audit_figure, int(args.dpi))

    primary_summary = fit_summary[fit_summary["surface_definition"].eq("preferred_orientation_abs_tf")]
    manifest = {
        "analysis": "rr100_zero_gaze_separable_sf_tf_factorization",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "unit_factorization_and_selected_example_checkpoint_not_population_inference",
        "source_csv": file_identity(source_csv),
        "source_manifest": file_identity(source_manifest),
        "n_units": int(primary_summary["rr100_index"].nunique()),
        "n_response_flag_passed": int(primary_summary["responsive_max_f1_flag"].sum()),
        "n_weak_response_controls": int((~primary_summary["responsive_max_f1_flag"]).sum()),
        "responsive_max_f1_threshold_hz": float(args.responsive_max_f1_hz),
        "response_contract": "dynamic F1 amplitude; phase averaged before fitting; static phase-averaged DC retained separately",
        "support_contract": "primary_fit_support from production: SF 1-11.313708 cpd and |TF| 0.5-32 Hz; 16 cpd and 45.254834 Hz excluded",
        "orientation_contract": "preferred orientation maximizes mean primary-support F1; preferred and orientation-marginal surfaces both retained",
        "direction_contract": "signed-TF surface and matched direction-difference metrics retained; |TF| fit averages matched directions",
        "fit_contract": "least-squares raw rank-one SVD with raw observed, reconstruction, residual, energy fraction, and centered R2 retained",
        "normalization_guardrail": "normalized factors are descriptive only; absolute maximum/range/gain and responsive flag remain beside them",
        "selected_roles": selected[["selection_role", "rr100_index"]].to_dict(orient="records"),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "artifacts": {
            "selected_figure": selected_figure.name,
            "audit_figure": audit_figure.name,
            "fit_summary": "separable_fit_unit_summary.csv",
            "factor_points": "separable_factor_points.csv",
            "surface_points": "separable_surface_and_residual_points.csv",
            "selected_units": "selected_units.csv",
            "static_control": "static_phase_averaged_dc_control.csv",
            "dynamic_mean_control": "dynamic_mean_rate_control.csv",
        },
    }
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
