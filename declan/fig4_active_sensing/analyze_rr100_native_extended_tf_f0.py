#!/usr/bin/env python3
"""Inspect and fit the native-readout RR100 F0 extension through 56 Hz.

The old 0.5--32 Hz surface and the new 34--56 Hz measurements are combined at
the *old* preferred orientation.  The repeated 32-Hz plane is reserved for an
exact replication audit and 60 Hz is reserved as a Nyquist-edge control.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.analyze_rr100_zero_gaze_separable_sf_tf_f0 import (
    fit_nonnegative_rank_one,
)
from declan.fit_rr100_joint_f0_parametric_and_validate_recorded import (
    fit_log_gaussian,
    log_gaussian,
)


OLD = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_separable_sf_tf_f0_factorization_v1"
OLD_PARAM = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_joint_f0_parametric_recorded_validation_v1"
NEW = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_zero_gaze_native_extended_tf_32_60_v1"
OUT = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_native_extended_tf_f0_analysis_v1"
SF_GRID = np.asarray([1, 2**0.5, 2, 2**1.5, 4, 2**2.5, 8, 2**3.5], dtype=float)
OLD_TF = np.asarray([0.5, 2**-0.5, 1, 2**0.5, 2, 2**1.5, 4, 2**2.5, 8, 2**3.5, 16, 2**4.5, 32], dtype=float)
EXT_TF = np.asarray([34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56], dtype=float)
NYQUIST_TF = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=OLD)
    parser.add_argument("--old-param-dir", type=Path, default=OLD_PARAM)
    parser.add_argument("--new-dir", type=Path, default=NEW)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def collect_new_rows(directory: Path) -> pd.DataFrame:
    aggregate = directory / "native_condition_unit_summary.csv"
    if aggregate.exists():
        return pd.read_csv(aggregate)
    paths = sorted((directory / "sessions").glob("*/condition_unit_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No native response summaries found under {directory}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def prepare_new_folded(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    rows = rows.drop_duplicates(["session", "condition_id", "rr100_index"], keep="last").copy()
    # The production aggregate already carries these derived columns, whereas
    # resumable per-session checkpoints do not.  Recompute from the blank rows
    # in both cases so aggregate and checkpoint inputs have one identical path.
    rows = rows.drop(columns=["blank_rate_hz", "mean_rate_above_blank_hz"], errors="ignore")
    blank = (
        rows[rows.condition_kind.eq("gray_blank")]
        .groupby(["session", "rr100_index"], as_index=False)
        .mean_rate_hz.mean()
        .rename(columns={"mean_rate_hz": "blank_rate_hz"})
    )
    dynamic = rows[rows.condition_kind.eq("drifting_grating")].merge(
        blank, on=["session", "rr100_index"], how="left", validate="many_to_one"
    )
    dynamic["signed_f0_hz"] = dynamic.mean_rate_hz - dynamic.blank_rate_hz
    dynamic["temporal_hz"] = dynamic.signed_temporal_hz.abs()
    phase = (
        dynamic.groupby(
            ["session", "rr100_index", "orientation_deg", "spatial_cpd", "temporal_hz", "signed_temporal_hz"],
            as_index=False,
        )
        .signed_f0_hz.mean()
    )
    folded = (
        phase.groupby(
            ["session", "rr100_index", "orientation_deg", "spatial_cpd", "temporal_hz"],
            as_index=False,
        )
        .agg(
            signed_f0_hz=("signed_f0_hz", "mean"),
            direction_difference_f0_hz=("signed_f0_hz", lambda x: float(np.max(x) - np.min(x))),
            n_directions=("signed_temporal_hz", "nunique"),
        )
    )
    folded["positive_f0_hz"] = folded.signed_f0_hz.clip(lower=0)
    expected = len(SF_GRID) * (len(EXT_TF) + 2) * 4
    counts = folded.groupby("rr100_index").size()
    complete = sorted(counts[counts.eq(expected)].index.astype(int).tolist())
    return folded, complete


def matrix(frame: pd.DataFrame, value: str, sf_grid: np.ndarray, tf_grid: np.ndarray) -> np.ndarray:
    values = np.full((len(sf_grid), len(tf_grid)), np.nan, dtype=float)
    for sf_index, sf in enumerate(sf_grid):
        for tf_index, tf in enumerate(tf_grid):
            match = frame[np.isclose(frame.spatial_cpd, sf) & np.isclose(frame.temporal_hz, tf)]
            if len(match) == 1:
                values[sf_index, tf_index] = float(match[value].iloc[0])
    if not np.isfinite(values).all():
        raise ValueError("Incomplete SFxTF matrix")
    return values


def centered_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - residual / total if total > 1e-15 else float("nan")


def select_raw_examples(raw: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    used: set[int] = set()

    def add(role: str, candidates: pd.DataFrame, metric: str, largest: bool, criterion: str) -> None:
        candidates = candidates[~candidates.rr100_index.isin(used)].dropna(subset=[metric])
        if candidates.empty:
            return
        row = candidates.loc[candidates[metric].idxmax() if largest else candidates[metric].idxmin()].copy()
        row["selection_role"] = role
        row["selection_metric"] = metric
        row["selection_value"] = float(row[metric])
        row["selection_criterion"] = criterion
        selected.append(row)
        used.add(int(row.rr100_index))

    boundary = raw[raw.old_sampled_tf_peak_hz.eq(32) & raw.old_responsive]
    add(
        "32-Hz boundary continues upward",
        boundary,
        "extension_to_32_ratio",
        True,
        "largest raw 34–56 Hz / repeated-32 Hz marginal among old 32-Hz-edge units",
    )
    add(
        "32-Hz boundary resolves downward",
        boundary,
        "extension_to_32_ratio",
        False,
        "smallest raw 34–56 Hz / repeated-32 Hz marginal among old 32-Hz-edge units",
    )
    add(
        "strong high-TF response",
        raw[raw.old_responsive],
        "maximum_extension_f0_hz",
        True,
        "largest raw positive F0 anywhere from 34–56 Hz",
    )
    add(
        "Nyquist-edge discontinuity control",
        raw,
        "nyquist_discontinuity_hz",
        True,
        "largest absolute 60-Hz minus 54/56-Hz marginal; 60 Hz is not fit",
    )
    return pd.DataFrame(selected)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    old_points = pd.read_csv(args.old_dir / "direction_folded_signed_f0_points.csv")
    old_summary = pd.read_csv(args.old_dir / "f0_separable_fit_unit_summary.csv")
    old_param = pd.read_csv(args.old_param_dir / "rr100_joint_f0_parametric_fit_summary.csv")
    new_rows = collect_new_rows(args.new_dir)
    new_folded, complete_units = prepare_new_folded(new_rows)
    if not complete_units:
        raise RuntimeError("No RR100 unit yet has a complete native 32–60 Hz surface")

    preferred = old_summary[["rr100_index", "preferred_orientation_deg"]].drop_duplicates()
    new_pref = new_folded.merge(
        preferred,
        on="rr100_index",
        how="inner",
        validate="many_to_one",
    )
    new_pref = new_pref[np.isclose(new_pref.orientation_deg, new_pref.preferred_orientation_deg)]
    old_pref = old_points[np.isclose(old_points.orientation_deg, old_points.preferred_orientation_deg)].copy()
    old_pref = old_pref[old_pref.rr100_index.isin(complete_units)]
    new_pref = new_pref[new_pref.rr100_index.isin(complete_units)]

    overlap = new_pref[np.isclose(new_pref.temporal_hz, 32)].merge(
        old_pref[np.isclose(old_pref.temporal_hz, 32)][
            ["rr100_index", "spatial_cpd", "signed_f0_hz", "positive_f0_hz"]
        ],
        on=["rr100_index", "spatial_cpd"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    overlap["signed_difference_hz"] = overlap.signed_f0_hz_new - overlap.signed_f0_hz_old
    overlap["absolute_signed_difference_hz"] = overlap.signed_difference_hz.abs()
    overlap.to_csv(args.out_dir / "repeated_32hz_point_audit.csv", index=False)

    raw_rows: list[dict[str, object]] = []
    combined_points: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    surfaces: dict[int, dict[str, np.ndarray]] = {}
    all_tf = np.concatenate([OLD_TF, EXT_TF])
    old_lookup = old_summary.set_index("rr100_index")
    old_param_lookup = old_param.set_index("rr100_index")

    for unit in complete_units:
        old_unit = old_pref[old_pref.rr100_index.eq(unit)].copy()
        new_unit = new_pref[new_pref.rr100_index.eq(unit)].copy()
        old_matrix = matrix(old_unit, "positive_f0_hz", SF_GRID, OLD_TF)
        extension_matrix = matrix(new_unit[new_unit.temporal_hz.isin(EXT_TF)], "positive_f0_hz", SF_GRID, EXT_TF)
        repeated_32 = matrix(new_unit[np.isclose(new_unit.temporal_hz, 32)], "positive_f0_hz", SF_GRID, np.asarray([32.0]))[:, 0]
        nyquist = matrix(new_unit[np.isclose(new_unit.temporal_hz, NYQUIST_TF)], "positive_f0_hz", SF_GRID, np.asarray([NYQUIST_TF]))[:, 0]
        combined = np.column_stack([old_matrix, extension_matrix])
        extension_marginal = extension_matrix.mean(axis=0)
        repeated_marginal = float(repeated_32.mean())
        high_reference = float(extension_marginal[-2:].mean())
        old_row = old_lookup.loc[unit]
        raw_rows.append(
            {
                "rr100_index": unit,
                "session": old_row.session,
                "preferred_orientation_deg": float(old_row.preferred_orientation_deg),
                "old_responsive": bool(old_row.responsive_positive_f0_flag),
                "old_sampled_tf_peak_hz": float(old_row.preferred_tf_hz_factor),
                "repeated_32hz_marginal_f0_hz": repeated_marginal,
                "maximum_extension_f0_hz": float(extension_matrix.max()),
                "maximum_extension_marginal_f0_hz": float(extension_marginal.max()),
                "extension_to_32_ratio": float(extension_marginal.max() / max(repeated_marginal, 1e-12)),
                "nyquist_60hz_marginal_f0_hz": float(nyquist.mean()),
                "nyquist_discontinuity_hz": float(abs(nyquist.mean() - high_reference)),
            }
        )
        rank1 = fit_nonnegative_rank_one(pd.DataFrame(combined, index=SF_GRID, columns=all_tf))
        gain = float(rank1["gain_hz"])
        sf_factor = np.asarray(rank1["sf_factor"], dtype=float)
        tf_factor = np.asarray(rank1["tf_factor"], dtype=float)
        prediction = np.asarray(rank1["reconstruction"], dtype=float)
        sf_fit = fit_log_gaussian(SF_GRID, sf_factor) if old_row.responsive_positive_f0_flag else {"fit_ok": False}
        tf_fit = fit_log_gaussian(all_tf, tf_factor) if old_row.responsive_positive_f0_flag else {"fit_ok": False}
        fit_row: dict[str, object] = {
            "rr100_index": unit,
            "session": old_row.session,
            "preferred_orientation_deg": float(old_row.preferred_orientation_deg),
            "responsive_positive_f0_flag": bool(old_row.responsive_positive_f0_flag),
            "old_tf_sampled_preferred_hz": float(old_row.preferred_tf_hz_factor),
            "old_tf_parametric_center_hz": float(old_param_lookup.loc[unit].tf_center_frequency),
            "extended_rank1_gain_f0_hz": float(gain),
            "extended_rank1_centered_r2": centered_r2(combined, prediction),
            "extended_rank1_relative_rmse": float(np.sqrt(np.mean((combined - prediction) ** 2)) / max(combined.max(), 1e-12)),
            "extended_sf_sampled_preferred_cpd": float(SF_GRID[np.argmax(sf_factor)]),
            "extended_tf_sampled_preferred_hz": float(all_tf[np.argmax(tf_factor)]),
        }
        for prefix, result in (("extended_sf", sf_fit), ("extended_tf", tf_fit)):
            fit_row[f"{prefix}_parametric_fit_ok"] = bool(result.get("fit_ok", False))
            for key in ("center_frequency", "sigma_octaves", "fwhm_octaves", "fit_r2", "fit_rmse", "preferred_within_support", "sampled_preferred", "sampled_peak_at_edge"):
                fit_row[f"{prefix}_{key}"] = result.get(key, np.nan)
        fit_rows.append(fit_row)
        surfaces[unit] = {"observed": combined, "predicted": prediction, "nyquist": nyquist}
        for axis, frequencies, values in (("spatial_frequency", SF_GRID, sf_factor), ("temporal_frequency", all_tf, tf_factor)):
            result = sf_fit if axis == "spatial_frequency" else tf_fit
            parametric = (
                log_gaussian(frequencies, result["baseline"], result["amplitude"], result["center_log2"], result["sigma_octaves"])
                if result.get("fit_ok", False)
                else np.full(len(frequencies), np.nan)
            )
            for frequency, value, predicted_value in zip(frequencies, values, parametric):
                factor_rows.append({"rr100_index": unit, "axis": axis, "frequency": float(frequency), "normalized_factor": float(value), "parametric_prediction": float(predicted_value)})
        for sf_index, sf in enumerate(SF_GRID):
            for tf_index, tf in enumerate(all_tf):
                combined_points.append({"rr100_index": unit, "spatial_cpd": float(sf), "temporal_hz": float(tf), "observed_positive_f0_hz": float(combined[sf_index, tf_index]), "rank1_prediction_f0_hz": float(prediction[sf_index, tf_index]), "source": "established_0p5_32" if tf <= 32 else "native_extension_34_56"})

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(args.out_dir / "raw_high_tf_unit_metrics.csv", index=False)
    examples = select_raw_examples(raw)
    examples.to_csv(args.out_dir / "selected_raw_map_examples.csv", index=False)
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(args.out_dir / "extended_f0_fit_unit_summary.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(args.out_dir / "extended_f0_factor_points.csv", index=False)
    pd.DataFrame(combined_points).to_csv(args.out_dir / "extended_f0_surface_points.csv", index=False)

    # Map-first example sheet: the left panel is the direct response map; the
    # fitted marginal is shown only after the reader can see the measurements.
    if not examples.empty:
        fig, axes = plt.subplots(len(examples), 3, figsize=(14, 3.5 * len(examples)), squeeze=False, constrained_layout=True)
        vmax = max(float(surfaces[int(unit)]["observed"].max()) for unit in examples.rr100_index)
        for row_index, example in enumerate(examples.itertuples(index=False)):
            unit = int(example.rr100_index)
            observed = surfaces[unit]["observed"]
            image = axes[row_index, 0].imshow(observed, origin="lower", aspect="auto", cmap="coolwarm", vmin=0, vmax=max(vmax, 1e-12))
            axes[row_index, 0].set_yticks(range(len(SF_GRID)), [f"{v:g}" for v in SF_GRID])
            tick_idx = [0, 4, 8, 12, 16, 20, 24]
            axes[row_index, 0].set_xticks(tick_idx, [f"{all_tf[i]:g}" for i in tick_idx])
            axes[row_index, 0].axvline(len(OLD_TF) - 0.5, color="white", ls="--", lw=1.2)
            axes[row_index, 0].set(xlabel="TF (Hz); dashed line begins new data", ylabel="SF (cpd)", title=f"{example.selection_role}\nRR100 {unit}: positive F0 at fixed preferred orientation")
            fig.colorbar(image, ax=axes[row_index, 0], label="F0 above blank (Hz)")

            old_curve = observed[:, : len(OLD_TF)].mean(axis=0)
            ext_curve = observed[:, len(OLD_TF) :].mean(axis=0)
            axes[row_index, 1].plot(OLD_TF, old_curve, "o-", color="#0072B2", label="established 0.5–32")
            axes[row_index, 1].plot(EXT_TF, ext_curve, "o-", color="#D55E00", label="new native 34–56")
            axes[row_index, 1].plot([60], [surfaces[unit]["nyquist"].mean()], "X", ms=8, color="#CC79A7", label="60-Hz control")
            axes[row_index, 1].set_xscale("log", base=2)
            axes[row_index, 1].set(xlabel="TF (Hz)", ylabel="mean positive F0 across SF (Hz)", title="Direct TF marginal (before fitting)")
            axes[row_index, 1].legend(frameon=False, fontsize=7)

            factor = pd.DataFrame(factor_rows)
            curve = factor[factor.rr100_index.eq(unit) & factor.axis.eq("temporal_frequency")]
            axes[row_index, 2].plot(curve.frequency, curve.normalized_factor, "o", color="black", label="rank-1 factor")
            axes[row_index, 2].plot(curve.frequency, curve.parametric_prediction, "-", color="#009E73", label="log-Gaussian")
            axes[row_index, 2].axvline(32, color="0.6", ls="--")
            axes[row_index, 2].set_xscale("log", base=2)
            axes[row_index, 2].set(xlabel="TF (Hz)", ylabel="normalized TF factor", title="Combined separable fit (60 Hz excluded)")
            axes[row_index, 2].legend(frameon=False, fontsize=7)
        fig.suptitle("Native-readout high-TF extension: raw maps first, fit second", fontsize=14, weight="bold")
        fig.savefig(args.out_dir / "selected_native_extended_tf_maps.png", dpi=args.dpi, bbox_inches="tight")
        fig.savefig(args.out_dir / "selected_native_extended_tf_maps.pdf", bbox_inches="tight")
        plt.close(fig)

    responsive = fits[fits.responsive_positive_f0_flag & fits.extended_tf_parametric_fit_ok].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].scatter(responsive.old_tf_parametric_center_hz, responsive.extended_tf_center_frequency, s=24, alpha=0.7)
    axes[0].plot([0.5, 60], [0.5, 60], color="0.6", ls="--")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log", base=2)
    axes[0].set(xlabel="old TF center (Hz; fit through 32)", ylabel="extended TF center (Hz; fit through 56)", title="How TF preference changes")
    bins = np.geomspace(0.5, 60, 18)
    axes[1].hist(responsive.old_tf_parametric_center_hz, bins=bins, alpha=0.55, label="old", color="#0072B2")
    axes[1].hist(responsive.extended_tf_center_frequency, bins=bins, alpha=0.55, label="extended", color="#D55E00")
    axes[1].set_xscale("log", base=2); axes[1].set(xlabel="parametric TF center (Hz)", ylabel="units", title="Population preference distribution")
    axes[1].legend(frameon=False)
    axes[2].scatter(responsive.old_tf_parametric_center_hz, responsive.extended_rank1_centered_r2, s=24, alpha=0.7, color="#009E73")
    axes[2].axhline(0.5, color="0.6", ls="--")
    axes[2].set_xscale("log", base=2); axes[2].set(xlabel="old TF center (Hz)", ylabel="extended rank-1 surface R²", title="Does separability survive extension?")
    fig.suptitle("Population consequences of extending native F0 support to 56 Hz", fontsize=14, weight="bold")
    fig.savefig(args.out_dir / "native_extended_tf_population.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out_dir / "native_extended_tf_population.pdf", bbox_inches="tight")
    plt.close(fig)

    overlap_signed_r = float(pearsonr(overlap.signed_f0_hz_old, overlap.signed_f0_hz_new).statistic) if len(overlap) > 2 else np.nan
    overlap_signed_rho = float(spearmanr(overlap.signed_f0_hz_old, overlap.signed_f0_hz_new).statistic) if len(overlap) > 2 else np.nan
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if len(complete_units) == 100 else "partial_complete_sessions_only",
        "n_complete_units": len(complete_units),
        "complete_units": complete_units,
        "contracts": {
            "readout": "session-native fitted-unit response, identical to established 0.5–32 Hz sweep",
            "orientation": "established preferred orientation held fixed",
            "replication": "new 32-Hz plane compared to old but excluded from combined fit",
            "fit": "old 0.5–32 plus new 34–56 Hz positive F0; nonnegative rank-one SVD and log-Gaussian factors",
            "nyquist": "60 Hz retained only as an edge control and excluded from fit",
        },
        "repeated_32hz": {
            "n_points": int(len(overlap)),
            "pearson_r": overlap_signed_r,
            "spearman_rho": overlap_signed_rho,
            "maximum_absolute_difference_hz": float(overlap.absolute_signed_difference_hz.max()),
            "median_absolute_difference_hz": float(overlap.absolute_signed_difference_hz.median()),
        },
        "population": {
            "n_responsive_parametric_fits": int(len(responsive)),
            "median_old_tf_center_hz": float(responsive.old_tf_parametric_center_hz.median()) if len(responsive) else None,
            "median_extended_tf_center_hz": float(responsive.extended_tf_center_frequency.median()) if len(responsive) else None,
            "median_extended_rank1_r2": float(responsive.extended_rank1_centered_r2.median()) if len(responsive) else None,
            "n_extended_sampled_peaks_above_32hz": int((responsive.extended_tf_sampled_preferred_hz > 32).sum()),
            "n_extended_sampled_peaks_at_56hz_edge": int(np.isclose(responsive.extended_tf_sampled_preferred_hz, 56).sum()),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
