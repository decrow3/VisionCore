#!/usr/bin/env python3
"""Expand the clean-history RF-local power checkpoint to 100 input-only conditions."""
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

from declan.fig4_active_sensing.spectral_cache_contract import (
    validate_artifact_not_superseded,
    validate_spectral_cache,
)

from declan.fig4_active_sensing.analyze_rr100_natural_image_rf_local_oriented_power_response_checkpoint import (
    audit_clean_history,
    load_response_rows,
    safe_correlation,
    zscore,
)
from declan.fig4_active_sensing.make_rr100_natural_image_rf_local_oriented_power_checkpoint import (
    build_metrics,
    load_selected_movies,
    spectral_storage_crosswalk,
    verify_reconstruction,
)
from declan.fig4_active_sensing.make_rr100_orientation_routing_input_checkpoint import (
    four_grating_channels,
)
from declan.fig4_active_sensing.make_rr100_recorded_grating_oriented_power_checkpoint import (
    load_rf_and_tuning,
)


ROOT = Path(__file__).resolve().parents[2]
RESPONSES = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
TRACE_FLAGS = RESPONSES / "quality_control/pre_fixation_history_trace_flags.csv"
ASSEMBLED = RESPONSES / "assembled/rounds_000_002_n003"
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
RF = ROOT / "outputs/fig4_active_sensing/rr100_recorded_grating_three_way_response_rf_local_v2"
TUNING = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_f0_map_checkpoint_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_expanded_n100_clean_history_v1"
EPS = np.finfo(float).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectral-dir", type=Path, required=True,
        help="Explicit frozen corrected spectral cache; superseded caches are rejected.",
    )
    parser.add_argument("--response-cache-dir", type=Path, default=RESPONSES)
    parser.add_argument("--trace-flags", type=Path, default=TRACE_FLAGS)
    parser.add_argument("--assembled-dir", type=Path, default=ASSEMBLED)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT)
    parser.add_argument("--rf-dir", type=Path, default=RF)
    parser.add_argument("--tuning-dir", type=Path, default=TUNING)
    parser.add_argument("--session", default="Logan_2020-02-29")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--n-conditions", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def load_spectral(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "condition_spectra.npz", allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def select_clean_conditions(
    spectral_dir: Path,
    trace_flags_path: Path,
    n_conditions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    spectral = load_spectral(spectral_dir)
    crosswalk = spectral_storage_crosswalk(spectral_dir)
    flags = pd.read_csv(trace_flags_path)
    clean_ids = set(
        flags.loc[
            flags.history_within_selected_fixation.eq(True)
            & flags.cache_eligibility.eq("clean_within_fixation_history"),
            "trace_index",
        ].astype(int)
    )
    eligible = crosswalk.loc[crosswalk.trace_index.astype(int).isin(clean_ids)].copy()
    images = sorted(eligible.image_index.astype(int).unique())
    if n_conditions != len(images):
        raise ValueError(
            f"This checkpoint requires one condition per available image: requested {n_conditions}, found {len(images)}"
        )

    rng = np.random.default_rng(seed)
    chosen_indices: list[int] | None = None
    for _ in range(10000):
        used_traces: set[int] = set()
        candidate_indices: list[int] = []
        success = True
        for image in rng.permutation(images):
            candidates = eligible.loc[eligible.image_index.astype(int).eq(int(image))]
            order = rng.permutation(candidates.index.to_numpy(int))
            available = [index for index in order if int(eligible.loc[index, "trace_index"]) not in used_traces]
            if not available:
                success = False
                break
            index = int(available[0])
            candidate_indices.append(index)
            used_traces.add(int(eligible.loc[index, "trace_index"]))
        if success:
            chosen_indices = candidate_indices
            break
    if chosen_indices is None:
        raise RuntimeError("Could not construct a one-image/one-trace clean-history matching")

    selected = crosswalk.loc[chosen_indices].copy().sort_values("image_index").reset_index(drop=True)
    selected["selection_role"] = "deterministic random clean-history condition; one per image and trace"
    selected["selection_criterion"] = (
        "input-only seeded matching with every image and trace identity used once"
    )
    selected["selection_seed"] = int(seed)
    selected["selection_order"] = np.arange(len(selected), dtype=int)

    channels = four_grating_channels(spectral["orientation_power"], spectral["orientation_edges_deg"])
    channel_power = channels.sum(axis=(1, 2))
    concentration = channel_power.max(axis=1) / np.maximum(channel_power.sum(axis=1), EPS)
    total_power = spectral["radial_power"].sum(axis=(1, 2)).astype(float)
    storage_rows = selected.spectral_storage_row.to_numpy(int)
    selected["global_dynamic_power"] = total_power[storage_rows]
    selected["global_orientation_concentration"] = concentration[storage_rows]

    eligibility = crosswalk[["spectral_storage_row", "matrix_row_index", "image_index", "trace_index", "round_index"]].copy()
    eligibility["clean_history_eligible"] = eligibility.trace_index.astype(int).isin(clean_ids)
    eligibility["selected_for_expanded_checkpoint"] = eligibility.spectral_storage_row.astype(int).isin(
        set(selected.spectral_storage_row.astype(int))
    )
    return selected, eligibility, spectral


def cross_validated_r2(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    prediction = np.full(len(y), np.nan, dtype=float)
    for held_out in np.unique(fold):
        test = fold == held_out
        train = ~test
        x_mean = float(x[train].mean())
        x_sd = max(float(x[train].std()), 1e-12)
        design = np.column_stack([np.ones(train.sum()), (x[train] - x_mean) / x_sd])
        coefficients, *_ = np.linalg.lstsq(design, y[train], rcond=None)
        prediction[test] = coefficients[0] + coefficients[1] * ((x[test] - x_mean) / x_sd)
    denominator = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - prediction) ** 2) / denominator)


def summarize_units(frame: pd.DataFrame, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for unit, group in frame.groupby("rr100_index", sort=True):
        group = group.sort_values("image_index")
        radial = group.radial_direct_f0_drive.to_numpy(float)
        oriented = group.oriented_direct_f0_drive.to_numpy(float)
        activation = group.full_twin_activation_rms_hz.to_numpy(float)
        signed = group.full_twin_delta_mean_rate_hz.to_numpy(float)
        folds = np.arange(len(group), dtype=int) % 5
        radial_r = safe_correlation(radial, activation)
        oriented_r = safe_correlation(oriented, activation)
        radial_mae = float(np.mean(np.abs(zscore(radial) - zscore(activation))))
        oriented_mae = float(np.mean(np.abs(zscore(oriented) - zscore(activation))))
        differences = np.empty(n_bootstrap, dtype=float)
        for index in range(n_bootstrap):
            sample = rng.integers(0, len(group), len(group))
            differences[index] = safe_correlation(oriented[sample], activation[sample]) - safe_correlation(
                radial[sample], activation[sample]
            )
        finite = differences[np.isfinite(differences)]
        summaries.append(
            {
                "rr100_index": int(unit),
                "n_conditions": int(len(group)),
                "orientation_collapsed_vs_modulation_r": radial_r,
                "orientation_aware_vs_modulation_r": oriented_r,
                "orientation_aware_minus_collapsed_r": oriented_r - radial_r,
                "orientation_collapsed_vs_modulation_spearman": float(spearmanr(radial, activation).statistic),
                "orientation_aware_vs_modulation_spearman": float(spearmanr(oriented, activation).statistic),
                "orientation_collapsed_modulation_z_mae": radial_mae,
                "orientation_aware_modulation_z_mae": oriented_mae,
                "orientation_aware_minus_collapsed_z_mae": oriented_mae - radial_mae,
                "orientation_collapsed_held_out_r2": cross_validated_r2(radial, activation, folds),
                "orientation_aware_held_out_r2": cross_validated_r2(oriented, activation, folds),
                "orientation_collapsed_vs_signed_mean_rate_r": safe_correlation(radial, signed),
                "orientation_aware_vs_signed_mean_rate_r": safe_correlation(oriented, signed),
                "bootstrap_r_difference_low": float(np.quantile(finite, 0.025)),
                "bootstrap_r_difference_high": float(np.quantile(finite, 0.975)),
                "bootstrap_fraction_r_difference_positive": float(np.mean(finite > 0)),
            }
        )
        bootstrap_rows.extend(
            {
                "rr100_index": int(unit),
                "bootstrap_index": int(index),
                "orientation_aware_minus_collapsed_r": float(value),
            }
            for index, value in enumerate(differences)
        )
    return pd.DataFrame(summaries), pd.DataFrame(bootstrap_rows)


def plot_input_selection(eligibility: pd.DataFrame, selected: pd.DataFrame, out: Path, dpi: int) -> None:
    selected_rows = set(selected.spectral_storage_row.astype(int))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    counts = eligibility.groupby("image_index").clean_history_eligible.sum()
    axes[0].bar(counts.index, counts.values, color="#0072B2")
    axes[0].set(
        xlabel="natural-image identity",
        ylabel="clean-history conditions available",
        title="Every image contributes one condition from its clean-history candidates",
    )
    axes[1].scatter(
        np.log10(np.maximum(selected.global_dynamic_power, EPS)),
        selected.global_orientation_concentration,
        s=26,
        alpha=0.75,
        color="#D55E00",
    )
    axes[1].set(
        xlabel="log10 total dynamic retinal-image power",
        ylabel="largest orientation-channel power fraction",
        title="Selected inputs span power and orientation concentration",
    )
    figure.suptitle(
        f"Input-only clean-history selection: {len(selected)} images and {selected.trace_index.nunique()} eye-movement traces",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_response_summary(frame: pd.DataFrame, summary: pd.DataFrame, out: Path, dpi: int) -> None:
    units = summary.rr100_index.astype(int).tolist()
    figure, axes = plt.subplots(len(units), 3, figsize=(17, 3.4 * len(units)), constrained_layout=True)
    for row, unit in enumerate(units):
        group = frame.loc[frame.rr100_index.eq(unit)].sort_values("image_index")
        activation = group.full_twin_activation_rms_hz.to_numpy(float)
        record = summary.loc[summary.rr100_index.eq(unit)].iloc[0]
        for column, predictor, color, label, correlation in (
            (0, group.radial_direct_f0_drive.to_numpy(float), "0.45", "orientation-collapsed power", record.orientation_collapsed_vs_modulation_r),
            (1, group.oriented_direct_f0_drive.to_numpy(float), "#D55E00", "orientation-aware power", record.orientation_aware_vs_modulation_r),
        ):
            standardized = zscore(predictor)
            axes[row, column].scatter(standardized, activation, s=18, alpha=0.65, color=color)
            coefficients = np.polyfit(standardized, activation, 1)
            grid = np.linspace(standardized.min(), standardized.max(), 100)
            axes[row, column].plot(grid, np.polyval(coefficients, grid), color=color, lw=2)
            axes[row, column].set(
                xlabel=f"{label} (standardized)",
                ylabel="digital-twin response-modulation RMS (Hz)",
                title=f"digital-twin unit {unit}: {label}\nPearson correlation r={correlation:+.2f}",
            )
        axes[row, 2].bar(
            ["orientation\ncollapsed", "orientation\naware"],
            [record.orientation_collapsed_vs_modulation_r, record.orientation_aware_vs_modulation_r],
            color=["0.55", "#D55E00"],
        )
        axes[row, 2].axhline(0, color="0.4", lw=0.8)
        axes[row, 2].set_ylim(-1, 1)
        axes[row, 2].set(
            ylabel="correlation with response-modulation magnitude",
            title=(
                f"difference={record.orientation_aware_minus_collapsed_r:+.2f}\n"
                f"condition-bootstrap 95% interval "
                f"[{record.bootstrap_r_difference_low:+.2f}, {record.bootstrap_r_difference_high:+.2f}]"
            ),
        )
    figure.suptitle(
        f"Expanded clean-history checkpoint: spectral power versus digital-twin response-modulation magnitude\n"
        f"{frame.matrix_row_index.nunique()} input-only conditions · one unique image and eye-movement trace per condition · descriptive five-unit test",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_spectral_cache(args.spectral_dir)
    validate_artifact_not_superseded(args.tuning_dir, label="orientation tuning")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected, eligibility, spectral = select_clean_conditions(
        args.spectral_dir, args.trace_flags, int(args.n_conditions), int(args.selection_seed)
    )
    history_audit = audit_clean_history(selected, args.trace_flags)
    payload = load_selected_movies(selected, args.cohort_dir, args.response_cache_dir, args.device)
    reconstruction = verify_reconstruction(selected, payload, spectral)
    units, apertures, radial_weights, oriented_weights, sf, tf = load_rf_and_tuning(
        args.rf_dir, args.tuning_dir, args.session
    )
    power, _, _ = build_metrics(
        selected, payload, units, apertures, radial_weights, oriented_weights, sf, tf, spectral
    )
    response, _, response_audit = load_response_rows(
        power, selected, args.response_cache_dir, args.assembled_dir
    )
    summary, bootstrap = summarize_units(response, int(args.n_bootstrap), int(args.bootstrap_seed))

    selected.to_csv(args.out_dir / "selected_conditions.csv", index=False)
    eligibility.to_csv(args.out_dir / "condition_history_eligibility.csv", index=False)
    history_audit.to_csv(args.out_dir / "selected_condition_history_audit.csv", index=False)
    reconstruction.to_csv(args.out_dir / "movie_and_spectrum_reconstruction_audit.csv", index=False)
    response.to_csv(args.out_dir / "condition_unit_power_response_metrics.csv", index=False)
    response_audit.to_csv(args.out_dir / "response_join_audit.csv", index=False)
    summary.to_csv(args.out_dir / "unit_expanded_condition_summary.csv", index=False)
    bootstrap.to_csv(args.out_dir / "condition_bootstrap_correlation_differences.csv", index=False)
    plot_input_selection(
        eligibility,
        selected,
        args.out_dir / "expanded_clean_history_input_selection",
        int(args.dpi),
    )
    plot_response_summary(
        response,
        summary,
        args.out_dir / "expanded_clean_history_power_response_summary",
        int(args.dpi),
    )

    audit_columns = [
        column
        for column in response_audit
        if column.endswith("_error_vs_assembled_hz") or column.endswith("_error_vs_shard_hz")
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "expanded clean-history natural-image RF-local power response checkpoint",
        "status": "expanded_targeted_checkpoint_complete_stop_before_population_claim",
        "scope": {
            "conditions": int(len(selected)),
            "unique_images": int(selected.image_index.nunique()),
            "unique_traces": int(selected.trace_index.nunique()),
            "units": int(response.rr100_index.nunique()),
            "condition_unit_pairs": int(len(response)),
        },
        "selection_contract": {
            "rule": "input-only seeded matching; exactly one clean-history condition per image and unique trace",
            "seed": int(args.selection_seed),
            "neural_outcomes_used": False,
        },
        "interpretation_contract": {
            "primary_outcome": "40-frame digital-twin RMS of moving minus matched stabilized firing rate",
            "primary_comparison": "orientation-aware versus orientation-collapsed receptive-field-local power",
            "inference": "descriptive targeted five-unit checkpoint; condition bootstrap does not support a population claim",
            "spectral_cache_limitation": "uses reconstructed image-major crosswalk for the old three-round arrays; every selected movie and spectrum is independently rerendered",
        },
        "verification": {
            "all_selected_histories_within_fixation": bool(history_audit.clean_history_gate_pass.all()),
            "maximum_cached_radial_reconstruction_relative_error": float(reconstruction.maximum_radial_relative_error.max()),
            "maximum_cached_oriented_reconstruction_relative_error": float(reconstruction.maximum_oriented_relative_error.max()),
            "maximum_orientation_sum_relative_error": float(reconstruction.orientation_sum_relative_error.max()),
            "maximum_response_join_or_formula_error_hz": float(response_audit[audit_columns].to_numpy(float).max()),
            "all_response_values_finite": bool(np.isfinite(response.select_dtypes("number")).all().all()),
        },
        "inputs": {
            "spectral_cache": file_identity(args.spectral_dir / "condition_spectra.npz"),
            "trace_history_flags": file_identity(args.trace_flags),
            "quarantine_manifest": file_identity(
                args.response_cache_dir / "quality_control/pre_fixation_history_quarantine_manifest.json"
            ),
        },
        "artifacts": {
            "input_figure": "expanded_clean_history_input_selection.png",
            "response_figure": "expanded_clean_history_power_response_summary.png",
            "selected_conditions": "selected_conditions.csv",
            "pair_metrics": "condition_unit_power_response_metrics.csv",
            "unit_summary": "unit_expanded_condition_summary.csv",
            "bootstrap": "condition_bootstrap_correlation_differences.csv",
            "history_audit": "selected_condition_history_audit.csv",
            "spectrum_audit": "movie_and_spectrum_reconstruction_audit.csv",
            "response_audit": "response_join_audit.csv",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
