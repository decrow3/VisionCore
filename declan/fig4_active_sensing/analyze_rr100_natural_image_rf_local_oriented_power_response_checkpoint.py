#!/usr/bin/env python3
"""Compare RF-local radial/oriented power with exact full-twin natural-image responses."""
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

from declan.fig4_active_sensing.make_rr100_natural_image_rf_local_oriented_power_checkpoint import (
    load_selected_movies,
)


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_input_checkpoint_v2"
RESPONSES = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
ASSEMBLED = RESPONSES / "assembled/rounds_000_002_n003"
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
RF = ROOT / "outputs/fig4_active_sensing/rr100_recorded_grating_three_way_response_rf_local_v2"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_response_checkpoint_v1"
FRAME_RATE_HZ = 120.0
EPS = np.finfo(float).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--response-cache-dir", type=Path, default=RESPONSES)
    parser.add_argument("--assembled-dir", type=Path, default=ASSEMBLED)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT)
    parser.add_argument("--rf-dir", type=Path, default=RF)
    parser.add_argument("--out-dir", type=Path, default=OUT)
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


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / max(float(values.std()), 1e-12)


def load_response_rows(
    power: pd.DataFrame,
    conditions: pd.DataFrame,
    response_cache_dir: Path,
    assembled_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[int, int], dict[str, np.ndarray]], pd.DataFrame]:
    condition_index = pd.read_csv(assembled_dir / "condition_index.csv").set_index("matrix_row_index")
    assembled_moving = {
        "mean_rate_hz": np.load(assembled_dir / "moving_mean_rate_hz.npy"),
        "activation_rms_hz": np.load(assembled_dir / "moving_temporal_rms_delta_from_stabilized_hz.npy"),
        "activation_mean_abs_hz": np.load(assembled_dir / "moving_temporal_mean_abs_delta_from_stabilized_hz.npy"),
    }
    assembled_baseline = np.load(assembled_dir / "stabilized_by_image_sufficient_statistics.npz")
    roles = conditions.set_index("matrix_row_index")["selection_role"]
    rows: list[dict[str, object]] = []
    timecourses: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    audit_rows: list[dict[str, object]] = []
    for condition_id in sorted(power.matrix_row_index.unique().astype(int)):
        identity = condition_index.loc[condition_id]
        round_index = int(identity.round_index)
        image_index = int(identity.image_index)
        trace_index = int(identity.trace_index)
        moving_path = response_cache_dir / "moving" / f"round_{round_index:03d}" / f"image_{image_index:03d}.npz"
        baseline_path = response_cache_dir / "baselines" / f"image_{image_index:03d}.npz"
        with np.load(moving_path, allow_pickle=False) as moving, np.load(baseline_path, allow_pickle=False) as baseline:
            trace_position = np.flatnonzero(moving["trace_index"].astype(int) == trace_index)
            if len(trace_position) != 1:
                raise ValueError(f"Trace {trace_index} is not unique in {moving_path}")
            trace_position = int(trace_position[0])
            for unit in power.loc[power.matrix_row_index.eq(condition_id), "rr100_index"].astype(int):
                moving_tc = moving["rate_timecourse_hz"][trace_position, :, unit].astype(float)
                baseline_tc = baseline["rate_timecourse_hz"][:, unit].astype(float)
                delta_tc = moving_tc - baseline_tc
                manual_rms = float(np.sqrt(np.mean(delta_tc**2)))
                manual_mean_abs = float(np.mean(np.abs(delta_tc)))
                shard_mean = float(moving["mean_rate_hz"][trace_position, unit])
                shard_rms = float(moving["temporal_rms_delta_from_stabilized_hz"][trace_position, unit])
                shard_mean_abs = float(moving["temporal_mean_abs_delta_from_stabilized_hz"][trace_position, unit])
                baseline_mean = float(baseline["mean_rate_hz"][unit])
                power_row = power.loc[
                    power.matrix_row_index.eq(condition_id) & power.rr100_index.eq(unit)
                ].iloc[0]
                rows.append(
                    {
                        **power_row.to_dict(),
                        "condition_selection_role": str(roles.loc[condition_id]),
                        "round_index": round_index,
                        "full_twin_moving_mean_rate_hz": shard_mean,
                        "full_twin_stabilized_mean_rate_hz": baseline_mean,
                        "full_twin_delta_mean_rate_hz": shard_mean - baseline_mean,
                        "full_twin_activation_rms_hz": shard_rms,
                        "full_twin_activation_mean_abs_hz": shard_mean_abs,
                    }
                )
                timecourses[(condition_id, int(unit))] = {
                    "moving_rate_hz": moving_tc,
                    "stabilized_rate_hz": baseline_tc,
                    "delta_rate_hz": delta_tc,
                }
                audit_rows.append(
                    {
                        "matrix_row_index": condition_id,
                        "rr100_index": int(unit),
                        "moving_mean_abs_error_vs_assembled_hz": abs(
                            shard_mean - float(assembled_moving["mean_rate_hz"][condition_id, unit])
                        ),
                        "activation_rms_abs_error_vs_assembled_hz": abs(
                            shard_rms - float(assembled_moving["activation_rms_hz"][condition_id, unit])
                        ),
                        "activation_mean_abs_error_vs_assembled_hz": abs(
                            shard_mean_abs - float(assembled_moving["activation_mean_abs_hz"][condition_id, unit])
                        ),
                        "baseline_mean_abs_error_vs_assembled_hz": abs(
                            baseline_mean - float(assembled_baseline["mean_rate_hz"][image_index, unit])
                        ),
                        "manual_rms_abs_error_vs_shard_hz": abs(manual_rms - shard_rms),
                        "manual_mean_abs_error_vs_shard_hz": abs(manual_mean_abs - shard_mean_abs),
                    }
                )
    return pd.DataFrame(rows), timecourses, pd.DataFrame(audit_rows)


def add_scale_free_comparisons(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    augmented = []
    summaries = []
    for unit, group in frame.groupby("rr100_index", sort=True):
        group = group.sort_values("matrix_row_index").copy()
        radial = group.radial_direct_f0_drive.to_numpy(float)
        oriented = group.oriented_direct_f0_drive.to_numpy(float)
        activation = group.full_twin_activation_rms_hz.to_numpy(float)
        delta_mean = group.full_twin_delta_mean_rate_hz.to_numpy(float)
        z_radial = zscore(radial)
        z_oriented = zscore(oriented)
        z_activation = zscore(activation)
        group["radial_drive_z_across_conditions"] = z_radial
        group["oriented_drive_z_across_conditions"] = z_oriented
        group["activation_rms_z_across_conditions"] = z_activation
        group["radial_absolute_z_error"] = np.abs(z_activation - z_radial)
        group["oriented_absolute_z_error"] = np.abs(z_activation - z_oriented)
        group["orientation_pair_error_improvement"] = (
            group.radial_absolute_z_error - group.oriented_absolute_z_error
        )
        augmented.append(group)
        summaries.append(
            {
                "rr100_index": int(unit),
                "n_conditions": int(len(group)),
                "radial_vs_activation_r": safe_correlation(radial, activation),
                "oriented_vs_activation_r": safe_correlation(oriented, activation),
                "orientation_minus_radial_activation_r": safe_correlation(oriented, activation)
                - safe_correlation(radial, activation),
                "radial_activation_z_mae": float(np.mean(np.abs(z_activation - z_radial))),
                "oriented_activation_z_mae": float(np.mean(np.abs(z_activation - z_oriented))),
                "orientation_minus_radial_activation_z_mae": float(
                    np.mean(np.abs(z_activation - z_oriented)) - np.mean(np.abs(z_activation - z_radial))
                ),
                "radial_vs_delta_mean_rate_r": safe_correlation(radial, delta_mean),
                "oriented_vs_delta_mean_rate_r": safe_correlation(oriented, delta_mean),
            }
        )
    return pd.concat(augmented, ignore_index=True), pd.DataFrame(summaries)


def select_examples(frame: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        (
            "orientation improves activation match",
            frame.orientation_pair_error_improvement,
            "max",
            "largest reduction in absolute across-condition activation z-error from radial to oriented drive",
        ),
        (
            "local condition mismatch despite unit-level improvement",
            frame.orientation_pair_error_improvement,
            "min",
            "largest single-condition increase in activation z-error, retained despite the unit's overall oriented-power improvement",
        ),
        (
            "radial-equivalent power control",
            frame.log2_orientation_to_radial.abs(),
            "min",
            "smallest absolute log2 oriented/radial drive",
        ),
        (
            "large full-twin activation",
            frame.full_twin_activation_rms_hz,
            "max",
            "largest full-twin moving-versus-stabilized activation RMS",
        ),
    ]
    selected: list[pd.Series] = []
    used: set[int] = set()
    for role, values, direction, criterion in definitions:
        available = values.loc[~values.index.isin(used)]
        index = available.idxmax() if direction == "max" else available.idxmin()
        row = frame.loc[index].copy()
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        row["selection_value"] = float(values.loc[index])
        selected.append(row)
        used.add(int(index))
    return pd.DataFrame(selected)


def plot_checkpoint(
    selected: pd.DataFrame,
    all_rows: pd.DataFrame,
    timecourses: dict[tuple[int, int], dict[str, np.ndarray]],
    payload: dict[int, dict[str, np.ndarray]],
    apertures: dict[int, np.ndarray],
    out: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(len(selected), 7, figsize=(26, 3.55 * len(selected)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    condition_labels = {
        int(row.matrix_row_index): str(row.condition_selection_role).replace(" input", "")
        for row in all_rows.itertuples(index=False)
    }
    for row_number, selection in enumerate(selected.itertuples(index=False)):
        condition = int(selection.matrix_row_index)
        unit = int(selection.rr100_index)
        item = payload[condition]
        movie = item["scored_movie"]
        aperture = apertures[unit]
        strip = np.concatenate([movie[0], movie[len(movie) // 2], movie[-1]], axis=1)
        aperture_strip = np.concatenate([aperture, aperture, aperture], axis=1)
        axes[row_number, 0].imshow(strip, cmap="gray")
        axes[row_number, 0].imshow(
            aperture_strip,
            cmap="viridis",
            alpha=0.42 * aperture_strip / max(float(aperture_strip.max()), EPS),
        )
        axes[row_number, 0].set_title(
            f"exact retinal frames + RF\nimage {int(selection.image_index)}, trace {int(selection.trace_index)}, RR100 {unit}"
        )
        axes[row_number, 0].axis("off")

        unit_rows = all_rows.loc[all_rows.rr100_index.eq(unit)].sort_values("matrix_row_index")
        x = np.arange(len(unit_rows))
        axes[row_number, 1].plot(x, unit_rows.radial_drive_z_across_conditions, "o-", color="0.5", label="radial power")
        axes[row_number, 1].plot(x, unit_rows.oriented_drive_z_across_conditions, "o-", color="#D55E00", label="oriented power")
        axes[row_number, 1].plot(x, unit_rows.activation_rms_z_across_conditions, "o-", color="#0072B2", label="full-twin activation")
        axes[row_number, 1].set_xticks(
            x,
            [condition_labels[int(value)] for value in unit_rows.matrix_row_index],
            rotation=25,
            ha="right",
            fontsize=7,
        )
        axes[row_number, 1].set(ylabel="within-unit z-score across 3 inputs", title=f"condition ordering\n{selection.selection_role}")
        axes[row_number, 1].axhline(0, color="0.75", lw=0.8)
        axes[row_number, 1].legend(frameon=False, fontsize=7)

        tc = timecourses[(condition, unit)]
        time_ms = np.arange(len(tc["moving_rate_hz"])) / FRAME_RATE_HZ * 1000.0
        axes[row_number, 2].plot(time_ms, tc["stabilized_rate_hz"], color="0.45", lw=1.5, label="stabilized")
        axes[row_number, 2].plot(time_ms, tc["moving_rate_hz"], color="#0072B2", lw=1.5, label="moving full twin")
        axes[row_number, 2].set(xlabel="time (ms)", ylabel="rate (Hz)", title="observed full-twin timecourses")
        axes[row_number, 2].legend(frameon=False, fontsize=7)

        axes[row_number, 3].plot(time_ms, tc["delta_rate_hz"], color="#0072B2", lw=1.5)
        axes[row_number, 3].fill_between(time_ms, 0, tc["delta_rate_hz"], color="#0072B2", alpha=0.2)
        axes[row_number, 3].axhline(0, color="0.5", ls="--")
        axes[row_number, 3].set(
            xlabel="time (ms)",
            ylabel="moving − stabilized rate (Hz)",
            title=f"activation RMS={selection.full_twin_activation_rms_hz:.3f} Hz\nmean change={selection.full_twin_delta_mean_rate_hz:+.3f} Hz",
        )

        axes[row_number, 4].scatter(
            unit_rows.radial_drive_z_across_conditions,
            unit_rows.activation_rms_z_across_conditions,
            s=65,
            color="0.5",
        )
        for point in unit_rows.itertuples(index=False):
            axes[row_number, 4].annotate(
                str(int(point.image_index)),
                (point.radial_drive_z_across_conditions, point.activation_rms_z_across_conditions),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axes[row_number, 4].set(xlabel="radial drive z", ylabel="activation RMS z", title=f"radial: r={selection.radial_vs_activation_r:+.2f}")

        axes[row_number, 5].scatter(
            unit_rows.oriented_drive_z_across_conditions,
            unit_rows.activation_rms_z_across_conditions,
            s=65,
            color="#D55E00",
        )
        for point in unit_rows.itertuples(index=False):
            axes[row_number, 5].annotate(
                str(int(point.image_index)),
                (point.oriented_drive_z_across_conditions, point.activation_rms_z_across_conditions),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axes[row_number, 5].set(xlabel="oriented drive z", ylabel="activation RMS z", title=f"oriented: r={selection.oriented_vs_activation_r:+.2f}")
        for axis in axes[row_number, 4:6]:
            axis.axhline(0, color="0.8", lw=0.8)
            axis.axvline(0, color="0.8", lw=0.8)

        axes[row_number, 6].axis("off")
        axes[row_number, 6].text(0.02, 0.95, selection.selection_role, va="top", fontsize=11, weight="bold")
        axes[row_number, 6].text(
            0.02,
            0.76,
            f"RR100 {unit}\n"
            f"oriented/radial drive = {selection.orientation_to_radial_ratio:.2f}\n"
            f"pair error improvement = {selection.orientation_pair_error_improvement:+.2f}\n"
            f"radial activation r = {selection.radial_vs_activation_r:+.2f}\n"
            f"oriented activation r = {selection.oriented_vs_activation_r:+.2f}\n\n"
            "Only 3 conditions: correlations are\nexample diagnostics, not inference.",
            va="top",
            fontsize=9.5,
            linespacing=1.35,
        )

    figure.suptitle(
        "Concrete natural-image response checkpoint: does RF-local orientation-aware power track full-twin activation better than radial power?\n"
        "Exact moving and stabilized full-twin timecourses · three input-selected conditions · no recorded responses or population claim",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    power = pd.read_csv(args.input_dir / "selected_condition_unit_metrics.csv")
    conditions = pd.read_csv(args.input_dir / "selected_conditions.csv")
    response_rows, timecourses, audit = load_response_rows(
        power, conditions, args.response_cache_dir, args.assembled_dir
    )
    response_rows, unit_summary = add_scale_free_comparisons(response_rows)
    response_rows = response_rows.merge(
        unit_summary[["rr100_index", "radial_vs_activation_r", "oriented_vs_activation_r"]],
        on="rr100_index",
        how="left",
        validate="many_to_one",
    )
    selected = select_examples(response_rows)
    payload = load_selected_movies(conditions, args.cohort_dir, args.response_cache_dir, args.device)
    with np.load(args.rf_dir / "unit_rf_apertures.npz", allow_pickle=False) as archive:
        rf_units = archive["rr100_index"].astype(int)
        apertures = {
            int(unit): archive["spectral_aperture"][index].astype(float)
            for index, unit in enumerate(rf_units)
        }
    response_rows.to_csv(args.out_dir / "condition_unit_power_response_metrics.csv", index=False)
    unit_summary.to_csv(args.out_dir / "unit_three_condition_summary.csv", index=False)
    selected.to_csv(args.out_dir / "selected_response_examples.csv", index=False)
    audit.to_csv(args.out_dir / "response_join_audit.csv", index=False)
    ordered_keys = [(int(row.matrix_row_index), int(row.rr100_index)) for row in response_rows.itertuples(index=False)]
    np.savez_compressed(
        args.out_dir / "selected_full_twin_timecourses.npz",
        matrix_row_index=np.asarray([key[0] for key in ordered_keys], dtype=int),
        rr100_index=np.asarray([key[1] for key in ordered_keys], dtype=int),
        moving_rate_hz=np.stack([timecourses[key]["moving_rate_hz"] for key in ordered_keys]).astype(np.float32),
        stabilized_rate_hz=np.stack([timecourses[key]["stabilized_rate_hz"] for key in ordered_keys]).astype(np.float32),
        delta_rate_hz=np.stack([timecourses[key]["delta_rate_hz"] for key in ordered_keys]).astype(np.float32),
    )
    figure_base = args.out_dir / "natural_image_rf_local_oriented_power_response_checkpoint"
    plot_checkpoint(selected, response_rows, timecourses, payload, apertures, figure_base, int(args.dpi))
    audit_columns = [column for column in audit if column.endswith("_error_vs_assembled_hz") or column.endswith("_error_vs_shard_hz")]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_natural_image_rf_local_oriented_power_full_twin_response_checkpoint",
        "status": "concrete_response_checkpoint_complete",
        "scope": {"conditions": int(response_rows.matrix_row_index.nunique()), "units": int(response_rows.rr100_index.nunique()), "pairs": int(len(response_rows))},
        "response_contract": {
            "primary": "full-twin RMS over 40 frames of moving rate minus matched stabilized rate",
            "diagnostic": "signed full-twin moving-minus-stabilized mean-rate change",
            "timecourses": "exact cached full-twin 40-frame rates for moving and matched stabilized conditions",
            "recorded_response": False,
        },
        "comparison_contract": {
            "radial": "RF-local direct-F0-weighted SFxTF power from the preceding verified input checkpoint",
            "oriented": "RF-local direct-F0-weighted SFxorientationxTF power from the preceding verified input checkpoint",
            "scale_free_example_metric": "within-unit z-score agreement across the three input-selected conditions",
            "inference": "none; n=3 condition correlations are descriptive example diagnostics only",
        },
        "verification": {
            "maximum_response_join_or_formula_error_hz": float(audit[audit_columns].to_numpy(float).max()),
            "all_response_values_finite": bool(np.isfinite(response_rows.select_dtypes("number")).all().all()),
        },
        "inputs": {
            "power_metrics": file_identity(args.input_dir / "selected_condition_unit_metrics.csv"),
            "condition_index": file_identity(args.assembled_dir / "condition_index.csv"),
            "assembled_manifest": file_identity(args.assembled_dir / "manifest.json"),
        },
        "artifacts": {
            "figure_png": figure_base.with_suffix(".png").name,
            "figure_pdf": figure_base.with_suffix(".pdf").name,
            "pair_metrics": "condition_unit_power_response_metrics.csv",
            "unit_summary": "unit_three_condition_summary.csv",
            "selected_examples": "selected_response_examples.csv",
            "join_audit": "response_join_audit.csv",
            "timecourses": "selected_full_twin_timecourses.npz",
        },
        "next_checkpoint": "expand only after deciding whether the visible positive, failure, and control cases support a population-scale RF-local rerun",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(unit_summary.to_string(index=False))


if __name__ == "__main__":
    main()
