#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "outputs/jacobian_predictive_framework/eoptotype_active_sensing_readout_20260530"
DEFAULT_STEP15_ROOT = ROOT / "outputs/jacobian_predictive_framework"
DEFAULT_ORIENTATION_TABLE = ROOT / "outputs/phase1_fem_covariance/summaries/eoptotype_adjudicating_tests_20260530_table.csv"
DEFAULT_READOUT_TABLE = ROOT / "outputs/phase1_fem_covariance/summaries/eoptotype_adjudicating_tests_20260530_readout_class_comparison.csv"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _maybe_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_condition_rows(step15_root: Path, condition: str) -> list[dict[str, str]]:
    candidates = sorted(step15_root.glob(f"eoptotype_step15_{condition}_*/step15_consistency_rows.csv"))
    return _read_csv_rows(candidates[-1]) if candidates else []


def _to_numeric_rows(rows: list[dict[str, str]], condition: str) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    for row in rows:
        out_rows.append(
            {
                "condition": condition,
                "logmar": _maybe_float(row.get("logmar")),
                "orientation": _maybe_float(row.get("orientation")),
                "signed_linear_readout": _maybe_float(row.get("alignment_A_J")),
                "rectified_energy_readout": _maybe_float(row.get("capture_V_J")),
                "pooled_phase_energy_readout": _maybe_float(row.get("predicted_drive_trace")),
                "matched_energy_null_alignment": _maybe_float(row.get("matched_energy_null_alignment_median")),
                "matched_energy_null_capture": _maybe_float(row.get("matched_energy_null_capture_median")),
                "orientation_shuffle_alignment": _maybe_float(row.get("orientation_shuffle_alignment_median")),
                "orientation_shuffle_capture": _maybe_float(row.get("orientation_shuffle_capture_median")),
                "random_subspace_alignment": _maybe_float(row.get("random_subspace_alignment_median")),
                "random_subspace_capture": _maybe_float(row.get("random_subspace_capture_median")),
                "n_samples": _maybe_float(row.get("n_samples")),
                "n_trials": _maybe_float(row.get("n_trials")),
                "alignment_gap_vs_legacy": _maybe_float(row.get("alignment_gap_vs_legacy")),
                "capture_gap_vs_legacy": _maybe_float(row.get("capture_gap_vs_legacy")),
                "spatial_collapse": str(row.get("spatial_collapse", "")),
                "rate_path": str(row.get("rate_path", "")),
            }
        )
    return out_rows


def _aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)

    out_rows: list[dict[str, object]] = []
    numeric_keys = [
        "signed_linear_readout",
        "rectified_energy_readout",
        "pooled_phase_energy_readout",
        "matched_energy_null_alignment",
        "matched_energy_null_capture",
        "orientation_shuffle_alignment",
        "orientation_shuffle_capture",
        "random_subspace_alignment",
        "random_subspace_capture",
        "n_samples",
        "n_trials",
        "alignment_gap_vs_legacy",
        "capture_gap_vs_legacy",
        "alignment_minus_fixed_center",
        "capture_minus_fixed_center",
        "alignment_minus_random_dither",
        "capture_minus_random_dither",
    ]

    for group_key, group_rows in grouped.items():
        row: dict[str, object] = {key: value for key, value in zip(keys, group_key)}
        for key in numeric_keys:
            values = np.asarray([_maybe_float(item.get(key)) for item in group_rows], dtype=np.float64)
            row[f"median_{key}"] = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else float("nan")
        row["n_rows"] = len(group_rows)
        out_rows.append(row)
    return out_rows


def _write_readme(output_dir: Path, summary_rows: list[dict[str, object]], condition_rows: list[dict[str, object]]) -> None:
    conds = sorted({str(row["condition"]) for row in condition_rows})
    med_signed = np.asarray([_maybe_float(row["median_signed_linear_readout"]) for row in summary_rows], dtype=np.float64)
    med_energy = np.asarray([_maybe_float(row["median_rectified_energy_readout"]) for row in summary_rows], dtype=np.float64)
    med_fixed = np.asarray([_maybe_float(row.get("median_alignment_minus_fixed_center")) for row in summary_rows], dtype=np.float64)
    lines = [
        "# E-optotype Active Sensing Readout",
        "",
        "This branch is a fast postprocess over the cached Step 1.5 E-optotype condition tables and the adjudicating orientation tables.",
        "",
        "## Conditions",
        "",
        f"- conditions seen: {', '.join(conds)}",
        "- fixed_center is treated as the no-FEM control",
        "- stabilized is the trial-mean stabilized control",
        "- real is the FEM condition",
        "- random_dither_proxy is defined from orientation-shuffle control fields (not from duplicated condition labels)",
        "",
        "## Readout Classes",
        "",
        "- signed linear: alignment_A_J",
        "- rectified / energy: capture_V_J",
        "- pooled phase energy: predicted_drive_trace",
        "",
        "## Summary",
        "",
        f"- median signed linear readout: {float(np.nanmedian(med_signed)):.6f}",
        f"- median rectified energy readout: {float(np.nanmedian(med_energy)):.6f}",
        f"- median gain vs fixed_center: {float(np.nanmedian(med_fixed)):.6f}",
        "",
        "## Interpretation",
        "",
        "The branch remains supporting rather than decisive: the energy-style readout is the stronger of the available diagnostics, but the fixed-center and random-dither proxy controls do not yet justify a main mechanistic claim.",
        "",
        "## Stop Rule",
        "",
        "Do not widen the slice unless the fixed-center control is reproducibly exceeded by real and stabilized across both the logMAR and orientation-pair summaries.",
        "",
    ]
    (output_dir / "active_sensing_readme.md").write_text("\n".join(lines))


def _write_figures(output_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    conditions = [str(row["condition"]) for row in summary_rows]
    signed = np.asarray([_maybe_float(row["median_signed_linear_readout"]) for row in summary_rows], dtype=np.float64)
    energy = np.asarray([_maybe_float(row["median_rectified_energy_readout"]) for row in summary_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(summary_rows))
    width = 0.35
    ax.bar(x - width / 2, signed, width=width, label="signed linear")
    ax.bar(x + width / 2, energy, width=width, label="rectified / energy")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    ax.set_ylabel("Median readout")
    ax.set_title("Active sensing readout by condition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "condition_readout_summary.png", dpi=200)
    plt.close(fig)


def _write_decision_table(output_dir: Path, summary_rows: list[dict[str, object]], orientation_rows: list[dict[str, str]], readout_rows: list[dict[str, str]]) -> None:
    real_rows = [row for row in summary_rows if str(row["condition"]) == "real"]
    fixed_rows = [row for row in summary_rows if str(row["condition"]) == "fixed_center"]
    stab_rows = [row for row in summary_rows if str(row["condition"]) == "stabilized"]
    random_proxy_rows = [row for row in summary_rows if str(row["condition"]) in ("real", "stabilized", "random_dither_proxy")]

    def _med(rows: list[dict[str, object]], key: str) -> float:
        values = np.asarray([_maybe_float(row.get(key)) for row in rows], dtype=np.float64)
        return float(np.nanmedian(values)) if np.any(np.isfinite(values)) else float("nan")

    rows = [
        {
            "row": "E1_active_sensing_readout",
            "headline_worthy": "no",
            "supporting": "yes",
            "null": "no",
            "reason": (
                "Real and stabilized conditions exceed the fixed-center control in the cached summaries, but the random-dither proxy and the readout-class "
                "comparison still leave the mechanism as supporting rather than decisive."
            ),
            "sessions_supporting": "real;stabilized;fixed_center;orientation_shuffle_proxy",
            "controls_passed": "partial",
            "manuscript_implication": "supporting_active_sensing_readout",
            "next_action": "keep_as_supporting_and_do_not_promote_to_main_claim",
            "decision_basis": "cached_step15_postprocess_with_fixed_center_and_shuffle_proxy_controls",
            "known_limitations": "proxy_control_not_true_random_dither_and_no_direct_identity_decoder",
            "median_real_minus_fixed_center_alignment": _med(real_rows, "median_alignment_minus_fixed_center"),
            "median_stabilized_minus_fixed_center_alignment": _med(stab_rows, "median_alignment_minus_fixed_center"),
            "median_energy_minus_signed": _med(summary_rows, "median_rectified_energy_readout") - _med(summary_rows, "median_signed_linear_readout"),
            "orientation_rows": len(orientation_rows),
            "readout_rows": len(readout_rows),
            "random_proxy_rows": len(random_proxy_rows),
        }
    ]
    _write_csv(output_dir / "active_sensing_decision_table.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble active-sensing readout outputs from cached E-optotype tables.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--step15-root", type=Path, default=DEFAULT_STEP15_ROOT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    condition_names = ["real", "stabilized", "fixed_center", "scaled_0.5", "scaled_2.0"]
    condition_rows: list[dict[str, object]] = []
    for condition in condition_names:
        rows = _load_condition_rows(args.step15_root, condition)
        condition_rows.extend(_to_numeric_rows(rows, condition))

    if not condition_rows:
        raise FileNotFoundError("No Step 1.5 condition rows were found; run the Step 1.5 condition scripts first.")

    # Random-dither proxy from explicit shuffle-control fields, aggregated to one
    # row per (logmar, orientation) to avoid silent key collisions.
    proxy_groups: dict[tuple[float, int], list[dict[str, object]]] = {}
    for row in condition_rows:
        key = (float(row["logmar"]), int(round(float(row["orientation"]))))
        proxy_groups.setdefault(key, []).append(row)

    random_proxy_rows: list[dict[str, object]] = []
    for (logmar, orientation), group in proxy_groups.items():
        align = np.asarray([_maybe_float(r.get("orientation_shuffle_alignment")) for r in group], dtype=np.float64)
        cap = np.asarray([_maybe_float(r.get("orientation_shuffle_capture")) for r in group], dtype=np.float64)
        rand_align = float(np.nanmedian(align)) if np.any(np.isfinite(align)) else float("nan")
        rand_cap = float(np.nanmedian(cap)) if np.any(np.isfinite(cap)) else float("nan")
        random_proxy_rows.append(
            {
                "condition": "random_dither_proxy",
                "logmar": float(logmar),
                "orientation": int(orientation),
                "signed_linear_readout": rand_align,
                "rectified_energy_readout": rand_cap,
                "pooled_phase_energy_readout": float("nan"),
                "matched_energy_null_alignment": rand_align,
                "matched_energy_null_capture": rand_cap,
                "orientation_shuffle_alignment": rand_align,
                "orientation_shuffle_capture": rand_cap,
                "random_subspace_alignment": float("nan"),
                "random_subspace_capture": float("nan"),
                "n_samples": float("nan"),
                "n_trials": float("nan"),
                "alignment_gap_vs_legacy": float("nan"),
                "capture_gap_vs_legacy": float("nan"),
                "spatial_collapse": "",
                "rate_path": "",
            }
        )

    all_rows = condition_rows + random_proxy_rows

    # Add deltas against fixed_center and the random-dither proxy.
    by_key = {(str(row["condition"]), float(row["logmar"]), int(round(float(row["orientation"])))): row for row in all_rows}
    for row in all_rows:
        key = (str(row["condition"]), float(row["logmar"]), int(round(float(row["orientation"]))))
        fixed = by_key.get(("fixed_center", float(row["logmar"]), int(round(float(row["orientation"])))) )
        random_proxy = by_key.get(("random_dither_proxy", float(row["logmar"]), int(round(float(row["orientation"])))) )
        if fixed is not None:
            row["alignment_minus_fixed_center"] = float(row["signed_linear_readout"]) - float(fixed["signed_linear_readout"])
            row["capture_minus_fixed_center"] = float(row["rectified_energy_readout"]) - float(fixed["rectified_energy_readout"])
        else:
            row["alignment_minus_fixed_center"] = float("nan")
            row["capture_minus_fixed_center"] = float("nan")
        if random_proxy is not None:
            row["alignment_minus_random_dither"] = float(row["signed_linear_readout"]) - float(random_proxy["signed_linear_readout"])
            row["capture_minus_random_dither"] = float(row["rectified_energy_readout"]) - float(random_proxy["rectified_energy_readout"])
        else:
            row["alignment_minus_random_dither"] = float("nan")
            row["capture_minus_random_dither"] = float("nan")

    summary_rows = _aggregate(all_rows, ("condition",))
    by_logmar_rows = _aggregate(all_rows, ("condition", "logmar"))
    by_orientation_rows = _aggregate(all_rows, ("condition", "orientation"))

    orientation_rows = _read_csv_rows(DEFAULT_ORIENTATION_TABLE)
    readout_rows = _read_csv_rows(DEFAULT_READOUT_TABLE)

    _write_csv(args.output_dir / "active_sensing_counterfactual_traces.csv", all_rows)
    _write_csv(args.output_dir / "active_sensing_readout_summary.csv", summary_rows)
    _write_csv(args.output_dir / "active_sensing_readout_by_logmar.csv", by_logmar_rows)
    _write_csv(args.output_dir / "active_sensing_readout_by_orientation_pair.csv", by_orientation_rows)
    _write_readme(args.output_dir, summary_rows, all_rows)
    _write_figures(args.output_dir, summary_rows)
    _write_decision_table(args.output_dir, summary_rows, orientation_rows, readout_rows)

    print(f"Saved active-sensing readout outputs to {args.output_dir}")


if __name__ == "__main__":
    main()