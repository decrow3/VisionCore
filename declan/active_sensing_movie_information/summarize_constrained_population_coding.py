"""Summarize constrained population-coding Check 6 outputs for Figure 5.

This is the Priority 2 dashboard for the reframed Figure 5 story. It reads
saved Check 6 pairwise dprime rows and reports whether covariance-aware
population coding is helped or hurt by retinal-motion conditions.

The default input is the current natural-image center-channel run. That run is
useful as a natural-image sanity check, but it is still not the canonical
756-channel comparison to the older e-optotype scaffold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN_DIR = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "figure5_natural_image_population_checks_5_to_9"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "constrained_population_coding"
)
DEFAULT_CONTRASTS = "stabilized,random_cov,random_amp,random_amp_cloud_matched,trajectory_order_shuffle"
METRICS = ("dprime2_pop", "dprime2_indep", "eta_pop_over_indep")


def parse_csv_text(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def stable_sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def stable_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def condition_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_condition.setdefault(str(row.get("condition", "")), []).append(row)

    out: list[dict[str, Any]] = []
    for condition, group in sorted(by_condition.items()):
        summary: dict[str, Any] = {
            "condition": condition,
            "n_pairs": len({row.get("pair", "") for row in group}),
            "n_rows": len(group),
            "source": group[0].get("source", "") if group else "",
            "ridge_fraction": fnum(group[0], "ridge_fraction") if group else float("nan"),
        }
        for metric in METRICS:
            vals = [fnum(row, metric) for row in group]
            summary[f"{metric}_mean"] = stable_mean(vals)
            summary[f"{metric}_sem"] = stable_sem(vals)
            summary[f"{metric}_median"] = stable_median(vals)
        out.append(summary)
    return out


def paired_contrasts(rows: list[dict[str, str]], reference: str, controls: list[str]) -> list[dict[str, Any]]:
    by_condition_pair: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        by_condition_pair[(str(row.get("condition", "")), str(row.get("pair", "")))] = row

    reference_pairs = {
        pair
        for condition, pair in by_condition_pair
        if condition == reference and pair
    }
    out: list[dict[str, Any]] = []
    for control in controls:
        control_pairs = {
            pair
            for condition, pair in by_condition_pair
            if condition == control and pair
        }
        common_pairs = sorted(reference_pairs & control_pairs)
        row_out: dict[str, Any] = {
            "reference_condition": reference,
            "control_condition": control,
            "n_common_pairs": len(common_pairs),
        }
        for metric in METRICS:
            ref_vals = [fnum(by_condition_pair[(reference, pair)], metric) for pair in common_pairs]
            control_vals = [fnum(by_condition_pair[(control, pair)], metric) for pair in common_pairs]
            deltas = [
                ref - ctrl
                for ref, ctrl in zip(ref_vals, control_vals, strict=True)
                if np.isfinite(ref) and np.isfinite(ctrl)
            ]
            ref_mean = stable_mean(ref_vals)
            control_mean = stable_mean(control_vals)
            delta_mean = stable_mean(deltas)
            row_out[f"{metric}_reference_mean"] = ref_mean
            row_out[f"{metric}_control_mean"] = control_mean
            row_out[f"{metric}_delta_mean"] = delta_mean
            row_out[f"{metric}_delta_sem"] = stable_sem(deltas)
            row_out[f"{metric}_delta_median"] = stable_median(deltas)
            row_out[f"{metric}_delta_positive_fraction"] = (
                float(np.mean(np.asarray(deltas, dtype=np.float64) > 0.0)) if deltas else float("nan")
            )
            row_out[f"{metric}_ratio_reference_over_control"] = (
                ref_mean / control_mean if np.isfinite(ref_mean) and np.isfinite(control_mean) and control_mean != 0 else float("nan")
            )
        row_out["priority2_read"] = priority2_read(row_out)
        out.append(row_out)
    return out


def priority2_read(row: dict[str, Any]) -> str:
    eta_delta = fnum(row, "eta_pop_over_indep_delta_mean")
    pop_delta = fnum(row, "dprime2_pop_delta_mean")
    control = str(row.get("control_condition", ""))
    if not np.isfinite(eta_delta) or not np.isfinite(pop_delta):
        return "missing"
    if control.startswith("random") and eta_delta <= 0:
        return "random_control_matches_or_exceeds_real_eta"
    if eta_delta > 0 and pop_delta < 0:
        return "higher_eta_but_lower_absolute_pop_dprime"
    if eta_delta > 0 and pop_delta >= 0:
        return "higher_eta_and_higher_absolute_pop_dprime"
    if eta_delta <= 0 and pop_delta < 0:
        return "lower_eta_and_lower_absolute_pop_dprime"
    return "lower_eta_but_higher_absolute_pop_dprime"


def write_summary_markdown(
    path: Path,
    condition_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    input_path: Path,
) -> None:
    lines = [
        "# Constrained Population-Coding Summary",
        "",
        f"Input: `{input_path}`",
        "",
        "This summarizes Check 6 with covariance-aware dprime and `eta = dprime2_pop / dprime2_indep`.",
        "The current default is the 16-channel natural-image center-response run, not the canonical 756-channel comparison.",
        "",
        "## Condition Means",
        "",
        "| Condition | n pairs | dprime2_pop | dprime2_indep | eta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in condition_rows:
        lines.append(
            "| {condition} | {n_pairs} | {pop:.3f} | {indep:.3f} | {eta:.3f} |".format(
                condition=row.get("condition", ""),
                n_pairs=int(fnum(row, "n_pairs")),
                pop=fnum(row, "dprime2_pop_mean"),
                indep=fnum(row, "dprime2_indep_mean"),
                eta=fnum(row, "eta_pop_over_indep_mean"),
            )
        )

    lines.extend([
        "",
        "## Paired Real-Minus-Control Contrasts",
        "",
        "| Control | n pairs | delta dprime2_pop | delta dprime2_indep | delta eta | eta positive frac | read |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in contrast_rows:
        lines.append(
            "| {control} | {n} | {pop:.3f} | {indep:.3f} | {eta:.3f} | {pos:.3f} | {read} |".format(
                control=row.get("control_condition", ""),
                n=int(fnum(row, "n_common_pairs")),
                pop=fnum(row, "dprime2_pop_delta_mean"),
                indep=fnum(row, "dprime2_indep_delta_mean"),
                eta=fnum(row, "eta_pop_over_indep_delta_mean"),
                pos=fnum(row, "eta_pop_over_indep_delta_positive_fraction"),
                read=row.get("priority2_read", ""),
            )
        )

    lines.extend([
        "",
        "## Working Interpretation",
        "",
        "- Real exceeds stabilized on `eta`, but has lower absolute covariance-aware `dprime2_pop`.",
        "- Random-motion controls are similar to or above real on `eta`, so this does not support real-trajectory optimality.",
        "- The result remains useful as a constrained-coding bridge: reafferent covariance can be benign/helpful by the `eta` ratio while still costing absolute population separability.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    input_path = in_dir / "check6_natural_image_constrained_dprime.csv"
    rows = read_csv_rows(input_path)
    if not rows:
        raise FileNotFoundError(f"No Check 6 rows found at {input_path}")

    controls = parse_csv_text(args.controls)
    condition_rows = condition_summary(rows)
    contrast_rows = paired_contrasts(rows, args.reference_condition, controls)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(out_dir / "constrained_population_condition_summary.csv", condition_rows)
    write_csv_rows(out_dir / "constrained_population_real_contrasts.csv", contrast_rows)
    write_summary_markdown(out_dir / "constrained_population_summary.md", condition_rows, contrast_rows, input_path)

    manifest = {
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "reference_condition": args.reference_condition,
        "controls": controls,
        "n_input_rows": len(rows),
        "outputs": [
            "constrained_population_condition_summary.csv",
            "constrained_population_real_contrasts.csv",
            "constrained_population_summary.md",
        ],
        "guardrail": "Default input is 16-channel natural-image center responses; canonical 756-channel natural-image Check 6 remains tabled.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Constrained population-coding summary complete")
    print(f"  input: {input_path}")
    print(f"  out_dir: {out_dir}")
    print(f"  input rows: {len(rows)}")
    print(f"  condition rows: {len(condition_rows)}")
    print(f"  contrast rows: {len(contrast_rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reference-condition", type=str, default="real")
    parser.add_argument("--controls", type=str, default=DEFAULT_CONTRASTS)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
