"""Summarize covariance-aware FEM optimality outputs.

Example
-------
.venv/bin/python declan/active_sensing_movie_information/summarize_covariance_optimality.py \
  --covopt-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu/covariance_optimality/covopt_smoke \
  --figure5-run-dir outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu \
  --out-dir outputs/active_sensing_movie_information/covariance_optimality/covopt_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUT_DIR = Path("outputs/active_sensing_movie_information/covariance_optimality")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_sem(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    if arr.size == 1:
        return float(arr[0]), 0.0, 1
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size)), int(arr.size)


def pair_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("example_id", "")),
        str(row.get("kind", "")),
        str(row.get("image_index", "")),
        str(row.get("crop_rank", "0")),
    )


def summarize_scale_curves(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, str], list[float]] = {}
    groups_spikes: dict[tuple[str, str, float, str], list[float]] = {}
    for row in rows:
        if not np.isclose(fnum(row, "rate_gain", 1.0), 1.0) or not np.isclose(fnum(row, "noise_floor_multiplier", 1.0), 1.0):
            continue
        key = (str(row["family"]), str(row.get("kind", "")), fnum(row, "scale_D"), str(row["regime"]))
        groups.setdefault(key, []).append(fnum(row, "final_fisher_trace_per_spike"))
        groups_spikes.setdefault(key, []).append(fnum(row, "final_expected_spikes"))
    out: list[dict[str, Any]] = []
    for (family, kind, scale, regime), vals in sorted(groups.items()):
        mean, sem, n = mean_sem(vals)
        spikes_mean, spikes_sem, _ = mean_sem(groups_spikes[(family, kind, scale, regime)])
        out.append({
            "family": family,
            "kind": kind,
            "scale_D": scale,
            "regime": regime,
            "metric": "final_fisher_trace_per_spike",
            "mean": mean,
            "sem": sem,
            "n": n,
            "expected_spikes_mean": spikes_mean,
            "expected_spikes_sem": spikes_sem,
        })
    return out


def paired_contrasts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[tuple[str, str, str, str], str, str, float], float] = {}
    for row in rows:
        if not np.isclose(fnum(row, "rate_gain", 1.0), 1.0) or not np.isclose(fnum(row, "noise_floor_multiplier", 1.0), 1.0):
            continue
        key = (pair_key(row), str(row["family"]), str(row.get("kind", "")), fnum(row, "scale_D"))
        by_key[(key, str(row["regime"]))] = fnum(row, "final_fisher_trace_per_spike")

    contrast_by_pair: dict[tuple[tuple[str, str, str, str], str, str, float, str], float] = {}
    keys = sorted({key for key, _regime in by_key})
    for key in keys:
        aware = by_key.get((key, "cov_pose_aware"))
        blind = by_key.get((key, "cov_pose_blind"))
        ind = by_key.get((key, "independent_pose_aware"))
        geometry_items = {
            regime: value
            for (candidate_key, regime), value in by_key.items()
            if candidate_key == key and str(regime).startswith("cov_geometry_aware")
        }
        _pair, family, kind, scale = key
        if aware is None:
            continue
        if blind is not None:
            contrast_by_pair[(_pair, family, kind, scale, "pose_gap")] = aware - blind
        if ind is not None:
            contrast_by_pair[(_pair, family, kind, scale, "independent_minus_cov_pose_aware")] = ind - aware
        if ind is not None and blind is not None:
            contrast_by_pair[(_pair, family, kind, scale, "independent_minus_cov_pose_blind")] = ind - blind
        for regime, geometry in geometry_items.items():
            suffix = str(regime).removeprefix("cov_geometry_aware_")
            contrast_by_pair[(_pair, family, kind, scale, f"pose_to_geometry_gap_{suffix}")] = aware - geometry
            if blind is not None:
                contrast_by_pair[(_pair, family, kind, scale, f"geometry_to_blind_gap_{suffix}")] = geometry - blind
                contrast_by_pair[(_pair, family, kind, scale, f"geometry_fraction_of_pose_gap_{suffix}")] = (
                    (geometry - blind) / (aware - blind)
                    if abs(aware - blind) > 1e-12
                    else float("nan")
                )

    corrected_by_pair: dict[tuple[tuple[str, str, str, str], str, str, float, str], float] = {}
    for (pair, family, kind, scale, contrast), value in contrast_by_pair.items():
        baseline = contrast_by_pair.get((pair, family, kind, 0.0, contrast))
        if baseline is None:
            continue
        corrected_by_pair[(pair, family, kind, scale, f"{contrast}_minus_D0")] = value - baseline

    contrasts: dict[tuple[str, str, float, str], list[float]] = {}
    for (_pair, family, kind, scale, contrast), value in {**contrast_by_pair, **corrected_by_pair}.items():
        contrasts.setdefault((family, kind, scale, contrast), []).append(value)

    out: list[dict[str, Any]] = []
    for (family, kind, scale, contrast), vals in sorted(contrasts.items()):
        mean, sem, n = mean_sem(vals)
        out.append({
            "family": family,
            "kind": kind,
            "scale_D": scale,
            "contrast": contrast,
            "mean": mean,
            "sem": sem,
            "n": n,
        })
    return out


def _shape_label(scales: np.ndarray, values: np.ndarray) -> str:
    keep = np.isfinite(scales) & np.isfinite(values)
    scales = scales[keep]
    values = values[keep]
    if values.size < 3 or np.nanmax(values) - np.nanmin(values) <= 1e-12:
        return "flat_or_unresolved"
    peak_i = int(np.nanargmax(values))
    peak_d = float(scales[peak_i])
    if np.all(np.diff(values) >= -1e-12):
        return "monotonic_increasing"
    if np.all(np.diff(values) <= 1e-12):
        return "monotonic_decreasing"
    empirical = values[np.argmin(np.abs(scales - 1.0))]
    peak = values[peak_i]
    if 0.75 <= peak_d <= 1.5:
        return "peak_near_empirical"
    if empirical >= 0.8 * peak:
        return "empirical_on_plateau"
    return "resolved_nonempirical_peak"


def decision_table(scale_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in scale_summary:
        groups.setdefault((str(row["family"]), str(row["kind"]), str(row["regime"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (family, kind, regime), rows in sorted(groups.items()):
        scales = np.asarray([fnum(row, "scale_D") for row in rows], dtype=np.float64)
        values = np.asarray([fnum(row, "mean") for row in rows], dtype=np.float64)
        if values.size == 0 or not np.any(np.isfinite(values)):
            continue
        peak_i = int(np.nanargmax(values))
        empirical_i = int(np.nanargmin(np.abs(scales - 1.0)))
        peak = float(values[peak_i])
        empirical = float(values[empirical_i])
        out.append({
            "family": family,
            "kind": kind,
            "metric": "final_fisher_trace_per_spike",
            "regime": regime,
            "D_empirical": 1.0,
            "empirical_value": empirical,
            "D_peak": float(scales[peak_i]),
            "peak_value": peak,
            "empirical_fraction_of_peak": empirical / peak if np.isfinite(peak) and peak != 0 else float("nan"),
            "empirical_on_80pct_plateau": bool(np.isfinite(peak) and empirical >= 0.8 * peak),
            "curve_shape_label": _shape_label(scales, values),
        })
    return out


def sensitivity_decision_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, float], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["family"]),
                str(row.get("kind", "")),
                fnum(row, "rate_gain", 1.0),
                fnum(row, "noise_floor_multiplier", 1.0),
            ),
            [],
        ).append(row)
    out: list[dict[str, Any]] = []
    for (family, kind, gain, noise), vals in sorted(groups.items()):
        by_scale: dict[float, list[float]] = {}
        for row in vals:
            by_scale.setdefault(fnum(row, "scale_D"), []).append(fnum(row, "final_fisher_trace_per_spike"))
        scales = np.asarray(sorted(by_scale), dtype=np.float64)
        values = np.asarray([mean_sem(by_scale[float(scale)])[0] for scale in scales], dtype=np.float64)
        if values.size == 0 or not np.any(np.isfinite(values)):
            continue
        peak_i = int(np.nanargmax(values))
        empirical_i = int(np.nanargmin(np.abs(scales - 1.0)))
        peak = float(values[peak_i])
        empirical = float(values[empirical_i])
        out.append({
            "family": family,
            "kind": kind,
            "metric": "final_fisher_trace_per_spike",
            "regime": "cov_pose_blind_sensitivity",
            "rate_gain": gain,
            "noise_floor_multiplier": noise,
            "D_empirical": 1.0,
            "empirical_value": empirical,
            "D_peak": float(scales[peak_i]),
            "peak_value": peak,
            "empirical_fraction_of_peak": empirical / peak if np.isfinite(peak) and peak != 0 else float("nan"),
            "empirical_on_80pct_plateau": bool(np.isfinite(peak) and empirical >= 0.8 * peak),
            "curve_shape_label": _shape_label(scales, values),
            "n_scales": int(scales.size),
        })
    return out


def summarize_alignment(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, int], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((str(row["family"]), str(row.get("kind", "")), fnum(row, "scale_D"), int(fnum(row, "k", 0))), []).append(row)
    out: list[dict[str, Any]] = []
    metrics = ("coding_variance_fem", "fem_variance_coding", "signal_variance_fem", "fem_variance_signal")
    for (family, kind, scale, k), vals in sorted(groups.items()):
        row: dict[str, Any] = {"family": family, "kind": kind, "scale_D": scale, "k": k, "n": len(vals)}
        for metric in metrics:
            mean, sem, _ = mean_sem([fnum(v, metric) for v in vals])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sem"] = sem
        out.append(row)
    return out


def plot_scale_curves(scale_summary: list[dict[str, Any]], path: Path) -> None:
    regimes = ["independent_pose_aware", "cov_pose_aware"]
    regimes.extend(sorted({str(row["regime"]) for row in scale_summary if str(row["regime"]).startswith("cov_geometry_aware")}))
    regimes.append("cov_pose_blind")
    colors = {
        "independent_pose_aware": "#2b6cb0",
        "cov_pose_aware": "#2f855a",
        "cov_pose_blind": "#c05621",
    }
    families = sorted({str(row["family"]) for row in scale_summary})
    kinds = sorted({str(row["kind"]) for row in scale_summary})
    fig, axs = plt.subplots(max(len(kinds), 1), max(len(families), 1), figsize=(5.2 * max(len(families), 1), 3.8 * max(len(kinds), 1)), squeeze=False)
    for r, kind in enumerate(kinds or [""]):
        for c, family in enumerate(families or [""]):
            ax = axs[r, c]
            for regime in regimes:
                rows = [row for row in scale_summary if row["family"] == family and row["kind"] == kind and row["regime"] == regime]
                if not rows:
                    continue
                rows.sort(key=lambda row: fnum(row, "scale_D"))
                x = np.asarray([fnum(row, "scale_D") for row in rows])
                y = np.asarray([fnum(row, "mean") for row in rows])
                e = np.asarray([fnum(row, "sem", 0.0) for row in rows])
                color = colors.get(regime, None)
                ax.plot(x, y, marker="o", lw=2.0, color=color, label=regime)
                ax.fill_between(x, y - e, y + e, color=color, alpha=0.15, linewidth=0)
            ax.axvline(1.0, color="0.2", lw=1.0, ls="--")
            ax.set_title(f"{family} / {kind}")
            ax.set_xlabel("movement scale D")
            ax.set_ylabel("Fisher trace / expected spike")
            ax.grid(color="0.9", lw=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axs[0, 0].legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pose_gap(contrasts: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in contrasts if row["contrast"] == "pose_gap"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for key in sorted({(row["family"], row["kind"]) for row in rows}):
        family, kind = key
        subset = [row for row in rows if (row["family"], row["kind"]) == key]
        subset.sort(key=lambda row: fnum(row, "scale_D"))
        x = np.asarray([fnum(row, "scale_D") for row in subset])
        y = np.asarray([fnum(row, "mean") for row in subset])
        e = np.asarray([fnum(row, "sem", 0.0) for row in subset])
        ax.plot(x, y, marker="o", lw=2.0, label=f"{family} / {kind}")
        ax.fill_between(x, y - e, y + e, alpha=0.15, linewidth=0)
    ax.axhline(0.0, color="0.2", lw=1.0)
    ax.axvline(1.0, color="0.2", lw=1.0, ls="--")
    ax.set_xlabel("movement scale D")
    ax.set_ylabel("pose-aware minus pose-blind FI/spike")
    ax.grid(color="0.9", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_covariance_spectra(rows: list[dict[str, str]], path: Path) -> None:
    primary = [row for row in rows if row.get("estimator") == "pooled_residual"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for key in sorted({(row["family"], row.get("kind", "")) for row in primary}):
        subset = [row for row in primary if (row["family"], row.get("kind", "")) == key]
        subset.sort(key=lambda row: fnum(row, "scale_D"))
        ax.plot([fnum(row, "scale_D") for row in subset], [fnum(row, "trace") for row in subset], marker="o", lw=2.0, label=f"{key[0]} / {key[1]}")
    ax.axvline(1.0, color="0.2", lw=1.0, ls="--")
    ax.set_xlabel("movement scale D")
    ax.set_ylabel("trace Sigma_FEM")
    ax.grid(color="0.9", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_alignment(rows: list[dict[str, Any]], path: Path) -> None:
    subset = [row for row in rows if int(row["k"]) == 10]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for metric, label in (
        ("coding_variance_fem_mean", "coding capture"),
        ("signal_variance_fem_mean", "signal capture"),
    ):
        grouped = {}
        for row in subset:
            grouped.setdefault((row["family"], row["kind"]), []).append(row)
        for key, vals in sorted(grouped.items()):
            vals.sort(key=lambda row: fnum(row, "scale_D"))
            ax.plot(
                [fnum(row, "scale_D") for row in vals],
                [fnum(row, metric) for row in vals],
                marker="o",
                lw=2.0,
                label=f"{label}: {key[0]} / {key[1]}",
            )
    ax.axvline(1.0, color="0.2", lw=1.0, ls="--")
    ax.set_xlabel("movement scale D")
    ax.set_ylabel("variance capture, k=10")
    ax.set_ylim(bottom=0.0)
    ax.grid(color="0.9", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary_md(
    path: Path,
    decision: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    sensitivity_decisions: list[dict[str, Any]],
) -> None:
    pose_at_empirical = [
        row for row in contrasts
        if row["contrast"] == "pose_gap" and np.isclose(fnum(row, "scale_D"), 1.0)
    ]
    pose_corrected_at_empirical = [
        row for row in contrasts
        if row["contrast"] == "pose_gap_minus_D0" and np.isclose(fnum(row, "scale_D"), 1.0)
    ]
    lines = [
        "# Covariance-Aware FEM Optimality Summary",
        "",
        "Primary interpretation: D=0-corrected pose-aware minus pose-blind covariance-aware Fisher efficiency.",
        "",
        "## Pose gap at empirical scale, raw",
        "",
    ]
    if pose_at_empirical:
        for row in pose_at_empirical:
            lines.append(
                f"- {row['family']} / {row['kind']}: mean={fnum(row, 'mean'):.6g}, "
                f"SEM={fnum(row, 'sem'):.6g}, n={int(row['n'])}"
            )
    else:
        lines.append("- No D=1 pose-gap rows found.")
    lines.extend(["", "## Pose gap at empirical scale, D0-corrected", ""])
    if pose_corrected_at_empirical:
        for row in pose_corrected_at_empirical:
            lines.append(
                f"- {row['family']} / {row['kind']}: mean={fnum(row, 'mean'):.6g}, "
                f"SEM={fnum(row, 'sem'):.6g}, n={int(row['n'])}"
            )
    else:
        lines.append("- No D=1 D0-corrected pose-gap rows found.")
    lines.extend(["", "## Curve labels", ""])
    for row in decision:
        lines.append(
            f"- {row['family']} / {row['kind']} / {row['regime']}: "
            f"{row['curve_shape_label']} (D_peak={fnum(row, 'D_peak'):.6g}, "
            f"empirical_fraction_of_peak={fnum(row, 'empirical_fraction_of_peak'):.3g})"
        )
    if sensitivity_decisions:
        lines.extend(["", "## Gain/noise sensitivity", ""])
        for row in sensitivity_decisions:
            lines.append(
                f"- {row['family']} / {row['kind']} / gain={fnum(row, 'rate_gain'):.3g} "
                f"/ noise={fnum(row, 'noise_floor_multiplier'):.3g}: "
                f"{row['curve_shape_label']} (D_peak={fnum(row, 'D_peak'):.6g}, "
                f"empirical_fraction_of_peak={fnum(row, 'empirical_fraction_of_peak'):.3g})"
            )
    lines.extend([
        "",
        "Caveat: peak language should be restricted to the independently checked linear-validity range and should be demoted if the gain/noise sensitivity grid moves the peak strongly.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_covariance_optimality(covopt_dir: Path, figure5_run_dir: Path | None, out_dir: Path) -> dict[str, Any]:
    del figure5_run_dir
    row_metrics = read_csv_rows(covopt_dir / "results" / "covopt_row_metrics.csv")
    sensitivity_metrics = read_csv_rows(covopt_dir / "results" / "covopt_sensitivity_row_metrics.csv")
    cov_rows = read_csv_rows(covopt_dir / "results" / "covopt_covariance_spectra.csv")
    align_rows = read_csv_rows(covopt_dir / "results" / "covopt_alignment_diagnostics.csv")
    if not row_metrics:
        raise FileNotFoundError(f"No row metrics found at {covopt_dir / 'results' / 'covopt_row_metrics.csv'}")

    scale_summary = summarize_scale_curves(row_metrics)
    contrasts = paired_contrasts(row_metrics)
    decisions = decision_table(scale_summary)
    sensitivity_decisions = sensitivity_decision_table(sensitivity_metrics)
    align_summary = summarize_alignment(align_rows)

    figures_dir = out_dir / "figures"
    write_csv_rows(out_dir / "covopt_scale_summary.csv", scale_summary)
    write_csv_rows(out_dir / "covopt_paired_contrasts.csv", contrasts)
    write_csv_rows(out_dir / "covopt_decision_table.csv", decisions)
    write_csv_rows(out_dir / "covopt_sensitivity_decision_table.csv", sensitivity_decisions)
    write_csv_rows(out_dir / "covopt_alignment_summary.csv", align_summary)
    plot_scale_curves(scale_summary, figures_dir / "covopt_scale_curves.pdf")
    plot_pose_gap(contrasts, figures_dir / "covopt_pose_gap.pdf")
    plot_covariance_spectra(cov_rows, figures_dir / "covopt_covariance_spectra.pdf")
    plot_alignment(align_summary, figures_dir / "covopt_alignment.pdf")
    write_summary_md(out_dir / "covopt_summary.md", decisions, contrasts, sensitivity_decisions)

    manifest = {
        "covopt_dir": str(covopt_dir),
        "out_dir": str(out_dir),
        "n_row_metrics": len(row_metrics),
        "n_scale_summary_rows": len(scale_summary),
        "n_contrast_rows": len(contrasts),
        "n_decision_rows": len(decisions),
        "n_sensitivity_decision_rows": len(sensitivity_decisions),
        "outputs": {
            "scale_summary": str(out_dir / "covopt_scale_summary.csv"),
            "paired_contrasts": str(out_dir / "covopt_paired_contrasts.csv"),
            "decision_table": str(out_dir / "covopt_decision_table.csv"),
            "sensitivity_decision_table": str(out_dir / "covopt_sensitivity_decision_table.csv"),
            "alignment_summary": str(out_dir / "covopt_alignment_summary.csv"),
            "summary_md": str(out_dir / "covopt_summary.md"),
            "figures": sorted(str(path) for path in figures_dir.glob("*.pdf")),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covopt-dir", required=True, type=Path)
    parser.add_argument("--figure5-run-dir", default=None, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = summarize_covariance_optimality(args.covopt_dir, args.figure5_run_dir, args.out_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
