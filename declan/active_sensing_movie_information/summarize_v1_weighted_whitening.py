#!/usr/bin/env python3
"""Summarize V1 temporal-sensitivity-weighted input whitening.

This script consumes ``retinal_temporal_psd_by_movie.csv`` from
``run_input_whitening_optimum.py --write-psd-rows true``.  It is intentionally
cache-first: it does not render retinal movies and does not run the digital
twin.

The scientific question is whether raw retinal whitening remains boundary-
limited after temporal frequencies are weighted by simple, predeclared proxies
for the frequencies foveal V1 can use.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from non_circular_fem_common import DEFAULT_STACK_OUT_DIR, parse_float, read_csv_rows, write_csv_rows, write_json


DEFAULT_IN_DIR = DEFAULT_STACK_OUT_DIR / "input_whitening_primary_psd"
DEFAULT_OUT_DIR = DEFAULT_STACK_OUT_DIR / "v1_weighted_whitening"
BIOLOGICAL_D = 1.0
ExternalWeights = dict[str, tuple[np.ndarray, np.ndarray]]


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    return parse_float(row.get(key), default)


def mean_sem(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    if arr.size == 1:
        return float(arr[0]), 0.0, 1
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size)), int(arr.size)


def spectral_flatness(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(vals + 1e-30))) / max(float(np.mean(vals)), 1e-30))


def spectral_entropy(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 2:
        return float("nan")
    probs = vals / max(float(np.sum(vals)), 1e-30)
    return float(-np.sum(probs * np.log(probs + 1e-30)) / np.log(vals.size))


def abs_loglog_slope(freq: np.ndarray, values: np.ndarray) -> float:
    f = np.asarray(freq, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(f) & np.isfinite(v) & (f > 0) & (v > 0)
    if int(np.sum(keep)) < 2:
        return float("nan")
    slope = float(np.polyfit(np.log10(f[keep]), np.log10(v[keep]), 1)[0])
    return abs(slope)


def load_external_weight_specs(path: Path | None, default_name: str) -> ExternalWeights:
    if path is None:
        return {}
    rows = read_csv_rows(path)
    if not rows:
        raise FileNotFoundError(f"No external temporal-weight rows found at {path}")
    freq_keys = ("temporal_frequency_hz", "frequency_hz", "freq_hz", "freq")
    weight_keys = (
        "normalized_weight",
        "weight",
        "mean_response_gain_sq",
        "response_gain_sq",
        "response_amp_rms",
        "mean_response_amp_rms",
    )
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        freq_val = float("nan")
        for key in freq_keys:
            if key in row:
                freq_val = fnum(row, key)
                break
        weight_val = float("nan")
        for key in weight_keys:
            if key in row:
                weight_val = fnum(row, key)
                break
        if not np.isfinite(freq_val) or not np.isfinite(weight_val):
            continue
        name = str(row.get("weight_name") or row.get("name") or default_name)
        grouped[name][float(freq_val)].append(float(weight_val))
    specs: ExternalWeights = {}
    for name, by_freq in grouped.items():
        freq = np.asarray(sorted(by_freq), dtype=np.float64)
        weights = np.asarray([np.mean(by_freq[float(f)]) for f in freq], dtype=np.float64)
        keep = np.isfinite(freq) & np.isfinite(weights) & (freq > 0) & (weights >= 0)
        freq = freq[keep]
        weights = weights[keep]
        if freq.size == 0:
            continue
        max_weight = float(np.nanmax(weights))
        if max_weight > 0:
            weights = weights / max_weight
        specs[name] = (freq, weights)
    if not specs:
        raise ValueError(f"Could not parse any usable external temporal weights from {path}")
    return specs


def temporal_weight(
    freq: np.ndarray,
    name: str,
    external_weights: ExternalWeights | None = None,
    renormalize_external_within_passband: bool = False,
) -> np.ndarray:
    f = np.asarray(freq, dtype=np.float64)
    w = np.zeros_like(f, dtype=np.float64)
    is_external = False
    if name == "uniform_1_30hz":
        w[(f >= 1.0) & (f <= 30.0)] = 1.0
    elif name == "bandpass_2_20hz":
        w[(f >= 2.0) & (f <= 20.0)] = 1.0
    elif name == "bandpass_4_30hz":
        w[(f >= 4.0) & (f <= 30.0)] = 1.0
    elif name == "soft_v1_log_gaussian_8hz":
        positive = f > 0
        w[positive] = np.exp(-0.5 * (np.log2(f[positive] / 8.0) / 0.85) ** 2)
    elif name == "soft_v1_log_gaussian_12hz":
        positive = f > 0
        w[positive] = np.exp(-0.5 * (np.log2(f[positive] / 12.0) / 0.85) ** 2)
    elif name == "lowpass_30hz":
        positive = f > 0
        w[positive] = 1.0 / (1.0 + (f[positive] / 30.0) ** 4)
    elif external_weights and name in external_weights:
        is_external = True
        wf, ww = external_weights[name]
        w = np.interp(f, wf, ww, left=0.0, right=0.0)
    else:
        raise ValueError(f"Unknown temporal weight: {name}")
    if (not is_external or renormalize_external_within_passband) and np.nanmax(w) > 0:
        w = w / float(np.nanmax(w))
    return w


def load_psd_groups(psd_path: Path, passband: str | None = None) -> dict[tuple[str, float, str], list[dict[str, Any]]]:
    rows = read_csv_rows(psd_path)
    groups: dict[tuple[str, float, str], dict[str, Any]] = {}
    for row in rows:
        if passband and row.get("passband_label") != passband:
            continue
        family = str(row.get("motion_family", ""))
        scale = fnum(row, "D_scale")
        movie_id = str(row.get("movie_id", ""))
        key = (family, scale, movie_id)
        item = groups.setdefault(
            key,
            {
                "motion_family": family,
                "D_scale": scale,
                "movie_id": movie_id,
                "kind": row.get("kind", ""),
                "passband_label": row.get("passband_label", ""),
                "freq": [],
                "psd": [],
            },
        )
        item["freq"].append(fnum(row, "temporal_frequency_hz"))
        item["psd"].append(fnum(row, "temporal_psd"))
    out: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for (_family, _scale, _movie_id), item in groups.items():
        freq = np.asarray(item.pop("freq"), dtype=np.float64)
        psd = np.asarray(item.pop("psd"), dtype=np.float64)
        order = np.argsort(freq)
        item["freq"] = freq[order]
        item["psd"] = psd[order]
        out[(str(item["motion_family"]), float(item["D_scale"]), str(item.get("passband_label", "")))].append(item)
    return out


def movie_weighted_metrics(
    freq: np.ndarray,
    psd: np.ndarray,
    weight_name: str,
    external_weights: ExternalWeights | None = None,
    renormalize_external_within_passband: bool = False,
) -> dict[str, float]:
    freq = np.asarray(freq, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    positive = np.isfinite(freq) & np.isfinite(psd) & (freq > 0) & (psd > 0)
    freq = freq[positive]
    psd = psd[positive]
    if freq.size == 0:
        return {
            "weighted_flatness": float("nan"),
            "weighted_entropy": float("nan"),
            "weighted_abs_loglog_slope": float("nan"),
            "usable_power_fraction": float("nan"),
            "out_of_weight_power_fraction": float("nan"),
            "total_power": float("nan"),
        }
    w = temporal_weight(
        freq,
        weight_name,
        external_weights=external_weights,
        renormalize_external_within_passband=renormalize_external_within_passband,
    )
    support = w > 1e-6
    weighted = psd * w
    usable = float(np.trapz(weighted, freq))
    total = float(np.trapz(psd, freq))
    frac = usable / total if total > 0 else float("nan")
    return {
        "weighted_flatness": spectral_flatness(weighted[support]),
        "weighted_entropy": spectral_entropy(weighted[support]),
        "weighted_abs_loglog_slope": abs_loglog_slope(freq[support], weighted[support]),
        "usable_power_fraction": frac,
        "out_of_weight_power_fraction": 1.0 - frac if np.isfinite(frac) else float("nan"),
        "total_power": total,
    }


def summarize_weighted(
    groups: dict[tuple[str, float, str], list[dict[str, Any]]],
    weight_names: list[str],
    external_weights: ExternalWeights | None = None,
    renormalize_external_within_passband: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    movie_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (family, scale, passband), movies in sorted(groups.items()):
        for weight_name in weight_names:
            metric_values: dict[str, list[float]] = defaultdict(list)
            for movie in movies:
                vals = movie_weighted_metrics(
                    movie["freq"],
                    movie["psd"],
                    weight_name,
                    external_weights=external_weights,
                    renormalize_external_within_passband=renormalize_external_within_passband,
                )
                row = {
                    "motion_family": family,
                    "D_scale": scale,
                    "movie_id": movie["movie_id"],
                    "kind": movie.get("kind", ""),
                    "passband_label": passband,
                    "weight_name": weight_name,
                    **vals,
                }
                movie_rows.append(row)
                for metric, value in vals.items():
                    metric_values[metric].append(float(value))
            for metric, vals in metric_values.items():
                m, s, n = mean_sem(vals)
                summary_rows.append(
                    {
                        "motion_family": family,
                        "D_scale": scale,
                        "passband_label": passband,
                        "weight_name": weight_name,
                        "metric": metric,
                        "mean": m,
                        "sem": s,
                        "n": n,
                    }
                )
    return movie_rows, summary_rows


def decision_table(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        groups[(str(row["motion_family"]), str(row["passband_label"]), str(row["weight_name"]), str(row["metric"]))].append(row)
    out: list[dict[str, Any]] = []
    maximize = {
        "weighted_flatness": True,
        "weighted_entropy": True,
        "usable_power_fraction": True,
        "total_power": True,
        "weighted_abs_loglog_slope": False,
        "out_of_weight_power_fraction": False,
    }
    for (family, passband, weight_name, metric), rows in sorted(groups.items()):
        if metric not in maximize:
            continue
        scales = np.asarray([fnum(row, "D_scale") for row in rows], dtype=np.float64)
        means = np.asarray([fnum(row, "mean") for row in rows], dtype=np.float64)
        keep = np.isfinite(scales) & np.isfinite(means)
        scales = scales[keep]
        means = means[keep]
        if means.size == 0:
            continue
        idx = int(np.nanargmax(means) if maximize[metric] else np.nanargmin(means))
        bio_idx = int(np.nanargmin(np.abs(scales - BIOLOGICAL_D)))
        opt = float(means[idx])
        bio = float(means[bio_idx])
        out.append(
            {
                "motion_family": family,
                "passband_label": passband,
                "weight_name": weight_name,
                "metric": metric,
                "optimum_rule": "argmax" if maximize[metric] else "argmin",
                "D_opt": float(scales[idx]),
                "value_at_opt": opt,
                "nearest_biological_D": float(scales[bio_idx]),
                "value_at_nearest_biological": bio,
                "biological_fraction_of_peak": bio / opt if maximize[metric] and opt != 0 else float("nan"),
                "distance_from_biological_D1": abs(float(scales[idx]) - BIOLOGICAL_D),
                "n_scales": int(scales.size),
                "boundary_call": "upper_boundary" if np.isclose(scales[idx], np.nanmax(scales)) else ("lower_boundary" if np.isclose(scales[idx], np.nanmin(scales)) else "interior"),
            }
        )
    return out


def write_figures(out_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("weighted_flatness", "weighted_entropy", "usable_power_fraction", "out_of_weight_power_fraction"):
        rows = [row for row in summary_rows if row.get("metric") == metric and row.get("motion_family") == "scaled_measured_drift_D"]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for weight_name in sorted({str(row["weight_name"]) for row in rows}):
            sub = [row for row in rows if row["weight_name"] == weight_name]
            sub = sorted(sub, key=lambda row: fnum(row, "D_scale"))
            ax.plot([fnum(row, "D_scale") for row in sub], [fnum(row, "mean") for row in sub], marker="o", label=weight_name)
        ax.axvline(BIOLOGICAL_D, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("D scale")
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " "))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{metric}.pdf", bbox_inches="tight")
        plt.close(fig)


def write_summary(out_dir: Path, decisions: list[dict[str, Any]], psd_path: Path, external_weight_csv: Path | None = None) -> None:
    rows = [row for row in decisions if row.get("motion_family") == "scaled_measured_drift_D"]
    upper = sum(1 for row in rows if row.get("boundary_call") == "upper_boundary")
    interior = sum(1 for row in rows if row.get("boundary_call") == "interior")
    lower = sum(1 for row in rows if row.get("boundary_call") == "lower_boundary")
    lines = [
        "# V1-Weighted Whitening Summary",
        "",
        f"Source PSD rows: `{psd_path}`",
        "",
        "This pass can combine analytic temporal weights with an optional model-derived temporal transfer function.",
        "",
        "## Decision Overview",
        "",
        f"- scaled measured drift decision rows: {len(rows)}",
        f"- upper-bound optima: {upper}",
        f"- interior optima: {interior}",
        f"- lower-bound optima: {lower}",
        "",
        "## Primary Interpretation",
        "",
        "If weighted metrics remain upper-bound limited, V1-like analytic temporal weighting has not rescued biological scale. If they move near `D=1`, this is evidence that raw whitening was too broad a frequency objective.",
        "",
        "## Guardrails",
        "",
        "- Analytic weights are predeclared proxies.",
        "- External weights are only as valid as their probe stimulus and response-gain estimator.",
        "- External weights are not renormalized inside each PSD passband unless explicitly requested.",
        "- PSD rows must come from the primary spatial passband export; the production full whitening run did not write PSD rows by default.",
        "",
    ]
    if external_weight_csv is not None:
        lines.extend(["## External Weights", "", f"- source: `{external_weight_csv}`", ""])
    primary = [
        row for row in rows
        if row.get("metric") in {"weighted_flatness", "weighted_entropy", "usable_power_fraction", "out_of_weight_power_fraction"}
    ]
    if primary:
        lines.extend(["## Key Optima", ""])
        for row in primary:
            lines.append(
                f"- {row['weight_name']} / {row['metric']}: D_opt={fnum(row, 'D_opt'):.6g}, "
                f"boundary={row['boundary_call']}, D=1 value={fnum(row, 'value_at_nearest_biological'):.6g}"
            )
        lines.append("")
    (out_dir / "v1_weighted_whitening_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    psd_path = Path(args.psd_file) if args.psd_file else in_dir / "retinal_temporal_psd_by_movie.csv"
    if not psd_path.exists() or psd_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"No PSD rows found at {psd_path}. Run run_input_whitening_optimum.py with "
            "--write-psd-rows true for the primary passband first."
        )
    weight_names = [part.strip() for part in str(args.weight_names).split(",") if part.strip()]
    external_weight_csv = Path(args.external_weight_csv) if args.external_weight_csv else None
    external_weights = load_external_weight_specs(external_weight_csv, args.external_weight_name)
    for name in external_weights:
        if name not in weight_names:
            weight_names.append(name)
    groups = load_psd_groups(psd_path, passband=args.passband or None)
    movie_rows, summary_rows = summarize_weighted(
        groups,
        weight_names,
        external_weights=external_weights,
        renormalize_external_within_passband=bool(args.renormalize_external_weight_within_passband),
    )
    decisions = decision_table(summary_rows)
    write_csv_rows(out_dir / "v1_weighted_whitening_movie_metrics.csv", movie_rows)
    write_csv_rows(out_dir / "v1_weighted_whitening_scale_summary.csv", summary_rows)
    write_csv_rows(out_dir / "v1_weighted_whitening_decision_table.csv", decisions)
    write_figures(out_dir, summary_rows)
    write_summary(out_dir, decisions, psd_path, external_weight_csv=external_weight_csv)
    write_json(
        out_dir / "v1_weighted_whitening_manifest.json",
        {
            "analysis": "v1_weighted_whitening",
            "in_dir": in_dir,
            "out_dir": out_dir,
            "psd_path": psd_path,
            "passband": args.passband,
            "weight_names": weight_names,
            "external_weight_csv": external_weight_csv,
            "external_weight_names": sorted(external_weights),
            "renormalize_external_weight_within_passband": bool(args.renormalize_external_weight_within_passband),
            "n_psd_groups": len(groups),
            "n_movie_metric_rows": len(movie_rows),
            "n_scale_summary_rows": len(summary_rows),
            "n_decision_rows": len(decisions),
            "claim_boundary": "Temporal weighting pass; external model-derived weights inherit the probe-stimulus assumptions.",
        },
    )
    print(f"Wrote V1-weighted whitening outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--psd-file", type=Path, default=None)
    parser.add_argument("--passband", default="spatial_4_40cpd__temporal_1_30hz")
    parser.add_argument(
        "--weight-names",
        default="uniform_1_30hz,bandpass_2_20hz,bandpass_4_30hz,soft_v1_log_gaussian_8hz,soft_v1_log_gaussian_12hz,lowpass_30hz",
    )
    parser.add_argument("--external-weight-csv", type=Path, default=None)
    parser.add_argument("--external-weight-name", default="model_response_gain_sq")
    parser.add_argument("--renormalize-external-weight-within-passband", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
