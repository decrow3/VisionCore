#!/usr/bin/env python3
"""BackImage-style line profiles for zero-gap Vernier contour surfaces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D


DEFAULT_SUMMARY = Path(
    "outputs/notebook_vernier_walkthrough/"
    "rr100_single_contour_panel_c_random_ori_blocks4_n20/"
    "rr100_single_contour_panel_c_high_sf_arcmin_binned_n8_from_grid4_random_ori_4blocks_"
    "arcmin_binned_component_surfaces_n8_summary.csv"
)

SF_COLOR = "#D55E00"
RELATION_ORDER = ("contour_matched", "contour_intermediate", "contour_orthogonal")
RELATION_TITLES = {
    "contour_matched": "Aligned high-SF units",
    "contour_intermediate": "Oblique high-SF units",
    "contour_orthogonal": "Orthogonal high-SF units",
}
COMPONENT_SPECS = {
    "across": {
        "label": "across-contour motion component",
        "short_label": "across contour",
        "linestyle": "-",
        "marker": "o",
        "bin_col": "across_bin",
        "bin_label_col": "across_bin_label",
        "median_col": "across_median_arcmin",
        "min_col": "across_min_arcmin",
        "max_col": "across_max_arcmin",
        "scale_col": "across_scale_median",
    },
    "along": {
        "label": "along-contour motion component",
        "short_label": "along contour",
        "linestyle": (0, (4.2, 2.0)),
        "marker": "s",
        "bin_col": "along_bin",
        "bin_label_col": "along_bin_label",
        "median_col": "along_median_arcmin",
        "min_col": "along_min_arcmin",
        "max_col": "along_max_arcmin",
        "scale_col": "along_scale_median",
    },
}
METRIC_LABELS = {
    "path": "path length",
    "rms": "RMS excursion",
}
TICK_CANDIDATES = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.2,
    1.5,
    2.0,
    2.5,
    3.0,
    5.0,
    10.0,
    20.0,
    25.0,
    35.0,
    50.0,
    65.0,
    90.0,
    120.0,
    160.0,
    240.0,
    320.0,
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def _ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _percent_delta(value: float, baseline: float) -> float:
    value = float(value)
    baseline = float(baseline)
    if not (math.isfinite(value) and math.isfinite(baseline)) or baseline == 0.0:
        return float("nan")
    return 100.0 * (value - baseline) / baseline


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if not np.any(ok):
        return float("nan")
    x = x[ok]
    w = w[ok]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cutoff = 0.5 * float(np.sum(w))
    return float(x[np.searchsorted(np.cumsum(w), cutoff, side="left")])


def _first_finite(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr[0]) if arr.size else float("nan")


def _collapse_one_component(surface: pd.DataFrame, *, component: str) -> pd.DataFrame:
    spec = COMPONENT_SPECS[component]
    rows: list[dict[str, Any]] = []
    bin_col = str(spec["bin_col"])
    median_col = str(spec["median_col"])
    min_col = str(spec["min_col"])
    max_col = str(spec["max_col"])
    scale_col = str(spec["scale_col"])
    for relation in RELATION_ORDER:
        rel = surface[surface["relation"].astype(str).eq(relation)].copy()
        if rel.empty:
            continue
        base_ssi = _first_finite(rel["static_population_ssi_bits_per_spike"])
        base_info = _first_finite(rel["static_information_bits_per_sample"])
        base_spikes = _first_finite(rel["static_expected_spikes_per_sample"])
        relation_title = str(rel["relation_title"].iloc[0]) if "relation_title" in rel else RELATION_TITLES[relation]
        relation_label = str(rel["relation_label"].iloc[0]) if "relation_label" in rel else relation
        for bin_id, group in rel.groupby(bin_col, sort=True):
            group = group.copy()
            info_num = float(pd.to_numeric(group["information_numerator_bits"], errors="coerce").sum())
            expected_spikes = float(pd.to_numeric(group["expected_spikes"], errors="coerce").sum())
            unit_trace_samples = int(pd.to_numeric(group["n_unit_trace_samples"], errors="coerce").fillna(0).sum())
            selected_trace_samples = int(
                pd.to_numeric(group["n_selected_trace_samples"], errors="coerce").fillna(0).sum()
            )
            trace_condition_samples = int(
                pd.to_numeric(group["n_trace_condition_samples"], errors="coerce").fillna(0).sum()
            )
            population_ssi = _ratio(info_num, expected_spikes)
            information_per_sample = _ratio(info_num, unit_trace_samples)
            spikes_per_sample = _ratio(expected_spikes, unit_trace_samples)
            rows.append(
                {
                    "metric_family": str(group["metric_family"].iloc[0]),
                    "metric_family_title": str(group["metric_family_title"].iloc[0]),
                    "relation": relation,
                    "relation_label": relation_label,
                    "relation_title": relation_title,
                    "component": component,
                    "component_label": str(spec["label"]),
                    "component_short_label": str(spec["short_label"]),
                    "component_bin": int(bin_id),
                    "component_bin_label": str(group[str(spec["bin_label_col"])].iloc[0]),
                    "component_min_arcmin": float(pd.to_numeric(group[min_col], errors="coerce").min()),
                    "component_max_arcmin": float(pd.to_numeric(group[max_col], errors="coerce").max()),
                    "component_median_arcmin": _weighted_median(
                        group[median_col],
                        group["n_trace_condition_samples"],
                    ),
                    "component_scale_median": _weighted_median(
                        group[scale_col],
                        group["n_trace_condition_samples"],
                    ),
                    "n_grid_cells": int(group.shape[0]),
                    "n_trace_condition_samples": trace_condition_samples,
                    "n_selected_trace_samples": selected_trace_samples,
                    "n_unit_trace_samples": unit_trace_samples,
                    "information_numerator_bits": info_num,
                    "expected_spikes": expected_spikes,
                    "population_ssi_bits_per_spike": population_ssi,
                    "population_ssi_percent_vs_static": _percent_delta(population_ssi, base_ssi),
                    "information_bits_per_sample": information_per_sample,
                    "information_bits_per_sample_percent_vs_static": _percent_delta(
                        information_per_sample,
                        base_info,
                    ),
                    "expected_spikes_per_sample": spikes_per_sample,
                    "expected_spikes_per_sample_percent_vs_static": _percent_delta(
                        spikes_per_sample,
                        base_spikes,
                    ),
                    "static_population_ssi_bits_per_spike": base_ssi,
                    "static_information_bits_per_sample": base_info,
                    "static_expected_spikes_per_sample": base_spikes,
                    "collapse_note": (
                        "Line points pool surface cells by summing information numerator and expected spikes "
                        "before computing spike-weighted population SSI."
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_line_profiles(surface: pd.DataFrame, *, metric_family: str) -> pd.DataFrame:
    metric = surface[surface["metric_family"].astype(str).eq(metric_family)].copy()
    if metric.empty:
        raise ValueError(f"No rows found for metric_family={metric_family!r}")
    return pd.concat(
        [
            _collapse_one_component(metric, component="across"),
            _collapse_one_component(metric, component="along"),
        ],
        ignore_index=True,
    )


def _find_reference_surface(summary_csv: Path) -> Path | None:
    candidates = sorted(summary_csv.parent.glob("*component_surface_grid*_component_surfaces_summary.csv"))
    return candidates[0] if candidates else None


def load_native_references(path: Path | None, *, metric_family: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    surface = pd.read_csv(path)
    required = {
        "metric_family",
        "relation",
        "across_scale",
        "along_scale",
        "across_median_arcmin",
        "along_median_arcmin",
    }
    if not required.issubset(surface.columns):
        return pd.DataFrame()
    work = surface[surface["metric_family"].astype(str).eq(metric_family)].copy()
    work["across_scale"] = pd.to_numeric(work["across_scale"], errors="coerce")
    work["along_scale"] = pd.to_numeric(work["along_scale"], errors="coerce")
    native = work[np.isclose(work["across_scale"], 1.0) & np.isclose(work["along_scale"], 1.0)].copy()
    rows: list[dict[str, Any]] = []
    for relation in RELATION_ORDER:
        rel = native[native["relation"].astype(str).eq(relation)]
        if rel.empty:
            continue
        first = rel.iloc[0]
        for component, col in [("across", "across_median_arcmin"), ("along", "along_median_arcmin")]:
            rows.append(
                {
                    "metric_family": metric_family,
                    "relation": relation,
                    "component": component,
                    "reference_label": "native_1x_randomized_trace_projection",
                    "reference_median_arcmin": float(first[col]),
                    "reference_surface_csv": str(path),
                }
            )
    return pd.DataFrame(rows)


def _x_broken_log(values: np.ndarray | pd.Series | list[float], *, min_pos: float, max_pos: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    mapped = np.zeros_like(x, dtype=np.float64)
    positive = x > 0.0
    if max_pos <= min_pos:
        max_pos = min_pos * 2.0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(x[positive] / min_pos) / np.log(max_pos / min_pos)
    return mapped


def _tick_label(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def _axis_bounds(line_summary: pd.DataFrame, refs: pd.DataFrame) -> tuple[float, float, list[float]]:
    values = pd.to_numeric(line_summary["component_median_arcmin"], errors="coerce").to_numpy(dtype=np.float64)
    if not refs.empty:
        ref_values = pd.to_numeric(refs["reference_median_arcmin"], errors="coerce").to_numpy(dtype=np.float64)
        values = np.concatenate([values, ref_values])
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0, 2.0, [0.0, 1.0, 2.0]
    min_pos = float(np.nanmin(values))
    max_pos = float(np.nanmax(values))
    if max_pos <= min_pos:
        max_pos = min_pos * 2.0
    min_pos *= 0.94
    max_pos *= 1.04
    ticks = [
        tick
        for tick in TICK_CANDIDATES
        if tick == 0.0 or (tick >= min_pos * 0.98 and tick <= max_pos * 1.02)
    ]
    if len(ticks) < 4:
        ticks = [0.0] + [
            float(np.exp(val))
            for val in np.linspace(math.log(min_pos), math.log(max_pos), num=4)
        ]
    ticks = _thin_ticks_for_broken_log(ticks, min_pos=min_pos, max_pos=max_pos)
    return min_pos, max_pos, ticks


def _thin_ticks_for_broken_log(ticks: list[float], *, min_pos: float, max_pos: float) -> list[float]:
    positives = [float(tick) for tick in ticks if tick > 0.0]
    if not positives:
        return [0.0]
    kept = [0.0]
    last_pos = -float("inf")
    for tick in positives:
        mapped = float(_x_broken_log([tick], min_pos=min_pos, max_pos=max_pos)[0])
        if mapped - last_pos >= 0.62:
            kept.append(float(tick))
            last_pos = mapped
    if kept[-1] != positives[-1]:
        last_tick = float(positives[-1])
        last_mapped = float(_x_broken_log([last_tick], min_pos=min_pos, max_pos=max_pos)[0])
        kept_mapped = float(_x_broken_log([kept[-1]], min_pos=min_pos, max_pos=max_pos)[0])
        if last_mapped - kept_mapped >= 0.44:
            kept.append(last_tick)
        else:
            kept[-1] = last_tick
    return kept


def _format_broken_log_axis(
    ax: plt.Axes,
    *,
    ticks: list[float],
    min_pos: float,
    max_pos: float,
    xlabel: str,
) -> None:
    right = float(_x_broken_log([max(max(ticks), max_pos)], min_pos=min_pos, max_pos=max_pos)[0])
    ax.set_xlim(-0.12, right + 0.25)
    ax.set_xticks(_x_broken_log(ticks, min_pos=min_pos, max_pos=max_pos))
    ax.set_xticklabels([_tick_label(tick) for tick in ticks])
    ax.set_xlabel(xlabel)
    ax.text(
        0.52,
        -0.075,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        rotation=-20,
        clip_on=False,
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.8)


def _draw_native_ticks(
    ax: plt.Axes,
    refs: pd.DataFrame,
    *,
    relation: str,
    min_pos: float,
    max_pos: float,
) -> None:
    rel = refs[refs["relation"].astype(str).eq(relation)] if not refs.empty else pd.DataFrame()
    if rel.empty:
        return
    strip_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for row in rel.itertuples(index=False):
        component = str(getattr(row, "component"))
        spec = COMPONENT_SPECS[component]
        x_val = float(getattr(row, "reference_median_arcmin"))
        if not math.isfinite(x_val) or x_val <= 0.0:
            continue
        x_plot = float(_x_broken_log([x_val], min_pos=min_pos, max_pos=max_pos)[0])
        ax.plot(
            [x_plot, x_plot],
            [-0.018, 0.065],
            transform=strip_transform,
            color="0.22",
            alpha=0.80,
            linestyle=spec["linestyle"],
            linewidth=1.2,
            clip_on=False,
            zorder=6,
        )


def _plot_component_line(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    component: str,
    min_pos: float,
    max_pos: float,
) -> None:
    spec = COMPONENT_SPECS[component]
    plot_rows = rows.sort_values("component_median_arcmin")
    x = _x_broken_log(plot_rows["component_median_arcmin"], min_pos=min_pos, max_pos=max_pos)
    y = pd.to_numeric(plot_rows["population_ssi_percent_vs_static"], errors="coerce").to_numpy(dtype=np.float64)
    ax.plot(
        x,
        y,
        color=SF_COLOR,
        linestyle=spec["linestyle"],
        linewidth=2.1,
        label=str(spec["label"]),
        zorder=3,
    )
    ax.errorbar(
        x,
        y,
        color=SF_COLOR,
        linestyle="none",
        marker=str(spec["marker"]),
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.25,
        linewidth=1.5,
        elinewidth=1.1,
        capsize=0.0,
        zorder=4,
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker=str(spec["marker"]),
        s=30,
        facecolors="white",
        edgecolors=SF_COLOR,
        linewidths=1.35,
        zorder=5,
    )


def plot_line_profiles(
    line_summary: pd.DataFrame,
    *,
    native_refs: pd.DataFrame,
    metric_family: str,
    out_dir: Path,
    out_stem: str,
    dpi: int,
) -> tuple[Path, Path]:
    min_pos, max_pos, ticks = _axis_bounds(line_summary, native_refs)
    y_values = pd.to_numeric(line_summary["population_ssi_percent_vs_static"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite_y = y_values[np.isfinite(y_values)]
    if finite_y.size:
        low = min(0.0, float(np.min(finite_y)))
        high = max(0.0, float(np.max(finite_y)))
        span = max(high - low, 1.0)
        ylim = (low - 0.14 * span, high + 0.16 * span)
    else:
        ylim = (-1.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.95), dpi=int(dpi), sharey=True)
    metric_label = METRIC_LABELS.get(metric_family, metric_family.replace("_", " "))
    for idx, relation in enumerate(RELATION_ORDER):
        ax = axes[idx]
        rel = line_summary[line_summary["relation"].astype(str).eq(relation)].copy()
        ax.axhline(0, color="0.35", lw=0.9, ls=":")
        for component in ("across", "along"):
            rows = rel[rel["component"].astype(str).eq(component)]
            if not rows.empty:
                _plot_component_line(ax, rows, component=component, min_pos=min_pos, max_pos=max_pos)
        _draw_native_ticks(ax, native_refs, relation=relation, min_pos=min_pos, max_pos=max_pos)
        _format_broken_log_axis(
            ax,
            ticks=ticks,
            min_pos=min_pos,
            max_pos=max_pos,
            xlabel="",
        )
        _style_axis(ax)
        ax.set_ylim(*ylim)
        ax.set_title(RELATION_TITLES.get(relation, relation.replace("_", " ")), fontsize=11.0, pad=7)
        if idx == 0:
            ax.set_ylabel("SSI change vs static center (%)")
        else:
            ax.set_ylabel("")
    handles = [
        Line2D(
            [0],
            [0],
            color=SF_COLOR,
            linestyle=COMPONENT_SPECS["across"]["linestyle"],
            marker=COMPONENT_SPECS["across"]["marker"],
            markerfacecolor="white",
            markeredgewidth=1.25,
            linewidth=2.1,
            label=COMPONENT_SPECS["across"]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=SF_COLOR,
            linestyle=COMPONENT_SPECS["along"]["linestyle"],
            marker=COMPONENT_SPECS["along"]["marker"],
            markerfacecolor="white",
            markeredgewidth=1.25,
            linewidth=2.1,
            label=COMPONENT_SPECS["along"]["label"],
        ),
    ]
    if not native_refs.empty:
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.22",
                linestyle="-",
                linewidth=1.2,
                label="x-axis ticks: native 1x randomized trace projection",
            )
        )
    fig.suptitle(
        f"Zero-gap Vernier contour: high-SF {metric_label} line profiles",
        fontsize=14.0,
        y=0.985,
    )
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=7.9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.918),
        ncol=len(handles),
        columnspacing=1.5,
        handlelength=2.8,
    )
    fig.supxlabel(
        f"{metric_label} bin median (arcmin; log scale after break)",
        fontsize=10.0,
        y=0.071,
    )
    fig.text(
        0.5,
        0.015,
        "Points marginalize the measured-arcmin surface cells by pooling information numerator and expected spikes first; "
        "the full-trajectory trace is intentionally omitted.",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="0.30",
    )
    fig.tight_layout(rect=(0.02, 0.13, 0.995, 0.875), w_pad=1.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{out_stem}.png"
    pdf = out_dir / f"{out_stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metric-family", choices=["path", "rms", "all"], default="path")
    parser.add_argument("--reference-surface-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-stem", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=230)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    surface = pd.read_csv(summary_csv)
    metric_families = (
        sorted(str(value) for value in surface["metric_family"].dropna().unique())
        if str(args.metric_family) == "all"
        else [str(args.metric_family)]
    )
    reference_surface = Path(args.reference_surface_csv) if args.reference_surface_csv else _find_reference_surface(summary_csv)
    out_dir = Path(args.out_dir) if args.out_dir else summary_csv.parent
    outputs: dict[str, Any] = {
        "summary_csv": summary_csv,
        "reference_surface_csv": reference_surface,
        "metric_families": metric_families,
        "line_summaries": {},
        "figures": {},
    }
    for metric_family in metric_families:
        line_summary = summarize_line_profiles(surface, metric_family=metric_family)
        native_refs = load_native_references(reference_surface, metric_family=metric_family)
        base_stem = args.out_stem or summary_csv.stem.replace("_summary", "")
        out_stem = f"{base_stem}_story_style_{metric_family}_lines"
        line_csv = out_dir / f"{out_stem}_summary.csv"
        line_summary.to_csv(line_csv, index=False)
        png, pdf = plot_line_profiles(
            line_summary,
            native_refs=native_refs,
            metric_family=metric_family,
            out_dir=out_dir,
            out_stem=out_stem,
            dpi=int(args.dpi),
        )
        outputs["line_summaries"][metric_family] = line_csv
        outputs["figures"][metric_family] = {"png": png, "pdf": pdf}
        print(line_csv)
        print(png)
        print(pdf)
    manifest = out_dir / f"{summary_csv.stem.replace('_summary', '')}_story_style_line_profiles_manifest.json"
    _write_json(manifest, outputs)
    print(manifest)


if __name__ == "__main__":
    main()
