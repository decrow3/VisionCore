#!/usr/bin/env python3
"""Contributor-dominance checks for BackImage RR100 instantaneous SSI.

This script is cache-backed. It reads the all-unit instantaneous SSI tables and
asks whether the 1x-to-3x spike-weighted population effect is carried by a small
set of high-leverage units.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_latest_v1"
)
DEFAULT_AXIS1_SSI_CSV = DEFAULT_RUN_DIR / "displayed_movie_instantaneous_ssi_all_units.csv"
DEFAULT_AXIS0_SSI_CSV = (
    DEFAULT_RUN_DIR
    / "population_ssi_summary"
    / "orientation_group_spike_weighted"
    / "opposing_axis0_displayed_movie_instantaneous_ssi_all_units_recomputed.csv"
)
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "population_ssi_summary" / "contributor_dominance"
DEFAULT_ORIENTATION_GROUPS_CSV = DEFAULT_RUN_DIR / "orientation_tuning_groups.csv"

VALUE_COL = "displayed_movie_time_resolved_ssi_bits_per_spike"
RATE_COL = "displayed_movie_mean_rate"
SPIKES_COL = "displayed_movie_expected_spikes_arbitrary_dt"
INFO_COL = "information_numerator_bits_arbitrary_dt"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis1-ssi-csv", type=Path, default=DEFAULT_AXIS1_SSI_CSV)
    parser.add_argument("--axis0-ssi-csv", type=Path, default=DEFAULT_AXIS0_SSI_CSV)
    parser.add_argument("--orientation-groups-csv", type=Path, default=DEFAULT_ORIENTATION_GROUPS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reference-scale", type=float, default=1.0)
    parser.add_argument("--endpoint-scale", type=float, default=3.0)
    parser.add_argument("--top-k", type=str, default="1,3,5,10,20")
    parser.add_argument("--cap-quantiles", type=str, default="0.9,0.95,0.99")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated integer.")
    return values


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def required_columns() -> set[str]:
    return {
        "unit_index",
        "unit_label",
        "axis_mode",
        "display_scale",
        "condition_index",
        "condition_id",
        "along_scale",
        "across_scale",
        VALUE_COL,
        RATE_COL,
        SPIKES_COL,
    }


def load_ssi_table(path: Path, panel_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing all-unit SSI table: {path}")
    df = pd.read_csv(path)
    missing = sorted(required_columns().difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    for col in (
        "unit_index",
        "display_scale",
        "condition_index",
        "along_scale",
        "across_scale",
        VALUE_COL,
        RATE_COL,
        SPIKES_COL,
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["unit_index"] = df["unit_index"].astype(int)
    df["unit_label"] = df["unit_label"].astype(str)
    df["axis_mode"] = df["axis_mode"].astype(str)
    df["condition_id"] = df["condition_id"].astype(str)
    df["panel"] = panel_label
    df[INFO_COL] = df[VALUE_COL].astype(float) * df[SPIKES_COL].astype(float)
    return df


def load_orientation_groups(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    groups = pd.read_csv(path)
    if "unit_index" not in groups.columns:
        return None
    groups["unit_index"] = pd.to_numeric(groups["unit_index"], errors="coerce").astype("Int64")
    groups = groups.dropna(subset=["unit_index"]).copy()
    groups["unit_index"] = groups["unit_index"].astype(int)
    wanted = [
        col
        for col in (
            "unit_index",
            "orientation_group",
            "orientation_group_label",
            "preferred_orientation_deg",
            "orientation_selectivity_index",
            "preferred_delta_from_contour_deg",
            "preferred_delta_from_across_deg",
        )
        if col in groups.columns
    ]
    return groups[wanted].drop_duplicates("unit_index")


def add_orientation_groups(df: pd.DataFrame, groups: pd.DataFrame | None) -> pd.DataFrame:
    if groups is None:
        return df
    return df.merge(groups, on="unit_index", how="left", validate="many_to_one")


def finite_sum(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nansum(arr[np.isfinite(arr)]))


def population_bits_per_spike(
    sub: pd.DataFrame,
    *,
    exclude_units: set[int] | None = None,
    weight_cap_quantile: float | None = None,
    value_cap_quantile: float | None = None,
    equal_weight: bool = False,
    median: bool = False,
) -> float:
    if exclude_units:
        sub = sub[~sub["unit_index"].isin(exclude_units)]
    y = sub[VALUE_COL].to_numpy(dtype=np.float64)
    w = sub[SPIKES_COL].to_numpy(dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(w) & (w >= 0.0)
    y = y[valid]
    w = w[valid]
    if y.size == 0:
        return float("nan")
    if median:
        return float(np.nanmedian(y))
    if equal_weight:
        return float(np.nanmean(y))
    if value_cap_quantile is not None:
        cap = float(np.nanquantile(y, float(value_cap_quantile)))
        y = np.minimum(y, cap)
    if weight_cap_quantile is not None:
        cap = float(np.nanquantile(w, float(weight_cap_quantile)))
        w = np.minimum(w, cap)
    return float(np.nansum(y * w) / max(float(np.nansum(w)), EPS))


def condition_sub(df: pd.DataFrame, panel: str, axis_mode: str, scale: float) -> pd.DataFrame:
    sub = df[
        (df["panel"] == panel)
        & (df["axis_mode"] == axis_mode)
        & np.isclose(df["display_scale"].astype(float), float(scale))
    ].copy()
    if sub.empty:
        raise ValueError(f"No rows for panel={panel}, axis_mode={axis_mode}, scale={scale:g}")
    return sub


def merged_reference_endpoint(df: pd.DataFrame, panel: str, axis_mode: str, reference_scale: float, endpoint_scale: float) -> pd.DataFrame:
    ref = condition_sub(df, panel, axis_mode, reference_scale)
    end = condition_sub(df, panel, axis_mode, endpoint_scale)
    cols = [
        "unit_index",
        "unit_label",
        VALUE_COL,
        RATE_COL,
        SPIKES_COL,
        INFO_COL,
    ]
    extra_cols = [
        col
        for col in (
            "orientation_group",
            "orientation_group_label",
            "preferred_orientation_deg",
            "orientation_selectivity_index",
            "preferred_delta_from_contour_deg",
            "preferred_delta_from_across_deg",
        )
        if col in df.columns
    ]
    pair = end[cols + extra_cols].merge(
        ref[cols],
        on=["unit_index", "unit_label"],
        how="inner",
        suffixes=("_endpoint", "_reference"),
        validate="one_to_one",
    )
    if pair.empty:
        raise ValueError(f"No paired units for panel={panel}, axis_mode={axis_mode}.")
    return pair


def compute_unit_contributions(
    df: pd.DataFrame,
    *,
    panel: str,
    axis_mode: str,
    reference_scale: float,
    endpoint_scale: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pair = merged_reference_endpoint(df, panel, axis_mode, reference_scale, endpoint_scale)
    n1 = pair[f"{INFO_COL}_reference"].to_numpy(dtype=np.float64)
    n3 = pair[f"{INFO_COL}_endpoint"].to_numpy(dtype=np.float64)
    w1 = pair[f"{SPIKES_COL}_reference"].to_numpy(dtype=np.float64)
    w3 = pair[f"{SPIKES_COL}_endpoint"].to_numpy(dtype=np.float64)
    y1 = pair[f"{VALUE_COL}_reference"].to_numpy(dtype=np.float64)
    y3 = pair[f"{VALUE_COL}_endpoint"].to_numpy(dtype=np.float64)

    valid = np.isfinite(n1) & np.isfinite(n3) & np.isfinite(w1) & np.isfinite(w3) & (w1 >= 0.0) & (w3 >= 0.0)
    pair = pair.loc[valid].copy()
    n1 = n1[valid]
    n3 = n3[valid]
    w1 = w1[valid]
    w3 = w3[valid]
    y1 = y1[valid]
    y3 = y3[valid]

    total_n1 = float(np.nansum(n1))
    total_n3 = float(np.nansum(n3))
    total_w1 = float(np.nansum(w1))
    total_w3 = float(np.nansum(w3))
    pop1 = total_n1 / max(total_w1, EPS)
    pop3 = total_n3 / max(total_w3, EPS)
    delta = pop3 - pop1

    additive = (n3 / max(total_w3, EPS)) - (n1 / max(total_w1, EPS))
    delta_without = np.empty_like(additive)
    for idx in range(additive.size):
        without_3 = (total_n3 - n3[idx]) / max(total_w3 - w3[idx], EPS)
        without_1 = (total_n1 - n1[idx]) / max(total_w1 - w1[idx], EPS)
        delta_without[idx] = without_3 - without_1
    loo_leverage = delta - delta_without

    pair["panel"] = panel
    pair["axis_mode"] = axis_mode
    pair["reference_scale"] = float(reference_scale)
    pair["endpoint_scale"] = float(endpoint_scale)
    pair["unit_bits_per_spike_reference"] = y1
    pair["unit_bits_per_spike_endpoint"] = y3
    pair["unit_delta_bits_per_spike"] = y3 - y1
    pair["unit_expected_spikes_reference"] = w1
    pair["unit_expected_spikes_endpoint"] = w3
    pair["unit_spike_weight_reference"] = w1 / max(total_w1, EPS)
    pair["unit_spike_weight_endpoint"] = w3 / max(total_w3, EPS)
    pair["unit_information_bits_reference"] = n1
    pair["unit_information_bits_endpoint"] = n3
    pair["population_delta_additive_contribution"] = additive
    pair["population_delta_leave_one_out_leverage"] = loo_leverage
    pair["population_delta_without_unit"] = delta_without
    pair["share_of_population_delta_additive"] = additive / max(delta, EPS) if delta >= 0.0 else np.nan
    pair["share_of_population_delta_loo"] = loo_leverage / max(delta, EPS) if delta >= 0.0 else np.nan
    pair["positive_additive_rank"] = (
        pair["population_delta_additive_contribution"].rank(method="first", ascending=False).astype(int)
    )
    pair["positive_loo_rank"] = (
        pair["population_delta_leave_one_out_leverage"].rank(method="first", ascending=False).astype(int)
    )
    pair["endpoint_spike_weight_rank"] = pair["unit_spike_weight_endpoint"].rank(method="first", ascending=False).astype(int)

    totals = {
        "population_bits_per_spike_reference": pop1,
        "population_bits_per_spike_endpoint": pop3,
        "population_delta_endpoint_minus_reference": delta,
        "population_information_bits_reference": total_n1,
        "population_information_bits_endpoint": total_n3,
        "population_expected_spikes_reference": total_w1,
        "population_expected_spikes_endpoint": total_w3,
    }
    return pair, totals


def recompute_delta(
    df: pd.DataFrame,
    *,
    panel: str,
    axis_mode: str,
    reference_scale: float,
    endpoint_scale: float,
    exclude_units: set[int] | None = None,
    weight_cap_quantile: float | None = None,
    value_cap_quantile: float | None = None,
    equal_weight: bool = False,
    median: bool = False,
) -> float:
    ref = condition_sub(df, panel, axis_mode, reference_scale)
    end = condition_sub(df, panel, axis_mode, endpoint_scale)
    ref_pop = population_bits_per_spike(
        ref,
        exclude_units=exclude_units,
        weight_cap_quantile=weight_cap_quantile,
        value_cap_quantile=value_cap_quantile,
        equal_weight=equal_weight,
        median=median,
    )
    end_pop = population_bits_per_spike(
        end,
        exclude_units=exclude_units,
        weight_cap_quantile=weight_cap_quantile,
        value_cap_quantile=value_cap_quantile,
        equal_weight=equal_weight,
        median=median,
    )
    return end_pop - ref_pop


def n_to_reach_fraction(sorted_positive: np.ndarray, target: float, delta: float) -> int | None:
    if sorted_positive.size == 0 or delta <= 0.0:
        return None
    threshold = float(target) * delta
    cumulative = np.cumsum(sorted_positive)
    idx = np.flatnonzero(cumulative >= threshold)
    return int(idx[0] + 1) if idx.size else None


def build_summary_rows(
    df: pd.DataFrame,
    unit_contrib: pd.DataFrame,
    totals: dict[tuple[str, str], dict[str, float]],
    *,
    reference_scale: float,
    endpoint_scale: float,
    top_k: list[int],
    cap_quantiles: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (panel, axis_mode), sub in unit_contrib.groupby(["panel", "axis_mode"], sort=True):
        total = totals[(panel, axis_mode)]
        delta = float(total["population_delta_endpoint_minus_reference"])
        contrib = sub["population_delta_additive_contribution"].to_numpy(dtype=np.float64)
        positive = np.sort(contrib[contrib > 0.0])[::-1]
        negative_sum = float(np.nansum(contrib[contrib < 0.0]))
        positive_sum = float(np.nansum(positive))
        positive_weights = positive / max(positive_sum, EPS) if positive_sum > 0 else positive
        hhi = float(np.nansum(positive_weights**2)) if positive_sum > 0 else float("nan")
        unit_delta = sub["unit_delta_bits_per_spike"].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            "panel": panel,
            "axis_mode": axis_mode,
            "reference_scale": float(reference_scale),
            "endpoint_scale": float(endpoint_scale),
            "n_units": int(sub["unit_index"].nunique()),
            **total,
            "equal_weight_delta": recompute_delta(
                df,
                panel=panel,
                axis_mode=axis_mode,
                reference_scale=reference_scale,
                endpoint_scale=endpoint_scale,
                equal_weight=True,
            ),
            "median_unit_delta": float(np.nanmedian(unit_delta[np.isfinite(unit_delta)])),
            "fraction_units_positive_delta": float(np.nanmean(unit_delta[np.isfinite(unit_delta)] > 0.0)),
            "positive_additive_contribution_sum": positive_sum,
            "negative_additive_contribution_sum": negative_sum,
            "positive_contribution_hhi": hhi,
            "positive_contribution_effective_n": 1.0 / hhi if hhi > 0.0 else float("nan"),
            "n_positive_contributors_to_reach_50pct_net_delta": n_to_reach_fraction(positive, 0.5, delta),
            "n_positive_contributors_to_reach_80pct_net_delta": n_to_reach_fraction(positive, 0.8, delta),
        }
        for k in top_k:
            top_units = set(
                sub.sort_values("population_delta_additive_contribution", ascending=False)
                .head(k)["unit_index"]
                .astype(int)
                .tolist()
            )
            top_sum = float(np.nansum(positive[: min(k, positive.size)]))
            row[f"top{k}_positive_additive_contribution_sum"] = top_sum
            row[f"top{k}_positive_additive_share_of_net_delta"] = top_sum / max(delta, EPS) if delta > 0 else float("nan")
            row[f"top{k}_positive_additive_share_of_positive_mass"] = (
                top_sum / max(positive_sum, EPS) if positive_sum > 0 else float("nan")
            )
            row[f"delta_after_dropping_top{k}_positive_contributors"] = recompute_delta(
                df,
                panel=panel,
                axis_mode=axis_mode,
                reference_scale=reference_scale,
                endpoint_scale=endpoint_scale,
                exclude_units=top_units,
            )
        for k in top_k:
            spike_units = set(
                sub.sort_values("unit_expected_spikes_endpoint", ascending=False)
                .head(k)["unit_index"]
                .astype(int)
                .tolist()
            )
            row[f"delta_after_dropping_top{k}_endpoint_spike_weight_units"] = recompute_delta(
                df,
                panel=panel,
                axis_mode=axis_mode,
                reference_scale=reference_scale,
                endpoint_scale=endpoint_scale,
                exclude_units=spike_units,
            )
        for q in cap_quantiles:
            suffix = f"p{int(round(q * 100)):02d}"
            row[f"delta_with_spike_weight_cap_{suffix}"] = recompute_delta(
                df,
                panel=panel,
                axis_mode=axis_mode,
                reference_scale=reference_scale,
                endpoint_scale=endpoint_scale,
                weight_cap_quantile=q,
            )
            row[f"delta_with_unit_ssi_cap_{suffix}"] = recompute_delta(
                df,
                panel=panel,
                axis_mode=axis_mode,
                reference_scale=reference_scale,
                endpoint_scale=endpoint_scale,
                value_cap_quantile=q,
            )
        top_labels = (
            sub.sort_values("population_delta_additive_contribution", ascending=False)
            .head(10)["unit_label"]
            .astype(str)
            .tolist()
        )
        row["top10_positive_contributor_unit_labels"] = ";".join(top_labels)
        rows.append(row)
    return rows


def build_curve_rows(
    df: pd.DataFrame,
    unit_contrib: pd.DataFrame,
    *,
    cap_quantiles: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    panels = sorted(df["panel"].dropna().unique())
    axis_modes = sorted(df["axis_mode"].dropna().unique())
    metrics: list[tuple[str, dict[str, Any]]] = [
        ("spike_weighted_all_units", {}),
        ("equal_unit_mean", {"equal_weight": True}),
        ("median_unit", {"median": True}),
    ]
    for q in cap_quantiles:
        suffix = f"p{int(round(q * 100)):02d}"
        metrics.append((f"spike_weight_cap_{suffix}", {"weight_cap_quantile": q}))
        metrics.append((f"unit_ssi_cap_{suffix}", {"value_cap_quantile": q}))

    top_sets: dict[tuple[str, str, str], set[int]] = {}
    for (panel, axis_mode), sub in unit_contrib.groupby(["panel", "axis_mode"], sort=True):
        ranked = sub.sort_values("population_delta_additive_contribution", ascending=False)
        for k in (1, 5, 10):
            top_sets[(panel, axis_mode, f"drop_top{k}_positive_contributors")] = set(
                ranked.head(k)["unit_index"].astype(int).tolist()
            )
            metrics.append((f"drop_top{k}_positive_contributors", {"exclude_lookup": f"drop_top{k}_positive_contributors"}))
        ranked_spikes = sub.sort_values("unit_expected_spikes_endpoint", ascending=False)
        for k in (5, 10):
            top_sets[(panel, axis_mode, f"drop_top{k}_endpoint_spike_weight_units")] = set(
                ranked_spikes.head(k)["unit_index"].astype(int).tolist()
            )
            metrics.append((f"drop_top{k}_endpoint_spike_weight_units", {"exclude_lookup": f"drop_top{k}_endpoint_spike_weight_units"}))

    deduped_metrics: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for name, kwargs in metrics:
        if name in seen:
            continue
        seen.add(name)
        deduped_metrics.append((name, kwargs))

    for panel in panels:
        for axis_mode in axis_modes:
            scales = sorted(
                df.loc[(df["panel"] == panel) & (df["axis_mode"] == axis_mode), "display_scale"]
                .dropna()
                .astype(float)
                .unique()
                .tolist()
            )
            for scale in scales:
                sub = condition_sub(df, panel, axis_mode, scale)
                for metric, kwargs in deduped_metrics:
                    call_kwargs = dict(kwargs)
                    exclude_lookup = call_kwargs.pop("exclude_lookup", None)
                    exclude_units = top_sets.get((panel, axis_mode, exclude_lookup), set()) if exclude_lookup else None
                    rows.append(
                        {
                            "panel": panel,
                            "axis_mode": axis_mode,
                            "display_scale": float(scale),
                            "metric": metric,
                            "population_bits_per_spike": population_bits_per_spike(
                                sub,
                                exclude_units=exclude_units,
                                **call_kwargs,
                            ),
                            "n_excluded_units": len(exclude_units) if exclude_units else 0,
                        }
                    )
    return rows


def plot_robust_curves(curves: pd.DataFrame, out_dir: Path, *, dpi: int) -> tuple[Path, Path]:
    plot_metrics = [
        ("spike_weighted_all_units", "all units", "black", 2.4),
        ("equal_unit_mean", "equal unit mean", "0.45", 1.7),
        ("spike_weight_cap_p95", "cap spike weights p95", "#1f77b4", 1.8),
        ("unit_ssi_cap_p95", "cap unit SSI p95", "#9467bd", 1.8),
        ("drop_top5_positive_contributors", "drop top 5 contributors", "#d62728", 1.8),
        ("drop_top10_positive_contributors", "drop top 10 contributors", "#ff7f0e", 1.8),
    ]
    panels = ["opposing axis fixed at 1x", "opposing axis fixed at 0x"]
    axis_modes = ["across_sweep", "along_sweep"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True, constrained_layout=True)
    for row_idx, panel in enumerate(panels):
        for col_idx, axis_mode in enumerate(axis_modes):
            ax = axes[row_idx, col_idx]
            sub = curves[(curves["panel"] == panel) & (curves["axis_mode"] == axis_mode)]
            for metric, label, color, lw in plot_metrics:
                msub = sub[sub["metric"] == metric].sort_values("display_scale")
                if msub.empty:
                    continue
                ax.plot(
                    msub["display_scale"].to_numpy(dtype=float),
                    msub["population_bits_per_spike"].to_numpy(dtype=float),
                    marker="o",
                    linewidth=lw,
                    label=label,
                    color=color,
                )
            ax.axvline(1.0, color="0.6", linestyle=":", linewidth=1.0)
            ax.set_title(f"{panel}\n{axis_mode.replace('_', ' ')}")
            ax.set_xlabel("display scale")
            ax.set_ylabel("population SSI (bits/spike)")
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)
    png = out_dir / "backimage_rr100_contributor_dominance_robust_curves.png"
    pdf = out_dir / "backimage_rr100_contributor_dominance_robust_curves.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_robust_delta_curves(curves: pd.DataFrame, out_dir: Path, *, dpi: int) -> tuple[Path, Path]:
    plot_metrics = [
        ("spike_weighted_all_units", "all units", "black", 2.4),
        ("equal_unit_mean", "equal unit mean", "0.45", 1.7),
        ("spike_weight_cap_p95", "cap spike weights p95", "#1f77b4", 1.8),
        ("unit_ssi_cap_p95", "cap unit SSI p95", "#9467bd", 1.8),
        ("drop_top5_positive_contributors", "drop top 5 contributors", "#d62728", 1.8),
        ("drop_top10_positive_contributors", "drop top 10 contributors", "#ff7f0e", 1.8),
    ]
    panels = ["opposing axis fixed at 1x", "opposing axis fixed at 0x"]
    axis_modes = ["across_sweep", "along_sweep"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True, sharey=True, constrained_layout=True)
    for row_idx, panel in enumerate(panels):
        for col_idx, axis_mode in enumerate(axis_modes):
            ax = axes[row_idx, col_idx]
            sub = curves[(curves["panel"] == panel) & (curves["axis_mode"] == axis_mode)]
            for metric, label, color, lw in plot_metrics:
                msub = sub[sub["metric"] == metric].sort_values("display_scale").copy()
                if msub.empty:
                    continue
                ref = msub[np.isclose(msub["display_scale"].astype(float), 1.0)]
                if ref.empty:
                    continue
                baseline = float(ref["population_bits_per_spike"].iloc[0])
                ax.plot(
                    msub["display_scale"].to_numpy(dtype=float),
                    msub["population_bits_per_spike"].to_numpy(dtype=float) - baseline,
                    marker="o",
                    linewidth=lw,
                    label=label,
                    color=color,
                )
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
            ax.axvline(1.0, color="0.6", linestyle=":", linewidth=1.0)
            ax.set_title(f"{panel}\n{axis_mode.replace('_', ' ')}")
            ax.set_xlabel("display scale")
            ax.set_ylabel("SSI delta from 1x (bits/spike)")
            ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)
    png = out_dir / "backimage_rr100_contributor_dominance_robust_delta_curves.png"
    pdf = out_dir / "backimage_rr100_contributor_dominance_robust_delta_curves.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_top_contributors(unit_contrib: pd.DataFrame, out_dir: Path, *, dpi: int) -> tuple[Path, Path]:
    panels = ["opposing axis fixed at 1x", "opposing axis fixed at 0x"]
    axis_modes = ["across_sweep", "along_sweep"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8), constrained_layout=True)
    for row_idx, panel in enumerate(panels):
        for col_idx, axis_mode in enumerate(axis_modes):
            ax = axes[row_idx, col_idx]
            sub = unit_contrib[(unit_contrib["panel"] == panel) & (unit_contrib["axis_mode"] == axis_mode)].copy()
            sub = sub.sort_values("population_delta_additive_contribution", ascending=False).head(15)
            labels = sub["unit_label"].astype(str).tolist()
            values = sub["population_delta_additive_contribution"].to_numpy(dtype=float)
            colors = ["#2ca02c" if val >= 0 else "#d62728" for val in values]
            x = np.arange(len(values))
            ax.bar(x, values, color=colors, alpha=0.8)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_title(f"{panel}\n{axis_mode.replace('_', ' ')}")
            ax.set_ylabel("additive contribution to 3x - 1x delta")
            ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Top positive unit contributions to spike-weighted population delta", fontsize=15)
    png = out_dir / "backimage_rr100_contributor_dominance_top_units.png"
    pdf = out_dir / "backimage_rr100_contributor_dominance_top_units.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_k = parse_int_list(args.top_k)
    cap_quantiles = parse_float_list(args.cap_quantiles)

    axis1 = load_ssi_table(Path(args.axis1_ssi_csv), "opposing axis fixed at 1x")
    axis0 = load_ssi_table(Path(args.axis0_ssi_csv), "opposing axis fixed at 0x")
    groups = load_orientation_groups(Path(args.orientation_groups_csv))
    df = add_orientation_groups(pd.concat([axis1, axis0], ignore_index=True), groups)

    contribution_frames: list[pd.DataFrame] = []
    totals: dict[tuple[str, str], dict[str, float]] = {}
    for panel in sorted(df["panel"].unique()):
        for axis_mode in sorted(df["axis_mode"].unique()):
            contrib, total = compute_unit_contributions(
                df,
                panel=str(panel),
                axis_mode=str(axis_mode),
                reference_scale=float(args.reference_scale),
                endpoint_scale=float(args.endpoint_scale),
            )
            contribution_frames.append(contrib)
            totals[(str(panel), str(axis_mode))] = total
    unit_contrib = pd.concat(contribution_frames, ignore_index=True)
    unit_contrib = unit_contrib.sort_values(
        ["panel", "axis_mode", "population_delta_additive_contribution"],
        ascending=[True, True, False],
    )

    summary_rows = build_summary_rows(
        df,
        unit_contrib,
        totals,
        reference_scale=float(args.reference_scale),
        endpoint_scale=float(args.endpoint_scale),
        top_k=top_k,
        cap_quantiles=cap_quantiles,
    )
    curve_rows = build_curve_rows(df, unit_contrib, cap_quantiles=cap_quantiles)

    unit_csv = out_dir / "contributor_dominance_unit_contributions.csv"
    summary_csv = out_dir / "contributor_dominance_summary.csv"
    curves_csv = out_dir / "contributor_dominance_robust_curves.csv"
    unit_contrib.to_csv(unit_csv, index=False)
    write_csv(summary_csv, summary_rows)
    write_csv(curves_csv, curve_rows)

    curves_df = pd.DataFrame(curve_rows)
    robust_png, robust_pdf = plot_robust_curves(curves_df, out_dir, dpi=int(args.dpi))
    robust_delta_png, robust_delta_pdf = plot_robust_delta_curves(curves_df, out_dir, dpi=int(args.dpi))
    top_png, top_pdf = plot_top_contributors(unit_contrib, out_dir, dpi=int(args.dpi))

    summary_json = out_dir / "summary.json"
    write_json(
        summary_json,
        {
            "analysis": "backimage_rr100_contributor_dominance",
            "axis1_ssi_csv": Path(args.axis1_ssi_csv),
            "axis0_ssi_csv": Path(args.axis0_ssi_csv),
            "orientation_groups_csv": Path(args.orientation_groups_csv),
            "reference_scale": float(args.reference_scale),
            "endpoint_scale": float(args.endpoint_scale),
            "definitions": {
                "population_delta_additive_contribution": (
                    "(unit_bits_3x / total_spikes_3x) - "
                    "(unit_bits_1x / total_spikes_1x); sums exactly to population 3x-1x delta"
                ),
                "population_delta_leave_one_out_leverage": (
                    "full population delta minus recomputed population delta after removing that unit"
                ),
                "robust_curves": (
                    "population SSI curves after equal-unit averaging, spike-weight caps, unit-SSI caps, "
                    "or excluding top contributors identified from the 3x-vs-1x delta"
                ),
            },
            "outputs": {
                "unit_contributions_csv": unit_csv,
                "summary_csv": summary_csv,
                "robust_curves_csv": curves_csv,
                "robust_curves_png": robust_png,
                "robust_curves_pdf": robust_pdf,
                "robust_delta_curves_png": robust_delta_png,
                "robust_delta_curves_pdf": robust_delta_pdf,
                "top_contributors_png": top_png,
                "top_contributors_pdf": top_pdf,
            },
        },
    )

    print(f"Wrote unit contributions: {unit_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote robust curves: {robust_pdf}")
    print(f"Wrote robust delta curves: {robust_delta_pdf}")
    print(f"Wrote top contributor plot: {top_pdf}")


if __name__ == "__main__":
    main()
