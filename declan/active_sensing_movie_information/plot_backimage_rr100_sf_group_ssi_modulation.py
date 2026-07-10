"""Compare BackImage SSI modulation for low- and high-SF-tuned RR100 units."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TUNING_DIR = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1"
)
DEFAULT_SSI_CSV = Path(
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_latest_v1/displayed_movie_instantaneous_ssi_all_units.csv"
)
DEFAULT_OUT_DIR = DEFAULT_TUNING_DIR / "sf_group_ssi_modulation"
VALUE_COL = "displayed_movie_time_resolved_ssi_bits_per_spike"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--ssi-csv", type=Path, default=DEFAULT_SSI_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sf-metric",
        choices=("dynamic_amp_weighted", "static_rate_weighted", "dynamic_peak", "static_peak"),
        default="dynamic_amp_weighted",
    )
    parser.add_argument("--tertile-n", type=int, default=None, help="Units per tail. Default is floor(n_units / 3).")
    parser.add_argument("--zscore-min-std", type=float, default=1e-8)
    return parser.parse_args()


def sem(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def weighted_log2_sf(grouped: pd.DataFrame, *, dynamic: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if dynamic:
        use = grouped[pd.to_numeric(grouped["temporal_hz"], errors="coerce") > 0].copy()
        weight_col = "response_amp_rms"
        metric_name = "dynamic_amp_weighted_sf_cpd"
    else:
        use = grouped[pd.to_numeric(grouped["temporal_hz"], errors="coerce") == 0].copy()
        weight_col = "mean_rate"
        metric_name = "static_rate_weighted_sf_cpd"

    use["spatial_cpd"] = pd.to_numeric(use["spatial_cpd"], errors="coerce")
    use[weight_col] = pd.to_numeric(use[weight_col], errors="coerce").clip(lower=0.0)
    use = use[use["spatial_cpd"] > 0].copy()
    use["log2_sf"] = np.log2(use["spatial_cpd"].to_numpy(dtype=float))

    for (unit_index, unit_label), sub in use.groupby(["unit_index", "unit_label"], sort=True):
        weights = sub[weight_col].to_numpy(dtype=float)
        logs = sub["log2_sf"].to_numpy(dtype=float)
        ok = np.isfinite(weights) & np.isfinite(logs)
        total = float(np.nansum(weights[ok]))
        if total > 0:
            log_pref = float(np.nansum(weights[ok] * logs[ok]) / total)
            sf_pref = float(2.0**log_pref)
        else:
            log_pref = float("nan")
            sf_pref = float("nan")
        rows.append(
            {
                "unit_index": int(unit_index),
                "unit_label": str(unit_label),
                f"{metric_name}_log2": log_pref,
                metric_name: sf_pref,
                f"{metric_name}_weight_sum": total,
            }
        )
    return pd.DataFrame(rows)


def build_sf_groups(tuning_dir: Path, sf_metric: str, tertile_n: int | None) -> pd.DataFrame:
    summary = pd.read_csv(tuning_dir / "frequency_tuning_summary.csv")
    grouped = pd.read_csv(tuning_dir / "frequency_tuning_grouped.csv")
    dynamic_weighted = weighted_log2_sf(grouped, dynamic=True)
    static_weighted = weighted_log2_sf(grouped, dynamic=False)
    units = summary.merge(dynamic_weighted, on=["unit_index", "unit_label"], how="left", validate="one_to_one")
    units = units.merge(static_weighted, on=["unit_index", "unit_label"], how="left", validate="one_to_one")

    metric_col = {
        "dynamic_amp_weighted": "dynamic_amp_weighted_sf_cpd",
        "static_rate_weighted": "static_rate_weighted_sf_cpd",
        "dynamic_peak": "dynamic_peak_spatial_cpd_by_amp",
        "static_peak": "static_peak_spatial_cpd_by_mean_rate",
    }[sf_metric]
    units["sf_split_metric"] = pd.to_numeric(units[metric_col], errors="coerce")
    units = units[np.isfinite(units["sf_split_metric"])].copy()
    units = units.sort_values(["sf_split_metric", "unit_index"], ascending=[True, True]).reset_index(drop=True)

    n_units = int(len(units))
    n_tail = int(tertile_n) if tertile_n is not None else n_units // 3
    if n_tail <= 0 or 2 * n_tail > n_units:
        raise ValueError(f"Invalid tertile-n {n_tail} for {n_units} units.")
    units["sf_rank_low_to_high"] = np.arange(1, n_units + 1)
    units["sf_group"] = "middle_sf"
    units.loc[units.index < n_tail, "sf_group"] = "low_sf"
    units.loc[units.index >= n_units - n_tail, "sf_group"] = "high_sf"
    units["sf_group_label"] = units["sf_group"].map(
        {
            "low_sf": f"low SF bottom third (n={n_tail})",
            "middle_sf": f"middle SF (n={n_units - 2 * n_tail})",
            "high_sf": f"high SF top third (n={n_tail})",
        }
    )
    units["sf_split_metric_name"] = sf_metric
    units["sf_split_metric_column"] = metric_col
    return units


def add_curve_metrics(ssi: pd.DataFrame, units: pd.DataFrame, zscore_min_std: float) -> pd.DataFrame:
    curves = ssi.merge(
        units[
            [
                "unit_index",
                "unit_label",
                "sf_group",
                "sf_group_label",
                "sf_rank_low_to_high",
                "sf_split_metric",
                "sf_split_metric_name",
                "dynamic_amp_weighted_sf_cpd",
                "static_rate_weighted_sf_cpd",
                "dynamic_peak_spatial_cpd_by_amp",
                "static_peak_spatial_cpd_by_mean_rate",
                "dynamic_peak_temporal_hz_by_amp",
                "prior_preferred_orientation_deg",
                "prior_orientation_selectivity_index",
            ]
        ],
        on=["unit_index", "unit_label"],
        how="inner",
        validate="many_to_one",
    ).copy()
    curves[VALUE_COL] = pd.to_numeric(curves[VALUE_COL], errors="coerce")
    curves["display_scale"] = pd.to_numeric(curves["display_scale"], errors="coerce")

    reference = curves[np.isclose(curves["display_scale"], 1.0)].copy()
    reference = reference[["unit_index", "axis_mode", VALUE_COL]].rename(columns={VALUE_COL: "ssi_at_scale_1"})
    curves = curves.merge(reference, on=["unit_index", "axis_mode"], how="left", validate="many_to_one")
    curves["ssi_delta_vs_1x"] = curves[VALUE_COL] - curves["ssi_at_scale_1"]

    stats = curves.groupby(["unit_index", "axis_mode"])[VALUE_COL].agg(["mean", "std"]).reset_index()
    stats = stats.rename(columns={"mean": "ssi_unit_axis_mean", "std": "ssi_unit_axis_std"})
    curves = curves.merge(stats, on=["unit_index", "axis_mode"], how="left", validate="many_to_one")
    curves["ssi_zscore_axis_mode"] = np.where(
        curves["ssi_unit_axis_std"].to_numpy(dtype=float) > float(zscore_min_std),
        (curves[VALUE_COL] - curves["ssi_unit_axis_mean"]) / curves["ssi_unit_axis_std"],
        np.nan,
    )
    curves["ssi_zscore_contract"] = "per-unit z-score across display scales within each axis_mode"
    return curves


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (sf_group, axis_mode, display_scale), sub in curves.groupby(["sf_group", "axis_mode", "display_scale"], sort=True):
        for value_col in [VALUE_COL, "ssi_delta_vs_1x", "ssi_zscore_axis_mode", "displayed_movie_mean_rate"]:
            values = pd.to_numeric(sub[value_col], errors="coerce")
            rows.append(
                {
                    "sf_group": sf_group,
                    "sf_group_label": str(sub["sf_group_label"].iloc[0]),
                    "axis_mode": axis_mode,
                    "display_scale": float(display_scale),
                    "value_name": value_col,
                    "n_units": int(sub["unit_index"].nunique()),
                    "mean": float(np.nanmean(values)),
                    "sem": sem(values),
                    "median": float(np.nanmedian(values)),
                    "q25": float(np.nanpercentile(values, 25)),
                    "q75": float(np.nanpercentile(values, 75)),
                }
            )
    return pd.DataFrame(rows)


def endpoint_summary(curves: pd.DataFrame) -> pd.DataFrame:
    endpoints = curves[curves["display_scale"].isin([1.0, 3.0])].copy()
    pivot = endpoints.pivot_table(
        index=["unit_index", "unit_label", "sf_group", "sf_group_label", "axis_mode"],
        columns="display_scale",
        values=VALUE_COL,
        aggfunc="first",
    ).reset_index()
    pivot["delta_3_minus_1"] = pivot[3.0] - pivot[1.0]
    rows: list[dict[str, object]] = []
    for (sf_group, axis_mode), sub in pivot.groupby(["sf_group", "axis_mode"], sort=True):
        rows.append(
            {
                "sf_group": sf_group,
                "sf_group_label": str(sub["sf_group_label"].iloc[0]),
                "axis_mode": axis_mode,
                "n_units": int(sub["unit_index"].nunique()),
                "mean_delta_3_minus_1": float(np.nanmean(sub["delta_3_minus_1"])),
                "sem_delta_3_minus_1": sem(sub["delta_3_minus_1"]),
                "median_delta_3_minus_1": float(np.nanmedian(sub["delta_3_minus_1"])),
            }
        )

    out = pd.DataFrame(rows)
    diff_rows: list[dict[str, object]] = []
    for axis_mode, sub in pivot.groupby("axis_mode", sort=True):
        low = sub[sub["sf_group"] == "low_sf"]["delta_3_minus_1"].to_numpy(dtype=float)
        high = sub[sub["sf_group"] == "high_sf"]["delta_3_minus_1"].to_numpy(dtype=float)
        diff_rows.append(
            {
                "axis_mode": axis_mode,
                "comparison": "high_sf_minus_low_sf",
                "mean_delta_diff": float(np.nanmean(high) - np.nanmean(low)),
                "median_delta_diff": float(np.nanmedian(high) - np.nanmedian(low)),
                "n_low": int(np.isfinite(low).sum()),
                "n_high": int(np.isfinite(high).sum()),
            }
        )
    return out.merge(pd.DataFrame(diff_rows), on="axis_mode", how="left")


def group_style(sf_group: str) -> tuple[str, float, int]:
    if sf_group == "low_sf":
        return "#2673a6", 1.0, 3
    if sf_group == "high_sf":
        return "#c74343", 1.0, 3
    return "0.62", 0.45, 2


def plot_curves(summary: pd.DataFrame, curves: pd.DataFrame, units: pd.DataFrame, out_dir: Path, sf_metric: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    axis_titles = {
        "across_sweep": "scale across; along=1",
        "along_sweep": "scale along; across=1",
    }
    value_panels = [
        (VALUE_COL, "raw SSI bits/spike", "mean SSI"),
        ("ssi_delta_vs_1x", "delta from 1x", "SSI - SSI at 1x"),
        ("ssi_zscore_axis_mode", "within-unit z-score", "SSI z-score"),
    ]
    group_order = ["low_sf", "middle_sf", "high_sf"]

    for row_i, axis_mode in enumerate(["across_sweep", "along_sweep"]):
        for col_i, (value_name, title, ylabel) in enumerate(value_panels):
            ax = axes[row_i, col_i]
            for sf_group in group_order:
                sub = summary[
                    (summary["axis_mode"].astype(str) == axis_mode)
                    & (summary["value_name"].astype(str) == value_name)
                    & (summary["sf_group"].astype(str) == sf_group)
                ].sort_values("display_scale")
                if sub.empty:
                    continue
                color, alpha, zorder = group_style(sf_group)
                label = str(sub["sf_group_label"].iloc[0])
                x = sub["display_scale"].to_numpy(dtype=float)
                y = sub["mean"].to_numpy(dtype=float)
                err = sub["sem"].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", color=color, alpha=alpha, lw=2.2, label=label, zorder=zorder)
                ax.fill_between(x, y - err, y + err, color=color, alpha=0.12 * alpha, zorder=zorder)
            ax.axvline(1.0, ls=":", color="0.6", lw=1.0)
            if value_name in {"ssi_delta_vs_1x", "ssi_zscore_axis_mode"}:
                ax.axhline(0.0, ls="--", color="0.7", lw=0.9)
            ax.set_title(f"{axis_titles[axis_mode]}\n{title}")
            ax.set_xlabel("motion scale")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="0.9", linewidth=0.8)
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "BackImage RR100 SSI modulation by SF-tuning tertile\n"
        f"SF split metric: {sf_metric}; SSI is averaged over instantaneous maps",
        fontsize=14,
    )
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_ssi_modulation_curves.png", dpi=180)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_ssi_modulation_curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    ax = axes[0]
    for sf_group in group_order:
        sub = units[units["sf_group"] == sf_group]
        color, alpha, zorder = group_style(sf_group)
        ax.scatter(
            sub["sf_rank_low_to_high"],
            sub["sf_split_metric"],
            color=color,
            alpha=alpha,
            s=35,
            label=str(sub["sf_group_label"].iloc[0]) if not sub.empty else sf_group,
            zorder=zorder,
        )
    ax.set_yscale("log")
    ax.set_title("SF split ranking")
    ax.set_xlabel("unit rank, low to high SF")
    ax.set_ylabel("SF split metric (cpd)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    vals = sorted(units["dynamic_peak_spatial_cpd_by_amp"].dropna().unique())
    x = np.arange(len(vals))
    width = 0.24
    for offset, sf_group in [(-width, "low_sf"), (0.0, "middle_sf"), (width, "high_sf")]:
        sub = units[units["sf_group"] == sf_group]
        color, alpha, _ = group_style(sf_group)
        counts = sub["dynamic_peak_spatial_cpd_by_amp"].value_counts().reindex(vals, fill_value=0)
        ax.bar(x + offset, counts.to_numpy(), width=width, color=color, alpha=max(alpha, 0.5), label=sf_group)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in vals], rotation=35, ha="right")
    ax.set_title("Dynamic peak SF bins")
    ax.set_xlabel("cpd")
    ax.set_ylabel("unit count")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    endpoint = endpoint_summary(curves)
    shown = endpoint[endpoint["sf_group"].isin(["low_sf", "high_sf"])].copy()
    xlabels = ["across_sweep", "along_sweep"]
    xpos = np.arange(len(xlabels))
    width = 0.32
    for offset, sf_group in [(-width / 2, "low_sf"), (width / 2, "high_sf")]:
        sub = shown[shown["sf_group"] == sf_group].set_index("axis_mode").reindex(xlabels)
        color, alpha, _ = group_style(sf_group)
        ax.bar(
            xpos + offset,
            sub["mean_delta_3_minus_1"].to_numpy(dtype=float),
            yerr=sub["sem_delta_3_minus_1"].to_numpy(dtype=float),
            width=width,
            color=color,
            alpha=alpha,
            capsize=3,
            label=str(sub["sf_group_label"].dropna().iloc[0]) if sub["sf_group_label"].notna().any() else sf_group,
        )
    ax.axhline(0.0, ls="--", color="0.7", lw=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["across\nalong=1", "along\nacross=1"])
    ax.set_title("Endpoint modulation")
    ax.set_ylabel("mean SSI(3x) - SSI(1x)")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("SF group definition and endpoint SSI modulation", fontsize=14)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_definition_and_endpoint_deltas.png", dpi=180)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_definition_and_endpoint_deltas.pdf")
    plt.close(fig)


def endpoint_unit_deltas(curves: pd.DataFrame) -> pd.DataFrame:
    endpoints = curves[curves["display_scale"].isin([1.0, 3.0])].copy()
    pivot = endpoints.pivot_table(
        index=["unit_index", "unit_label", "sf_group", "sf_group_label", "axis_mode"],
        columns="display_scale",
        values=VALUE_COL,
        aggfunc="first",
    ).reset_index()
    pivot["delta_3_minus_1"] = pivot[3.0] - pivot[1.0]
    return pivot


def contribution_summary(curves: pd.DataFrame) -> pd.DataFrame:
    key = ["unit_index", "unit_label", "sf_group", "sf_group_label", "axis_mode"]
    endpoints = curves[curves["display_scale"].isin([1.0, 3.0])].copy()
    one = endpoints[endpoints["display_scale"].eq(1.0)][
        key + [VALUE_COL, "displayed_movie_expected_spikes_arbitrary_dt"]
    ].rename(
        columns={
            VALUE_COL: "ssi_1x",
            "displayed_movie_expected_spikes_arbitrary_dt": "expected_spikes_1x",
        }
    )
    three = endpoints[endpoints["display_scale"].eq(3.0)][
        key + [VALUE_COL, "displayed_movie_expected_spikes_arbitrary_dt"]
    ].rename(
        columns={
            VALUE_COL: "ssi_3x",
            "displayed_movie_expected_spikes_arbitrary_dt": "expected_spikes_3x",
        }
    )
    unit = one.merge(three, on=key, how="inner", validate="one_to_one")
    unit["ssi_delta_3_minus_1"] = unit["ssi_3x"] - unit["ssi_1x"]
    unit["expected_spikes_mean_endpoint"] = (unit["expected_spikes_1x"] + unit["expected_spikes_3x"]) / 2.0
    unit["spike_weighted_ssi_delta_numerator"] = (
        unit["ssi_delta_3_minus_1"] * unit["expected_spikes_mean_endpoint"]
    )
    unit["information_numerator_1x"] = unit["ssi_1x"] * unit["expected_spikes_1x"]
    unit["information_numerator_3x"] = unit["ssi_3x"] * unit["expected_spikes_3x"]
    unit["information_numerator_delta_3_minus_1"] = (
        unit["information_numerator_3x"] - unit["information_numerator_1x"]
    )

    rows: list[dict[str, object]] = []
    for axis_mode, axis_sub in unit.groupby("axis_mode", sort=True):
        net_equal_unit_delta = float(axis_sub["ssi_delta_3_minus_1"].sum())
        positive_equal_unit_delta = float(axis_sub.loc[axis_sub["ssi_delta_3_minus_1"] > 0, "ssi_delta_3_minus_1"].sum())
        net_spike_weighted_delta = float(axis_sub["spike_weighted_ssi_delta_numerator"].sum())
        net_information_numerator_delta = float(axis_sub["information_numerator_delta_3_minus_1"].sum())
        for sf_group, group_sub in axis_sub.groupby("sf_group", sort=True):
            group_equal_delta = float(group_sub["ssi_delta_3_minus_1"].sum())
            group_positive_delta = float(
                group_sub.loc[group_sub["ssi_delta_3_minus_1"] > 0, "ssi_delta_3_minus_1"].sum()
            )
            group_spike_weighted_delta = float(group_sub["spike_weighted_ssi_delta_numerator"].sum())
            group_information_delta = float(group_sub["information_numerator_delta_3_minus_1"].sum())
            rows.append(
                {
                    "axis_mode": axis_mode,
                    "sf_group": sf_group,
                    "sf_group_label": str(group_sub["sf_group_label"].iloc[0]),
                    "n_units": int(group_sub["unit_index"].nunique()),
                    "mean_delta_3_minus_1": float(np.nanmean(group_sub["ssi_delta_3_minus_1"])),
                    "median_delta_3_minus_1": float(np.nanmedian(group_sub["ssi_delta_3_minus_1"])),
                    "sum_equal_unit_delta_3_minus_1": group_equal_delta,
                    "share_of_net_equal_unit_delta": (
                        group_equal_delta / net_equal_unit_delta if net_equal_unit_delta else float("nan")
                    ),
                    "sum_positive_equal_unit_delta_3_minus_1": group_positive_delta,
                    "share_of_positive_equal_unit_delta": (
                        group_positive_delta / positive_equal_unit_delta
                        if positive_equal_unit_delta
                        else float("nan")
                    ),
                    "spike_weighted_ssi_delta_numerator": group_spike_weighted_delta,
                    "share_of_net_spike_weighted_ssi_delta": (
                        group_spike_weighted_delta / net_spike_weighted_delta
                        if net_spike_weighted_delta
                        else float("nan")
                    ),
                    "information_numerator_delta_3_minus_1": group_information_delta,
                    "share_of_information_numerator_delta": (
                        group_information_delta / net_information_numerator_delta
                        if net_information_numerator_delta
                        else float("nan")
                    ),
                    "net_equal_unit_delta_all_units": net_equal_unit_delta,
                    "positive_equal_unit_delta_all_units": positive_equal_unit_delta,
                    "net_spike_weighted_ssi_delta_all_units": net_spike_weighted_delta,
                    "net_information_numerator_delta_all_units": net_information_numerator_delta,
                }
            )
    return pd.DataFrame(rows)


def plot_unit_level_views(summary: pd.DataFrame, curves: pd.DataFrame, out_dir: Path, sf_metric: str) -> None:
    axis_titles = {
        "across_sweep": "scale across; along=1",
        "along_sweep": "scale along; across=1",
    }
    value_panels = [
        (VALUE_COL, "raw SSI bits/spike", "SSI"),
        ("ssi_delta_vs_1x", "delta from 1x", "SSI - SSI at 1x"),
        ("ssi_zscore_axis_mode", "within-unit z-score", "SSI z-score"),
    ]
    group_order = ["low_sf", "middle_sf", "high_sf"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    for row_i, axis_mode in enumerate(["across_sweep", "along_sweep"]):
        for col_i, (value_name, title, ylabel) in enumerate(value_panels):
            ax = axes[row_i, col_i]
            for sf_group in group_order:
                group_curves = curves[
                    (curves["axis_mode"].astype(str) == axis_mode)
                    & (curves["sf_group"].astype(str) == sf_group)
                ]
                color, alpha, zorder = group_style(sf_group)
                label = None
                for unit_index, unit_sub in group_curves.groupby("unit_index", sort=False):
                    unit_sub = unit_sub.sort_values("display_scale")
                    ax.plot(
                        unit_sub["display_scale"].to_numpy(dtype=float),
                        pd.to_numeric(unit_sub[value_name], errors="coerce").to_numpy(dtype=float),
                        color=color,
                        alpha=0.22 if sf_group != "middle_sf" else 0.14,
                        lw=0.85,
                        zorder=zorder,
                    )
                    label = str(unit_sub["sf_group_label"].iloc[0])
                mean_sub = summary[
                    (summary["axis_mode"].astype(str) == axis_mode)
                    & (summary["value_name"].astype(str) == value_name)
                    & (summary["sf_group"].astype(str) == sf_group)
                ].sort_values("display_scale")
                if not mean_sub.empty:
                    ax.plot(
                        mean_sub["display_scale"].to_numpy(dtype=float),
                        mean_sub["mean"].to_numpy(dtype=float),
                        color=color,
                        alpha=max(alpha, 0.8),
                        lw=2.7,
                        marker="o",
                        ms=4,
                        label=label,
                        zorder=zorder + 4,
                    )
            ax.axvline(1.0, ls=":", color="0.6", lw=1.0)
            if value_name in {"ssi_delta_vs_1x", "ssi_zscore_axis_mode"}:
                ax.axhline(0.0, ls="--", color="0.72", lw=0.9)
            ax.set_title(f"{axis_titles[axis_mode]}\n{title}")
            ax.set_xlabel("motion scale")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="0.9", linewidth=0.8)
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "BackImage RR100 SF groups: every unit color-coded by group\n"
        f"SF split metric: {sf_metric}; thick lines are group means",
        fontsize=14,
    )
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_all_unit_colorcoded_curves.png", dpi=180)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_all_unit_colorcoded_curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.2), sharey=True, constrained_layout=True)
    finite_delta = pd.to_numeric(curves["ssi_delta_vs_1x"], errors="coerce").to_numpy(dtype=float)
    finite_delta = finite_delta[np.isfinite(finite_delta)]
    ylim = None
    if finite_delta.size:
        pad = max(0.004, 0.08 * (float(np.nanmax(finite_delta)) - float(np.nanmin(finite_delta))))
        ylim = (float(np.nanmin(finite_delta) - pad), float(np.nanmax(finite_delta) + pad))
    for row_i, sf_group in enumerate(group_order):
        color, alpha, zorder = group_style(sf_group)
        for col_i, axis_mode in enumerate(["across_sweep", "along_sweep"]):
            ax = axes[row_i, col_i]
            sub = curves[
                (curves["axis_mode"].astype(str) == axis_mode)
                & (curves["sf_group"].astype(str) == sf_group)
            ]
            for unit_index, unit_sub in sub.groupby("unit_index", sort=False):
                unit_sub = unit_sub.sort_values("display_scale")
                ax.plot(
                    unit_sub["display_scale"].to_numpy(dtype=float),
                    unit_sub["ssi_delta_vs_1x"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.32 if sf_group != "middle_sf" else 0.22,
                    lw=1.0,
                    zorder=zorder,
                )
            mean_sub = summary[
                (summary["axis_mode"].astype(str) == axis_mode)
                & (summary["value_name"].astype(str) == "ssi_delta_vs_1x")
                & (summary["sf_group"].astype(str) == sf_group)
            ].sort_values("display_scale")
            if not mean_sub.empty:
                ax.plot(
                    mean_sub["display_scale"].to_numpy(dtype=float),
                    mean_sub["mean"].to_numpy(dtype=float),
                    color="black",
                    lw=2.4,
                    marker="o",
                    ms=4,
                    label="group mean",
                    zorder=10,
                )
            ax.axvline(1.0, ls=":", color="0.6", lw=1.0)
            ax.axhline(0.0, ls="--", color="0.72", lw=0.9)
            if ylim is not None:
                ax.set_ylim(*ylim)
            group_label = str(sub["sf_group_label"].iloc[0]) if not sub.empty else sf_group
            ax.set_title(f"{group_label}\n{axis_titles[axis_mode]}")
            ax.set_xlabel("motion scale")
            if col_i == 0:
                ax.set_ylabel("SSI - SSI at 1x")
            ax.grid(True, color="0.9", linewidth=0.8)
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "BackImage RR100 SF groups: unit-level SSI modulation within each group\n"
        "Each thin line is one unit; panels share the y-axis",
        fontsize=14,
    )
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_faceted_unit_delta_curves.png", dpi=180)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_faceted_unit_delta_curves.pdf")
    plt.close(fig)

    unit_deltas = endpoint_unit_deltas(curves)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True, constrained_layout=True)
    for ax, axis_mode in zip(axes, ["across_sweep", "along_sweep"], strict=True):
        data = []
        labels = []
        colors = []
        for sf_group in group_order:
            sub = unit_deltas[
                (unit_deltas["axis_mode"].astype(str) == axis_mode)
                & (unit_deltas["sf_group"].astype(str) == sf_group)
            ]
            data.append(sub["delta_3_minus_1"].to_numpy(dtype=float))
            labels.append(str(sub["sf_group_label"].iloc[0]) if not sub.empty else sf_group)
            colors.append(group_style(sf_group)[0])
        parts = ax.violinplot(data, positions=np.arange(1, len(data) + 1), showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors, strict=True):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.22)
        parts["cmeans"].set_color("black")
        for i, (values, color) in enumerate(zip(data, colors, strict=True), start=1):
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                continue
            jitter = np.linspace(-0.11, 0.11, finite.size)
            ax.scatter(np.full(finite.size, i) + jitter, finite, s=24, color=color, alpha=0.72, edgecolor="white", linewidth=0.35)
        ax.axhline(0.0, ls="--", color="0.72", lw=0.9)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(axis_titles[axis_mode])
        ax.set_ylabel("SSI(3x) - SSI(1x)")
        ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    fig.suptitle("BackImage RR100 SF groups: endpoint delta distribution by unit", fontsize=14)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_endpoint_delta_distributions.png", dpi=180)
    fig.savefig(out_dir / f"backimage_rr100_{sf_metric}_sf_group_endpoint_delta_distributions.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    units = build_sf_groups(args.tuning_dir, args.sf_metric, args.tertile_n)
    ssi = pd.read_csv(args.ssi_csv)
    curves = add_curve_metrics(ssi, units, zscore_min_std=float(args.zscore_min_std))
    summary = summarize_curves(curves)
    endpoints = endpoint_summary(curves)
    contributions = contribution_summary(curves)

    units.to_csv(args.out_dir / f"{args.sf_metric}_sf_tuning_unit_groups.csv", index=False)
    curves.to_csv(args.out_dir / f"{args.sf_metric}_sf_group_ssi_curves_long.csv", index=False)
    summary.to_csv(args.out_dir / f"{args.sf_metric}_sf_group_ssi_summary.csv", index=False)
    endpoints.to_csv(args.out_dir / f"{args.sf_metric}_sf_group_endpoint_delta_summary.csv", index=False)
    contributions.to_csv(args.out_dir / f"{args.sf_metric}_sf_group_3x_increase_contribution_summary.csv", index=False)
    plot_curves(summary, curves, units, args.out_dir, args.sf_metric)
    plot_unit_level_views(summary, curves, args.out_dir, args.sf_metric)

    print(f"Wrote SF-group SSI modulation outputs to {args.out_dir}")
    print(units["sf_group"].value_counts().to_string())
    print(endpoints.to_string(index=False))


if __name__ == "__main__":
    main()
