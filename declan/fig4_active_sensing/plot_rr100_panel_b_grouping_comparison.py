#!/usr/bin/env python3
"""Compare corrected Figure-4B path curves across tuning-only unit splits."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2 import compose_ssi_figure_v4_corrected_sf_quartiles as figure


ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = ROOT / (
    "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/"
    "assembled/rounds_000_031_n032_clean_history_snapshot_v1"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_panel_b_grouping_comparison_v2"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GOLD = "#E69F00"
PINK = "#CC79A7"
PURPLE = "#7B61A8"


def rank_split(frame: pd.DataFrame, metric: str, low_name: str, high_name: str) -> pd.Series:
    ordered = frame.sort_values([metric, "rr100_index"]).index.to_numpy()
    low, high = np.array_split(ordered, 2)
    out = pd.Series("excluded", index=frame.index, dtype=object)
    out.loc[low] = low_name
    out.loc[high] = high_name
    return out


def build_assignments(audit: pd.DataFrame) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    valid = audit[audit.recorded_validation_pass.astype(bool)].copy()
    if len(valid) != 61:
        raise ValueError(f"Expected 61 validated units, found {len(valid)}")
    valid["preferred_speed_dps"] = valid.preferred_tf_hz / valid.preferred_sf_cpd
    log_tf = np.log2(valid.preferred_tf_hz.to_numpy(float))
    log_sf = np.log2(valid.preferred_sf_cpd.to_numpy(float))
    residual = log_sf - np.polyval(np.polyfit(log_tf, log_sf, 1), log_tf)
    valid["tf_adjusted_log2_sf"] = residual

    sf_order = valid.sort_values(["preferred_sf_cpd", "rr100_index"]).index.to_numpy()
    sf_outer = pd.Series("excluded", index=valid.index, dtype=object)
    sf_outer.loc[sf_order[:20]] = "bottom"
    sf_outer.loc[sf_order[-20:]] = "top"

    schemes: dict[str, dict[str, object]] = {
        "sf_quartiles": {
            "title": "Preferred-SF quartiles",
            "assignment": valid.sf_quartile.copy(),
            "groups": ("sf_q1", "sf_q2", "sf_q3", "sf_q4"),
            "labels": {"sf_q1": "Q1", "sf_q2": "Q2", "sf_q3": "Q3", "sf_q4": "Q4"},
            "colors": {"sf_q1": BLUE, "sf_q2": GREEN, "sf_q3": GOLD, "sf_q4": PINK},
        },
        "sf_halves": {
            "title": "Preferred-SF halves",
            "assignment": rank_split(valid, "preferred_sf_cpd", "low", "high"),
            "groups": ("low", "high"),
            "labels": {"low": "low SF", "high": "high SF"},
            "colors": {"low": BLUE, "high": ORANGE},
        },
        "sf_outer_thirds": {
            "title": "Preferred-SF outer thirds",
            "assignment": sf_outer,
            "groups": ("bottom", "top"),
            "labels": {"bottom": "bottom SF", "top": "top SF"},
            "colors": {"bottom": BLUE, "top": ORANGE},
        },
        "tf_halves": {
            "title": "Preferred-TF halves",
            "assignment": rank_split(valid, "preferred_tf_hz", "low", "high"),
            "groups": ("low", "high"),
            "labels": {"low": "low TF", "high": "high TF"},
            "colors": {"low": PURPLE, "high": GREEN},
        },
        "speed_halves": {
            "title": "Preferred-speed halves (TF/SF)",
            "assignment": rank_split(valid, "preferred_speed_dps", "slow", "fast"),
            "groups": ("slow", "fast"),
            "labels": {"slow": "slow pref.", "fast": "fast pref."},
            "colors": {"slow": PURPLE, "fast": GOLD},
        },
        "tf_adjusted_sf_halves": {
            "title": "TF-adjusted preferred-SF halves",
            "assignment": rank_split(valid, "tf_adjusted_log2_sf", "low", "high"),
            "groups": ("low", "high"),
            "labels": {"low": "lower SF | TF", "high": "higher SF | TF"},
            "colors": {"low": BLUE, "high": ORANGE},
        },
    }
    rows = []
    for scheme_name, spec in schemes.items():
        assignment = spec["assignment"]
        for index, row in valid.iterrows():
            rows.append({
                "scheme": scheme_name,
                "rr100_index": int(row.rr100_index),
                "group": str(assignment.loc[index]),
                "preferred_sf_cpd": float(row.preferred_sf_cpd),
                "preferred_tf_hz": float(row.preferred_tf_hz),
                "preferred_speed_dps": float(row.preferred_speed_dps),
                "tf_adjusted_log2_sf": float(row.tf_adjusted_log2_sf),
            })
    return schemes, pd.DataFrame(rows)


def compute_curves(data: dict[str, object], schemes: dict[str, dict[str, object]]) -> pd.DataFrame:
    condition = data["condition"]
    images = data["images"]
    audit = data["quartiles"]
    image_ids = condition.image_index.to_numpy(int)
    path = condition.corrected_dpi_crop120_path_length_arcmin.to_numpy(float)
    context = condition.context.to_numpy(str)
    coherence = images.corrected_reconstruction_orientation_coherence.to_numpy(float)
    strong_images = np.isfinite(coherence) & (coherence >= figure.COHERENCE_MIN)
    rng = np.random.default_rng(20260814)
    output = []
    for scheme_name, spec in schemes.items():
        assignment = spec["assignment"]
        for group in spec["groups"]:
            member_ids = audit.loc[assignment.index[assignment.eq(group)], "rr100_index"].to_numpy(int)
            mask = np.zeros((100, 100), dtype=bool)
            mask[np.ix_(strong_images, member_ids)] = True
            stats = figure.condition_sufficient_statistics(
                data["moving_info"], data["moving_spikes"], data["baseline_info"],
                data["baseline_spikes"], image_ids, mask,
            )
            table = figure.summarize_binned(
                x=path, context=context, image_ids=image_ids,
                info=stats[0], spikes=stats[1], base_info=stats[2], base_spikes=stats[3], valid=stats[4],
                group=str(group), relation="strong_contours",
                bins_by_context={"drift_only": 7, "microsaccade": 3}, rng=rng,
            )
            table["scheme"] = scheme_name
            table["n_units"] = int(len(member_ids))
            output.append(table)
    return pd.concat(output, ignore_index=True)


def draw(curves: pd.DataFrame, schemes: dict[str, dict[str, object]]) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 10,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.4), sharey=True, constrained_layout=True)
    low = float(curves.ci95_low.min()); high = float(curves.ci95_high.max())
    pad = 0.06 * (high - low)
    for panel, (ax, (scheme_name, spec)) in enumerate(zip(axes.flat, schemes.items()), start=1):
        groups = spec["groups"]; colors = spec["colors"]; labels = spec["labels"]
        figure.add_segmented_zero_anchor(ax, [colors[group] for group in groups])
        for group in groups:
            for context_name, filled in (("drift_only", False), ("microsaccade", True)):
                sub = curves[
                    curves.scheme.eq(scheme_name) & curves.sf_quartile.eq(group) & curves.context.eq(context_name)
                ].sort_values("bin")
                yerr = np.vstack([sub.delta_percent - sub.ci95_low, sub.ci95_high - sub.delta_percent])
                ax.errorbar(
                    figure.path_broken_log(sub.x_median), sub.delta_percent, yerr=yerr,
                    color=colors[group], marker="o", mfc=colors[group] if filled else "white",
                    mec=colors[group], ms=3.8, lw=1.35, ls="-", capsize=1.5,
                    label=f"{labels[group]} (n={int(sub.n_units.iloc[0])})" if context_name == "drift_only" else None,
                )
        figure.format_broken_path_axis(ax)
        ax.axhline(0, color="0.45", lw=0.7, ls=":")
        ax.set_ylim(low - pad, high + pad)
        ax.set_title(f"{chr(64 + panel)}  {spec['title']}", loc="left", weight="bold")
        ax.set_xlabel("retinal path length (arcmin)")
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="best", handlelength=1.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("SSI change (%) vs matched stabilized")
    axes[1, 0].set_ylabel("SSI change (%) vs matched stabilized")
    fig.suptitle(
        "Corrected Figure-4B path-length curves under tuning-only population splits\n"
        "same movies, trace labels, stabilized baseline, spike weighting, and image bootstrap in every panel",
        fontsize=14, weight="bold",
    )
    fig.text(0.995, 0.005, "open: drift only · filled: scored-window microsaccade", ha="right", fontsize=7.5)
    png = OUT / "rr100_panel_b_grouping_comparison.png"
    pdf = OUT / "rr100_panel_b_grouping_comparison.pdf"
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    figure.ASSEMBLED = ASSEMBLED
    data = figure.build_inputs()
    schemes, assignments = build_assignments(data["quartiles"])
    curves = compute_curves(data, schemes)
    assignments.to_csv(OUT / "unit_assignments_all_schemes.csv", index=False)
    curves.to_csv(OUT / "panel_b_curves_all_schemes.csv", index=False)
    png, pdf = draw(curves, schemes)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "panel_b_grouping_comparison_complete",
        "assembled_snapshot": str(ASSEMBLED.resolve()),
        "n_conditions": int(len(data["condition"])),
        "scope": "exploratory comparison; all splits defined from validated tuning only",
        "schemes": list(schemes),
        "shared_estimand": (
            "strong corrected contours; pooled spike-weighted SSI percent change versus image-matched stabilized; "
            "7 drift-only and 3 microsaccade equal-count path bins; image-cluster bootstrap"
        ),
        "outputs": {"png": str(png.resolve()), "pdf": str(pdf.resolve())},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
