#!/usr/bin/env python3
"""Checkpoint 5: page-14 aligned-versus-orthogonal overlay with new SF quartiles."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.analyze_backimage_real_trace_ssi_matrix_phase1_phase2 import (
    add_trace_path_context_bands,
)
from declan.fig4_active_sensing.rerun_backimage_real_trace_contour_matched_sf_quartiles import (
    COLORS,
    EXTREMES,
    GROUPS,
    LABELS,
    MATCH_MAX_DEG,
    MIN_MATCHED_IMAGES,
    MIN_OSI,
    ORTHOGONAL_MIN_DEG,
    ROOT,
    configure_matplotlib,
    file_identity,
    save_figure,
)


CHECK_ROOT = ROOT / "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1"
DEFAULT_MATCHED = CHECK_ROOT / "checkpoint_03_contour_matched_sf_quartiles"
DEFAULT_ORTHOGONAL = CHECK_ROOT / "checkpoint_04_contour_orthogonal_sf_quartiles"
DEFAULT_OUT = CHECK_ROOT / "checkpoint_05_aligned_vs_orthogonal_sf_quartiles"
ABSOLUTE = "unit_contour_matched_ssi_bits_per_spike"
DELTA = "unit_contour_matched_ssi_delta_vs_stabilized"
RELATIONS = (
    ("aligned", "contour aligned", "-", "o"),
    ("orthogonal", "contour orthogonal", "--", "s"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-dir", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--orthogonal-dir", type=Path, default=DEFAULT_ORTHOGONAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def mean_sem(values: pd.Series) -> tuple[float, float]:
    arr = values.dropna().to_numpy(float)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, sem


def plot_relation_curves(
    ax: plt.Axes,
    summaries: dict[str, pd.DataFrame],
    groups: tuple[str, ...],
    value_name: str,
    context: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    include_labels: bool,
) -> None:
    add_trace_path_context_bands(ax, context, include_legend=False)
    for relation, relation_label, linestyle, marker in RELATIONS:
        summary = summaries[relation]
        for group in groups:
            sub = summary[
                summary["sf_group"].astype(str).eq(group)
                & summary["value_name"].astype(str).eq(value_name)
            ].sort_values("trace_path_length_bin_median_arcmin")
            if sub.empty:
                continue
            x = sub["trace_path_length_bin_median_arcmin"].to_numpy(float)
            y = sub["mean"].to_numpy(float)
            e = sub["sem"].to_numpy(float)
            label = f"{LABELS[group]}, {relation_label}" if include_labels else None
            ax.plot(
                x, y, color=COLORS[group], linestyle=linestyle, marker=marker,
                lw=2.2, ms=4.3, label=label, zorder=4,
            )
            ax.fill_between(x, y - e, y + e, color=COLORS[group], alpha=0.10, lw=0)
    if value_name == DELTA:
        ax.axhline(0, color="0.4", ls=":", lw=1)
    ax.set(xlabel="trace path length bin median (arcmin)", ylabel=ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, color="0.92", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def make_overlay_figures(
    summaries: dict[str, pd.DataFrame],
    context: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    plot_relation_curves(
        axes[0], summaries, EXTREMES, ABSOLUTE, context,
        title="Absolute SSI on selected windows", ylabel="SSI (bits/spike)", include_labels=True,
    )
    plot_relation_curves(
        axes[1], summaries, EXTREMES, DELTA, context,
        title="Movement modulation", ylabel="SSI minus stabilized baseline (bits/spike)", include_labels=False,
    )
    color_handles = [
        mlines.Line2D([], [], color=COLORS[g], lw=2.5, label=LABELS[g]) for g in EXTREMES
    ]
    relation_handles = [
        mlines.Line2D([], [], color="0.25", lw=2.2, ls=ls, marker=marker, label=label)
        for _relation, label, ls, marker in RELATIONS
    ]
    fig.legend(
        handles=color_handles + relation_handles, loc="lower center", ncol=4,
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Contour-aligned versus contour-orthogonal unit-window pairs: real trace scale\n"
        "new SF extremes; aligned <= 22.5 deg; orthogonal >= 67.5 deg; OSI >= 0.05",
        fontsize=11.2,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.84), w_pad=2.2)
    save_figure(fig, out_dir, "014_phase2_contour_aligned_vs_orthogonal_sf_q1_q4_curves")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), sharex=True)
    for ax, group in zip(axes.ravel(), GROUPS, strict=True):
        plot_relation_curves(
            ax, summaries, (group,), DELTA, context,
            title=f"{LABELS[group]}", ylabel="SSI minus stabilized", include_labels=True,
        )
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Checkpoint 5: aligned-versus-orthogonal movement modulation",
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=2.0)
    save_figure(fig, out_dir, "checkpoint_05_aligned_vs_orthogonal_all_quartiles")


def paired_endpoint_table(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    relation_changes = {}
    for relation, frame in curves.items():
        pivot = frame.pivot(
            index=["unit_index", "unit_label", "sf_group"],
            columns="trace_path_length_bin", values=ABSOLUTE,
        ).reset_index()
        pivot[f"{relation}_long_minus_short"] = pivot["q06"] - pivot["q01"]
        relation_changes[relation] = pivot[
            ["unit_index", "unit_label", "sf_group", f"{relation}_long_minus_short"]
        ]
    paired = relation_changes["aligned"].merge(
        relation_changes["orthogonal"],
        on=["unit_index", "unit_label", "sf_group"], how="inner", validate="one_to_one",
    )
    paired["aligned_minus_orthogonal_endpoint_change"] = (
        paired["aligned_long_minus_short"] - paired["orthogonal_long_minus_short"]
    )
    return paired


def summarize_paired(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    relation_rows = []
    contrast_rows = []
    for group in GROUPS:
        sub = paired[paired["sf_group"].eq(group)]
        for relation in ("aligned", "orthogonal"):
            column = f"{relation}_long_minus_short"
            mean, sem = mean_sem(sub[column])
            relation_rows.append(
                {
                    "sf_quartile": group, "relation": relation, "n_paired_units": int(len(sub)),
                    "mean_long_minus_short_ssi": mean, "sem_long_minus_short_ssi": sem,
                    "median_long_minus_short_ssi": float(sub[column].median()),
                    "fraction_positive": float((sub[column] > 0).mean()),
                }
            )
        column = "aligned_minus_orthogonal_endpoint_change"
        mean, sem = mean_sem(sub[column])
        contrast_rows.append(
            {
                "sf_quartile": group, "n_paired_units": int(len(sub)),
                "mean_aligned_minus_orthogonal_endpoint_change": mean,
                "sem_aligned_minus_orthogonal_endpoint_change": sem,
                "median_aligned_minus_orthogonal_endpoint_change": float(sub[column].median()),
                "fraction_aligned_greater": float((sub[column] > 0).mean()),
            }
        )
    return pd.DataFrame(relation_rows), pd.DataFrame(contrast_rows)


def select_examples(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        sub = paired[paired["sf_group"].eq(group)].copy()
        column = "aligned_minus_orthogonal_endpoint_change"
        median = float(sub[column].median())
        choices = (
            ("largest_aligned_advantage", sub[column].idxmax(), "maximum aligned-minus-orthogonal endpoint change"),
            ("largest_orthogonal_advantage", sub[column].idxmin(), "minimum aligned-minus-orthogonal endpoint change"),
            ("median_relation_contrast", (sub[column] - median).abs().idxmin(), "closest to median relation contrast"),
        )
        for role, index, criterion in choices:
            row = sub.loc[index]
            rows.append(
                {
                    "selection_method": "algorithmic", "selection_role": role,
                    "criterion": criterion, "sf_quartile": group,
                    "unit_index": int(row["unit_index"]), "unit_label": str(row["unit_label"]),
                    "aligned_long_minus_short": float(row["aligned_long_minus_short"]),
                    "orthogonal_long_minus_short": float(row["orthogonal_long_minus_short"]),
                    "aligned_minus_orthogonal_endpoint_change": float(row[column]),
                }
            )
    return pd.DataFrame(rows)


def make_paired_figure(
    paired: pd.DataFrame,
    relation_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    x = np.arange(len(GROUPS), dtype=float)
    offsets = {"aligned": -0.13, "orthogonal": 0.13}
    markers = {"aligned": "o", "orthogonal": "s"}
    for relation in ("aligned", "orthogonal"):
        for idx, group in enumerate(GROUPS):
            values = paired.loc[paired["sf_group"].eq(group), f"{relation}_long_minus_short"].to_numpy(float)
            jitter = np.linspace(-0.045, 0.045, len(values)) if len(values) else np.array([])
            axes[0].scatter(
                idx + offsets[relation] + jitter, values, s=18, marker=markers[relation],
                facecolor=COLORS[group] if relation == "aligned" else "none",
                edgecolor=COLORS[group], alpha=0.55, lw=0.7,
            )
            row = relation_summary[
                relation_summary["sf_quartile"].eq(group)
                & relation_summary["relation"].eq(relation)
            ].iloc[0]
            axes[0].errorbar(
                idx + offsets[relation], row["mean_long_minus_short_ssi"],
                yerr=row["sem_long_minus_short_ssi"], color="black",
                marker=markers[relation], mfc="black" if relation == "aligned" else "white",
                ms=5, capsize=3, lw=1.2, zorder=5,
            )
    for idx, group in enumerate(GROUPS):
        values = paired.loc[
            paired["sf_group"].eq(group), "aligned_minus_orthogonal_endpoint_change"
        ].to_numpy(float)
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) else np.array([])
        axes[1].scatter(idx + jitter, values, color=COLORS[group], s=22, alpha=0.58, edgecolor="white", lw=0.3)
        row = contrast_summary[contrast_summary["sf_quartile"].eq(group)].iloc[0]
        axes[1].errorbar(
            idx, row["mean_aligned_minus_orthogonal_endpoint_change"],
            yerr=row["sem_aligned_minus_orthogonal_endpoint_change"],
            color="black", marker="D", ms=5, capsize=3, lw=1.3, zorder=5,
        )
    for ax in axes:
        ax.axhline(0, color="0.45", ls=":", lw=0.9)
        ax.set_xticks(x, [g.replace("sf_", "").upper() for g in GROUPS])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="0.92", lw=0.7)
    axes[0].set(
        ylabel="long minus short SSI (bits/spike)",
        title="A. Relation-specific endpoint changes",
    )
    axes[1].set(
        ylabel="aligned minus orthogonal endpoint change",
        title="B. Within-unit relation contrast",
    )
    handles = [
        mlines.Line2D([], [], color="black", marker="o", ls="none", label="aligned"),
        mlines.Line2D([], [], color="black", marker="s", mfc="white", ls="none", label="orthogonal"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=8)
    fig.suptitle(
        "Paired unit audit of contour relation effects",
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.4)
    save_figure(fig, out_dir, "checkpoint_05_paired_relation_endpoint_audit")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    matched_dir = Path(args.matched_dir)
    orthogonal_dir = Path(args.orthogonal_dir)
    summaries = {
        "aligned": pd.read_csv(matched_dir / "contour_matched_group_summary.csv"),
        "orthogonal": pd.read_csv(orthogonal_dir / "contour_orthogonal_group_summary.csv"),
    }
    curves = {
        "aligned": pd.read_csv(matched_dir / "contour_matched_unit_curves.csv"),
        "orthogonal": pd.read_csv(orthogonal_dir / "contour_orthogonal_unit_curves.csv"),
    }
    context = pd.read_csv(matched_dir / "trace_path_context_windows.csv")
    context_orthogonal = pd.read_csv(orthogonal_dir / "trace_path_context_windows.csv")
    if not context.equals(context_orthogonal):
        raise ValueError("Matched and orthogonal checkpoints use different trace-path context windows")
    for directory in (matched_dir, orthogonal_dir):
        validation = json.loads((directory / "historical_contract_validation.json").read_text())
        if not validation.get("passed", False):
            raise ValueError(f"Upstream historical reconstruction did not pass: {directory}")

    make_overlay_figures(summaries, context, args.out_dir)
    paired = paired_endpoint_table(curves)
    relation_summary, contrast_summary = summarize_paired(paired)
    examples = select_examples(paired)
    make_paired_figure(paired, relation_summary, contrast_summary, args.out_dir)

    paired.to_csv(args.out_dir / "paired_unit_relation_endpoint_changes.csv", index=False)
    relation_summary.to_csv(args.out_dir / "paired_relation_change_summary.csv", index=False)
    contrast_summary.to_csv(args.out_dir / "paired_aligned_minus_orthogonal_summary.csv", index=False)
    examples.to_csv(args.out_dir / "paired_relation_algorithmic_example_units.csv", index=False)
    context.to_csv(args.out_dir / "trace_path_context_windows.csv", index=False)

    counts = paired.groupby("sf_group")["unit_index"].nunique().reindex(GROUPS, fill_value=0).to_dict()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_05_complete",
        "source_test": "PDF page 14: contour-aligned versus contour-orthogonal low/high-SF overlay",
        "exact_replacement": "Q1 versus Q4 overlay; all quartiles and paired within-unit contrasts retained as audits",
        "selection_contract": {
            "aligned_max_deg": MATCH_MAX_DEG,
            "orthogonal_min_deg": ORTHOGONAL_MIN_DEG,
            "minimum_orientation_selectivity_index": MIN_OSI,
            "minimum_images_per_relation_per_unit": MIN_MATCHED_IMAGES,
        },
        "paired_unit_counts": counts,
        "upstream_sources": {
            "matched_summary": file_identity(matched_dir / "contour_matched_group_summary.csv"),
            "orthogonal_summary": file_identity(orthogonal_dir / "contour_orthogonal_group_summary.csv"),
            "matched_historical_validation": file_identity(matched_dir / "historical_contract_validation.json"),
            "orthogonal_historical_validation": file_identity(orthogonal_dir / "historical_contract_validation.json"),
        },
        "artifacts": {
            "exact_page_14_analog": "014_phase2_contour_aligned_vs_orthogonal_sf_q1_q4_curves.{png,pdf,svg}",
            "all_quartile_overlay": "checkpoint_05_aligned_vs_orthogonal_all_quartiles.{png,pdf,svg}",
            "paired_endpoint_audit": "checkpoint_05_paired_relation_endpoint_audit.{png,pdf,svg}",
            "algorithmic_examples": "paired_relation_algorithmic_example_units.csv",
        },
        "not_run": "No targeted relation-contrast map drill-down, trace-component conditioning, or later test was run.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = []
    for row in contrast_summary.itertuples(index=False):
        lines.append(
            f"- {LABELS[row.sf_quartile]}: n={row.n_paired_units}; mean aligned-minus-orthogonal endpoint "
            f"contrast={row.mean_aligned_minus_orthogonal_endpoint_change:+.4f} +/- "
            f"{row.sem_aligned_minus_orthogonal_endpoint_change:.4f} bits/spike; median="
            f"{row.median_aligned_minus_orthogonal_endpoint_change:+.4f}."
        )
    readme = f"""# Checkpoint 5: aligned versus orthogonal SF quartiles

This reproduces the old page-14 overlay using Q1 versus Q4 and retains all SF
quartiles in a companion overlay. Both upstream relation summaries exactly
reproduce their saved historical low/high-SF contracts.

## Paired relation contrasts

{chr(10).join(lines)}

The paired contrast is each unit's long-minus-short change on aligned image
windows minus its long-minus-short change on orthogonal windows. It is a
difference of differences, not a ratio. The algorithmic example table records
the largest aligned advantage, largest orthogonal advantage, and median
relation contrast in each quartile for a possible map-level follow-up.

No targeted relation-contrast map drill-down or trace-component analysis was
run here.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(relation_summary.to_string(index=False))
    print(contrast_summary.to_string(index=False))
    print(examples.to_string(index=False))


if __name__ == "__main__":
    main()
