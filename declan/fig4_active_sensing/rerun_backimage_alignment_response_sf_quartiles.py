#!/usr/bin/env python3
"""Checkpoint 6: page-15 aligned-versus-orthogonal response controls with new SF quartiles."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.rerun_backimage_real_trace_contour_matched_sf_quartiles import (
    COLORS,
    DEFAULT_ASSIGNMENTS,
    DEFAULT_MATRIX_DIR,
    GROUPS,
    LABELS,
    MATCH_MAX_DEG,
    MIN_OSI,
    ORTHOGONAL_MIN_DEG,
    ROOT,
    configure_matplotlib,
    file_identity,
    save_figure,
)


DEFAULT_OUT = ROOT / (
    "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/"
    "checkpoint_06_alignment_response_sf_quartiles"
)
RELATIONS = ("aligned_preferred", "orthogonal")
RELATION_LABELS = {"aligned_preferred": "preferred/aligned", "orthogonal": "orthogonal"}
METRICS = ("mean_rate", "expected_spikes", "ssi")
MOVING_COLUMNS = tuple(f"moving_{metric}" for metric in METRICS)
STABILIZED_COLUMNS = tuple(f"stabilized_{metric}" for metric in METRICS)
METRIC_LABELS = {
    "mean_rate": "mean rate",
    "expected_spikes": "expected spikes/window",
    "ssi": "SSI (bits/spike)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def axis_delta_deg(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs(((np.asarray(a, dtype=float) - float(b) + 90.0) % 180.0) - 90.0)


def build_per_unit_response(
    movie: pd.DataFrame,
    image: pd.DataFrame,
    unit: pd.DataFrame,
    moving_arrays: dict[str, np.ndarray],
    stabilized_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    image_axes = image["image_edge_axis_deg"].to_numpy(float)
    image_ids = image["image_index"].astype(int).to_numpy()
    movie_image_ids = movie["image_index"].astype(int).to_numpy()
    rows = []
    for row in unit.itertuples(index=False):
        unit_index = int(row.unit_index)
        delta = axis_delta_deg(image_axes, float(row.prior_preferred_orientation_deg))
        relation_masks = {
            "aligned_preferred": delta <= MATCH_MAX_DEG,
            "orthogonal": delta >= ORTHOGONAL_MIN_DEG,
        }
        for relation, image_mask in relation_masks.items():
            selected_images = image_ids[image_mask]
            movie_mask = np.isin(movie_image_ids, selected_images)
            record = {
                "unit_index": unit_index,
                "unit_label": str(row.unit_label),
                "prior_preferred_orientation_deg": float(row.prior_preferred_orientation_deg),
                "prior_orientation_selectivity_index": float(row.prior_orientation_selectivity_index),
                "historical_sf_group": str(row.sf_group),
                "relation": relation,
                "n_images": int(image_mask.sum()),
                "n_movies": int(movie_mask.sum()),
            }
            for metric in METRICS:
                record[f"moving_{metric}"] = float(np.mean(moving_arrays[metric][movie_mask, unit_index]))
                record[f"stabilized_{metric}"] = float(np.mean(stabilized_arrays[metric][image_mask, unit_index]))
            rows.append(record)
    return pd.DataFrame(rows)


def attach_quartiles(per_unit: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rr100_index", "model_valid", "sf_quartile", "sf_quartile_label",
        "preferred_sf_cpd", "preferred_tf_hz", "joint_parametric_surface_r2",
    ]
    out = per_unit.merge(
        assignments[columns], left_on="unit_index", right_on="rr100_index",
        how="left", validate="many_to_one",
    )
    return out[
        out["model_valid"].fillna(False).astype(bool)
        & out["sf_quartile"].isin(GROUPS)
        & out["prior_orientation_selectivity_index"].ge(MIN_OSI)
    ].copy()


def summarize_long(per_unit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for relation in RELATIONS:
            sub = per_unit[
                per_unit["sf_quartile"].eq(group) & per_unit["relation"].eq(relation)
            ]
            for condition, columns in (("moving", MOVING_COLUMNS), ("stabilized", STABILIZED_COLUMNS)):
                for column in columns:
                    metric = column.removeprefix(f"{condition}_")
                    values = sub[column].dropna().to_numpy(float)
                    rows.append(
                        {
                            "sf_quartile": group, "relation": relation,
                            "condition": condition, "metric": metric,
                            "n_units": int(len(values)), "mean": float(np.mean(values)),
                            "sem": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
                            "median": float(np.median(values)),
                        }
                    )
    return pd.DataFrame(rows)


def paired_wide(per_unit: pd.DataFrame) -> pd.DataFrame:
    value_columns = list(MOVING_COLUMNS + STABILIZED_COLUMNS)
    wide = per_unit.pivot(
        index=["unit_index", "unit_label", "sf_quartile", "preferred_sf_cpd", "preferred_tf_hz"],
        columns="relation", values=value_columns,
    )
    wide.columns = [f"{value}_{relation}" for value, relation in wide.columns]
    wide = wide.reset_index()
    for condition in ("moving", "stabilized"):
        for metric in METRICS:
            wide[f"{condition}_{metric}_aligned_minus_orthogonal"] = (
                wide[f"{condition}_{metric}_aligned_preferred"]
                - wide[f"{condition}_{metric}_orthogonal"]
            )
    return wide


def summarize_contrasts(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        sub = wide[wide["sf_quartile"].eq(group)]
        for condition in ("moving", "stabilized"):
            for metric in METRICS:
                column = f"{condition}_{metric}_aligned_minus_orthogonal"
                values = sub[column].dropna().to_numpy(float)
                rows.append(
                    {
                        "sf_quartile": group, "condition": condition, "metric": metric,
                        "n_units": int(len(values)), "mean_aligned_minus_orthogonal": float(np.mean(values)),
                        "sem_aligned_minus_orthogonal": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
                        "median_aligned_minus_orthogonal": float(np.median(values)),
                        "fraction_aligned_greater": float(np.mean(values > 0)),
                    }
                )
    return pd.DataFrame(rows)


def leave_one_out_sensitivity(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        sub = wide[wide["sf_quartile"].eq(group)]
        for metric in METRICS:
            column = f"moving_{metric}_aligned_minus_orthogonal"
            full_mean = float(sub[column].mean())
            candidates = []
            for row in sub.itertuples(index=False):
                leave_out = float(sub.loc[sub["unit_index"].ne(row.unit_index), column].mean())
                candidates.append((abs(leave_out - full_mean), row, leave_out))
            _, influential, leave_out = max(candidates, key=lambda item: item[0])
            rows.append(
                {
                    "sf_quartile": group, "metric": metric, "n_units": int(len(sub)),
                    "full_mean_aligned_minus_orthogonal": full_mean,
                    "most_influential_unit_index": int(influential.unit_index),
                    "most_influential_unit_label": str(influential.unit_label),
                    "most_influential_unit_contrast": float(getattr(influential, column)),
                    "mean_excluding_most_influential_unit": leave_out,
                }
            )
    return pd.DataFrame(rows)


def validate_historical(per_unit: pd.DataFrame, matrix_dir: Path) -> dict:
    saved_path = matrix_dir / (
        "phase1_phase2_conditioning_v1/phase2_alignment_response_per_unit_by_image_subset_long.csv"
    )
    saved = pd.read_csv(saved_path)
    saved = saved[saved["subset"].eq("all_images")].copy()
    reconstructed = per_unit[
        per_unit["historical_sf_group"].isin(["low_sf", "high_sf"])
        & per_unit["prior_orientation_selectivity_index"].ge(MIN_OSI)
    ].copy()
    reconstructed["sf_group"] = reconstructed["historical_sf_group"]
    relation_map = {"aligned_preferred": "aligned_preferred", "orthogonal": "orthogonal"}
    reconstructed["relation"] = reconstructed["relation"].map(relation_map)
    keys = ["unit_index", "sf_group", "relation"]
    columns = ["n_images", "n_movies"] + list(MOVING_COLUMNS + STABILIZED_COLUMNS)
    compare = reconstructed.merge(saved[keys + columns], on=keys, suffixes=("_new", "_saved"), validate="one_to_one")
    diffs = {
        column: float(np.max(np.abs(compare[f"{column}_new"] - compare[f"{column}_saved"])))
        for column in columns
    }
    passed = len(compare) == len(saved) == len(reconstructed) and max(diffs.values()) < 1e-6
    return {
        "saved_table": file_identity(saved_path),
        "n_rows_reconstructed": int(len(reconstructed)),
        "n_rows_saved": int(len(saved)),
        "n_rows_matched": int(len(compare)),
        "max_absolute_differences": diffs,
        "passed": bool(passed),
    }


def plot_metric_panel(
    ax: plt.Axes,
    wide: pd.DataFrame,
    groups: tuple[str, ...],
    metric: str,
    *,
    title: str,
) -> None:
    x = np.arange(len(groups), dtype=float)
    width = 0.30
    for group_index, group in enumerate(groups):
        sub = wide[wide["sf_quartile"].eq(group)]
        aligned = sub[f"moving_{metric}_aligned_preferred"].to_numpy(float)
        orthogonal = sub[f"moving_{metric}_orthogonal"].to_numpy(float)
        for a, o in zip(aligned, orthogonal, strict=True):
            ax.plot(
                [group_index - width / 2, group_index + width / 2], [a, o],
                color=COLORS[group], alpha=0.18, lw=0.8, zorder=1,
            )
        for relation_index, (relation, values) in enumerate(
            (("aligned_preferred", aligned), ("orthogonal", orthogonal))
        ):
            xpos = group_index + (-width / 2 if relation_index == 0 else width / 2)
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            ax.bar(
                xpos, mean, width=width, color=COLORS[group],
                alpha=0.90 if relation == "aligned_preferred" else 0.48,
                edgecolor="0.25", linewidth=0.6, zorder=2,
            )
            ax.errorbar(xpos, mean, yerr=sem, color="0.2", capsize=3, lw=1.1, zorder=3)
    ax.set_xticks(x, [g.replace("sf_", "Q").replace("q", "") for g in groups])
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", color="0.92", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def make_figures(wide: pd.DataFrame, contrast_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
    for ax, metric in zip(axes, METRICS, strict=True):
        plot_metric_panel(ax, wide, ("sf_q1", "sf_q4"), metric, title=METRIC_LABELS[metric])
    axes[0].bar([], [], color="0.35", alpha=0.90, label="preferred/aligned")
    axes[0].bar([], [], color="0.35", alpha=0.48, label="orthogonal")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Preferred/aligned versus orthogonal image windows: response and SSI checks\n"
        "new SF extremes; moving real-trace movies; paired units",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.2)
    save_figure(fig, out_dir, "015_phase2_alignment_response_sf_q1_q4")

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
    for ax, metric in zip(axes, METRICS, strict=True):
        plot_metric_panel(ax, wide, GROUPS, metric, title=METRIC_LABELS[metric])
    axes[0].bar([], [], color="0.35", alpha=0.90, label="preferred/aligned")
    axes[0].bar([], [], color="0.35", alpha=0.48, label="orthogonal")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Checkpoint 6: all-quartile response-strength audit",
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.2)
    save_figure(fig, out_dir, "checkpoint_06_alignment_response_all_quartiles")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))
    x = np.arange(len(GROUPS))
    offsets = {"moving": -0.12, "stabilized": 0.12}
    for ax, metric in zip(axes, METRICS, strict=True):
        for condition in ("moving", "stabilized"):
            sub = contrast_summary[
                contrast_summary["condition"].eq(condition)
                & contrast_summary["metric"].eq(metric)
            ].set_index("sf_quartile").reindex(GROUPS)
            ax.errorbar(
                x + offsets[condition], sub["mean_aligned_minus_orthogonal"],
                yerr=sub["sem_aligned_minus_orthogonal"], marker="o" if condition == "moving" else "s",
                color="0.15" if condition == "moving" else "0.55", lw=1.5, capsize=3,
                label=condition,
            )
        ax.axhline(0, color="0.45", ls=":", lw=0.9)
        ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
        ax.set_ylabel(f"aligned minus orthogonal\n{METRIC_LABELS[metric]}")
        ax.set_title(METRIC_LABELS[metric], fontweight="bold")
        ax.grid(True, axis="y", color="0.92", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Moving and stabilized relation contrasts",
        x=0.02, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.4)
    save_figure(fig, out_dir, "checkpoint_06_moving_vs_stabilized_relation_contrasts")


def select_examples(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    column = "moving_mean_rate_aligned_minus_orthogonal"
    for group in GROUPS:
        sub = wide[wide["sf_quartile"].eq(group)].copy()
        median = float(sub[column].median())
        choices = (
            ("largest_aligned_rate_advantage", sub[column].idxmax(), "maximum moving aligned-minus-orthogonal mean rate"),
            ("largest_orthogonal_rate_advantage", sub[column].idxmin(), "minimum moving aligned-minus-orthogonal mean rate"),
            ("median_rate_contrast", (sub[column] - median).abs().idxmin(), "closest to median moving rate contrast"),
            ("largest_aligned_ssi_advantage", sub["moving_ssi_aligned_minus_orthogonal"].idxmax(), "maximum moving aligned-minus-orthogonal SSI"),
            ("largest_orthogonal_ssi_advantage", sub["moving_ssi_aligned_minus_orthogonal"].idxmin(), "minimum moving aligned-minus-orthogonal SSI"),
        )
        for role, index, criterion in choices:
            row = sub.loc[index]
            rows.append(
                {
                    "selection_method": "algorithmic", "selection_role": role,
                    "criterion": criterion, "sf_quartile": group,
                    "unit_index": int(row["unit_index"]), "unit_label": str(row["unit_label"]),
                    "moving_mean_rate_aligned_minus_orthogonal": float(row[column]),
                    "moving_ssi_aligned_minus_orthogonal": float(row["moving_ssi_aligned_minus_orthogonal"]),
                    "preferred_sf_cpd": float(row["preferred_sf_cpd"]),
                    "preferred_tf_hz": float(row["preferred_tf_hz"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    matrix_dir = Path(args.matrix_dir)
    movie = pd.read_csv(matrix_dir / "movie_feature_table.csv")
    image = pd.read_csv(matrix_dir / "image_feature_table.csv")
    unit_path = matrix_dir / "phase1_phase2_conditioning_v1/phase1_unit_summary_with_ssi.csv"
    unit = pd.read_csv(unit_path)
    moving_arrays = {
        "mean_rate": np.load(matrix_dir / "mean_rate_matrix.npy", mmap_mode="r"),
        "expected_spikes": np.load(matrix_dir / "expected_spikes_matrix.npy", mmap_mode="r"),
        "ssi": np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r"),
    }
    stabilized_arrays = {
        "mean_rate": np.load(matrix_dir / "stabilized_mean_rate_by_image.npy", mmap_mode="r"),
        "expected_spikes": np.load(matrix_dir / "stabilized_expected_spikes_by_image.npy", mmap_mode="r"),
        "ssi": np.load(matrix_dir / "stabilized_ssi_by_image.npy", mmap_mode="r"),
    }
    reconstructed = build_per_unit_response(movie, image, unit, moving_arrays, stabilized_arrays)
    validation = validate_historical(reconstructed, matrix_dir)
    if not validation["passed"]:
        raise ValueError(f"Historical response-table reconstruction failed: {validation}")
    assignments = pd.read_csv(args.assignments_csv)
    per_unit = attach_quartiles(reconstructed, assignments)
    summary = summarize_long(per_unit)
    wide = paired_wide(per_unit)
    contrast_summary = summarize_contrasts(wide)
    sensitivity = leave_one_out_sensitivity(wide)
    examples = select_examples(wide)
    make_figures(wide, contrast_summary, args.out_dir)

    per_unit.to_csv(args.out_dir / "alignment_response_per_unit_long.csv", index=False)
    wide.to_csv(args.out_dir / "alignment_response_per_unit_wide.csv", index=False)
    summary.to_csv(args.out_dir / "alignment_response_summary.csv", index=False)
    contrast_summary.to_csv(args.out_dir / "alignment_response_contrast_summary.csv", index=False)
    sensitivity.to_csv(args.out_dir / "alignment_response_leave_one_out_sensitivity.csv", index=False)
    examples.to_csv(args.out_dir / "alignment_response_algorithmic_example_units.csv", index=False)
    (args.out_dir / "historical_contract_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    counts = wide.groupby("sf_quartile")["unit_index"].nunique().reindex(GROUPS, fill_value=0).to_dict()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_06_complete",
        "source_test": "PDF page 15: preferred/aligned versus orthogonal response and SSI checks",
        "exact_replacement": "Q1 versus Q4; all quartiles retained in response and baseline audits",
        "estimand": "equal-unit mean across relation-selected image windows and all 1000 real traces per image",
        "selection_contract": {
            "image_axis": "image_edge_axis_deg", "aligned_max_deg": MATCH_MAX_DEG,
            "orthogonal_min_deg": ORTHOGONAL_MIN_DEG, "minimum_orientation_selectivity_index": MIN_OSI,
        },
        "quartile_unit_counts": counts,
        "sources": {
            "assignments": file_identity(Path(args.assignments_csv)),
            "mean_rate_matrix": file_identity(matrix_dir / "mean_rate_matrix.npy"),
            "expected_spikes_matrix": file_identity(matrix_dir / "expected_spikes_matrix.npy"),
            "ssi_matrix": file_identity(matrix_dir / "ssi_matrix.npy"),
        },
        "historical_contract_validation": validation,
        "not_run": "No targeted response-map drill-down or page-16 population summary was regenerated.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    moving = contrast_summary[contrast_summary["condition"].eq("moving")]
    lines = []
    for group in GROUPS:
        bits = []
        for metric in METRICS:
            row = moving[moving["sf_quartile"].eq(group) & moving["metric"].eq(metric)].iloc[0]
            bits.append(f"{metric} {row['mean_aligned_minus_orthogonal']:+.4f}")
        lines.append(f"- {LABELS[group]}: " + "; ".join(bits) + ".")
    readme = f"""# Checkpoint 6: preferred/aligned versus orthogonal response controls

This reproduces page 15 with Q1 versus Q4 and adds all-quartile and stabilized
baseline audits. The reconstructed historical low/high-SF per-unit table matches
the saved table; see `historical_contract_validation.json`.

## Moving aligned-minus-orthogonal contrasts

{chr(10).join(lines)}

Mean rate and expected spikes test response strength separately from SSI. The
stabilized comparison shows how much of each relation contrast predates retinal
motion. No targeted response-map drill-down or page-16 summary was run here.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(moving.to_string(index=False))
    print(examples.to_string(index=False))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
