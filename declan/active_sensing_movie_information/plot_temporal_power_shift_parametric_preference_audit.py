#!/usr/bin/env python3
"""Audit the RR100 parametric SF/TF preferences used by temporal-power figures."""

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

from declan.active_sensing_movie_information.analyze_temporal_remapping_sftf_power_explanation import (
    DEFAULT_PARAMETRIC_MODEL_CSV,
    DEFAULT_RUN_DIR,
    sf_group_for_values,
)


DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "map_first_power_shift_parametric_preference_audit_v1"
DEFAULT_DENSE_FIT_CSV = Path(__file__).resolve().parents[2] / (
    "outputs/active_sensing_movie_information/backimage_rr100_dense_sf_tf_speed_pref_groups_v1/"
    "cycle_valid_dense_sf_tf_fit_unit_summary.csv"
)
OLD_EXAMPLE_UNITS = (86, 92, 8)
PARAMETRIC_EXAMPLE_UNITS = (50, 14, 6)

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "grey": "#777777",
    "black": "#222222",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
GROUP_ORDER = ["low_sf", "middle_sf", "high_sf"]
GROUP_LABELS = {"low_sf": "Low SF", "middle_sf": "Middle SF", "high_sf": "High SF"}
GROUP_COLORS = {"low_sf": OKABE_ITO["blue"], "middle_sf": OKABE_ITO["green"], "high_sf": OKABE_ITO["orange"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dense-fit-csv", type=Path, default=DEFAULT_DENSE_FIT_CSV)
    parser.add_argument("--parametric-model-csv", type=Path, default=DEFAULT_PARAMETRIC_MODEL_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--old-example-units", type=str, default=",".join(str(v) for v in OLD_EXAMPLE_UNITS))
    parser.add_argument(
        "--parametric-example-units",
        type=str,
        default=",".join(str(v) for v in PARAMETRIC_EXAMPLE_UNITS),
    )
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def parse_units(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


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
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")


def load_audit_table(run_dir: Path, dense_fit_csv: Path, parametric_model_csv: Path) -> pd.DataFrame:
    retiming = pd.read_csv(
        run_dir / "retiming_unit_observations.csv",
        usecols=["unit_index", "unit_label", "sf_group", "preferred_sf_cpd", "preferred_sf_source_column"],
    ).drop_duplicates("unit_index")
    dense = pd.read_csv(dense_fit_csv)
    dense = dense[["unit_index", "fit_ok", "fit_pref_sf_cpd", "fit_pref_tf_hz", "fit_r2", "fit_edge_tf"]].copy()
    param = pd.read_csv(parametric_model_csv)
    param = param[
        [
            "rr100_index",
            "model_valid",
            "preferred_sf_cpd",
            "preferred_tf_hz",
            "sf_fit_r2",
            "tf_fit_r2",
            "joint_parametric_surface_r2",
        ]
    ].rename(
        columns={
            "rr100_index": "unit_index",
            "preferred_sf_cpd": "parametric_preferred_sf_cpd",
            "preferred_tf_hz": "parametric_preferred_tf_hz",
        }
    )
    out = retiming.rename(
        columns={
            "sf_group": "legacy_sf_group",
            "preferred_sf_cpd": "legacy_preferred_sf_cpd",
            "preferred_sf_source_column": "legacy_preferred_sf_source_column",
        }
    )
    out = out.merge(dense, on="unit_index", how="left")
    out = out.merge(param, on="unit_index", how="left")
    groups = sf_group_for_values(out["parametric_preferred_sf_cpd"])
    out["parametric_sf_group"] = groups["sf_group"]
    out["parametric_sf_group_label"] = groups["sf_group_label"]
    out["legacy_sf_group_label"] = out["legacy_sf_group"].map(lambda group: GROUP_LABELS.get(str(group), str(group)))
    return out


def example_table(audit: pd.DataFrame, old_units: list[int], param_units: list[int]) -> pd.DataFrame:
    rows = []
    for role, units in [("old example", old_units), ("parametric example", param_units)]:
        for unit in units:
            row = audit[audit["unit_index"].eq(int(unit))]
            if row.empty:
                rows.append({"example_role": role, "unit_index": int(unit), "note": "unit not found"})
                continue
            item = row.iloc[0].to_dict()
            item["example_role"] = role
            if role == "old example" and not bool(item.get("model_valid", False)):
                item["note"] = "not valid in parametric table"
            elif role == "old example":
                item["note"] = "legacy working example"
            else:
                item["note"] = "chosen from parametric low/middle/high bands"
            rows.append(item)
    columns = [
        "example_role",
        "unit_index",
        "unit_label",
        "legacy_sf_group",
        "legacy_preferred_sf_cpd",
        "fit_pref_tf_hz",
        "model_valid",
        "parametric_sf_group",
        "parametric_preferred_sf_cpd",
        "parametric_preferred_tf_hz",
        "joint_parametric_surface_r2",
        "note",
    ]
    return pd.DataFrame(rows)[columns]


def plot_audit(audit: pd.DataFrame, examples: pd.DataFrame, out_dir: Path, *, dpi: int) -> tuple[Path, Path]:
    valid = audit[audit["model_valid"].astype(bool)].copy()
    fig = plt.figure(figsize=(14.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92], width_ratios=[1.12, 1.05, 1.18])

    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "A")
    for group in GROUP_ORDER:
        sub = valid[valid["parametric_sf_group"].eq(group)]
        ax.scatter(
            sub["parametric_preferred_sf_cpd"],
            sub["parametric_preferred_tf_hz"],
            s=34,
            color=GROUP_COLORS[group],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.35,
            label=f"{GROUP_LABELS[group]} (n={len(sub)})",
        )
    for _, row in examples[examples["model_valid"].astype(bool)].iterrows():
        ax.scatter(
            row["parametric_preferred_sf_cpd"],
            row["parametric_preferred_tf_hz"],
            marker="*",
            s=180,
            color=OKABE_ITO["purple"] if row["example_role"] == "old example" else OKABE_ITO["black"],
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        ax.annotate(
            str(row["unit_label"]),
            xy=(float(row["parametric_preferred_sf_cpd"]), float(row["parametric_preferred_tf_hz"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parametric preferred SF (cycles/deg)")
    ax.set_ylabel("parametric preferred TF (Hz)")
    ax.set_title("New SF/TF preferences")
    ax.set_xticks([1, 2, 4, 8, 12], ["1", "2", "4", "8", "12"])
    ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32], ["0.5", "1", "2", "4", "8", "16", "32"])
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "B")
    for group in GROUP_ORDER:
        sub = valid[valid["parametric_sf_group"].eq(group)]
        ax.scatter(
            sub["legacy_preferred_sf_cpd"],
            sub["parametric_preferred_sf_cpd"],
            s=30,
            color=GROUP_COLORS[group],
            alpha=0.70,
            edgecolor="white",
            linewidth=0.3,
        )
    ax.plot([0.02, 12], [0.02, 12], color="#777777", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("legacy preferred SF used by retiming run")
    ax.set_ylabel("parametric preferred SF")
    ax.set_title("Old and new SF are not interchangeable")
    ax.set_xlim(0.02, 14)
    ax.set_ylim(0.8, 14)
    ax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10], ["0.03", "0.1", "0.3", "1", "3", "10"])
    ax.set_yticks([1, 2, 4, 8, 12], ["1", "2", "4", "8", "12"])

    ax = fig.add_subplot(gs[0, 2])
    add_panel_label(ax, "C")
    valid_for_cross = valid[valid["legacy_sf_group"].isin(GROUP_ORDER) & valid["parametric_sf_group"].isin(GROUP_ORDER)]
    counts = pd.crosstab(valid_for_cross["legacy_sf_group"], valid_for_cross["parametric_sf_group"]).reindex(
        index=GROUP_ORDER, columns=GROUP_ORDER, fill_value=0
    )
    image = ax.imshow(counts.to_numpy(dtype=float), cmap="cividis")
    for i, old_group in enumerate(GROUP_ORDER):
        for j, new_group in enumerate(GROUP_ORDER):
            ax.text(j, i, str(int(counts.loc[old_group, new_group])), ha="center", va="center", color="white", fontsize=12)
    ax.set_xticks(range(3), [GROUP_LABELS[g] for g in GROUP_ORDER], rotation=25, ha="right")
    ax.set_yticks(range(3), [GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_xlabel("parametric SF group")
    ax.set_ylabel("legacy SF group")
    ax.set_title("Group labels change")
    fig.colorbar(image, ax=ax, fraction=0.047, pad=0.02, label="units")

    ax = fig.add_subplot(gs[1, :])
    add_panel_label(ax, "D")
    ax.axis("off")
    header = (
        "role     unit   legacy SF / TF         parametric SF / TF       note\n"
        "--------------------------------------------------------------------------"
    )
    lines = [header]
    for _, row in examples.iterrows():
        legacy = f"{row['legacy_preferred_sf_cpd']:.3g} cpd / {row['fit_pref_tf_hz']:.3g} Hz"
        if bool(row.get("model_valid", False)):
            param = f"{row['parametric_preferred_sf_cpd']:.3g} cpd / {row['parametric_preferred_tf_hz']:.3g} Hz"
        else:
            param = "no valid parametric model"
        lines.append(
            f"{row['example_role']:<8} {row['unit_label']:<5} {legacy:<22} {param:<26} {row['note']}"
        )
    ax.text(
        0.01,
        0.97,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10.2,
        color="#222222",
    )
    ax.text(
        0.01,
        0.08,
        "Rule for this parametric branch: low = 1-2 cpd, middle = 2-4 cpd, high = 4+ cpd. "
        "Motion-induced TF is recomputed as across-contour speed x parametric preferred SF.",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#333333",
    )

    for panel in fig.axes:
        if panel.has_data():
            panel.grid(True, color="#e8e8e8", lw=0.65)
    fig.suptitle("RR100 temporal-power analysis: parametric preference audit", fontsize=15)
    png = out_dir / "rr100_temporal_power_parametric_preference_audit.png"
    pdf = out_dir / "rr100_temporal_power_parametric_preference_audit.pdf"
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    old_units = parse_units(str(args.old_example_units))
    param_units = parse_units(str(args.parametric_example_units))
    audit = load_audit_table(Path(args.run_dir), Path(args.dense_fit_csv), Path(args.parametric_model_csv))
    examples = example_table(audit, old_units, param_units)
    audit_path = out_dir / "rr100_temporal_power_parametric_preference_audit_table.csv"
    example_path = out_dir / "rr100_temporal_power_parametric_example_units.csv"
    audit.to_csv(audit_path, index=False)
    examples.to_csv(example_path, index=False)
    png, pdf = plot_audit(audit, examples, out_dir, dpi=int(args.dpi))
    write_json(
        out_dir / "rr100_temporal_power_parametric_preference_audit_metadata.json",
        {
            "analysis": "rr100_temporal_power_parametric_preference_audit",
            "run_dir": Path(args.run_dir),
            "dense_fit_csv": Path(args.dense_fit_csv),
            "parametric_model_csv": Path(args.parametric_model_csv),
            "out_dir": out_dir,
            "old_example_units": old_units,
            "parametric_example_units": param_units,
            "parametric_sf_group_rule": "low=1-2 cpd, middle=2-4 cpd, high=4+ cpd",
            "outputs": {"figure_png": png, "figure_pdf": pdf, "audit_table": audit_path, "example_table": example_path},
        },
    )
    print(f"Wrote {png}")
    print(f"Wrote {example_path}")


if __name__ == "__main__":
    main()
