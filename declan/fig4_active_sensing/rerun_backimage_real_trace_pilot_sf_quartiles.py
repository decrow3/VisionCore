#!/usr/bin/env python3
"""Checkpoint 1: rerun the real-trace pilot SF test with new SF quartiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
DEFAULT_ASSIGNMENTS = (
    ROOT
    / "outputs/fig4_active_sensing/rr100_sf_quartile_iteration_checks_v1/"
    "sf_quartile_unit_assignments.csv"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/fig4_active_sensing/backimage_real_trace_sf_quartile_checks_v1/"
    "checkpoint_01_pilot_sf_by_trace_path"
)
QUARTILE_ORDER = ("sf_q1", "sf_q2", "sf_q3", "sf_q4")
QUARTILE_LABELS = {
    "sf_q1": "SF Q1 (lowest)",
    "sf_q2": "SF Q2",
    "sf_q3": "SF Q3",
    "sf_q4": "SF Q4 (highest)",
}
QUARTILE_COLORS = {
    "sf_q1": "#46327E",
    "sf_q2": "#2A788E",
    "sf_q3": "#2FB47C",
    "sf_q4": "#BDDF26",
}
OLD_ORDER = ("low_sf", "middle_sf", "high_sf")
OLD_LABELS = {"low_sf": "old low SF", "middle_sf": "old middle SF", "high_sf": "old high SF"}
OLD_COLORS = {"low_sf": "#2A9D8F", "middle_sf": "#4C78A8", "high_sf": "#B279A2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-drift-bins", type=int, default=8)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def microsaccade_mask(table: pd.DataFrame) -> pd.Series:
    for key in ("rendered_n_microsaccade_events", "n_microsaccade_events", "source_n_microsaccade_events"):
        if key in table.columns:
            return pd.to_numeric(table[key], errors="coerce").fillna(0).gt(0)
    return pd.Series(False, index=table.index)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_dir = Path(args.matrix_dir)
    ssi = np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r")
    movie = pd.read_csv(matrix_dir / "movie_feature_table.csv")
    trace = pd.read_csv(matrix_dir / "trace_feature_table.csv")
    unit = pd.read_csv(matrix_dir / "unit_feature_table.csv")
    assignments = pd.read_csv(args.assignments_csv)
    if ssi.shape != (len(movie), len(unit)):
        raise ValueError(f"SSI shape {ssi.shape} does not match movie/unit tables {(len(movie), len(unit))}")
    required = {"rr100_index", "model_valid", "sf_quartile", "preferred_sf_cpd", "preferred_tf_hz"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Assignment table is missing {sorted(missing)}")
    if assignments["rr100_index"].duplicated().any() or len(assignments) != len(unit):
        raise ValueError("Assignment table must contain one row for every RR100 unit")
    assignment_cols = [
        "rr100_index", "model_valid", "sf_quartile", "sf_quartile_label",
        "preferred_sf_cpd", "preferred_tf_hz", "joint_parametric_surface_r2",
    ]
    merged = unit.merge(
        assignments[assignment_cols],
        left_on="unit_index",
        right_on="rr100_index",
        how="left",
        validate="one_to_one",
    )
    if merged["rr100_index"].isna().any():
        raise ValueError("Some SSI units have no new SF-quartile assignment row")
    return ssi, movie, trace, unit, merged


def trace_index_column(trace: pd.DataFrame) -> str:
    return "trace_bank_index" if "trace_bank_index" in trace.columns else "trace_index"


def build_per_trace_table(
    ssi: np.ndarray,
    movie: pd.DataFrame,
    trace: pd.DataFrame,
    units: pd.DataFrame,
) -> pd.DataFrame:
    movie_trace = pd.to_numeric(movie["trace_index"], errors="raise").astype(int)
    trace_id_col = trace_index_column(trace)
    trace_meta = trace.copy()
    trace_meta["trace_index"] = pd.to_numeric(trace_meta[trace_id_col], errors="raise").astype(int)
    trace_meta["has_microsaccade"] = microsaccade_mask(trace_meta).to_numpy(bool)
    keep_cols = [
        "trace_index",
        "rendered_path_length_arcmin",
        "has_microsaccade",
    ]
    for col in ("rendered_diffusion_constant_arcmin2_s", "rendered_rms_displacement_arcmin"):
        if col in trace_meta.columns:
            keep_cols.append(col)
    per_trace = trace_meta[keep_cols].copy()

    for group in QUARTILE_ORDER:
        ids = units.loc[units["sf_quartile"].eq(group) & units["model_valid"].astype(bool), "unit_index"].astype(int).to_numpy()
        if len(ids) == 0:
            raise ValueError(f"No valid units assigned to {group}")
        movie_mean = np.nanmean(np.asarray(ssi[:, ids], dtype=np.float64), axis=1)
        trace_mean = pd.DataFrame({"trace_index": movie_trace, "value": movie_mean}).groupby("trace_index", sort=False)["value"].mean()
        per_trace[f"{group}_mean_ssi_bits_per_spike"] = per_trace["trace_index"].map(trace_mean)

    for group in OLD_ORDER:
        ids = units.loc[units["sf_group"].eq(group), "unit_index"].astype(int).to_numpy()
        movie_mean = np.nanmean(np.asarray(ssi[:, ids], dtype=np.float64), axis=1)
        trace_mean = pd.DataFrame({"trace_index": movie_trace, "value": movie_mean}).groupby("trace_index", sort=False)["value"].mean()
        per_trace[f"old_{group}_mean_ssi_bits_per_spike"] = per_trace["trace_index"].map(trace_mean)
    return per_trace.sort_values("rendered_path_length_arcmin").reset_index(drop=True)


def drift_bin_summary(per_trace: pd.DataFrame, n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    drift = per_trace[~per_trace["has_microsaccade"]].copy()
    drift["path_bin"] = pd.qcut(drift["rendered_path_length_arcmin"], q=n_bins, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    trends: list[dict[str, Any]] = []
    for group in QUARTILE_ORDER:
        col = f"{group}_mean_ssi_bits_per_spike"
        for path_bin, sub in drift.groupby("path_bin", sort=True):
            values = sub[col].to_numpy(float)
            rows.append(
                {
                    "sf_quartile": group,
                    "path_bin": int(path_bin),
                    "path_arcmin_mean": float(sub["rendered_path_length_arcmin"].mean()),
                    "path_arcmin_min": float(sub["rendered_path_length_arcmin"].min()),
                    "path_arcmin_max": float(sub["rendered_path_length_arcmin"].max()),
                    "mean_ssi_bits_per_spike": float(np.nanmean(values)),
                    "sem_ssi_bits_per_spike": float(np.nanstd(values, ddof=1) / math.sqrt(np.isfinite(values).sum())),
                    "n_traces": int(len(sub)),
                }
            )
        valid = drift[["rendered_path_length_arcmin", col]].dropna()
        rho = float(valid.corr(method="spearman").iloc[0, 1])
        slope, intercept = np.polyfit(valid["rendered_path_length_arcmin"], valid[col], deg=1)
        trends.append(
            {
                "sf_quartile": group,
                "n_drift_traces": int(len(valid)),
                "spearman_path_vs_ssi": rho,
                "linear_slope_bits_per_spike_per_arcmin": float(slope),
                "linear_intercept_bits_per_spike": float(intercept),
                "first_bin_mean_ssi": rows[-n_bins]["mean_ssi_bits_per_spike"],
                "last_bin_mean_ssi": rows[-1]["mean_ssi_bits_per_spike"],
                "last_minus_first_bin_mean_ssi": rows[-1]["mean_ssi_bits_per_spike"] - rows[-n_bins]["mean_ssi_bits_per_spike"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(trends)


def plot_exact_quartile_analog(per_trace: pd.DataFrame, units: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ms = per_trace["has_microsaccade"].to_numpy(bool)
    for group in QUARTILE_ORDER:
        col = f"{group}_mean_ssi_bits_per_spike"
        n_units = int((units["sf_quartile"].eq(group) & units["model_valid"].astype(bool)).sum())
        ax.plot(
            per_trace["rendered_path_length_arcmin"],
            per_trace[col],
            color=QUARTILE_COLORS[group],
            lw=1.6,
            alpha=0.88,
            label=f"{QUARTILE_LABELS[group]} (n={n_units})",
        )
        ax.scatter(
            per_trace.loc[ms, "rendered_path_length_arcmin"],
            per_trace.loc[ms, col],
            s=18,
            facecolors="white",
            edgecolors=QUARTILE_COLORS[group],
            linewidths=0.8,
            zorder=3,
        )
    ax.set_xlabel("Trace path length (arcmin)")
    ax.set_ylabel("Mean unit SSI (bits/spike)")
    ax.set_title("New SF-quartile information across real trace scale")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, out_dir, "002_pilot_sf_quartile_ssi_vs_trace_path_length")


def plot_lines(
    ax: plt.Axes,
    per_trace: pd.DataFrame,
    *,
    groups: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    column_prefix: str,
    title: str,
) -> None:
    ms = per_trace["has_microsaccade"].to_numpy(bool)
    for group in groups:
        col = f"{column_prefix}{group}_mean_ssi_bits_per_spike"
        ax.plot(per_trace["rendered_path_length_arcmin"], per_trace[col], color=colors[group], lw=1.35, alpha=0.83, label=labels[group])
        ax.scatter(
            per_trace.loc[ms, "rendered_path_length_arcmin"], per_trace.loc[ms, col],
            s=12, facecolors="white", edgecolors=colors[group], linewidths=0.65, zorder=3,
        )
    ax.set(xlabel="trace path length (arcmin)", ylabel="mean unit SSI (bits/spike)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_audit_sheet(
    per_trace: pd.DataFrame,
    drift_bins: pd.DataFrame,
    units: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    plot_lines(
        axes[0, 0], per_trace, groups=QUARTILE_ORDER, labels=QUARTILE_LABELS,
        colors=QUARTILE_COLORS, column_prefix="", title="A. Exact page-2 analog: new SF quartiles",
    )
    ax = axes[0, 1]
    for group in QUARTILE_ORDER:
        sub = drift_bins[drift_bins["sf_quartile"].eq(group)]
        ax.errorbar(
            sub["path_arcmin_mean"], sub["mean_ssi_bits_per_spike"], yerr=sub["sem_ssi_bits_per_spike"],
            color=QUARTILE_COLORS[group], marker="o", ms=3.5, lw=1.7, capsize=1.5, label=QUARTILE_LABELS[group],
        )
    ax.set(xlabel="drift-only trace path length (arcmin)", ylabel="mean unit SSI (bits/spike)")
    ax.set_title("B. Drift-only quantile bins", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)

    plot_lines(
        axes[1, 0], per_trace, groups=OLD_ORDER, labels=OLD_LABELS,
        colors=OLD_COLORS, column_prefix="old_", title="C. Historical labels on the same traces",
    )
    ax = axes[1, 1]
    cross = pd.crosstab(units["sf_group"], units["sf_quartile"], dropna=False)
    columns = [q for q in QUARTILE_ORDER if q in cross.columns] + (["invalid_model"] if "invalid_model" in cross.columns else [])
    rows = [q for q in OLD_ORDER if q in cross.index]
    cross = cross.reindex(index=rows, columns=columns, fill_value=0)
    image = ax.imshow(cross.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    for i in range(cross.shape[0]):
        for j in range(cross.shape[1]):
            ax.text(j, i, str(int(cross.iloc[i, j])), ha="center", va="center", color="black", fontsize=8)
    ax.set_xticks(range(len(columns)), [q.replace("sf_", "").replace("invalid_model", "invalid") for q in columns])
    ax.set_yticks(range(len(rows)), [OLD_LABELS[q] for q in rows])
    ax.set(xlabel="new assignment", ylabel="historical assignment")
    ax.set_title("D. Old-to-new unit crosswalk", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="units")

    fig.suptitle("BackImage real-trace pilot: SF regrouping checkpoint", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(
        0.02, 0.935,
        "Open circles mark traces containing a detected microsaccade. Absolute SSI is averaged across units in each group; no stabilized baseline is used in this pilot test.",
        fontsize=8.3, color="0.35",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.5, w_pad=2.3)
    save_figure(fig, out_dir, "checkpoint_01_pilot_sf_quartile_audit")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    ssi, movie, trace, _unit, units = load_inputs(args)
    per_trace = build_per_trace_table(ssi, movie, trace, units)
    drift_bins, trends = drift_bin_summary(per_trace, int(args.n_drift_bins))
    plot_exact_quartile_analog(per_trace, units, args.out_dir)
    plot_audit_sheet(per_trace, drift_bins, units, args.out_dir)

    assignments_used = units[
        [
            "unit_index", "unit_label", "sf_group", "sf_group_label", "model_valid", "sf_quartile",
            "sf_quartile_label", "preferred_sf_cpd", "preferred_tf_hz", "joint_parametric_surface_r2",
        ]
    ].copy()
    assignments_used.to_csv(args.out_dir / "unit_assignments_used.csv", index=False)
    per_trace.to_csv(args.out_dir / "pilot_sf_quartile_ssi_by_trace.csv", index=False)
    drift_bins.to_csv(args.out_dir / "pilot_sf_quartile_drift_bin_summary.csv", index=False)
    trends.to_csv(args.out_dir / "pilot_sf_quartile_drift_trends.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checkpoint_01_complete",
        "source_test": "PDF page 2: pilot SF-group information across trace scale",
        "source_generator": "declan/active_sensing_movie_information/plot_backimage_real_trace_ssi_matrix_pilot.py",
        "estimand": "for each trace, mean SSI across images and then across valid units within each new SF quartile",
        "baseline": "none; absolute SSI only",
        "matrix_dir": str(Path(args.matrix_dir).resolve()),
        "assignments": file_identity(Path(args.assignments_csv)),
        "ssi_matrix": file_identity(Path(args.matrix_dir) / "ssi_matrix.npy"),
        "n_movies": int(len(movie)),
        "n_traces": int(len(per_trace)),
        "n_microsaccade_traces": int(per_trace["has_microsaccade"].sum()),
        "n_valid_units": int(units["model_valid"].sum()),
        "n_invalid_units_excluded": int((~units["model_valid"].astype(bool)).sum()),
        "quartile_counts": units.loc[units["model_valid"].astype(bool), "sf_quartile"].value_counts().sort_index().to_dict(),
        "n_drift_bins": int(args.n_drift_bins),
        "artifacts": {
            "exact_page_2_analog": "002_pilot_sf_quartile_ssi_vs_trace_path_length.{png,pdf,svg}",
            "audit_sheet": "checkpoint_01_pilot_sf_quartile_audit.{png,pdf,svg}",
            "unit_assignments": "unit_assignments_used.csv",
            "per_trace_values": "pilot_sf_quartile_ssi_by_trace.csv",
            "drift_bin_summary": "pilot_sf_quartile_drift_bin_summary.csv",
            "drift_trends": "pilot_sf_quartile_drift_trends.csv",
        },
        "not_run": "No phase-2 conditioning, contour relation, component-path, or Figure 4 panels were regenerated.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    trend_lines = []
    for row in trends.itertuples(index=False):
        label = QUARTILE_LABELS[row.sf_quartile]
        trend_lines.append(
            f"- {label}: Spearman rho = {row.spearman_path_vs_ssi:.3f}; "
            f"first-to-last drift-bin change = {row.last_minus_first_bin_mean_ssi:+.4f} bits/spike."
        )
    readme = f"""# Checkpoint 1: pilot SF quartiles across trace scale

This is the page-2 analog from the prior test collection, regenerated using the
new parametric-fit SF quartiles. It estimates mean absolute SSI for each trace,
averaged over images and over the valid units in a quartile. The 15 invalid fits
are excluded rather than imputed. No stabilized baseline or phase conditioning
is used at this checkpoint.

## New grouping

- Q1: 22 units
- Q2: 21 units
- Q3: 21 units
- Q4: 21 units
- Invalid fits excluded: 15 units

## Drift-only trace-scale trends

{chr(10).join(trend_lines)}

The open-circle traces in the exact analog contain detected microsaccades and
are shown for context, but the numerical trend audit above uses the 800
drift-only traces.

## Interpretation boundary

This checkpoint establishes how absolute pilot SSI varies with trace path after
the new unit split. It does not test stabilized-vs-moving gain, phase-2
conditioning, contour relations, component paths, or any final Figure 4 claim.
Those analyses have not yet been regenerated.

See `manifest.json` for source identities and `pilot_sf_quartile_drift_trends.csv`
for the full-precision values.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(trends.to_string(index=False))


if __name__ == "__main__":
    main()
