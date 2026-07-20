#!/usr/bin/env python3
"""Plot a BackImage real-trace x image RR100 SSI pilot matrix."""

from __future__ import annotations

import argparse
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


SF_COLORS = {
    "low_sf": "#1f77b4",
    "middle_sf": "#8a8a8a",
    "high_sf": "#d62728",
}
DRIFT_COLOR = "#1f77b4"
MICROSACCADE_COLOR = "#d62728"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--fig-dir", type=Path, default=None)
    return parser.parse_args()


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": fig_dir / f"{stem}.png",
        "pdf": fig_dir / f"{stem}.pdf",
    }
    fig.savefig(paths["png"], dpi=220, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def microsaccade_mask(table: pd.DataFrame) -> np.ndarray:
    for key in ("rendered_n_microsaccade_events", "n_microsaccade_events", "source_n_microsaccade_events"):
        if key in table.columns:
            return pd.to_numeric(table[key], errors="coerce").fillna(0).to_numpy() > 0
    return np.zeros(table.shape[0], dtype=bool)


def load_pilot(pilot_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ssi = np.load(pilot_dir / "ssi_matrix.npy")
    population = np.load(pilot_dir / "population_ssi.npy")
    movie = pd.read_csv(pilot_dir / "movie_feature_table.csv")
    trace = pd.read_csv(pilot_dir / "trace_feature_table.csv")
    unit = pd.read_csv(pilot_dir / "unit_feature_table.csv")
    if ssi.shape[0] != movie.shape[0]:
        raise ValueError(f"SSI rows {ssi.shape[0]} do not match movie table rows {movie.shape[0]}.")
    if ssi.shape[1] != unit.shape[0]:
        raise ValueError(f"SSI columns {ssi.shape[1]} do not match unit table rows {unit.shape[0]}.")
    if population.shape[0] != movie.shape[0]:
        raise ValueError(f"Population SSI rows {population.shape[0]} do not match movie table rows {movie.shape[0]}.")
    return ssi, population, movie, trace, unit


def add_group_ssi(movie: pd.DataFrame, ssi: np.ndarray, population: np.ndarray, unit: pd.DataFrame) -> pd.DataFrame:
    out = movie.copy()
    out["population_ssi"] = population
    if "sf_group" not in unit.columns:
        out["all_units_mean_ssi"] = np.nanmean(ssi, axis=1)
        return out
    for group in ("low_sf", "middle_sf", "high_sf"):
        idx = unit.index[unit["sf_group"].astype(str) == group].to_numpy()
        if idx.size:
            out[f"{group}_mean_ssi"] = np.nanmean(ssi[:, idx], axis=1)
    return out


def trace_summary(movie_aug: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, tr in trace.reset_index(drop=True).iterrows():
        trace_index = int(tr.get("trace_bank_index", tr.name))
        sub = movie_aug[movie_aug["trace_index"].astype(int) == trace_index]
        row = {
            "trace_index": trace_index,
            "rendered_path_length_arcmin": float(tr["rendered_path_length_arcmin"]),
            "rendered_diffusion_constant_arcmin2_s": float(tr.get("rendered_diffusion_constant_arcmin2_s", np.nan)),
            "has_microsaccade": bool(microsaccade_mask(pd.DataFrame([tr]))[0]),
            "population_ssi_mean": float(sub["population_ssi"].mean()),
            "population_ssi_sem": float(sub["population_ssi"].sem()),
        }
        for col in [c for c in movie_aug.columns if c.endswith("_mean_ssi")]:
            row[col] = float(sub[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rendered_path_length_arcmin").reset_index(drop=True)


def plot_population_vs_trace_scale(movie_aug: pd.DataFrame, per_trace: pd.DataFrame, fig_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    mask = microsaccade_mask(movie_aug)
    x = pd.to_numeric(movie_aug["rendered_path_length_arcmin"], errors="coerce").to_numpy()
    y = pd.to_numeric(movie_aug["population_ssi"], errors="coerce").to_numpy()
    ax.scatter(x[~mask], y[~mask], s=13, color=DRIFT_COLOR, alpha=0.16, linewidths=0, label="drift-only movies")
    ax.scatter(x[mask], y[mask], s=16, color=MICROSACCADE_COLOR, alpha=0.25, linewidths=0, label="microsaccade movies")
    for has_ms, color, label in [
        (False, DRIFT_COLOR, "drift-only trace mean"),
        (True, MICROSACCADE_COLOR, "microsaccade trace mean"),
    ]:
        sub = per_trace[per_trace["has_microsaccade"] == has_ms]
        if sub.empty:
            continue
        ax.errorbar(
            sub["rendered_path_length_arcmin"],
            sub["population_ssi_mean"],
            yerr=sub["population_ssi_sem"].fillna(0),
            fmt="o",
            ms=4.5,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=1.5,
            label=label,
        )
    ax.set_xlabel("Trace path length (arcmin)")
    ax.set_ylabel("Population SSI (bits/spike)")
    ax.set_title("Population information across real trace scale")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, fig_dir, "population_ssi_vs_trace_path_length")


def plot_sf_groups(per_trace: pd.DataFrame, fig_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for group, color in SF_COLORS.items():
        col = f"{group}_mean_ssi"
        if col not in per_trace.columns:
            continue
        ax.plot(
            per_trace["rendered_path_length_arcmin"],
            per_trace[col],
            color=color,
            lw=1.8,
            alpha=0.9,
            label=group.replace("_", " "),
        )
        ms = per_trace["has_microsaccade"].to_numpy(dtype=bool)
        ax.scatter(
            per_trace.loc[ms, "rendered_path_length_arcmin"],
            per_trace.loc[ms, col],
            s=22,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            zorder=3,
        )
    ax.set_xlabel("Trace path length (arcmin)")
    ax.set_ylabel("Mean unit SSI (bits/spike)")
    ax.set_title("SF-group information across trace scale")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, fig_dir, "sf_group_ssi_vs_trace_path_length")


def plot_population_distribution(movie_aug: pd.DataFrame, fig_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    mask = microsaccade_mask(movie_aug)
    bins = np.linspace(
        float(np.nanpercentile(movie_aug["population_ssi"], 1)),
        float(np.nanpercentile(movie_aug["population_ssi"], 99)),
        34,
    )
    ax.hist(movie_aug.loc[~mask, "population_ssi"], bins=bins, density=True, histtype="step", lw=2.0, color=DRIFT_COLOR, label="drift-only")
    ax.hist(movie_aug.loc[mask, "population_ssi"], bins=bins, density=True, histtype="step", lw=2.0, color=MICROSACCADE_COLOR, label="microsaccade")
    ax.set_xlabel("Population SSI (bits/spike)")
    ax.set_ylabel("Density")
    ax.set_title("Population SSI by microsaccade contamination")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, fig_dir, "population_ssi_distribution_by_microsaccade")


def unit_order(unit: pd.DataFrame) -> np.ndarray:
    work = unit.copy()
    work["_unit_order"] = np.arange(work.shape[0])
    if "sf_group" in work.columns:
        sf_order = {"low_sf": 0, "middle_sf": 1, "high_sf": 2}
        work["_sf_order"] = work["sf_group"].astype(str).map(sf_order).fillna(9)
    else:
        work["_sf_order"] = 0
    sort_cols = ["_sf_order"]
    for col in (
        "dynamic_log_gaussian_marginal_sf_cpd",
        "dynamic_amp_weighted_sf_cpd",
        "static_rate_weighted_sf_cpd",
    ):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            sort_cols.append(col)
            break
    sort_cols.append("_unit_order")
    return work.sort_values(sort_cols, kind="mergesort")["_unit_order"].to_numpy(dtype=int)


def plot_unit_trace_heatmap(ssi: np.ndarray, movie: pd.DataFrame, trace: pd.DataFrame, unit: pd.DataFrame, fig_dir: Path) -> dict[str, str]:
    trace_indices = trace["trace_bank_index"].to_numpy(dtype=int) if "trace_bank_index" in trace.columns else np.arange(trace.shape[0])
    trace_order = trace.sort_values("rendered_path_length_arcmin").index.to_numpy()
    u_order = unit_order(unit)
    trace_unit = np.full((trace.shape[0], ssi.shape[1]), np.nan, dtype=np.float64)
    for row_idx, trace_index in enumerate(trace_indices):
        movie_mask = movie["trace_index"].astype(int).to_numpy() == int(trace_index)
        trace_unit[row_idx] = np.nanmean(ssi[movie_mask], axis=0)
    heat = trace_unit[trace_order][:, u_order].T
    vmin = float(np.nanpercentile(heat, 2))
    vmax = float(np.nanpercentile(heat, 98))
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    im = ax.imshow(heat, aspect="auto", interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    sorted_trace = trace.iloc[trace_order].reset_index(drop=True)
    ms = microsaccade_mask(sorted_trace)
    if ms.any():
        ax.scatter(np.flatnonzero(ms), np.full(int(ms.sum()), -1.7), marker="|", s=60, color=MICROSACCADE_COLOR, clip_on=False, label="microsaccade trace")
    ax.set_xlabel("Trace index sorted by path length")
    ax.set_ylabel("RR100 unit sorted by SF group")
    ax.set_title("Mean unit SSI for each trace")
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label("SSI (bits/spike)")
    if ms.any():
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    return save_figure(fig, fig_dir, "unit_by_trace_ssi_heatmap")


def main() -> None:
    args = parse_args()
    pilot_dir = Path(args.pilot_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir is not None else pilot_dir / "pilot_figures"
    ssi, population, movie, trace, unit = load_pilot(pilot_dir)
    movie_aug = add_group_ssi(movie, ssi, population, unit)
    movie_aug.to_csv(pilot_dir / "movie_feature_table_with_group_ssi.csv", index=False)
    per_trace = trace_summary(movie_aug, trace)
    per_trace.to_csv(pilot_dir / "trace_summary_with_ssi.csv", index=False)

    figures = {
        "population_ssi_vs_trace_path_length": plot_population_vs_trace_scale(movie_aug, per_trace, fig_dir),
        "sf_group_ssi_vs_trace_path_length": plot_sf_groups(per_trace, fig_dir),
        "population_ssi_distribution_by_microsaccade": plot_population_distribution(movie_aug, fig_dir),
        "unit_by_trace_ssi_heatmap": plot_unit_trace_heatmap(ssi, movie, trace, unit, fig_dir),
    }
    trace_ms = microsaccade_mask(trace)
    movie_ms = microsaccade_mask(movie_aug)
    summary = {
        "n_movies": int(movie.shape[0]),
        "n_images": int(movie["image_index"].nunique()),
        "n_traces": int(trace.shape[0]),
        "n_units": int(unit.shape[0]),
        "n_microsaccade_traces": int(trace_ms.sum()),
        "n_microsaccade_movies": int(movie_ms.sum()),
        "population_ssi_min": float(np.nanmin(population)),
        "population_ssi_median": float(np.nanmedian(population)),
        "population_ssi_max": float(np.nanmax(population)),
        "trace_path_length_arcmin_min": float(np.nanmin(trace["rendered_path_length_arcmin"])),
        "trace_path_length_arcmin_median": float(np.nanmedian(trace["rendered_path_length_arcmin"])),
        "trace_path_length_arcmin_max": float(np.nanmax(trace["rendered_path_length_arcmin"])),
        "figures": figures,
    }
    save_json(pilot_dir / "pilot_analysis_summary.json", summary)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
