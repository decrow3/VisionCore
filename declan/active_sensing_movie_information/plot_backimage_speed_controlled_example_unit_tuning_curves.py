#!/usr/bin/env python3
"""Plot example unit SF, TF, and speed tuning fits from the speed-controlled grating probe."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_speed_controlled_sf_tf_speed_tuning_fits_v1"
)
DEFAULT_OUT_DIR = DEFAULT_FIT_DIR / "example_unit_tuning_curves"
DIMENSIONS = [
    ("sf", "spatial frequency (cpd)"),
    ("tf", "temporal frequency (Hz)"),
    ("speed", "speed (deg/s)"),
]
GROUP_ORDER = ["high_speed_preferring", "low_speed_preferring"]
GROUP_LABELS = {
    "high_speed_preferring": "high-speed pref.",
    "low_speed_preferring": "low-speed pref.",
}
GROUP_COLORS = {
    "high_speed_preferring": "#1f77b4",
    "low_speed_preferring": "#d62728",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--speed-family", choices=("cycle_valid", "subcycle_control"), default="cycle_valid")
    parser.add_argument("--units", type=str, default="")
    parser.add_argument("--examples-per-group", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def log_gaussian(log_x: np.ndarray, baseline: float, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    return baseline + amplitude * np.exp(-0.5 * np.square((log_x - mu) / max(float(sigma), EPS)))


def parse_units(text: str) -> list[int]:
    if not str(text).strip():
        return []
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part.startswith("u"):
            part = part[1:]
        out.append(int(part))
    return out


def choose_units(fits: pd.DataFrame, *, family: str, examples_per_group: int) -> list[int]:
    sub = fits[(fits["speed_family"].eq(family)) & (fits["fit_ok"])].copy()
    wide = sub.pivot_table(
        index=["unit_index", "unit_label", "speed_pref_group", "speed_pref_label"],
        columns="dimension",
        values=["fit_r2", "fit_status", "fit_preferred_value"],
        aggfunc="first",
    ).reset_index()
    wide.columns = ["_".join([str(x) for x in col if str(x)]) for col in wide.columns]
    needed = ["fit_r2_sf", "fit_r2_tf", "fit_r2_speed"]
    for col in needed:
        if col not in wide.columns:
            wide[col] = np.nan
    wide["min_r2"] = wide[needed].min(axis=1)
    edge_cols = [c for c in ["fit_status_sf", "fit_status_tf", "fit_status_speed"] if c in wide.columns]
    edge_penalty = wide[edge_cols].isin(["lower_edge", "upper_edge"]).sum(axis=1) if edge_cols else 0
    wide["score"] = wide["min_r2"] - 0.12 * edge_penalty

    chosen: list[int] = []
    for group in GROUP_ORDER:
        group_rows = wide[wide["speed_pref_group"].eq(group)].copy()
        if group_rows.empty:
            continue
        if group == "high_speed_preferring" and "fit_preferred_value_speed" in group_rows:
            threshold = float(group_rows["fit_preferred_value_speed"].median())
            preferred = group_rows[group_rows["fit_preferred_value_speed"] >= threshold]
            if preferred.shape[0] >= examples_per_group:
                group_rows = preferred
        if group == "low_speed_preferring" and "fit_preferred_value_speed" in group_rows:
            threshold = float(group_rows["fit_preferred_value_speed"].median())
            preferred = group_rows[group_rows["fit_preferred_value_speed"] <= threshold]
            if preferred.shape[0] >= examples_per_group:
                group_rows = preferred
        chosen.extend(
            int(v)
            for v in group_rows.sort_values(["score", "min_r2"], ascending=False)["unit_index"]
            .head(int(examples_per_group))
            .to_list()
        )
    return chosen


def plot_examples(
    points: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    units: list[int],
    family: str,
    out_dir: Path,
    dpi: int,
) -> Path:
    n_rows = len(units)
    png = out_dir / f"{family}_example_unit_sf_tf_speed_tuning_curves.png"
    fig, axes = plt.subplots(n_rows, 3, figsize=(13.8, max(2.0 * n_rows, 4.2)), squeeze=False, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.065, top=0.9, hspace=0.62, wspace=0.3)
    fig.suptitle("Example unit tuning curves from speed-controlled gratings", y=0.975, fontsize=15)
    fig.text(
        0.5,
        0.94,
        "Cycle-valid family unless noted; points are RMS modulation amplitudes, curves are bounded log-Gaussian fits",
        ha="center",
        fontsize=10.5,
        color="0.35",
    )
    for row, unit in enumerate(units):
        unit_fits = fits[(fits["unit_index"].eq(unit)) & (fits["speed_family"].eq(family))]
        if unit_fits.empty:
            continue
        group = str(unit_fits["speed_pref_group"].iloc[0])
        color = GROUP_COLORS.get(group, "0.25")
        unit_label = str(unit_fits["unit_label"].iloc[0])
        group_label = GROUP_LABELS.get(group, group)
        for col, (dim, dim_label) in enumerate(DIMENSIONS):
            ax = axes[row, col]
            unit_points = points[
                (points["unit_index"].eq(unit))
                & (points["speed_family"].eq(family))
                & (points["dimension"].eq(dim))
            ].sort_values("stimulus_value")
            fit = unit_fits[unit_fits["dimension"].eq(dim)]
            if unit_points.empty:
                ax.axis("off")
                continue
            x = unit_points["stimulus_value"].to_numpy(dtype=float)
            log_x = unit_points["log2_stimulus_value"].to_numpy(dtype=float)
            y = unit_points["response_amp_rms_mean"].to_numpy(dtype=float)
            ax.plot(x, y, color=color, marker="o", lw=0, ms=4.8, alpha=0.9)
            if not fit.empty and bool(fit["fit_ok"].iloc[0]):
                rec = fit.iloc[0]
                grid = np.linspace(float(np.nanmin(log_x)), float(np.nanmax(log_x)), 240)
                pred = log_gaussian(
                    grid,
                    float(rec["fit_baseline"]),
                    float(rec["fit_amplitude"]),
                    float(rec["fit_preferred_log2_value"]),
                    float(rec["fit_sigma_log2"]),
                )
                ax.plot(2.0**grid, pred, color=color, lw=2.0)
                pref = float(rec["fit_preferred_value"])
                ax.axvline(pref, color=color, ls="--", lw=1.2, alpha=0.75)
                status = str(rec["fit_status"])
                r2 = float(rec["fit_r2"])
                title_suffix = f"pref={pref:.3g}, R2={r2:.2f}"
                if status != "interior":
                    title_suffix += f", {status}"
            else:
                title_suffix = "fit unavailable"
            ax.set_xscale("log", base=2)
            ax.grid(True, color="0.9")
            if row == 0:
                ax.set_title(dim_label)
            if col == 0:
                ax.set_ylabel(f"{unit_label}\n{group_label}\nRMS amp", color=color)
            ax.set_xlabel(title_suffix, fontsize=8.5)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return png


def main() -> None:
    args = parse_args()
    fit_dir = Path(args.fit_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(fit_dir / "sf_tf_speed_tuning_curve_points.csv")
    fits = pd.read_csv(fit_dir / "sf_tf_speed_tuning_fit_unit_summary.csv")
    units = parse_units(args.units)
    if not units:
        units = choose_units(fits, family=str(args.speed_family), examples_per_group=int(args.examples_per_group))
    png = plot_examples(
        points,
        fits,
        units=units,
        family=str(args.speed_family),
        out_dir=out_dir,
        dpi=int(args.dpi),
    )
    pd.DataFrame({"unit_index": units, "unit_label": [f"u{u:03d}" for u in units]}).to_csv(
        out_dir / f"{args.speed_family}_example_units.csv",
        index=False,
    )
    print(f"Wrote {png}")
    print("units:", ",".join(f"u{u:03d}" for u in units))


if __name__ == "__main__":
    main()
