#!/usr/bin/env python3
"""Compare native-readout synthetic RR100 SF tuning with held-out gratings."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.active_sensing_movie_information.run_backimage_rr100_frequency_tuning_probe import (
    make_windowed_drifting_grating_movie,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_native_readout_common_sf_probe_v1"
DEFAULT_HELDOUT_DIR = ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1"
COMMON_SFS = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0], dtype=float)
COLORS = {
    "synthetic_native_twin": "#0072B2",
    "heldout_fitted_twin": "#D55E00",
    "heldout_recorded": "#222222",
}
LABELS = {
    "synthetic_native_twin": "synthetic native twin",
    "heldout_fitted_twin": "held-out fitted twin",
    "heldout_recorded": "held-out recorded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--heldout-dir", type=Path, default=DEFAULT_HELDOUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROBE_DIR / "common_sf_comparison")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def nearest_common_sf(values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(raw.shape, np.nan, dtype=float)
    finite = np.isfinite(raw)
    if finite.any():
        distance = np.abs(raw[finite, None] - COMMON_SFS[None, :])
        idx = np.argmin(distance, axis=1)
        chosen = COMMON_SFS[idx]
        valid = np.min(distance, axis=1) < 1e-5
        vals = np.full(chosen.shape, np.nan, dtype=float)
        vals[valid] = chosen[valid]
        out[finite] = vals
    return pd.Series(out, index=values.index, dtype=float)


def curve_strength(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    vmax = float(np.max(finite))
    return float((vmax - float(np.min(finite))) / max(abs(vmax), 1e-12))


def prepare_curves(grouped_path: Path, maps_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = pd.read_csv(grouped_path)
    grouped["spatial_cpd"] = nearest_common_sf(grouped["spatial_cpd"])
    grouped["temporal_hz"] = pd.to_numeric(grouped["temporal_hz"], errors="coerce")
    grouped["response_amp_rms"] = pd.to_numeric(grouped["response_amp_rms"], errors="coerce")
    dynamic = grouped[(grouped["temporal_hz"] > 0) & grouped["spatial_cpd"].notna()].copy()
    synthetic = (
        dynamic.groupby(["unit_index", "spatial_cpd"], as_index=False, sort=True)
        .agg(value=("response_amp_rms", "mean"), n_orientation_tf_points=("response_amp_rms", "size"))
        .rename(columns={"unit_index": "rr100_index", "spatial_cpd": "sf_cpd"})
    )
    synthetic["source"] = "synthetic_native_twin"

    maps = pd.read_csv(maps_path)
    maps["sf_cpd"] = nearest_common_sf(maps["spatial_frequency_cpd"])
    maps["rate_hz"] = pd.to_numeric(maps["rate_hz"], errors="coerce")
    heldout = maps[maps["sf_cpd"].notna()].copy()
    support = (
        heldout.groupby(["rr100_index", "session"], as_index=False, sort=True)["sf_cpd"]
        .agg(lambda values: tuple(sorted(set(float(v) for v in values))))
        .rename(columns={"sf_cpd": "common_support"})
    )
    support["common_support_sf_cpd"] = support["common_support"].map(lambda values: ",".join(f"{v:g}" for v in values))
    support["common_support_n_bins"] = support["common_support"].map(len)
    support["common_support_min_sf_cpd"] = support["common_support"].map(min)
    support["common_support_max_sf_cpd"] = support["common_support"].map(max)
    support_lookup = support.set_index("rr100_index")["common_support"].to_dict()
    heldout = (
        heldout.groupby(["rr100_index", "source", "sf_cpd"], as_index=False, sort=True)
        .agg(value=("rate_hz", "max"), n_orientation_tf_points=("rate_hz", "count"))
    )
    heldout["source"] = heldout["source"].map(
        {"twin": "heldout_fitted_twin", "recorded": "heldout_recorded"}
    )
    curves = pd.concat(
        [synthetic[["rr100_index", "source", "sf_cpd", "value", "n_orientation_tf_points"]], heldout],
        ignore_index=True,
    )
    curves["curve_contract"] = curves["source"].map(
        {
            "synthetic_native_twin": "TF>0 phase-RMS amplitude averaged over orientation and temporal frequency",
            "heldout_fitted_twin": "mean fitted response maximized over orientation at each positive SF",
            "heldout_recorded": "mean recorded response maximized over orientation at each positive SF",
        }
    )
    curves["in_unit_common_support"] = [
        float(sf) in support_lookup[int(unit)]
        for unit, sf in zip(curves["rr100_index"], curves["sf_cpd"], strict=True)
    ]

    rows: list[dict[str, object]] = []
    for unit in range(100):
        row: dict[str, object] = {"rr100_index": unit}
        unit_support = tuple(float(v) for v in support_lookup[unit])
        for source in LABELS:
            sub = curves[
                (curves["rr100_index"] == unit)
                & (curves["source"] == source)
                & curves["in_unit_common_support"]
            ].sort_values("sf_cpd")
            vals = sub["value"].to_numpy(dtype=float)
            valid = np.isfinite(vals)
            pref = float(sub.iloc[int(np.nanargmax(vals))]["sf_cpd"]) if valid.any() else float("nan")
            row[f"{source}_preferred_sf_cpd"] = pref
            row[f"{source}_curve_strength"] = curve_strength(vals)
            row[f"{source}_peak_is_boundary"] = bool(pref in (unit_support[0], unit_support[-1])) if np.isfinite(pref) else False
            row[f"{source}_n_valid_sf_bins"] = int(valid.sum())
        rows.append(row)
    units = pd.DataFrame(rows).merge(
        support.drop(columns="common_support"), on="rr100_index", how="left", validate="one_to_one"
    )
    support_families = (
        support.groupby(["common_support_sf_cpd", "common_support_n_bins"], as_index=False)
        .agg(
            n_rr100_units=("rr100_index", "size"),
            n_sessions=("session", "nunique"),
            sessions=("session", lambda values: ",".join(sorted(set(str(v) for v in values)))),
        )
        .sort_values("common_support_sf_cpd")
    )
    return curves, units, support_families


def pair_stats(units: pd.DataFrame, left: str, right: str) -> dict[str, object]:
    x = units[f"{left}_preferred_sf_cpd"]
    y = units[f"{right}_preferred_sf_cpd"]
    valid = x.notna() & y.notna()
    xv = x[valid].to_numpy(dtype=float)
    yv = y[valid].to_numpy(dtype=float)
    delta = np.abs(np.log2(xv / yv)) if xv.size else np.asarray([], dtype=float)
    rho = float(pd.Series(xv).corr(pd.Series(yv), method="spearman")) if xv.size >= 3 else float("nan")
    return {
        "comparison": f"{left}_vs_{right}",
        "left_source": left,
        "right_source": right,
        "n_pairs": int(xv.size),
        "exact_bin_agreement_n": int(np.sum(delta == 0)) if delta.size else 0,
        "exact_bin_agreement_fraction": float(np.mean(delta == 0)) if delta.size else float("nan"),
        "within_one_octave_n": int(np.sum(delta <= 1)) if delta.size else 0,
        "within_one_octave_fraction": float(np.mean(delta <= 1)) if delta.size else float("nan"),
        "median_absolute_octave_difference": float(np.median(delta)) if delta.size else float("nan"),
        "spearman_rho": rho,
    }


def select_examples(units: pd.DataFrame) -> pd.DataFrame:
    frame = units.copy()
    s = frame["synthetic_native_twin_preferred_sf_cpd"]
    t = frame["heldout_fitted_twin_preferred_sf_cpd"]
    r = frame["heldout_recorded_preferred_sf_cpd"]
    valid = s.notna() & t.notna() & r.notna()
    frame["mean_curve_strength"] = frame[
        [f"{source}_curve_strength" for source in LABELS]
    ].mean(axis=1)
    frame["synthetic_twin_gap_oct"] = np.abs(np.log2(s / t))
    frame["synthetic_recorded_gap_oct"] = np.abs(np.log2(s / r))
    picks: list[tuple[str, int, str]] = []

    def pick(role: str, mask: pd.Series, score: pd.Series, reason: str) -> None:
        candidates = frame[mask & ~frame["rr100_index"].isin([item[1] for item in picks])]
        if candidates.empty:
            return
        idx = score.loc[candidates.index].astype(float).idxmax()
        picks.append((role, int(frame.loc[idx, "rr100_index"]), reason))

    pick(
        "three_way_match",
        valid & (s == t) & (t == r),
        frame["mean_curve_strength"],
        "all three discrete preferences match; strongest mean curve modulation among matches",
    )
    pick(
        "synthetic_twin_dissociation",
        valid & (s == t) & (t != r),
        np.abs(np.log2(t / r)),
        "synthetic and held-out twins match while recorded preference differs; largest recorded gap",
    )
    pick(
        "synthetic_recorded_dissociation",
        valid & (s == r) & (s != t),
        np.abs(np.log2(s / t)),
        "synthetic and recorded preferences match while held-out twin differs; largest twin gap",
    )
    pick(
        "cross_probe_mismatch",
        valid & (s != t) & (s != r),
        frame["synthetic_twin_gap_oct"] + frame["synthetic_recorded_gap_oct"],
        "synthetic preference differs from both held-out estimates; largest summed octave gap",
    )
    pick(
        "weak_control",
        valid,
        -frame["mean_curve_strength"],
        "weakest mean normalized curve modulation among complete units",
    )
    selected = pd.DataFrame(picks, columns=["selection_role", "rr100_index", "selection_reason"])
    return selected.merge(frame, on="rr100_index", how="left", validate="one_to_one")


def plot_stimulus_support(out_dir: Path, support_families: pd.DataFrame, dpi: int) -> Path:
    ppd = 37.50476617
    image_size = 51
    fov_deg = image_size / ppd
    fig, axes = plt.subplots(1, len(COMMON_SFS), figsize=(12.5, 2.8), constrained_layout=True)
    for ax, sf in zip(axes, COMMON_SFS, strict=True):
        movie = make_windowed_drifting_grating_movie(
            image_size=image_size,
            orientation_deg=78.75,
            spatial_cpd=float(sf),
            temporal_hz=3.2,
            phase_rad=0.0,
            n_valid_frames=1,
            n_lags=32,
            frame_rate_hz=120.0,
            ppd=ppd,
            contrast=0.8,
            window_sigma_frac=0.28,
        )
        ax.imshow(movie[-1], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{sf:g} cpd\n{sf * fov_deg:.2f} cycles/window")
        ax.set_xticks([])
        ax.set_yticks([])
    family_text = "; ".join(
        f"{row.common_support_sf_cpd} cpd (n={int(row.n_rr100_units)})"
        for row in support_families.itertuples(index=False)
    )
    fig.suptitle(
        f"Cycle-valid union in the native 51 px window ({fov_deg:.2f}°, floor {1/fov_deg:.3f} cpd)\n"
        f"per-unit comparisons use their session's four-bin intersection: {family_text}",
        fontsize=10.5,
    )
    path = out_dir / "common_sf_stimulus_support.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def normalize_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return values
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    return (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)


def plot_examples(out_dir: Path, curves: pd.DataFrame, selected: pd.DataFrame, dpi: int) -> Path:
    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 2.35 * n), constrained_layout=True, squeeze=False)
    for ax, (_, meta) in zip(axes[:, 0], selected.iterrows(), strict=True):
        unit = int(meta["rr100_index"])
        support = {float(value) for value in str(meta["common_support_sf_cpd"]).split(",")}
        for source in LABELS:
            sub = curves[
                (curves["rr100_index"] == unit)
                & (curves["source"] == source)
                & curves["sf_cpd"].isin(support)
            ].sort_values("sf_cpd")
            ax.plot(
                sub["sf_cpd"], normalize_curve(sub["value"].to_numpy(dtype=float)), marker="o", lw=1.8,
                color=COLORS[source], label=LABELS[source],
            )
        prefs = ", ".join(
            f"{LABELS[source]}={meta[f'{source}_preferred_sf_cpd']:g}"
            for source in LABELS
            if np.isfinite(float(meta[f"{source}_preferred_sf_cpd"]))
        )
        ax.set_xscale("log", base=2)
        ticks = sorted(support)
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.set_ylim(-0.06, 1.06)
        ax.grid(alpha=0.18)
        ax.set_ylabel("normalized tuning")
        ax.set_title(f"u{unit:03d} · {meta['selection_role']}\n{prefs}", loc="left", fontsize=9.5)
    axes[-1, 0].set_xlabel("spatial frequency (cpd; exact common bins)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    path = out_dir / "common_sf_selected_example_curves.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def confusion(ax: plt.Axes, units: pd.DataFrame, left: str, right: str, title: str) -> None:
    x = units[f"{left}_preferred_sf_cpd"]
    y = units[f"{right}_preferred_sf_cpd"]
    valid = x.notna() & y.notna()
    table = pd.crosstab(y[valid], x[valid]).reindex(index=COMMON_SFS, columns=COMMON_SFS, fill_value=0)
    image = ax.imshow(table.to_numpy(), origin="lower", cmap="Blues", vmin=0)
    for row in range(len(COMMON_SFS)):
        for col in range(len(COMMON_SFS)):
            value = int(table.iloc[row, col])
            if value:
                ax.text(col, row, str(value), ha="center", va="center", fontsize=8,
                        color="white" if value > 0.55 * table.to_numpy().max() else "black")
    ax.set_xticks(range(len(COMMON_SFS)), [f"{v:g}" for v in COMMON_SFS])
    ax.set_yticks(range(len(COMMON_SFS)), [f"{v:g}" for v in COMMON_SFS])
    ax.set_xlabel(LABELS[left] + " pref")
    ax.set_ylabel(LABELS[right] + " pref")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="units")


def plot_population(out_dir: Path, units: pd.DataFrame, stats: pd.DataFrame, dpi: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.4), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(len(COMMON_SFS), dtype=float)
    width = 0.24
    for offset, source in zip((-width, 0.0, width), LABELS, strict=True):
        counts = units[f"{source}_preferred_sf_cpd"].value_counts().reindex(COMMON_SFS, fill_value=0)
        ax.bar(x + offset, counts.to_numpy(), width=width, color=COLORS[source], label=LABELS[source])
    ax.set_xticks(x, [f"{v:g}" for v in COMMON_SFS])
    ax.set_xlabel("preferred SF (cpd)")
    ax.set_ylabel("unit count")
    ax.set_title("A  Exact common-bin distributions", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.18)

    pairs = [
        ("synthetic_native_twin", "heldout_fitted_twin", "B  Same model, different probe"),
        ("synthetic_native_twin", "heldout_recorded", "C  Synthetic twin vs recorded"),
        ("heldout_fitted_twin", "heldout_recorded", "D  Held-out twin vs recorded"),
    ]
    for ax, (left, right, title) in zip(axes.flat[1:], pairs, strict=True):
        rec = stats[stats["comparison"] == f"{left}_vs_{right}"].iloc[0]
        confusion(ax, units, left, right, title)
        ax.text(
            0.03, 0.97,
            f"exact={float(rec['exact_bin_agreement_fraction']):.2f}; ≤1 oct={float(rec['within_one_octave_fraction']):.2f}\n"
            f"Spearman ρ={float(rec['spearman_rho']):.2f}; n={int(rec['n_pairs'])}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
        )
    fig.suptitle("RR100 SF tuning on native readouts and genuinely shared cycle-valid support", fontsize=12)
    path = out_dir / "common_sf_population_comparison.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    probe_dir = args.probe_dir.resolve()
    heldout_dir = args.heldout_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped_path = probe_dir / "frequency_tuning_grouped.csv"
    maps_path = heldout_dir / "rr100_grating_tuning_maps_long.csv"
    metrics_path = heldout_dir / "rr100_grating_tuning_metrics.csv"
    identity_path = probe_dir / "frequency_tuning_request_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    sampling = identity["stimulus_sampling"]
    if int(identity["image_size"]) != 51:
        raise ValueError("Native-readout comparison requires a 51 px probe")
    if not np.allclose(sorted(identity["spatial_cpds"]), COMMON_SFS):
        raise ValueError("Synthetic probe does not use the exact common SF support")
    one_cycle = float(sampling["one_cycle_across_window_cpd"])
    nyquist = float(sampling["spatial_nyquist_cpd"])
    if np.any(COMMON_SFS < one_cycle) or np.any(COMMON_SFS > nyquist):
        raise ValueError("Requested common SF support is not cycle-valid and Nyquist-valid")

    curves, units, support_families = prepare_curves(grouped_path, maps_path)
    metrics = pd.read_csv(metrics_path)[
        ["rr100_index", "canonical_channel", "session", "source_unit_index", "ccnorm", "real_tuning_strength", "twin_tuning_strength"]
    ]
    units = units.merge(metrics, on="rr100_index", how="left", validate="one_to_one")
    stats = pd.DataFrame(
        [
            pair_stats(units, "synthetic_native_twin", "heldout_fitted_twin"),
            pair_stats(units, "synthetic_native_twin", "heldout_recorded"),
            pair_stats(units, "heldout_fitted_twin", "heldout_recorded"),
        ]
    )
    selected = select_examples(units)

    curves_path = out_dir / "common_sf_curves_long.csv"
    units_path = out_dir / "common_sf_unit_summary.csv"
    stats_path = out_dir / "common_sf_agreement_stats.csv"
    selected_path = out_dir / "common_sf_selected_examples.csv"
    support_families_path = out_dir / "common_sf_support_families.csv"
    curves.to_csv(curves_path, index=False)
    units.to_csv(units_path, index=False)
    stats.to_csv(stats_path, index=False)
    selected.to_csv(selected_path, index=False)
    support_families.to_csv(support_families_path, index=False)
    support_figure = plot_stimulus_support(out_dir, support_families, int(args.dpi))
    example_figure = plot_examples(out_dir, curves, selected, int(args.dpi))
    population_figure = plot_population(out_dir, units, stats, int(args.dpi))

    manifest = {
        "analysis": "rr100_native_common_sf_vs_heldout",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_probe_union_sf_cpd": COMMON_SFS.tolist(),
        "per_unit_comparison_support": "intersection of the synthetic union with the four positive SF bins present in that unit's held-out session",
        "native_readout_evidence": "51 px input produces a 1x1 spatial output from the learned Gaussian readout",
        "cycle_validity": {
            "fov_deg": float(sampling["fov_deg"]),
            "one_cycle_floor_cpd": one_cycle,
            "spatial_nyquist_cpd": nyquist,
            "all_common_bins_valid": True,
        },
        "synthetic_contract": "native 1x1 learned unit readout; dynamic TF>0 phase-RMS amplitude averaged across orientation and TF; discrete argmax",
        "heldout_contract": "recorded-selected lag; mean response maximized across orientation at each shared positive SF; discrete argmax",
        "remaining_non_equivalence": "synthetic preference uses dynamic phase-RMS amplitude while held-out preferences use mean responses",
        "sources": [file_identity(grouped_path), file_identity(maps_path), file_identity(metrics_path), file_identity(identity_path)],
        "outputs": {
            "curves": str(curves_path), "units": str(units_path), "stats": str(stats_path), "selected_examples": str(selected_path),
            "support_families": str(support_families_path),
            "stimulus_support_figure": str(support_figure), "example_figure": str(example_figure), "population_figure": str(population_figure),
        },
    }
    (out_dir / "common_sf_comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(stats.to_string(index=False))
    print("\nSelected examples:")
    print(selected[["selection_role", "rr100_index", "selection_reason"]].to_string(index=False))
    print(f"\nWrote comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
