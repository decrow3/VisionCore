#!/usr/bin/env python3
"""Plot the new RR100 parametric SF/TF preferences for Figure 4 review.

This is deliberately a checkpoint rather than a silent replacement of the
existing Figure 4 SF groups.  The historical groups were defined on a probe
with a different SF support, so the script shows that comparison explicitly
and saves an auditable role-based set of example tuning surfaces.
"""

from __future__ import annotations

import argparse
import hashlib
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = (
    ROOT
    / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_models_v1/rr100_sf_tf_parametric_model_arrays.npz"
)
DEFAULT_UNIT_TABLE = (
    ROOT
    / "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged/unit_feature_table.csv"
)
DEFAULT_OUT = ROOT / "outputs/fig4_active_sensing/rr100_sf_tf_parametric_preference_plots_v1"
OLD_GROUP_COLORS = {"low_sf": "#0072B2", "middle_sf": "#559F76", "high_sf": "#D55E00"}
OLD_GROUP_LABELS = {"low_sf": "old low SF", "middle_sf": "old middle SF", "high_sf": "old high SF"}
SF_SUPPORT = (1.0, 11.313708498984761)
TF_SUPPORT = (0.5, 32.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNIT_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
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


def file_identity(path: Path) -> dict[str, object]:
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


def load_data(npz_path: Path, unit_table_path: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    with np.load(npz_path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    required = {
        "rr100_index",
        "model_valid",
        "preferred_sf_cpd",
        "preferred_tf_hz",
        "sf_evaluation_grid_cpd",
        "tf_evaluation_grid_hz",
        "sf_factor_normalized_curves",
        "tf_factor_normalized_curves",
        "joint_parametric_surface_r2",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"NPZ is missing required arrays: {missing}")

    old = pd.read_csv(unit_table_path)
    old = old[["unit_index", "sf_split_metric", "sf_group"]].copy()
    frame = pd.DataFrame(
        {
            "rr100_index": arrays["rr100_index"].astype(int),
            "model_valid": arrays["model_valid"].astype(bool),
            "preferred_sf_cpd": arrays["preferred_sf_cpd"].astype(float),
            "preferred_tf_hz": arrays["preferred_tf_hz"].astype(float),
            "joint_parametric_surface_r2": arrays["joint_parametric_surface_r2"].astype(float),
        }
    )
    frame = frame.merge(old, left_on="rr100_index", right_on="unit_index", how="left", validate="one_to_one")
    frame.drop(columns="unit_index", inplace=True)
    return arrays, frame


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_preference_checkpoint(frame: pd.DataFrame, out_dir: Path) -> dict[str, float]:
    valid = frame[frame["model_valid"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35))

    ax = axes[0]
    for group in ("low_sf", "middle_sf", "high_sf"):
        sub = valid[valid["sf_group"].eq(group)]
        ax.scatter(
            sub["preferred_sf_cpd"],
            sub["preferred_tf_hz"],
            s=27,
            color=OLD_GROUP_COLORS[group],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            label=f"{OLD_GROUP_LABELS[group]} (n={len(sub)})",
        )
    ax.set(xscale="log", yscale="log", xlabel="new preferred SF (cycles/deg)", ylabel="new preferred |TF| (Hz)")
    ax.set_xlim(0.85, 13.5)
    ax.set_ylim(0.42, 40)
    ax.set_xticks([1, 2, 4, 8, 12], ["1", "2", "4", "8", "12"])
    ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32], ["0.5", "1", "2", "4", "8", "16", "32"])
    ax.set_title("A. New joint SF/TF preferences", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.3, loc="upper right")

    ax = axes[1]
    for group in ("low_sf", "middle_sf", "high_sf"):
        sub = valid[valid["sf_group"].eq(group)]
        ax.scatter(
            sub["sf_split_metric"],
            sub["preferred_sf_cpd"],
            s=24,
            color=OLD_GROUP_COLORS[group],
            alpha=0.74,
            edgecolor="white",
            linewidth=0.3,
        )
    rho = float(valid[["sf_split_metric", "preferred_sf_cpd"]].corr(method="spearman").iloc[0, 1])
    ax.axvline(0.5, color="0.35", lw=1.0, ls="--")
    ax.axhline(0.5, color="0.35", lw=1.0, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="historical SF split metric (cycles/deg)", ylabel="new preferred SF (cycles/deg)")
    ax.set_xlim(0.009, 16)
    ax.set_ylim(0.42, 16)
    ax.set_title(f"B. Old versus new SF (Spearman rho={rho:+.2f})", loc="left", fontweight="bold")
    ax.text(0.03, 0.97, "old 0.5-cpd split\ncollapses new valid models", transform=ax.transAxes, va="top", fontsize=6.8)

    ax = axes[2]
    sf_log = np.log2(valid["preferred_sf_cpd"].to_numpy(float))
    tf_log = np.log2(valid["preferred_tf_hz"].to_numpy(float))
    sf_edges = np.linspace(np.log2(SF_SUPPORT[0]), np.log2(SF_SUPPORT[1]), 13)
    tf_edges = np.linspace(np.log2(TF_SUPPORT[0]), np.log2(TF_SUPPORT[1]), 13)
    ax.hist(sf_log, bins=sf_edges, color="#3B75AF", alpha=0.72, label="SF preference")
    ax.hist(tf_log, bins=tf_edges, color="#C4683D", alpha=0.55, label="TF preference")
    ticks = np.arange(-1, 6, dtype=float)
    ax.set_xticks(ticks, [f"{2**v:g}" for v in ticks])
    ax.set_xlabel("preferred frequency (log2 axis; SF in cpd, TF in Hz)")
    ax.set_ylabel("valid units")
    ax.set_title("C. Preference marginals", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.8)

    for panel in axes:
        panel.grid(True, color="0.91", lw=0.55)
        panel.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RR100 parametric SF/TF preferences: Figure 4 checkpoint", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out_dir, "rr100_sf_tf_parametric_preferences_checkpoint")
    return {"spearman_old_vs_new_sf": rho}


def select_examples(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["model_valid"]].copy()
    valid["sf_percentile"] = valid["preferred_sf_cpd"].rank(method="average", pct=True)
    valid["tf_percentile"] = valid["preferred_tf_hz"].rank(method="average", pct=True)
    roles = [
        ("low_sf_low_tf_control", 0.15, 0.15),
        ("low_sf_high_tf_dissociation", 0.15, 0.85),
        ("central_preference", 0.50, 0.50),
        ("high_sf_low_tf_dissociation", 0.85, 0.15),
        ("high_sf_high_tf_example", 0.85, 0.85),
    ]
    selected: list[dict[str, object]] = []
    used: set[int] = set()
    for role, sf_target, tf_target in roles:
        candidates = valid[~valid["rr100_index"].isin(used)].copy()
        candidates["criterion_value"] = np.hypot(
            candidates["sf_percentile"] - sf_target,
            candidates["tf_percentile"] - tf_target,
        )
        row = candidates.sort_values(["criterion_value", "rr100_index"]).iloc[0]
        used.add(int(row["rr100_index"]))
        item = row.to_dict()
        item.update(
            {
                "selection_role": role,
                "criterion_name": f"distance_to_sf_tf_percentile_target_{sf_target:.2f}_{tf_target:.2f}",
                "selection_kind": "algorithmic",
            }
        )
        selected.append(item)

    candidates = valid[~valid["rr100_index"].isin(used)].sort_values(
        ["joint_parametric_surface_r2", "rr100_index"], na_position="last"
    )
    row = candidates.iloc[0]
    item = row.to_dict()
    item.update(
        {
            "selection_role": "weak_joint_fit_control",
            "criterion_name": "minimum_joint_parametric_surface_r2_among_remaining_valid_units",
            "criterion_value": float(row["joint_parametric_surface_r2"]),
            "selection_kind": "algorithmic",
        }
    )
    selected.append(item)
    columns = [
        "selection_role",
        "selection_kind",
        "criterion_name",
        "criterion_value",
        "rr100_index",
        "preferred_sf_cpd",
        "preferred_tf_hz",
        "sf_percentile",
        "tf_percentile",
        "joint_parametric_surface_r2",
        "sf_split_metric",
        "sf_group",
    ]
    return pd.DataFrame(selected)[columns]


def plot_selected_surfaces(arrays: dict[str, np.ndarray], selected: pd.DataFrame, out_dir: Path) -> None:
    sf_grid = arrays["sf_evaluation_grid_cpd"].astype(float)
    tf_grid = arrays["tf_evaluation_grid_hz"].astype(float)
    support_mask = sf_grid <= SF_SUPPORT[1] * (1.0 + 1e-10)
    sf_support_indices = np.flatnonzero(support_mask)
    sf_indices = np.unique(np.r_[sf_support_indices[::4], sf_support_indices[-1]])
    tf_indices = np.unique(np.r_[np.arange(len(tf_grid))[::4], len(tf_grid) - 1])
    sf = sf_grid[sf_indices]
    tf = tf_grid[tf_indices]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.8), sharex=True, sharey=True)
    mesh = None
    for ax, (_, row) in zip(axes.ravel(), selected.iterrows()):
        unit = int(row["rr100_index"])
        sf_curve = arrays["sf_factor_normalized_curves"][unit, sf_indices].astype(float)
        tf_curve = arrays["tf_factor_normalized_curves"][unit, tf_indices].astype(float)
        surface = np.outer(tf_curve, sf_curve)
        surface /= np.nanmax(surface)
        mesh = ax.pcolormesh(sf, tf, surface, shading="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.scatter([row["preferred_sf_cpd"]], [row["preferred_tf_hz"]], marker="x", s=38, lw=1.2, color="#46E1FF")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(SF_SUPPORT)
        ax.set_ylim(TF_SUPPORT)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32], ["0.5", "1", "2", "4", "8", "16", "32"])
        role = str(row["selection_role"]).replace("_", " ")
        ax.set_title(
            f"RR100 {unit:02d} | {role}\nSF {row['preferred_sf_cpd']:.2g} cpd, TF {row['preferred_tf_hz']:.2g} Hz, joint R2 {row['joint_parametric_surface_r2']:.2f}",
            loc="left",
            fontsize=7.2,
            fontweight="bold",
        )
    for ax in axes[-1, :]:
        ax.set_xlabel("spatial frequency (cycles/deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("|temporal frequency| (Hz)")
    assert mesh is not None
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.09, top=0.89, wspace=0.16, hspace=0.28)
    cbar_ax = fig.add_axes([0.905, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("within-unit response / fitted maximum")
    fig.suptitle(
        "Auditable RR100 SFxTF tuning examples (cyan x = fitted preference)",
        x=0.02,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    save_figure(fig, out_dir, "selected_rr100_sf_tf_parametric_surfaces")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    arrays, frame = load_data(args.npz, args.unit_table)
    comparison_path = args.out_dir / "rr100_old_new_sf_tf_preference_comparison.csv"
    frame.to_csv(comparison_path, index=False)
    statistics = plot_preference_checkpoint(frame, args.out_dir)
    selected = select_examples(frame)
    selected_path = args.out_dir / "selected_rr100_sf_tf_examples.csv"
    selected.to_csv(selected_path, index=False)
    plot_selected_surfaces(arrays, selected, args.out_dir)

    valid = frame[frame["model_valid"]]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "map_first_checkpoint",
        "source_npz": file_identity(args.npz),
        "historical_unit_table": file_identity(args.unit_table),
        "n_rr100_units": int(len(frame)),
        "n_valid_new_models": int(frame["model_valid"].sum()),
        "n_invalid_new_models": int((~frame["model_valid"]).sum()),
        "new_preferred_sf_range_cpd": [float(valid["preferred_sf_cpd"].min()), float(valid["preferred_sf_cpd"].max())],
        "new_preferred_tf_range_hz": [float(valid["preferred_tf_hz"].min()), float(valid["preferred_tf_hz"].max())],
        "n_valid_below_historical_0p5_cpd_split": int((valid["preferred_sf_cpd"] < 0.5).sum()),
        **statistics,
        "surface_scaling": "each unit divided by its fitted maximum on the declared SF/TF fit support; shared 0-1 color scale",
        "artifacts": {
            "preference_checkpoint": "rr100_sf_tf_parametric_preferences_checkpoint.{png,pdf,svg}",
            "selected_surfaces": "selected_rr100_sf_tf_parametric_surfaces.{png,pdf,svg}",
            "selected_examples": selected_path.name,
            "old_new_comparison": comparison_path.name,
        },
        "decision_boundary": (
            "Do not reuse the historical 0.5-cpd threshold with the new estimates: all valid new models are above it. "
            "A revised Figure 4 grouping rule requires an explicit scientific choice."
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir.resolve()}")
    print(selected[["selection_role", "rr100_index", "preferred_sf_cpd", "preferred_tf_hz", "joint_parametric_surface_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
