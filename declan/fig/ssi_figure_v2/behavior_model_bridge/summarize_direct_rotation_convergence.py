"""Summarize exact-pair rotation-grid convergence for Figure 4 panel G.

This script never evaluates or interpolates a dose curve.  It joins independent
fresh model evaluations at 4, 8, 16, and 32 deterministic midpoint angles for
the same local and preselected representative-offset image--trajectory pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/panel_g_direct_rotation_convergence"
)
DEFAULT_K8 = Path(
    "outputs/fig/ssi_figure_v2/behavior_model_bridge/"
    "panel_g_direct_exact_pair_ssi_targeted_v1"
)
ROLES = ("oblique-local-positive", "motor-prior-dissociation")
ROLE_LABELS = {
    "oblique-local-positive": "Oblique local-positive",
    "motor-prior-dissociation": "Motor-prior dissociation",
}
POPULATION = "high_sf_aligned"
METRICS = (
    ("bits_per_spike", "bits/spike"),
    ("information_bits_per_sample", "information bits/sample"),
    ("expected_spikes_per_sample", "expected spikes/sample"),
)
COLORS = {"local": "#2468a2", "representative_offset": "#d77c22", "locality": "#27845d"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--k8-dir", type=Path, default=DEFAULT_K8)
    return parser.parse_args()


def _paths(root: Path, k8_dir: Path) -> dict[int, Path]:
    return {4: root / "k04", 8: k8_dir, 16: root / "k16", 32: root / "k32"}


def _selected_locations(reference_dir: Path) -> dict[str, tuple[str, str]]:
    manifest = pd.read_csv(reference_dir / "direct_location_manifest.csv")
    selected: dict[str, tuple[str, str]] = {}
    for role in ROLES:
        group = manifest[manifest["example_role"].astype(str).eq(role)]
        local = group[group["location_kind"].astype(str).eq("local")]
        offset = group[group["location_kind"].astype(str).eq("offset")]
        if len(local) != 1 or len(offset) != 1:
            raise RuntimeError(f"Expected one local and one representative offset for {role}")
        selected[role] = (str(local.iloc[0]["location_id"]), str(offset.iloc[0]["location_id"]))
    return selected


def _load_values(paths: dict[int, Path], selected: dict[str, tuple[str, str]]) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for n_angles, directory in paths.items():
        table = pd.read_csv(directory / "direct_location_rotation_contrasts.csv")
        keep = np.zeros(len(table), dtype=bool)
        for role, location_ids in selected.items():
            keep |= table["example_role"].astype(str).eq(role) & table["location_id"].astype(str).isin(location_ids)
        table = table.loc[keep].copy()
        table["n_angles"] = n_angles
        table["effect_kind"] = np.where(
            table["location_kind"].astype(str).eq("local"), "local", "representative_offset"
        )
        tables.append(table)
    values = pd.concat(tables, ignore_index=True)
    expected = len(paths) * len(ROLES) * 2 * values["population"].nunique()
    if len(values) != expected:
        raise RuntimeError(f"Incomplete convergence join: got {len(values)} rows, expected {expected}")
    return values


def _add_locality(values: pd.DataFrame) -> pd.DataFrame:
    effect_columns = [column for column in values if column.startswith("real_minus_rotation_")]
    rows: list[dict[str, object]] = []
    for keys, group in values.groupby(["n_angles", "example_role", "population"], sort=False):
        indexed = group.set_index("effect_kind")
        if not {"local", "representative_offset"}.issubset(indexed.index):
            raise RuntimeError(f"Missing local or representative offset for {keys}")
        row = indexed.loc["local"].to_dict()
        row.update({"n_angles": keys[0], "example_role": keys[1], "population": keys[2]})
        row["location_id"] = "local-minus-representative-offset"
        row["location_kind"] = "derived"
        row["effect_kind"] = "locality"
        for column in effect_columns:
            row[column] = float(indexed.loc["local", column]) - float(indexed.loc["representative_offset", column])
        rows.append(row)
    return pd.concat([values, pd.DataFrame(rows)], ignore_index=True)


def _convergence_summary(values: pd.DataFrame) -> pd.DataFrame:
    primary = values[values["population"].astype(str).eq(POPULATION)]
    rows: list[dict[str, object]] = []
    column = "real_minus_rotation_bits_per_spike"
    for (role, effect_kind), group in primary.groupby(["example_role", "effect_kind"], sort=False):
        estimates = group.set_index("n_angles")[column].astype(float)
        reference = float(estimates.loc[32])
        absolute_tolerance = max(0.005, 0.05 * abs(reference))
        row: dict[str, object] = {
            "example_role": role,
            "effect_kind": effect_kind,
            "reference_n_angles": 32,
            "absolute_tolerance_bits_per_spike": absolute_tolerance,
        }
        for n_angles in (4, 8, 16, 32):
            value = float(estimates.loc[n_angles])
            error = abs(value - reference)
            row[f"estimate_k{n_angles:02d}_bits_per_spike"] = value
            row[f"absolute_error_k{n_angles:02d}_vs_k32"] = error
            row[f"relative_error_k{n_angles:02d}_vs_k32"] = error / max(abs(reference), 1e-12)
            row[f"adequate_k{n_angles:02d}"] = bool(error <= absolute_tolerance)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_convergence(values: pd.DataFrame, out_dir: Path) -> None:
    primary = values[values["population"].astype(str).eq(POPULATION)]
    fig, axes = plt.subplots(len(ROLES), len(METRICS), figsize=(12.2, 6.8), squeeze=False)
    for row_index, role in enumerate(ROLES):
        role_values = primary[primary["example_role"].astype(str).eq(role)]
        for column_index, (metric, label) in enumerate(METRICS):
            ax = axes[row_index, column_index]
            value_column = f"real_minus_rotation_{metric}"
            for effect_kind, marker, linestyle in (
                ("local", "o", "-"),
                ("representative_offset", "s", "--"),
                ("locality", "^", "-."),
            ):
                curve = role_values[role_values["effect_kind"].astype(str).eq(effect_kind)].sort_values("n_angles")
                ax.plot(
                    curve["n_angles"], curve[value_column], marker=marker, linestyle=linestyle,
                    linewidth=1.8, markersize=5, color=COLORS[effect_kind],
                    label=effect_kind.replace("_", " "),
                )
            ax.axhline(0.0, color="0.65", linewidth=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xticks([4, 8, 16, 32], labels=["4", "8", "16", "32"])
            ax.grid(alpha=0.20, linewidth=0.6)
            if row_index == len(ROLES) - 1:
                ax.set_xlabel("fresh rotation angles")
            if column_index == 0:
                ax.set_ylabel(f"{ROLE_LABELS[role]}\nreal − rotation mean\n{label}")
            else:
                ax.set_ylabel(label)
            if row_index == 0:
                ax.set_title(label)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Panel G exact-pair rotation-grid convergence\n"
        "same image–trajectory pairs; locally aligned high-SF units",
        fontsize=12, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "direct_rotation_grid_convergence.png", dpi=220)
    fig.savefig(out_dir / "direct_rotation_grid_convergence.pdf")
    plt.close(fig)


def _plot_angle_curves(k32_dir: Path, selected: dict[str, tuple[str, str]], out_dir: Path) -> None:
    values = pd.read_csv(k32_dir / "direct_population_metrics.csv")
    values = values[values["population"].astype(str).eq(POPULATION)]
    fig, axes = plt.subplots(len(ROLES), 2, figsize=(10.5, 6.8), squeeze=False)
    for row_index, role in enumerate(ROLES):
        for column_index, (effect_kind, location_id) in enumerate(
            zip(("local", "representative offset"), selected[role])
        ):
            ax = axes[row_index, column_index]
            group = values[
                values["example_role"].astype(str).eq(role)
                & values["location_id"].astype(str).eq(location_id)
            ]
            real = float(group[group["condition_kind"].astype(str).eq("real")]["bits_per_spike"].iloc[0])
            rotations = group[group["condition_kind"].astype(str).eq("rotation")].sort_values("rotation_angle_deg")
            x = rotations["rotation_angle_deg"].to_numpy(dtype=float)
            y = rotations["bits_per_spike"].to_numpy(dtype=float)
            rotation_mean = float(np.mean(y))
            ax.plot(np.r_[x, x[0] + 360.0], np.r_[y, y[0]], color="#6d4c9b", linewidth=1.7)
            ax.scatter(x, y, color="#6d4c9b", s=11, zorder=3)
            ax.axhline(real, color="#1e1e1e", linestyle="--", linewidth=1.2, label="real trajectory")
            ax.axhline(rotation_mean, color="#2468a2", linestyle=":", linewidth=1.2, label="rotation mean")
            ax.fill_between([0, 360], real, y2=rotation_mean, color="#2468a2", alpha=0.08)
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.grid(alpha=0.18, linewidth=0.6)
            ax.set_title(f"{ROLE_LABELS[role]} — {effect_kind}", fontsize=10)
            if row_index == len(ROLES) - 1:
                ax.set_xlabel("image rotation (deg)")
            ax.set_ylabel("direct SSI (bits/spike)")
            if row_index == 0 and column_index == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "32-angle exact model evaluations\n"
        "curves are individual rotated image–trajectory movies, not interpolated doses",
        fontsize=12, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "direct_rotation_angle_curves_k32.png", dpi=220)
    fig.savefig(out_dir / "direct_rotation_angle_curves_k32.pdf")
    plt.close(fig)


def _angular_exceedance(k32_dir: Path, selected: dict[str, tuple[str, str]]) -> pd.DataFrame:
    values = pd.read_csv(k32_dir / "direct_population_metrics.csv")
    values = values[values["population"].astype(str).eq(POPULATION)]
    rows: list[dict[str, object]] = []
    for role, (local_id, offset_id) in selected.items():
        for effect_kind, location_id in (("local", local_id), ("representative_offset", offset_id)):
            group = values[
                values["example_role"].astype(str).eq(role)
                & values["location_id"].astype(str).eq(location_id)
            ]
            real = float(group[group["condition_kind"].astype(str).eq("real")]["bits_per_spike"].iloc[0])
            rotations = group[group["condition_kind"].astype(str).eq("rotation")].sort_values("rotation_angle_deg")
            rotation_values = rotations["bits_per_spike"].to_numpy(dtype=float)
            angles = rotations["rotation_angle_deg"].to_numpy(dtype=float)
            exceeds = rotation_values >= real
            rows.append(
                {
                    "example_role": role,
                    "effect_kind": effect_kind,
                    "location_id": location_id,
                    "n_angles": len(rotation_values),
                    "real_bits_per_spike": real,
                    "rotation_mean_bits_per_spike": float(np.mean(rotation_values)),
                    "real_minus_rotation_mean_bits_per_spike": float(real - np.mean(rotation_values)),
                    "rotation_min_bits_per_spike": float(np.min(rotation_values)),
                    "rotation_max_bits_per_spike": float(np.max(rotation_values)),
                    "n_rotations_at_or_above_real": int(np.sum(exceeds)),
                    "fraction_rotations_below_real": float(np.mean(rotation_values < real)),
                    "angles_at_or_above_real_deg": ";".join(f"{angle:.3f}" for angle in angles[exceeds]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {key: value.resolve() for key, value in _paths(root, args.k8_dir.resolve()).items()}
    selected = _selected_locations(paths[32])
    values = _add_locality(_load_values(paths, selected))
    summary = _convergence_summary(values)
    angular_exceedance = _angular_exceedance(paths[32], selected)
    values.to_csv(root / "direct_rotation_convergence_values.csv", index=False)
    summary.to_csv(root / "direct_rotation_convergence_summary.csv", index=False)
    angular_exceedance.to_csv(root / "direct_rotation_angular_exceedance_k32.csv", index=False)
    _plot_convergence(values, root)
    _plot_angle_curves(paths[32], selected, root)
    metadata = {
        "analysis": "exact image-trajectory rotation-grid convergence",
        "model_evaluation": "fresh for every condition; no interpolation or dose curve",
        "n_angles": [4, 8, 16, 32],
        "paths": {str(key): str(value) for key, value in paths.items()},
        "selected_locations": selected,
        "population": POPULATION,
        "adequacy_rule": "absolute K-vs-K32 error <= max(0.005 bits/spike, 5% of |K32|)",
    }
    (root / "convergence_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\n32-angle angular exceedance:\n" + angular_exceedance.to_string(index=False))
    print(f"[rotation-convergence] wrote diagnostics to {root}")


if __name__ == "__main__":
    main()
