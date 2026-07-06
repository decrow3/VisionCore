#!/usr/bin/env python3
"""Summarize BackImage local-feature screens across image patch radii."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_patch_radius_sensitivity_v1"
)

RADII = (
    ("r0p25", 0.25),
    ("r0p5", 0.5),
    ("r1p0", 1.0),
)

IMAGE_TABLES = {
    0.25: (
        Path("outputs")
        / "fixation_statistics_by_stimulus_all_sessions_after_review"
        / "backimage_image_structure_patch_radius_0p25_v1"
        / "run_metadata.json"
    ),
    0.5: (
        Path("outputs")
        / "fixation_statistics_by_stimulus_all_sessions_after_review"
        / "backimage_image_structure_patch_radius_0p5_v1"
        / "run_metadata.json"
    ),
    1.0: (
        Path("outputs")
        / "fixation_statistics_by_stimulus_all_sessions_after_review"
        / "backimage_image_structure_reviewed_v2_screenfiltered_yfix_slope_v1"
        / "run_metadata.json"
    ),
}

LOCAL_KEY_ROWS = (
    ("orientation_coherence", "drift_edge_cos2"),
    ("orientation_coherence", "rms_across_arcmin"),
    ("orientation_coherence", "rms_delta_along_minus_across_arcmin"),
    ("spectrum_anisotropy", "drift_edge_cos2"),
    ("spectrum_anisotropy", "rms_across_arcmin"),
    ("spectrum_anisotropy", "rms_delta_along_minus_across_arcmin"),
    ("oriented_8plus_cpd_power", "drift_edge_cos2"),
    ("oriented_8plus_cpd_power", "rms_across_arcmin"),
    ("oriented_8plus_cpd_power", "rms_radius_arcmin"),
)

SF_KEY_FEATURES = ("abs_power_4_8_cpd", "abs_power_8plus_cpd")
SF_KEY_METRICS = (
    "rms_radius_arcmin",
    "rms_across_arcmin",
    "rms_along_arcmin",
    "rms_delta_along_minus_across_arcmin",
)


@dataclass(frozen=True)
class PatchRadiusSummaryConfig:
    root: str


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_window_counts() -> pd.DataFrame:
    rows = []
    for radius, path in IMAGE_TABLES.items():
        meta = _read_json(path)
        rows.append(
            {
                "patch_radius_deg": radius,
                "patch_full_width_deg": 2.0 * radius,
                "n_raw_augmented_windows": int(meta.get("n_raw_augmented_windows", np.nan)),
                "n_windows": int(meta.get("n_windows", np.nan)),
                "n_failed_image_feature_windows": int(meta.get("n_failed_image_feature_windows", np.nan)),
                "n_excluded_patch_contamination_windows": int(
                    meta.get("n_excluded_patch_contamination_windows", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("patch_radius_deg")


def _load_local(root: Path) -> pd.DataFrame:
    frames = []
    for label, radius in RADII:
        path = root / f"local_feature_poles_{label}" / "pole_eye_metric_high_low_contrasts.csv"
        df = pd.read_csv(path)
        df.insert(0, "patch_radius_deg", radius)
        df.insert(1, "patch_full_width_deg", 2.0 * radius)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_sf(root: Path) -> pd.DataFrame:
    frames = []
    for label, radius in RADII:
        path = root / f"sf_scaling_{label}" / "sf_controlled_slope_summary.csv"
        df = pd.read_csv(path)
        df.insert(0, "patch_radius_deg", radius)
        df.insert(1, "patch_full_width_deg", 2.0 * radius)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _plot_ci_series(ax: plt.Axes, df: pd.DataFrame, *, y: str, label: str | None = None, color: str = "C0") -> None:
    work = df.sort_values("patch_radius_deg")
    x = work["patch_full_width_deg"].to_numpy(dtype=float)
    vals = work[y].to_numpy(dtype=float)
    lo = work["ci95_low"].to_numpy(dtype=float)
    hi = work["ci95_high"].to_numpy(dtype=float)
    yerr = np.vstack([vals - lo, hi - vals])
    ax.errorbar(x, vals, yerr=yerr, marker="o", lw=1.7, capsize=3, label=label, color=color)


def plot_local_key_effects(local_key: pd.DataFrame, out_path: Path) -> None:
    pairs = list(LOCAL_KEY_ROWS)
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.8), sharex=True)
    for ax, (feature, metric) in zip(axes.ravel(), pairs, strict=True):
        sub = local_key[(local_key["feature"] == feature) & (local_key["eye_metric"] == metric)]
        if sub.empty:
            ax.set_visible(False)
            continue
        row0 = sub.iloc[0]
        _plot_ci_series(ax, sub, y="median_delta", color="#315f72")
        ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.6)
        ax.set_title(f"{row0['feature_label']}\n{row0['eye_metric_label']}", fontsize=9)
        ax.grid(axis="y", color="0.88", lw=0.8)
        ax.set_xlim(0.35, 2.1)
    for ax in axes[-1, :]:
        ax.set_xlabel("image patch full width (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("high - low pole delta")
    fig.suptitle("Patch-radius sensitivity of local-feature pole effects", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_sf_key_slopes(sf_key: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=True)
    colors = {
        "abs_power_4_8_cpd": "#7a5195",
        "abs_power_8plus_cpd": "#2f8f63",
    }
    for ax, metric in zip(axes.ravel(), SF_KEY_METRICS, strict=True):
        sub_metric = sf_key[sf_key["eye_metric"] == metric]
        for feature in SF_KEY_FEATURES:
            sub = sub_metric[sub_metric["feature"] == feature]
            if sub.empty:
                continue
            label = str(sub.iloc[0]["feature_label"])
            _plot_ci_series(
                ax,
                sub,
                y="controlled_beta_z_median",
                label=label,
                color=colors.get(feature, "C0"),
            )
        label = str(sub_metric.iloc[0]["eye_metric_label"]) if not sub_metric.empty else metric
        ax.set_title(label, fontsize=10)
        ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.6)
        ax.grid(axis="y", color="0.88", lw=0.8)
        ax.set_xlim(0.35, 2.1)
    for ax in axes[-1, :]:
        ax.set_xlabel("image patch full width (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("controlled beta")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Patch-radius sensitivity of controlled spatial-frequency slopes", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(window_counts: pd.DataFrame, local_key: pd.DataFrame, sf_key: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# BackImage patch-radius sensitivity",
        "",
        "Patch radius is the half-width passed to the image-feature extractor. Full patch width is `2 * radius`.",
        "",
        "## Window counts",
        "",
        "| radius deg | full width deg | valid windows | contamination excluded | feature failures |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in window_counts.to_dict("records"):
        lines.append(
            f"| {row['patch_radius_deg']:.2f} | {row['patch_full_width_deg']:.2f} | "
            f"{int(row['n_windows'])} | {int(row['n_excluded_patch_contamination_windows'])} | "
            f"{int(row['n_failed_image_feature_windows'])} |"
        )

    lines.extend(
        [
            "",
            "## Key local-feature pole effects",
            "",
            "| radius | feature | eye metric | delta | CI |",
            "|---:|---|---|---:|---|",
        ]
    )
    focus_local = local_key[
        local_key["eye_metric"].isin(
            ["drift_edge_cos2", "rms_across_arcmin", "rms_delta_along_minus_across_arcmin"]
        )
    ].copy()
    for row in focus_local.sort_values(["feature", "eye_metric", "patch_radius_deg"]).to_dict("records"):
        lines.append(
            f"| {row['patch_radius_deg']:.2f} | {row['feature_label']} | {row['eye_metric_label']} | "
            f"{row['median_delta']:.4g} | [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] |"
        )

    lines.extend(
        [
            "",
            "## Controlled SF slopes",
            "",
            "| radius | band | eye metric | beta | CI |",
            "|---:|---|---|---:|---|",
        ]
    )
    for row in sf_key.sort_values(["feature", "eye_metric", "patch_radius_deg"]).to_dict("records"):
        lines.append(
            f"| {row['patch_radius_deg']:.2f} | {row['feature_label']} | {row['eye_metric_label']} | "
            f"{row['controlled_beta_z_median']:.4g} | [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}] |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def run(args: argparse.Namespace) -> Path:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    window_counts = _load_window_counts()
    local = _load_local(root)
    sf = _load_sf(root)

    local_key = local[
        local.apply(lambda r: (r["feature"], r["eye_metric"]) in LOCAL_KEY_ROWS, axis=1)
    ].copy()
    sf_key = sf[sf["feature"].isin(SF_KEY_FEATURES) & sf["eye_metric"].isin(SF_KEY_METRICS)].copy()

    window_counts.to_csv(root / "patch_radius_window_counts.csv", index=False)
    local_key.to_csv(root / "patch_radius_key_local_feature_effects.csv", index=False)
    sf_key.to_csv(root / "patch_radius_key_sf_controlled_slopes.csv", index=False)

    plot_local_key_effects(local_key, root / "patch_radius_key_local_feature_effects")
    plot_sf_key_slopes(sf_key, root / "patch_radius_key_sf_controlled_slopes")
    write_report(window_counts, local_key, sf_key, root / "summary_report.md")
    (root / "run_metadata.json").write_text(
        json.dumps({"config": asdict(PatchRadiusSummaryConfig(root=str(root)))}, indent=2) + "\n"
    )
    print(f"Wrote BackImage patch-radius sensitivity summary to {root}")
    return root


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
