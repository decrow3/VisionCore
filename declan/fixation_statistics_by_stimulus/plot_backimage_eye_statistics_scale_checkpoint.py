"""Map-first checkpoint for eye statistics across BackImage display sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review")
WINDOWS = ROOT / "backimage_image_structure_reviewed_v2_screenfiltered_yfix" / "backimage_image_fem_windows.csv"
TRIALS = ROOT / "backimage_trial_scale_audit" / "backimage_trial_scale_audit.csv"
OUT = ROOT / "backimage_eye_statistics_scale_checkpoint"

METRICS = [
    "rms_radius_deg",
    "anisotropy",
    "speed_median_deg_s",
    "path_length_deg_s",
    "direction_persistence",
    "return_to_center_strength",
    "fraction_within_0p10deg",
]
SCALE_ORDER = ["full", "32deg", "16deg", "8deg", "4deg"]
SCALE_VALUE = {"full": 0, "32deg": 1, "16deg": 2, "8deg": 3, "4deg": 4}
COLORS = {"32deg": "#4c78a8", "16deg": "#f58518", "8deg": "#54a24b", "4deg": "#e45756"}


def prepare(windows_path: Path, trials_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = pd.read_csv(windows_path)
    trials = pd.read_csv(trials_path)
    trials["scale_label"] = "full"
    reduced = ~np.isclose(trials["screen_area_fraction"], 1.0)
    trials.loc[reduced, "scale_label"] = trials.loc[reduced, "nominal_size_deg"].round().astype(int).astype(str) + "deg"
    cols = ["session", "trial_idx", "session_trial_order", "image_file", "scale_label", "nominal_size_deg", "screen_area_fraction"]
    joined = windows.merge(trials[cols], on=["session", "trial_idx"], validate="many_to_one")
    trial = joined.groupby(cols, dropna=False, sort=False)[METRICS].median().reset_index()

    full_ref = (
        trial[trial.scale_label == "full"]
        .groupby(["session", "image_file"], sort=False)[METRICS]
        .median()
        .add_suffix("_matched_full")
        .reset_index()
    )
    matched = trial[trial.scale_label != "full"].merge(full_ref, on=["session", "image_file"], how="inner", validate="many_to_one")
    for metric in METRICS:
        matched[f"delta_{metric}"] = matched[metric] - matched[f"{metric}_matched_full"]
    return trial, matched


def _heatmap_matrix(trial: pd.DataFrame, sessions: list[str], value: str, *, zscore: bool = False) -> np.ndarray:
    max_order = int(trial.session_trial_order.max())
    mat = np.full((len(sessions), max_order + 1), np.nan)
    for i, session in enumerate(sessions):
        sub = trial[trial.session == session]
        values = sub[value].to_numpy(float) if value != "scale_label" else sub.scale_label.map(SCALE_VALUE).to_numpy(float)
        if zscore:
            med = np.nanmedian(values)
            scale = np.nanmedian(np.abs(values - med)) * 1.4826
            values = (values - med) / scale if scale > 1e-12 else values * np.nan
        mat[i, sub.session_trial_order.to_numpy(int)] = values
    return mat


def plot(trial: pd.DataFrame, matched: pd.DataFrame, output: Path) -> None:
    sessions = sorted(trial.loc[trial.scale_label != "full", "session"].unique())
    subset = trial[trial.session.isin(sessions)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))

    ax = axes[0, 0]
    scale = _heatmap_matrix(subset, sessions, "scale_label")
    cmap = plt.matplotlib.colors.ListedColormap(["#d9d9d9", COLORS["32deg"], COLORS["16deg"], COLORS["8deg"], COLORS["4deg"]])
    ax.imshow(scale, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=4.5)
    ax.set_title("Recorded scale by trial order")
    ax.set_ylabel("session")
    ax.set_yticks(np.arange(len(sessions)), [s.replace("Allen_", "") for s in sessions], fontsize=8)
    ax.set_xlabel("BackImage trial order")

    for ax, metric, title in [
        (axes[0, 1], "rms_radius_deg", "RMS fixation spread\n(within-session robust z)"),
        (axes[0, 2], "speed_median_deg_s", "Median eye speed\n(within-session robust z)"),
    ]:
        mat = _heatmap_matrix(subset, sessions, metric, zscore=True)
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-2.5, vmax=2.5)
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xlabel("BackImage trial order")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    scatter_specs = [
        ("rms_radius_deg", "RMS spread (deg)"),
        ("speed_median_deg_s", "Median speed (deg/s)"),
        ("anisotropy", "Drift anisotropy"),
    ]
    for ax, (metric, label) in zip(axes[1], scatter_specs, strict=True):
        pooled = np.r_[matched[metric].to_numpy(float), matched[f"{metric}_matched_full"].to_numpy(float)]
        lo, hi = np.nanmin(pooled), np.nanmax(pooled)
        ax.plot([lo, hi], [lo, hi], color="0.55", lw=1, zorder=0)
        for scale_label in SCALE_ORDER[1:]:
            sub = matched[matched.scale_label == scale_label]
            ax.scatter(sub[f"{metric}_matched_full"], sub[metric], s=28, alpha=0.8, color=COLORS[scale_label], label=scale_label)
        ax.set_xlabel("same session×image, full-screen median")
        ax.set_ylabel(f"reduced trial: {label}")
        ax.set_title(f"Matched raw values: {label}")
        ax.grid(alpha=0.2)
    axes[1, 2].legend(frameon=False, fontsize=8)

    fig.suptitle("BackImage scaling checkpoint: late-block structure and same-image raw eye statistics", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=WINDOWS)
    parser.add_argument("--trials", type=Path, default=TRIALS)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trial, matched = prepare(args.windows, args.trials)
    trial.to_csv(args.out_dir / "trial_level_eye_statistics.csv", index=False)
    matched.to_csv(args.out_dir / "same_session_image_scaled_vs_full.csv", index=False)
    plot(trial, matched, args.out_dir / "backimage_eye_statistics_scale_checkpoint.png")
    print(trial.groupby("scale_label").agg(trials=("trial_idx", "size"), sessions=("session", "nunique"), images=("image_file", "nunique")).reindex(SCALE_ORDER).dropna().to_string())
    print("\nMatched reduced trials:", len(matched), "across", matched.session.nunique(), "sessions and", matched.image_file.nunique(), "images")


if __name__ == "__main__":
    main()
