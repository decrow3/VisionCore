"""Map-first checkpoint: pooled versus image-specific horizontal bias."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review")
WINDOWS = ROOT / "backimage_contour_motion_component_plots_v1" / "contour_motion_component_windows.csv"
TRIALS = ROOT / "backimage_trial_scale_audit" / "backimage_trial_scale_audit.csv"
OUT = ROOT / "backimage_image_eye_horizontal_bias_checkpoint"


def prepare(windows_path: Path, trials_path: Path, coherence_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = pd.read_csv(windows_path)
    trials = pd.read_csv(trials_path, usecols=["session", "trial_idx", "image_file"])
    values = windows.merge(trials, on=["session", "trial_idx"], validate="many_to_one")
    values = values[values.image_orientation_coherence >= coherence_threshold].copy()
    values["image_horizontal_cos2"] = np.cos(2 * np.deg2rad(values.image_edge_axis_deg))
    values["fem_horizontal_cos2"] = np.cos(2 * np.deg2rad(values.drift_orientation_deg))
    values["fem_horizontal_minus_vertical_arcmin"] = 60 * (
        np.sqrt(np.maximum(values.cov_xx_deg2, 0)) - np.sqrt(np.maximum(values.cov_yy_deg2, 0))
    )
    metrics = ["image_horizontal_cos2", "fem_horizontal_cos2", "fem_horizontal_minus_vertical_arcmin"]
    trial = values.groupby(["subject", "session", "trial_idx", "image_file"], sort=False)[metrics].median().reset_index()
    session_image = trial.groupby(["subject", "session", "image_file"], sort=False)[metrics].median().reset_index()
    image = session_image.groupby("image_file", sort=False)[metrics].median().reset_index()
    image["n_sessions"] = image.image_file.map(session_image.groupby("image_file").session.nunique())
    image["n_trials"] = image.image_file.map(trial.groupby("image_file").size())
    image["n_windows"] = image.image_file.map(values.groupby("image_file").size())
    return values, session_image, image


def plot(values: pd.DataFrame, session_image: pd.DataFrame, image_table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    ax = axes[0, 0]
    bins = np.linspace(-90, 90, 25)
    image_axis = ((values.image_edge_axis_deg + 90) % 180) - 90
    fem_axis = ((values.drift_orientation_deg + 90) % 180) - 90
    ax.hist(image_axis, bins=bins, density=True, histtype="step", lw=2.2, color="#4e79a7", label="local image contour")
    ax.hist(fem_axis, bins=bins, density=True, histtype="step", lw=2.2, color="#e15759", label="FEM major axis")
    ax.axvline(0, color="0.5", lw=1)
    ax.set_xlabel("screen-frame axial orientation (deg; 0° = horizontal)")
    ax.set_ylabel("density")
    ax.set_title("A  Pooled marginals are both horizontal")
    ax.legend(frameon=False)

    ordered = image_table.sort_values("image_horizontal_cos2").reset_index(drop=True)
    y = np.arange(len(ordered))
    ax = axes[0, 1]
    ax.hlines(y, ordered.image_horizontal_cos2, ordered.fem_horizontal_cos2, color="0.78", lw=1)
    ax.scatter(ordered.image_horizontal_cos2, y, color="#4e79a7", s=25, label="sampled image contours")
    ax.scatter(ordered.fem_horizontal_cos2, y, color="#e15759", s=25, label="associated FEM axes")
    ax.axvline(0, color="0.5", lw=1)
    ax.set_yticks(y, ordered.image_file.str.replace(r"\.(JPG|jpg)$", "", regex=True), fontsize=7)
    ax.set_xlabel("median cos(2θ); +1 horizontal, −1 vertical")
    ax.set_title("B  Image-by-image biases often dissociate")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1, 0]
    ax.scatter(image_table.image_horizontal_cos2, image_table.fem_horizontal_cos2,
               s=25 + 3 * np.sqrt(image_table.n_windows), color="#6f4e7c", alpha=0.85)
    for row in image_table.itertuples():
        label = Path(row.image_file).stem.replace("Rochester_", "R_").replace("Hawaii_", "H_")
        ax.annotate(label, (row.image_horizontal_cos2, row.fem_horizontal_cos2), xytext=(3, 2),
                    textcoords="offset points", fontsize=6.2, alpha=0.85)
    rho = spearmanr(image_table.image_horizontal_cos2, image_table.fem_horizontal_cos2).statistic
    ax.axhline(0, color="0.6", lw=1)
    ax.axvline(0, color="0.6", lw=1)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("image-specific horizontal contour score")
    ax.set_ylabel("associated FEM horizontal-axis score")
    ax.set_title(f"C  Across 27 images: Spearman ρ = {rho:+.2f}")
    ax.grid(alpha=0.2)

    centered = session_image.copy()
    for col in ["image_horizontal_cos2", "fem_horizontal_cos2"]:
        centered[col] -= centered.groupby("session")[col].transform("mean")
    rho_within = spearmanr(centered.image_horizontal_cos2, centered.fem_horizontal_cos2).statistic
    ax = axes[1, 1]
    for subject, block, color in [("Allen", centered[centered.subject == "Allen"], "#59a14f"),
                                  ("Logan", centered[centered.subject == "Logan"], "#f28e2b")]:
        ax.scatter(block.image_horizontal_cos2, block.fem_horizontal_cos2, s=16, alpha=0.45, color=color, label=subject)
    ax.axhline(0, color="0.6", lw=1)
    ax.axvline(0, color="0.6", lw=1)
    ax.set_xlabel("image contour score, demeaned within session")
    ax.set_ylabel("FEM axis score, demeaned within session")
    ax.set_title(f"D  Same-session image cells (n={len(centered)}): ρ = {rho_within:+.2f}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.suptitle("BackImage horizontal bias: marginal overlap does not imply image-specific matching\n"
                 "coherence ≥ 0.3; medians windows→trial→session×image→image", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=WINDOWS)
    parser.add_argument("--trials", type=Path, default=TRIALS)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--coherence-threshold", type=float, default=0.3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    values, session_image, image_table = prepare(args.windows, args.trials, args.coherence_threshold)
    image_table.sort_values("image_horizontal_cos2").to_csv(args.out_dir / "per_image_horizontal_bias.csv", index=False)
    session_image.to_csv(args.out_dir / "per_session_image_horizontal_bias.csv", index=False)
    plot(values, session_image, image_table, args.out_dir / "backimage_image_eye_horizontal_bias_checkpoint.png")
    print("windows", len(values), "session-image cells", len(session_image), "images", len(image_table))
    print("pooled image cos2", values.image_horizontal_cos2.mean(), "pooled FEM cos2", values.fem_horizontal_cos2.mean())
    print("per-image Spearman", spearmanr(image_table.image_horizontal_cos2, image_table.fem_horizontal_cos2).statistic)
    centered = session_image.copy()
    for col in ["image_horizontal_cos2", "fem_horizontal_cos2"]:
        centered[col] -= centered.groupby("session")[col].transform("mean")
    print("within-session Spearman", spearmanr(centered.image_horizontal_cos2, centered.fem_horizontal_cos2).statistic)


if __name__ == "__main__":
    main()
