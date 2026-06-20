"""Build the cache-only Figure 4C joint-observer accuracy panel."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OBSERVER_SUMMARY = (
    REPO_ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1"
    / "observer_summary.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "declan/figure4_active_sensing_atlas/figures"

CANDIDATE_LABELS = {
    "hard_negative_structure": "Hard negatives",
    "matched_static_response": "Matched static",
}
PRIOR_LABELS = {
    "empirical": "Joint empirical",
    "ou": "Joint OU",
}
COLORS = {
    "known": "#222222",
    "zero": "#8e9aa6",
    "empirical": "#2f8f6a",
    "ou": "#3366aa",
}


def _load_primary_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "candidate_set_mode",
        "prior_family",
        "observation_scale",
        "prior_scale",
        "likelihood_scale",
        "known_eye_accuracy",
        "zero_eye_accuracy",
        "joint_eye_accuracy",
        "median_N_eff_fraction",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    work = df[
        (df["trajectory_prior_mode"] == "leave_one_out")
        & (df["likelihood_scale"].astype(float) == 1.0)
        & (df["observation_scale"].astype(float).isin([0.5, 1.0]))
        & (df["prior_scale"].astype(float).isin([0.5, 1.0]))
    ].copy()
    work = work[work["observation_scale"].astype(float) == work["prior_scale"].astype(float)].copy()
    work["scale_label"] = work["observation_scale"].astype(float).map({0.5: "0.5x", 1.0: "1.0x"})
    return work.sort_values(["candidate_set_mode", "observation_scale", "prior_family"])


def _plot_panel(rows: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True, constrained_layout=True)
    for ax, candidate_mode in zip(axes, ["hard_negative_structure", "matched_static_response"], strict=True):
        sub = rows[rows["candidate_set_mode"] == candidate_mode].copy()
        scales = [0.5, 1.0]
        x = range(len(scales))

        known = []
        zero = []
        empirical = []
        ou = []
        for scale in scales:
            scale_rows = sub[sub["observation_scale"].astype(float) == scale]
            known.append(float(scale_rows["known_eye_accuracy"].iloc[0]))
            zero.append(float(scale_rows["zero_eye_accuracy"].iloc[0]))
            empirical.append(float(scale_rows[scale_rows["prior_family"] == "empirical"]["joint_eye_accuracy"].iloc[0]))
            ou.append(float(scale_rows[scale_rows["prior_family"] == "ou"]["joint_eye_accuracy"].iloc[0]))

        ax.plot(x, known, marker="o", color=COLORS["known"], linewidth=2.0, label="Known-eye")
        ax.plot(x, zero, marker="o", color=COLORS["zero"], linewidth=2.0, label="Zero-eye")
        ax.plot(x, empirical, marker="o", color=COLORS["empirical"], linewidth=2.0, label=PRIOR_LABELS["empirical"])
        ax.plot(x, ou, marker="o", color=COLORS["ou"], linewidth=2.0, label=PRIOR_LABELS["ou"])
        ax.set_title(CANDIDATE_LABELS[candidate_mode], fontsize=10)
        ax.set_xticks(list(x), ["0.5x", "1.0x"])
        ax.set_xlabel("motion scale")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", color="#d8dde3", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

        if candidate_mode == "matched_static_response":
            ax.annotate(
                "joint recovers\nlatent-pose loss",
                xy=(1, empirical[-1]),
                xytext=(0.25, 0.55),
                textcoords="data",
                arrowprops={"arrowstyle": "->", "color": "#56616b", "linewidth": 1.0},
                fontsize=8,
                color="#303840",
            )

    axes[0].set_ylabel("image-identification accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Trajectory marginalization rescues image identity under latent eye motion", fontsize=11)

    png_path = out_dir / "panel_C_joint_observer_accuracy.png"
    pdf_path = out_dir / "panel_C_joint_observer_accuracy.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _write_caption(rows: pd.DataFrame, out_dir: Path) -> None:
    key = rows[
        (rows["candidate_set_mode"] == "matched_static_response")
        & (rows["observation_scale"].astype(float) == 1.0)
    ]
    emp = key[key["prior_family"] == "empirical"].iloc[0]
    ou = key[key["prior_family"] == "ou"].iloc[0]
    caption = f"""# Panel C Joint Observer

Cache-only panel generated from:

```text
{DEFAULT_OBSERVER_SUMMARY}
```

At matched-static 1.0x, known-eye accuracy was {emp.known_eye_accuracy:.3f},
zero-eye accuracy was {emp.zero_eye_accuracy:.3f}, joint-eye accuracy was
{emp.joint_eye_accuracy:.3f} with the empirical trajectory prior and
{ou.joint_eye_accuracy:.3f} with the OU prior. Median N_eff / K was
{emp.median_N_eff_fraction:.3f} for empirical and {ou.median_N_eff_fraction:.3f}
for OU. This panel supports the claim that trajectory marginalization over
exact natural-image response tables recovers image identity lost by a zero-eye
observer. It does not by itself identify compact geometry as the mechanism.
"""
    (out_dir / "panel_C_joint_observer_accuracy_caption.md").write_text(caption)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer-summary", type=Path, default=DEFAULT_OBSERVER_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = _load_primary_rows(args.observer_summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_dir / "panel_C_joint_observer_values.csv", index=False)
    _plot_panel(rows, args.out_dir)
    _write_caption(rows, args.out_dir)


if __name__ == "__main__":
    main()
