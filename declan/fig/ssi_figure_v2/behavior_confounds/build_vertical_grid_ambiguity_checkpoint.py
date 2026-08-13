#!/usr/bin/env python3
"""Diagnose gridlike false positives in the expanded vertical-contour cohort."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig.ssi_figure_v2.behavior_confounds.build_contour_axis_consensus_examples import (
    ROOT,
    analyze_direct,
    axial_distance_deg,
    load_candidates,
)


SOURCE_DIR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "contour_axis_consensus_vertical_expansion_v1"
)
OUT_DIR = (
    ROOT / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "vertical_grid_ambiguity_checkpoint_v1"
)
ROLES = {
    ("Allen", 1): "clean_vertical_control",
    ("Allen", 8): "user_flagged_grid",
    ("Logan", 1): "clean_vertical_control",
    ("Logan", 2): "user_flagged_grid",
    ("Logan", 3): "user_flagged_grid",
}


def orientation_profile(patch01: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    gx = cv2.Scharr(patch01, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Scharr(patch01, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REPLICATE)
    contour = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
    energy = gx * gx + gy * gy
    bins = np.arange(0.0, 185.0, 5.0)
    profile, _ = np.histogram(contour[mask], bins=bins, weights=energy[mask])
    profile = profile / max(float(np.sum(profile)), 1e-12)
    centers = (bins[:-1] + bins[1:]) / 2.0
    return centers, profile, float(np.sum(energy[mask]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SOURCE_DIR / "selected_strict_consensus_examples.csv")
    sources = load_candidates(0, (90.0,))
    keys = ["subject", "session", "trial_idx"]
    rows = selected[keys + ["display_rank_within_cell"]].merge(sources, on=keys, validate="one_to_one")
    rows["selection_role"] = [ROLES.get((str(r.subject), int(r.display_rank_within_cell)), "") for r in rows.itertuples()]
    rows = rows[rows.selection_role.ne("")].sort_values(["subject", "display_rank_within_cell"])

    audit_rows = []
    panels = []
    for row in rows.itertuples(index=False):
        source = pd.Series(row._asdict())
        base, arrays = analyze_direct(source)
        centers, profile, _ = orientation_profile(np.asarray(arrays["patch01"]), np.asarray(arrays["circle_mask"]))
        ref = float(base["sobel_square_axis_deg"])
        parallel = float(np.sum(profile[np.asarray(axial_distance_deg(centers, ref)) <= 15.0]))
        orthogonal = float(np.sum(profile[np.asarray(axial_distance_deg(centers, ref + 90.0)) <= 15.0]))
        audit_rows.append({
            "subject": row.subject, "display_row": int(row.display_rank_within_cell),
            "session": row.session, "trial_idx": int(row.trial_idx), "image_file": row.image_file,
            "selection_role": row.selection_role, "sobel_square_coherence": base["sobel_square_coherence"],
            "parallel_band_energy_fraction": parallel, "orthogonal_band_energy_fraction": orthogonal,
            "parallel_to_orthogonal_band_ratio": parallel / max(orthogonal, 1e-12),
        })
        panels.append((row, base, arrays, centers, profile))

    fig, axes = plt.subplots(len(panels), 3, figsize=(10.5, 2.6 * len(panels)), constrained_layout=True)
    for i, (row, base, arrays, centers, profile) in enumerate(panels):
        patch = np.asarray(arrays["patch"])
        axes[i, 0].imshow(patch, cmap="gray", interpolation="nearest")
        axes[i, 0].scatter([(patch.shape[1] - 1) / 2], [(patch.shape[0] - 1) / 2], marker="+", c="#E23D3D")
        axes[i, 0].axis("off")
        axes[i, 0].set_ylabel(f"{row.subject} row {int(row.display_rank_within_cell)}\n{row.selection_role}", weight="bold", fontsize=8)
        axes[i, 1].imshow(arrays["canny_edges"], cmap="gray", interpolation="nearest")
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Canny edges; coherence={base['sobel_square_coherence']:.3f}", fontsize=8)
        axes[i, 2].bar(centers, profile, width=4.5, color="#6C8EBF")
        axes[i, 2].axvline(base["sobel_square_axis_deg"] % 180.0, color="#D1495B", lw=2, label="chosen axis")
        axes[i, 2].axvline((base["sobel_square_axis_deg"] + 90.0) % 180.0, color="#E6A23C", lw=1.5, ls="--", label="orthogonal")
        axes[i, 2].set(xlim=(0, 180), xlabel="local contour orientation (deg)", ylabel="Scharr energy fraction")
        if i == 0:
            axes[i, 2].legend(frameon=False, fontsize=7)
    for ax, title in zip(axes[0], ["raw fixation patch", "edge map", "orientation-energy distribution"], strict=True):
        ax.text(.5, 1.22, title, transform=ax.transAxes, ha="center", weight="bold", fontsize=9)
    fig.suptitle("Why estimator agreement admits gridlike vertical false positives", weight="bold")
    fig.savefig(OUT_DIR / "vertical_grid_ambiguity_diagnostic.png", dpi=220)
    fig.savefig(OUT_DIR / "vertical_grid_ambiguity_diagnostic.pdf")
    plt.close(fig)
    pd.DataFrame(audit_rows).to_csv(OUT_DIR / "flagged_and_control_patch_metrics.csv", index=False)

    full = pd.read_csv(SOURCE_DIR / "candidate_validation_audit.csv")
    strict = full[full.passes_strict_consensus.fillna(False)].copy()
    thresholds = []
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        kept = strict[strict.sobel_square_coherence.ge(threshold)]
        for subject in ("Allen", "Logan"):
            thresholds.append({"sobel_coherence_min": threshold, "subject": subject,
                               "n_retained": int(kept.subject.astype(str).eq(subject).sum())})
    pd.DataFrame(thresholds).to_csv(OUT_DIR / "proposed_purity_threshold_support.csv", index=False)
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print(pd.DataFrame(thresholds).to_string(index=False))


if __name__ == "__main__":
    main()
