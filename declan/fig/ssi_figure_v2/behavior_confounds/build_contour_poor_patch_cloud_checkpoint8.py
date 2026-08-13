#!/usr/bin/env python3
"""Checkpoint 8: inspect what near-zero orientation coherence selects.

This targeted map-first figure selects examples using image measurements only.
It pairs each gaze-centered BackImage patch with the recorded, mean-centered eye
path in absolute screen coordinates.  No contour axis is drawn because the
purpose is to inspect patches for which that axis is not reliable.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas
from declan.fixation_statistics_by_stimulus.plot_backimage_contour_motion_components import (
    _patch_orientation_features,
    _window_trace,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
    / "contour_motion_component_windows.csv"
)
OUT_DIR = (
    ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "contour_poor_patch_cloud_checkpoint8_v1"
)
SUBJECTS = ("Allen", "Logan")
ROLE_ORDER = ("near_zero_textured", "typical_low_coherence", "low_gradient_control")
ROLE_LABELS = {
    "near_zero_textured": "Near-zero coherence,\ntextured",
    "typical_low_coherence": "Typical low coherence,\ntextured",
    "low_gradient_control": "Lower-gradient\ncontrol",
}
ROLE_RULES = {
    "near_zero_textured": "coherence<=0.02; energy>=subject median among coherence<=0.05",
    "typical_low_coherence": "0.02<coherence<=0.05; subject energy IQR",
    "low_gradient_control": "coherence<=0.05; energy<=subject 25th percentile",
}
COLORS = {"Allen": "#3B6FB6", "Logan": "#C56A2D"}


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidates() -> pd.DataFrame:
    values = pd.read_csv(SOURCE)
    numeric = (
        "image_orientation_coherence",
        "image_gradient_energy",
        "image_patch_std",
        "image_patch_fraction_background",
        "image_patch_center_x_px",
        "image_patch_center_y_px",
        "image_patch_radius_px",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
        "mean_x_deg",
        "mean_y_deg",
        "rms_radius_deg",
        "anisotropy",
    )
    ok = values["subject"].isin(SUBJECTS) & values["image_feature_ok"].astype(bool)
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce")
        ok &= np.isfinite(values[column])
    ok &= values["image_orientation_coherence"].le(0.05)
    ok &= values["image_patch_fraction_background"].le(0.05)
    return values.loc[ok].copy().reset_index(drop=True)


def role_mask(block: pd.DataFrame, role: str) -> pd.Series:
    coherence = block["image_orientation_coherence"]
    energy = block["image_gradient_energy"]
    q25, q50, q75 = energy.quantile([0.25, 0.50, 0.75])
    if role == "near_zero_textured":
        return coherence.le(0.02) & energy.ge(q50)
    if role == "typical_low_coherence":
        return coherence.gt(0.02) & coherence.le(0.05) & energy.between(q25, q75)
    if role == "low_gradient_control":
        return coherence.le(0.05) & energy.le(q25)
    raise ValueError(role)


def representative_score(candidates: pd.DataFrame) -> pd.Series:
    """Robust distance to the image-feature medoid; behavior is not used."""
    columns = ("image_orientation_coherence", "image_gradient_energy", "image_patch_std")
    score = pd.Series(0.0, index=candidates.index)
    for column in columns:
        transformed = np.log1p(candidates[column]) if column == "image_gradient_energy" else candidates[column]
        median = float(transformed.median())
        iqr = float(transformed.quantile(0.75) - transformed.quantile(0.25))
        scale = iqr if iqr > 0 else max(float(transformed.std()), 1e-12)
        score += np.abs(transformed - median) / scale
    return score


def select_examples(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections: list[pd.Series] = []
    support_rows: list[dict[str, object]] = []
    used_trials: set[tuple[str, str, int]] = set()
    for role in ROLE_ORDER:
        for subject in SUBJECTS:
            subject_values = values[values["subject"].eq(subject)].copy()
            candidates = subject_values[role_mask(subject_values, role)].copy()
            support_rows.append(
                {
                    "example_role": role,
                    "subject": subject,
                    "eligible_windows": int(len(candidates)),
                    "eligible_trials": int(candidates.groupby(["session", "trial_idx"]).ngroups),
                    "eligible_sessions": int(candidates["session"].nunique()),
                }
            )
            candidates["selection_score"] = representative_score(candidates)
            candidates = candidates.sort_values(
                ["selection_score", "session", "trial_idx", "global_start"], kind="stable"
            )
            keep = []
            for index, row in candidates.iterrows():
                trial_key = (subject, str(row["session"]), int(row["trial_idx"]))
                if trial_key not in used_trials:
                    keep.append(index)
            if not keep:
                raise RuntimeError(f"No unused-trial candidate for {subject} {role}")
            selected = candidates.loc[keep[0]].copy()
            used_trials.add((subject, str(selected["session"]), int(selected["trial_idx"])))
            selected["example_role"] = role
            selected["selection_rule"] = ROLE_RULES[role]
            selected["eligible_candidates"] = int(len(candidates))
            selected["selection_uses_behavior"] = False
            selections.append(selected)
    return pd.DataFrame(selections).reset_index(drop=True), pd.DataFrame(support_rows)


def extract_patch(row: pd.Series) -> tuple[np.ndarray, float, dict[str, float]]:
    canvas, ppd, _shape = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    cx = int(round(float(row["image_patch_center_x_px"])))
    cy = int(round(float(row["image_patch_center_y_px"])))
    radius = int(round(float(row["image_patch_radius_px"])))
    patch = np.asarray(canvas[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1])
    if patch.size == 0:
        raise RuntimeError(f"Empty patch for {row['session']} trial {row['trial_idx']}")
    return patch, float(ppd), _patch_orientation_features(patch)


def add_covariance_ellipse(ax: plt.Axes, row: pd.Series) -> None:
    covariance = np.asarray(
        [
            [float(row["cov_xx_deg2"]), float(row["cov_xy_deg2"])],
            [float(row["cov_xy_deg2"]), float(row["cov_yy_deg2"])],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ax.add_patch(
        Ellipse(
            (0, 0),
            width=4 * math.sqrt(max(float(eigenvalues[0]), 0.0)),
            height=4 * math.sqrt(max(float(eigenvalues[1]), 0.0)),
            angle=angle,
            facecolor="none",
            edgecolor="#121619",
            lw=1.15,
            zorder=5,
        )
    )


def render(
    selected: pd.DataFrame,
    patches: list[np.ndarray],
    traces: list[np.ndarray],
    common_limit: float,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 4, figsize=(12.0, 9.4), constrained_layout=False)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.075, top=0.80, hspace=0.43, wspace=0.34)
    for role_index, role in enumerate(ROLE_ORDER):
        for subject_index, subject in enumerate(SUBJECTS):
            match = selected.index[
                selected["example_role"].eq(role) & selected["subject"].eq(subject)
            ]
            index = int(match[0])
            row = selected.loc[index]
            patch = patches[index]
            trace = traces[index]
            centered = trace - np.mean(trace, axis=0, keepdims=True)
            patch_ax = axes[role_index, subject_index * 2]
            cloud_ax = axes[role_index, subject_index * 2 + 1]

            patch_ax.imshow(patch, cmap="gray", origin="upper", interpolation="nearest")
            cy, cx = (np.asarray(patch.shape) - 1) / 2
            patch_ax.scatter([cx], [cy], marker="+", s=23, linewidths=1.1, c="#D93A32")
            patch_ax.set_xticks([])
            patch_ax.set_yticks([])
            patch_ax.set_title(
                f"coh={float(row['image_orientation_coherence']):.3f}  "
                f"energy={float(row['image_gradient_energy']):,.0f}\n"
                f"contrast={float(row['image_patch_rms_contrast']):.2f}",
                fontsize=7.7,
            )

            times = np.arange(len(centered))
            cloud_ax.plot(centered[:, 0], centered[:, 1], color="#7B8187", lw=0.65, zorder=1)
            cloud_ax.scatter(
                centered[:, 0], centered[:, 1], c=times, cmap="viridis", s=7, alpha=0.72, zorder=2
            )
            cloud_ax.scatter(centered[0, 0], centered[0, 1], c="#2878B5", s=22, zorder=6)
            cloud_ax.scatter(
                centered[-1, 0], centered[-1, 1], c="#D44B3E", marker="s", s=22, zorder=6
            )
            add_covariance_ellipse(cloud_ax, row)
            cloud_ax.axhline(0, color="#C1C6CB", lw=0.6)
            cloud_ax.axvline(0, color="#C1C6CB", lw=0.6)
            cloud_ax.set_xlim(-common_limit, common_limit)
            cloud_ax.set_ylim(-common_limit, common_limit)
            cloud_ax.set_aspect("equal")
            cloud_ax.grid(color="#E0E3E6", lw=0.45)
            hmv = 60.0 * (
                math.sqrt(max(float(row["cov_xx_deg2"]), 0.0))
                - math.sqrt(max(float(row["cov_yy_deg2"]), 0.0))
            )
            cloud_ax.set_title(
                f"cloud axis={float(row['drift_orientation_deg']):+.0f}°  "
                f"aniso={float(row['anisotropy']):.2f}\nH−V RMS={hmv:+.2f} arcmin",
                fontsize=7.7,
            )
            if role_index == 2:
                cloud_ax.set_xlabel("screen x, centered (deg)")
            if subject_index == 0:
                cloud_ax.set_ylabel("screen y, centered (deg)")

            if subject_index == 0:
                patch_ax.set_ylabel(
                    ROLE_LABELS[role], rotation=0, ha="right", va="center", labelpad=12,
                    fontsize=8.5, weight="semibold",
                )

    fig.text(0.375, 0.865, "Allen", ha="center", fontsize=11, color=COLORS["Allen"], weight="bold")
    fig.text(0.785, 0.865, "Logan", ha="center", fontsize=11, color=COLORS["Logan"], weight="bold")
    fig.text(0.265, 0.835, "GAZE-CENTERED PATCH", ha="center", fontsize=8.2, weight="bold")
    fig.text(0.475, 0.835, "RECORDED CLOUD: SCREEN AXES", ha="center", fontsize=8.2, weight="bold")
    fig.text(0.675, 0.835, "GAZE-CENTERED PATCH", ha="center", fontsize=8.2, weight="bold")
    fig.text(0.885, 0.835, "RECORDED CLOUD: SCREEN AXES", ha="center", fontsize=8.2, weight="bold")
    fig.suptitle(
        "Checkpoint 8: what does near-zero local orientation coherence look like?\n"
        "Examples are selected from image features only; all cloud panels share one spatial scale",
        y=0.98,
        fontsize=12.1,
        weight="bold",
    )
    fig.text(
        0.985,
        0.022,
        "Patch cross = mean gaze. Cloud points run blue→yellow with first/last marked by a blue circle/red square; "
        "black ellipse = 2 SD. No contour axis is shown because it is unreliable at low coherence.",
        ha="right",
        fontsize=6.8,
        color="#4E545A",
    )
    return fig


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values = load_candidates()
    selected, support = select_examples(values)

    patches: list[np.ndarray] = []
    traces: list[np.ndarray] = []
    trace_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    max_abs = 0.0
    for index, row in selected.iterrows():
        patch, ppd, recomputed = extract_patch(row)
        trace = np.asarray(_window_trace(row), dtype=float)
        if trace.ndim != 2 or trace.shape[1] != 2 or not np.isfinite(trace).all():
            raise RuntimeError(f"Invalid trace for {row['session']} trial {row['trial_idx']}")
        centered = trace - np.mean(trace, axis=0, keepdims=True)
        max_abs = max(max_abs, float(np.max(np.abs(centered))))
        patches.append(patch)
        traces.append(trace)
        for sample_index, point in enumerate(centered):
            trace_rows.append(
                {
                    "example_role": row["example_role"],
                    "subject": row["subject"],
                    "session": row["session"],
                    "trial_idx": int(row["trial_idx"]),
                    "sample_index": sample_index,
                    "centered_x_deg": float(point[0]),
                    "centered_y_deg": float(point[1]),
                }
            )
        audit_rows.append(
            {
                "selection_index": index,
                "ppd": ppd,
                "stored_coherence": float(row["image_orientation_coherence"]),
                "recomputed_coherence": float(recomputed["image_orientation_coherence"]),
                "coherence_abs_error": abs(
                    float(row["image_orientation_coherence"])
                    - float(recomputed["image_orientation_coherence"])
                ),
                "trace_mean_x_error_deg": float(np.mean(trace[:, 0]) - row["mean_x_deg"]),
                "trace_mean_y_error_deg": float(np.mean(trace[:, 1]) - row["mean_y_deg"]),
                "trace_rms_recomputed_deg": float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))),
                "trace_rms_stored_deg": float(row["rms_radius_deg"]),
            }
        )

    common_limit = max(0.12, 1.08 * max_abs)
    figure = render(selected, patches, traces, common_limit)
    outputs = {}
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"contour_poor_patch_cloud_examples.{suffix}"
        figure.savefig(path, transparent=False, **kwargs)
        outputs[suffix] = str(path.relative_to(ROOT))
    plt.close(figure)

    selection_columns = [
        "example_role",
        "selection_rule",
        "selection_score",
        "selection_uses_behavior",
        "eligible_candidates",
        "subject",
        "session",
        "trial_idx",
        "global_start",
        "global_stop",
        "phase",
        "image_orientation_coherence",
        "image_gradient_energy",
        "image_patch_std",
        "image_patch_rms_contrast",
        "image_patch_fraction_background",
        "mean_x_deg",
        "mean_y_deg",
        "abs_mean_radius_deg",
        "rms_radius_deg",
        "anisotropy",
        "drift_orientation_deg",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
    ]
    selected[selection_columns].to_csv(OUT_DIR / "selected_examples.csv", index=False)
    support.to_csv(OUT_DIR / "selection_support.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(OUT_DIR / "selected_trace_values.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_DIR / "recomputation_audit.csv", index=False)

    report = [
        "# Near-zero local orientation coherence: map-first checkpoint 8",
        "",
        "This checkpoint asks what the low-coherence image class looks like before using it as",
        "evidence for an image-independent movement prior. Six examples were selected using only",
        "image coherence, gradient energy, and patch contrast; behavior did not enter selection.",
        "",
        "The sheet separates the observed image patch from the recorded, mean-centered eye path.",
        "All eye-path panels use the same absolute screen axes and spatial scale. No local contour",
        "axis is drawn because an axis estimated at coherence <=0.05 is not treated as meaningful.",
        "",
        "Interpretation remains a human checkpoint: low coherence can mean mixed, curved, or",
        "crossing structure, as well as weak gradients. It does not literally prove that the patch",
        "contains no contours. The low-gradient row is included to expose that distinction.",
        "",
        f"Common cloud half-range: {common_limit:.4f} deg.",
        f"Maximum patch-coherence recomputation error: {audit['coherence_abs_error'].max():.8f}.",
        "",
    ]
    (OUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "stage": "map-first checkpoint 8; concrete low-coherence patch and drift-cloud inspection",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "primary_coherence_max": 0.05,
        "strict_near_zero_max": 0.02,
        "max_patch_fraction_background": 0.05,
        "selection_uses_behavior": False,
        "common_cloud_half_range_deg": common_limit,
        "role_order": list(ROLE_ORDER),
        "role_rules": ROLE_RULES,
        "outputs": outputs,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(ROOT / outputs["png"])
    print(OUT_DIR / "selected_examples.csv")
    print(OUT_DIR / "summary_report.md")


if __name__ == "__main__":
    main()
