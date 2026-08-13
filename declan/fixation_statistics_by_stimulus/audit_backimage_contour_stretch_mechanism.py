"""Map-first audit of BackImage aspect stretching and local contour axes.

The existing contour analysis correctly measures the recorded, resized canvas.
This diagnostic asks a counterfactual question: at the same native-image
location and vertical angular scale, what local contour axis would be obtained
if horizontal and vertical magnification were equal?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from DataYatesV1.exp.support import get_backimage_directory


BASE = Path("outputs/fixation_statistics_by_stimulus_all_sessions_after_review")
DEFAULT_WINDOWS = (
    BASE
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)
DEFAULT_SCALE_AUDIT = BASE / "backimage_trial_scale_audit" / "backimage_trial_scale_audit.csv"
DEFAULT_OUT = BASE / "backimage_contour_stretch_mechanism_checkpoint"
PATCH_RADIUS_DEG = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--scale-audit", type=Path, default=DEFAULT_SCALE_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def circular_axis_delta_deg(a_deg: float | np.ndarray, b_deg: float | np.ndarray) -> np.ndarray:
    return 0.5 * np.degrees(
        np.angle(np.exp(2j * np.radians(np.asarray(a_deg) - np.asarray(b_deg))))
    )


def patch_orientation(patch: np.ndarray) -> tuple[float, float]:
    """Return contour-axis angle in gaze coordinates and orientation coherence."""
    arr = np.asarray(patch, dtype=np.float64)
    gx = ndimage.sobel(arr, axis=1, mode="nearest")
    gy = ndimage.sobel(arr, axis=0, mode="nearest")
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    den = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2) / den if den > 0 else np.nan
    gradient_axis_array = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    edge_axis_gaze = -(gradient_axis_array + np.pi / 2.0)
    return float(np.degrees(edge_axis_gaze)), float(coherence)


def _sample_patch(
    source: np.ndarray,
    *,
    center_x_source: float,
    center_y_source: float,
    radius_display_px: int,
    source_px_per_display_px_x: float,
    source_px_per_display_px_y: float,
) -> tuple[np.ndarray, float]:
    offsets = np.arange(-radius_display_px, radius_display_px + 1, dtype=np.float64)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    xx = center_x_source + dx * source_px_per_display_px_x
    yy = center_y_source + dy * source_px_per_display_px_y
    inside = (xx >= 0) & (xx <= source.shape[1] - 1) & (yy >= 0) & (yy <= source.shape[0] - 1)
    patch = ndimage.map_coordinates(source, [yy, xx], order=1, mode="constant", cval=127.0)
    return patch, float(np.mean(inside))


def compute_window_audit(windows: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    trial_cols = [
        "session",
        "trial_idx",
        "image_file",
        "source_width_px",
        "source_height_px",
        "dest_x0_px",
        "dest_y0_px",
        "dest_width_px",
        "dest_height_px",
        "screen_width_px",
        "screen_height_px",
        "pix_per_deg",
        "screen_area_fraction",
        "relative_horizontal_magnification",
    ]
    merged = windows.reset_index(names="window_row").merge(
        trials[trial_cols], on=["session", "trial_idx"], how="left", validate="many_to_one"
    )
    if merged["image_file"].isna().any():
        raise ValueError("Scale audit is missing trials used by the reviewed BackImage window table.")

    image_root = get_backimage_directory()
    source_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    for rec in merged.to_dict("records"):
        image_file = str(rec["image_file"])
        if image_file not in source_cache:
            with Image.open(image_root / image_file) as image:
                source_cache[image_file] = np.asarray(image.convert("L"), dtype=np.float64)
        source = source_cache[image_file]
        sx = float(rec["dest_width_px"]) / float(rec["source_width_px"])
        sy = float(rec["dest_height_px"]) / float(rec["source_height_px"])
        ppd = float(rec["pix_per_deg"])
        radius_px = max(2, int(round(PATCH_RADIUS_DEG * ppd)))
        cx_screen = float(rec["screen_width_px"]) / 2.0 + float(rec["mean_x_deg"]) * ppd
        cy_screen = float(rec["screen_height_px"]) / 2.0 - float(rec["mean_y_deg"]) * ppd
        # PIL/texture resize center convention: display pixel center -> source pixel center.
        cx_source = ((cx_screen - float(rec["dest_x0_px"])) + 0.5) / sx - 0.5
        cy_source = ((cy_screen - float(rec["dest_y0_px"])) + 0.5) / sy - 0.5
        actual_patch, actual_inside = _sample_patch(
            source,
            center_x_source=cx_source,
            center_y_source=cy_source,
            radius_display_px=radius_px,
            source_px_per_display_px_x=1.0 / sx,
            source_px_per_display_px_y=1.0 / sy,
        )
        preserved_patch, preserved_inside = _sample_patch(
            source,
            center_x_source=cx_source,
            center_y_source=cy_source,
            radius_display_px=radius_px,
            source_px_per_display_px_x=1.0 / sy,
            source_px_per_display_px_y=1.0 / sy,
        )
        actual_reconstructed_axis, actual_reconstructed_coherence = patch_orientation(actual_patch)
        preserved_axis, preserved_coherence = patch_orientation(preserved_patch)
        actual_axis = float(rec["image_edge_axis_deg"])
        drift_axis = float(rec["drift_orientation_deg"])
        actual_alignment = float(np.cos(2.0 * np.radians(drift_axis - actual_axis)))
        preserved_alignment = float(np.cos(2.0 * np.radians(drift_axis - preserved_axis)))
        theta = np.radians(preserved_axis)
        predicted_axis = np.degrees(np.arctan2(sy * np.sin(theta), sx * np.cos(theta)))

        row = {
            "window_row": int(rec["window_row"]),
            "session": rec["session"],
            "trial_idx": int(rec["trial_idx"]),
            "global_start": int(rec["global_start"]),
            "global_stop": int(rec["global_stop"]),
            "phase": rec["phase"],
            "image_file": image_file,
            "mean_x_deg": float(rec["mean_x_deg"]),
            "mean_y_deg": float(rec["mean_y_deg"]),
            "screen_area_fraction": float(rec["screen_area_fraction"]),
            "relative_horizontal_magnification": float(rec["relative_horizontal_magnification"]),
            "source_center_x_px": cx_source,
            "source_center_y_px": cy_source,
            "patch_radius_display_px": radius_px,
            "source_px_per_display_px_x_actual": 1.0 / sx,
            "source_px_per_display_px_y": 1.0 / sy,
            "actual_patch_inside_source_fraction": actual_inside,
            "preserved_patch_inside_source_fraction": preserved_inside,
            "actual_axis_saved_deg": actual_axis,
            "actual_axis_reconstructed_deg": actual_reconstructed_axis,
            "actual_axis_reconstruction_error_deg": float(
                circular_axis_delta_deg(actual_reconstructed_axis, actual_axis)
            ),
            "actual_coherence_saved": float(rec["image_orientation_coherence"]),
            "actual_coherence_reconstructed": actual_reconstructed_coherence,
            "preserved_axis_deg": preserved_axis,
            "preserved_coherence": preserved_coherence,
            "axis_shift_actual_minus_preserved_deg": float(circular_axis_delta_deg(actual_axis, preserved_axis)),
            "axis_shift_geometry_prediction_deg": float(circular_axis_delta_deg(predicted_axis, preserved_axis)),
            "drift_axis_deg": drift_axis,
            "drift_alignment_actual_cos2": actual_alignment,
            "drift_alignment_preserved_cos2": preserved_alignment,
            "alignment_change_actual_minus_preserved": actual_alignment - preserved_alignment,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def selected_patches(selected: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    image_root = get_backimage_directory()
    source_cache: dict[str, np.ndarray] = {}
    patches: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row in selected.itertuples():
        if row.image_file not in source_cache:
            with Image.open(image_root / row.image_file) as image:
                source_cache[row.image_file] = np.asarray(image.convert("L"), dtype=np.float64)
        source = source_cache[row.image_file]
        actual, _ = _sample_patch(
            source,
            center_x_source=float(row.source_center_x_px),
            center_y_source=float(row.source_center_y_px),
            radius_display_px=int(row.patch_radius_display_px),
            source_px_per_display_px_x=float(row.source_px_per_display_px_x_actual),
            source_px_per_display_px_y=float(row.source_px_per_display_px_y),
        )
        preserved, _ = _sample_patch(
            source,
            center_x_source=float(row.source_center_x_px),
            center_y_source=float(row.source_center_y_px),
            radius_display_px=int(row.patch_radius_display_px),
            source_px_per_display_px_x=float(row.source_px_per_display_px_y),
            source_px_per_display_px_y=float(row.source_px_per_display_px_y),
        )
        patches[int(row.window_row)] = (preserved, actual)
    return patches


def select_examples(audit: pd.DataFrame) -> pd.DataFrame:
    work = audit[
        (audit["actual_patch_inside_source_fraction"] >= 0.999)
        & (audit["preserved_patch_inside_source_fraction"] >= 0.999)
        & (np.abs(audit["actual_axis_reconstruction_error_deg"]) <= 5.0)
    ].copy()
    work["minimum_coherence"] = work[["actual_coherence_saved", "preserved_coherence"]].min(axis=1)
    strong = work[work["minimum_coherence"] >= 0.20].copy()
    full = strong[np.isclose(strong["screen_area_fraction"], 1.0, atol=0.01)]
    native = full[np.isclose(full["relative_horizontal_magnification"], 1.0, atol=0.01)]
    four_three = full[
        (full["relative_horizontal_magnification"] >= 1.32)
        & (full["relative_horizontal_magnification"] <= 1.35)
    ]
    square = full[full["relative_horizontal_magnification"] >= 1.70]
    native_row = native.sort_values("minimum_coherence", ascending=False).iloc[0]
    rotation_row = (
        four_three.assign(abs_shift=np.abs(four_three["axis_shift_actual_minus_preserved_deg"]))
        .sort_values("abs_shift", ascending=False)
        .iloc[0]
    )
    used_trials = {(str(rotation_row["session"]), int(rotation_row["trial_idx"]))}

    def unused_trials(frame: pd.DataFrame) -> pd.DataFrame:
        keep = [
            (str(row.session), int(row.trial_idx)) not in used_trials
            for row in frame.itertuples()
        ]
        return frame.loc[keep]

    gain_row = unused_trials(four_three).sort_values(
        "alignment_change_actual_minus_preserved", ascending=False
    ).iloc[0]
    used_trials.add((str(gain_row["session"]), int(gain_row["trial_idx"])))
    loss_row = unused_trials(four_three).sort_values(
        "alignment_change_actual_minus_preserved", ascending=True
    ).iloc[0]
    square_row = (
        square.assign(abs_shift=np.abs(square["axis_shift_actual_minus_preserved_deg"]))
        .sort_values("abs_shift", ascending=False)
        .iloc[0]
    )
    specs = [
        (
            "native_16x9_control",
            native_row,
            "highest minimum actual/preserved coherence among native-16:9 full-screen windows",
        ),
        (
            "largest_axis_rotation_4x3",
            rotation_row,
            "largest absolute axis shift among coherent full-screen 4:3-source windows",
        ),
        (
            "largest_matching_gain_4x3",
            gain_row,
            "largest actual-minus-preserved drift/contour cos2 change on a different trial from the rotation example",
        ),
        (
            "largest_matching_loss_4x3",
            loss_row,
            "smallest actual-minus-preserved drift/contour cos2 change on a distinct 4:3-source trial",
        ),
        (
            "square_source_example",
            square_row,
            "largest absolute axis shift among coherent full-screen square-source windows",
        ),
    ]
    rows = []
    for role, row, criterion in specs:
        out = row.to_dict()
        out["selection_role"] = role
        out["selection_criterion"] = criterion
        rows.append(out)
    return pd.DataFrame(rows)


def plot_geometry(output_path: Path) -> None:
    native_deg = np.linspace(0.0, 90.0, 361)
    theta = np.radians(native_deg)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ratio, label, color in [
        (1.0, "native 16:9 (1.00×)", "#444444"),
        (4.0 / 3.0, "4:3 source (1.33× horizontal)", "#2b8cbe"),
        (16.0 / 9.0, "square source (1.78× horizontal)", "#d95f0e"),
    ]:
        displayed = np.degrees(np.arctan2(np.sin(theta), ratio * np.cos(theta)))
        axes[0].plot(native_deg, displayed, label=label, color=color, linewidth=2)
        axes[1].plot(native_deg, displayed - native_deg, label=label, color=color, linewidth=2)
    axes[0].plot([0, 90], [0, 90], linestyle="--", color="#999999", linewidth=1)
    axes[0].set(xlabel="native contour angle (deg)", ylabel="displayed contour angle (deg)")
    axes[1].axhline(0, linestyle="--", color="#999999", linewidth=1)
    axes[1].set(xlabel="native contour angle (deg)", ylabel="displayed − native angle (deg)")
    for ax in axes:
        ax.set_xlim(0, 90)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Predicted contour-axis rotation from BackImage horizontal stretching")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_axis(ax: plt.Axes, angle_deg: float, *, color: str, linestyle: str = "-", label: str | None = None) -> None:
    theta = np.radians(angle_deg)
    length = 0.78
    ax.plot(
        [-length * np.cos(theta), length * np.cos(theta)],
        [-length * np.sin(theta), length * np.sin(theta)],
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=label,
    )


def plot_examples(
    selected: pd.DataFrame,
    patches: dict[int, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(selected), 3, figsize=(11, 3.0 * len(selected)), constrained_layout=True)
    axes[0, 0].set_title("Aspect-restored counterfactual")
    axes[0, 1].set_title("Recorded displayed patch")
    axes[0, 2].set_title("Recorded − restored luminance")
    for row_idx, row in selected.reset_index(drop=True).iterrows():
        preserved, actual = patches[int(row["window_row"])]
        lo, hi = np.percentile(np.concatenate([preserved.ravel(), actual.ravel()]), [1, 99])
        diff = actual - preserved
        dlim = max(float(np.percentile(np.abs(diff), 99)), 1.0)
        axes[row_idx, 0].imshow(
            preserved, cmap="gray", vmin=lo, vmax=hi, origin="upper", extent=[-1, 1, -1, 1]
        )
        axes[row_idx, 1].imshow(
            actual, cmap="gray", vmin=lo, vmax=hi, origin="upper", extent=[-1, 1, -1, 1]
        )
        axes[row_idx, 2].imshow(
            diff, cmap="coolwarm", vmin=-dlim, vmax=dlim, origin="upper", extent=[-1, 1, -1, 1]
        )
        _draw_axis(axes[row_idx, 0], float(row["preserved_axis_deg"]), color="#00ffff", label="contour")
        _draw_axis(axes[row_idx, 1], float(row["actual_axis_saved_deg"]), color="#00ffff", label="contour")
        _draw_axis(axes[row_idx, 0], float(row["drift_axis_deg"]), color="#ffcc00", linestyle="--", label="drift")
        _draw_axis(axes[row_idx, 1], float(row["drift_axis_deg"]), color="#ffcc00", linestyle="--", label="drift")
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_idx, 0].set_ylabel(str(row["selection_role"]), fontsize=8)
        axes[row_idx, 2].text(
            1.03,
            0.5,
            f"{row['session']} trial {int(row['trial_idx'])}\n{row['image_file']}\n"
            f"axis: {row['preserved_axis_deg']:.1f}° → {row['actual_axis_saved_deg']:.1f}° "
            f"(Δ {row['axis_shift_actual_minus_preserved_deg']:+.1f}°)\n"
            f"drift match cos2: {row['drift_alignment_preserved_cos2']:+.2f} → "
            f"{row['drift_alignment_actual_cos2']:+.2f}",
            transform=axes[row_idx, 2].transAxes,
            va="center",
            fontsize=8,
        )
    axes[0, 0].legend(loc="lower left", fontsize=7, frameon=True)
    fig.suptitle(
        "BackImage stretch mechanism at matched native-image locations\n"
        "cyan = local contour axis; dashed yellow = measured drift axis"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(args.windows)
    trials = pd.read_csv(args.scale_audit)
    audit = compute_window_audit(windows, trials)
    selected = select_examples(audit)
    patches = selected_patches(selected)
    audit.to_csv(args.out_dir / "backimage_contour_stretch_window_audit.csv", index=False)
    selected.to_csv(args.out_dir / "selected_examples.csv", index=False)
    plot_geometry(args.out_dir / "backimage_contour_stretch_geometry.png")
    plot_examples(selected, patches, args.out_dir / "backimage_contour_stretch_local_examples.png")
    np.savez_compressed(
        args.out_dir / "selected_example_patches.npz",
        **{
            f"window_{int(row.window_row)}_{kind}": patches[int(row.window_row)][index].astype(np.float32)
            for row in selected.itertuples()
            for index, kind in enumerate(["preserved", "actual"])
        },
    )
    metadata = {
        "analysis": "targeted_map_first_backimage_contour_stretch_mechanism_checkpoint",
        "windows": str(args.windows),
        "scale_audit": str(args.scale_audit),
        "n_windows": int(len(audit)),
        "patch_radius_deg": PATCH_RADIUS_DEG,
        "counterfactual": (
            "same native-image center and recorded vertical angular scale; horizontal source sampling "
            "set equal to vertical sampling"
        ),
        "actual_axis_contract": "saved image_edge_axis_deg from recorded resized static canvas at mean gaze",
        "status": "input_mechanism_checkpoint_not_population_inference",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
