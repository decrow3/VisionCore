#!/usr/bin/env python3
"""Local complex-pyramid phase-scramble checkpoint on Checkpoint 42 inputs.

This reuses the exact three corrected source patches and shared corrected FEM
trace from the global source-scramble checkpoint.  Within the source region
swept by the complete history + scored trace, it independently randomizes the
phase of every complex coefficient in a four-level, four-orientation steerable
pyramid while retaining coefficient magnitudes and real residuals.  The
reconstruction is clipped to the intact source range before the identical FEM
trace is rendered.  No neural model is evaluated.

This is the existing, direct twininfo control.  It is intentionally not called
a pooled-energy metamer: its coefficient magnitudes are constrained pointwise,
not averaged in model-matched pooling windows.  Saved RR100 readout-scale
quantiles are carried into the checkpoint as the calibration target for a later
optimization-based metamer only if this simpler direct control is inadequate.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from declan.fig4_active_sensing.input_only_retinal_renderer import (
    render_retinal_frames_lag_zero,
)
from declan.fig4_active_sensing.make_rr100_global_source_phase_scramble_checkpoint import (
    _correlation,
    _difference,
    _font,
    _gray,
    _montage,
    _phase_difference_coherence,
    _write_mp4,
    file_identity,
)
from declan.fig4_active_sensing.make_rr100_phase_surrogate_input_checkpoint import (
    EPS,
    N_SCORE,
    PPD,
    movie_audit,
    power_audit,
    relative_db,
)
from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    FRAME_RATE_HZ,
    SF_EDGES_CPD,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
)
from jake.twininfo.retinal_examples import pyramid_local_image_controls
from jake.twininfo.retinal_examples import (
    _padded_even_patch,
    _patch_to_tensor,
    _phase_scramble_pyramid_coeffs,
    _steerable_pyramid,
    local_phase_scramble_roi,
)


ROOT = Path(__file__).resolve().parents[2]
GLOBAL = ROOT / "outputs/fig4_active_sensing/rr100_global_source_phase_scramble_checkpoint_42_v1"
SCALES = ROOT / "outputs/fig4_active_sensing/rr100_spatial_filter_pooling_scale_checkpoint_41_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_local_pyramid_phase_scramble_checkpoint_43_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-checkpoint", type=Path, default=GLOBAL)
    parser.add_argument("--scale-checkpoint", type=Path, default=SCALES)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def _prefixed(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _roi_values(values: np.ndarray, audit: dict[str, Any]) -> np.ndarray:
    return np.asarray(values)[
        int(audit["roi_y0"]):int(audit["roi_y1"]),
        int(audit["roi_x0"]):int(audit["roi_x1"]),
    ]


def _weighted_phase_coherence(delta: np.ndarray, weight: np.ndarray) -> float:
    phase = np.asarray(delta, dtype=np.float64).ravel()
    weights = np.asarray(weight, dtype=np.float64).ravel()
    valid = np.isfinite(phase) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    vector = np.sum(weights[valid] * np.exp(1j * phase[valid])) / np.sum(weights[valid])
    return float(np.abs(vector))


def _coefficient_phase_audit(
    source: np.ndarray,
    full_retinal_trace: np.ndarray,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Audit the randomized phases before redundant-pyramid reconstruction.

    The cross-level statistic compares the original and randomized adjacent-
    scale phase-relation maps, phi_fine - 2*phi_coarse, within orientation.
    Values near zero mean the original relationship was destroyed.
    """
    roi = local_phase_scramble_roi(
        full_retinal_trace,
        source.shape,
        ppd=PPD,
        out_size=(51, 51),
    )
    patch = np.asarray(source, dtype=np.float32)[
        int(roi["roi_y0"]):int(roi["roi_y1"]),
        int(roi["roi_x0"]):int(roi["roi_x1"]),
    ]
    padded, _ = _padded_even_patch(patch)
    pyramid = _steerable_pyramid(padded.shape, height=4, order=3)
    original = pyramid(_patch_to_tensor(padded))
    randomized, _, _ = _phase_scramble_pyramid_coeffs(original, np.random.default_rng(seed))
    rows: list[dict[str, Any]] = []
    for level in range(4):
        original_level = original[level].detach().cpu().numpy()[0, 0]
        randomized_level = randomized[level].detach().cpu().numpy()[0, 0]
        for orientation in range(original_level.shape[0]):
            before = original_level[orientation]
            after = randomized_level[orientation]
            rows.append(
                {
                    "audit_type": "within_band_original_to_randomized_phase",
                    "fine_level": level,
                    "coarse_level": -1,
                    "orientation": orientation,
                    "phase_relation_retention_coherence": _weighted_phase_coherence(
                        np.angle(after * np.conj(before)),
                        np.abs(before),
                    ),
                }
            )
    for fine_level in range(3):
        coarse_level = fine_level + 1
        original_fine = original[fine_level].detach().cpu().numpy()[0, 0]
        original_coarse = original[coarse_level].detach().cpu().numpy()[0, 0]
        randomized_fine = randomized[fine_level].detach().cpu().numpy()[0, 0]
        randomized_coarse = randomized[coarse_level].detach().cpu().numpy()[0, 0]
        for orientation in range(original_fine.shape[0]):
            original_relation = np.angle(original_fine[orientation]) - 2 * np.angle(original_coarse[orientation])
            randomized_relation = np.angle(randomized_fine[orientation]) - 2 * np.angle(randomized_coarse[orientation])
            weight = np.sqrt(np.abs(original_fine[orientation]) * np.abs(original_coarse[orientation]))
            rows.append(
                {
                    "audit_type": "adjacent_scale_phase_relationship_retention",
                    "fine_level": fine_level,
                    "coarse_level": coarse_level,
                    "orientation": orientation,
                    "phase_relation_retention_coherence": _weighted_phase_coherence(
                        randomized_relation - original_relation,
                        weight,
                    ),
                }
            )
    return rows


def _video_frames(
    intact: np.ndarray,
    global_scramble: np.ndarray,
    local_scramble: np.ndarray,
    *,
    title: str,
    repeat: int = 2,
) -> Iterable[Image.Image]:
    local_delta = np.asarray(local_scramble, dtype=np.float64) - np.asarray(intact, dtype=np.float64)
    limit = float(np.quantile(np.abs(local_delta), 0.99))
    title_font = _font(21)
    label_font = _font(16)
    body_font = _font(14)
    scale = 5
    panel = 51 * scale
    gap = 14
    margin = 14
    width = 2 * margin + 4 * panel + 3 * gap
    height = 103 + panel
    for frame_index in range(intact.shape[0]):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 5), title, fill="black", font=title_font)
        positions = [margin + index * (panel + gap) for index in range(4)]
        panels = (
            _gray(intact[frame_index], scale),
            _gray(global_scramble[frame_index], scale),
            _gray(local_scramble[frame_index], scale),
            _difference(local_delta[frame_index], limit, scale),
        )
        labels = (
            "Intact + FEM",
            "Global + same FEM",
            "Local pyramid + same FEM",
            f"Local - intact (±{limit:.1f})",
        )
        for x, image, label in zip(positions, panels, labels, strict=True):
            canvas.paste(image, (x, 103))
            draw.text((x, 50), label, fill="black", font=label_font)
        draw.text(
            (margin, 78),
            f"frame {frame_index:02d}/39 | local frame r={_correlation(intact[frame_index], local_scramble[frame_index]):+.3f}"
            f" | local movie r={_correlation(intact, local_scramble):+.3f}",
            fill="black",
            font=body_font,
        )
        for _ in range(int(repeat)):
            yield canvas


def _plot_source_movie(
    selections: pd.DataFrame,
    payloads: list[dict[str, Any]],
    combined: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(len(payloads), 6, figsize=(20.0, 3.4 * len(payloads)), constrained_layout=True)
    for row_index, (selection, payload) in enumerate(zip(selections.itertuples(index=False), payloads, strict=True)):
        audit = payload["local_native_audit"]
        source_panels = (
            (payload["source_intact"], "Corrected source"),
            (payload["source_global"], "Global Fourier phase"),
            (payload["source_local"], "Local complex-pyramid phase"),
        )
        for column, (values, title) in enumerate(source_panels):
            ax = axes[row_index, column]
            ax.imshow(values, cmap="gray", vmin=0, vmax=255, origin="lower")
            if column in (0, 2):
                ax.add_patch(
                    Rectangle(
                        (audit["roi_x0"], audit["roi_y0"]),
                        audit["roi_width_px"],
                        audit["roi_height_px"],
                        fill=False,
                        edgecolor="#e66101",
                        linewidth=1.2,
                    )
                )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        movie_panels = (
            (payload["movie_intact"], "Intact retinal movie"),
            (payload["movie_global"], "Global retinal movie"),
            (payload["movie_local"], "Local retinal movie"),
        )
        for column, (movie, title) in enumerate(movie_panels, start=3):
            ax = axes[row_index, column]
            ax.imshow(_montage(movie), cmap="gray", vmin=0, vmax=255, origin="lower")
            if column == 5:
                record = combined[(combined.example_number == row_index + 1) & (combined.control == "local_pyramid")].iloc[0]
                title += (
                    f"\ncontrast={record.spatial_contrast_ratio_to_intact:.2f}x; "
                    f"power={record.hann_supported_power_ratio:.2f}x"
                )
            ax.set_title(title + "\nt=5, 19, 34", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row_index, 0].set_ylabel(
            f"{selection.selection_role}\nimage {int(selection.image_index)}",
            fontsize=9,
        )
    fig.suptitle(
        "Checkpoint 43: direct local complex-pyramid phase scramble\n"
        "Four levels × four orientations; trace-swept ROI; exact same corrected FEM; no neural scoring",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path, dpi=175)
    plt.close(fig)


def _plot_power(
    payloads: list[dict[str, Any]],
    combined: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(len(payloads), 2, figsize=(10.2, 3.5 * len(payloads)), constrained_layout=True)
    sf_centers = 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])
    tf_hz = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]
    mesh = None
    for row_index, payload in enumerate(payloads):
        for column, (control, radial_key, title) in enumerate(
            (
                ("global_source", "radial_global", "Global source scramble"),
                ("local_pyramid", "radial_local", "Local pyramid scramble"),
            )
        ):
            record = combined[(combined.example_number == row_index + 1) & (combined.control == control)].iloc[0]
            delta_db = relative_db(payload[radial_key]) - relative_db(payload["radial_intact"])
            ax = axes[row_index, column]
            mesh = ax.pcolormesh(sf_centers, tf_hz, delta_db, cmap="coolwarm", vmin=-12, vmax=12, shading="nearest")
            ax.set_xscale("log")
            ax.set_xlabel("SF (cpd)")
            ax.set_ylabel("TF (Hz)")
            ax.set_title(
                f"{title}\ncos={record.hann_supported_power_cosine:.3f}; ratio={record.hann_supported_power_ratio:.2f}; "
                f"contrast={record.spatial_contrast_ratio_to_intact:.2f}x",
                fontsize=9,
            )
    assert mesh is not None
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="control minus intact relative power (dB)")
    fig.suptitle("Retinal SF×TF audit: global versus direct local phase controls", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=185)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {out_dir}")
    (out_dir / "movies").mkdir(parents=True)
    (out_dir / "data").mkdir()

    selections_path = args.global_checkpoint / "selected_images.csv"
    trace_path = args.global_checkpoint / "selected_trace.csv"
    global_audit_path = args.global_checkpoint / "global_source_phase_scramble_audit.csv"
    scale_path = args.scale_checkpoint / "spatial_readout_scale_quantiles.csv"
    selections = pd.read_csv(selections_path)
    trace_selection = pd.read_csv(trace_path).iloc[0]
    global_audit = pd.read_csv(global_audit_path)
    scale_quantiles = pd.read_csv(scale_path)
    rr100_scales = scale_quantiles[scale_quantiles.population.eq("rr100_movie_medoids")].copy()
    rr100_scales.to_csv(out_dir / "rr100_readout_scale_calibration.csv", index=False)
    selections.to_csv(out_dir / "selected_images.csv", index=False)
    pd.DataFrame([trace_selection.to_dict()]).to_csv(out_dir / "selected_trace.csv", index=False)

    common = _load_twin_common()
    combined_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    coefficient_phase_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for selection in selections.itertuples(index=False):
        example_number = int(selection.Index + 1) if hasattr(selection, "Index") else len(payloads) + 1
        image_index = int(selection.image_index)
        source_path = args.global_checkpoint / "data" / f"example_{example_number}_image_{image_index:03d}.npz"
        with np.load(source_path, allow_pickle=False) as data:
            source_intact = np.asarray(data["source_intact"], dtype=np.float32)
            source_global = np.asarray(data["source_global_phase_scrambled"], dtype=np.float32)
            movie_intact = np.asarray(data["movie_intact"], dtype=np.float32)
            movie_global = np.asarray(data["movie_global_phase_scrambled"], dtype=np.float32)
            history = np.asarray(data["history_xy_deg"], dtype=np.float32)
            score = np.asarray(data["score_xy_deg"], dtype=np.float32)
            retinal_score = np.asarray(data["retinal_score_trace_xy_deg"], dtype=np.float32)
            radial_intact = np.asarray(data["radial_power_intact"], dtype=np.float64)
            radial_global = np.asarray(data["radial_power_global_phase_scrambled"], dtype=np.float64)
            trace_index = int(data["trace_index"].item())
        full_retinal_trace = -np.concatenate([history, score], axis=0)
        local_seed = int(args.seed) + image_index * 1009
        phase_rows = _coefficient_phase_audit(
            source_intact,
            full_retinal_trace,
            seed=local_seed,
        )
        coefficient_phase_rows.extend(
            {
                "example_number": example_number,
                "image_index": image_index,
                "trace_index": trace_index,
                **row,
            }
            for row in phase_rows
        )
        controls, local_audits = pyramid_local_image_controls(
            source_intact,
            full_retinal_trace,
            np.random.default_rng(local_seed),
            clip=True,
            ppd=PPD,
            out_size=(51, 51),
            height=4,
            order=3,
            sf_bands=(),
        )
        source_local = controls["pyramid_phase_scrambled"]
        local_native = dict(local_audits[0])
        movie_local = (
            render_retinal_frames_lag_zero(common, source_local, retinal_score, ppd=PPD)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        local_power, radial_local = power_audit(movie_intact, movie_local, ppd=PPD)
        local_movie = movie_audit(movie_intact, movie_local)
        intact_movie = movie_audit(movie_intact, movie_intact)
        global_record = global_audit[global_audit.example_number.eq(example_number)].iloc[0]
        global_power_fields = {
            key: global_record[key]
            for key in local_power
        }
        global_movie_fields = {
            key: global_record[key]
            for key in local_movie
        }
        for control, movie_fields, power_fields in (
            ("global_source", global_movie_fields, global_power_fields),
            ("local_pyramid", local_movie, local_power),
        ):
            combined_rows.append(
                {
                    "example_number": example_number,
                    "selection_role": selection.selection_role,
                    "image_index": image_index,
                    "trace_index": trace_index,
                    "control": control,
                    **movie_fields,
                    **power_fields,
                    "spatial_contrast_ratio_to_intact": float(movie_fields["spatial_contrast_rms"])
                    / max(float(intact_movie["spatial_contrast_rms"]), EPS),
                }
            )
        intact_roi = _roi_values(source_intact, local_native)
        local_roi = _roi_values(source_local, local_native)
        native_rows.append(
            {
                "example_number": example_number,
                "selection_role": selection.selection_role,
                "image_index": image_index,
                "trace_index": trace_index,
                "local_seed": local_seed,
                **local_native,
                "roi_pixel_correlation": _correlation(intact_roi, local_roi),
                "roi_phase_difference_vector_coherence": _phase_difference_coherence(intact_roi, local_roi),
                "roi_std_ratio_to_intact": float(np.std(local_roi)) / max(float(np.std(intact_roi)), EPS),
                "full_source_pixel_correlation": _correlation(source_intact, source_local),
            }
        )
        payload = {
            "source_intact": source_intact,
            "source_global": source_global,
            "source_local": source_local,
            "movie_intact": movie_intact,
            "movie_global": movie_global,
            "movie_local": movie_local,
            "radial_intact": radial_intact,
            "radial_global": radial_global,
            "radial_local": radial_local,
            "local_native_audit": local_native,
        }
        payloads.append(payload)
        data_path = out_dir / "data" / f"example_{example_number}_image_{image_index:03d}.npz"
        np.savez_compressed(
            data_path,
            source_intact=source_intact,
            source_global_phase_scrambled=source_global,
            source_local_pyramid_phase_scrambled=source_local,
            movie_intact=movie_intact,
            movie_global_phase_scrambled=movie_global,
            movie_local_pyramid_phase_scrambled=movie_local,
            history_xy_deg=history,
            score_xy_deg=score,
            retinal_score_trace_xy_deg=retinal_score,
            full_retinal_trace_xy_deg=full_retinal_trace,
            radial_power_intact=radial_intact,
            radial_power_global_phase_scrambled=radial_global,
            radial_power_local_pyramid_phase_scrambled=radial_local,
            image_index=np.asarray(image_index),
            trace_index=np.asarray(trace_index),
            local_seed=np.asarray(local_seed),
        )
        movie_path = out_dir / "movies" / f"example_{example_number}_image_{image_index:03d}_global_vs_local.mp4"
        _write_mp4(
            _video_frames(
                movie_intact,
                movie_global,
                movie_local,
                title=f"Example {example_number}: image {image_index}, shared trace {trace_index}",
            ),
            movie_path,
            fps=int(args.fps),
        )

    combined = pd.DataFrame(combined_rows)
    native = pd.DataFrame(native_rows)
    coefficient_phase = pd.DataFrame(coefficient_phase_rows)
    combined_path = out_dir / "global_vs_local_retinal_power_contrast_audit.csv"
    native_path = out_dir / "local_pyramid_source_audit.csv"
    coefficient_phase_path = out_dir / "local_pyramid_coefficient_phase_relationship_audit.csv"
    combined.to_csv(combined_path, index=False)
    native.to_csv(native_path, index=False)
    coefficient_phase.to_csv(coefficient_phase_path, index=False)
    source_figure = out_dir / "local_pyramid_source_and_movie_checkpoint.png"
    power_figure = out_dir / "global_vs_local_retinal_power_audit.png"
    _plot_source_movie(selections, payloads, combined, source_figure)
    _plot_power(payloads, combined, power_figure)

    local = combined[combined.control.eq("local_pyramid")]
    manifest = {
        "analysis": "rr100_direct_local_complex_pyramid_phase_scramble_input_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "input_only_human_checkpoint_no_neural_scoring",
        "control_contract": (
            "independent random phase at every complex coefficient in a four-level, four-orientation "
            "steerable pyramid within the complete-trace-swept source ROI; coefficient magnitudes and "
            "real residuals retained; reconstruction clipped to intact source range; identical FEM rendering"
        ),
        "not_a_claim": (
            "This direct control is not an optimization-based pooled-energy metamer and does not use the "
            "RR100 readout radii as synthesis windows. The scale distribution is saved only as the required "
            "calibration target if an expanded metamer synthesis is needed."
        ),
        "paper_comparison": (
            "Broderick et al. synthesize images by optimization to match Gaussian-window pooled luminance "
            "and multi-scale/orientation complex-pyramid energy. Their foveated log-polar windows and human "
            "perceptual objective are not directly transplanted to this fixed-retina, unit-distributed setting."
        ),
        "n_examples": int(len(payloads)),
        "pyramid_height": 4,
        "pyramid_order": 3,
        "seed": int(args.seed),
        "median_local_hann_supported_power_ratio": float(local.hann_supported_power_ratio.median()),
        "median_local_hann_supported_power_cosine": float(local.hann_supported_power_cosine.median()),
        "median_local_spatial_contrast_ratio": float(local.spatial_contrast_ratio_to_intact.median()),
        "maximum_local_out_of_range_fraction": float(local.out_of_0_255_fraction.max()),
        "maximum_complex_coefficient_magnitude_relative_error": float(native.complex_coeff_magnitude_relative_error.max()),
        "maximum_pyramid_reconstruction_relative_error": float(native.pyramid_reconstruction_relative_error.max()),
        "maximum_outside_roi_changed_fraction": float(native.outside_roi_changed_fraction.max()),
        "maximum_within_band_phase_retention_coherence": float(
            coefficient_phase[
                coefficient_phase.audit_type.eq("within_band_original_to_randomized_phase")
            ].phase_relation_retention_coherence.max()
        ),
        "maximum_adjacent_scale_phase_relationship_retention_coherence": float(
            coefficient_phase[
                coefficient_phase.audit_type.eq("adjacent_scale_phase_relationship_retention")
            ].phase_relation_retention_coherence.max()
        ),
        "inputs": {
            "global_manifest": file_identity(args.global_checkpoint / "manifest.json"),
            "selected_images": file_identity(selections_path),
            "selected_trace": file_identity(trace_path),
            "global_audit": file_identity(global_audit_path),
            "rr100_readout_scale_quantiles": file_identity(scale_path),
        },
        "outputs": {
            "retinal_audit": file_identity(combined_path),
            "local_source_audit": file_identity(native_path),
            "coefficient_phase_relationship_audit": file_identity(coefficient_phase_path),
            "source_movie_figure": file_identity(source_figure),
            "power_figure": file_identity(power_figure),
            "movies": [file_identity(path) for path in sorted((out_dir / "movies").glob("*.mp4"))],
        },
        "next_checkpoint_if_approved": (
            "either score intact/global/local in the frozen twins, or build optimization-based pooled-energy "
            "surrogates with pooling windows sampled from the RR100 spatial-filter distribution"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Checkpoint 43: direct local complex-pyramid phase scramble\n\n"
        "The three examples and trace are exactly those in Checkpoint 42. Inside the complete-trace-swept "
        "source ROI, phases of all complex steerable-pyramid coefficients are randomized independently; "
        "magnitudes and real residuals are retained. The clipped reconstruction is moved with the identical "
        "corrected FEM trace.\n\n"
        "Start with `local_pyramid_source_and_movie_checkpoint.png` and the three MP4s, then inspect "
        "`global_vs_local_retinal_power_audit.png`. This is the existing direct twininfo control, not the "
        "optimization-based pooled-energy construction in Broderick et al. Their paper motivates matching "
        "feature energy in explicit pooling windows, but its eccentricity-scaled human-perception windows "
        "are not directly applicable here. The RR100 readout-scale distribution is copied into this checkpoint "
        "as the calibration for that possible expansion.\n\n"
        "No digital-twin response or SSI was computed.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
