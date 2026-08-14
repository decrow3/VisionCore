#!/usr/bin/env python3
"""Human-review summary for Cartographer Stage 3 spatial-contract checkpoint."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_v1"
CONTEXT = ROOT / "outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_context_audit_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_review_v1"


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def plot_review(
    equivalence: pd.DataFrame,
    modulation: pd.DataFrame,
    translation: pd.DataFrame,
    apertures: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    units = sorted(equivalence.response_rr100_index.unique())
    colors = {unit: color for unit, color in zip(units, plt.cm.tab10(np.linspace(0, 1, len(units))), strict=True)}
    point_colors = [colors[int(unit)] for unit in equivalence.response_rr100_index]

    for axis, frame, x_column, y_column, title, x_label, y_label in (
        (
            axes[0, 0], equivalence, "native_rate_hz", "large_embedded_center_rate_hz",
            "Absolute firing rate is not native-to-large equivalent",
            "native 51×51 response (Hz)", "large-canvas central response (Hz)",
        ),
        (
            axes[0, 1], modulation, "native_modulation_hz", "large_embedded_modulation_hz",
            "Separate blank subtraction does not remove the mismatch",
            "native probe minus native blank (Hz)", "large probe minus large blank (Hz)",
        ),
    ):
        axis.scatter(frame[x_column], frame[y_column], c=point_colors, s=30, alpha=0.8)
        low = float(min(frame[x_column].min(), frame[y_column].min()))
        high = float(max(frame[x_column].max(), frame[y_column].max()))
        axis.plot([low, high], [low, high], color="0.35", ls="--")
        axis.set(xlabel=x_label, ylabel=y_label, title=title)

    axes[0, 2].scatter(
        equivalence.large_embedded_center_rate_hz,
        equivalence.large_extended_center_rate_hz,
        c=point_colors, s=30, alpha=0.8,
    )
    low = float(min(equivalence.large_embedded_center_rate_hz.min(), equivalence.large_extended_center_rate_hz.min()))
    high = float(max(equivalence.large_embedded_center_rate_hz.max(), equivalence.large_extended_center_rate_hz.max()))
    axes[0, 2].plot([low, high], [low, high], color="0.35", ls="--")
    axes[0, 2].set(
        xlabel="exactly embedded probe central response (Hz)",
        ylabel="analytically extended probe central response (Hz)",
        title="Outer probe signal is not the source of the mismatch",
    )

    nonzero = translation.loc[
        translation.translation_x_input_px.ne(0) | translation.translation_y_input_px.ne(0)
    ].copy()
    nonzero["translation_distance_group"] = np.maximum(
        np.abs(nonzero.translation_x_input_px), np.abs(nonzero.translation_y_input_px)
    )
    distance_groups = sorted(nonzero.translation_distance_group.unique())
    for position, distance in enumerate(distance_groups):
        values = nonzero.loc[
            nonzero.translation_distance_group.eq(distance),
            "translation_map_normalized_root_mean_square_error",
        ]
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.asarray([0.0])
        axes[1, 0].scatter(np.full(len(values), position) + jitter, values, s=12, alpha=0.45)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(range(len(distance_groups)), [f"{value} input pixels" for value in distance_groups])
    axes[1, 0].set(
        xlabel="largest translation component", ylabel="normalized map root-mean-square error",
        title="Two input pixels per activation-map bin passes, including edge probes",
    )

    candidate_order = [
        "learned readout back-projection", "grating input-gradient energy", "translated-probe response envelope"
    ]
    labels = ["learned readout\nback-projection", "grating input-gradient\nenergy", "translated-probe\nresponse envelope"]
    for unit in units:
        frame = apertures.loc[apertures.rr100_index.eq(unit)].set_index("candidate_aperture").loc[candidate_order]
        axes[1, 1].plot(range(3), frame.energy_radius_90_input_px, marker="o", label=f"RR100 unit {unit}")
    axes[1, 1].set_xticks(range(3), labels)
    axes[1, 1].set(
        ylabel="90% energy radius (input pixels)",
        title="Candidate spatial supports disagree in scale and shape",
    )
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.98,
        "Stage 3 decision\n\n"
        "• The spatial coordinate rule is supported: translating the input by two pixels translates the activation map by one bin.\n\n"
        "• Native 51×51 grating responses cannot be transferred directly to the central large-canvas pathway; the mismatch changes stimulus modulation, not merely baseline.\n\n"
        "• The learned-readout back-projection is broader than the grating gradient, while the translated-probe envelope is probe- and response-dependent and is not a direct receptive-field aperture.\n\n"
        "• Therefore no aperture or Stage 4 power-to-rate calibration is frozen at this checkpoint.",
        va="top", fontsize=11, wrap=True,
    )
    figure.suptitle(
        "Cartographer Stage 3 review: translation coordinates pass, but native response transfer and aperture selection do not",
        fontsize=15, weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    equivalence = pd.read_csv(SOURCE / "native_to_large_canvas_response_equivalence.csv")
    translation = pd.read_csv(SOURCE / "translation_equivariance_metrics.csv")
    apertures = pd.read_csv(SOURCE / "candidate_aperture_geometry.csv")
    modulation = pd.read_csv(CONTEXT / "blank_subtracted_response_equivalence.csv")
    nonzero = translation.loc[
        translation.translation_x_input_px.ne(0) | translation.translation_y_input_px.ne(0)
    ]
    interior = nonzero.loc[
        np.maximum(np.abs(nonzero.translation_x_input_px), np.abs(nonzero.translation_y_input_px)).lt(40)
    ]
    edge = nonzero.loc[
        np.maximum(np.abs(nonzero.translation_x_input_px), np.abs(nonzero.translation_y_input_px)).ge(40)
    ]
    embedded_extended_error = np.abs(
        equivalence.large_embedded_center_rate_hz - equivalence.large_extended_center_rate_hz
    )
    summary = pd.DataFrame(
        [
            ("native_to_large_absolute_rate_pearson_r", float(np.corrcoef(equivalence.native_rate_hz, equivalence.large_embedded_center_rate_hz)[0, 1])),
            ("native_to_large_maximum_absolute_rate_error_hz", float(equivalence.embedded_absolute_error_hz.max())),
            ("native_to_large_mean_absolute_rate_error_hz", float(equivalence.embedded_absolute_error_hz.mean())),
            ("blank_subtracted_modulation_pearson_r", float(np.corrcoef(modulation.native_modulation_hz, modulation.large_embedded_modulation_hz)[0, 1])),
            ("blank_subtracted_maximum_modulation_error_hz", float(modulation.embedded_modulation_absolute_error_hz.max())),
            ("embedded_vs_extended_maximum_central_error_hz", float(embedded_extended_error.max())),
            ("interior_translation_minimum_map_pearson_r", float(interior.translation_map_pearson_r.min())),
            ("interior_translation_maximum_normalized_rmse", float(interior.translation_map_normalized_root_mean_square_error.max())),
            ("edge_translation_minimum_map_pearson_r", float(edge.translation_map_pearson_r.min())),
            ("edge_translation_maximum_normalized_rmse", float(edge.translation_map_normalized_root_mean_square_error.max())),
            ("readout_backprojection_median_radius90_input_px", float(apertures.loc[apertures.candidate_aperture.eq("learned readout back-projection"), "energy_radius_90_input_px"].median())),
            ("grating_gradient_median_radius90_input_px", float(apertures.loc[apertures.candidate_aperture.eq("grating input-gradient energy"), "energy_radius_90_input_px"].median())),
            ("translated_probe_envelope_median_radius90_input_px", float(apertures.loc[apertures.candidate_aperture.eq("translated-probe response envelope"), "energy_radius_90_input_px"].median())),
        ],
        columns=["metric", "value"],
    )
    summary_path = OUT / "stage3_decision_metrics.csv"
    summary.to_csv(summary_path, index=False)
    figure_path = OUT / "01_stage3_decision_review"
    plot_review(equivalence, modulation, translation, apertures, figure_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "cartographer_stage3_human_review_summary",
        "status": "stage3_human_checkpoint_complete_gate_not_passed",
        "supported": "two input pixels correspond to one activation-map bin, with near-exact translation covariance",
        "unsupported": (
            "native 51x51 grating response scale transfer to large-canvas maps and selection of a single effective aperture"
        ),
        "decision_gate": (
            "do not begin Stage 4; next test direct large-canvas grating calibration and distinguish architectural "
            "support from operating-point sensitivity before freezing a spatial power aperture"
        ),
        "reserved_final_test_identities_opened": False,
        "sources": {
            "stage3": identity(SOURCE / "manifest.json"),
            "context_audit": identity(CONTEXT / "manifest.json"),
        },
        "outputs": {
            "decision_metrics": identity(summary_path),
            "decision_figure": identity(figure_path.with_suffix(".png")),
        },
        "runner": identity(Path(__file__)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Cartographer Stage 3 human-review checkpoint\n\n"
        "The two-input-pixel to one-map-bin coordinate rule passes. Native 51×51 response scale does not "
        "transfer to the central 151×151 pathway, even after separate blank subtraction, and the candidate "
        "apertures disagree substantially. The Stage 3 gate is not passed; do not begin Stage 4.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
