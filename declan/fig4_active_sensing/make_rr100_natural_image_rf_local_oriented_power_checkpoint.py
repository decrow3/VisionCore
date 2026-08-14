#!/usr/bin/env python3
"""RF-local oriented-power input checkpoint for corrected natural-image movies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from declan.fig4_active_sensing.input_only_retinal_renderer import render_retinal_frames_lag_zero
from declan.fig4_active_sensing.make_rr100_orientation_routing_input_checkpoint import four_grating_channels
from declan.fig4_active_sensing.make_rr100_recorded_grating_oriented_power_checkpoint import (
    GRATING_ORIENTATIONS,
    load_rf_and_tuning,
    localized_oriented_spectrum,
    relative_db,
)
from declan.fig4_active_sensing.make_rr100_recorded_grating_three_way_response_checkpoint import indices_for_support
from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    ORIENTATION_EDGES_DEG,
    spectral_statistics,
)
from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import (
    _load_twin_common,
    _standardize_uint_like,
)


ROOT = Path(__file__).resolve().parents[2]
SPECTRAL = ROOT / "outputs/fig4_active_sensing/rr100_corrected_three_round_spectral_cache_v1"
COHORT = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_production_cohort_v1"
RESPONSES = ROOT / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1"
RF = ROOT / "outputs/fig4_active_sensing/rr100_recorded_grating_three_way_response_rf_local_v2"
TUNING = ROOT / "outputs/fig4_active_sensing/rr100_orientation_aware_f0_map_checkpoint_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_natural_image_rf_local_oriented_power_input_checkpoint_v2"
PPD = 37.50476617
EPS = np.finfo(float).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-dir", type=Path, default=SPECTRAL)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT)
    parser.add_argument("--response-cache-dir", type=Path, default=RESPONSES)
    parser.add_argument("--rf-dir", type=Path, default=RF)
    parser.add_argument("--tuning-dir", type=Path, default=TUNING)
    parser.add_argument("--session", default="Logan_2020-02-29")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {"path": str(path.resolve()), "size_bytes": int(stat.st_size), "sha256": digest.hexdigest()}


def spectral_storage_crosswalk(spectral_dir: Path) -> pd.DataFrame:
    """Reconstruct the image-major append order used by the existing cache builder."""
    condition_path = (
        ROOT
        / "outputs/fig4_active_sensing/rr100_corrected100x1000_response_cache_v1/assembled/rounds_000_002_n003/condition_index.csv"
    )
    conditions = pd.read_csv(condition_path).sort_values("matrix_row_index").reset_index(drop=True)
    storage = pd.concat(
        [frame.sort_values("matrix_row_index") for _, frame in conditions.groupby("image_index", sort=True)],
        ignore_index=True,
    )
    storage.insert(0, "spectral_storage_row", np.arange(len(storage), dtype=int))
    with np.load(spectral_dir / "condition_spectra.npz", allow_pickle=False) as archive:
        declared = pd.DataFrame(
            {
                "declared_matrix_row_index": archive["matrix_row_index"].astype(int),
                "declared_image_index": archive["image_index"].astype(int),
                "declared_trace_index": archive["trace_index"].astype(int),
                "declared_round_index": archive["round_index"].astype(int),
            }
        )
    return pd.concat([storage.reset_index(drop=True), declared], axis=1)


def select_conditions(spectral_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    with np.load(spectral_dir / "condition_spectra.npz", allow_pickle=False) as archive:
        spectral = {key: np.asarray(archive[key]) for key in archive.files}
    crosswalk = spectral_storage_crosswalk(spectral_dir)
    channels = four_grating_channels(spectral["orientation_power"], spectral["orientation_edges_deg"])
    channel_power = channels.sum(axis=(1, 2))
    fractions = channel_power / np.maximum(channel_power.sum(axis=1, keepdims=True), EPS)
    concentration = fractions.max(axis=1)
    total = spectral["radial_power"].sum(axis=(1, 2)).astype(float)
    middle = (total >= np.quantile(total, 0.25)) & (total <= np.quantile(total, 0.75))
    candidates = np.flatnonzero(middle)
    roles = [
        (
            "orientation-concentrated input",
            int(candidates[np.argmax(concentration[candidates])]),
            "largest four-channel concentration among middle-50% total-power conditions",
        ),
        (
            "orientation-balanced input",
            int(candidates[np.argmin(concentration[candidates])]),
            "smallest four-channel concentration among middle-50% total-power conditions",
        ),
    ]
    z_power = (np.log10(np.maximum(total, EPS)) - np.median(np.log10(np.maximum(total, EPS)))) / max(
        float(np.std(np.log10(np.maximum(total, EPS)))), 1e-12
    )
    z_concentration = (concentration - np.median(concentration)) / max(float(np.std(concentration)), 1e-12)
    typical_score = np.abs(z_power) + np.abs(z_concentration)
    used = {row for _, row, _ in roles}
    typical_order = np.argsort(typical_score)
    typical = int(next(row for row in typical_order if int(row) not in used))
    roles.append(("typical input control", typical, "closest jointly to median log-power and orientation concentration"))
    selected = []
    for role, row_index, criterion in roles:
        row = crosswalk.iloc[row_index].copy()
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        row["selection_value"] = float(concentration[row_index] if "orientation" in role else typical_score[row_index])
        row["four_channel_concentration"] = float(concentration[row_index])
        for index, orientation in enumerate(GRATING_ORIENTATIONS):
            row[f"global_power_fraction_{int(orientation)}deg"] = float(fractions[row_index, index])
        selected.append(row)
    return pd.DataFrame(selected), spectral, crosswalk


def load_selected_movies(
    selected: pd.DataFrame,
    cohort_dir: Path,
    response_cache_dir: Path,
    device: str,
) -> dict[int, dict[str, np.ndarray]]:
    images = pd.read_csv(cohort_dir / "corrected100_images.csv").set_index("image_index")
    trace_path = response_cache_dir / "input_cache/corrected_trace_segments.npz"
    with np.load(trace_path, allow_pickle=False) as archive:
        trace_ids = archive["trace_index"].astype(int)
        score = archive["score_xy_deg"].astype(float)
    trace_lookup = {int(value): index for index, value in enumerate(trace_ids)}
    common = _load_twin_common()
    payload: dict[int, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for row in selected.itertuples(index=False):
            image = images.loc[int(row.image_index)]
            with np.load(Path(str(image.corrected_patch_npz)), allow_pickle=False) as archive:
                patch = _standardize_uint_like(np.asarray(archive[str(image.corrected_patch_key)], dtype=np.float32))
            retinal_trace = -score[trace_lookup[int(row.trace_index)]]
            tensor = render_retinal_frames_lag_zero(common, patch, retinal_trace, ppd=PPD, device=device)
            movie = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            payload[int(row.matrix_row_index)] = {
                "source_patch": patch,
                "retinal_trace": retinal_trace,
                "scored_movie": movie,
            }
    return payload


def verify_reconstruction(
    selected: pd.DataFrame, payload: dict[int, dict[str, np.ndarray]], spectral: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for condition in selected.itertuples(index=False):
        index = int(condition.spectral_storage_row)
        condition_index = int(condition.matrix_row_index)
        radial, oriented, _ = spectral_statistics(payload[condition_index]["scored_movie"], ppd=PPD)
        saved_radial = spectral["radial_power"][index].astype(float)
        saved_oriented = spectral["orientation_power"][index].astype(float)
        rows.append(
            {
                "spectral_storage_row": index,
                "matrix_row_index": condition_index,
                "image_index": int(condition.image_index),
                "trace_index": int(condition.trace_index),
                "maximum_radial_relative_error": float(np.max(np.abs(radial - saved_radial)) / max(float(saved_radial.max()), EPS)),
                "maximum_oriented_relative_error": float(np.max(np.abs(oriented - saved_oriented)) / max(float(saved_oriented.max()), EPS)),
                "orientation_sum_relative_error": float(np.max(np.abs(oriented.sum(axis=-1) - radial)) / max(float(radial.max()), EPS)),
            }
        )
    return pd.DataFrame(rows)


def build_metrics(
    selected_conditions: pd.DataFrame,
    payload: dict[int, dict[str, np.ndarray]],
    units: pd.DataFrame,
    apertures: dict[int, np.ndarray],
    radial_weights: dict[int, np.ndarray],
    oriented_weights: dict[int, np.ndarray],
    tuning_sf: np.ndarray,
    tuning_tf: np.ndarray,
    spectral: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sf_centers = 0.5 * (spectral["sf_edges_cpd"][:-1] + spectral["sf_edges_cpd"][1:])
    sf_index = indices_for_support(sf_centers, tuning_sf, "SF")
    tf_index = indices_for_support(spectral["tf_hz"], tuning_tf, "TF")
    condition_roles = selected_conditions.set_index("matrix_row_index")
    rows: list[dict[str, object]] = []
    radial_arrays: list[np.ndarray] = []
    oriented_arrays: list[np.ndarray] = []
    for condition in selected_conditions.itertuples(index=False):
        condition_index = int(condition.matrix_row_index)
        movie = payload[condition_index]["scored_movie"]
        for unit in units.rr100_index.astype(int):
            radial_full, oriented_full = localized_oriented_spectrum(
                movie, ppd=PPD, spatial_aperture=apertures[int(unit)]
            )
            radial = radial_full[np.ix_(tf_index, sf_index)]
            oriented = oriented_full[tf_index][:, sf_index, :]
            radial_map = radial * radial_weights[int(unit)]
            oriented_map = np.sum(oriented * oriented_weights[int(unit)], axis=-1)
            radial_drive = float(radial_map.sum())
            oriented_drive = float(oriented_map.sum())
            channels = four_grating_channels(oriented, ORIENTATION_EDGES_DEG).sum(axis=(0, 1))
            fractions = channels / max(float(channels.sum()), EPS)
            rows.append(
                {
                    "array_row": len(rows),
                    "matrix_row_index": condition_index,
                    "image_index": int(condition.image_index),
                    "trace_index": int(condition.trace_index),
                    "condition_selection_role": str(condition_roles.loc[condition_index, "selection_role"]),
                    "rr100_index": int(unit),
                    "radial_direct_f0_drive": radial_drive,
                    "oriented_direct_f0_drive": oriented_drive,
                    "orientation_delta_drive": oriented_drive - radial_drive,
                    "orientation_to_radial_ratio": oriented_drive / max(radial_drive, EPS),
                    "log2_orientation_to_radial": float(np.log2(max(oriented_drive, EPS) / max(radial_drive, EPS))),
                    "rf_local_orientation_concentration": float(fractions.max()),
                    **{f"rf_local_power_fraction_{int(value)}deg": float(fractions[index]) for index, value in enumerate(GRATING_ORIENTATIONS)},
                }
            )
            radial_arrays.append(radial.astype(np.float32))
            oriented_arrays.append(oriented.astype(np.float32))
    return pd.DataFrame(rows), np.stack(radial_arrays), np.stack(oriented_arrays)


def select_pairs(metrics: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("orientation-aligned gain", metrics.log2_orientation_to_radial, "max", "largest log2 oriented/radial drive"),
        ("orientation-mismatch loss", metrics.log2_orientation_to_radial, "min", "smallest log2 oriented/radial drive"),
        ("radial-equivalent control", metrics.log2_orientation_to_radial.abs(), "min", "smallest absolute log2 oriented/radial drive"),
        ("most orientation-structured RF input", metrics.rf_local_orientation_concentration, "max", "largest RF-local four-channel power concentration"),
    ]
    selected: list[pd.Series] = []
    used: set[int] = set()
    for role, values, direction, criterion in definitions:
        available = values.loc[~values.index.isin(used)]
        index = available.idxmax() if direction == "max" else available.idxmin()
        row = metrics.loc[index].copy()
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        row["selection_value"] = float(
            row.rf_local_orientation_concentration if "structured" in role else row.log2_orientation_to_radial
        )
        selected.append(row)
        used.add(int(index))
    return pd.DataFrame(selected)


def plot_checkpoint(
    selected: pd.DataFrame,
    payload: dict[int, dict[str, np.ndarray]],
    apertures: dict[int, np.ndarray],
    radial_weights: dict[int, np.ndarray],
    oriented_weights: dict[int, np.ndarray],
    radial_arrays: np.ndarray,
    oriented_arrays: np.ndarray,
    sf: np.ndarray,
    tf: np.ndarray,
    out: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(len(selected), 7, figsize=(26, 3.6 * len(selected)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_number, selection in enumerate(selected.itertuples(index=False)):
        array_row = int(selection.array_row)
        condition = int(selection.matrix_row_index)
        unit = int(selection.rr100_index)
        item = payload[condition]
        movie = item["scored_movie"]
        aperture = apertures[unit]
        radial = radial_arrays[array_row].astype(float)
        oriented = oriented_arrays[array_row].astype(float)
        radial_map = radial * radial_weights[unit]
        oriented_map = np.sum(oriented * oriented_weights[unit], axis=-1)
        difference = oriented_map - radial_map
        maximum = max(float(radial_map.max()), float(oriented_map.max()), EPS)

        axes[row_number, 0].imshow(item["source_patch"], cmap="gray")
        trace = item["retinal_trace"] * 60.0
        center = np.asarray(item["source_patch"].shape[::-1]) / 2.0
        axes[row_number, 0].plot(center[0] + trace[:, 0] * 2.0, center[1] + trace[:, 1] * 2.0, color="#D55E00", lw=1.1)
        axes[row_number, 0].set_title(f"source image {int(selection.image_index)} + gaze path\n{selection.condition_selection_role}")
        axes[row_number, 0].axis("off")

        strip = np.concatenate([movie[0], movie[len(movie) // 2], movie[-1]], axis=1)
        aperture_strip = np.concatenate([aperture, aperture, aperture], axis=1)
        axes[row_number, 1].imshow(strip, cmap="gray")
        axes[row_number, 1].imshow(aperture_strip, cmap="viridis", alpha=0.42 * aperture_strip / max(float(aperture_strip.max()), EPS))
        axes[row_number, 1].set_title(f"exact retinal frames + RF aperture\nRR100 {unit}: start · middle · end")
        axes[row_number, 1].axis("off")

        fractions = np.asarray([getattr(selection, f"rf_local_power_fraction_{int(value)}deg") for value in GRATING_ORIENTATIONS])
        axes[row_number, 2].bar([f"{value:g}°" for value in GRATING_ORIENTATIONS], fractions, color=["#0072B2", "#E69F00", "#009E73", "#D55E00"])
        axes[row_number, 2].set_ylim(0, 1)
        axes[row_number, 2].set(ylabel="fraction of supported power", title="derived RF-local orientation composition")

        for column, values, title in (
            (3, radial_map, "radial SF×TF accepted power"),
            (4, oriented_map, "oriented SF×θ×TF accepted power"),
        ):
            image = axes[row_number, column].imshow(relative_db(values, maximum), origin="lower", aspect="auto", cmap="magma", vmin=-50, vmax=0)
            axes[row_number, column].set_xticks(range(len(sf)), [f"{value:.2g}" for value in sf], rotation=45)
            axes[row_number, column].set_yticks([0, 5, 10, 15, 17], [f"{tf[index]:g}" for index in [0, 5, 10, 15, 17]])
            axes[row_number, column].set(xlabel="SF (cpd)", ylabel="TF (Hz)" if column == 3 else "", title=title)
        figure.colorbar(image, ax=[axes[row_number, 3], axes[row_number, 4]], label="dB (shared row scale)", fraction=0.03)

        scale = max(float(np.max(np.abs(difference))), EPS)
        image = axes[row_number, 5].imshow(difference, origin="lower", aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale)
        axes[row_number, 5].set_xticks(range(len(sf)), [f"{value:.2g}" for value in sf], rotation=45)
        axes[row_number, 5].set_yticks([0, 5, 10, 15, 17], [f"{tf[index]:g}" for index in [0, 5, 10, 15, 17]])
        axes[row_number, 5].set(xlabel="SF (cpd)", title="orientation correction\noriented − radial")
        figure.colorbar(image, ax=axes[row_number, 5], label="signed accepted power", fraction=0.046)

        axes[row_number, 6].bar(
            ["radial", "oriented"],
            [selection.radial_direct_f0_drive, selection.oriented_direct_f0_drive],
            color=["0.58", "#D55E00"],
        )
        axes[row_number, 6].set_yticks([])
        axes[row_number, 6].set_title(f"{selection.selection_role}\nratio={selection.orientation_to_radial_ratio:.2f}")

    figure.suptitle(
        "Natural-image RF-local orientation checkpoint: exact gaze-cropped movies routed through the grating-validated SF×orientation×TF filters\n"
        "Observed inputs are kept separate from derived power; no neural response is fit or selected",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_conditions, spectral, crosswalk = select_conditions(args.spectral_dir)
    payload = load_selected_movies(selected_conditions, args.cohort_dir, args.response_cache_dir, args.device)
    reconstruction = verify_reconstruction(selected_conditions, payload, spectral)
    units, apertures, radial_weights, oriented_weights, sf, tf = load_rf_and_tuning(
        args.rf_dir, args.tuning_dir, args.session
    )
    metrics, radial_arrays, oriented_arrays = build_metrics(
        selected_conditions,
        payload,
        units,
        apertures,
        radial_weights,
        oriented_weights,
        sf,
        tf,
        spectral,
    )
    selected_pairs = select_pairs(metrics)
    selected_conditions.to_csv(args.out_dir / "selected_conditions.csv", index=False)
    crosswalk.to_csv(args.out_dir / "spectral_storage_crosswalk.csv", index=False)
    reconstruction.to_csv(args.out_dir / "movie_reconstruction_audit.csv", index=False)
    metrics.to_csv(args.out_dir / "selected_condition_unit_metrics.csv", index=False)
    selected_pairs.to_csv(args.out_dir / "selected_condition_unit_examples.csv", index=False)
    np.savez_compressed(
        args.out_dir / "selected_rf_local_oriented_power_arrays.npz",
        radial_power=radial_arrays,
        orientation_power=oriented_arrays,
        sf_cpd=sf,
        tf_hz=tf,
        fourier_orientation_deg=0.5 * (ORIENTATION_EDGES_DEG[:-1] + ORIENTATION_EDGES_DEG[1:]),
        array_row=metrics.array_row.to_numpy(int),
        matrix_row_index=metrics.matrix_row_index.to_numpy(int),
        rr100_index=metrics.rr100_index.to_numpy(int),
    )
    figure_base = args.out_dir / "natural_image_rf_local_oriented_power_checkpoint"
    plot_checkpoint(
        selected_pairs,
        payload,
        apertures,
        radial_weights,
        oriented_weights,
        radial_arrays,
        oriented_arrays,
        sf,
        tf,
        figure_base,
        int(args.dpi),
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_natural_image_rf_local_oriented_power_input_checkpoint",
        "status": "input_mechanism_checkpoint_complete",
        "supersedes": "rr100_natural_image_rf_local_oriented_power_input_checkpoint_v1 (failed reconstruction because the source spectral cache's arrays are image-major but its saved identity axes are matrix-row-major)",
        "scope": {
            "source_condition_bank": 3000,
            "selected_conditions": int(len(selected_conditions)),
            "rf_local_units": int(len(units)),
            "selected_condition_unit_pairs": int(len(metrics)),
        },
        "contracts": {
            "condition_selection": "input-only global spectral quantities; no neural outcomes",
            "spectral_row_mapping": "corrected image-major storage order reconstructed from the cache builder; the cache's embedded condition identity arrays are not aligned",
            "movie": "exact corrected lag-zero 40-frame gaze-cropped natural-image movie reconstructed from frozen image and trace caches",
            "spatial_localization": "same frozen unit-specific RF apertures verified on recorded gratings",
            "spectral_tensor": "positive-TF P(SF magnitude, Fourier-wavevector orientation modulo 180, TF)",
            "neural_weight": "same direct positive-F0 SFxorientationxTF weights used in the grating checkpoint; not squared",
            "response_use": "none",
        },
        "verification": {
            "maximum_cached_radial_reconstruction_relative_error": float(reconstruction.maximum_radial_relative_error.max()),
            "maximum_cached_oriented_reconstruction_relative_error": float(reconstruction.maximum_oriented_relative_error.max()),
            "maximum_orientation_sum_relative_error": float(reconstruction.orientation_sum_relative_error.max()),
            "fraction_cache_rows_with_declared_identity_mismatch": float(
                (
                    (crosswalk.matrix_row_index != crosswalk.declared_matrix_row_index)
                    | (crosswalk.image_index != crosswalk.declared_image_index)
                    | (crosswalk.trace_index != crosswalk.declared_trace_index)
                ).mean()
            ),
            "minimum_oriented_to_radial_drive_ratio": float(metrics.orientation_to_radial_ratio.min()),
            "maximum_oriented_to_radial_drive_ratio": float(metrics.orientation_to_radial_ratio.max()),
        },
        "inputs": {
            "spectral_cache": file_identity(args.spectral_dir / "condition_spectra.npz"),
            "image_cohort": file_identity(args.cohort_dir / "corrected100_images.csv"),
            "trace_cache": file_identity(args.response_cache_dir / "input_cache/corrected_trace_segments.npz"),
            "rf_apertures": file_identity(args.rf_dir / "unit_rf_apertures.npz"),
            "orientation_tuning": file_identity(args.tuning_dir / "orientation_aware_f0_tuning_and_routing.npz"),
        },
        "artifacts": {
            "figure_png": figure_base.with_suffix(".png").name,
            "figure_pdf": figure_base.with_suffix(".pdf").name,
            "selected_conditions": "selected_conditions.csv",
            "spectral_storage_crosswalk": "spectral_storage_crosswalk.csv",
            "reconstruction_audit": "movie_reconstruction_audit.csv",
            "pair_metrics": "selected_condition_unit_metrics.csv",
            "selected_examples": "selected_condition_unit_examples.csv",
            "arrays": "selected_rf_local_oriented_power_arrays.npz",
        },
        "next_checkpoint": "compare radial and oriented RF-local predictors with full-twin natural-image response changes on these same conditions",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
