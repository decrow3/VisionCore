#!/usr/bin/env python3
"""Input checkpoint for power in exact recorded-grating retinal movies.

This deliberately stops before neural prediction.  It loads the held-out
grating trials used by the RR100 recorded-response validation, extracts exact
40-frame gaze-cropped/shifter-corrected stimulus windows, computes the same
positive-TF spectral sufficient statistics used for the corrected natural-
image cache, and renders auditable low/typical/high-power examples.
"""
from __future__ import annotations

import argparse
import copy
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

from models.config_loader import load_dataset_configs
from models.data import prepare_data
from declan.fig4_active_sensing.run_interim_input_spectral_cache import (
    FRAME_RATE_HZ,
    N_HISTORY,
    N_SCORE,
    ORIENTATION_EDGES_DEG,
    SF_EDGES_CPD,
    SF_FIT_MAX_CPD,
    SF_FIT_MIN_CPD,
    spectral_statistics,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/dataset_configs/multi_basic_120_long_legacy.yaml"
DEFAULT_OUT = (
    ROOT
    / "outputs/fig4_active_sensing/rr100_recorded_grating_retinal_power_input_checkpoint_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--session", default="Logan_2020-02-29")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stride", type=int, default=N_SCORE)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.resolve().stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def load_heldout_grating_dataset(
    config_path: Path, session: str, *, preserve_config_cids: bool = False
):
    configs = load_dataset_configs(config_path)
    matches = [row for row in configs if str(row["session"]) == str(session)]
    if len(matches) != 1:
        raise ValueError(f"Expected one configuration for {session}; found {len(matches)}")
    config = copy.deepcopy(matches[0])
    config["types"] = ["gratings"]
    # Unit columns do not affect the input checkpoint. Avoid session-specific
    # cluster-ID indexing by default while preserving the deterministic split.
    # Response-alignment analyses may retain the fitted Allen cids so reloaded
    # robs columns exactly match the historical response cache. Materialized
    # Logan files are already subsetted and must continue to use cids=None.
    if not preserve_config_cids:
        config["cids"] = None
    config["transforms"] = {}
    config["keys_lags"] = {
        "robs": 0,
        "stim": list(range(N_HISTORY + 1)),
        "dfs": 0,
    }
    _, validation, loaded = prepare_data(config, strict=True)
    dataset_indices = validation.get_dataset_inds("gratings")
    if dataset_indices.numel() == 0:
        raise RuntimeError(f"No held-out grating samples for {session}")
    dset_ids = np.unique(dataset_indices[:, 0].detach().cpu().numpy())
    if dset_ids.size != 1:
        raise ValueError(f"Expected one grating dataset; got IDs {dset_ids.tolist()}")
    dset_id = int(dset_ids[0])
    local = np.sort(dataset_indices[:, 1].detach().cpu().numpy().astype(int))
    return validation.dsets[dset_id], local, loaded


def contiguous_validation_runs(local: np.ndarray, trial: np.ndarray) -> list[tuple[int, int]]:
    if local.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    run_start = 0
    for position in range(1, len(local) + 1):
        boundary = position == len(local)
        if not boundary:
            previous = int(local[position - 1])
            current = int(local[position])
            boundary = current != previous + 1 or int(trial[current]) != int(trial[previous])
        if boundary:
            first = int(local[run_start])
            stop = int(local[position - 1]) + 1
            runs.append((first, stop))
            run_start = position
    return runs


def sf_centers() -> np.ndarray:
    # Match the corrected natural-image cache exactly: the routing analysis
    # labels each annular bin by the arithmetic midpoint of its fixed edges.
    return 0.5 * (SF_EDGES_CPD[:-1] + SF_EDGES_CPD[1:])


def candidate_windows(
    dset, local: np.ndarray, stride: int, max_windows: int, *, session: str
):
    if stride <= 0:
        raise ValueError("--stride must be positive")
    stim = dset["stim"].detach().cpu().numpy()
    sf = dset["sf"].detach().cpu().numpy().astype(float)
    ori = dset["ori"].detach().cpu().numpy().astype(float)
    trial = dset["trial_inds"].detach().cpu().numpy().astype(int)
    time_s = dset["t_bins"].detach().cpu().numpy().astype(float)
    dpi_pix = dset["dpi_pix"].detach().cpu().numpy().astype(float)
    ppd = float(np.asarray(dset.metadata["ppd"]).reshape(-1)[0])
    centers = sf_centers()
    tf = np.fft.rfftfreq(N_SCORE, d=1.0 / FRAME_RATE_HZ)[1:]

    rows: list[dict[str, object]] = []
    payload: dict[int, dict[str, np.ndarray]] = {}
    window_index = 0
    for run_start, run_stop in contiguous_validation_runs(local, trial):
        for start in range(run_start, run_stop - N_SCORE + 1, stride):
            stop = start + N_SCORE
            if not np.all(trial[start:stop] == trial[start]):
                continue
            movie_uint8 = np.asarray(stim[start:stop], dtype=np.uint8)
            movie = (movie_uint8.astype(np.float32) - 127.0) / 255.0
            radial, oriented, scalar = spectral_statistics(movie, ppd=ppd)
            if radial.shape != (len(tf), len(centers)):
                raise ValueError(f"Unexpected radial spectrum shape {radial.shape}")
            crop = dpi_pix[start:stop]
            crop_xy_deg = np.column_stack(
                [
                    (crop[:, 1] - crop[:, 1].mean()) / ppd,
                    -(crop[:, 0] - crop[:, 0].mean()) / ppd,
                ]
            )
            steps = np.diff(crop_xy_deg, axis=0)
            fitted_sf = (centers >= SF_FIT_MIN_CPD) & (centers <= SF_FIT_MAX_CPD)
            fitted_power = float(radial[:, fitted_sf].sum())
            positive_power = float(radial.sum())
            rows.append(
                {
                    "window_index": window_index,
                    "session": str(session),
                    "trial_index": int(trial[start]),
                    "start_index_120hz": start,
                    "stop_index_120hz_exclusive": stop,
                    "start_time_s": float(time_s[start]),
                    "stop_time_s": float(time_s[stop - 1] + 1.0 / FRAME_RATE_HZ),
                    "n_frames": N_SCORE,
                    "duration_s": N_SCORE / FRAME_RATE_HZ,
                    "n_unique_spatial_frequencies": int(np.unique(sf[start:stop]).size),
                    "n_unique_orientations": int(np.unique(ori[start:stop]).size),
                    "total_positive_tf_power": positive_power,
                    "fitted_sf_positive_tf_power": fitted_power,
                    "fitted_sf_power_fraction": fitted_power / max(positive_power, 1e-30),
                    "crop_path_length_arcmin": float(np.linalg.norm(steps, axis=1).sum() * 60.0),
                    "crop_rms_radius_arcmin": float(
                        np.sqrt(np.mean(np.sum(crop_xy_deg**2, axis=1))) * 60.0
                    ),
                }
            )
            payload[window_index] = {
                "movie_uint8": movie_uint8,
                "sf_cpd": sf[start:stop].astype(np.float32),
                "orientation_deg": ori[start:stop].astype(np.float32),
                "crop_xy_deg": crop_xy_deg.astype(np.float32),
                "radial_power_tf_sf": radial.astype(np.float32),
                "oriented_power_tf_sf_ori": oriented.astype(np.float32),
                "scalar_power": scalar.astype(np.float64),
            }
            window_index += 1
            if max_windows > 0 and window_index >= max_windows:
                return pd.DataFrame(rows), payload, centers, tf, ppd
    return pd.DataFrame(rows), payload, centers, tf, ppd


def choose_examples(metrics: pd.DataFrame) -> pd.DataFrame:
    if len(metrics) < 3:
        raise RuntimeError(f"Need at least three candidate windows; found {len(metrics)}")
    log_power = np.log10(np.maximum(metrics["total_positive_tf_power"].to_numpy(float), 1e-30))
    roles = [
        ("lowest dynamic power", int(np.argmin(log_power)), "minimum total positive-TF power"),
        (
            "typical dynamic power",
            int(np.argmin(np.abs(log_power - np.median(log_power)))),
            "closest to median log10 total positive-TF power",
        ),
        ("highest dynamic power", int(np.argmax(log_power)), "maximum total positive-TF power"),
    ]
    selected = []
    used: set[int] = set()
    for role, position, criterion in roles:
        row = metrics.iloc[position].copy()
        window_index = int(row["window_index"])
        if window_index in used:
            alternatives = metrics.loc[~metrics["window_index"].isin(used)].copy()
            if role == "lowest dynamic power":
                row = alternatives.loc[alternatives["total_positive_tf_power"].idxmin()].copy()
            elif role == "highest dynamic power":
                row = alternatives.loc[alternatives["total_positive_tf_power"].idxmax()].copy()
            else:
                delta = np.abs(
                    np.log10(np.maximum(alternatives["total_positive_tf_power"], 1e-30))
                    - np.median(log_power)
                )
                row = alternatives.loc[delta.idxmin()].copy()
            window_index = int(row["window_index"])
        used.add(window_index)
        row["selection_role"] = role
        row["selection_criterion"] = criterion
        row["selection_value"] = float(row["total_positive_tf_power"])
        selected.append(row)
    return pd.DataFrame(selected)


def plot_checkpoint(
    selected: pd.DataFrame,
    payload: dict[int, dict[str, np.ndarray]],
    centers: np.ndarray,
    tf: np.ndarray,
    out: Path,
    dpi: int,
) -> None:
    selected_powers = [payload[int(row.window_index)]["radial_power_tf_sf"] for row in selected.itertuples()]
    reference = max(float(np.max(power)) for power in selected_powers)
    fitted = (centers >= SF_FIT_MIN_CPD) & (centers <= SF_FIT_MAX_CPD)
    show_sf = centers > 0
    figure, axes = plt.subplots(len(selected), 5, figsize=(18.5, 3.7 * len(selected)), constrained_layout=True)
    for row_number, row in enumerate(selected.itertuples(index=False)):
        item = payload[int(row.window_index)]
        movie = item["movie_uint8"]
        frame_strip = np.concatenate([movie[0], movie[len(movie) // 2], movie[-1]], axis=1)
        axes[row_number, 0].imshow(frame_strip, cmap="gray", vmin=0, vmax=255)
        axes[row_number, 0].set_title(
            f"{row.selection_role}\nexact frames: start · middle · end\ntrial {int(row.trial_index)}, window {int(row.window_index)}"
        )
        axes[row_number, 0].axis("off")

        crop = item["crop_xy_deg"] * 60.0
        axes[row_number, 1].plot(crop[:, 0], crop[:, 1], "-", color="0.2", lw=1.2)
        axes[row_number, 1].scatter(crop[0, 0], crop[0, 1], s=24, color="#009E73", zorder=3)
        axes[row_number, 1].set_aspect("equal", adjustable="datalim")
        axes[row_number, 1].set(
            xlabel="crop x (arcmin, centered)",
            ylabel="crop y (arcmin, centered)",
            title=f"dataset crop path\nlength {row.crop_path_length_arcmin:.1f} arcmin",
        )
        axes[row_number, 1].grid(alpha=0.2)

        t_ms = np.arange(N_SCORE) / FRAME_RATE_HZ * 1000.0
        color = np.mod(item["orientation_deg"], 180.0)
        scatter = axes[row_number, 2].scatter(t_ms, item["sf_cpd"], c=color, cmap="hsv", vmin=0, vmax=180, s=24)
        axes[row_number, 2].set_yscale("symlog", linthresh=0.5)
        axes[row_number, 2].set(
            xlabel="time in window (ms)",
            ylabel="nominal SF (cpd)",
            title=f"display sequence\n{int(row.n_unique_spatial_frequencies)} SFs · {int(row.n_unique_orientations)} orientations",
        )
        figure.colorbar(scatter, ax=axes[row_number, 2], label="orientation (deg)", fraction=0.046)

        power = item["radial_power_tf_sf"]
        db = 10.0 * np.log10(np.maximum(power[:, show_sf], reference * 1e-6) / max(reference, 1e-30))
        image = axes[row_number, 3].imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=[np.log2(centers[show_sf][0]), np.log2(centers[show_sf][-1]), tf[0], tf[-1]],
            cmap="magma",
            vmin=-60,
            vmax=0,
        )
        xticks = np.asarray([0.5, 1, 2, 4, 8, 16, 32], dtype=float)
        xticks = xticks[(xticks >= centers[show_sf][0]) & (xticks <= centers[show_sf][-1])]
        axes[row_number, 3].set_xticks(np.log2(xticks), [f"{value:g}" for value in xticks])
        axes[row_number, 3].set(
            xlabel="spatial frequency (cpd)",
            ylabel="temporal frequency (Hz)",
            title="exact retinal-movie dynamic power",
        )
        figure.colorbar(image, ax=axes[row_number, 3], label="dB relative to selected maximum", fraction=0.046)

        fitted_power = power[:, fitted]
        bands = np.asarray(
            [
                fitted_power[tf <= 32.0].sum(),
                fitted_power[(tf > 32.0) & (tf <= 45.25)].sum(),
                fitted_power[tf > 45.25].sum(),
            ],
            dtype=float,
        )
        fractions = 100.0 * bands / max(float(bands.sum()), 1e-30)
        axes[row_number, 4].bar(["≤32", "33–45", "48–60"], fractions, color=["#0072B2", "#E69F00", "#D55E00"])
        axes[row_number, 4].set_ylim(0, 100)
        axes[row_number, 4].set(
            xlabel="TF band (Hz)",
            ylabel="fraction of fitted-SF power (%)",
            title=f"power summary\ntotal={row.total_positive_tf_power:.2e} a.u.",
        )

    figure.suptitle(
        "Recorded gratings: exact gaze-cropped retinal movies and their frequency-domain power\n"
        "Held-out trials · 40 frames at 120 Hz · examples selected from input power only; no neural response used",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dset, local, loaded = load_heldout_grating_dataset(args.dataset_config, args.session)
    metrics, payload, centers, tf, ppd = candidate_windows(
        dset, local, int(args.stride), int(args.max_windows), session=args.session
    )
    if metrics.empty:
        raise RuntimeError("No complete held-out 40-frame grating windows were found")
    selected = choose_examples(metrics)
    metrics.to_csv(args.out_dir / "candidate_input_windows.csv", index=False)
    selected.to_csv(args.out_dir / "selected_input_windows.csv", index=False)

    long_rows = []
    archive: dict[str, np.ndarray] = {
        "sf_bin_centers_cpd": centers.astype(np.float64),
        "sf_bin_edges_cpd": SF_EDGES_CPD.astype(np.float64),
        "orientation_bin_edges_deg": ORIENTATION_EDGES_DEG.astype(np.float64),
        "positive_temporal_frequency_hz": tf.astype(np.float64),
    }
    for row in selected.itertuples(index=False):
        window_index = int(row.window_index)
        item = payload[window_index]
        prefix = f"window_{window_index:04d}"
        for key, values in item.items():
            archive[f"{prefix}_{key}"] = values
        for tf_index, temporal in enumerate(tf):
            for sf_index, spatial in enumerate(centers):
                long_rows.append(
                    {
                        "window_index": window_index,
                        "selection_role": row.selection_role,
                        "temporal_frequency_hz": float(temporal),
                        "spatial_frequency_bin_center_cpd": float(spatial),
                        "dynamic_power": float(item["radial_power_tf_sf"][tf_index, sf_index]),
                    }
                )
    np.savez_compressed(args.out_dir / "selected_retinal_movies_and_power.npz", **archive)
    pd.DataFrame(long_rows).to_csv(args.out_dir / "selected_power_long.csv", index=False)
    figure_base = args.out_dir / "recorded_grating_retinal_power_input_checkpoint"
    plot_checkpoint(selected, payload, centers, tf, figure_base, int(args.dpi))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_recorded_grating_retinal_power_input_checkpoint",
        "status": "input_checkpoint_complete_no_neural_test",
        "session": args.session,
        "n_candidate_windows": int(len(metrics)),
        "n_selected_windows": int(len(selected)),
        "contracts": {
            "data_split": "deterministic held-out grating trials from the RR100 dataset configuration",
            "retinal_movie": "exact stored 51x51 gaze-cropped and shifter-corrected stimulus frames",
            "sampling": "dataset 240-to-120-Hz even-frame stimulus decimation",
            "window": f"{N_SCORE} contiguous frames ({N_SCORE / FRAME_RATE_HZ:.6f} s)",
            "spectrum": "temporal-mean residual; Hann temporal and spatial windows; rFFT over time and centered 2D FFT over space; negative-TF partner restored",
            "selection": "low, median-log, and high total positive-TF power; no recorded or model response used",
        },
        "caveats": [
            "The grating dataset is a rapidly changing SF/orientation sequence, not a stationary single-grating epoch.",
            "The displayed dpi_pix path is descriptive; the stored retinal frames are the source of truth for power.",
            "This checkpoint does not yet test whether power predicts recorded firing.",
        ],
        "pixels_per_degree": ppd,
        "loaded_config": {
            "session": str(loaded["session"]),
            "sampling": loaded.get("sampling"),
            "seed": loaded.get("seed"),
            "train_val_split": loaded.get("train_val_split"),
        },
        "inputs": {"dataset_config": file_identity(args.dataset_config)},
        "artifacts": {
            "candidate_windows": "candidate_input_windows.csv",
            "selected_windows": "selected_input_windows.csv",
            "selected_movies_and_power": "selected_retinal_movies_and_power.npz",
            "selected_power_long": "selected_power_long.csv",
            "figure_png": "recorded_grating_retinal_power_input_checkpoint.png",
            "figure_pdf": "recorded_grating_retinal_power_input_checkpoint.pdf",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(selected[["selection_role", "window_index", "trial_index", "total_positive_tf_power", "crop_path_length_arcmin"]].to_string(index=False))
    print(f"Wrote input checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
