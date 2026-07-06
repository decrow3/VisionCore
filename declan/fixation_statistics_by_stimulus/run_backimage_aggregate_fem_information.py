"""Aggregate BackImage FEM information pilot with empirical trajectory controls.

This runner is deliberately narrower than the fixed-axis local screens.  It
asks whether an ensemble of natural-image patches is better represented under
empirical FEM-like motion distributions than under matched synthetic controls.
The primary controls are OU trajectories matched to empirical RMS/autocorrelation,
Brownian trajectories matched to RMS, and rotated empirical trajectories.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from tqdm import tqdm

try:
    from .extraction import _as_numpy, _load_dict_dataset
    from .image_features import _backimage_canvas, gaze_deg_to_screen_px
    from .run_backimage_latent_information_screen import (
        CanonicalTwinScorer,
        HAVE_STEERABLE_PYRAMID,
        _align_response_to_trace,
        _central_crop,
        _clip_patch,
        _cross_validated_decode,
        _dct_features,
        _extract_latents,
        _mean_r2,
        GABOR_ENERGY_GRID_BY_NAME,
        GABOR_ENERGY_ORIENTCOV_GRID_BY_NAME,
        GABOR_ENERGY_ORIENTCOV_TYPEBALANCED_NAMES,
        GABOR_LOCAL_FIELD_GRID4,
        _gabor_energy_bands_from_filter_bands,
        _gabor_energy_orientcov_type_balance_weights,
        _gabor_features,
        _gabor_filter_bands,
        _gabor_features_from_bands,
        _gabor_local_energy_from_bands,
        _gabor_local_energy_orientcov_from_bands,
        _parse_float_list,
        _parse_int_list,
        _parse_str_list,
        _pyramid_complex_energy_bands,
        _pyramid_energy_orientcov_type_balance_weights,
        _pyramid_local_energy_from_bands,
        _pyramid_local_energy_orientcov_from_bands,
        PYRAMID_ENERGY_GRID_BY_NAME,
        PYRAMID_ENERGY_ORIENTCOV_GRID_BY_NAME,
        PYRAMID_ENERGY_ORIENTCOV_TYPEBALANCED_NAMES,
        _pyramid_phase_preserving_scale_balance_weights,
        _pyramid_features,
        _pyramid_features_phase_preserving,
        _scale_token,
        _static_trace,
        _standardize_uint_like,
        _standardize_train_test,
        _split_outer,
        _trace_xy_to_twin_helper_order,
        _trace_rms,
        _write_json,
    )
    from .run_fixation_statistics_by_stimulus import load_sessions
    from jake.twininfo.eye_controls import detect_microsaccade_events, speed_threshold_mad
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.fixation_statistics_by_stimulus.extraction import _as_numpy, _load_dict_dataset
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_latent_information_screen import (
        CanonicalTwinScorer,
        HAVE_STEERABLE_PYRAMID,
        _align_response_to_trace,
        _central_crop,
        _clip_patch,
        _cross_validated_decode,
        _dct_features,
        _extract_latents,
        _mean_r2,
        GABOR_ENERGY_GRID_BY_NAME,
        GABOR_ENERGY_ORIENTCOV_GRID_BY_NAME,
        GABOR_ENERGY_ORIENTCOV_TYPEBALANCED_NAMES,
        GABOR_LOCAL_FIELD_GRID4,
        _gabor_energy_bands_from_filter_bands,
        _gabor_energy_orientcov_type_balance_weights,
        _gabor_features,
        _gabor_filter_bands,
        _gabor_features_from_bands,
        _gabor_local_energy_from_bands,
        _gabor_local_energy_orientcov_from_bands,
        _parse_float_list,
        _parse_int_list,
        _parse_str_list,
        _pyramid_complex_energy_bands,
        _pyramid_energy_orientcov_type_balance_weights,
        _pyramid_local_energy_from_bands,
        _pyramid_local_energy_orientcov_from_bands,
        PYRAMID_ENERGY_GRID_BY_NAME,
        PYRAMID_ENERGY_ORIENTCOV_GRID_BY_NAME,
        PYRAMID_ENERGY_ORIENTCOV_TYPEBALANCED_NAMES,
        _pyramid_phase_preserving_scale_balance_weights,
        _pyramid_features,
        _pyramid_features_phase_preserving,
        _scale_token,
        _static_trace,
        _standardize_uint_like,
        _standardize_train_test,
        _split_outer,
        _trace_xy_to_twin_helper_order,
        _trace_rms,
        _write_json,
    )
    from declan.fixation_statistics_by_stimulus.run_fixation_statistics_by_stimulus import load_sessions
    from jake.twininfo.eye_controls import detect_microsaccade_events, speed_threshold_mad


PHASE_PRESERVING_LATENT = "pyramid_local_field_phase_preserving"
PHASE_PRESERVING_SCALE_BALANCED_LATENT = "pyramid_local_field_phase_preserving_scale_balanced"
PYRAMID_STATS_ENERGY_ORIENTCOV_LATENT = "pyramid_local_stats_energy_orientcov"


DEFAULT_INPUT = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_aggregate_fem_information_pilot"
)


@dataclass(frozen=True)
class AggregateConfig:
    input: str
    out_dir: str
    window_manifest: str | None
    max_images: int
    trace_samples_per_condition: int
    motion_families: list[str]
    observed_rms_scales: list[float]
    patch_size_px: int
    latent_crop_px: int
    center_crop_px: int
    local_field_grid: int
    feature_target_mode: str
    spatial_readout_mode: str
    spatial_readout_radius: int
    n_timepoints: int
    temporal_pc_components: int
    pca_k_list: list[int]
    latent_names: list[str]
    ridge_alphas: list[float]
    fixed_ridge_alpha: float | None
    outer_folds: int
    inner_folds: int
    decode_group_mode: str
    reliable_image_coherence_min: float
    reliable_drift_anisotropy_min: float
    min_duration_s: float
    min_patch_image_margin_px: float
    max_rms_deg: float
    max_trace_source_rms_deg: float | None
    max_trace_source_radius_deg: float | None
    max_trace_source_path_length_deg: float | None
    max_rendered_trace_path_length_deg: float | None
    max_source_trace_path_length_deg: float | None
    max_trace_source_speed_p95_deg_s: float | None
    max_trace_source_microsaccade_events: int | None
    microsaccade_speed_threshold_dps: float | None
    microsaccade_threshold_z: float
    microsaccade_pad_frames: int
    reuse_trace_sources_across_scales: bool
    twin_batch_size: int
    twin_trace_batch_size: int
    device: str
    progress_every: int
    seed: int
    dry_run: bool
    save_response_sample_arrays: bool
    compute_ssi_features: bool
    ssi_summary_names: list[str]
    ssi_incremental_base_summaries: list[str]


def _progress(message: str) -> None:
    print(f"[backimage-aggregate-fem] {message}", flush=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class AggregateSsiTwinScorer(CanonicalTwinScorer):
    """Canonical scorer that also exposes spatial-SSI summaries before max-pooling."""

    def __init__(self, *, device: str, batch_size: int, empty_cache_every_batch: bool = False):
        super().__init__(device=device, batch_size=batch_size, empty_cache_every_batch=empty_cache_every_batch)
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from scripts.spatial_info import spatial_ssi_population

        self.spatial_ssi_population = spatial_ssi_population

    def _ssi_from_rate_map(self, rate_map: Any) -> dict[str, np.ndarray]:
        rate_map = rate_map.clamp_min(0.0)
        ispike_t, irate_t, i_tn = self.spatial_ssi_population(rate_map, dt=1.0)
        rbar_tn = rate_map.reshape(rate_map.shape[0], rate_map.shape[1], -1).mean(dim=2)
        return {
            "ssi_itn": i_tn.detach().cpu().numpy().astype(np.float32, copy=False),
            "ssi_rbar_tn": rbar_tn.detach().cpu().numpy().astype(np.float32, copy=False),
            "ssi_ispike_t": ispike_t.detach().cpu().numpy().astype(np.float32, copy=False),
            "ssi_irate_t": irate_t.detach().cpu().numpy().astype(np.float32, copy=False),
        }

    def responses_with_ssi(
        self,
        patch: np.ndarray,
        traces: list[np.ndarray],
        *,
        trace_batch_size: int = 1,
    ) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
        if not traces:
            return []
        image = _standardize_uint_like(patch)
        trace_batch_size = max(1, int(trace_batch_size))
        out: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
        for start in range(0, len(traces), trace_batch_size):
            trace_chunk = traces[start : start + trace_batch_size]
            stims = []
            lengths = []
            for trace in trace_chunk:
                trace = np.asarray(trace, dtype=np.float32)
                full_stack = np.broadcast_to(
                    image[None, :, :],
                    (trace.shape[0] + self.common.N_LAGS + 1, *image.shape),
                ).copy()
                eye = self.torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
                stim = self.common.make_counterfactual_stim(
                    full_stack,
                    eye,
                    ppd=self.common.PPD,
                    scale_factor=1.0,
                    n_lags=self.common.N_LAGS,
                    out_size=self.common.OUT_SIZE,
                )
                stims.append((stim - 127.0) / 255.0)
                lengths.append(int(stim.shape[0]))
            rate_map = self._compute_rate_map_batched(self.torch.cat(stims, dim=0))
            rates = rate_map.amax(dim=(-2, -1)).detach().cpu().numpy().astype(np.float32, copy=False)
            ssi_all = self._ssi_from_rate_map(rate_map)
            offset = 0
            for length in lengths:
                response = rates[offset : offset + length]
                ssi = {key: value[offset : offset + length] for key, value in ssi_all.items()}
                out.append((response, ssi))
                offset += length
            del stims, rate_map, rates, ssi_all
        return out


def _rate_map_shift_scale(scorer: CanonicalTwinScorer, rate_map_shape: tuple[int, int]) -> tuple[float, float]:
    h_map, w_map = int(rate_map_shape[0]), int(rate_map_shape[1])
    space_weights = getattr(scorer.ctx.readout, "space_weights", None)
    if space_weights is not None:
        kernel_h = int(space_weights.shape[-2])
        kernel_w = int(space_weights.shape[-1])
    else:
        kernel_h = kernel_w = 14
    core_h = h_map + kernel_h - 1
    core_w = w_map + kernel_w - 1
    out_h, out_w = int(scorer.common.OUT_SIZE[0]), int(scorer.common.OUT_SIZE[1])
    return core_h / float(out_h), core_w / float(out_w)


def _pool_rate_map_response(
    rate_map: Any,
    trace: np.ndarray,
    scorer: CanonicalTwinScorer,
    *,
    mode: str,
    radius: int,
) -> np.ndarray:
    values = rate_map.detach().cpu().numpy().astype(np.float32, copy=False)
    if str(mode) == "amax":
        return values.max(axis=(-2, -1)).astype(np.float32, copy=False)
    if values.ndim != 4:
        raise ValueError(f"Expected rate_map shape (T, N, H, W), got {values.shape}")
    T, n_units, h_map, w_map = values.shape
    radius = max(0, int(radius))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float64)
    offsets_y = yy.reshape(-1)
    offsets_x = xx.reshape(-1)
    center_y = (float(h_map) - 1.0) / 2.0
    center_x = (float(w_map) - 1.0) / 2.0
    trace_rs = _resample_trace(np.asarray(trace, dtype=np.float64), int(T))
    scale_y, scale_x = _rate_map_shift_scale(scorer, (h_map, w_map))
    unit_idx = np.repeat(np.arange(n_units, dtype=np.float64), offsets_x.size)
    out = np.empty((T, n_units), dtype=np.float32)
    for t in range(T):
        if str(mode) == "center_mean":
            dx_map = 0.0
            dy_map = 0.0
        elif str(mode) == "trace_registered_center_mean":
            dx_map = float(trace_rs[t, 0]) * float(scorer.common.PPD) * scale_x
            dy_map = -float(trace_rs[t, 1]) * float(scorer.common.PPD) * scale_y
        else:
            raise ValueError(f"Unknown spatial readout mode {mode!r}")
        sample_y = center_y + dy_map + offsets_y
        sample_x = center_x + dx_map + offsets_x
        coords_y = np.tile(sample_y, n_units)
        coords_x = np.tile(sample_x, n_units)
        sampled = map_coordinates(
            values[t],
            [unit_idx, coords_y, coords_x],
            order=1,
            mode="nearest",
            prefilter=False,
        ).reshape(n_units, offsets_x.size)
        out[t] = np.mean(sampled, axis=1)
    return out


def _responses_with_spatial_readout(
    scorer: CanonicalTwinScorer,
    patch: np.ndarray,
    traces: list[np.ndarray],
    *,
    trace_batch_size: int,
    spatial_readout_mode: str,
    spatial_readout_radius: int,
) -> list[np.ndarray]:
    if not traces:
        return []
    image = _standardize_uint_like(patch)
    trace_batch_size = max(1, int(trace_batch_size))
    out: list[np.ndarray] = []
    for start in range(0, len(traces), trace_batch_size):
        trace_chunk = traces[start : start + trace_batch_size]
        stims = []
        lengths = []
        for trace in trace_chunk:
            trace = np.asarray(trace, dtype=np.float32)
            full_stack = np.broadcast_to(
                image[None, :, :],
                (trace.shape[0] + scorer.common.N_LAGS + 1, *image.shape),
            ).copy()
            eye = scorer.torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
            stim = scorer.common.make_counterfactual_stim(
                full_stack,
                eye,
                ppd=scorer.common.PPD,
                scale_factor=1.0,
                n_lags=scorer.common.N_LAGS,
                out_size=scorer.common.OUT_SIZE,
            )
            stims.append((stim - 127.0) / 255.0)
            lengths.append(int(stim.shape[0]))
        rate_map = scorer._compute_rate_map_batched(scorer.torch.cat(stims, dim=0))
        offset = 0
        for trace, length in zip(trace_chunk, lengths, strict=True):
            pooled = _pool_rate_map_response(
                rate_map[offset : offset + length],
                trace,
                scorer,
                mode=str(spatial_readout_mode),
                radius=int(spatial_readout_radius),
            )
            out.append(pooled.astype(np.float32, copy=False))
            offset += length
        del stims, rate_map
    return out


def _prepare_windows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    df["source_row"] = np.arange(df.shape[0], dtype=int)
    required = [
        "session",
        "trial_idx",
        "global_start",
        "global_stop",
        "mean_x_deg",
        "mean_y_deg",
        "anisotropy",
        "image_orientation_coherence",
        "image_patch_distance_to_image_border_px",
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "duration_s" not in df.columns:
        df["duration_s"] = df.get("epoch_duration_s", np.nan)
    margin = float(args.min_patch_image_margin_px) if args.min_patch_image_margin_px is not None else float(args.patch_size_px) / 2.0
    keep = (
        np.isfinite(df["mean_x_deg"].astype(float))
        & np.isfinite(df["mean_y_deg"].astype(float))
        & (df["anisotropy"].astype(float) >= float(args.reliable_drift_anisotropy_min))
        & (df["image_orientation_coherence"].astype(float) >= float(args.reliable_image_coherence_min))
        & (df["duration_s"].astype(float) >= float(args.min_duration_s))
        & (df["image_patch_distance_to_image_border_px"].astype(float) >= margin)
    )
    work = df.loc[keep].copy()
    if args.window_manifest is not None:
        manifest = pd.read_csv(args.window_manifest)
        if "source_row" in manifest.columns:
            requested = manifest["source_row"].astype(int).drop_duplicates().to_list()
            available = set(work["source_row"].astype(int).to_list())
            missing_ids = sorted(set(requested).difference(available))
            if missing_ids:
                preview = ", ".join(str(v) for v in missing_ids[:10])
                suffix = "..." if len(missing_ids) > 10 else ""
                raise ValueError(f"--window-manifest source_row values do not survive filters: {preview}{suffix}")
            work = work.set_index("source_row", drop=False).loc[requested].reset_index(drop=True)
        else:
            matched_rows: list[int] = []
            missing_rows: list[int] = []
            float_pairs = [
                ("mean_x_deg", "mean_x_deg"),
                ("mean_y_deg", "mean_y_deg"),
                ("image_orientation_coherence", "image_orientation_coherence"),
                ("drift_anisotropy", "anisotropy"),
                ("observed_rms_radius_deg", "rms_radius_deg"),
                ("actual_observed_rms_deg", "rms_radius_deg"),
            ]
            for manifest_idx, manifest_row in manifest.drop_duplicates().iterrows():
                mask = np.ones(work.shape[0], dtype=bool)
                if "session" in manifest_row.index and "session" in work.columns and pd.notna(manifest_row["session"]):
                    mask &= work["session"].astype(str).to_numpy() == str(manifest_row["session"])
                if "trial_idx" in manifest_row.index and "trial_idx" in work.columns and pd.notna(manifest_row["trial_idx"]):
                    mask &= work["trial_idx"].astype(int).to_numpy() == int(manifest_row["trial_idx"])
                if "phase" in manifest_row.index and "phase" in work.columns and pd.notna(manifest_row["phase"]):
                    mask &= work["phase"].astype(str).to_numpy() == str(manifest_row["phase"])
                for manifest_col, work_col in float_pairs:
                    if manifest_col in manifest_row.index and work_col in work.columns and pd.notna(manifest_row[manifest_col]):
                        mask &= np.isclose(
                            work[work_col].astype(float).to_numpy(),
                            float(manifest_row[manifest_col]),
                            rtol=1e-7,
                            atol=1e-7,
                            equal_nan=True,
                        )
                matches = np.flatnonzero(mask)
                if matches.size == 0:
                    missing_rows.append(int(manifest_idx))
                    continue
                if matches.size > 1:
                    raise ValueError(
                        "Ambiguous --window-manifest row without source_row; multiple rows match "
                        f"session={manifest_row.get('session', '')!r}, trial_idx={manifest_row.get('trial_idx', '')!r}."
                    )
                matched_rows.append(int(matches[0]))
            if missing_rows:
                preview = ", ".join(str(v) for v in missing_rows[:10])
                suffix = "..." if len(missing_rows) > 10 else ""
                raise ValueError(
                    "--window-manifest lacks source_row and some rows could not be matched by "
                    f"session/trial/geometry after current filters: manifest rows {preview}{suffix}"
                )
            work = work.iloc[matched_rows].reset_index(drop=True)
    elif int(args.max_images) > 0 and work.shape[0] > int(args.max_images):
        work = work.sample(n=int(args.max_images), replace=False, random_state=int(args.seed))
        work = work.sort_values(["session", "trial_idx", "source_row"])
    work["image_index"] = np.arange(work.shape[0], dtype=int)
    return work.reset_index(drop=True)


def _session_dataset_cache(sessions: list[str]) -> dict[str, np.ndarray]:
    session_objects = {str(getattr(s, "name", s)): s for s in load_sessions(",".join(sorted(set(sessions))))}
    out: dict[str, np.ndarray] = {}
    for name in sorted(set(sessions)):
        session = session_objects[name]
        dset_path = Path(session.sess_dir) / "datasets" / "backimage.dset"
        dset = _load_dict_dataset(dset_path)
        out[name] = _as_numpy(dset["eyepos"]).astype(np.float64)
    return out


def _extract_requested_latents(
    patch: np.ndarray,
    *,
    latent_crop_px: int,
    center_crop_px: int,
    local_field_grid: int,
    requested: set[str],
) -> dict[str, np.ndarray]:
    if not requested:
        return _extract_latents(
            patch,
            latent_crop_px=int(latent_crop_px),
            center_crop_px=int(center_crop_px),
            local_field_grid=int(local_field_grid),
        )
    image = _standardize_uint_like(patch)
    out: dict[str, np.ndarray] = {}
    need_field = any(
        name.endswith("_local_field")
        or name in {PHASE_PRESERVING_LATENT, PHASE_PRESERVING_SCALE_BALANCED_LATENT, PYRAMID_STATS_ENERGY_ORIENTCOV_LATENT}
        or name in PYRAMID_ENERGY_GRID_BY_NAME
        or name in PYRAMID_ENERGY_ORIENTCOV_GRID_BY_NAME
        or name == GABOR_LOCAL_FIELD_GRID4
        or name in GABOR_ENERGY_GRID_BY_NAME
        or name in GABOR_ENERGY_ORIENTCOV_GRID_BY_NAME
        for name in requested
    )
    need_center = any(name.endswith("_center") for name in requested)
    field_crop = _central_crop(image, int(latent_crop_px)) if need_field else None
    center_crop = _central_crop(image, int(center_crop_px)) if need_center else None
    if "dct_center" in requested and center_crop is not None:
        out["dct_center"] = _dct_features(center_crop, n_freq=8)
    if "dct_local_field" in requested and field_crop is not None:
        out["dct_local_field"] = _dct_features(field_crop, n_freq=8)
    if "gabor_center" in requested and center_crop is not None:
        out["gabor_center"] = _gabor_features(center_crop, scope="center", local_grid=int(local_field_grid))
    requested_gabor_energy_names = set(GABOR_ENERGY_GRID_BY_NAME).union(GABOR_ENERGY_ORIENTCOV_GRID_BY_NAME).intersection(requested)
    requested_gabor_field_names = {"gabor_local_field", GABOR_LOCAL_FIELD_GRID4}.intersection(requested)
    gabor_bands = _gabor_filter_bands(field_crop) if field_crop is not None and (requested_gabor_energy_names or requested_gabor_field_names) else []
    if "gabor_local_field" in requested and field_crop is not None:
        out["gabor_local_field"] = _gabor_features_from_bands(gabor_bands, scope="local_field", local_grid=int(local_field_grid))
    if GABOR_LOCAL_FIELD_GRID4 in requested and field_crop is not None:
        out[GABOR_LOCAL_FIELD_GRID4] = _gabor_features_from_bands(gabor_bands, scope="local_field", local_grid=4)
    if requested_gabor_energy_names and field_crop is not None:
        gabor_energy_bands = _gabor_energy_bands_from_filter_bands(gabor_bands)
        for name, grid in GABOR_ENERGY_GRID_BY_NAME.items():
            if name in requested:
                out[name] = _gabor_local_energy_from_bands(gabor_energy_bands, local_grid=int(grid))
        for name, grid in GABOR_ENERGY_ORIENTCOV_GRID_BY_NAME.items():
            if name in requested:
                out[name] = _gabor_local_energy_orientcov_from_bands(gabor_energy_bands, local_grid=int(grid))
    if HAVE_STEERABLE_PYRAMID and "pyramid_center" in requested and center_crop is not None:
        out["pyramid_center"] = _pyramid_features(center_crop, scope="center", local_grid=int(local_field_grid))
    if HAVE_STEERABLE_PYRAMID and "pyramid_local_field" in requested and field_crop is not None:
        out["pyramid_local_field"] = _pyramid_features(field_crop, scope="local_field", local_grid=int(local_field_grid))
    phase_requested = bool({PHASE_PRESERVING_LATENT, PHASE_PRESERVING_SCALE_BALANCED_LATENT}.intersection(requested))
    if HAVE_STEERABLE_PYRAMID and phase_requested and field_crop is not None:
        phase_features = _pyramid_features_phase_preserving(field_crop, scope="local_field", local_grid=int(local_field_grid))
        if PHASE_PRESERVING_LATENT in requested:
            out[PHASE_PRESERVING_LATENT] = phase_features
        if PHASE_PRESERVING_SCALE_BALANCED_LATENT in requested:
            out[PHASE_PRESERVING_SCALE_BALANCED_LATENT] = phase_features
    if HAVE_STEERABLE_PYRAMID and field_crop is not None:
        requested_energy_names = set(PYRAMID_ENERGY_GRID_BY_NAME).union(PYRAMID_ENERGY_ORIENTCOV_GRID_BY_NAME).intersection(requested)
        energy_bands = _pyramid_complex_energy_bands(field_crop) if requested_energy_names else []
        for name, grid in PYRAMID_ENERGY_GRID_BY_NAME.items():
            if name in requested:
                out[name] = _pyramid_local_energy_from_bands(energy_bands, local_grid=int(grid))
        for name, grid in PYRAMID_ENERGY_ORIENTCOV_GRID_BY_NAME.items():
            if name in requested:
                out[name] = _pyramid_local_energy_orientcov_from_bands(energy_bands, local_grid=int(grid))
    return {key: value for key, value in out.items() if value.size > 0}


def _clip_patch_subpixel(canvas: np.ndarray, center_xy_px: tuple[float, float], size_px: int) -> np.ndarray:
    size_px = int(size_px)
    half = size_px // 2
    cx, cy = float(center_xy_px[0]), float(center_xy_px[1])
    x = cx + np.arange(size_px, dtype=np.float64) - float(half)
    y = cy + np.arange(size_px, dtype=np.float64) - float(half)
    xx, yy = np.meshgrid(x, y)
    fill = float(np.nanmean(canvas))
    patch = map_coordinates(
        np.asarray(canvas, dtype=np.float32),
        [yy, xx],
        order=1,
        mode="constant",
        cval=fill,
        prefilter=False,
    )
    return np.asarray(patch, dtype=np.float32)


def _trace_registered_latents(
    canvas: np.ndarray,
    center_px: np.ndarray,
    trace: np.ndarray,
    *,
    ppd: float,
    patch_size_px: int,
    latent_crop_px: int,
    center_crop_px: int,
    local_field_grid: int,
    requested: set[str],
) -> dict[str, np.ndarray]:
    """Average feature targets over source-image locations sampled by a trace."""
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise ValueError(f"Expected trace shape (T, 2), got {trace.shape}")
    center = np.asarray(center_px, dtype=np.float64).reshape(2)
    if trace.shape[0] <= 1 or float(np.nanmax(np.linalg.norm(trace - trace[:1], axis=1))) <= 1e-12:
        xy = trace[0] if trace.shape[0] else np.zeros(2, dtype=np.float64)
        frame_center = (
            float(center[0] - float(xy[0]) * float(ppd)),
            float(center[1] + float(xy[1]) * float(ppd)),
        )
        frame_patch = _clip_patch_subpixel(canvas, frame_center, int(patch_size_px))
        return _extract_requested_latents(
            frame_patch,
            latent_crop_px=int(latent_crop_px),
            center_crop_px=int(center_crop_px),
            local_field_grid=int(local_field_grid),
            requested=requested,
        )
    by_name: dict[str, list[np.ndarray]] = {}
    for xy in trace:
        # Match make_counterfactual_stim: source x samples move opposite the
        # x trace, while positive y samples lower screen rows after the helper
        # converts gaze degrees to grid_sample coordinates.
        frame_center = (
            float(center[0] - float(xy[0]) * float(ppd)),
            float(center[1] + float(xy[1]) * float(ppd)),
        )
        frame_patch = _clip_patch_subpixel(canvas, frame_center, int(patch_size_px))
        latents = _extract_requested_latents(
            frame_patch,
            latent_crop_px=int(latent_crop_px),
            center_crop_px=int(center_crop_px),
            local_field_grid=int(local_field_grid),
            requested=requested,
        )
        for name, value in latents.items():
            by_name.setdefault(name, []).append(np.asarray(value, dtype=np.float32))
    return {
        name: np.mean(np.vstack(values), axis=0).astype(np.float32)
        for name, values in by_name.items()
        if values
    }


def _resample_trace(trace: np.ndarray, n_timepoints: int) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2:
        return np.zeros((int(n_timepoints), 2), dtype=np.float32)
    idx = np.linspace(0, trace.shape[0] - 1, int(n_timepoints))
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    frac = idx - lo
    out = trace[lo] * (1.0 - frac[:, None]) + trace[hi] * frac[:, None]
    finite = np.isfinite(out).all(axis=1)
    if not np.all(finite):
        good = np.flatnonzero(finite)
        if good.size == 0:
            out = np.zeros_like(out)
        else:
            bad = np.flatnonzero(~finite)
            for dim in range(2):
                out[bad, dim] = np.interp(bad, good, out[good, dim])
    out -= np.mean(out, axis=0, keepdims=True)
    return out.astype(np.float32)


def _scale_to_rms(trace: np.ndarray, target_rms: float, *, max_rms_deg: float) -> tuple[np.ndarray, dict[str, Any]]:
    trace = np.asarray(trace, dtype=np.float64)
    base_rms = _trace_rms(trace)
    requested = float(target_rms)
    effective_target = min(max(requested, 0.0), float(max_rms_deg))
    clipped_high = bool(requested > float(max_rms_deg))
    if base_rms <= 1e-12 or effective_target <= 0.0:
        scaled = np.zeros_like(trace)
    else:
        scaled = trace * (effective_target / base_rms)
        scaled -= np.mean(scaled, axis=0, keepdims=True)
    return scaled.astype(np.float32), {
        "base_rms_deg": float(base_rms),
        "requested_rms_deg": float(requested),
        "effective_rms_deg": float(_trace_rms(scaled)),
        "rms_clipped_high": clipped_high,
    }


def _path_length(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trace, axis=0), axis=1)))


def _speed_summary(trace: np.ndarray, dt: float) -> dict[str, float]:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2:
        return {"speed_mean_deg_s": 0.0, "speed_median_deg_s": 0.0, "speed_p95_deg_s": 0.0}
    speed = np.linalg.norm(np.diff(trace, axis=0), axis=1) / float(dt)
    return {
        "speed_mean_deg_s": float(np.nanmean(speed)),
        "speed_median_deg_s": float(np.nanmedian(speed)),
        "speed_p95_deg_s": float(np.nanpercentile(speed, 95.0)),
    }


def _microsaccade_stats(
    trace: np.ndarray,
    *,
    dt: float,
    threshold_dps: float | None,
    threshold_z: float,
    pad_frames: int,
) -> dict[str, float | int]:
    threshold = (
        float(threshold_dps)
        if threshold_dps is not None
        else speed_threshold_mad(np.asarray(trace, dtype=np.float64), dt=float(dt), z=float(threshold_z))
    )
    events, sample_mask, _threshold = detect_microsaccade_events(
        np.asarray(trace, dtype=np.float64),
        dt=float(dt),
        threshold_deg_s=threshold,
        min_samples=1,
        pad_samples=max(0, int(pad_frames)),
    )
    peak = max((float(event["peak_speed_deg_s"]) for event in events), default=0.0)
    return {
        "microsaccade_threshold_dps": float(threshold),
        "n_microsaccade_events": int(len(events)),
        "fraction_microsaccade_samples": float(np.mean(sample_mask)) if sample_mask.size else 0.0,
        "peak_microsaccade_speed_dps": float(peak),
    }


def _trace_covariance_shape(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    cov = np.cov(trace, rowvar=False) if trace.shape[0] > 1 else np.eye(2)
    if not np.all(np.isfinite(cov)):
        cov = np.eye(2)
    vals, vecs = np.linalg.eigh(cov + 1e-9 * np.eye(2))
    vals = np.maximum(vals, 1e-9)
    shape = vecs @ np.diag(np.sqrt(vals / np.mean(vals))) @ vecs.T
    return shape.astype(np.float64)


def _trace_covariance_anisotropy(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=np.float64)
    cov = np.cov(trace, rowvar=False) if trace.shape[0] > 1 else np.eye(2)
    if not np.all(np.isfinite(cov)):
        return float("nan")
    vals = np.linalg.eigvalsh(cov + 1e-12 * np.eye(2))
    vals = np.maximum(vals, 0.0)
    total = float(np.sum(vals))
    if total <= 1e-12:
        return 0.0
    return float((np.max(vals) - np.min(vals)) / total)


def _lag1_autocorr(trace: np.ndarray) -> float:
    x = np.asarray(trace, dtype=np.float64)
    if x.shape[0] < 3:
        return 0.0
    vals = []
    for dim in range(2):
        a = x[:-1, dim] - np.mean(x[:-1, dim])
        b = x[1:, dim] - np.mean(x[1:, dim])
        den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if den > 1e-12:
            vals.append(float(np.sum(a * b) / den))
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals), -0.95, 0.98))


def _brownian_trace(n_timepoints: int, rng: np.random.Generator) -> np.ndarray:
    inc = rng.normal(size=(int(n_timepoints), 2))
    trace = np.cumsum(inc, axis=0)
    trace -= np.mean(trace, axis=0, keepdims=True)
    return trace.astype(np.float32)


def _ou_trace(n_timepoints: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    rho = float(np.clip(rho, -0.95, 0.98))
    x = np.zeros((int(n_timepoints), 2), dtype=np.float64)
    sigma = float(np.sqrt(max(1e-6, 1.0 - rho * rho)))
    x[0] = rng.normal(size=2)
    for t in range(1, int(n_timepoints)):
        x[t] = rho * x[t - 1] + sigma * rng.normal(size=2)
    x -= np.mean(x, axis=0, keepdims=True)
    return x.astype(np.float32)


def _rotated_trace(trace: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    rot = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    out = np.asarray(trace, dtype=np.float64) @ rot.T
    out -= np.mean(out, axis=0, keepdims=True)
    return out.astype(np.float32)


def _build_trace_bank(
    work: pd.DataFrame,
    eyepos_by_session: dict[str, np.ndarray],
    n_timepoints: int,
    *,
    microsaccade_speed_threshold_dps: float | None,
    microsaccade_threshold_z: float,
    microsaccade_pad_frames: int,
) -> list[dict[str, Any]]:
    bank: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        eyepos = eyepos_by_session[str(row["session"])]
        start = int(row["global_start"])
        stop = int(row["global_stop"])
        source_trace = _resample_trace(eyepos[start:stop], max(2, int(stop - start)))
        duration_s = float(row.get("duration_s", np.nan))
        source_dt = (
            duration_s / float(max(1, source_trace.shape[0] - 1))
            if np.isfinite(duration_s) and duration_s > 0.0
            else 1.0 / 120.0
        )
        source_ms = _microsaccade_stats(
            source_trace,
            dt=source_dt,
            threshold_dps=microsaccade_speed_threshold_dps,
            threshold_z=float(microsaccade_threshold_z),
            pad_frames=int(microsaccade_pad_frames),
        )
        trace = _resample_trace(eyepos[start:stop], int(n_timepoints))
        rendered_ms = _microsaccade_stats(
            trace,
            dt=1.0 / 120.0,
            threshold_dps=microsaccade_speed_threshold_dps,
            threshold_z=float(microsaccade_threshold_z),
            pad_frames=int(microsaccade_pad_frames),
        )
        bank.append(
            {
                "source_row": int(row["source_row"]),
                "session": str(row["session"]),
                "trial_idx": int(row.get("trial_idx", -1)),
                "global_start": int(start),
                "global_stop": int(stop),
                "mean_x_deg": float(row.get("mean_x_deg", np.nan)),
                "mean_y_deg": float(row.get("mean_y_deg", np.nan)),
                "trace": trace,
                "observed_rms_deg": float(_trace_rms(trace)),
                "source_trace_observed_rms_deg": float(_trace_rms(source_trace)),
                "source_rms_radius_deg": float(row.get("rms_radius_deg", np.nan)),
                "source_max_radius_deg": float(row.get("max_radius_deg", np.nan)),
                "path_length_deg": _path_length(trace),
                "source_path_length_deg": float(row.get("path_length_deg", np.nan)),
                "source_speed_p95_deg_s": float(row.get("speed_p95_deg_s", np.nan)),
                "duration_s": duration_s,
                "lag1_autocorr": _lag1_autocorr(trace),
                "covariance_shape": _trace_covariance_shape(trace),
                "trace_cov_anisotropy": _trace_covariance_anisotropy(trace),
                "source_trace_cov_anisotropy": _trace_covariance_anisotropy(source_trace),
                "source_anisotropy": float(row.get("anisotropy", np.nan)),
                "source_microsaccade_threshold_dps": float(source_ms["microsaccade_threshold_dps"]),
                "source_n_microsaccade_events": int(source_ms["n_microsaccade_events"]),
                "source_fraction_microsaccade_samples": float(source_ms["fraction_microsaccade_samples"]),
                "source_peak_microsaccade_speed_dps": float(source_ms["peak_microsaccade_speed_dps"]),
                "rendered_microsaccade_threshold_dps": float(rendered_ms["microsaccade_threshold_dps"]),
                "rendered_n_microsaccade_events": int(rendered_ms["n_microsaccade_events"]),
                "rendered_fraction_microsaccade_samples": float(rendered_ms["fraction_microsaccade_samples"]),
                "rendered_peak_microsaccade_speed_dps": float(rendered_ms["peak_microsaccade_speed_dps"]),
                "microsaccade_threshold_dps": float(source_ms["microsaccade_threshold_dps"]),
                "n_microsaccade_events": int(source_ms["n_microsaccade_events"]),
                "fraction_microsaccade_samples": float(source_ms["fraction_microsaccade_samples"]),
                "peak_microsaccade_speed_dps": float(source_ms["peak_microsaccade_speed_dps"]),
            }
        )
    return bank


def _eligible_trace_bank_indices(
    trace_bank: list[dict[str, Any]],
    *,
    current_source_row: int,
    max_trace_source_rms_deg: float | None,
    max_trace_source_radius_deg: float | None,
    max_trace_source_path_length_deg: float | None,
    max_rendered_trace_path_length_deg: float | None,
    max_source_trace_path_length_deg: float | None,
    max_trace_source_speed_p95_deg_s: float | None,
    max_trace_source_microsaccade_events: int | None,
) -> list[int]:
    rendered_path_limit = (
        max_rendered_trace_path_length_deg
        if max_rendered_trace_path_length_deg is not None
        else max_trace_source_path_length_deg
    )

    def over_limit(value: Any, limit: float | None) -> bool:
        if limit is None:
            return False
        val = float(value)
        return (not np.isfinite(val)) or val > float(limit)

    eligible = []
    for j, item in enumerate(trace_bank):
        if int(item["source_row"]) == int(current_source_row):
            continue
        if over_limit(item["observed_rms_deg"], max_trace_source_rms_deg):
            continue
        if over_limit(item["source_max_radius_deg"], max_trace_source_radius_deg):
            continue
        if over_limit(item["path_length_deg"], rendered_path_limit):
            continue
        if over_limit(item["source_path_length_deg"], max_source_trace_path_length_deg):
            continue
        if over_limit(item["source_speed_p95_deg_s"], max_trace_source_speed_p95_deg_s):
            continue
        if (
            max_trace_source_microsaccade_events is not None
            and int(item["n_microsaccade_events"]) > int(max_trace_source_microsaccade_events)
        ):
            continue
        eligible.append(j)
    return eligible


def _family_raw_trace(
    family: str,
    source_trace: np.ndarray,
    source_rho: float,
    *,
    rng: np.random.Generator,
    max_rms_deg: float,
    source_shape: np.ndarray | None = None,
    selection_rms: float | None = None,
    target_path_length: float | None = None,
) -> np.ndarray:
    if family == "empirical":
        raw = np.asarray(source_trace, dtype=np.float32)
    elif family == "rotated":
        raw = _rotated_trace(source_trace, rng)
    elif family == "brownian":
        raw = _brownian_trace(source_trace.shape[0], rng)
    elif family == "ou":
        best_raw = None
        best_loss = float("inf")
        shape = np.eye(2) if source_shape is None else np.asarray(source_shape, dtype=np.float64)
        for _ in range(12):
            candidate = np.asarray(_ou_trace(source_trace.shape[0], source_rho, rng), dtype=np.float64) @ shape.T
            candidate -= np.mean(candidate, axis=0, keepdims=True)
            if target_path_length is None or not np.isfinite(target_path_length):
                loss = 0.0
            else:
                cand_scaled, _ = _scale_to_rms(
                    candidate,
                    float(selection_rms) if selection_rms is not None else _trace_rms(source_trace),
                    max_rms_deg=max_rms_deg,
                )
                loss = abs(_path_length(cand_scaled) - float(target_path_length))
            if loss < best_loss:
                best_loss = float(loss)
                best_raw = candidate
        raw = np.asarray(best_raw, dtype=np.float32)
    else:
        raise ValueError(f"Unknown motion family {family!r}")
    raw = np.asarray(raw, dtype=np.float64)
    raw -= np.mean(raw, axis=0, keepdims=True)
    return raw.astype(np.float32)


def _scale_family_raw_trace(
    raw: np.ndarray,
    target_rms: float,
    *,
    max_rms_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    trace, meta = _scale_to_rms(raw, target_rms, max_rms_deg=max_rms_deg)
    meta["generated_lag1_autocorr"] = _lag1_autocorr(trace)
    meta["path_length_deg"] = _path_length(trace)
    meta.update(_speed_summary(trace, dt=1.0 / 120.0))
    return trace, meta


def _family_trace(
    family: str,
    source_trace: np.ndarray,
    source_rho: float,
    target_rms: float,
    *,
    rng: np.random.Generator,
    max_rms_deg: float,
    source_shape: np.ndarray | None = None,
    target_path_length: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = _family_raw_trace(
        family,
        source_trace,
        source_rho,
        rng=rng,
        max_rms_deg=max_rms_deg,
        source_shape=source_shape,
        selection_rms=target_rms,
        target_path_length=target_path_length,
    )
    return _scale_family_raw_trace(raw, target_rms, max_rms_deg=max_rms_deg)


def _fit_temporal_basis(responses: list[np.ndarray], n_components: int) -> np.ndarray:
    if not responses:
        raise ValueError("No responses available for temporal basis")
    T = int(responses[0].shape[0])
    cov = np.zeros((T, T), dtype=np.float64)
    count = 0
    for resp in responses:
        arr = np.asarray(resp, dtype=np.float64)
        if arr.shape[0] != T:
            raise ValueError("All responses must have the same time length for temporal PCA")
        arr = arr - np.mean(arr, axis=0, keepdims=True)
        cov += arr @ arr.T
        count += int(arr.shape[1])
    cov /= max(1, count)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    basis = vecs[:, order[: int(min(n_components, T))]]
    for j in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, j])))
        if basis[pivot, j] < 0:
            basis[:, j] *= -1.0
    return basis.astype(np.float32)


def _fixed_dct_basis(n_timepoints: int, n_components: int) -> np.ndarray:
    t = np.arange(int(n_timepoints), dtype=np.float64)
    basis = []
    for k in range(1, int(n_components) + 1):
        vec = np.cos(np.pi * (t + 0.5) * float(k) / float(n_timepoints))
        vec = vec - np.mean(vec)
        vec = vec / (np.sqrt(np.sum(vec * vec)) + 1e-12)
        basis.append(vec)
    return np.column_stack(basis).astype(np.float32)


def _summarize_response(response: np.ndarray, static: np.ndarray, basis: np.ndarray) -> dict[str, np.ndarray]:
    response = np.asarray(response, dtype=np.float32)
    static = np.asarray(static, dtype=np.float32)
    delta = response - static
    return {
        "temporal_pca": (basis.T @ response).reshape(-1).astype(np.float32),
        "temporal_delta_pca": (basis.T @ delta).reshape(-1).astype(np.float32),
        "mean": np.mean(response, axis=0).astype(np.float32),
        "delta_mean": np.mean(delta, axis=0).astype(np.float32),
    }


def _align_ssi_to_trace(ssi: dict[str, np.ndarray], n_timepoints: int) -> dict[str, np.ndarray]:
    return {key: _align_response_to_trace(value, int(n_timepoints)) for key, value in ssi.items()}


def _summarize_ssi_features(
    ssi: dict[str, np.ndarray],
    summary_names: list[str],
    *,
    static_ssi: dict[str, np.ndarray] | None = None,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    i_tn = np.asarray(ssi["ssi_itn"], dtype=np.float32)
    rbar_tn = np.asarray(ssi["ssi_rbar_tn"], dtype=np.float32)
    ispike_t = np.asarray(ssi["ssi_ispike_t"], dtype=np.float32)
    irate_t = np.asarray(ssi["ssi_irate_t"], dtype=np.float32)
    if i_tn.shape != rbar_tn.shape:
        raise ValueError(f"SSI I_tn shape {i_tn.shape} does not match rbar shape {rbar_tn.shape}")
    weighted_unit = np.sum(i_tn * rbar_tn, axis=0) / (np.sum(rbar_tn, axis=0) + float(eps))
    available = {
        "ssi_itn": i_tn.reshape(-1).astype(np.float32),
        "ssi_unit_mean": np.mean(i_tn, axis=0).astype(np.float32),
        "ssi_unit_spike_weighted_mean": weighted_unit.astype(np.float32),
        "ssi_rbar_itn": rbar_tn.reshape(-1).astype(np.float32),
        "ssi_itn_plus_rbar": np.concatenate([i_tn.reshape(-1), rbar_tn.reshape(-1)]).astype(np.float32),
        "ssi_population_time": np.concatenate([ispike_t.reshape(-1), irate_t.reshape(-1)]).astype(np.float32),
    }
    if static_ssi is not None:
        static_i_tn = np.asarray(static_ssi["ssi_itn"], dtype=np.float32)
        static_rbar_tn = np.asarray(static_ssi["ssi_rbar_tn"], dtype=np.float32)
        static_ispike_t = np.asarray(static_ssi["ssi_ispike_t"], dtype=np.float32)
        static_irate_t = np.asarray(static_ssi["ssi_irate_t"], dtype=np.float32)
        if static_i_tn.shape != i_tn.shape:
            raise ValueError(f"Static SSI I_tn shape {static_i_tn.shape} does not match response SSI shape {i_tn.shape}")
        if static_rbar_tn.shape != rbar_tn.shape:
            raise ValueError(f"Static SSI rbar shape {static_rbar_tn.shape} does not match response rbar shape {rbar_tn.shape}")
        delta_i_tn = i_tn - static_i_tn
        delta_rbar_tn = rbar_tn - static_rbar_tn
        delta_ispike_t = ispike_t - static_ispike_t
        delta_irate_t = irate_t - static_irate_t
        weighted_delta_unit = np.sum(delta_i_tn * rbar_tn, axis=0) / (np.sum(rbar_tn, axis=0) + float(eps))
        available.update(
            {
                "delta_ssi_itn": delta_i_tn.reshape(-1).astype(np.float32),
                "delta_ssi_unit_mean": np.mean(delta_i_tn, axis=0).astype(np.float32),
                "delta_ssi_unit_spike_weighted_mean": weighted_delta_unit.astype(np.float32),
                "delta_ssi_rbar_itn": delta_rbar_tn.reshape(-1).astype(np.float32),
                "delta_ssi_itn_plus_rbar": np.concatenate([delta_i_tn.reshape(-1), delta_rbar_tn.reshape(-1)]).astype(np.float32),
                "delta_ssi_population_time": np.concatenate([delta_ispike_t.reshape(-1), delta_irate_t.reshape(-1)]).astype(np.float32),
            }
        )
    missing = sorted(set(summary_names).difference(available))
    if missing:
        raise ValueError(f"Unknown SSI summary names: {missing}; available={sorted(available)}")
    return {name: available[name] for name in summary_names}


def _delta_ssi_summary_name(name: str) -> str:
    name = str(name)
    return name if name.startswith("delta_ssi_") else f"delta_{name}"


def _expand_ssi_summary_names(summary_names: list[str]) -> list[str]:
    out: list[str] = []
    for name in summary_names:
        for candidate in (str(name), _delta_ssi_summary_name(str(name))):
            if candidate not in out:
                out.append(candidate)
    return out


def _ssi_component_for_incremental_base(
    summaries: dict[str, np.ndarray],
    *,
    base: str,
    ssi_name: str,
) -> str:
    if str(base).startswith("delta_"):
        delta_name = _delta_ssi_summary_name(str(ssi_name))
        if delta_name in summaries:
            return delta_name
    return str(ssi_name)


def _add_ssi_incremental_summaries(
    summaries: dict[str, np.ndarray],
    *,
    base_summaries: list[str],
    ssi_summary_names: list[str],
) -> list[str]:
    added: list[str] = []
    for base in base_summaries:
        if base not in summaries:
            raise ValueError(f"Cannot build SSI incremental summary; missing base summary {base!r}")
        for ssi_name in ssi_summary_names:
            component_name = _ssi_component_for_incremental_base(summaries, base=str(base), ssi_name=str(ssi_name))
            if component_name not in summaries:
                raise ValueError(f"Cannot build SSI incremental summary; missing SSI summary {component_name!r}")
            out_name = f"{base}_plus_{component_name}"
            summaries[out_name] = np.concatenate(
                [
                    np.asarray(summaries[base], dtype=np.float32).reshape(-1),
                    np.asarray(summaries[component_name], dtype=np.float32).reshape(-1),
                ]
            ).astype(np.float32)
            added.append(out_name)
    return added


def _is_ssi_summary_name(name: str) -> bool:
    text = str(name)
    return text.startswith("ssi_") or text.startswith("delta_ssi_") or "_plus_ssi_" in text or "_plus_delta_ssi_" in text


def _add_temporal_basis_summaries(
    out: dict[str, np.ndarray],
    response: np.ndarray,
    static: np.ndarray,
    basis: np.ndarray,
    *,
    prefix: str,
) -> None:
    response = np.asarray(response, dtype=np.float32)
    static = np.asarray(static, dtype=np.float32)
    delta = response - static
    out[f"{prefix}"] = (basis.T @ response).reshape(-1).astype(np.float32)
    out[f"{prefix}_delta"] = (basis.T @ delta).reshape(-1).astype(np.float32)


def _stack_condition_features(
    records: list[dict[str, Any]],
    summaries: dict[int, dict[str, np.ndarray]],
    summary_name: str,
) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec["family"] == "static":
            key = ("static", "static")
        else:
            key = (str(rec["family"]), str(rec["scale_id"]))
        grouped[key][int(rec["image_index"])].append(summaries[int(rec["response_id"])][summary_name])
    out: dict[tuple[str, str], np.ndarray] = {}
    for key, by_image in grouped.items():
        image_features = []
        for image_index in sorted(by_image):
            image_features.append(np.mean(np.vstack(by_image[image_index]), axis=0))
        out[key] = np.vstack(image_features).astype(np.float32)
    return out


def _stack_condition_sample_features(
    records: list[dict[str, Any]],
    summaries: dict[int, dict[str, np.ndarray]],
    summary_name: str,
) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec["family"] == "static":
            key = ("static", "static")
        else:
            key = (str(rec["family"]), str(rec["scale_id"]))
        grouped[key][int(rec["image_index"])].append(summaries[int(rec["response_id"])][summary_name])
    out: dict[tuple[str, str], np.ndarray] = {}
    for key, by_image in grouped.items():
        image_indices = sorted(by_image)
        counts = {len(by_image[i]) for i in image_indices}
        if len(counts) != 1:
            raise ValueError(f"Condition {key} has unequal sample counts across images: {sorted(counts)}")
        if key == ("static", "static"):
            out[key] = np.vstack([by_image[i][0] for i in image_indices]).astype(np.float32)
            continue
        out[key] = np.stack([np.vstack(by_image[i]) for i in image_indices], axis=0).astype(np.float32)
    return out


def _stack_condition_targets(
    records: list[dict[str, Any]],
    target_latents_by_response: dict[int, dict[str, np.ndarray]],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    grouped: dict[tuple[str, str], dict[int, list[dict[str, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        response_id = int(rec["response_id"])
        if response_id not in target_latents_by_response:
            raise ValueError(f"Missing feature target latents for response_id={response_id}")
        if rec["family"] == "static":
            key = ("static", "static")
        else:
            key = (str(rec["family"]), str(rec["scale_id"]))
        grouped[key][int(rec["image_index"])].append(target_latents_by_response[response_id])

    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, by_image in grouped.items():
        names = sorted({name for items in by_image.values() for item in items for name in item})
        condition_targets: dict[str, list[np.ndarray]] = {name: [] for name in names}
        for image_index in sorted(by_image):
            items = by_image[image_index]
            for name in names:
                values = [np.asarray(item[name], dtype=np.float32) for item in items if name in item]
                if len(values) != len(items):
                    raise ValueError(f"Condition {key} image {image_index} is missing registered target {name!r}")
                condition_targets[name].append(np.mean(np.vstack(values), axis=0).astype(np.float32))
        out[key] = {name: np.vstack(values).astype(np.float32) for name, values in condition_targets.items()}
    return out


def _decode_groups_from_images(images: pd.DataFrame, mode: str) -> np.ndarray:
    mode = str(mode)
    if mode == "image":
        return images["image_index"].to_numpy(dtype=int)
    if mode == "source_trial":
        missing = sorted({"session", "trial_idx"}.difference(images.columns))
        if missing:
            raise ValueError(f"source_trial decode grouping requires columns {missing}")
        return (
            images["session"].astype(str)
            + "::trial_"
            + images["trial_idx"].astype(int).astype(str)
        ).to_numpy()
    if mode == "session":
        return images["session"].to_numpy()
    raise ValueError(f"Unknown decode_group_mode={mode!r}")


def _bootstrap_condition_delta(
    per_image_a: np.ndarray,
    per_image_b: np.ndarray,
    sessions: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    delta = np.asarray(per_image_a, dtype=np.float64) - np.asarray(per_image_b, dtype=np.float64)
    obs = float(np.nanmean(delta))
    if int(n_bootstrap) <= 0:
        return obs, float("nan"), float("nan")
    unique_sessions = np.unique(sessions)
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for j in range(int(n_bootstrap)):
        sampled_sessions = rng.choice(unique_sessions, size=unique_sessions.size, replace=True)
        idx = np.concatenate([np.flatnonzero(sessions == sess) for sess in sampled_sessions])
        boot[j] = float(np.nanmean(delta[idx]))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return obs, float(lo), float(hi)


def _parse_contrast_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in _parse_str_list(text):
        if ":" in part:
            lhs, rhs = part.split(":", 1)
        elif ">" in part:
            lhs, rhs = part.split(">", 1)
        elif "-" in part:
            lhs, rhs = part.split("-", 1)
        else:
            raise ValueError(
                "Contrast pairs must use lhs:rhs, lhs>rhs, or lhs-rhs syntax; "
                f"got {part!r}"
            )
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not lhs or not rhs:
            raise ValueError(f"Invalid empty contrast pair entry: {part!r}")
        pairs.append((lhs, rhs))
    return pairs


def _condition_motion_tensor(
    records: list[dict[str, Any]],
    summaries: dict[int, dict[str, np.ndarray]],
    summary_name: str,
    key: tuple[str, str],
) -> np.ndarray:
    by_image: dict[int, list[np.ndarray]] = defaultdict(list)
    for rec in records:
        rec_key = ("static", "static") if rec["family"] == "static" else (str(rec["family"]), str(rec["scale_id"]))
        if rec_key != key:
            continue
        by_image[int(rec["image_index"])].append(summaries[int(rec["response_id"])][summary_name])
    arrays = [np.vstack(by_image[i]) for i in sorted(by_image)]
    return np.asarray(arrays, dtype=np.float32)


def _subspace_overlap(A: np.ndarray, B: np.ndarray, max_dim: int) -> float:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] < 2 or B.shape[0] < 2:
        return float("nan")
    A -= np.mean(A, axis=0, keepdims=True)
    B -= np.mean(B, axis=0, keepdims=True)
    _, _, vt_a = np.linalg.svd(A, full_matrices=False)
    _, _, vt_b = np.linalg.svd(B, full_matrices=False)
    dim = int(min(max_dim, vt_a.shape[0], vt_b.shape[0]))
    if dim < 1:
        return float("nan")
    overlap = np.linalg.norm(vt_a[:dim] @ vt_b[:dim].T, ord="fro") ** 2 / float(dim)
    return float(overlap)


def _covariance_rows(
    records: list[dict[str, Any]],
    summaries: dict[int, dict[str, np.ndarray]],
    summary_names: list[str],
    condition_keys: list[tuple[str, str]],
    *,
    overlap_dim: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_name in summary_names:
        for family, scale_id in condition_keys:
            tensor = _condition_motion_tensor(records, summaries, summary_name, (family, scale_id))
            if tensor.ndim != 3:
                continue
            mu = np.mean(tensor, axis=1)
            residual = (tensor - mu[:, None, :]).reshape(-1, tensor.shape[-1])
            signal_trace = float(np.sum(np.var(mu, axis=0, ddof=1))) if mu.shape[0] > 1 else float("nan")
            motion_trace = float(np.sum(np.var(residual, axis=0, ddof=1))) if residual.shape[0] > 1 else float("nan")
            rows.append(
                {
                    "summary": summary_name,
                    "family": family,
                    "scale_id": scale_id,
                    "n_images": int(tensor.shape[0]),
                    "n_trace_samples": int(tensor.shape[1]),
                    "feature_dim": int(tensor.shape[2]),
                    "signal_cov_trace": signal_trace,
                    "motion_cov_trace": motion_trace,
                    "signal_motion_trace_ratio": signal_trace / (motion_trace + 1e-12) if np.isfinite(signal_trace) else float("nan"),
                    "signal_motion_subspace_overlap": _subspace_overlap(mu, residual, int(overlap_dim)),
                }
            )
    return rows


def _decode_rows(
    feature_by_condition: dict[tuple[str, str], np.ndarray],
    latent_arrays: dict[str, np.ndarray],
    groups: np.ndarray,
    args: argparse.Namespace,
    *,
    latent_arrays_by_condition: dict[tuple[str, str], dict[str, np.ndarray]] | None = None,
    target_weight_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, int], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    per_image: dict[tuple[str, str, str, int], np.ndarray] = {}
    alphas = _parse_float_list(args.ridge_alphas)
    fixed_alpha = float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else float(alphas[len(alphas) // 2])
    for (family, scale_id), X in sorted(feature_by_condition.items()):
        condition_key = (family, scale_id)
        condition_latents = (
            latent_arrays
            if latent_arrays_by_condition is None
            else latent_arrays_by_condition.get(condition_key, {})
        )
        if not condition_latents:
            raise ValueError(f"No latent targets available for condition {condition_key}")
        for latent_name, Z in sorted(condition_latents.items()):
            target_weights = None if target_weight_arrays is None else target_weight_arrays.get(latent_name)
            if target_weights is None:
                target_weighting = "none"
            elif latent_name in PYRAMID_ENERGY_ORIENTCOV_TYPEBALANCED_NAMES:
                target_weighting = "pyramid_energy_orientcov_type_balanced_after_zscore"
            elif latent_name in GABOR_ENERGY_ORIENTCOV_TYPEBALANCED_NAMES:
                target_weighting = "gabor_energy_orientcov_type_balanced_after_zscore"
            else:
                target_weighting = "pyramid_scale_balanced_after_zscore"
            for k in _parse_int_list(args.pca_k_list):
                result = _cross_validated_decode(
                    X,
                    Z,
                    groups,
                    k=int(k),
                    alphas=alphas,
                    alpha_mode="fixed",
                    fixed_alpha=fixed_alpha,
                    outer_folds=int(args.outer_folds),
                    inner_folds=int(args.inner_folds),
                    seed=int(args.seed),
                    target_feature_weights=target_weights,
                )
                key = (family, scale_id, latent_name, int(k))
                per_image[key] = np.asarray(result["per_window_score"], dtype=np.float64)
                rows.append(
                    {
                        "family": family,
                        "scale_id": scale_id,
                        "latent": latent_name,
                        "k": int(k),
                        "mean_neg_mse": float(result["mean_neg_mse"]),
                        "r2": float(result["r2"]),
                        "chosen_alpha_median": float(result["chosen_alpha_median"]),
                        "ridge_alpha_mode": "fixed",
                        "fixed_ridge_alpha": fixed_alpha,
                        "target_dim": int(result["target_dim"]),
                        "n_images": int(X.shape[0]),
                        "decode_group_mode": str(args.decode_group_mode),
                        "n_decode_groups": int(np.unique(groups).size),
                        "feature_dim": int(X.shape[1]),
                        "target_weighting": target_weighting,
                        "feature_target_mode": str(getattr(args, "feature_target_mode", "static")),
                    }
                )
    return rows, per_image


def _target_weighting_label(latent_name: str, target_weights: np.ndarray | None) -> str:
    if target_weights is None:
        return "none"
    if latent_name in PYRAMID_ENERGY_ORIENTCOV_TYPEBALANCED_NAMES:
        return "pyramid_energy_orientcov_type_balanced_after_zscore"
    if latent_name in GABOR_ENERGY_ORIENTCOV_TYPEBALANCED_NAMES:
        return "gabor_energy_orientcov_type_balanced_after_zscore"
    return "pyramid_scale_balanced_after_zscore"


def _standardize_with_train_stats(values: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - mean) / sd


def _decode_rows_shared_target_basis(
    feature_by_condition: dict[tuple[str, str], np.ndarray],
    latent_arrays_by_condition: dict[tuple[str, str], dict[str, np.ndarray]],
    groups: np.ndarray,
    args: argparse.Namespace,
    *,
    target_weight_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, int], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    per_image: dict[tuple[str, str, str, int], np.ndarray] = {}
    alphas = _parse_float_list(args.ridge_alphas)
    fixed_alpha = float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else float(alphas[len(alphas) // 2])
    groups = np.asarray(groups)
    condition_keys = sorted(feature_by_condition)
    latent_names = sorted(
        set.intersection(
            *[
                set(latent_arrays_by_condition.get(condition_key, {}))
                for condition_key in condition_keys
            ]
        )
    )
    if not latent_names:
        raise ValueError("No shared registered target latent names are available across all conditions")
    splits = _split_outer(groups, int(args.outer_folds), int(args.seed))
    min_train_n = min((len(train_idx) for train_idx, _ in splits), default=int(groups.size))

    for latent_name in latent_names:
        target_by_condition = {
            condition_key: np.asarray(latent_arrays_by_condition[condition_key][latent_name], dtype=np.float64)
            for condition_key in condition_keys
        }
        target_dims = {Z.shape[1] for Z in target_by_condition.values()}
        if len(target_dims) != 1:
            raise ValueError(f"Registered target {latent_name!r} has inconsistent dimensions: {sorted(target_dims)}")
        target_dim = int(next(iter(target_dims)))
        target_weights = None if target_weight_arrays is None else target_weight_arrays.get(latent_name)
        target_weights_arr = None
        if target_weights is not None:
            target_weights_arr = np.asarray(target_weights, dtype=np.float64).reshape(-1)
            if target_weights_arr.shape[0] != target_dim:
                raise ValueError(
                    f"Target feature weights length {target_weights_arr.shape[0]} "
                    f"does not match target feature dimension {target_dim}"
                )
        target_weighting = _target_weighting_label(latent_name, target_weights_arr)
        for k in _parse_int_list(args.pca_k_list):
            n = int(groups.size)
            k_eff = int(min(int(k), target_dim, max(1, min_train_n * len(condition_keys))))
            pred_by_condition = {
                condition_key: np.full((n, k_eff), np.nan, dtype=np.float64)
                for condition_key in condition_keys
            }
            target_by_condition_pc = {
                condition_key: np.full((n, k_eff), np.nan, dtype=np.float64)
                for condition_key in condition_keys
            }
            fold_r2s_by_condition: dict[tuple[str, str], list[float]] = {condition_key: [] for condition_key in condition_keys}
            chosen_alphas_by_condition: dict[tuple[str, str], list[float]] = {condition_key: [] for condition_key in condition_keys}
            for fold, (train_idx, test_idx) in enumerate(splits):
                ref_train = np.vstack([Z[train_idx] for Z in target_by_condition.values()])
                target_mean = np.nanmean(ref_train, axis=0, keepdims=True)
                target_sd = np.nanstd(ref_train, axis=0, keepdims=True)
                target_sd[~np.isfinite(target_sd) | (target_sd <= 1e-12)] = 1.0
                ref_train_std = _standardize_with_train_stats(ref_train, target_mean, target_sd)
                if target_weights_arr is not None:
                    ref_train_std = ref_train_std * target_weights_arr[None, :]
                pca = PCA(n_components=k_eff, svd_solver="full")
                pca.fit(ref_train_std)
                for condition_key in condition_keys:
                    X = np.asarray(feature_by_condition[condition_key], dtype=np.float64)
                    Z = target_by_condition[condition_key]
                    if X.shape[0] != n or Z.shape[0] != n:
                        raise ValueError(
                            f"Condition {condition_key} has X/Z rows {X.shape[0]}/{Z.shape[0]}, expected {n}"
                        )
                    X_train_raw, X_test = _standardize_train_test(X[train_idx], X[test_idx])
                    Z_train_raw = _standardize_with_train_stats(Z[train_idx], target_mean, target_sd)
                    Z_test_raw = _standardize_with_train_stats(Z[test_idx], target_mean, target_sd)
                    if target_weights_arr is not None:
                        Z_train_raw = Z_train_raw * target_weights_arr[None, :]
                        Z_test_raw = Z_test_raw * target_weights_arr[None, :]
                    Y_train = pca.transform(Z_train_raw)
                    Y_test = pca.transform(Z_test_raw)
                    alpha = fixed_alpha
                    model = Ridge(alpha=float(alpha), fit_intercept=True)
                    model.fit(X_train_raw, Y_train)
                    Y_pred = np.asarray(model.predict(X_test), dtype=np.float64)
                    if Y_pred.ndim == 1:
                        Y_pred = Y_pred[:, None]
                    pred_by_condition[condition_key][test_idx] = Y_pred
                    target_by_condition_pc[condition_key][test_idx] = Y_test
                    fold_r2s_by_condition[condition_key].append(_mean_r2(Y_test, Y_pred))
                    chosen_alphas_by_condition[condition_key].append(float(alpha))
            for condition_key in condition_keys:
                family, scale_id = condition_key
                target = target_by_condition_pc[condition_key]
                pred = pred_by_condition[condition_key]
                mse = np.mean((target - pred) ** 2, axis=1)
                key = (family, scale_id, latent_name, int(k))
                per_image[key] = -mse
                valid_fold_r2s = [float(v) for v in fold_r2s_by_condition[condition_key] if np.isfinite(v)]
                rows.append(
                    {
                        "family": family,
                        "scale_id": scale_id,
                        "latent": latent_name,
                        "k": int(k),
                        "mean_neg_mse": float(np.nanmean(-mse)),
                        "r2": float(np.mean(valid_fold_r2s)) if valid_fold_r2s else float("nan"),
                        "chosen_alpha_median": float(np.nanmedian(chosen_alphas_by_condition[condition_key])),
                        "ridge_alpha_mode": "fixed",
                        "fixed_ridge_alpha": fixed_alpha,
                        "target_dim": int(k_eff),
                        "raw_target_dim": int(target_dim),
                        "n_images": n,
                        "decode_group_mode": str(args.decode_group_mode),
                        "n_decode_groups": int(np.unique(groups).size),
                        "feature_dim": int(feature_by_condition[condition_key].shape[1]),
                        "target_weighting": target_weighting,
                        "feature_target_mode": str(getattr(args, "feature_target_mode", "static")),
                        "target_basis_mode": "shared_registered_train_conditions",
                    }
                )
    return rows, per_image


def _ssi_incremental_decode_rows(
    decode_rows: list[dict[str, Any]],
    per_image: dict[tuple[str, str, str, str, int], np.ndarray],
    sessions: np.ndarray,
    args: argparse.Namespace,
    *,
    incremental_summary_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not incremental_summary_names:
        return rows
    rng = np.random.default_rng(int(args.seed) + 303)
    row_lookup = {
        (
            str(row["response_summary"]),
            str(row["family"]),
            str(row["scale_id"]),
            str(row["latent"]),
            int(row["k"]),
        ): row
        for row in decode_rows
    }
    incremental_set = set(incremental_summary_names)
    for row in decode_rows:
        combined_summary = str(row["response_summary"])
        if combined_summary not in incremental_set or "_plus_" not in combined_summary:
            continue
        base_summary, ssi_summary = combined_summary.split("_plus_", 1)
        key = (
            combined_summary,
            str(row["family"]),
            str(row["scale_id"]),
            str(row["latent"]),
            int(row["k"]),
        )
        base_key = (
            base_summary,
            str(row["family"]),
            str(row["scale_id"]),
            str(row["latent"]),
            int(row["k"]),
        )
        if key not in per_image or base_key not in per_image or base_key not in row_lookup:
            continue
        mean, lo, hi = _bootstrap_condition_delta(
            per_image[key],
            per_image[base_key],
            sessions,
            n_bootstrap=int(args.n_bootstrap),
            rng=rng,
        )
        base_row = row_lookup[base_key]
        rows.append(
            {
                "base_summary": base_summary,
                "ssi_summary": ssi_summary,
                "combined_summary": combined_summary,
                "family": str(row["family"]),
                "scale_id": str(row["scale_id"]),
                "latent": str(row["latent"]),
                "k": int(row["k"]),
                "combined_mean_neg_mse": float(row["mean_neg_mse"]),
                "base_mean_neg_mse": float(base_row["mean_neg_mse"]),
                "mean_incremental_neg_mse": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_images": int(sessions.size),
                "base_feature_dim": int(base_row["feature_dim"]),
                "combined_feature_dim": int(row["feature_dim"]),
            }
        )
    return rows


def _contrast_rows(
    decode_rows: list[dict[str, Any]],
    per_image: dict[tuple[str, str, str, int], np.ndarray],
    sessions: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed) + 101)
    static_key_by_latent_k = {
        (str(row["latent"]), int(row["k"])): ("static", "static", str(row["latent"]), int(row["k"]))
        for row in decode_rows
        if row["family"] == "static"
    }
    contrast_text = getattr(args, "contrast_pairs", "empirical:ou,empirical:brownian,empirical:rotated,ou:brownian")
    contrasts = _parse_contrast_pairs(contrast_text)
    row_lookup = {
        (str(row["family"]), str(row["scale_id"]), str(row["latent"]), int(row["k"])): row
        for row in decode_rows
    }

    def condition_key(family: str, scale_id: str, latent: str, k: int) -> tuple[str, str, str, int] | None:
        if family == "static":
            return static_key_by_latent_k.get((latent, k))
        return (family, scale_id, latent, k)

    for row in decode_rows:
        scale_id = str(row["scale_id"])
        latent = str(row["latent"])
        k = int(row["k"])
        if str(row["family"]) == "static":
            continue
        for lhs_family, rhs_family in contrasts:
            if str(row["family"]) != lhs_family:
                continue
            lhs_key = condition_key(lhs_family, scale_id, latent, k)
            rhs_key = condition_key(rhs_family, scale_id, latent, k)
            if lhs_key is None or rhs_key is None or lhs_key not in per_image or rhs_key not in per_image:
                continue
            mean, lo, hi = _bootstrap_condition_delta(
                per_image[lhs_key],
                per_image[rhs_key],
                sessions,
                n_bootstrap=int(args.n_bootstrap),
                rng=rng,
            )
            rows.append(
                {
                    "lhs_family": lhs_family,
                    "rhs_family": rhs_family,
                    "scale_id": scale_id,
                    "latent": latent,
                    "k": k,
                    "lhs_mean_neg_mse": float(row_lookup[lhs_key]["mean_neg_mse"]),
                    "rhs_mean_neg_mse": float(row_lookup[rhs_key]["mean_neg_mse"]),
                    "mean_delta_neg_mse": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_images": int(sessions.size),
                }
            )
        if str(row["family"]) != "empirical" or ("empirical", "static") in contrasts:
            continue
        lhs_key = ("empirical", scale_id, latent, k)
        rhs_key = static_key_by_latent_k.get((latent, k))
        if rhs_key is not None:
            if rhs_key not in per_image:
                continue
            mean, lo, hi = _bootstrap_condition_delta(
                per_image[lhs_key],
                per_image[rhs_key],
                sessions,
                n_bootstrap=int(args.n_bootstrap),
                rng=rng,
            )
            rows.append(
                {
                    "lhs_family": "empirical",
                    "rhs_family": "static",
                    "scale_id": scale_id,
                    "latent": latent,
                    "k": k,
                    "lhs_mean_neg_mse": float(row_lookup[lhs_key]["mean_neg_mse"]),
                    "rhs_mean_neg_mse": float(row_lookup[rhs_key]["mean_neg_mse"]),
                    "mean_delta_neg_mse": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_images": int(sessions.size),
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-manifest", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--trace-samples-per-condition", type=int, default=4)
    parser.add_argument("--motion-families", default="empirical,ou,brownian,rotated")
    parser.add_argument("--observed-rms-scales", default="0.25,0.5,1.0")
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--min-patch-image-margin-px", type=float, default=None)
    parser.add_argument("--latent-crop-px", type=int, default=151)
    parser.add_argument("--center-crop-px", type=int, default=41)
    parser.add_argument("--local-field-grid", type=int, default=8)
    parser.add_argument(
        "--feature-target-mode",
        choices=("static", "static_subpixel", "trace_registered"),
        default="static",
        help=(
            "Feature target extraction mode. static uses the centered BackImage crop for every condition. "
            "static_subpixel uses the same centered target with bilinear subpixel cropping. "
            "trace_registered averages targets over source-image locations sampled by each eye trace, "
            "so motion responses are decoded against pose-registered image features."
        ),
    )
    parser.add_argument(
        "--spatial-readout-mode",
        choices=("amax", "center_mean", "trace_registered_center_mean"),
        default="amax",
        help=(
            "How to pool the twin rate map before response summarization. amax is the historical "
            "spatial max over the full map. center_mean averages a fixed central map crop. "
            "trace_registered_center_mean uses eye position to sample the central source region "
            "from the shifted activation map before averaging."
        ),
    )
    parser.add_argument(
        "--spatial-readout-radius",
        type=int,
        default=1,
        help="Radius, in 51x51 rate-map bins, for center_mean and trace_registered_center_mean pooling.",
    )
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--temporal-pc-components", type=int, default=4)
    parser.add_argument(
        "--latent-names",
        default="gabor_local_field,pyramid_local_field",
        help=(
            "Comma-separated feature targets. Baselines include gabor_local_field,pyramid_local_field,dct_local_field. "
            "V1-stat targets include pyramid_energy_global,pyramid_energy_grid4,pyramid_energy_grid8,"
            "pyramid_energy_orientcov_grid4_typebalanced,pyramid_energy_orientcov_grid8_typebalanced,"
            "gabor_local_field_grid4,gabor_energy_global,gabor_energy_grid4,gabor_energy_grid8,"
            "gabor_energy_orientcov_grid4_typebalanced,gabor_energy_orientcov_grid8_typebalanced."
        ),
    )
    parser.add_argument("--pca-k-list", default="4,8")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--fixed-ridge-alpha", type=float, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--decode-group-mode",
        choices=("image", "source_trial", "session"),
        default="image",
        help=(
            "CV grouping for feature decoding. image keeps each image/source row in one fold; "
            "source_trial groups all windows from the same session/trial; "
            "session is stricter across sessions. Response arrays are already image-averaged."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--contrast-pairs",
        default="empirical:ou,empirical:brownian,empirical:rotated,ou:brownian,empirical:static",
        help="Comma-separated lhs:rhs family contrasts for decode_contrasts.csv.",
    )
    parser.add_argument("--reliable-image-coherence-min", type=float, default=0.20)
    parser.add_argument("--reliable-drift-anisotropy-min", type=float, default=0.20)
    parser.add_argument("--min-duration-s", type=float, default=0.10)
    parser.add_argument("--max-rms-deg", type=float, default=0.12)
    parser.add_argument(
        "--max-trace-source-rms-deg",
        type=float,
        default=None,
        help=(
            "Optional common-unclipped trace-pool restriction. For a max scale S and cap C, "
            "set this to C/S so every sampled empirical source trace remains unclipped at S."
        ),
    )
    parser.add_argument("--max-trace-source-radius-deg", type=float, default=None)
    parser.add_argument(
        "--max-trace-source-path-length-deg",
        type=float,
        default=None,
        help=(
            "Deprecated alias for --max-rendered-trace-path-length-deg. This filters the "
            "resampled n-timepoint trace path length, not the source-table path_length_deg."
        ),
    )
    parser.add_argument(
        "--max-rendered-trace-path-length-deg",
        type=float,
        default=None,
        help="Filter trace-bank entries by path length after resampling to --n-timepoints.",
    )
    parser.add_argument(
        "--max-source-trace-path-length-deg",
        type=float,
        default=None,
        help="Filter trace-bank entries by the source table path_length_deg column.",
    )
    parser.add_argument("--max-trace-source-speed-p95-deg-s", type=float, default=None)
    parser.add_argument(
        "--max-trace-source-microsaccade-events",
        type=int,
        default=None,
        help="Optional Jake-detector event-count filter for the trace source bank. Use 0 for drift-only.",
    )
    parser.add_argument(
        "--microsaccade-speed-threshold-dps",
        type=float,
        default=None,
        help="Fixed microsaccade speed threshold. Defaults to Jake/MAD threshold per trace.",
    )
    parser.add_argument("--microsaccade-threshold-z", type=float, default=6.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    parser.add_argument(
        "--reuse-trace-sources-across-scales",
        action="store_true",
        help="Sample each family/sample trace source once per image and reuse it across all requested scales.",
    )
    parser.add_argument("--twin-batch-size", type=int, default=48)
    parser.add_argument("--twin-trace-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Prepare images/traces/latents but skip twin evaluation.")
    parser.add_argument(
        "--save-response-sample-arrays",
        action="store_true",
        help="Also save per-image x per-trace response summary arrays for hidden-trajectory posthocs.",
    )
    parser.add_argument(
        "--compute-ssi-features",
        action="store_true",
        help=(
            "Compute spatial-SSI-derived response summaries from the full readout maps before spatial max-pooling, "
            "then decode them with the same grouped ridge endpoint."
        ),
    )
    parser.add_argument(
        "--ssi-summary-names",
        default="ssi_itn,ssi_unit_mean,ssi_itn_plus_rbar",
        help=(
            "Comma-separated SSI summaries to decode when --compute-ssi-features is set. "
            "Available: ssi_itn, ssi_unit_mean, ssi_unit_spike_weighted_mean, "
            "ssi_rbar_itn, ssi_itn_plus_rbar, ssi_population_time, and their "
            "static-referenced delta_ssi_* variants."
        ),
    )
    parser.add_argument(
        "--ssi-incremental-base-summaries",
        default="mean,delta_mean",
        help=(
            "Comma-separated ordinary response summaries to concatenate with each SSI summary "
            "for incremental decode tests. Delta bases use static-referenced SSI deltas, "
            "e.g. delta_mean_plus_delta_ssi_itn."
        ),
    )
    return parser


def _trace_filter_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_trace_source_rms_deg": float(args.max_trace_source_rms_deg) if args.max_trace_source_rms_deg is not None else None,
        "max_trace_source_radius_deg": float(args.max_trace_source_radius_deg) if args.max_trace_source_radius_deg is not None else None,
        "max_trace_source_path_length_deg": (
            float(args.max_trace_source_path_length_deg) if args.max_trace_source_path_length_deg is not None else None
        ),
        "max_rendered_trace_path_length_deg": (
            float(args.max_rendered_trace_path_length_deg) if args.max_rendered_trace_path_length_deg is not None else None
        ),
        "max_source_trace_path_length_deg": (
            float(args.max_source_trace_path_length_deg) if args.max_source_trace_path_length_deg is not None else None
        ),
        "max_trace_source_speed_p95_deg_s": (
            float(args.max_trace_source_speed_p95_deg_s) if args.max_trace_source_speed_p95_deg_s is not None else None
        ),
        "max_trace_source_microsaccade_events": (
            int(args.max_trace_source_microsaccade_events) if args.max_trace_source_microsaccade_events is not None else None
        ),
    }


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    work = _prepare_windows(args)
    if work.empty:
        raise ValueError("No BackImage windows survived the aggregate filters.")
    families = _parse_str_list(args.motion_families)
    valid_families = {"empirical", "ou", "brownian", "rotated"}
    invalid = sorted(set(families).difference(valid_families))
    if invalid:
        raise ValueError(f"Unknown --motion-families entries: {invalid}")
    if bool(args.compute_ssi_features) and str(args.spatial_readout_mode) != "amax":
        raise ValueError("--compute-ssi-features currently requires --spatial-readout-mode amax")
    scales = _parse_float_list(args.observed_rms_scales)
    latent_filter = set(_parse_str_list(args.latent_names))
    cfg = AggregateConfig(
        input=str(args.input),
        out_dir=str(out_dir),
        window_manifest=str(args.window_manifest) if args.window_manifest is not None else None,
        max_images=int(args.max_images),
        trace_samples_per_condition=int(args.trace_samples_per_condition),
        motion_families=families,
        observed_rms_scales=scales,
        patch_size_px=int(args.patch_size_px),
        latent_crop_px=int(args.latent_crop_px),
        center_crop_px=int(args.center_crop_px),
        local_field_grid=int(args.local_field_grid),
        feature_target_mode=str(args.feature_target_mode),
        spatial_readout_mode=str(args.spatial_readout_mode),
        spatial_readout_radius=int(args.spatial_readout_radius),
        n_timepoints=int(args.n_timepoints),
        temporal_pc_components=int(args.temporal_pc_components),
        pca_k_list=_parse_int_list(args.pca_k_list),
        latent_names=sorted(latent_filter),
        ridge_alphas=_parse_float_list(args.ridge_alphas),
        fixed_ridge_alpha=float(args.fixed_ridge_alpha) if args.fixed_ridge_alpha is not None else None,
        outer_folds=int(args.outer_folds),
        inner_folds=int(args.inner_folds),
        decode_group_mode=str(args.decode_group_mode),
        reliable_image_coherence_min=float(args.reliable_image_coherence_min),
        reliable_drift_anisotropy_min=float(args.reliable_drift_anisotropy_min),
        min_duration_s=float(args.min_duration_s),
        min_patch_image_margin_px=(
            float(args.min_patch_image_margin_px) if args.min_patch_image_margin_px is not None else float(args.patch_size_px) / 2.0
        ),
        max_rms_deg=float(args.max_rms_deg),
        max_trace_source_rms_deg=float(args.max_trace_source_rms_deg) if args.max_trace_source_rms_deg is not None else None,
        max_trace_source_radius_deg=float(args.max_trace_source_radius_deg) if args.max_trace_source_radius_deg is not None else None,
        max_trace_source_path_length_deg=(
            float(args.max_trace_source_path_length_deg) if args.max_trace_source_path_length_deg is not None else None
        ),
        max_rendered_trace_path_length_deg=(
            float(args.max_rendered_trace_path_length_deg) if args.max_rendered_trace_path_length_deg is not None else None
        ),
        max_source_trace_path_length_deg=(
            float(args.max_source_trace_path_length_deg) if args.max_source_trace_path_length_deg is not None else None
        ),
        max_trace_source_speed_p95_deg_s=(
            float(args.max_trace_source_speed_p95_deg_s) if args.max_trace_source_speed_p95_deg_s is not None else None
        ),
        max_trace_source_microsaccade_events=(
            int(args.max_trace_source_microsaccade_events) if args.max_trace_source_microsaccade_events is not None else None
        ),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps) if args.microsaccade_speed_threshold_dps is not None else None
        ),
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
        reuse_trace_sources_across_scales=bool(args.reuse_trace_sources_across_scales),
        twin_batch_size=int(args.twin_batch_size),
        twin_trace_batch_size=int(args.twin_trace_batch_size),
        device=str(args.device),
        progress_every=int(args.progress_every),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        save_response_sample_arrays=bool(args.save_response_sample_arrays),
        compute_ssi_features=bool(args.compute_ssi_features),
        ssi_summary_names=_parse_str_list(args.ssi_summary_names) if bool(args.compute_ssi_features) else [],
        ssi_incremental_base_summaries=(
            _parse_str_list(args.ssi_incremental_base_summaries) if bool(args.compute_ssi_features) else []
        ),
    )
    _write_json(out_dir / "run_metadata.json", {"config": asdict(cfg), "steerable_pyramid": HAVE_STEERABLE_PYRAMID})
    _progress(
        f"prepared {work.shape[0]} images; families={families}; scales={scales}; "
        f"K={args.trace_samples_per_condition}; dry_run={args.dry_run}; output={out_dir}"
    )

    eyepos_by_session = _session_dataset_cache(work["session"].astype(str).to_list())
    trace_bank = _build_trace_bank(
        work,
        eyepos_by_session,
        int(args.n_timepoints),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps) if args.microsaccade_speed_threshold_dps is not None else None
        ),
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
    )
    trace_rows = [
        {
            "bank_index": j,
            "source_row": int(item["source_row"]),
            "session": str(item["session"]),
            "trial_idx": int(item["trial_idx"]),
            "global_start": int(item["global_start"]),
            "global_stop": int(item["global_stop"]),
            "observed_rms_deg": float(item["observed_rms_deg"]),
            "source_trace_observed_rms_deg": float(item["source_trace_observed_rms_deg"]),
            "source_rms_radius_deg": float(item["source_rms_radius_deg"]),
            "source_max_radius_deg": float(item["source_max_radius_deg"]),
            "path_length_deg": float(item["path_length_deg"]),
            "source_path_length_deg": float(item["source_path_length_deg"]),
            "source_speed_p95_deg_s": float(item["source_speed_p95_deg_s"]),
            "duration_s": float(item["duration_s"]),
            "lag1_autocorr": float(item["lag1_autocorr"]),
            "source_anisotropy": float(item["source_anisotropy"]),
            "trace_cov_anisotropy": float(item["trace_cov_anisotropy"]),
            "source_trace_cov_anisotropy": float(item["source_trace_cov_anisotropy"]),
            "microsaccade_threshold_dps": float(item["microsaccade_threshold_dps"]),
            "n_microsaccade_events": int(item["n_microsaccade_events"]),
            "fraction_microsaccade_samples": float(item["fraction_microsaccade_samples"]),
            "peak_microsaccade_speed_dps": float(item["peak_microsaccade_speed_dps"]),
            "rendered_microsaccade_threshold_dps": float(item["rendered_microsaccade_threshold_dps"]),
            "rendered_n_microsaccade_events": int(item["rendered_n_microsaccade_events"]),
            "rendered_fraction_microsaccade_samples": float(item["rendered_fraction_microsaccade_samples"]),
            "rendered_peak_microsaccade_speed_dps": float(item["rendered_peak_microsaccade_speed_dps"]),
        }
        for j, item in enumerate(trace_bank)
    ]
    _write_csv(out_dir / "trace_bank_metadata.csv", trace_rows)
    trace_pool = _eligible_trace_bank_indices(
        trace_bank,
        current_source_row=-1,
        **_trace_filter_kwargs(args),
    )
    if any(
        value is not None
        for value in (
            args.max_trace_source_rms_deg,
            args.max_trace_source_radius_deg,
            args.max_trace_source_path_length_deg,
            args.max_rendered_trace_path_length_deg,
            args.max_source_trace_path_length_deg,
            args.max_trace_source_speed_p95_deg_s,
            args.max_trace_source_microsaccade_events,
        )
    ):
        _progress(
            "trace source filter keeps "
            f"{len(trace_pool)}/{len(trace_bank)} traces "
            f"(rms<={args.max_trace_source_rms_deg}, radius<={args.max_trace_source_radius_deg}, "
            f"rendered_path<={args.max_rendered_trace_path_length_deg or args.max_trace_source_path_length_deg}, "
            f"source_path<={args.max_source_trace_path_length_deg}, speed_p95<={args.max_trace_source_speed_p95_deg_s}, "
            f"events<={args.max_trace_source_microsaccade_events})"
        )

    if args.dry_run:
        scorer = None
    elif bool(args.compute_ssi_features):
        scorer = AggregateSsiTwinScorer(device=str(args.device), batch_size=int(args.twin_batch_size))
    else:
        scorer = CanonicalTwinScorer(device=str(args.device), batch_size=int(args.twin_batch_size))
    image_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    latent_values: dict[str, list[np.ndarray]] = {}
    target_latents_by_response: dict[int, dict[str, np.ndarray]] = {}
    raw_responses: list[np.ndarray] = []
    raw_ssi_features: list[dict[str, np.ndarray] | None] = []
    records: list[dict[str, Any]] = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}

    for image_index, row in tqdm(work.iterrows(), total=work.shape[0], desc="aggregate FEM responses"):
        canvas_key = (str(row["session"]), int(row["trial_idx"]))
        if canvas_key not in canvas_cache:
            canvas_cache[canvas_key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
        canvas, ppd, screen_shape = canvas_cache[canvas_key]
        center_px = gaze_deg_to_screen_px(
            np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
            ppd=ppd,
            screen_shape=screen_shape,
        )
        patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(args.patch_size_px))
        if str(args.feature_target_mode) in {"static_subpixel", "trace_registered"}:
            latents = _trace_registered_latents(
                canvas,
                center_px,
                _static_trace(int(args.n_timepoints)),
                ppd=float(ppd),
                patch_size_px=int(args.patch_size_px),
                latent_crop_px=int(args.latent_crop_px),
                center_crop_px=int(args.center_crop_px),
                local_field_grid=int(args.local_field_grid),
                requested=latent_filter,
            )
        else:
            latents = _extract_requested_latents(
                patch,
                latent_crop_px=int(args.latent_crop_px),
                center_crop_px=int(args.center_crop_px),
                local_field_grid=int(args.local_field_grid),
                requested=latent_filter,
            )
        if not latents:
            raise ValueError(f"No requested latent features were available for image {image_index}.")
        for name, value in latents.items():
            latent_values.setdefault(name, []).append(value)
        image_rows.append(
            {
                "image_index": int(image_index),
                "source_row": int(row["source_row"]),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "phase": str(row.get("phase", "")),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
                "drift_anisotropy": float(row["anisotropy"]),
                "source_observed_rms_radius_deg": float(row.get("rms_radius_deg", np.nan)),
            }
        )
        if args.dry_run:
            continue
        traces = [_static_trace(int(args.n_timepoints))]
        trace_specs: list[dict[str, Any]] = [
            {
                "family": "static",
                "scale_id": "static",
                "scale": 0.0,
                "sample_index": 0,
                "trace_bank_index": -1,
                "source_trace_rms_deg": 0.0,
                "source_trace_lag1": np.nan,
                "requested_rms_deg": 0.0,
                "effective_rms_deg": 0.0,
                "rms_clipped_high": False,
            }
        ]
        reusable_sources: dict[tuple[str, int], int] = {}
        reusable_raw_traces: dict[tuple[str, int], np.ndarray] = {}
        if bool(args.reuse_trace_sources_across_scales):
            eligible = _eligible_trace_bank_indices(
                trace_bank,
                current_source_row=int(row["source_row"]),
                **_trace_filter_kwargs(args),
            )
            if not eligible:
                raise ValueError("Unpaired sampling has no eligible trace-bank entries after source-RMS filtering.")
            for family in families:
                for sample_index in range(int(args.trace_samples_per_condition)):
                    key = (family, sample_index)
                    bank_index = int(eligible[int(rng.integers(0, len(eligible)))])
                    item = trace_bank[bank_index]
                    reusable_sources[key] = bank_index
                    reusable_raw_traces[key] = _family_raw_trace(
                        family,
                        item["trace"],
                        float(item["lag1_autocorr"]),
                        rng=rng,
                        max_rms_deg=float(args.max_rms_deg),
                        source_shape=item.get("covariance_shape"),
                        selection_rms=float(item["observed_rms_deg"]),
                        target_path_length=float(item["path_length_deg"]),
                    )
        for scale in scales:
            scale_id = f"rel_{_scale_token(scale)}x"
            for family in families:
                for sample_index in range(int(args.trace_samples_per_condition)):
                    eligible = _eligible_trace_bank_indices(
                        trace_bank,
                        current_source_row=int(row["source_row"]),
                        **_trace_filter_kwargs(args),
                    )
                    if not eligible:
                        raise ValueError("Unpaired sampling has no eligible trace-bank entries after source-RMS filtering.")
                    if bool(args.reuse_trace_sources_across_scales):
                        reuse_key = (family, int(sample_index))
                        bank_index = reusable_sources[reuse_key]
                    else:
                        bank_index = int(eligible[int(rng.integers(0, len(eligible)))])
                    item = trace_bank[bank_index]
                    target_rms = float(scale) * float(item["observed_rms_deg"])
                    target_path = float(scale) * float(item["path_length_deg"])
                    if bool(args.reuse_trace_sources_across_scales):
                        trace, meta = _scale_family_raw_trace(
                            reusable_raw_traces[reuse_key],
                            target_rms,
                            max_rms_deg=float(args.max_rms_deg),
                        )
                    else:
                        trace, meta = _family_trace(
                            family,
                            item["trace"],
                            float(item["lag1_autocorr"]),
                            target_rms,
                            rng=rng,
                            max_rms_deg=float(args.max_rms_deg),
                            source_shape=item.get("covariance_shape"),
                            target_path_length=target_path,
                        )
                    traces.append(trace)
                    trace_specs.append(
                        {
                            "family": family,
                            "pairing_mode": "unpaired_ensemble",
                            "scale_id": scale_id,
                            "scale": float(scale),
                            "sample_index": int(sample_index),
                            "trace_bank_index": bank_index,
                            "trace_source_row": int(item["source_row"]),
                            "trace_source_session": str(item["session"]),
                            "source_trace_rms_deg": float(item["observed_rms_deg"]),
                            "source_trace_path_length_deg": float(item["path_length_deg"]),
                            "rendered_source_trace_path_length_deg": float(item["path_length_deg"]),
                            "source_table_path_length_deg": float(item["source_path_length_deg"]),
                            "source_trace_duration_s": float(item["duration_s"]),
                            "source_trace_lag1": float(item["lag1_autocorr"]),
                            "raw_trace_reused_across_scales": bool(args.reuse_trace_sources_across_scales),
                            "requested_rms_deg": float(meta["requested_rms_deg"]),
                            "effective_rms_deg": float(meta["effective_rms_deg"]),
                            "effective_to_requested_rms": (
                                float(meta["effective_rms_deg"]) / float(meta["requested_rms_deg"])
                                if float(meta["requested_rms_deg"]) > 0.0
                                else np.nan
                            ),
                            "rms_clipped_high": bool(meta["rms_clipped_high"]),
                            "generated_lag1_autocorr": float(meta["generated_lag1_autocorr"]),
                            "target_path_length_deg": target_path,
                            "path_length_deg": float(meta["path_length_deg"]),
                            "path_to_target_ratio": (
                                float(meta["path_length_deg"]) / target_path
                                if target_path > 0.0
                                else np.nan
                            ),
                            "speed_mean_deg_s": float(meta["speed_mean_deg_s"]),
                            "speed_median_deg_s": float(meta["speed_median_deg_s"]),
                            "speed_p95_deg_s": float(meta["speed_p95_deg_s"]),
                        }
                    )
        if str(args.feature_target_mode) == "trace_registered":
            target_latents_for_specs = [
                _trace_registered_latents(
                    canvas,
                    center_px,
                    trace,
                    ppd=float(ppd),
                    patch_size_px=int(args.patch_size_px),
                    latent_crop_px=int(args.latent_crop_px),
                    center_crop_px=int(args.center_crop_px),
                    local_field_grid=int(args.local_field_grid),
                    requested=latent_filter,
                )
                for trace in traces
            ]
        else:
            target_latents_for_specs = [latents for _ in traces]
        if bool(args.compute_ssi_features):
            assert isinstance(scorer, AggregateSsiTwinScorer)
            response_items = scorer.responses_with_ssi(patch, traces, trace_batch_size=int(args.twin_trace_batch_size))
            aligned_items = [
                (
                    _align_response_to_trace(resp, int(args.n_timepoints)),
                    _align_ssi_to_trace(ssi, int(args.n_timepoints)),
                )
                for resp, ssi in response_items
            ]
        else:
            if str(args.spatial_readout_mode) == "amax":
                responses = scorer.responses(patch, traces, trace_batch_size=int(args.twin_trace_batch_size))
            else:
                responses = _responses_with_spatial_readout(
                    scorer,
                    patch,
                    traces,
                    trace_batch_size=int(args.twin_trace_batch_size),
                    spatial_readout_mode=str(args.spatial_readout_mode),
                    spatial_readout_radius=int(args.spatial_readout_radius),
                )
            aligned_items = [(_align_response_to_trace(resp, int(args.n_timepoints)), None) for resp in responses]
        for spec, target_latents, (resp, ssi) in zip(trace_specs, target_latents_for_specs, aligned_items, strict=True):
            response_id = len(raw_responses)
            raw_responses.append(resp.astype(np.float32, copy=False))
            raw_ssi_features.append(ssi)
            target_latents_by_response[response_id] = target_latents
            records.append(
                {
                    "response_id": response_id,
                    "image_index": int(image_index),
                    "family": str(spec["family"]),
                    "scale_id": str(spec["scale_id"]),
                    "sample_index": int(spec.get("sample_index", 0)),
                }
            )
            motion_row = {
                "response_id": response_id,
                "image_index": int(image_index),
                "source_row": int(row["source_row"]),
                **spec,
                "response_frames": int(resp.shape[0]),
                "response_units": int(resp.shape[1]),
            }
            motion_rows.append(motion_row)
        done = int(image_index) + 1
        if done == 1 or done == work.shape[0] or (int(args.progress_every) > 0 and done % int(args.progress_every) == 0):
            _progress(f"images {done}/{work.shape[0]}; responses={len(raw_responses)}")

    image_df = pd.DataFrame(image_rows)
    image_df.to_csv(out_dir / "analysis_images.csv", index=False)
    _write_csv(out_dir / "aggregate_motion_metadata.csv", motion_rows)
    if motion_rows:
        motion_df = pd.DataFrame(motion_rows)
        summary_cols = [
            "family",
            "scale_id",
            "scale",
            "pairing_mode",
            "effective_rms_deg",
            "requested_rms_deg",
            "effective_to_requested_rms",
            "path_length_deg",
            "path_to_target_ratio",
            "speed_mean_deg_s",
            "speed_median_deg_s",
            "speed_p95_deg_s",
            "generated_lag1_autocorr",
            "rms_clipped_high",
        ]
        available = [col for col in summary_cols if col in motion_df.columns]
        grouped = motion_df.loc[motion_df["family"] != "static", available].groupby(["family", "scale_id"], dropna=False)
        motion_summary = grouped.agg(
            n=("effective_rms_deg", "size"),
            median_effective_rms_deg=("effective_rms_deg", "median"),
            iqr_effective_rms_deg=("effective_rms_deg", lambda x: float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))),
            median_effective_to_requested_rms=("effective_to_requested_rms", "median"),
            median_path_length_deg=("path_length_deg", "median"),
            median_path_to_target_ratio=("path_to_target_ratio", "median"),
            median_speed_mean_deg_s=("speed_mean_deg_s", "median"),
            median_generated_lag1_autocorr=("generated_lag1_autocorr", "median"),
            clipped_fraction=("rms_clipped_high", "mean"),
        ).reset_index()
        motion_summary.to_csv(out_dir / "aggregate_motion_summary.csv", index=False)
    latent_arrays = {name: np.vstack(values).astype(np.float32) for name, values in latent_values.items()}
    np.savez_compressed(out_dir / "latent_feature_arrays.npz", **latent_arrays)
    latent_arrays_by_condition: dict[tuple[str, str], dict[str, np.ndarray]] | None = None
    if str(args.feature_target_mode) == "trace_registered" and target_latents_by_response:
        latent_arrays_by_condition = _stack_condition_targets(records, target_latents_by_response)
        registered_arrays = {
            f"{name}__{family}__{scale_id}": arr
            for (family, scale_id), condition_latents in sorted(latent_arrays_by_condition.items())
            for name, arr in sorted(condition_latents.items())
        }
        np.savez_compressed(out_dir / "latent_feature_arrays_trace_registered_by_condition.npz", **registered_arrays)
    target_weight_arrays: dict[str, np.ndarray] = {}
    if PHASE_PRESERVING_SCALE_BALANCED_LATENT in latent_arrays:
        weights = _pyramid_phase_preserving_scale_balance_weights(local_grid=int(args.local_field_grid))
        if weights.shape[0] != latent_arrays[PHASE_PRESERVING_SCALE_BALANCED_LATENT].shape[1]:
            raise ValueError(
                f"{PHASE_PRESERVING_SCALE_BALANCED_LATENT} weights have length {weights.shape[0]}, "
                f"but latent dim is {latent_arrays[PHASE_PRESERVING_SCALE_BALANCED_LATENT].shape[1]}"
            )
        target_weight_arrays[PHASE_PRESERVING_SCALE_BALANCED_LATENT] = weights.astype(np.float32)
    for name in sorted(PYRAMID_ENERGY_ORIENTCOV_TYPEBALANCED_NAMES):
        if name not in latent_arrays:
            continue
        grid = int(PYRAMID_ENERGY_ORIENTCOV_GRID_BY_NAME[name])
        weights = _pyramid_energy_orientcov_type_balance_weights(local_grid=grid)
        if weights.shape[0] != latent_arrays[name].shape[1]:
            raise ValueError(f"{name} weights have length {weights.shape[0]}, but latent dim is {latent_arrays[name].shape[1]}")
        target_weight_arrays[name] = weights.astype(np.float32)
    for name in sorted(GABOR_ENERGY_ORIENTCOV_TYPEBALANCED_NAMES):
        if name not in latent_arrays:
            continue
        grid = int(GABOR_ENERGY_ORIENTCOV_GRID_BY_NAME[name])
        weights = _gabor_energy_orientcov_type_balance_weights(local_grid=grid)
        if weights.shape[0] != latent_arrays[name].shape[1]:
            raise ValueError(f"{name} weights have length {weights.shape[0]}, but latent dim is {latent_arrays[name].shape[1]}")
        target_weight_arrays[name] = weights.astype(np.float32)
    if target_weight_arrays:
        np.savez_compressed(out_dir / "latent_feature_weights.npz", **target_weight_arrays)

    if args.dry_run:
        _progress("dry run complete; skipped twin responses and summaries")
        return out_dir

    _progress("fitting temporal response basis and writing compact response summaries")
    basis = _fit_temporal_basis(raw_responses, int(args.temporal_pc_components))
    dct_basis = _fixed_dct_basis(int(args.n_timepoints), int(args.temporal_pc_components))
    static_by_image = {
        int(rec["image_index"]): raw_responses[int(rec["response_id"])]
        for rec in records
        if rec["family"] == "static"
    }
    static_ssi_by_image: dict[int, dict[str, np.ndarray]] = {}
    if bool(args.compute_ssi_features):
        for rec in records:
            if rec["family"] != "static":
                continue
            image_index = int(rec["image_index"])
            ssi = raw_ssi_features[int(rec["response_id"])]
            if ssi is None:
                raise ValueError("SSI features were requested but one static response has no SSI feature dictionary")
            static_ssi_by_image[image_index] = ssi
    response_summaries = {}
    ssi_summary_names = _parse_str_list(args.ssi_summary_names) if bool(args.compute_ssi_features) else []
    ssi_standalone_summary_names = _expand_ssi_summary_names(ssi_summary_names) if bool(args.compute_ssi_features) else []
    ssi_incremental_base_summaries = (
        _parse_str_list(args.ssi_incremental_base_summaries) if bool(args.compute_ssi_features) else []
    )
    ssi_incremental_names: list[str] = []
    for rec in records:
        response_id = int(rec["response_id"])
        image_index = int(rec["image_index"])
        summaries = _summarize_response(raw_responses[response_id], static_by_image[image_index], basis)
        _add_temporal_basis_summaries(
            summaries,
            raw_responses[response_id],
            static_by_image[image_index],
            dct_basis,
            prefix="temporal_dct",
        )
        if bool(args.compute_ssi_features):
            ssi = raw_ssi_features[response_id]
            if ssi is None:
                raise ValueError("SSI features were requested but one response has no SSI feature dictionary")
            summaries.update(
                _summarize_ssi_features(
                    ssi,
                    ssi_standalone_summary_names,
                    static_ssi=static_ssi_by_image[image_index],
                )
            )
            for name in _add_ssi_incremental_summaries(
                summaries,
                base_summaries=ssi_incremental_base_summaries,
                ssi_summary_names=ssi_summary_names,
            ):
                if name not in ssi_incremental_names:
                    ssi_incremental_names.append(name)
        response_summaries[response_id] = summaries
    base_summary_names = ["temporal_pca", "temporal_delta_pca", "temporal_dct", "temporal_dct_delta", "mean", "delta_mean"]
    summary_names = base_summary_names + ssi_standalone_summary_names + ssi_incremental_names
    summary_arrays: dict[str, np.ndarray] = {}
    for summary in summary_names:
        by_condition = _stack_condition_features(records, response_summaries, summary)
        for (family, scale_id), arr in by_condition.items():
            summary_arrays[f"{summary}__{family}__{scale_id}"] = arr
    np.savez_compressed(out_dir / "response_summary_arrays.npz", temporal_basis=basis, temporal_dct_basis=dct_basis, **summary_arrays)
    if bool(args.save_response_sample_arrays):
        sample_arrays: dict[str, np.ndarray] = {}
        for summary in summary_names:
            by_condition = _stack_condition_sample_features(records, response_summaries, summary)
            for (family, scale_id), arr in by_condition.items():
                sample_arrays[f"{summary}__{family}__{scale_id}"] = arr
        np.savez_compressed(
            out_dir / "response_sample_summary_arrays.npz",
            temporal_basis=basis,
            temporal_dct_basis=dct_basis,
            **sample_arrays,
        )

    sessions = image_df["session"].to_numpy()
    decode_groups = _decode_groups_from_images(image_df, str(args.decode_group_mode))
    all_decode_rows: list[dict[str, Any]] = []
    all_per_image: dict[tuple[str, str, str, str, int], np.ndarray] = {}
    for summary in summary_names:
        by_condition = _stack_condition_features(records, response_summaries, summary)
        if latent_arrays_by_condition is None:
            rows, per_image = _decode_rows(
                by_condition,
                latent_arrays,
                decode_groups,
                args,
                target_weight_arrays=target_weight_arrays,
            )
        else:
            rows, per_image = _decode_rows_shared_target_basis(
                by_condition,
                latent_arrays_by_condition,
                decode_groups,
                args,
                target_weight_arrays=target_weight_arrays,
            )
        for row in rows:
            row["response_summary"] = summary
            all_decode_rows.append(row)
        for key, values in per_image.items():
            all_per_image[(summary, *key)] = values
        _progress(f"decoded summary={summary}; jobs={len(rows)}")
    _write_csv(out_dir / "decode_summary.csv", all_decode_rows)
    ssi_incremental_decode_rows = _ssi_incremental_decode_rows(
        all_decode_rows,
        all_per_image,
        sessions,
        args,
        incremental_summary_names=ssi_incremental_names,
    )
    _write_csv(out_dir / "ssi_incremental_decode.csv", ssi_incremental_decode_rows)

    contrast_input_rows = []
    contrast_rows: list[dict[str, Any]] = []
    for summary in summary_names:
        rows = [row for row in all_decode_rows if row["response_summary"] == summary]
        per_image = {
            key[1:]: value
            for key, value in all_per_image.items()
            if key[0] == summary
        }
        for crow in _contrast_rows(rows, per_image, sessions, args):
            crow["response_summary"] = summary
            contrast_rows.append(crow)
        contrast_input_rows.extend(rows)
    _write_csv(out_dir / "decode_contrasts.csv", contrast_rows)

    condition_keys = sorted({("static", "static")} | {(str(rec["family"]), str(rec["scale_id"])) for rec in records if rec["family"] != "static"})
    covariance_summary_names = [name for name in summary_names if not _is_ssi_summary_name(name)]
    cov_rows = _covariance_rows(records, response_summaries, covariance_summary_names, condition_keys, overlap_dim=5)
    _write_csv(out_dir / "covariance_summary.csv", cov_rows)

    report = [
        "# BackImage Aggregate FEM Information Pilot",
        "",
        f"- Images: {work.shape[0]}",
        f"- Trace samples per family/scale/image: {args.trace_samples_per_condition}",
        f"- Families: {', '.join(families)}",
        f"- Scales: {', '.join(str(v) for v in scales)}",
        f"- Temporal basis components: {basis.shape[1]}",
        f"- Latents: {', '.join(latent_arrays)}",
        f"- Feature target mode: {args.feature_target_mode}",
        f"- Spatial readout mode: {args.spatial_readout_mode}; radius `{args.spatial_readout_radius}`",
        f"- Target basis mode: {'shared train-condition PCA' if str(args.feature_target_mode) == 'trace_registered' else 'condition-local PCA'}",
        f"- Target feature weighting: {', '.join(sorted(target_weight_arrays)) if target_weight_arrays else 'none'}",
        f"- SSI feature summaries: {', '.join(ssi_standalone_summary_names) if ssi_standalone_summary_names else 'not computed'}",
        f"- SSI incremental bases: {', '.join(ssi_incremental_base_summaries) if ssi_incremental_base_summaries else 'none'}",
        "",
        "Primary files:",
        "- `decode_summary.csv`",
        "- `decode_contrasts.csv`",
        "- `ssi_incremental_decode.csv`",
        "- `covariance_summary.csv`",
        "- `response_summary_arrays.npz`",
    ]
    if str(args.feature_target_mode) == "trace_registered":
        report.append("- `latent_feature_arrays_trace_registered_by_condition.npz`")
    if ssi_summary_names:
        report.extend(
            [
                "",
                "SSI adjudication:",
                "- SSI summaries were decoded with the same grouped ridge endpoint as the ordinary response summaries.",
                "- Static-referenced `delta_ssi_*` summaries are generated alongside absolute SSI summaries.",
                "- Incremental summaries concatenate an ordinary response summary with the matched SSI component; delta response bases use delta SSI components.",
                "- `ssi_incremental_decode.csv` reports same-condition held-out decode gain from adding SSI to the base response summary.",
                "- Covariance/subspace diagnostics are kept to ordinary response summaries to avoid treating high-dimensional SSI features as a separate covariance claim.",
            ]
        )
    (out_dir / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _progress(f"complete; wrote summaries to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
