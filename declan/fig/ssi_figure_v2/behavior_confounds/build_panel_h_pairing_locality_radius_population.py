#!/usr/bin/env python3
"""Population audit of the Figure-4H fixation-centered image-support curve.

The analysis holds every measured position cloud and animal-selected fixation
center fixed while changing only the radius over which the local image axis is
estimated.  At every radius it compares the real local pairing with the same
set of matched, within-session real-trajectory reassignments.  A companion
curve evaluates image patches centered 5 degrees away on the same image.

Adjacent extraction windows overlap.  Window-level values are retained for
auditability, but uncertainty is computed after collapsing to trials and uses
an equal-subject, equal-session hierarchical bootstrap.  With two animals the
interval is conditional on these animals and is not an animal-population CI.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from declan.fixation_statistics_by_stimulus.image_features import (
    _backimage_canvas,
    backimage_trial_geometry,
    gaze_deg_to_screen_px,
    image_axis_rad_to_gaze_axis_rad,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = (
    REPO_ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
)
WINDOW_FEATURES = SOURCE_ROOT / "window_features.csv"
LOCAL_SWEEP = (
    SOURCE_ROOT
    / "backimage_patch_radius_sensitivity_v1"
    / "patch_radius_alignment_sweep_windows.csv"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "outputs/fig/ssi_figure_v2/behavior_confounds_map_first_v1"
    / "panel_h_pairing_locality_radius_population_v1"
)

OFFSET_DISTANCE_DEG = 5.0
OFFSET_ANGLES_DEG = tuple(float(x) for x in range(0, 360, 45))
MIN_PATCH_FRACTION_INSIDE = 0.98
MAX_PATCH_FRACTION_BACKGROUND = 0.05
COHERENCE_SLOPE_MIN = 0.3
MATCH_FEATURES = (
    "log_rms_radius",
    "anisotropy",
    "gaze_eccentricity",
    "log_samples_since_event",
)
MATCH_BIN_SCHEMES = (
    MATCH_FEATURES,
    MATCH_FEATURES[:3],
    ("log_rms_radius", "gaze_eccentricity"),
    ("log_rms_radius",),
    (),
)
MIN_STRATUM_SIZE = 8

SUBJECT_COLORS = {"Allen": "#2c6e9b", "Logan": "#c27520"}
LOCAL_COLOR = "#176b87"
OFFSET_COLOR = "#969da5"
LOCALITY_COLOR = "#9c3d6e"
NULL_COLOR = "#8b78a5"
RMS_COLOR = "#315f72"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _integral_image(values: np.ndarray) -> np.ndarray:
    return np.pad(
        np.cumsum(np.cumsum(values, axis=0), axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )


def _rect_sum(integral: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> float:
    return float(
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )


def _offset_gaze(gaze: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.radians(float(angle_deg))
    return np.asarray(gaze, dtype=np.float64) + OFFSET_DISTANCE_DEG * np.asarray(
        [np.cos(theta), np.sin(theta)], dtype=np.float64
    )


def _load_complete_local(max_sessions: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    sweep = pd.read_csv(LOCAL_SWEEP)
    radii = np.asarray(sorted(sweep["patch_radius_deg"].unique()), dtype=float)
    complete = sweep.groupby("source_window_index")["patch_radius_deg"].nunique()
    complete_idx = complete[complete.eq(len(radii))].index
    sweep = sweep[sweep["source_window_index"].isin(complete_idx)].copy()

    base = pd.read_csv(WINDOW_FEATURES)
    base.insert(0, "source_window_index", np.arange(len(base), dtype=int))
    base = base[base["source_window_index"].isin(complete_idx)].copy()
    base["subject"] = base["session"].astype(str).str.split("_", n=1).str[0]

    if max_sessions > 0:
        sessions = sorted(base["session"].astype(str).unique())[:max_sessions]
        base = base[base["session"].astype(str).isin(sessions)].copy()
        sweep = sweep[sweep["session"].astype(str).isin(sessions)].copy()

    expected = len(base) * len(radii)
    if len(sweep) != expected:
        raise RuntimeError(f"Incomplete local support after filtering: {len(sweep)} != {expected}")
    return base.reset_index(drop=True), sweep.reset_index(drop=True), radii


def _compute_offset_features(
    base: pd.DataFrame,
    radii: np.ndarray,
    cache_path: Path,
    *,
    force: bool,
) -> pd.DataFrame:
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path)
        expected_idx = set(base["source_window_index"].astype(int))
        if set(cached["source_window_index"].astype(int)).issubset(expected_idx):
            return cached

    rows: list[dict[str, Any]] = []
    grouped = base.groupby(["session", "trial_idx"], sort=False)
    for (session, trial_idx), block in tqdm(
        grouped,
        total=len(grouped),
        desc="5-deg offset image axes",
    ):
        try:
            canvas, ppd, (height, width) = _backimage_canvas(str(session), int(trial_idx))
            geometry = backimage_trial_geometry(str(session), int(trial_idx))
        except Exception:
            continue

        arr = np.asarray(canvas, dtype=np.float64)
        gx = ndimage.sobel(arr, axis=1, mode="nearest")
        gy = ndimage.sobel(arr, axis=0, mode="nearest")
        jxx_int = _integral_image(gx * gx)
        jyy_int = _integral_image(gy * gy)
        jxy_int = _integral_image(gx * gy)

        dest_x0, dest_y0, dest_x1, dest_y1 = geometry["dest_rect"]
        yy, xx = np.indices(arr.shape)
        inside = (
            (xx >= dest_x0)
            & (xx < dest_x1)
            & (yy >= dest_y0)
            & (yy < dest_y1)
        ).astype(np.float64)
        background = np.isclose(
            arr, float(geometry["background"]), atol=1e-6
        ).astype(np.float64)
        inside_int = _integral_image(inside)
        background_int = _integral_image(background)

        for row in block.itertuples(index=False):
            local_gaze = np.asarray([row.mean_x_deg, row.mean_y_deg], dtype=np.float64)
            for angle in OFFSET_ANGLES_DEG:
                gaze = _offset_gaze(local_gaze, angle)
                cx, cy = gaze_deg_to_screen_px(
                    gaze, ppd=ppd, screen_shape=(height, width)
                )
                for radius in radii:
                    rad = max(2, int(round(float(radius) * ppd)))
                    x0 = max(0, int(round(cx)) - rad)
                    x1 = min(width, int(round(cx)) + rad + 1)
                    y0 = max(0, int(round(cy)) - rad)
                    y1 = min(height, int(round(cy)) + rad + 1)
                    area = int((x1 - x0) * (y1 - y0))
                    if area < 16:
                        continue
                    fraction_inside = _rect_sum(inside_int, x0, x1, y0, y1) / area
                    fraction_background = _rect_sum(
                        background_int, x0, x1, y0, y1
                    ) / area
                    if (
                        fraction_inside < MIN_PATCH_FRACTION_INSIDE
                        or fraction_background > MAX_PATCH_FRACTION_BACKGROUND
                    ):
                        continue
                    jxx = _rect_sum(jxx_int, x0, x1, y0, y1) / area
                    jyy = _rect_sum(jyy_int, x0, x1, y0, y1) / area
                    jxy = _rect_sum(jxy_int, x0, x1, y0, y1) / area
                    den = jxx + jyy
                    if den <= 0:
                        continue
                    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / den
                    gradient_axis = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
                    edge_axis = image_axis_rad_to_gaze_axis_rad(
                        gradient_axis + np.pi / 2.0
                    )
                    rows.append(
                        {
                            "source_window_index": int(row.source_window_index),
                            "session": str(session),
                            "trial_idx": int(trial_idx),
                            "patch_radius_deg": float(radius),
                            "offset_angle_deg": float(angle),
                            "offset_center_x_deg": float(gaze[0]),
                            "offset_center_y_deg": float(gaze[1]),
                            "image_edge_axis_deg": float(np.degrees(edge_axis)),
                            "image_orientation_coherence": float(coherence),
                            "image_patch_fraction_inside_image": float(fraction_inside),
                            "image_patch_fraction_background": float(fraction_background),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(cache_path, index=False, compression="gzip")
    return out


def _prepare_ordered_windows(base: pd.DataFrame) -> pd.DataFrame:
    ordered = base.sort_values("source_window_index").reset_index(drop=True).copy()
    ordered["window_order"] = np.arange(len(ordered), dtype=int)
    ordered["log_rms_radius"] = np.log1p(
        pd.to_numeric(ordered["rms_radius_deg"], errors="coerce").clip(lower=0)
    )
    ordered["gaze_eccentricity"] = pd.to_numeric(
        ordered["abs_mean_radius_deg"], errors="coerce"
    )
    ordered["log_samples_since_event"] = np.log1p(
        pd.to_numeric(ordered["samples_since_event"], errors="coerce").clip(lower=0)
    )
    return ordered


def _binary_rank(values: pd.Series) -> pd.Series:
    return (values.rank(method="first", pct=True) > 0.5).astype(int)


def _valid_partition(block: pd.DataFrame, keys: tuple[str, ...]) -> bool:
    if not keys:
        groups: Iterable[tuple[Any, pd.DataFrame]] = [("all", block)]
    else:
        labels = pd.DataFrame(index=block.index)
        for key in keys:
            labels[key] = _binary_rank(block[key])
        groups = block.groupby([labels[k] for k in keys], sort=False)
    for _label, group in groups:
        n = len(group)
        counts = group["trial_idx"].value_counts()
        if n < MIN_STRATUM_SIZE or len(counts) < 2 or int(counts.max()) * 2 > n:
            return False
    return True


def _assign_match_strata(ordered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = ordered.copy()
    work["match_stratum"] = ""
    manifest_rows: list[dict[str, Any]] = []
    for (session, phase), block in work.groupby(["session", "phase"], sort=False):
        chosen: tuple[str, ...] | None = None
        for keys in MATCH_BIN_SCHEMES:
            if _valid_partition(block, keys):
                chosen = keys
                break
        if chosen is None:
            raise RuntimeError(f"No feasible match partition for {session} {phase}")

        labels = pd.DataFrame(index=block.index)
        for key in chosen:
            labels[key] = _binary_rank(block[key])
        if chosen:
            signatures = labels.astype(str).agg("".join, axis=1)
        else:
            signatures = pd.Series("all", index=block.index)
        for signature, idx in signatures.groupby(signatures).groups.items():
            stratum = f"{session}|{phase}|{'+'.join(chosen) or 'unbinned'}|{signature}"
            work.loc[list(idx), "match_stratum"] = stratum
            part = work.loc[list(idx)]
            manifest_rows.append(
                {
                    "match_stratum": stratum,
                    "session": str(session),
                    "phase": str(phase),
                    "bin_features": "+".join(chosen) or "unbinned",
                    "n_windows": int(len(part)),
                    "n_trials": int(part["trial_idx"].nunique()),
                    "maximum_windows_from_one_trial": int(
                        part["trial_idx"].value_counts().max()
                    ),
                }
            )
    if work["match_stratum"].eq("").any():
        raise RuntimeError("Some windows were not assigned to a match stratum")
    return work, pd.DataFrame(manifest_rows)


def _random_valid_mapping(
    idx: np.ndarray,
    trial: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(idx)
    # Cheap cyclic attempts handle diffuse strata.  Dense strata with several
    # windows per trial are sent promptly to the exact assignment fallback;
    # long random-search loops dominate runtime without changing the null.
    for _attempt in range(8):
        order_position = rng.permutation(n)
        shifts = rng.permutation(np.arange(1, n, dtype=int))
        for shift in shifts[: min(len(shifts), 8)]:
            donor_position = np.roll(order_position, int(shift))
            mapping = np.empty(n, dtype=int)
            mapping[order_position] = idx[donor_position]
            if np.all(trial[idx] != trial[mapping]):
                return mapping

    costs = rng.random((n, n))
    forbidden = trial[idx, None] == trial[idx][None, :]
    costs[forbidden] = 1e9
    row, col = linear_sum_assignment(costs)
    if np.any(forbidden[row, col]):
        raise RuntimeError("Could not construct a different-trial donor assignment")
    return idx[col]


def _build_donor_permutations(
    ordered: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(ordered)
    donors = np.empty((n_permutations, n), dtype=np.int32)
    trial = ordered["trial_idx"].to_numpy(dtype=int)
    groups = [
        group["window_order"].to_numpy(dtype=int)
        for _key, group in ordered.groupby("match_stratum", sort=False)
    ]
    for permutation in tqdm(range(n_permutations), desc="matched trajectory reassignments"):
        for idx in groups:
            donors[permutation, idx] = _random_valid_mapping(idx, trial, rng)
    if np.any(trial[None, :] == trial[donors]):
        raise RuntimeError("A donor assignment retained a same-trial pair")
    for permutation in range(n_permutations):
        if len(np.unique(donors[permutation])) != n:
            raise RuntimeError("A donor assignment did not preserve the trajectory marginal")
    return donors


def _matching_quality(
    ordered: pd.DataFrame,
    donors: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sample = donors[: min(32, len(donors))]
    for feature in MATCH_FEATURES:
        values = ordered[feature].to_numpy(dtype=float)
        z = np.empty_like(values)
        for (_session, _phase), idx in ordered.groupby(["session", "phase"]).groups.items():
            positions = np.asarray(list(idx), dtype=int)
            current = values[positions]
            scale = np.nanstd(current, ddof=1)
            z[positions] = (current - np.nanmean(current)) / (scale if scale > 0 else 1.0)
        delta = np.abs(z[None, :] - z[sample])
        rows.append(
            {
                "feature": feature,
                "mean_absolute_standardized_difference": float(np.nanmean(delta)),
                "median_absolute_standardized_difference": float(np.nanmedian(delta)),
                "p95_absolute_standardized_difference": float(np.nanpercentile(delta, 95)),
                "n_permutations_audited": int(len(sample)),
            }
        )
    return pd.DataFrame(rows)


def _rms_delta_arcmin(
    cxx: np.ndarray,
    cxy: np.ndarray,
    cyy: np.ndarray,
    edge_deg: np.ndarray,
) -> np.ndarray:
    theta = np.radians(edge_deg)
    co = np.cos(theta)
    si = np.sin(theta)
    parallel_var = cxx * co * co + 2.0 * cxy * co * si + cyy * si * si
    normal_var = cxx * si * si - 2.0 * cxy * co * si + cyy * co * co
    return 60.0 * (
        np.sqrt(np.maximum(parallel_var, 0.0))
        - np.sqrt(np.maximum(normal_var, 0.0))
    )


def _offset_matrices(
    ordered: pd.DataFrame,
    offsets: pd.DataFrame,
    radii: np.ndarray,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], np.ndarray]:
    n_radii = len(radii)
    pair_counts = offsets.groupby(
        ["source_window_index", "offset_angle_deg"]
    )["patch_radius_deg"].nunique()
    common_pairs = pair_counts[pair_counts.eq(n_radii)].index
    key_frame = pd.DataFrame(
        common_pairs.tolist(),
        columns=["source_window_index", "offset_angle_deg"],
    )
    common = offsets.merge(
        key_frame,
        on=["source_window_index", "offset_angle_deg"],
        how="inner",
    )
    window_ids = ordered["source_window_index"].to_numpy(dtype=int)
    angle_axis = pd.Index(OFFSET_ANGLES_DEG, name="offset_angle_deg")
    axes: dict[float, np.ndarray] = {}
    coherence: dict[float, np.ndarray] = {}
    for radius in radii:
        block = common[np.isclose(common["patch_radius_deg"], radius)]
        axis_pivot = block.pivot(
            index="source_window_index",
            columns="offset_angle_deg",
            values="image_edge_axis_deg",
        ).reindex(index=window_ids, columns=angle_axis)
        coherence_pivot = block.pivot(
            index="source_window_index",
            columns="offset_angle_deg",
            values="image_orientation_coherence",
        ).reindex(index=window_ids, columns=angle_axis)
        axes[float(radius)] = axis_pivot.to_numpy(dtype=float)
        coherence[float(radius)] = coherence_pivot.to_numpy(dtype=float)
    direction_count = np.sum(np.isfinite(axes[float(radii[0])]), axis=1)
    return axes, coherence, direction_count


def _score_population(
    ordered: pd.DataFrame,
    local: pd.DataFrame,
    radii: np.ndarray,
    offset_axes: dict[float, np.ndarray],
    offset_coherence: dict[float, np.ndarray],
    common_offset_count: np.ndarray,
    donors: np.ndarray,
) -> pd.DataFrame:
    n = len(ordered)
    ordered_ids = ordered["source_window_index"].to_numpy(dtype=int)
    local_indexed = local.set_index(["patch_radius_deg", "source_window_index"])
    trajectory_axis = ordered.apply(
        lambda row: 0.5
        * np.degrees(
            np.arctan2(
                2.0 * float(row["cov_xy_deg2"]),
                float(row["cov_xx_deg2"]) - float(row["cov_yy_deg2"]),
            )
        ),
        axis=1,
    ).to_numpy(dtype=float)
    cxx = ordered["cov_xx_deg2"].to_numpy(dtype=float)
    cxy = ordered["cov_xy_deg2"].to_numpy(dtype=float)
    cyy = ordered["cov_yy_deg2"].to_numpy(dtype=float)
    donor_axis = trajectory_axis[donors]
    donor_cxx = cxx[donors]
    donor_cxy = cxy[donors]
    donor_cyy = cyy[donors]

    frames: list[pd.DataFrame] = []
    for radius in tqdm(radii, desc="radius scores"):
        block = local_indexed.loc[float(radius)].reindex(ordered_ids)
        edge = block["image_edge_axis_deg"].to_numpy(dtype=float)
        coherence = block["image_orientation_coherence"].to_numpy(dtype=float)
        observed_cos = np.cos(2.0 * np.radians(trajectory_axis - edge))
        null_cos = np.mean(
            np.cos(2.0 * np.radians(donor_axis - edge[None, :])), axis=0
        )
        observed_rms = _rms_delta_arcmin(cxx, cxy, cyy, edge)
        null_rms = np.mean(
            _rms_delta_arcmin(
                donor_cxx,
                donor_cxy,
                donor_cyy,
                edge[None, :],
            ),
            axis=0,
        )

        off_edge = offset_axes[float(radius)]
        off_coh = offset_coherence[float(radius)]
        has_offset = common_offset_count > 0
        with np.errstate(invalid="ignore"):
            observed_offset_cos = np.nanmean(
                np.cos(
                    2.0
                    * np.radians(trajectory_axis[:, None] - off_edge)
                ),
                axis=1,
            )
            observed_offset_rms = np.nanmean(
                _rms_delta_arcmin(
                    cxx[:, None], cxy[:, None], cyy[:, None], off_edge
                ),
                axis=1,
            )
            mean_offset_coherence = np.nanmean(off_coh, axis=1)

        null_offset_cos_acc = np.zeros((len(donors), n), dtype=np.float64)
        null_offset_rms_acc = np.zeros((len(donors), n), dtype=np.float64)
        valid_counts = np.zeros(n, dtype=float)
        for direction in range(off_edge.shape[1]):
            axis = off_edge[:, direction]
            valid = np.isfinite(axis)
            if not np.any(valid):
                continue
            null_offset_cos_acc[:, valid] += np.cos(
                2.0
                * np.radians(
                    donor_axis[:, valid] - axis[None, valid]
                )
            )
            null_offset_rms_acc[:, valid] += _rms_delta_arcmin(
                donor_cxx[:, valid],
                donor_cxy[:, valid],
                donor_cyy[:, valid],
                axis[None, valid],
            )
            valid_counts[valid] += 1.0
        null_offset_cos_acc[:, has_offset] /= valid_counts[None, has_offset]
        null_offset_rms_acc[:, has_offset] /= valid_counts[None, has_offset]
        null_offset_cos = np.full(n, np.nan)
        null_offset_rms = np.full(n, np.nan)
        null_offset_cos[has_offset] = np.mean(
            null_offset_cos_acc[:, has_offset], axis=0
        )
        null_offset_rms[has_offset] = np.mean(
            null_offset_rms_acc[:, has_offset], axis=0
        )

        scored = ordered[
            [
                "source_window_index",
                "subject",
                "session",
                "trial_idx",
                "phase",
                "global_start",
                "global_stop",
                "rms_radius_deg",
                "anisotropy",
                "abs_mean_radius_deg",
                "samples_since_event",
                "match_stratum",
            ]
        ].copy()
        scored["patch_radius_deg"] = float(radius)
        scored["local_image_edge_axis_deg"] = edge
        scored["local_image_orientation_coherence"] = coherence
        scored["observed_local_cos2"] = observed_cos
        scored["matched_null_local_cos2"] = null_cos
        scored["D_pair_cos2"] = observed_cos - null_cos
        scored["observed_local_rms_delta_arcmin"] = observed_rms
        scored["matched_null_local_rms_delta_arcmin"] = null_rms
        scored["D_pair_rms_delta_arcmin"] = observed_rms - null_rms
        scored["n_common_valid_offset_directions"] = common_offset_count
        scored["mean_offset_image_orientation_coherence"] = mean_offset_coherence
        scored["observed_offset_cos2"] = observed_offset_cos
        scored["matched_null_offset_cos2"] = null_offset_cos
        scored["D_offset_cos2"] = observed_offset_cos - null_offset_cos
        scored["D_locality_cos2"] = scored["D_pair_cos2"] - scored["D_offset_cos2"]
        scored["observed_offset_rms_delta_arcmin"] = observed_offset_rms
        scored["matched_null_offset_rms_delta_arcmin"] = null_offset_rms
        scored["D_offset_rms_delta_arcmin"] = observed_offset_rms - null_offset_rms
        scored["D_locality_rms_delta_arcmin"] = (
            scored["D_pair_rms_delta_arcmin"] - scored["D_offset_rms_delta_arcmin"]
        )
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


MEAN_METRICS = (
    "observed_local_cos2",
    "matched_null_local_cos2",
    "D_pair_cos2",
    "D_offset_cos2",
    "D_locality_cos2",
    "observed_local_rms_delta_arcmin",
    "matched_null_local_rms_delta_arcmin",
    "D_pair_rms_delta_arcmin",
    "D_offset_rms_delta_arcmin",
    "D_locality_rms_delta_arcmin",
    "local_image_orientation_coherence",
    "mean_offset_image_orientation_coherence",
    "n_common_valid_offset_directions",
)


def _trial_values(window_values: pd.DataFrame) -> pd.DataFrame:
    return (
        window_values.groupby(
            ["subject", "session", "trial_idx", "patch_radius_deg"],
            as_index=False,
        )[list(MEAN_METRICS)]
        .mean()
    )


def _hierarchical_mean_summary(
    trial_values: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    subjects = sorted(trial_values["subject"].astype(str).unique())
    for radius, radius_block in trial_values.groupby("patch_radius_deg"):
        for metric in MEAN_METRICS:
            subject_points: dict[str, float] = {}
            subject_draws: dict[str, np.ndarray] = {}
            for subject in subjects:
                block = radius_block[radius_block["subject"].astype(str).eq(subject)]
                session_arrays = [
                    group[metric].dropna().to_numpy(dtype=float)
                    for _session, group in block.groupby("session", sort=False)
                ]
                session_arrays = [values for values in session_arrays if len(values)]
                if not session_arrays:
                    continue
                n_sessions = len(session_arrays)
                subject_points[subject] = float(
                    np.mean([np.mean(values) for values in session_arrays])
                )
                sampled_session = rng.integers(
                    0, n_sessions, size=(n_bootstrap, n_sessions)
                )
                selected = np.empty((n_bootstrap, n_sessions), dtype=float)
                for session_index, values in enumerate(session_arrays):
                    trial_index = rng.integers(
                        0,
                        len(values),
                        size=(n_bootstrap, n_sessions, len(values)),
                    )
                    session_draws = np.mean(values[trial_index], axis=2)
                    mask = sampled_session == session_index
                    selected[mask] = session_draws[mask]
                draws = np.mean(selected, axis=1)
                subject_draws[subject] = draws
                finite = draws[np.isfinite(draws)]
                rows.append(
                    {
                        "scope": "subject",
                        "subject": subject,
                        "patch_radius_deg": float(radius),
                        "metric": metric,
                        "estimate": subject_points[subject],
                        "ci95_low": float(np.percentile(finite, 2.5)),
                        "ci95_high": float(np.percentile(finite, 97.5)),
                        "n_sessions": int(n_sessions),
                        "n_trials": int(block["trial_idx"].count()),
                    }
                )
            if not subject_points:
                continue
            common_subjects = sorted(subject_points)
            grand_draws = np.nanmean(
                np.vstack([subject_draws[s] for s in common_subjects]), axis=0
            )
            rows.append(
                {
                    "scope": "grand_equal_subject",
                    "subject": "equal_subject_mean",
                    "patch_radius_deg": float(radius),
                    "metric": metric,
                    "estimate": float(np.mean([subject_points[s] for s in common_subjects])),
                    "ci95_low": float(np.nanpercentile(grand_draws, 2.5)),
                    "ci95_high": float(np.nanpercentile(grand_draws, 97.5)),
                    "n_sessions": int(radius_block["session"].nunique()),
                    "n_trials": int(radius_block["trial_idx"].count()),
                }
            )
    return pd.DataFrame(rows)


def _slope_from_moments(moments: np.ndarray) -> np.ndarray:
    """Return slopes from [..., (E[x], E[y], E[x2], E[xy])] moments."""
    moments = np.asarray(moments, dtype=float)
    variance = moments[..., 2] - moments[..., 0] ** 2
    covariance = moments[..., 3] - moments[..., 0] * moments[..., 1]
    return np.divide(
        covariance,
        variance,
        out=np.full_like(covariance, np.nan),
        where=variance > 0,
    )


def _trial_slope_moments(block: pd.DataFrame, y: str) -> pd.DataFrame:
    work = block[
        block["local_image_orientation_coherence"].astype(float)
        > COHERENCE_SLOPE_MIN
    ].copy()
    x = work["local_image_orientation_coherence"].to_numpy(dtype=float)
    values = work[y].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(values)
    work = work.loc[ok, ["subject", "session", "trial_idx"]].copy()
    work["x"] = x[ok]
    work["y"] = values[ok]
    work["x2"] = work["x"] ** 2
    work["xy"] = work["x"] * work["y"]
    return work.groupby(
        ["subject", "session", "trial_idx"], as_index=False
    )[["x", "y", "x2", "xy"]].mean()


def _slope_summary(
    window_values: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_columns = (
        "observed_local_cos2",
        "matched_null_local_cos2",
        "D_pair_cos2",
    )
    rows: list[dict[str, Any]] = []
    subjects = sorted(window_values["subject"].astype(str).unique())
    for radius, radius_block in window_values.groupby("patch_radius_deg"):
        for y in y_columns:
            trial_moments = _trial_slope_moments(radius_block, y)
            subject_points: dict[str, float] = {}
            subject_draws: dict[str, np.ndarray] = {}
            for subject in subjects:
                block = trial_moments[
                    trial_moments["subject"].astype(str).eq(subject)
                ]
                session_arrays = [
                    group[["x", "y", "x2", "xy"]].to_numpy(dtype=float)
                    for _session, group in block.groupby("session", sort=False)
                ]
                session_arrays = [values for values in session_arrays if len(values)]
                if not session_arrays:
                    continue
                n_sessions = len(session_arrays)
                point_slopes = [
                    float(_slope_from_moments(np.mean(values, axis=0)))
                    for values in session_arrays
                ]
                subject_points[subject] = float(np.nanmean(point_slopes))
                sampled_session = rng.integers(
                    0, n_sessions, size=(n_bootstrap, n_sessions)
                )
                selected = np.empty((n_bootstrap, n_sessions), dtype=float)
                for session_index, values in enumerate(session_arrays):
                    trial_index = rng.integers(
                        0,
                        len(values),
                        size=(n_bootstrap, n_sessions, len(values)),
                    )
                    boot_moments = np.mean(values[trial_index], axis=2)
                    session_draws = _slope_from_moments(boot_moments)
                    mask = sampled_session == session_index
                    selected[mask] = session_draws[mask]
                draws = np.nanmean(selected, axis=1)
                subject_draws[subject] = draws
                finite = draws[np.isfinite(draws)]
                rows.append(
                    {
                        "scope": "subject",
                        "subject": subject,
                        "patch_radius_deg": float(radius),
                        "response": y,
                        "slope_vs_coherence": subject_points[subject],
                        "ci95_low": float(np.percentile(finite, 2.5)),
                        "ci95_high": float(np.percentile(finite, 97.5)),
                        "coherence_min": COHERENCE_SLOPE_MIN,
                    }
                )
            common_subjects = sorted(subject_points)
            if common_subjects:
                grand_draws = np.nanmean(
                    np.vstack([subject_draws[s] for s in common_subjects]), axis=0
                )
                rows.append(
                    {
                        "scope": "grand_equal_subject",
                        "subject": "equal_subject_mean",
                        "patch_radius_deg": float(radius),
                        "response": y,
                        "slope_vs_coherence": float(
                            np.mean([subject_points[s] for s in common_subjects])
                        ),
                        "ci95_low": float(np.nanpercentile(grand_draws, 2.5)),
                        "ci95_high": float(np.nanpercentile(grand_draws, 97.5)),
                        "coherence_min": COHERENCE_SLOPE_MIN,
                    }
                )
    return pd.DataFrame(rows)


def _curve(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    work = summary[
        summary["scope"].eq("grand_equal_subject")
        & summary["metric"].eq(metric)
    ].sort_values("patch_radius_deg")
    x = work["patch_radius_deg"].to_numpy(dtype=float)
    y = work["estimate"].to_numpy(dtype=float)
    lo = work["ci95_low"].to_numpy(dtype=float)
    hi = work["ci95_high"].to_numpy(dtype=float)
    ax.plot(x, y, color=color, lw=1.7, marker="o", ms=3.2, ls=linestyle, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0)


def _plot_results(
    mean_summary: pd.DataFrame,
    slope_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True)
    ax = axes[0, 0]
    _curve(ax, mean_summary, "observed_local_cos2", color=LOCAL_COLOR, label="real local")
    _curve(
        ax,
        mean_summary,
        "matched_null_local_cos2",
        color=NULL_COLOR,
        label="matched reassignment",
        linestyle="--",
    )
    ax.set_title("Raw alignment and matched real-pair null")
    ax.set_ylabel("mean axial cos2")

    ax = axes[0, 1]
    _curve(ax, mean_summary, "D_pair_cos2", color=LOCAL_COLOR, label="local Dpair")
    _curve(ax, mean_summary, "D_offset_cos2", color=OFFSET_COLOR, label="offset Dpair")
    _curve(ax, mean_summary, "D_locality_cos2", color=LOCALITY_COLOR, label="Dlocality")
    ax.axhline(0, color="0.45", lw=0.8, ls=":")
    ax.set_title("Pairing and fixation-locality curves")
    ax.set_ylabel("null-subtracted axial cos2")

    ax = axes[0, 2]
    _curve(
        ax,
        mean_summary,
        "D_pair_rms_delta_arcmin",
        color=RMS_COLOR,
        label="local Dpair",
    )
    _curve(
        ax,
        mean_summary,
        "D_offset_rms_delta_arcmin",
        color=OFFSET_COLOR,
        label="offset Dpair",
    )
    _curve(
        ax,
        mean_summary,
        "D_locality_rms_delta_arcmin",
        color=LOCALITY_COLOR,
        label="Dlocality",
    )
    ax.axhline(0, color="0.45", lw=0.8, ls=":")
    ax.set_title("Position-confinement counterpart")
    ax.set_ylabel("parallel − normal RMS advantage (arcmin)")

    ax = axes[1, 0]
    for response, color, label, ls in (
        ("observed_local_cos2", LOCAL_COLOR, "raw slope", "-"),
        ("matched_null_local_cos2", NULL_COLOR, "null slope", "--"),
        ("D_pair_cos2", LOCALITY_COLOR, "pairing-corrected slope", "-"),
    ):
        work = slope_summary[
            slope_summary["scope"].eq("grand_equal_subject")
            & slope_summary["response"].eq(response)
        ].sort_values("patch_radius_deg")
        x = work["patch_radius_deg"].to_numpy(dtype=float)
        y = work["slope_vs_coherence"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, lw=1.6, marker="o", ms=3, ls=ls, label=label)
        ax.fill_between(
            x,
            work["ci95_low"].to_numpy(dtype=float),
            work["ci95_high"].to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            lw=0,
        )
    ax.axhline(0, color="0.45", lw=0.8, ls=":")
    ax.set_title(f"Direct correction of current 4H slope (coh > {COHERENCE_SLOPE_MIN:g})")
    ax.set_ylabel("alignment/coherence slope")

    ax = axes[1, 1]
    _curve(
        ax,
        mean_summary,
        "local_image_orientation_coherence",
        color=LOCAL_COLOR,
        label="local",
    )
    _curve(
        ax,
        mean_summary,
        "mean_offset_image_orientation_coherence",
        color=OFFSET_COLOR,
        label="5° offsets",
    )
    ax.set_title("Image-axis measurement quality")
    ax.set_ylabel("mean structure-tensor coherence")

    ax = axes[1, 2]
    subject = mean_summary[
        mean_summary["scope"].eq("subject")
        & mean_summary["metric"].eq("D_locality_cos2")
    ]
    for name, block in subject.groupby("subject"):
        work = block.sort_values("patch_radius_deg")
        ax.plot(
            work["patch_radius_deg"],
            work["estimate"],
            marker="o",
            ms=3,
            lw=1.5,
            color=SUBJECT_COLORS.get(str(name), "0.4"),
            label=str(name),
        )
    ax.axhline(0, color="0.45", lw=0.8, ls=":")
    ax.set_title("Animal-specific fixation locality")
    ax.set_ylabel("Dlocality axial cos2")

    for ax in axes.ravel():
        ax.set_xlabel("fixation-centered patch radius (deg)")
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.suptitle(
        "Figure 4H audit: where fixation-centered image content stops predicting the measured position cloud\n"
        "Same windows and matched real-trajectory assignments at every radius; ribbons are hierarchical 95% bootstrap intervals conditional on Allen and Logan",
        y=0.985,
        fontsize=12,
        weight="bold",
    )
    fig.savefig(
        out_dir / "panel_h_pairing_locality_radius_population.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    fig.savefig(
        out_dir / "panel_h_pairing_locality_radius_population.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)


def _write_report(
    mean_summary: pd.DataFrame,
    slope_summary: pd.DataFrame,
    ordered: pd.DataFrame,
    common_offset_count: np.ndarray,
    out_path: Path,
) -> None:
    grand = mean_summary[mean_summary["scope"].eq("grand_equal_subject")]
    grand_slopes = slope_summary[
        slope_summary["scope"].eq("grand_equal_subject")
    ]
    corrected = grand_slopes[grand_slopes["response"].eq("D_pair_cos2")]
    peak_corrected = corrected.loc[corrected["slope_vs_coherence"].idxmax()]
    peak_radius = float(peak_corrected["patch_radius_deg"])
    peak_block = grand_slopes[np.isclose(grand_slopes["patch_radius_deg"], peak_radius)].set_index(
        "response"
    )
    raw_peak = float(peak_block.loc["observed_local_cos2", "slope_vs_coherence"])
    null_peak = float(peak_block.loc["matched_null_local_cos2", "slope_vs_coherence"])
    n_pair_supported = int(
        np.sum(
            (grand[grand["metric"].eq("D_pair_cos2")]["ci95_low"] > 0)
            | (grand[grand["metric"].eq("D_pair_cos2")]["ci95_high"] < 0)
        )
    )
    n_locality_supported = int(
        np.sum(
            (grand[grand["metric"].eq("D_locality_cos2")]["ci95_low"] > 0)
            | (grand[grand["metric"].eq("D_locality_cos2")]["ci95_high"] < 0)
        )
    )
    n_slope_supported = int(
        np.sum((corrected["ci95_low"] > 0) | (corrected["ci95_high"] < 0))
    )
    lines = [
        "# Figure 4H pairing/locality radius audit",
        "",
        "This is a production population summary of the behavior-side spatial-support question.",
        "Every radius uses the same complete-support windows and the same matched within-session real-trajectory reassignments.",
        "Uncertainty is hierarchical over session and trial with Allen and Logan held fixed and equally weighted.",
        "",
        f"- complete-support windows: {len(ordered)}",
        f"- sessions: {ordered['session'].nunique()}",
        f"- trials: {ordered[['session', 'trial_idx']].drop_duplicates().shape[0]}",
        f"- windows with at least one 5-deg offset direction valid at every radius: {int(np.sum(common_offset_count > 0))}",
        "",
        "## Result summary",
        "",
        f"- The mean local pairing advantage has a hierarchical interval excluding zero at {n_pair_supported} of {len(corrected)} radii.",
        f"- The local-minus-offset locality advantage has an interval excluding zero at {n_locality_supported} of {len(corrected)} radii.",
        f"- The pairing-corrected alignment/coherence slope has an interval excluding zero at {n_slope_supported} of {len(corrected)} radii.",
        f"- Its largest point estimate is at {peak_radius:g} deg: raw slope {raw_peak:+.4f}, matched-null slope {null_peak:+.4f}, corrected slope {float(peak_corrected['slope_vs_coherence']):+.4f} with 95% interval [{float(peak_corrected['ci95_low']):+.4f}, {float(peak_corrected['ci95_high']):+.4f}].",
        "- Thus the pre-existing 1.25-deg peak is suggestive after correction, but the corrected data do not establish a radius at which local image information stops being predictive.",
        "",
        "## Pairing and locality curves",
        "",
        "| radius (deg) | Dpair cos2 | 95% CI | Doffset cos2 | Dlocality cos2 | 95% CI |",
        "|---:|---:|---|---:|---:|---|",
    ]
    for radius in sorted(grand["patch_radius_deg"].unique()):
        block = grand[np.isclose(grand["patch_radius_deg"], radius)].set_index("metric")
        pair = block.loc["D_pair_cos2"]
        off = block.loc["D_offset_cos2"]
        loc = block.loc["D_locality_cos2"]
        lines.append(
            f"| {radius:.2f} | {pair['estimate']:+.4f} | [{pair['ci95_low']:+.4f}, {pair['ci95_high']:+.4f}] | "
            f"{off['estimate']:+.4f} | {loc['estimate']:+.4f} | [{loc['ci95_low']:+.4f}, {loc['ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Pairing-corrected version of the displayed alignment/coherence slope",
            "",
            "| radius (deg) | raw slope | matched-null slope | corrected slope | 95% CI |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    slopes = grand_slopes
    for radius in sorted(slopes["patch_radius_deg"].unique()):
        block = slopes[np.isclose(slopes["patch_radius_deg"], radius)].set_index("response")
        raw = block.loc["observed_local_cos2"]
        null = block.loc["matched_null_local_cos2"]
        corrected = block.loc["D_pair_cos2"]
        lines.append(
            f"| {radius:.2f} | {raw['slope_vs_coherence']:+.4f} | "
            f"{null['slope_vs_coherence']:+.4f} | {corrected['slope_vs_coherence']:+.4f} | "
            f"[{corrected['ci95_low']:+.4f}, {corrected['ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Radius-dependent changes in image axis or coherence are part of the intended manipulation, not automatically confounds.",
            "- Weak-coherence points remain measurement-limited and are shown explicitly.",
            "- The curve estimates behavioral image-prediction support; it is not by itself a neural integration window or causal guidance estimate.",
            "- If the corrected curve remains elevated at the largest tested radius, this analysis provides only a lower bound and must be extended beyond 3 deg.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-permutations", type=int, default=256)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4129)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--force-offset-cache", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base, local, radii = _load_complete_local(int(args.max_sessions))
    ordered = _prepare_ordered_windows(base)
    ordered, stratum_manifest = _assign_match_strata(ordered)

    cache = out_dir / "panel_h_offset_patch_radius_features.csv.gz"
    offsets = _compute_offset_features(
        ordered,
        radii,
        cache,
        force=bool(args.force_offset_cache),
    )
    offset_axes, offset_coherence, common_offset_count = _offset_matrices(
        ordered, offsets, radii
    )

    donors = _build_donor_permutations(
        ordered,
        n_permutations=int(args.n_permutations),
        seed=int(args.seed),
    )
    np.savez_compressed(
        out_dir / "panel_h_matched_trajectory_reassignments.npz",
        donors=donors,
        source_window_index=ordered["source_window_index"].to_numpy(dtype=int),
    )
    match_quality = _matching_quality(ordered, donors)
    window_values = _score_population(
        ordered,
        local,
        radii,
        offset_axes,
        offset_coherence,
        common_offset_count,
        donors,
    )
    trial_values = _trial_values(window_values)
    mean_summary = _hierarchical_mean_summary(
        trial_values,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed) + 101,
    )
    slope_summary = _slope_summary(
        window_values,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed) + 202,
    )

    ordered.to_csv(out_dir / "panel_h_complete_support_windows.csv", index=False)
    stratum_manifest.to_csv(out_dir / "panel_h_match_strata.csv", index=False)
    match_quality.to_csv(out_dir / "panel_h_matching_quality.csv", index=False)
    window_values.to_csv(out_dir / "panel_h_window_values.csv.gz", index=False, compression="gzip")
    trial_values.to_csv(out_dir / "panel_h_trial_values.csv", index=False)
    mean_summary.to_csv(out_dir / "panel_h_hierarchical_mean_curves.csv", index=False)
    slope_summary.to_csv(out_dir / "panel_h_hierarchical_slope_curves.csv", index=False)
    _plot_results(mean_summary, slope_summary, out_dir)
    _write_report(mean_summary, slope_summary, ordered, common_offset_count, out_dir / "summary_report.md")

    _write_json(
        out_dir / "run_metadata.json",
        {
            "artifact_type": "production_population_behavior_radius_audit",
            "panel": "Figure 4H (currently rendered by panel_k_patch_radius_alignment_slope.py)",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(["python", "-m", __spec__.name if __spec__ else __name__]),
            "source_local_sweep": LOCAL_SWEEP,
            "source_window_features": WINDOW_FEATURES,
            "radii_deg": radii,
            "offset_distance_deg": OFFSET_DISTANCE_DEG,
            "offset_angles_deg": OFFSET_ANGLES_DEG,
            "complete_support_contract": "same local windows and same offset directions at every radius",
            "pairing_null_contract": (
                "within session and phase; exact trajectory marginal preserved; different trial; "
                "adaptive binary strata over movement RMS, anisotropy, gaze eccentricity, and time since event; "
                "same donor assignments at every radius and local/offset location"
            ),
            "n_permutations": int(args.n_permutations),
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_contract": (
                "overlapping windows collapsed to trials; sessions and trials resampled; "
                "Allen and Logan fixed and equally weighted"
            ),
            "coherence_slope_min": COHERENCE_SLOPE_MIN,
            "n_complete_support_windows": int(len(ordered)),
            "n_sessions": int(ordered["session"].nunique()),
            "n_trials": int(ordered[["session", "trial_idx"]].drop_duplicates().shape[0]),
            "n_windows_with_common_valid_offsets": int(np.sum(common_offset_count > 0)),
            "git_revision": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "population_inference_performed": True,
            "animal_population_inference": False,
        },
    )
    print(f"Wrote Figure 4H pairing/locality radius audit to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
