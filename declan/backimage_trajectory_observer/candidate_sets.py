"""Candidate-set construction for BackImage image-identity observers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STRUCTURE_FEATURE_COLUMNS = [
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_edge_density",
    "image_orientation_coherence",
    "image_spectrum_anisotropy",
    "image_high_freq_power_fraction",
    "image_power_0_2_cpd_fraction",
    "image_power_2_4_cpd_fraction",
    "image_power_4_8_cpd_fraction",
    "image_power_8plus_cpd_fraction",
]


SUPPORTED_CANDIDATE_SET_MODES = (
    "self_lookup",
    "random_global",
    "same_session_region",
    "matched_structure_bins",
    "hard_negative_structure",
    "matched_static_response",
)


def _stable_row_id(row: pd.Series, fallback_index: int) -> str:
    if "source_row" in row and pd.notna(row["source_row"]):
        return f"source_row:{int(row['source_row'])}"
    return f"row:{int(fallback_index)}"


def candidate_id_list(windows: pd.DataFrame, indices: list[int]) -> list[str]:
    """Stable candidate ids for selected positional indices."""
    ids = []
    for idx in indices:
        row = windows.iloc[int(idx)]
        ids.append(_stable_row_id(row, int(idx)))
    return ids


def _feature_matrix(windows: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    available = [col for col in columns if col in windows.columns]
    if not available:
        return np.zeros((windows.shape[0], 0), dtype=np.float64), []
    x = windows[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    med = np.nanmedian(x, axis=0)
    med[~np.isfinite(med)] = 0.0
    x = np.where(np.isfinite(x), x, med[None, :])
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return (x - np.nanmean(x, axis=0, keepdims=True)) / sd[None, :], available


def _fill_with_random(
    selected: list[int],
    *,
    n_total: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> list[int]:
    seen = set(int(v) for v in selected)
    pool = [idx for idx in range(int(n_total)) if idx not in seen]
    if len(pool) < max(0, int(n_candidates) - len(selected)):
        raise ValueError("Not enough unique windows to fill candidate set")
    if len(selected) < int(n_candidates):
        extra = rng.choice(pool, size=int(n_candidates) - len(selected), replace=False)
        selected.extend(int(v) for v in extra)
    return selected[: int(n_candidates)]


def _finish_candidate_set(
    selected: list[int],
    *,
    n_total: int,
    n_candidates: int,
    rng: np.random.Generator,
    allow_random_fallback: bool,
    mode: str,
) -> tuple[list[int], int]:
    n_before = len({int(v) for v in selected})
    if n_before < int(n_candidates) and not bool(allow_random_fallback):
        raise ValueError(
            f"candidate_set_mode={mode!r} found only {n_before} unique matched candidates; "
            f"need {int(n_candidates)}. Use allow_random_fallback=True only for smoke/debug runs."
        )
    out = _fill_with_random(selected, n_total=n_total, n_candidates=n_candidates, rng=rng)
    n_random = max(0, int(n_candidates) - n_before)
    return out, n_random


def _nearest_by_feature(
    windows: pd.DataFrame,
    true_index: int,
    *,
    n_needed: int,
    pool: np.ndarray | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[list[int], np.ndarray, list[str]]:
    cols = STRUCTURE_FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    x, used = _feature_matrix(windows, cols)
    if x.shape[1] == 0:
        return [], np.full(windows.shape[0], np.nan), used
    true = int(true_index)
    candidate_pool = np.arange(windows.shape[0], dtype=int) if pool is None else np.asarray(pool, dtype=int)
    candidate_pool = candidate_pool[candidate_pool != true]
    if candidate_pool.size == 0:
        return [], np.full(windows.shape[0], np.nan), used
    d = np.linalg.norm(x - x[true][None, :], axis=1)
    order = candidate_pool[np.argsort(d[candidate_pool], kind="mergesort")]
    return [int(v) for v in order[: int(n_needed)]], d, used


def _static_response_feature_columns(windows: pd.DataFrame) -> list[str]:
    prefixes = (
        "static_response_",
        "mean_static_response_",
        "static_mean_rate_",
        "response_static_",
    )
    return [col for col in windows.columns if any(str(col).startswith(prefix) for prefix in prefixes)]


def _same_session_region_pool(windows: pd.DataFrame, true_index: int) -> np.ndarray:
    true_row = windows.iloc[int(true_index)]
    mask = np.ones(windows.shape[0], dtype=bool)
    if "session" in windows.columns:
        mask &= windows["session"].astype(str).to_numpy() == str(true_row["session"])
    if {"image_patch_center_x_px", "image_patch_center_y_px"}.issubset(windows.columns):
        xy = windows[["image_patch_center_x_px", "image_patch_center_y_px"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        true_xy = xy[int(true_index)]
        dist = np.linalg.norm(xy - true_xy[None, :], axis=1)
        finite = np.isfinite(dist)
        if np.any(mask & finite):
            keep_n = max(8, min(windows.shape[0], int(np.ceil(0.25 * windows.shape[0]))))
            threshold = np.partition(dist[mask & finite], min(keep_n - 1, int(np.sum(mask & finite)) - 1))[
                min(keep_n - 1, int(np.sum(mask & finite)) - 1)
            ]
            mask &= finite & (dist <= threshold)
    mask[int(true_index)] = False
    return np.flatnonzero(mask)


def build_candidate_set(
    windows: pd.DataFrame,
    true_index: int,
    *,
    mode: str,
    n_candidates: int,
    rng: np.random.Generator,
    allow_random_fallback: bool = False,
) -> dict[str, Any]:
    """Build one candidate set containing the true window exactly once."""
    if str(mode) not in SUPPORTED_CANDIDATE_SET_MODES:
        raise ValueError(f"Unsupported candidate_set_mode={mode!r}; expected {SUPPORTED_CANDIDATE_SET_MODES}")
    if int(n_candidates) < 1:
        raise ValueError("n_candidates must be at least 1")
    if windows.empty:
        raise ValueError("windows table is empty")
    true = int(true_index)
    if true < 0 or true >= windows.shape[0]:
        raise ValueError(f"true_index {true} outside windows table")
    if int(n_candidates) > windows.shape[0]:
        raise ValueError(f"n_candidates={n_candidates} exceeds available windows={windows.shape[0]}")

    selected = [true]
    structure_distance = np.full(windows.shape[0], np.nan, dtype=np.float64)
    used_features: list[str] = []
    n_random_fallback = 0
    if str(mode) == "self_lookup":
        selected = _fill_with_random(selected, n_total=windows.shape[0], n_candidates=int(n_candidates), rng=rng)
        n_random_fallback = max(0, int(n_candidates) - 1)
    elif str(mode) == "random_global":
        selected = _fill_with_random(selected, n_total=windows.shape[0], n_candidates=int(n_candidates), rng=rng)
        n_random_fallback = max(0, int(n_candidates) - 1)
    elif str(mode) == "same_session_region":
        pool = _same_session_region_pool(windows, true)
        near, structure_distance, used_features = _nearest_by_feature(
            windows,
            true,
            n_needed=int(n_candidates) - 1,
            pool=pool,
        )
        selected.extend(near)
        selected, n_random_fallback = _finish_candidate_set(
            selected,
            n_total=windows.shape[0],
            n_candidates=int(n_candidates),
            rng=rng,
            allow_random_fallback=bool(allow_random_fallback),
            mode=str(mode),
        )
    elif str(mode) == "matched_structure_bins":
        x, used_features = _feature_matrix(windows, STRUCTURE_FEATURE_COLUMNS)
        if x.shape[1] == 0:
            selected, n_random_fallback = _finish_candidate_set(
                selected,
                n_total=windows.shape[0],
                n_candidates=int(n_candidates),
                rng=rng,
                allow_random_fallback=bool(allow_random_fallback),
                mode=str(mode),
            )
        else:
            close_mask = np.ones(windows.shape[0], dtype=bool)
            for col in [c for c in STRUCTURE_FEATURE_COLUMNS if c in windows.columns][:4]:
                vals = pd.to_numeric(windows[col], errors="coerce").to_numpy(dtype=np.float64)
                true_val = vals[true]
                if np.isfinite(true_val) and np.isfinite(vals).sum() >= 4:
                    qs = np.nanquantile(vals, [0.25, 0.5, 0.75])
                    bin_id = int(np.searchsorted(qs, true_val, side="right"))
                    close_mask &= np.searchsorted(qs, vals, side="right") == bin_id
            pool = np.flatnonzero(close_mask)
            near, structure_distance, _used = _nearest_by_feature(
                windows,
                true,
                n_needed=int(n_candidates) - 1,
                pool=pool,
            )
            selected.extend(near)
            selected, n_random_fallback = _finish_candidate_set(
                selected,
                n_total=windows.shape[0],
                n_candidates=int(n_candidates),
                rng=rng,
                allow_random_fallback=bool(allow_random_fallback),
                mode=str(mode),
            )
    elif str(mode) == "hard_negative_structure":
        near, structure_distance, used_features = _nearest_by_feature(
            windows,
            true,
            n_needed=int(n_candidates) - 1,
        )
        selected.extend(near)
        selected, n_random_fallback = _finish_candidate_set(
            selected,
            n_total=windows.shape[0],
            n_candidates=int(n_candidates),
            rng=rng,
            allow_random_fallback=bool(allow_random_fallback),
            mode=str(mode),
        )
    elif str(mode) == "matched_static_response":
        static_cols = _static_response_feature_columns(windows)
        if not static_cols:
            raise ValueError(
                "candidate_set_mode='matched_static_response' requires static-response feature columns "
                "in the window table or a precomputed candidate manifest"
            )
        near, structure_distance, used_features = _nearest_by_feature(
            windows,
            true,
            n_needed=int(n_candidates) - 1,
            feature_columns=static_cols,
        )
        selected.extend(near)
        selected, n_random_fallback = _finish_candidate_set(
            selected,
            n_total=windows.shape[0],
            n_candidates=int(n_candidates),
            rng=rng,
            allow_random_fallback=bool(allow_random_fallback),
            mode=str(mode),
        )

    # Preserve true candidate at index 0 and remove accidental duplicates.
    deduped = []
    seen = set()
    for idx in selected:
        if int(idx) in seen:
            continue
        deduped.append(int(idx))
        seen.add(int(idx))
    if len(deduped) < int(n_candidates):
        if not bool(allow_random_fallback) and str(mode) not in {"self_lookup", "random_global"}:
            raise ValueError(f"candidate_set_mode={mode!r} lost candidates after de-duplication")
        selected = _fill_with_random(deduped, n_total=windows.shape[0], n_candidates=int(n_candidates), rng=rng)
        n_random_fallback += max(0, int(n_candidates) - len(deduped))
    else:
        selected = deduped[: int(n_candidates)]
    ids = candidate_id_list(windows, selected)
    duplicate_flag = len(set(ids)) != len(ids)
    if ids.count(ids[0]) != 1:
        raise ValueError("True candidate id does not appear exactly once")

    contrast_distance = float("nan")
    if "image_patch_rms_contrast" in windows.columns and len(selected) > 1:
        vals = pd.to_numeric(windows["image_patch_rms_contrast"], errors="coerce").to_numpy(dtype=np.float64)
        diffs = np.abs(vals[selected[1:]] - vals[true])
        contrast_distance = float(np.nanmin(diffs)) if np.isfinite(diffs).any() else float("nan")

    structure_nearest = float("nan")
    if np.isfinite(structure_distance).any() and len(selected) > 1:
        vals = structure_distance[selected[1:]]
        structure_nearest = float(np.nanmin(vals)) if np.isfinite(vals).any() else float("nan")

    return {
        "candidate_indices": selected,
        "candidate_ids": ids,
        "true_candidate_index": 0,
        "candidate_set_mode": str(mode),
        "n_candidates": int(len(selected)),
        "candidate_duplicate_flag": bool(duplicate_flag),
        "near_duplicate_flag": bool(structure_nearest <= 1e-9) if np.isfinite(structure_nearest) else False,
        "n_matched_distractors": int(max(0, len(selected) - 1 - n_random_fallback)),
        "n_random_fallback_distractors": int(n_random_fallback),
        "random_fallback_used": bool(n_random_fallback > 0),
        "contrast_distance_to_nearest_distractor": contrast_distance,
        "structure_distance_to_nearest_distractor": structure_nearest,
        "structure_feature_columns": ",".join(used_features),
    }
