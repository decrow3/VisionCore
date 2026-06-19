"""Post-hoc feature-posterior scoring for BackImage trajectory-table runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .likelihood import effective_count, entropy, rank_desc, true_margin
    from .observer import (
        feature_recovery_metrics,
        posterior_weighted_feature,
        score_image_identity_score_vectors,
    )
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
        _extract_requested_latents,
        _parse_int_list,
        _parse_str_list,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _clip_patch
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.backimage_trajectory_observer.likelihood import effective_count, entropy, rank_desc, true_margin
    from declan.backimage_trajectory_observer.observer import (
        feature_recovery_metrics,
        posterior_weighted_feature,
        score_image_identity_score_vectors,
    )
    from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas, gaze_deg_to_screen_px
    from declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information import (
        _extract_requested_latents,
        _parse_int_list,
        _parse_str_list,
    )
    from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _clip_patch


AXIS_PARALLEL = "axis_edge_parallel"
AXIS_ORTHOGONAL = "axis_edge_orthogonal"
OBSERVER_MODES = ("known", "zero", "joint", "best_single_tau", "motion_delta")
SUMMARY_UNCERTAINTY_METRICS = (
    "joint_minus_zero_feature_gain",
    "motion_delta_minus_zero_feature_gain",
)
CONTRAST_METRICS = (
    "joint",
    "joint_minus_zero_feature_gain",
    "known_minus_joint_pose_cost",
    "motion_delta",
    "motion_delta_minus_zero_feature_gain",
)


def _progress(message: str) -> None:
    print(f"[feature-posterior] {message}", flush=True)


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    try:
        if pd.isna(value):
            return bool(default)
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return bool(default)


def _finite_float_array(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def _uncertainty_stats(
    values: Any,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    confidence: float,
) -> dict[str, Any]:
    vals = _finite_float_array(values)
    n = int(vals.size)
    out: dict[str, Any] = {
        "n": n,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "permutation_p_two_sided": float("nan"),
        "bootstrap_n": int(max(0, n_bootstrap)),
        "permutation_n": int(max(0, n_permutations)),
        "fraction_positive": float("nan"),
        "fraction_negative": float("nan"),
        "fraction_zero": float("nan"),
    }
    if n <= 0:
        return out
    out["fraction_positive"] = float(np.mean(vals > 0.0))
    out["fraction_negative"] = float(np.mean(vals < 0.0))
    out["fraction_zero"] = float(np.mean(vals == 0.0))
    alpha = float(np.clip((1.0 - float(confidence)) / 2.0, 0.0, 0.5))
    if int(n_bootstrap) > 0:
        boot = np.empty(int(n_bootstrap), dtype=np.float64)
        for boot_i in range(int(n_bootstrap)):
            sample = vals[rng.integers(0, n, size=n)]
            boot[boot_i] = float(np.mean(sample))
        out["ci_low"] = float(np.quantile(boot, alpha))
        out["ci_high"] = float(np.quantile(boot, 1.0 - alpha))
    if int(n_permutations) > 0:
        obs = abs(float(np.mean(vals)))
        count = 0
        for _ in range(int(n_permutations)):
            signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=n)
            stat = abs(float(np.mean(vals * signs)))
            if stat >= obs - 1e-12:
                count += 1
        out["permutation_p_two_sided"] = float((count + 1) / (int(n_permutations) + 1))
    return out


def _add_uncertainty_fields(
    row: dict[str, Any],
    *,
    prefix: str,
    values: Any,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    confidence: float,
) -> None:
    stats = _uncertainty_stats(
        values,
        rng=rng,
        n_bootstrap=int(n_bootstrap),
        n_permutations=int(n_permutations),
        confidence=float(confidence),
    )
    row[f"{prefix}_n"] = int(stats["n"])
    row[f"{prefix}_ci_low"] = float(stats["ci_low"])
    row[f"{prefix}_ci_high"] = float(stats["ci_high"])
    row[f"{prefix}_permutation_p_two_sided"] = float(stats["permutation_p_two_sided"])
    row[f"{prefix}_bootstrap_n"] = int(stats["bootstrap_n"])
    row[f"{prefix}_permutation_n"] = int(stats["permutation_n"])
    row[f"{prefix}_fraction_positive"] = float(stats["fraction_positive"])
    row[f"{prefix}_fraction_negative"] = float(stats["fraction_negative"])
    row[f"{prefix}_fraction_zero"] = float(stats["fraction_zero"])


def _copy_oriented_uncertainty_fields(
    *,
    source: dict[str, Any],
    target: dict[str, Any],
    source_prefix: str,
    target_prefix: str,
    sign: float,
) -> None:
    for suffix in ("n", "bootstrap_n", "permutation_n", "permutation_p_two_sided"):
        key = f"{source_prefix}_{suffix}"
        if key in source:
            target[f"{target_prefix}_{suffix}"] = source[key]
    low_key = f"{source_prefix}_ci_low"
    high_key = f"{source_prefix}_ci_high"
    if low_key in source and high_key in source:
        low = _safe_float(source[low_key])
        high = _safe_float(source[high_key])
        if sign >= 0:
            target[f"{target_prefix}_ci_low"] = low
            target[f"{target_prefix}_ci_high"] = high
        else:
            target[f"{target_prefix}_ci_low"] = -high
            target[f"{target_prefix}_ci_high"] = -low
    pos_key = f"{source_prefix}_fraction_positive"
    neg_key = f"{source_prefix}_fraction_negative"
    zero_key = f"{source_prefix}_fraction_zero"
    if pos_key in source and neg_key in source:
        target[f"{target_prefix}_fraction_positive"] = source[pos_key] if sign >= 0 else source[neg_key]
        target[f"{target_prefix}_fraction_negative"] = source[neg_key] if sign >= 0 else source[pos_key]
    if zero_key in source:
        target[f"{target_prefix}_fraction_zero"] = source[zero_key]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _scalar_int(table: dict[str, np.ndarray], key: str, default: int = -1) -> int:
    if key not in table:
        return int(default)
    arr = np.asarray(table[key]).reshape(-1)
    return int(arr[0]) if arr.size else int(default)


def _candidate_ids(table: dict[str, np.ndarray], n_candidates: int) -> list[str]:
    if "candidate_ids" not in table:
        return [str(i) for i in range(int(n_candidates))]
    return [str(v) for v in np.asarray(table["candidate_ids"]).tolist()]


def _candidate_set_lookup(candidate_sets: pd.DataFrame) -> dict[tuple[int, str], pd.Series]:
    lookup: dict[tuple[int, str], pd.Series] = {}
    if candidate_sets.empty:
        return lookup
    for _, row in candidate_sets.iterrows():
        key = (int(row["trial_id"]), str(row["candidate_set_mode"]))
        if key in lookup:
            raise ValueError(f"Duplicate candidate_sets row for trial_id={key[0]} candidate_set_mode={key[1]!r}")
        lookup[key] = row
    return lookup


def _parse_semicolon_ints(value: Any) -> list[int]:
    if pd.isna(value):
        return []
    return [int(part.strip()) for part in str(value).split(";") if part.strip()]


def _parse_semicolon_strings(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _source_row_from_candidate_id(candidate_id: str) -> int | None:
    text = str(candidate_id)
    if text.startswith("source_row:"):
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _candidate_window_indices(
    *,
    manifest_row: pd.Series,
    candidate_ids: list[str],
    candidate_lookup: dict[tuple[int, str], pd.Series],
    source_row_to_pos: dict[int, int],
    n_windows: int,
) -> tuple[list[int], str]:
    def validate_indices(indices: list[int], source: str) -> tuple[list[int], str]:
        if len(indices) != len(candidate_ids):
            raise ValueError(
                f"{source} supplied {len(indices)} candidate indices, but response table has "
                f"{len(candidate_ids)} candidate ids"
            )
        bad = [idx for idx in indices if idx < 0 or idx >= int(n_windows)]
        if bad:
            preview = ", ".join(str(v) for v in bad[:5])
            raise ValueError(f"{source} has candidate indices outside selected_windows bounds: {preview}")
        for candidate_id, idx in zip(candidate_ids, indices, strict=True):
            source_row = _source_row_from_candidate_id(candidate_id)
            if source_row is not None and source_row in source_row_to_pos and int(source_row_to_pos[source_row]) != int(idx):
                raise ValueError(
                    f"{source} index mismatch for {candidate_id!r}: candidate_sets points to row {idx}, "
                    f"but selected_windows source_row lookup points to row {source_row_to_pos[source_row]}"
                )
        return [int(idx) for idx in indices], source

    key = (int(manifest_row["trial_id"]), str(manifest_row["candidate_set_mode"]))
    if key in candidate_lookup:
        candidate_row = candidate_lookup[key]
        recorded_ids = _parse_semicolon_strings(candidate_row.get("candidate_ids", ""))
        if recorded_ids and recorded_ids != candidate_ids:
            raise ValueError(
                f"candidate_sets candidate_ids do not match response table for trial_id={key[0]} "
                f"candidate_set_mode={key[1]!r}"
            )
        indices = _parse_semicolon_ints(candidate_row.get("candidate_indices", ""))
        if indices:
            return validate_indices(indices, "candidate_sets_candidate_indices")
    indices = []
    for candidate_id in candidate_ids:
        source_row = _source_row_from_candidate_id(candidate_id)
        if source_row is None or source_row not in source_row_to_pos:
            raise ValueError(f"Cannot map candidate id {candidate_id!r} to selected_windows row")
        indices.append(int(source_row_to_pos[source_row]))
    return validate_indices(indices, "candidate_ids_source_row")


def _identity_vector(values: Any, *, name: str, expected_len: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != int(expected_len):
        raise ValueError(f"{name} identity length {arr.shape[0]} does not match selected_windows rows={expected_len}")
    if name in {"source_row", "image_index"}:
        nums = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(nums).all():
            raise ValueError(f"{name} identity contains non-numeric or non-finite values")
        return nums.astype(np.int64)
    return arr.astype(str)


def _feature_identity_sources(
    *,
    args: argparse.Namespace,
    npz_data: Any,
    n_windows: int,
) -> dict[str, np.ndarray]:
    aliases = {
        "source_row": ("source_row", "selected_source_row", "window_source_row"),
        "image_index": ("image_index", "selected_image_index", "window_image_index"),
    }
    out: dict[str, np.ndarray] = {}
    files = set(getattr(npz_data, "files", []))
    for canonical, names in aliases.items():
        for name in names:
            if name in files:
                out[canonical] = _identity_vector(npz_data[name], name=canonical, expected_len=n_windows)
                break
    if args.feature_manifest is not None:
        manifest = pd.read_csv(Path(args.feature_manifest))
        if manifest.shape[0] != int(n_windows):
            raise ValueError(
                f"--feature-manifest has {manifest.shape[0]} rows, but selected_windows has {n_windows}"
            )
        for canonical, names in aliases.items():
            if canonical in out:
                continue
            for name in names:
                if name in manifest.columns:
                    out[canonical] = _identity_vector(manifest[name], name=canonical, expected_len=n_windows)
                    break
    return out


def _validate_feature_row_identity(
    *,
    args: argparse.Namespace,
    npz_data: Any,
    windows: pd.DataFrame,
    feature_path: Path,
) -> str:
    n_windows = int(windows.shape[0])
    identities = _feature_identity_sources(args=args, npz_data=npz_data, n_windows=n_windows)
    validated: list[str] = []
    for name in ("source_row", "image_index"):
        if name not in windows.columns or name not in identities:
            continue
        expected = _identity_vector(windows[name], name=name, expected_len=n_windows)
        if not np.array_equal(expected, identities[name]):
            raise ValueError(
                f"Feature NPZ row identity mismatch for {name!r}: {feature_path} is not aligned "
                "to this run's selected_windows.csv"
            )
        validated.append(name)
    if validated:
        return "validated_" + "+".join(validated)
    if bool(args.trust_feature_row_order):
        return "trusted_row_order"
    available = ", ".join(sorted(identities)) or "none"
    raise ValueError(
        f"Feature NPZ {feature_path} has no usable row identity for this run "
        f"(available identities: {available}). Add source_row/image_index to the NPZ, pass "
        "--feature-manifest with a matching column, or explicitly pass --trust-feature-row-order."
    )


def _extract_patch(row: pd.Series, canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]], patch_size_px: int) -> np.ndarray:
    key = (str(row["session"]), int(row["trial_idx"]))
    if key not in canvas_cache:
        canvas_cache[key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    canvas, ppd, screen_shape = canvas_cache[key]
    center_px = gaze_deg_to_screen_px(
        np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=ppd,
        screen_shape=screen_shape,
    )
    return _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(patch_size_px))


def _load_or_compute_latents(args: argparse.Namespace, windows: pd.DataFrame, out_dir: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], str]:
    latent_names = _parse_str_list(args.latent_names)
    if not latent_names:
        raise ValueError("--latent-names must request at least one feature family")
    identity_status = "computed_from_selected_windows"
    if args.feature_npz is not None:
        path = Path(args.feature_npz)
        with np.load(path, allow_pickle=True) as data:
            identity_status = _validate_feature_row_identity(
                args=args,
                npz_data=data,
                windows=windows,
                feature_path=path,
            )
            arrays = {name: np.asarray(data[name], dtype=np.float32) for name in latent_names if name in data.files}
            missing = sorted(set(latent_names).difference(arrays))
            if missing:
                raise ValueError(f"Feature NPZ {path} is missing requested arrays: {missing}")
        source = f"feature_npz:{path}"
    else:
        requested = set(latent_names)
        canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
        values: dict[str, list[np.ndarray]] = {name: [] for name in latent_names}
        progress_every = max(1, int(args.progress_every))
        window_items = list(windows.iterrows())
        for latent_i, (_, row) in enumerate(tqdm(window_items, desc="feature posterior latents"), start=1):
            patch = _extract_patch(row, canvas_cache, int(args.patch_size_px))
            latents = _extract_requested_latents(
                patch,
                latent_crop_px=int(args.latent_crop_px),
                center_crop_px=int(args.center_crop_px),
                local_field_grid=int(args.local_field_grid),
                requested=requested,
            )
            missing = sorted(requested.difference(latents))
            if missing:
                raise ValueError(f"Missing requested latent features for source_row={row.get('source_row', '')}: {missing}")
            for name in latent_names:
                values[name].append(np.asarray(latents[name], dtype=np.float32))
            if latent_i == 1 or latent_i == len(window_items) or latent_i % progress_every == 0:
                _progress(f"latent extraction {latent_i}/{len(window_items)} windows")
        arrays = {name: np.vstack(values[name]).astype(np.float32, copy=False) for name in latent_names}
        source = "computed_backimage_patch_latents"
    for name, arr in arrays.items():
        if arr.ndim != 2:
            raise ValueError(f"Feature array {name!r} must be 2D, got {arr.shape}")
        if arr.shape[0] != windows.shape[0]:
            raise ValueError(
                f"Feature array {name!r} has {arr.shape[0]} rows, but selected_windows has {windows.shape[0]}"
            )
    identity_payload: dict[str, np.ndarray] = {"feature_row_index": np.arange(windows.shape[0], dtype=np.int64)}
    for name in ("source_row", "image_index"):
        if name in windows.columns and name not in arrays:
            identity_payload[name] = _identity_vector(windows[name], name=name, expected_len=windows.shape[0])
    np.savez_compressed(out_dir / "feature_latent_arrays.npz", **arrays, **identity_payload)
    qc = [
        {
            "qc_type": "latent_array",
            "latent": name,
            "n_windows": int(arr.shape[0]),
            "raw_feature_dim": int(arr.shape[1]),
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(arr))),
            "feature_source": source,
            "feature_identity_status": identity_status,
        }
        for name, arr in arrays.items()
    ]
    return arrays, qc, source


def _fit_feature_spaces(
    arrays: dict[str, np.ndarray],
    k_list: list[int],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    spaces: dict[tuple[str, int], dict[str, Any]] = {}
    qc_rows: list[dict[str, Any]] = []
    for latent, arr_raw in arrays.items():
        arr = np.asarray(arr_raw, dtype=np.float64)
        nonfinite_count = int(np.count_nonzero(~np.isfinite(arr)))
        if nonfinite_count:
            med = np.nanmedian(arr, axis=0)
            med[~np.isfinite(med)] = 0.0
            arr = np.where(np.isfinite(arr), arr, med[None, :])
        mean = np.mean(arr, axis=0, keepdims=True)
        sd = np.std(arr, axis=0, keepdims=True)
        sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
        z = (arr - mean) / sd
        if z.shape[0] < 2 or z.shape[1] < 1:
            raise ValueError(f"Feature array {latent!r} is too small for PCA: {z.shape}")
        _u, s, vt = np.linalg.svd(z, full_matrices=False)
        var = s * s
        total_var = float(np.sum(var))
        for k in k_list:
            k_requested = int(k)
            if k_requested < 1:
                raise ValueError("--pca-k-list entries must be positive")
            k_eff = int(min(k_requested, vt.shape[0], z.shape[1]))
            components = vt[:k_eff]
            scores = z @ components.T
            frac = float(np.sum(var[:k_eff]) / total_var) if total_var > 0.0 else float("nan")
            spaces[(latent, k_requested)] = {
                "scores": scores.astype(np.float32, copy=False),
                "k_eff": k_eff,
                "raw_feature_dim": int(arr.shape[1]),
                "variance_fraction": frac,
            }
            qc_rows.append(
                {
                    "qc_type": "feature_space",
                    "latent": latent,
                    "requested_k": k_requested,
                    "k_eff": k_eff,
                    "n_windows": int(arr.shape[0]),
                    "raw_feature_dim": int(arr.shape[1]),
                    "standardization": "selected_windows_zscore",
                    "pca_scope": "selected_windows",
                    "variance_fraction": frac,
                    "nonfinite_count": nonfinite_count,
                }
            )
    return spaces, qc_rows


def _auto_likelihood_scales(run_dir: Path, text: str) -> list[float]:
    if str(text).strip().lower() != "auto":
        return _parse_float_list(str(text))
    trials_path = run_dir / "observer_trials.csv"
    if trials_path.exists() and trials_path.stat().st_size > 0:
        trials = pd.read_csv(trials_path, usecols=["likelihood_scale"])
        vals = sorted(pd.to_numeric(trials["likelihood_scale"], errors="coerce").dropna().unique())
        if vals:
            return [float(v) for v in vals]
    return [1.0]


def _load_observer_trial_metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "observer_trials.csv"
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    columns = pd.read_csv(path, nrows=0).columns
    wanted = [
        "response_cache_path",
        "observation_condition",
        "observation_scale",
        "prior_condition",
        "prior_scale",
        "trajectory_prior_mode",
        "zero_reference_mode",
        "bin_seconds",
    ]
    usecols = [col for col in wanted if col in columns]
    if "response_cache_path" not in usecols:
        return {}
    frame = pd.read_csv(path, usecols=usecols)
    out: dict[str, dict[str, Any]] = {}
    for _, row in frame.drop_duplicates("response_cache_path").iterrows():
        out[str(row["response_cache_path"])] = {col: row[col] for col in usecols if col != "response_cache_path"}
    return out


def _filter_manifest(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = manifest.copy()
    if args.candidate_set_modes:
        keep = set(_parse_str_list(args.candidate_set_modes))
        out = out[out["candidate_set_mode"].astype(str).isin(keep)]
    if args.priors:
        keep = set(_parse_str_list(args.priors))
        out = out[out["prior_family"].astype(str).isin(keep)]
    if args.motion_scales:
        scales = [round(v, 10) for v in _parse_float_list(args.motion_scales)]
        out = out[out["scale"].astype(float).round(10).isin(scales)]
    if int(args.max_tables) > 0:
        out = out.head(int(args.max_tables))
    return out.reset_index(drop=True)


def _response_cache_path_valid(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def _filter_response_cache_manifest(manifest: pd.DataFrame, run_dir: Path) -> tuple[pd.DataFrame, int]:
    if "response_cache_path" not in manifest.columns:
        raise ValueError("response_cache_manifest.csv is missing required column 'response_cache_path'")
    valid_mask = manifest["response_cache_path"].map(_response_cache_path_valid)
    skipped = int((~valid_mask).sum())
    out = manifest[valid_mask].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(
            "No response cache tables available after filtering response_cache_manifest.csv. "
            "This usually means the producer run was a dry-run or used --skip-response-cache."
        )
    missing = [
        str(path)
        for path in out["response_cache_path"].astype(str)
        if not (run_dir / path).exists()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"response_cache_manifest.csv references missing response cache files: {preview}{suffix}")
    return out, skipped


def _mode_row(
    *,
    base_cols: dict[str, Any],
    observer_mode: str,
    scores: np.ndarray,
    candidate_features: np.ndarray,
    z_true: np.ndarray,
    true_idx: int,
    temperature: float,
) -> dict[str, Any]:
    z_hat, posterior = posterior_weighted_feature(scores, candidate_features, temperature=float(temperature))
    metrics = feature_recovery_metrics(z_hat, z_true)
    top = int(np.nanargmax(posterior)) if posterior.size and np.isfinite(posterior).any() else -1
    row = {
        **base_cols,
        "observer_mode": str(observer_mode),
        "score_interpretation": (
            "candidate_log_likelihood_ratio_joint_minus_zero"
            if observer_mode == "motion_delta"
            else "candidate_log_score"
        ),
        "posterior_temperature": float(temperature),
        "candidate_posterior_true_mass": float(posterior[true_idx]) if 0 <= true_idx < posterior.size else float("nan"),
        "candidate_posterior_entropy": entropy(posterior),
        "candidate_posterior_N_eff": effective_count(posterior),
        "candidate_posterior_N_eff_fraction": (
            float(effective_count(posterior) / posterior.size)
            if posterior.size and np.isfinite(effective_count(posterior))
            else float("nan")
        ),
        "posterior_top_candidate_index": top,
        "posterior_top_is_true": bool(top == true_idx) if top >= 0 else False,
        "score_true_rank": rank_desc(scores, true_idx),
        "score_true_margin": true_margin(scores, true_idx),
        "score_true_value": float(scores[true_idx]) if 0 <= true_idx < len(scores) else float("nan"),
    }
    row.update(metrics)
    return row


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = [
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_condition",
        "prior_family",
        "prior_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_prior_mode",
        "likelihood_scale",
        "latent",
        "requested_k",
        "k_eff",
    ]
    out: list[dict[str, Any]] = []
    for key, grp in df.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, key, strict=True)}
        row["n_trial_mode_rows"] = int(len(grp))
        row["n_trials"] = int(grp["trial_id"].nunique())
        for mode in OBSERVER_MODES:
            sub = grp[grp["observer_mode"].eq(mode)]
            row[f"{mode}_mean_neg_mse"] = float(sub["feature_neg_mse"].mean()) if not sub.empty else float("nan")
            row[f"{mode}_median_neg_mse"] = float(sub["feature_neg_mse"].median()) if not sub.empty else float("nan")
            row[f"{mode}_mean_cosine"] = float(sub["feature_cosine"].mean()) if not sub.empty else float("nan")
            row[f"{mode}_mean_true_mass"] = float(sub["candidate_posterior_true_mass"].mean()) if not sub.empty else float("nan")
            row[f"{mode}_median_candidate_N_eff_fraction"] = (
                float(sub["candidate_posterior_N_eff_fraction"].median()) if not sub.empty else float("nan")
            )
        row["joint_minus_zero_feature_gain"] = row["joint_mean_neg_mse"] - row["zero_mean_neg_mse"]
        row["known_minus_joint_pose_cost"] = row["known_mean_neg_mse"] - row["joint_mean_neg_mse"]
        row["motion_delta_minus_zero_feature_gain"] = row["motion_delta_mean_neg_mse"] - row["zero_mean_neg_mse"]
        out.append(row)
    return out


def _wide_trial_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "prior_condition" not in df.columns and "prior_family" in df.columns:
        df["prior_condition"] = df["prior_family"]
    index_cols = [
        "trial_id",
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_prior_mode",
        "zero_reference_mode",
        "bin_seconds",
        "likelihood_scale",
        "latent",
        "requested_k",
        "k_eff",
        "prior_condition",
        "prior_family",
    ]
    duplicate_key = index_cols + ["observer_mode"]
    dupes = df.duplicated(duplicate_key, keep=False)
    if bool(dupes.any()):
        sample_cols = [col for col in duplicate_key if col in df.columns]
        sample = df.loc[dupes, sample_cols].head(5).to_dict("records")
        raise ValueError(f"Duplicate feature-posterior rows for contrast pivot key: {sample}")
    wide = df.pivot(
        index=index_cols,
        columns="observer_mode",
        values="feature_neg_mse",
    ).reset_index()
    for mode in OBSERVER_MODES:
        if mode not in wide.columns:
            wide[mode] = np.nan
    wide["joint_minus_zero_feature_gain"] = wide["joint"] - wide["zero"]
    wide["known_minus_joint_pose_cost"] = wide["known"] - wide["joint"]
    wide["motion_delta_minus_zero_feature_gain"] = wide["motion_delta"] - wide["zero"]
    return wide


def _add_summary_uncertainty(
    summary_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    confidence: float,
) -> None:
    if not summary_rows or not rows:
        return
    wide = _wide_trial_metrics(rows)
    if wide.empty:
        return
    group_cols = [
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_condition",
        "prior_family",
        "prior_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_prior_mode",
        "likelihood_scale",
        "latent",
        "requested_k",
        "k_eff",
    ]
    row_lookup: dict[tuple[Any, ...], dict[str, Any]] = {
        tuple(row.get(col) for col in group_cols): row for row in summary_rows
    }
    for key, grp in wide.groupby(group_cols, dropna=False):
        row = row_lookup.get(tuple(key))
        if row is None:
            continue
        for metric in SUMMARY_UNCERTAINTY_METRICS:
            if metric not in grp.columns:
                continue
            _add_uncertainty_fields(
                row,
                prefix=metric,
                values=grp[metric],
                rng=rng,
                n_bootstrap=int(n_bootstrap),
                n_permutations=int(n_permutations),
                confidence=float(confidence),
            )


def _pairwise_motion_contrasts(
    rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator | None = None,
    n_bootstrap: int = 0,
    n_permutations: int = 0,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    wide = _wide_trial_metrics(rows)
    if wide.empty:
        return []
    if rng is None:
        rng = np.random.default_rng(0)
    index_cols = [
        "trial_id",
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_prior_mode",
        "zero_reference_mode",
        "bin_seconds",
        "likelihood_scale",
        "latent",
        "requested_k",
        "k_eff",
    ]
    metrics = list(CONTRAST_METRICS)
    per_trial: list[dict[str, Any]] = []
    for key, grp in wide.groupby(index_cols, dropna=False):
        fams = sorted(str(v) for v in grp["prior_family"].dropna().unique())
        for i, lhs in enumerate(fams):
            for rhs in fams[i + 1 :]:
                left = grp[grp["prior_family"].astype(str).eq(lhs)].iloc[0]
                right = grp[grp["prior_family"].astype(str).eq(rhs)].iloc[0]
                row = {col: value for col, value in zip(index_cols, key, strict=True)}
                row.update({"lhs_prior_family": lhs, "rhs_prior_family": rhs})
                for metric in metrics:
                    row[f"{metric}_lhs_minus_rhs"] = _safe_float(left[metric]) - _safe_float(right[metric])
                per_trial.append(row)
    if not per_trial:
        return []
    frame = pd.DataFrame(per_trial)
    group_cols = [col for col in frame.columns if col not in {f"{metric}_lhs_minus_rhs" for metric in metrics} and col != "trial_id"]
    out: list[dict[str, Any]] = []
    for key, grp in frame.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, key, strict=True)}
        row["n_trials"] = int(grp["trial_id"].nunique())
        for metric in metrics:
            vals = pd.to_numeric(grp[f"{metric}_lhs_minus_rhs"], errors="coerce")
            row[f"mean_{metric}_lhs_minus_rhs"] = float(vals.mean())
            row[f"median_{metric}_lhs_minus_rhs"] = float(vals.median())
            _add_uncertainty_fields(
                row,
                prefix=f"mean_{metric}_lhs_minus_rhs",
                values=vals,
                rng=rng,
                n_bootstrap=int(n_bootstrap),
                n_permutations=int(n_permutations),
                confidence=float(confidence),
            )
        out.append(row)
    return out


def _axis_contrasts_from_pairwise(all_contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in all_contrasts:
        lhs = str(row.get("lhs_prior_family", ""))
        rhs = str(row.get("rhs_prior_family", ""))
        if {lhs, rhs} != {AXIS_PARALLEL, AXIS_ORTHOGONAL}:
            continue
        if not _safe_bool(row.get("axis_shared_source_catalog", False), default=False):
            continue
        oriented = dict(row)
        sign = 1.0 if lhs == AXIS_PARALLEL else -1.0
        oriented["axis_lhs"] = AXIS_PARALLEL
        oriented["axis_rhs"] = AXIS_ORTHOGONAL
        for metric in CONTRAST_METRICS:
            for stat in ("mean", "median"):
                source_key = f"{stat}_{metric}_lhs_minus_rhs"
                if source_key in row:
                    oriented[f"{stat}_{metric}_parallel_minus_orthogonal"] = sign * float(row[source_key])
            _copy_oriented_uncertainty_fields(
                source=row,
                target=oriented,
                source_prefix=f"mean_{metric}_lhs_minus_rhs",
                target_prefix=f"mean_{metric}_parallel_minus_orthogonal",
                sign=sign,
            )
        out.append(oriented)
    return out


def _axis_contrasts(
    rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator | None = None,
    n_bootstrap: int = 0,
    n_permutations: int = 0,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    all_contrasts = _pairwise_motion_contrasts(
        rows,
        rng=rng,
        n_bootstrap=int(n_bootstrap),
        n_permutations=int(n_permutations),
        confidence=float(confidence),
    )
    return _axis_contrasts_from_pairwise(all_contrasts)


def _uncertainty_report_rows(
    *,
    summary_rows: list[dict[str, Any]],
    motion_contrasts: list[dict[str, Any]],
    axis_contrasts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    summary_context = [
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_condition",
        "prior_family",
        "prior_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_prior_mode",
        "likelihood_scale",
        "latent",
        "requested_k",
        "k_eff",
        "n_trials",
    ]
    contrast_context = [
        "candidate_set_mode",
        "observation_condition",
        "observation_family",
        "observation_scale",
        "prior_scale",
        "axis_catalog_mode",
        "axis_shared_source_catalog",
        "trajectory_prior_mode",
        "likelihood_scale",
        "latent",
        "requested_k",
        "k_eff",
        "n_trials",
        "lhs_prior_family",
        "rhs_prior_family",
    ]

    def append_record(scope: str, row: dict[str, Any], metric: str, estimate_key: str, prefix: str, context_cols: list[str]) -> None:
        if estimate_key not in row or f"{prefix}_ci_low" not in row:
            return
        payload = {col: row.get(col) for col in context_cols if col in row}
        payload.update(
            {
                "contrast_scope": scope,
                "metric": metric,
                "estimate_column": estimate_key,
                "estimate": _safe_float(row.get(estimate_key)),
                "ci_low": _safe_float(row.get(f"{prefix}_ci_low")),
                "ci_high": _safe_float(row.get(f"{prefix}_ci_high")),
                "permutation_p_two_sided": _safe_float(row.get(f"{prefix}_permutation_p_two_sided")),
                "bootstrap_n": int(_safe_float(row.get(f"{prefix}_bootstrap_n"), 0.0)),
                "permutation_n": int(_safe_float(row.get(f"{prefix}_permutation_n"), 0.0)),
                "paired_n": int(_safe_float(row.get(f"{prefix}_n"), 0.0)),
                "fraction_positive": _safe_float(row.get(f"{prefix}_fraction_positive")),
                "fraction_negative": _safe_float(row.get(f"{prefix}_fraction_negative")),
                "fraction_zero": _safe_float(row.get(f"{prefix}_fraction_zero")),
            }
        )
        out.append(payload)

    for row in summary_rows:
        for metric in SUMMARY_UNCERTAINTY_METRICS:
            append_record("within_prior", row, metric, metric, metric, summary_context)
    for row in motion_contrasts:
        for metric in CONTRAST_METRICS:
            key = f"mean_{metric}_lhs_minus_rhs"
            append_record("pairwise_prior_lhs_minus_rhs", row, metric, key, key, contrast_context)
    axis_context = contrast_context + ["axis_lhs", "axis_rhs"]
    for row in axis_contrasts:
        for metric in CONTRAST_METRICS:
            key = f"mean_{metric}_parallel_minus_orthogonal"
            append_record("axis_parallel_minus_orthogonal", row, metric, key, key, axis_context)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--feature-npz", type=Path, default=None)
    parser.add_argument("--feature-manifest", type=Path, default=None)
    parser.add_argument("--latent-names", default="gabor_local_field,pyramid_local_field")
    parser.add_argument("--pca-k-list", default="4,8")
    parser.add_argument("--likelihood-scales", default="auto")
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--candidate-set-modes", default="")
    parser.add_argument("--priors", default="")
    parser.add_argument("--motion-scales", default="")
    parser.add_argument("--max-tables", type=int, default=0)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--latent-crop-px", type=int, default=151)
    parser.add_argument("--center-crop-px", type=int, default=41)
    parser.add_argument("--local-field-grid", type=int, default=8)
    parser.add_argument("--trust-feature-row-order", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--uncertainty-confidence", type=float, default=0.95)
    parser.add_argument("--uncertainty-seed", type=int, default=0)
    return parser


def analyze(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "feature_posterior_posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(run_dir / "selected_windows.csv")
    manifest = pd.read_csv(run_dir / "response_cache_manifest.csv")
    manifest = _filter_manifest(manifest, args)
    if manifest.empty:
        raise ValueError("No response cache rows remain after filtering")
    manifest, skipped_cache_rows = _filter_response_cache_manifest(manifest, run_dir)
    _progress(
        f"selected {manifest.shape[0]} response tables from {run_dir}; "
        f"skipped_no_cache={skipped_cache_rows}; windows={windows.shape[0]}"
    )
    candidate_sets_path = run_dir / "candidate_sets.csv"
    candidate_sets = pd.read_csv(candidate_sets_path) if candidate_sets_path.exists() and candidate_sets_path.stat().st_size > 0 else pd.DataFrame()
    candidate_lookup = _candidate_set_lookup(candidate_sets)
    source_row_to_pos: dict[int, int] = {}
    if "source_row" in windows.columns:
        source_row_to_pos = {int(row["source_row"]): int(pos) for pos, row in windows.iterrows()}
    k_list = _parse_int_list(args.pca_k_list)
    likelihood_scales = _auto_likelihood_scales(run_dir, str(args.likelihood_scales))
    trial_metadata = _load_observer_trial_metadata(run_dir)
    latent_arrays, latent_qc, feature_source = _load_or_compute_latents(args, windows, out_dir)
    feature_spaces, feature_qc = _fit_feature_spaces(latent_arrays, k_list)
    uncertainty_rng = np.random.default_rng(int(args.uncertainty_seed))
    _progress(
        f"feature spaces ready: latents={','.join(_parse_str_list(args.latent_names))}; "
        f"k={','.join(str(k) for k in k_list)}; likelihood_scales={','.join(str(v) for v in likelihood_scales)}"
    )

    trial_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    progress_every = max(1, int(args.progress_every))
    manifest_items = list(manifest.iterrows())
    for progress_i, (table_index, man_row) in enumerate(
        tqdm(manifest_items, desc="feature posterior tables"),
        start=1,
    ):
        table_path = run_dir / str(man_row["response_cache_path"])
        table = _load_npz(table_path)
        prior = np.asarray(table["prior_lambda_counts"], dtype=np.float64)
        known = np.asarray(table["known_lambda_counts"], dtype=np.float64)
        zero = np.asarray(table["zero_lambda_counts"], dtype=np.float64)
        y_obs = np.asarray(table["y_obs_counts"], dtype=np.float64)
        true_idx = _scalar_int(table, "true_candidate_index", 0)
        candidate_ids = _candidate_ids(table, prior.shape[0])
        candidate_indices, candidate_index_source = _candidate_window_indices(
            manifest_row=man_row,
            candidate_ids=candidate_ids,
            candidate_lookup=candidate_lookup,
            source_row_to_pos=source_row_to_pos,
            n_windows=int(windows.shape[0]),
        )
        alignment_rows.append(
            {
                "qc_type": "candidate_alignment",
                "table_index": int(table_index),
                "trial_id": int(man_row["trial_id"]),
                "response_cache_path": str(man_row["response_cache_path"]),
                "candidate_set_mode": str(man_row["candidate_set_mode"]),
                "n_candidates": int(len(candidate_ids)),
                "candidate_index_source": candidate_index_source,
                "candidate_indices": ";".join(str(v) for v in candidate_indices),
                "candidate_ids": ";".join(candidate_ids),
            }
        )
        for likelihood_scale in likelihood_scales:
            vectors = score_image_identity_score_vectors(
                y_obs_counts=y_obs,
                prior_lambda_counts=prior,
                known_lambda_counts=known,
                zero_lambda_counts=zero,
                true_candidate_index=true_idx,
                candidate_ids=candidate_ids,
                eps=float(args.eps),
                likelihood_scale=float(likelihood_scale),
            )
            score_by_mode = {
                "known": np.asarray(vectors["known_scores"], dtype=np.float64),
                "zero": np.asarray(vectors["zero_scores"], dtype=np.float64),
                "joint": np.asarray(vectors["joint_scores"], dtype=np.float64),
                "best_single_tau": np.asarray(vectors["best_single_tau_scores"], dtype=np.float64),
            }
            score_by_mode["motion_delta"] = score_by_mode["joint"] - score_by_mode["zero"]
            for (latent, k_requested), space in feature_spaces.items():
                features_all = np.asarray(space["scores"], dtype=np.float64)
                candidate_features = features_all[np.asarray(candidate_indices, dtype=int)]
                z_true = candidate_features[int(true_idx)]
                meta = trial_metadata.get(str(man_row["response_cache_path"]), {})
                base_cols = {
                    "table_index": int(table_index),
                    "trial_id": int(man_row["trial_id"]),
                    "response_cache_path": str(man_row["response_cache_path"]),
                    "candidate_set_mode": str(man_row["candidate_set_mode"]),
                    "observation_condition": str(meta.get("observation_condition", man_row.get("observation_family", ""))),
                    "observation_family": str(man_row.get("observation_family", "")),
                    "observation_scale": float(meta.get("observation_scale", man_row.get("scale", np.nan))),
                    "prior_condition": str(meta.get("prior_condition", man_row.get("prior_family", ""))),
                    "prior_family": str(man_row.get("prior_family", "")),
                    "prior_scale": float(meta.get("prior_scale", man_row.get("scale", np.nan))),
                    "axis_catalog_mode": str(man_row.get("axis_catalog_mode", "shared")),
                    "axis_shared_source_catalog": _safe_bool(man_row.get("axis_shared_source_catalog", False)),
                    "trajectory_prior_mode": str(
                        meta.get("trajectory_prior_mode", man_row.get("trajectory_prior_mode", "unknown"))
                    ),
                    "zero_reference_mode": str(meta.get("zero_reference_mode", man_row.get("zero_reference_mode", ""))),
                    "bin_seconds": float(meta.get("bin_seconds", man_row.get("bin_seconds", np.nan))),
                    "likelihood_scale": float(likelihood_scale),
                    "likelihood_family": "poisson_expected_count",
                    "eps": float(args.eps),
                    "n_candidates": int(prior.shape[0]),
                    "n_trajectories": int(prior.shape[1]),
                    "n_timebins": int(prior.shape[2]),
                    "n_units": int(prior.shape[3]),
                    "true_candidate_index": int(true_idx),
                    "true_image_id": str(candidate_ids[int(true_idx)]),
                    "latent": str(latent),
                    "requested_k": int(k_requested),
                    "k_eff": int(space["k_eff"]),
                    "raw_feature_dim": int(space["raw_feature_dim"]),
                    "feature_variance_fraction": float(space["variance_fraction"]),
                    "feature_space": "selected_windows_zscore_pca",
                    "feature_source": feature_source,
                }
                for mode, scores in score_by_mode.items():
                    trial_rows.append(
                        _mode_row(
                            base_cols=base_cols,
                            observer_mode=mode,
                            scores=scores,
                            candidate_features=candidate_features,
                            z_true=z_true,
                            true_idx=int(true_idx),
                            temperature=float(args.posterior_temperature),
                        )
                    )
        if progress_i == 1 or progress_i == len(manifest_items) or progress_i % progress_every == 0:
            _progress(f"scored {progress_i}/{len(manifest_items)} response tables")

    qc_rows = latent_qc + feature_qc + alignment_rows
    qc_rows.append(
        {
            "qc_type": "response_cache_manifest",
            "n_manifest_rows_after_cli_filters": int(manifest.shape[0] + skipped_cache_rows),
            "n_response_cache_rows_scored": int(manifest.shape[0]),
            "n_manifest_rows_without_response_cache": int(skipped_cache_rows),
        }
    )
    summary_rows = _summary_rows(trial_rows)
    _add_summary_uncertainty(
        summary_rows,
        trial_rows,
        rng=uncertainty_rng,
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        confidence=float(args.uncertainty_confidence),
    )
    motion_contrasts = _pairwise_motion_contrasts(
        trial_rows,
        rng=uncertainty_rng,
        n_bootstrap=int(args.n_bootstrap),
        n_permutations=int(args.n_permutations),
        confidence=float(args.uncertainty_confidence),
    )
    axis_contrasts = _axis_contrasts_from_pairwise(motion_contrasts)
    uncertainty_rows = _uncertainty_report_rows(
        summary_rows=summary_rows,
        motion_contrasts=motion_contrasts,
        axis_contrasts=axis_contrasts,
    )
    _write_csv(out_dir / "feature_posterior_trials.csv", trial_rows)
    _write_csv(out_dir / "feature_posterior_summary.csv", summary_rows)
    _write_csv(out_dir / "feature_axis_contrasts.csv", axis_contrasts)
    _write_csv(out_dir / "feature_motion_evidence_contrasts.csv", motion_contrasts)
    _write_csv(out_dir / "feature_posterior_uncertainty.csv", uncertainty_rows)
    _write_csv(out_dir / "feature_posterior_qc.csv", qc_rows)
    _write_json(
        out_dir / "feature_posterior_metadata.json",
        {
            "run_dir": run_dir,
            "n_selected_tables": int(manifest.shape[0]),
            "n_manifest_rows_without_response_cache": int(skipped_cache_rows),
            "likelihood_scales": likelihood_scales,
            "feature_source": feature_source,
            "outputs": [
                "feature_posterior_trials.csv",
                "feature_posterior_summary.csv",
                "feature_axis_contrasts.csv",
                "feature_motion_evidence_contrasts.csv",
                "feature_posterior_uncertainty.csv",
                "feature_posterior_qc.csv",
            ],
            "uncertainty": {
                "n_bootstrap": int(args.n_bootstrap),
                "n_permutations": int(args.n_permutations),
                "confidence": float(args.uncertainty_confidence),
                "seed": int(args.uncertainty_seed),
                "permutation_test": "paired random sign flip against zero mean contrast",
                "bootstrap_target": "paired trial-level mean contrast",
            },
            "config": vars(args),
        },
    )
    report = [
        "# Feature-Posterior Joint Decoding",
        "",
        f"- Source run: `{run_dir}`",
        f"- Response tables scored: {manifest.shape[0]}",
        f"- Feature source: `{feature_source}`",
        f"- Latents: {', '.join(_parse_str_list(args.latent_names))}",
        f"- k list: {', '.join(str(k) for k in k_list)}",
        f"- Likelihood scales: {', '.join(str(v) for v in likelihood_scales)}",
        "",
        "Primary files:",
        "- `feature_posterior_trials.csv`",
        "- `feature_posterior_summary.csv`",
        "- `feature_axis_contrasts.csv`",
        "- `feature_motion_evidence_contrasts.csv`",
        "- `feature_posterior_uncertainty.csv`",
        "- `feature_posterior_qc.csv`",
        "",
        "Uncertainty:",
        f"- Bootstrap resamples: {int(args.n_bootstrap)}",
        f"- Sign-flip permutations: {int(args.n_permutations)}",
        f"- Confidence level: {float(args.uncertainty_confidence):.3f}",
        "",
        "`motion_delta` is scored as a candidate-wise log-likelihood ratio "
        "(`joint - zero`), then normalized as a contrast diagnostic rather than "
        "as an independent generative likelihood.",
        "`feature_axis_contrasts.csv` only includes axis-conditioned pairs whose "
        "`axis_shared_source_catalog` metadata is true.",
    ]
    (out_dir / "feature_posterior_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _progress(f"wrote feature-posterior outputs to {out_dir}")
    return out_dir


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
