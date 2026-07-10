#!/usr/bin/env python3
"""Audit BackImage contour-axis inputs for RR100 SSI cache reuse.

This is the freeze/audit step before running the BackImage contour-axis SSI
revival. It checks three contracts:

1. the selected n=128 contour-axis run is complete and shared-source;
2. cached 756-unit response tables can be reduced to RR100 for readout sanity
   checks;
3. full-756 spatial-map caches, if present, can be reused directly for RR100
   SSI postprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.redundancy_resolved_v1_population import load_population_view


RR100_MOVIE_MEDOID_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
DEFAULT_AXIS_RUN_DIR = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n128_c4_k16_scales_0p5_1_2_bconsistent_v1"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "active_sensing_movie_information"
    / "backimage_contour_axis_rr100_input_audit"
)
DEFAULT_SPATIAL_SEARCH_ROOTS = (
    DEFAULT_AXIS_RUN_DIR,
    ROOT / "outputs" / "active_sensing_movie_information",
    ROOT / "outputs" / "twininfo",
)
REQUIRED_AXIS_FILES = (
    "selected_windows.csv",
    "candidate_sets.csv",
    "motion_catalog.csv",
    "axis_trajectory_catalog.csv",
    "response_cache_manifest.csv",
    "run_metadata.json",
)
RESPONSE_KEYS = (
    "prior_lambda_counts",
    "known_lambda_counts",
    "zero_lambda_counts",
    "y_obs_counts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-run-dir", type=Path, default=DEFAULT_AXIS_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rr100-version", type=str, default=RR100_MOVIE_MEDOID_VERSION)
    parser.add_argument(
        "--spatial-search-root",
        action="append",
        type=Path,
        default=None,
        help="Directory or NPZ to scan for full-756 spatial-map arrays. May be repeated.",
    )
    parser.add_argument("--expected-selected-windows", type=int, default=128)
    parser.add_argument("--expected-response-units", type=int, default=756)
    parser.add_argument("--sample-response-tables", type=int, default=12)
    parser.add_argument("--max-npz-scan", type=int, default=5000)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_check(rows: list[dict[str, Any]], name: str, status: str, detail: str, **extra: Any) -> None:
    row = {"check": str(name), "status": str(status), "detail": str(detail)}
    row.update(extra)
    rows.append(row)


def as_bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def sorted_unique(values: pd.Series) -> list[Any]:
    out = []
    for value in values.dropna().unique().tolist():
        if isinstance(value, np.generic):
            value = value.item()
        out.append(value)
    return sorted(out, key=lambda item: str(item))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_indices(n_total: int, n_sample: int) -> list[int]:
    if n_total <= 0 or n_sample <= 0:
        return []
    n = min(int(n_total), int(n_sample))
    if n == n_total:
        return list(range(n_total))
    return sorted(set(int(round(v)) for v in np.linspace(0, n_total - 1, n)))


def audit_required_files(axis_run_dir: Path, checks: list[dict[str, Any]]) -> dict[str, Path]:
    paths = {name: axis_run_dir / name for name in REQUIRED_AXIS_FILES}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        add_check(checks, "required_axis_files", "FAIL", f"missing files: {', '.join(missing)}")
    else:
        add_check(checks, "required_axis_files", "PASS", "all required axis-run files are present")
    return paths


def audit_rr100(version: str, checks: list[dict[str, Any]], expected_units: int) -> tuple[Any, dict[str, Any]]:
    view = load_population_view(version_name=str(version))
    membership = np.asarray(view.membership, dtype=np.float32)
    status = "PASS" if membership.shape == (100, int(expected_units)) else "FAIL"
    add_check(
        checks,
        "rr100_membership_shape",
        status,
        f"RR100 membership shape is {membership.shape}",
        rr100_version=view.name,
        input_channels=int(view.input_channels),
        n_units=int(view.n_units),
        pooling_mode=str(view.meta.get("pooling_mode", "")),
    )
    row_nnz = np.count_nonzero(membership, axis=1)
    add_check(
        checks,
        "rr100_membership_finite",
        "PASS" if np.isfinite(membership).all() else "FAIL",
        "membership matrix is finite" if np.isfinite(membership).all() else "membership matrix contains non-finite values",
        row_nnz_min=int(row_nnz.min()) if row_nnz.size else 0,
        row_nnz_max=int(row_nnz.max()) if row_nnz.size else 0,
    )
    meta = {
        "version": view.name,
        "input_channels": int(view.input_channels),
        "n_units": int(view.n_units),
        "membership_shape": [int(v) for v in membership.shape],
        "pooling_mode": str(view.meta.get("pooling_mode", "")),
        "membership_row_nnz_min": int(row_nnz.min()) if row_nnz.size else 0,
        "membership_row_nnz_max": int(row_nnz.max()) if row_nnz.size else 0,
    }
    return view, meta


def audit_selected_windows(df: pd.DataFrame, checks: list[dict[str, Any]], expected_rows: int) -> dict[str, Any]:
    n_rows = int(df.shape[0])
    add_check(
        checks,
        "selected_window_count",
        "PASS" if n_rows == int(expected_rows) else "WARN",
        f"selected_windows has {n_rows} rows",
        expected=int(expected_rows),
    )
    if "source_row" in df.columns:
        n_unique = int(df["source_row"].nunique())
        add_check(
            checks,
            "selected_window_source_rows_unique",
            "PASS" if n_unique == n_rows else "WARN",
            f"{n_unique}/{n_rows} source_row values are unique",
        )
    if "image_feature_ok" in df.columns:
        ok = as_bool_series(df["image_feature_ok"])
        add_check(
            checks,
            "selected_window_image_features",
            "PASS" if bool(ok.all()) else "FAIL",
            f"{int(ok.sum())}/{n_rows} selected windows have image_feature_ok",
        )
    return {
        "n_selected_windows": n_rows,
        "source_row_unique": int(df["source_row"].nunique()) if "source_row" in df.columns else None,
        "sessions": sorted_unique(df["session"]) if "session" in df.columns else [],
        "n_time_samples_min": float(df["n_samples"].min()) if "n_samples" in df.columns else None,
        "n_time_samples_max": float(df["n_samples"].max()) if "n_samples" in df.columns else None,
    }


def audit_manifest(
    manifest: pd.DataFrame,
    axis_run_dir: Path,
    checks: list[dict[str, Any]],
    *,
    expected_units: int,
) -> dict[str, Any]:
    n_rows = int(manifest.shape[0])
    add_check(checks, "response_manifest_nonempty", "PASS" if n_rows > 0 else "FAIL", f"manifest has {n_rows} rows")

    n_units = sorted_unique(manifest["n_units"]) if "n_units" in manifest.columns else []
    add_check(
        checks,
        "response_manifest_n_units",
        "PASS" if n_units == [int(expected_units)] else "FAIL",
        f"manifest n_units values: {n_units}",
    )

    if "dry_run" in manifest.columns:
        dry = as_bool_series(manifest["dry_run"])
        add_check(
            checks,
            "response_manifest_not_dry_run",
            "PASS" if not bool(dry.any()) else "FAIL",
            f"{int(dry.sum())} rows marked dry_run",
        )

    missing_paths = []
    duplicate_paths = int(manifest["response_cache_path"].duplicated().sum()) if "response_cache_path" in manifest.columns else 0
    if "response_cache_path" in manifest.columns:
        for rel in manifest["response_cache_path"].astype(str):
            if not (axis_run_dir / rel).exists():
                missing_paths.append(rel)
    add_check(
        checks,
        "response_table_files_exist",
        "PASS" if not missing_paths else "FAIL",
        "all response table files exist" if not missing_paths else f"{len(missing_paths)} response table files are missing",
        missing_preview=";".join(missing_paths[:8]),
    )
    add_check(
        checks,
        "response_table_paths_unique",
        "PASS" if duplicate_paths == 0 else "FAIL",
        f"{duplicate_paths} duplicate response_cache_path rows",
    )

    shared_fraction = None
    if "axis_shared_source_catalog" in manifest.columns:
        shared = as_bool_series(manifest["axis_shared_source_catalog"])
        shared_fraction = float(shared.mean()) if shared.size else None
        add_check(
            checks,
            "axis_shared_source_catalog",
            "PASS" if bool(shared.all()) else "FAIL",
            f"{int(shared.sum())}/{shared.size} rows have axis_shared_source_catalog=True",
        )

    group_ok = None
    if {"trial_id", "axis_shared_sampled_source_rows", "prior_family"}.issubset(manifest.columns):
        good = 0
        total = 0
        bad_preview = []
        for trial_id, group in manifest.groupby("trial_id", sort=False):
            families = set(group["prior_family"].astype(str))
            if not {"axis_edge_parallel", "axis_edge_orthogonal"}.issubset(families):
                continue
            total += 1
            sampled = set(group["axis_shared_sampled_source_rows"].astype(str))
            if len(sampled) == 1:
                good += 1
            elif len(bad_preview) < 8:
                bad_preview.append(str(trial_id))
        group_ok = float(good / total) if total else None
        add_check(
            checks,
            "axis_parallel_orthogonal_shared_samples",
            "PASS" if total > 0 and good == total else "FAIL",
            f"{good}/{total} trial groups share sampled source rows across parallel/orthogonal families",
            bad_trial_preview=";".join(bad_preview),
        )

    if "n_timebins" in manifest.columns:
        bins = sorted_unique(manifest["n_timebins"])
        add_check(
            checks,
            "response_cache_timebins_auxiliary",
            "WARN",
            f"cached response tables have n_timebins={bins}; use as readout sanity checks, not the spatial-SSI timebase",
        )

    return {
        "n_response_manifest_rows": n_rows,
        "candidate_set_modes": sorted_unique(manifest["candidate_set_mode"]) if "candidate_set_mode" in manifest.columns else [],
        "prior_families": sorted_unique(manifest["prior_family"]) if "prior_family" in manifest.columns else [],
        "scales": sorted_unique(manifest["scale"]) if "scale" in manifest.columns else [],
        "n_units": n_units,
        "n_timebins": sorted_unique(manifest["n_timebins"]) if "n_timebins" in manifest.columns else [],
        "axis_shared_source_catalog_fraction": shared_fraction,
        "axis_shared_sample_fraction": group_ok,
    }


def audit_axis_catalog(axis_catalog: pd.DataFrame, motion_catalog: pd.DataFrame, checks: list[dict[str, Any]]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "n_axis_trajectory_rows": int(axis_catalog.shape[0]),
        "n_motion_catalog_rows": int(motion_catalog.shape[0]),
    }
    if "family" in axis_catalog.columns:
        meta["axis_families"] = sorted_unique(axis_catalog["family"])
    if "axis_match_status" in axis_catalog.columns:
        status_counts = axis_catalog["axis_match_status"].astype(str).value_counts(dropna=False).to_dict()
        prior_axis = axis_catalog[axis_catalog.get("role", "").astype(str).eq("prior")] if "role" in axis_catalog.columns else axis_catalog
        bad = prior_axis[~prior_axis["axis_match_status"].astype(str).isin({"matched", "", "nan"})]
        add_check(
            checks,
            "axis_catalog_match_status",
            "PASS" if bad.empty else "FAIL",
            f"axis_match_status counts: {status_counts}",
            bad_rows=int(bad.shape[0]),
        )
        meta["axis_match_status_counts"] = {str(k): int(v) for k, v in status_counts.items()}
    if "clipping_fraction" in axis_catalog.columns:
        clip = pd.to_numeric(axis_catalog["clipping_fraction"], errors="coerce")
        max_clip = float(clip.max()) if clip.notna().any() else float("nan")
        add_check(
            checks,
            "axis_catalog_clipping_fraction",
            "PASS" if not np.isfinite(max_clip) or max_clip <= 1e-8 else "WARN",
            f"max clipping_fraction={max_clip}",
        )
        meta["max_axis_clipping_fraction"] = max_clip
    return meta


def response_table_sample_audit(
    axis_run_dir: Path,
    manifest: pd.DataFrame,
    rr100_membership: np.ndarray,
    *,
    n_sample: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    if "response_cache_path" not in manifest.columns:
        return rows, ["response_cache_manifest lacks response_cache_path"]
    for idx in sample_indices(int(manifest.shape[0]), int(n_sample)):
        man_row = manifest.iloc[int(idx)]
        rel = str(man_row["response_cache_path"])
        path = axis_run_dir / rel
        row: dict[str, Any] = {
            "manifest_index": int(idx),
            "response_cache_path": rel,
            "prior_family": str(man_row.get("prior_family", "")),
            "scale": float(man_row.get("scale", np.nan)),
            "status": "PASS",
        }
        if not path.exists():
            row["status"] = "FAIL"
            row["error"] = "missing_response_table"
            failures.append(rel)
            rows.append(row)
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in RESPONSE_KEYS:
                    if key not in data.files:
                        row[f"{key}_present"] = False
                        continue
                    arr = np.asarray(data[key])
                    row[f"{key}_present"] = True
                    row[f"{key}_shape"] = "x".join(str(int(v)) for v in arr.shape)
                    row[f"{key}_dtype"] = str(arr.dtype)
                    row[f"{key}_finite"] = bool(np.isfinite(arr).all())
                    row[f"{key}_unit_dim"] = int(arr.shape[-1]) if arr.ndim else -1
                    if arr.ndim >= 2 and int(arr.shape[-1]) == rr100_membership.shape[1]:
                        reduced = np.einsum("...tc,rc->...tr", arr, rr100_membership, optimize=True)
                        row[f"{key}_rr100_shape"] = "x".join(str(int(v)) for v in reduced.shape)
                        row[f"{key}_rr100_finite"] = bool(np.isfinite(reduced).all())
                    elif arr.ndim >= 1:
                        row[f"{key}_rr100_shape"] = ""
                        row[f"{key}_rr100_finite"] = False
                        if key in {"prior_lambda_counts", "known_lambda_counts", "zero_lambda_counts", "y_obs_counts"}:
                            row["status"] = "FAIL"
                            failures.append(f"{rel}:{key}:unit_dim={arr.shape[-1]}")
        except Exception as exc:
            row["status"] = "FAIL"
            row["error"] = str(exc)
            failures.append(f"{rel}:{exc}")
        rows.append(row)
    return rows, failures


def npz_headers(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            with zf.open(name) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            rows.append(
                {
                    "key": name[:-4],
                    "shape": tuple(int(v) for v in shape),
                    "dtype": str(dtype),
                    "fortran_order": bool(fortran_order),
                }
            )
    return rows


def iter_npz_paths(roots: list[Path], max_n: int) -> tuple[list[Path], list[str], bool]:
    paths: list[Path] = []
    missing: list[str] = []
    truncated = False
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            missing.append(str(root))
            continue
        if root.is_file():
            candidates = [root] if root.suffix == ".npz" else []
        else:
            candidates = []
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    if filename.endswith(".npz"):
                        candidates.append(Path(dirpath) / filename)
        for path in sorted(candidates):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
            if len(paths) >= int(max_n):
                truncated = True
                return paths, missing, truncated
    return paths, missing, truncated


def audit_spatial_map_inventory(
    roots: list[Path],
    checks: list[dict[str, Any]],
    *,
    expected_units: int,
    max_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    npz_paths, missing_roots, truncated = iter_npz_paths(roots, int(max_n))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in npz_paths:
        try:
            headers = npz_headers(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        for header in headers:
            shape = tuple(header["shape"])
            is_tchw = len(shape) == 4 and shape[1] == int(expected_units) and shape[2] > 1 and shape[3] > 1
            is_possible_5d = (
                len(shape) == 5
                and int(expected_units) in shape
                and shape[-1] > 1
                and shape[-2] > 1
            )
            if not is_tchw and not is_possible_5d:
                continue
            rows.append(
                {
                    "npz_path": str(path),
                    "key": str(header["key"]),
                    "shape": "x".join(str(int(v)) for v in shape),
                    "dtype": str(header["dtype"]),
                    "file_size_bytes": int(path.stat().st_size),
                    "candidate_type": "T_x_756_x_H_x_W" if is_tchw else "possible_5d_full756_spatial",
                }
            )
    if missing_roots:
        add_check(
            checks,
            "spatial_search_roots_exist",
            "WARN",
            f"{len(missing_roots)} spatial-search roots are missing",
            missing_roots=";".join(missing_roots),
        )
    else:
        add_check(checks, "spatial_search_roots_exist", "PASS", "all spatial-search roots exist")
    if rows:
        add_check(
            checks,
            "full756_spatial_map_inventory",
            "PASS",
            f"found {len(rows)} candidate full-756 spatial-map arrays across {len(npz_paths)} scanned NPZ files",
            truncated=bool(truncated),
        )
    else:
        add_check(
            checks,
            "full756_spatial_map_inventory",
            "WARN",
            f"found no candidate full-756 spatial-map arrays across {len(npz_paths)} scanned NPZ files",
            truncated=bool(truncated),
            header_errors=len(errors),
        )
    return rows, {
        "n_npz_scanned": int(len(npz_paths)),
        "n_candidate_full756_spatial_arrays": int(len(rows)),
        "truncated": bool(truncated),
        "missing_roots": missing_roots,
        "header_errors": errors[:20],
    }


def main() -> None:
    args = parse_args()
    axis_run_dir = Path(args.axis_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    paths = audit_required_files(axis_run_dir, checks)
    rr100_view, rr100_meta = audit_rr100(str(args.rr100_version), checks, int(args.expected_response_units))
    rr100_membership = np.asarray(rr100_view.membership, dtype=np.float32)

    run_metadata = read_json(paths["run_metadata.json"]) if paths["run_metadata.json"].exists() else {}
    selected = pd.read_csv(paths["selected_windows.csv"]) if paths["selected_windows.csv"].exists() else pd.DataFrame()
    manifest = pd.read_csv(paths["response_cache_manifest.csv"]) if paths["response_cache_manifest.csv"].exists() else pd.DataFrame()
    motion_catalog = pd.read_csv(paths["motion_catalog.csv"]) if paths["motion_catalog.csv"].exists() else pd.DataFrame()
    axis_catalog = pd.read_csv(paths["axis_trajectory_catalog.csv"]) if paths["axis_trajectory_catalog.csv"].exists() else pd.DataFrame()

    selected_meta = audit_selected_windows(selected, checks, int(args.expected_selected_windows)) if not selected.empty else {}
    manifest_meta = (
        audit_manifest(manifest, axis_run_dir, checks, expected_units=int(args.expected_response_units))
        if not manifest.empty
        else {}
    )
    axis_meta = (
        audit_axis_catalog(axis_catalog, motion_catalog, checks)
        if not axis_catalog.empty and not motion_catalog.empty
        else {}
    )

    sample_rows, sample_failures = response_table_sample_audit(
        axis_run_dir,
        manifest,
        rr100_membership,
        n_sample=int(args.sample_response_tables),
    )
    add_check(
        checks,
        "sample_response_tables_reduce_to_rr100",
        "PASS" if not sample_failures else "FAIL",
        f"{len(sample_rows) - len(sample_failures)}/{len(sample_rows)} sampled response tables reduced cleanly to RR100",
        failure_preview=";".join(sample_failures[:8]),
    )

    spatial_roots = [Path(p) for p in (args.spatial_search_root or DEFAULT_SPATIAL_SEARCH_ROOTS)]
    spatial_rows, spatial_meta = audit_spatial_map_inventory(
        spatial_roots,
        checks,
        expected_units=int(args.expected_response_units),
        max_n=int(args.max_npz_scan),
    )

    n_fail = sum(1 for row in checks if row["status"] == "FAIL")
    n_warn = sum(1 for row in checks if row["status"] == "WARN")
    summary = {
        "axis_run_dir": axis_run_dir,
        "out_dir": out_dir,
        "rr100": rr100_meta,
        "run_metadata_config": run_metadata.get("config", {}),
        "selected_windows": selected_meta,
        "response_manifest": manifest_meta,
        "axis_catalog": axis_meta,
        "spatial_map_inventory": spatial_meta,
        "n_checks": int(len(checks)),
        "n_fail": int(n_fail),
        "n_warn": int(n_warn),
        "overall_status": "FAIL" if n_fail else ("WARN" if n_warn else "PASS"),
        "next_action": (
            "derive RR100 SSI from matching full-756 spatial-map caches"
            if spatial_meta.get("n_candidate_full756_spatial_arrays", 0) > 0
            else "no matching full-756 spatial-map cache found in scanned roots; generate missing maps before RR100 SSI"
        ),
    }

    write_csv_rows(out_dir / "audit_checks.csv", checks)
    write_csv_rows(out_dir / "response_table_sample_audit.csv", sample_rows)
    write_csv_rows(out_dir / "full756_spatial_map_inventory.csv", spatial_rows)
    write_json(out_dir / "frozen_input_audit_summary.json", summary)

    print(f"Wrote audit summary to {out_dir / 'frozen_input_audit_summary.json'}")
    print(f"Overall status: {summary['overall_status']} ({n_fail} fail, {n_warn} warn)")
    print(f"Next action: {summary['next_action']}")
    if n_fail and bool(args.fail_on_error):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
