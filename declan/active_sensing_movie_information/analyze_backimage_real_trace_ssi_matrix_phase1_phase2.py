#!/usr/bin/env python3
"""Phase 1/2 conditioning analysis for the BackImage real-trace SSI matrix.

This script turns the matrix-first 100k x RR100 SSI dataset into joined
movie/image/trace/unit analysis tables and recreates the older hand-built
condition logic as post hoc metadata-conditioned summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
DEFAULT_DENSE_TUNING_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_dense_sf_tf_speed_pref_groups_v1/"
    "cycle_valid_dense_sf_tf_fit_unit_summary.csv"
)
DEFAULT_STRONG_RAMP_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "strong_monotonic_1x_to_3x_rampers/strong_monotonic_ramping_units.csv"
)
DEFAULT_DROP_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_spatial_ssi_n128_across_sweep_v1/"
    "drop_1_to_3_unit_image_maps_top6_img6_v1/drop_1_to_3_unit_selection.csv"
)
DEFAULT_TOP_DELTA_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_top_delta_drivers_v1/"
    "selected_units.csv"
)
DEFAULT_ORIENTATION_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_instantaneous_unit_maps_top_delta_drivers_v1/"
    "orientation_tuning_groups.csv"
)
DEFAULT_TRACE_PATH_CONTEXT_REFERENCE_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_trace_bank_diffusion_large_fixation_sample_n5000_n40_v1/"
    "filtered_path_length_le350arcmin/trace_bank_metadata_filtered.csv"
)
STABILIZED_BASELINE_FILES = {
    "ssi": "stabilized_ssi_by_image.npy",
    "expected": "stabilized_expected_spikes_by_image.npy",
    "mean_rate": "stabilized_mean_rate_by_image.npy",
    "population": "stabilized_population_ssi_by_image.npy",
    "table": "stabilized_movie_feature_table.csv",
    "summary": "stabilized_baseline_summary.json",
}

EPS = 1e-12
LOW_BLUE = "#1f77b4"
HIGH_RED = "#d62728"
MID_GRAY = "#8a8a8a"
ACCENT_ORANGE = "#ff7f0e"
ACCENT_PURPLE = "#9467bd"
POP_COLOR = LOW_BLUE
NO_MS_COLOR = LOW_BLUE
MS_COLOR = HIGH_RED
GROUP_COLORS = [
    LOW_BLUE,
    HIGH_RED,
    MID_GRAY,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
]
LABEL_COLORS = {
    "low_sf": LOW_BLUE,
    "low_tf_pref": LOW_BLUE,
    "low_speed_preferring": LOW_BLUE,
    "low_speed_pref_tertile": LOW_BLUE,
    "small_scale_preferring": LOW_BLUE,
    "strong_monotonic_ramp": LOW_BLUE,
    "no_microsaccade": LOW_BLUE,
    "unit_matched_to_image_contour": LOW_BLUE,
    "matched_units": LOW_BLUE,
    "along_contour_axis": LOW_BLUE,
    "reliable": LOW_BLUE,
    "middle_sf": MID_GRAY,
    "middle_tf_pref": MID_GRAY,
    "middle_speed_pref_tertile": MID_GRAY,
    "oblique": MID_GRAY,
    "oblique_units": MID_GRAY,
    "other": MID_GRAY,
    "weak_or_unreliable": MID_GRAY,
    "low_anisotropy": ACCENT_ORANGE,
    "high_sf": HIGH_RED,
    "high_tf_pref": HIGH_RED,
    "high_speed_preferring": HIGH_RED,
    "high_speed_pref_tertile": HIGH_RED,
    "large_scale_preferring": HIGH_RED,
    "microsaccade": HIGH_RED,
    "orthogonal_units": HIGH_RED,
    "across_contour_axis": HIGH_RED,
    "strong": HIGH_RED,
    "drop_1_to_3": HIGH_RED,
    "top_delta_driver": ACCENT_ORANGE,
}


def color_for_label(label: object, fallback_index: int = 0) -> str:
    key = sanitize(label)
    if key in LABEL_COLORS:
        return LABEL_COLORS[key]
    for token, color in LABEL_COLORS.items():
        if token in key:
            return color
    return GROUP_COLORS[int(fallback_index) % len(GROUP_COLORS)]


def ordered_groups(unit_col: str, groups: list[object]) -> list[object]:
    preferred = {
        "sf_group": ["low_sf", "middle_sf", "high_sf"],
        "analysis_tf_pref_group": ["low_tf_pref", "middle_tf_pref", "high_tf_pref"],
        "speed_pref_group": ["low_speed_preferring", "high_speed_preferring"],
        "analysis_speed_pref_tertile": [
            "low_speed_pref_tertile",
            "middle_speed_pref_tertile",
            "high_speed_pref_tertile",
        ],
        "old_scale_curve_group": ["small_scale_preferring", "large_scale_preferring"],
        "analysis_ramp_drop_group": ["strong_monotonic_ramp", "drop_1_to_3", "top_delta_driver", "other"],
        "old_axis_orientation_group": ["contour_biased", "off_axis_or_mixed", "across_biased"],
    }.get(unit_col, [])
    group_text = {str(group): group for group in groups}
    ordered = [group_text.pop(key) for key in preferred if key in group_text]
    ordered.extend(group_text[key] for key in sorted(group_text))
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dense-tuning-csv", type=Path, default=DEFAULT_DENSE_TUNING_CSV)
    parser.add_argument("--strong-ramp-csv", type=Path, default=DEFAULT_STRONG_RAMP_CSV)
    parser.add_argument("--drop-unit-csv", type=Path, default=DEFAULT_DROP_CSV)
    parser.add_argument("--top-delta-csv", type=Path, default=DEFAULT_TOP_DELTA_CSV)
    parser.add_argument("--orientation-groups-csv", type=Path, default=DEFAULT_ORIENTATION_GROUPS_CSV)
    parser.add_argument("--image-axis-col", type=str, default="image_edge_axis_deg")
    parser.add_argument("--trace-axis-col", type=str, default="rendered_cov_orientation_deg")
    parser.add_argument("--n-trace-bins", type=int, default=6)
    parser.add_argument("--n-diffusion-bins", type=int, default=6)
    parser.add_argument("--n-image-bins", type=int, default=4)
    parser.add_argument("--match-max-deg", type=float, default=22.5)
    parser.add_argument("--orthogonal-min-deg", type=float, default=67.5)
    parser.add_argument("--trace-axis-min-anisotropy", type=float, default=0.5)
    parser.add_argument("--contour-match-sf-groups", type=str, default="low_sf,high_sf")
    parser.add_argument("--min-contour-match-osi", type=float, default=0.05)
    parser.add_argument("--min-contour-matched-images-per-unit", type=int, default=1)
    parser.add_argument(
        "--trace-path-context-reference-csv",
        type=Path,
        default=DEFAULT_TRACE_PATH_CONTEXT_REFERENCE_CSV,
        help="Optional larger fixation trace-bank CSV used only for path-length reference bands.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[backimage-real-trace-phase1-phase2] {message}", flush=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"png": out_dir / f"{stem}.png", "pdf": out_dir / f"{stem}.pdf"}
    fig.savefig(paths["png"], dpi=220, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def sanitize(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "missing"


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def sem(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def axis_delta_deg(a_deg: Any, b_deg: Any) -> np.ndarray:
    a = np.asarray(a_deg, dtype=float)
    b = np.asarray(b_deg, dtype=float)
    return np.abs(0.5 * np.degrees(np.angle(np.exp(2j * np.radians(a - b)))))


def coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for col in columns:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def microsaccade_count(frame: pd.DataFrame) -> pd.Series:
    for col in ("rendered_n_microsaccade_events", "n_microsaccade_events", "source_n_microsaccade_events"):
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(0).clip(lower=0).astype(int)
    return pd.Series(np.zeros(frame.shape[0], dtype=int), index=frame.index)


def add_quantile_bin(
    frame: pd.DataFrame,
    source_col: str,
    out_col: str,
    *,
    n_bins: int,
    table_name: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = frame.copy()
    out[out_col] = pd.NA
    if source_col not in out.columns:
        return out, []
    values = pd.to_numeric(out[source_col], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return out, []
    unique_count = int(finite.nunique(dropna=True))
    q = min(int(n_bins), unique_count)
    if q <= 1:
        out.loc[finite.index, out_col] = "q01"
    else:
        codes = pd.qcut(finite, q=q, labels=False, duplicates="drop")
        out.loc[finite.index, out_col] = [f"q{int(code) + 1:02d}" if pd.notna(code) else pd.NA for code in codes]
    rows: list[dict[str, Any]] = []
    for order, (label, sub) in enumerate(out[out[out_col].notna()].groupby(out_col, sort=True), start=1):
        vals = pd.to_numeric(sub[source_col], errors="coerce").dropna()
        rows.append(
            {
                "table": table_name,
                "source_col": source_col,
                "bin_col": out_col,
                "bin_label": str(label),
                "bin_order": order,
                "n": int(vals.shape[0]),
                "min": float(vals.min()),
                "median": float(vals.median()),
                "max": float(vals.max()),
            }
        )
    return out, rows


def add_tertile_group(frame: pd.DataFrame, source_col: str, out_col: str, labels: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out, rows = add_quantile_bin(frame, source_col, out_col, n_bins=3, table_name="unit")
    label_map = {f"q{idx + 1:02d}": label for idx, label in enumerate(labels)}
    out[out_col] = out[out_col].map(label_map).fillna(out[out_col])
    for row in rows:
        row["bin_label_original"] = row["bin_label"]
        row["bin_label"] = label_map.get(str(row["bin_label"]), str(row["bin_label"]))
    return out, rows


def classify_image_contours(image: pd.DataFrame) -> pd.Series:
    strong = image.get("image_contour_strong", pd.Series(False, index=image.index)).fillna(False).astype(bool)
    reliable = image.get("image_contour_reliable", pd.Series(False, index=image.index)).fillna(False).astype(bool)
    out = pd.Series("weak_or_unreliable", index=image.index, dtype=object)
    out.loc[reliable] = "reliable"
    out.loc[strong] = "strong"
    return out


def load_matrix_dataset(matrix_dir: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ssi = np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r")
    expected_path = matrix_dir / "expected_spikes_matrix.npy"
    expected = np.load(expected_path, mmap_mode="r") if expected_path.exists() else None
    population = np.load(matrix_dir / "population_ssi.npy", mmap_mode="r")
    movie = pd.read_csv(matrix_dir / "movie_feature_table.csv")
    image = pd.read_csv(matrix_dir / "image_feature_table.csv")
    trace = pd.read_csv(matrix_dir / "trace_feature_table.csv")
    unit = pd.read_csv(matrix_dir / "unit_feature_table.csv")

    if ssi.shape[0] != movie.shape[0]:
        raise ValueError(f"SSI rows {ssi.shape[0]} do not match movie rows {movie.shape[0]}.")
    if ssi.shape[1] != unit.shape[0]:
        raise ValueError(f"SSI columns {ssi.shape[1]} do not match unit rows {unit.shape[0]}.")
    if population.shape[0] != movie.shape[0]:
        raise ValueError(f"Population SSI rows {population.shape[0]} do not match movie rows {movie.shape[0]}.")
    if expected is not None and expected.shape != ssi.shape:
        raise ValueError(f"Expected-spikes shape {expected.shape} does not match SSI shape {ssi.shape}.")
    return ssi, expected, population, movie, image, trace, unit


def load_stabilized_baseline(matrix_dir: Path, image: pd.DataFrame, unit: pd.DataFrame) -> dict[str, Any] | None:
    ssi_path = matrix_dir / STABILIZED_BASELINE_FILES["ssi"]
    if not ssi_path.exists():
        return None
    ssi = np.load(ssi_path, mmap_mode="r")
    if ssi.ndim != 2:
        raise ValueError(f"Expected stabilized SSI shape (images, units), got {ssi.shape}.")
    if ssi.shape[0] != image.shape[0]:
        raise ValueError(f"Stabilized SSI rows {ssi.shape[0]} do not match image rows {image.shape[0]}.")
    if ssi.shape[1] != unit.shape[0]:
        raise ValueError(f"Stabilized SSI columns {ssi.shape[1]} do not match unit rows {unit.shape[0]}.")

    baseline: dict[str, Any] = {"ssi": ssi, "source_dir": matrix_dir}
    expected_path = matrix_dir / STABILIZED_BASELINE_FILES["expected"]
    if expected_path.exists():
        expected = np.load(expected_path, mmap_mode="r")
        if expected.shape != ssi.shape:
            raise ValueError(f"Stabilized expected-spikes shape {expected.shape} does not match SSI shape {ssi.shape}.")
        baseline["expected"] = expected
    else:
        baseline["expected"] = None
    mean_rate_path = matrix_dir / STABILIZED_BASELINE_FILES["mean_rate"]
    if mean_rate_path.exists():
        mean_rate = np.load(mean_rate_path, mmap_mode="r")
        if mean_rate.shape != ssi.shape:
            raise ValueError(f"Stabilized mean-rate shape {mean_rate.shape} does not match SSI shape {ssi.shape}.")
        baseline["mean_rate"] = mean_rate
    population_path = matrix_dir / STABILIZED_BASELINE_FILES["population"]
    if population_path.exists():
        population = np.load(population_path, mmap_mode="r")
        if population.shape[0] != ssi.shape[0]:
            raise ValueError(f"Stabilized population rows {population.shape[0]} do not match SSI rows {ssi.shape[0]}.")
        baseline["population"] = population
    table_path = matrix_dir / STABILIZED_BASELINE_FILES["table"]
    if table_path.exists():
        table = pd.read_csv(table_path)
    else:
        table = image[["image_index"]].copy()
        table.insert(0, "baseline_row_index", np.arange(table.shape[0], dtype=int))
    if "baseline_row_index" not in table.columns:
        table = table.copy()
        table.insert(0, "baseline_row_index", np.arange(table.shape[0], dtype=int))
    if "image_index" not in table.columns:
        raise ValueError(f"{table_path} must contain image_index.")
    if table["image_index"].astype(int).nunique() != table.shape[0]:
        raise ValueError("Stabilized baseline table has duplicate image_index values.")
    baseline["table"] = table
    summary_path = matrix_dir / STABILIZED_BASELINE_FILES["summary"]
    if summary_path.exists():
        baseline["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return baseline


def baseline_image_lookup(stabilized_baseline: dict[str, Any]) -> dict[int, int]:
    if "_image_index_to_row" not in stabilized_baseline:
        table = stabilized_baseline["table"]
        stabilized_baseline["_image_index_to_row"] = {
            int(row.image_index): int(row.baseline_row_index)
            for row in table[["image_index", "baseline_row_index"]].itertuples(index=False)
        }
    return stabilized_baseline["_image_index_to_row"]


def baseline_rows_for_image_indices(image_indices: Any, stabilized_baseline: dict[str, Any]) -> np.ndarray:
    lookup = baseline_image_lookup(stabilized_baseline)
    rows: list[int] = []
    missing: list[int] = []
    for image_index in np.asarray(image_indices, dtype=int).reshape(-1):
        if int(image_index) not in lookup:
            missing.append(int(image_index))
        else:
            rows.append(int(lookup[int(image_index)]))
    if missing:
        preview = ", ".join(str(value) for value in missing[:8])
        raise ValueError(f"Stabilized baseline is missing image_index values: {preview}")
    return np.asarray(rows, dtype=int)


def baseline_values_for_movie(values_by_image: np.ndarray, movie: pd.DataFrame, stabilized_baseline: dict[str, Any]) -> np.ndarray:
    rows = baseline_rows_for_image_indices(movie["image_index"].astype(int).to_numpy(), stabilized_baseline)
    return np.asarray(values_by_image)[rows]


def merge_if_present(
    unit: pd.DataFrame,
    csv_path: Path,
    *,
    columns: dict[str, str],
    source_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {
        "source": source_name,
        "path": str(csv_path),
        "exists": bool(csv_path.exists()),
        "n_rows": 0,
        "n_units_joined": 0,
    }
    if not csv_path.exists():
        return unit, info
    extra = pd.read_csv(csv_path)
    info["n_rows"] = int(extra.shape[0])
    if "unit_index" not in extra.columns:
        info["error"] = "missing unit_index"
        return unit, info
    keep_cols = ["unit_index"] + [col for col in columns if col in extra.columns]
    if len(keep_cols) == 1:
        return unit, info
    extra = extra[keep_cols].drop_duplicates("unit_index").rename(columns=columns)
    info["n_units_joined"] = int(extra["unit_index"].nunique())
    return unit.merge(extra, on="unit_index", how="left", validate="one_to_one"), info


def enrich_unit_table(unit: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    out = unit.copy()
    sources: list[dict[str, Any]] = []

    out, info = merge_if_present(
        out,
        Path(args.dense_tuning_csv),
        columns={
            "curve_group": "old_scale_curve_group",
            "curve_group_label": "old_scale_curve_group_label",
            "speed_pref_group": "speed_pref_group",
            "speed_pref_label": "speed_pref_label",
            "fit_ok": "dense_fit_ok",
            "fit_status": "dense_fit_status",
            "fit_r2": "dense_fit_r2",
            "fit_pref_sf_cpd": "dense_fit_pref_sf_cpd",
            "fit_pref_tf_hz": "dense_fit_pref_tf_hz",
            "fit_pref_speed_dps": "dense_fit_pref_speed_dps",
            "fit_fwhm_sf_octaves": "dense_fit_fwhm_sf_octaves",
            "fit_fwhm_tf_octaves": "dense_fit_fwhm_tf_octaves",
            "observed_peak_tf_hz": "dense_observed_peak_tf_hz",
            "observed_peak_speed_dps": "dense_observed_peak_speed_dps",
        },
        source_name="dense_sf_tf_speed_pref_groups",
    )
    sources.append(info)

    out["is_strong_monotonic_ramp_unit"] = False
    if Path(args.strong_ramp_csv).exists():
        ramp = pd.read_csv(args.strong_ramp_csv)
        ids = set(pd.to_numeric(ramp.get("unit_index", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
        out["is_strong_monotonic_ramp_unit"] = out["unit_index"].astype(int).isin(ids)
        sources.append(
            {
                "source": "strong_monotonic_ramp_units",
                "path": str(args.strong_ramp_csv),
                "exists": True,
                "n_rows": int(ramp.shape[0]),
                "n_units_joined": int(len(ids)),
            }
        )
        keep = [col for col in ["unit_index", "delta_3_minus_1", "strong_delta_threshold"] if col in ramp.columns]
        if keep:
            out = out.merge(
                ramp[keep].drop_duplicates("unit_index").rename(
                    columns={
                        "delta_3_minus_1": "strong_ramp_delta_3_minus_1",
                        "strong_delta_threshold": "strong_ramp_delta_threshold",
                    }
                ),
                on="unit_index",
                how="left",
                validate="one_to_one",
            )
    else:
        sources.append({"source": "strong_monotonic_ramp_units", "path": str(args.strong_ramp_csv), "exists": False})

    out["is_drop_1_to_3_unit"] = False
    if Path(args.drop_unit_csv).exists():
        drop = pd.read_csv(args.drop_unit_csv)
        ids = set(pd.to_numeric(drop.get("unit_index", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
        out["is_drop_1_to_3_unit"] = out["unit_index"].astype(int).isin(ids)
        sources.append(
            {
                "source": "drop_1_to_3_units",
                "path": str(args.drop_unit_csv),
                "exists": True,
                "n_rows": int(drop.shape[0]),
                "n_units_joined": int(len(ids)),
            }
        )
        keep = [col for col in ["unit_index", "drop_1_to_3"] if col in drop.columns]
        if keep:
            out = out.merge(
                drop[keep].drop_duplicates("unit_index").rename(columns={"drop_1_to_3": "drop_1_to_3_metric"}),
                on="unit_index",
                how="left",
                validate="one_to_one",
            )
    else:
        sources.append({"source": "drop_1_to_3_units", "path": str(args.drop_unit_csv), "exists": False})

    out["is_top_delta_driver_unit"] = False
    if Path(args.top_delta_csv).exists():
        top = pd.read_csv(args.top_delta_csv)
        ids = set(pd.to_numeric(top.get("unit_index", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
        out["is_top_delta_driver_unit"] = out["unit_index"].astype(int).isin(ids)
        sources.append(
            {
                "source": "top_delta_driver_units",
                "path": str(args.top_delta_csv),
                "exists": True,
                "n_rows": int(top.shape[0]),
                "n_units_joined": int(len(ids)),
            }
        )
    else:
        sources.append({"source": "top_delta_driver_units", "path": str(args.top_delta_csv), "exists": False})

    out, info = merge_if_present(
        out,
        Path(args.orientation_groups_csv),
        columns={
            "orientation_group": "old_axis_orientation_group",
            "orientation_group_label": "old_axis_orientation_group_label",
            "orientation_group_rank": "old_axis_orientation_group_rank",
            "preferred_orientation_deg": "orientation_probe_preferred_orientation_deg",
            "orientation_selectivity_index": "orientation_probe_selectivity_index",
        },
        source_name="old_axis_orientation_tuning_groups",
    )
    sources.append(info)

    out["analysis_preferred_orientation_deg"] = coalesce_numeric(
        out,
        [
            "orientation_probe_preferred_orientation_deg",
            "prior_preferred_orientation_deg",
            "dynamic_peak_orientation_deg_by_amp",
            "static_peak_orientation_deg_by_mean_rate",
        ],
    )
    out["analysis_tf_pref_hz"] = coalesce_numeric(
        out,
        ["dense_fit_pref_tf_hz", "dense_observed_peak_tf_hz", "dynamic_peak_temporal_hz_by_amp"],
    )
    out["analysis_speed_pref_dps"] = coalesce_numeric(out, ["dense_fit_pref_speed_dps", "dense_observed_peak_speed_dps"])
    if out["analysis_speed_pref_dps"].isna().all() and {"dynamic_peak_temporal_hz_by_amp", "dynamic_log_gaussian_marginal_sf_cpd"}.issubset(out.columns):
        tf = pd.to_numeric(out["dynamic_peak_temporal_hz_by_amp"], errors="coerce")
        sf = pd.to_numeric(out["dynamic_log_gaussian_marginal_sf_cpd"], errors="coerce")
        out["analysis_speed_pref_dps"] = tf / sf.where(sf > 0)

    out["orientation_pref_bin"] = pd.cut(
        out["analysis_preferred_orientation_deg"] % 180.0,
        bins=[0, 30, 60, 90, 120, 150, 180],
        labels=["ori_0_30", "ori_30_60", "ori_60_90", "ori_90_120", "ori_120_150", "ori_150_180"],
        include_lowest=True,
        right=False,
    ).astype(object)

    bin_rows: list[dict[str, Any]] = []
    out, rows = add_tertile_group(
        out,
        "analysis_tf_pref_hz",
        "analysis_tf_pref_group",
        labels=["low_tf_pref", "middle_tf_pref", "high_tf_pref"],
    )
    bin_rows.extend(rows)
    out, rows = add_tertile_group(
        out,
        "analysis_speed_pref_dps",
        "analysis_speed_pref_tertile",
        labels=["low_speed_pref_tertile", "middle_speed_pref_tertile", "high_speed_pref_tertile"],
    )
    bin_rows.extend(rows)

    out["analysis_ramp_drop_group"] = "other"
    out.loc[out["is_top_delta_driver_unit"].fillna(False).astype(bool), "analysis_ramp_drop_group"] = "top_delta_driver"
    out.loc[out["is_drop_1_to_3_unit"].fillna(False).astype(bool), "analysis_ramp_drop_group"] = "drop_1_to_3"
    out.loc[out["is_strong_monotonic_ramp_unit"].fillna(False).astype(bool), "analysis_ramp_drop_group"] = "strong_monotonic_ramp"
    return out, sources, bin_rows


def weighted_bits_for_mask(
    ssi: np.ndarray,
    expected: np.ndarray | None,
    unit_mask: np.ndarray,
    *,
    row_indices: np.ndarray | None = None,
) -> np.ndarray:
    if not np.any(unit_mask):
        n_rows = int(ssi.shape[0]) if row_indices is None else int(row_indices.size)
        return np.full(n_rows, np.nan, dtype=np.float32)
    block = ssi[:, unit_mask] if row_indices is None else ssi[row_indices][:, unit_mask]
    if expected is None:
        return np.nanmean(block, axis=1).astype(np.float32)
    exp_block = expected[:, unit_mask] if row_indices is None else expected[row_indices][:, unit_mask]
    numer = np.sum(np.asarray(block, dtype=np.float64) * np.asarray(exp_block, dtype=np.float64), axis=1)
    denom = np.sum(np.asarray(exp_block, dtype=np.float64), axis=1)
    return np.divide(numer, np.maximum(denom, EPS)).astype(np.float32)


def mean_bits_for_mask(ssi: np.ndarray, unit_mask: np.ndarray, *, row_indices: np.ndarray | None = None) -> np.ndarray:
    if not np.any(unit_mask):
        n_rows = int(ssi.shape[0]) if row_indices is None else int(row_indices.size)
        return np.full(n_rows, np.nan, dtype=np.float32)
    block = ssi[:, unit_mask] if row_indices is None else ssi[row_indices][:, unit_mask]
    return np.nanmean(block, axis=1).astype(np.float32)


def add_unit_group_columns(
    movie: pd.DataFrame,
    ssi: np.ndarray,
    expected: np.ndarray | None,
    unit: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, list[str]]]:
    out = movie.copy()
    registry: list[dict[str, Any]] = []
    plot_groups: dict[str, list[str]] = {}
    specs = [
        ("sf_group", "sf"),
        ("analysis_tf_pref_group", "tf_pref"),
        ("speed_pref_group", "speed_pref_legacy"),
        ("analysis_speed_pref_tertile", "speed_pref_tertile"),
        ("old_scale_curve_group", "old_scale_curve"),
        ("analysis_ramp_drop_group", "ramp_drop"),
        ("orientation_pref_bin", "unit_orientation_pref"),
        ("old_axis_orientation_group", "old_axis_orientation"),
    ]
    for unit_col, prefix in specs:
        if unit_col not in unit.columns:
            continue
        values = unit[unit_col].astype(object)
        groups = [g for g in pd.Series(values).dropna().unique().tolist() if str(g).lower() != "nan"]
        if not groups:
            continue
        groups = ordered_groups(unit_col, groups)
        plot_groups[prefix] = []
        for group in groups:
            mask = values.astype(str).to_numpy() == str(group)
            if not np.any(mask):
                continue
            safe = sanitize(group)
            weighted_col = f"{prefix}_{safe}_weighted_ssi"
            mean_col = f"{prefix}_{safe}_mean_ssi"
            out[weighted_col] = weighted_bits_for_mask(ssi, expected, mask)
            out[mean_col] = mean_bits_for_mask(ssi, mask)
            plot_groups[prefix].append(weighted_col)
            registry.append(
                {
                    "condition_family": prefix,
                    "unit_feature_column": unit_col,
                    "unit_group": str(group),
                    "weighted_ssi_column": weighted_col,
                    "mean_ssi_column": mean_col,
                    "n_units": int(np.count_nonzero(mask)),
                }
            )
    return out, registry, plot_groups


def add_stabilized_baseline_columns(
    movie: pd.DataFrame,
    unit: pd.DataFrame,
    registry: list[dict[str, Any]],
    stabilized_baseline: dict[str, Any] | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = movie.copy()
    if stabilized_baseline is None:
        out["has_stabilized_baseline"] = False
        return out, registry

    out["has_stabilized_baseline"] = True
    baseline_ssi = stabilized_baseline["ssi"]
    baseline_expected = stabilized_baseline.get("expected")
    baseline_population = stabilized_baseline.get("population")
    if baseline_population is not None:
        out["stabilized_population_ssi"] = baseline_values_for_movie(baseline_population, out, stabilized_baseline).astype(np.float32)
        out["population_ssi_delta_vs_stabilized"] = out["population_ssi"] - out["stabilized_population_ssi"]

    values_by_unit_col: dict[str, pd.Series] = {}
    for row in registry:
        unit_col = str(row.get("unit_feature_column", ""))
        group = str(row.get("unit_group", ""))
        weighted_col = str(row.get("weighted_ssi_column", ""))
        mean_col = str(row.get("mean_ssi_column", ""))
        if not unit_col or unit_col not in unit.columns:
            continue
        if unit_col not in values_by_unit_col:
            values_by_unit_col[unit_col] = unit[unit_col].astype(str)
        mask = values_by_unit_col[unit_col].to_numpy() == group
        if not np.any(mask):
            continue
        weighted_baseline_by_image = weighted_bits_for_mask(baseline_ssi, baseline_expected, mask)
        mean_baseline_by_image = mean_bits_for_mask(baseline_ssi, mask)
        if weighted_col in out.columns:
            baseline_col = weighted_col.replace("_weighted_ssi", "_weighted_stabilized_ssi")
            delta_col = weighted_col.replace("_weighted_ssi", "_weighted_ssi_delta_vs_stabilized")
            if baseline_col == weighted_col:
                baseline_col = f"{weighted_col}_stabilized"
            if delta_col == weighted_col:
                delta_col = f"{weighted_col}_delta_vs_stabilized"
            out[baseline_col] = baseline_values_for_movie(weighted_baseline_by_image, out, stabilized_baseline)
            out[delta_col] = out[weighted_col] - out[baseline_col]
            row["weighted_stabilized_ssi_column"] = baseline_col
            row["weighted_ssi_delta_vs_stabilized_column"] = delta_col
        if mean_col in out.columns:
            baseline_col = mean_col.replace("_mean_ssi", "_mean_stabilized_ssi")
            delta_col = mean_col.replace("_mean_ssi", "_mean_ssi_delta_vs_stabilized")
            if baseline_col == mean_col:
                baseline_col = f"{mean_col}_stabilized"
            if delta_col == mean_col:
                delta_col = f"{mean_col}_delta_vs_stabilized"
            out[baseline_col] = baseline_values_for_movie(mean_baseline_by_image, out, stabilized_baseline)
            out[delta_col] = out[mean_col] - out[baseline_col]
            row["mean_stabilized_ssi_column"] = baseline_col
            row["mean_ssi_delta_vs_stabilized_column"] = delta_col
    return out, registry


def add_orientation_match_columns(
    movie: pd.DataFrame,
    ssi: np.ndarray,
    expected: np.ndarray | None,
    unit: pd.DataFrame,
    *,
    image_axis_col: str,
    match_max_deg: float,
    orthogonal_min_deg: float,
    stabilized_baseline: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = movie.copy()
    n_rows = out.shape[0]
    pref = pd.to_numeric(unit["analysis_preferred_orientation_deg"], errors="coerce").to_numpy(dtype=float)
    image_axis = pd.to_numeric(out[image_axis_col], errors="coerce").to_numpy(dtype=float)
    out["unit_image_orientation_match_mean_ssi"] = np.nan
    out["unit_image_orientation_match_weighted_ssi"] = np.nan
    out["unit_image_orientation_oblique_mean_ssi"] = np.nan
    out["unit_image_orientation_oblique_weighted_ssi"] = np.nan
    out["unit_image_orientation_orthogonal_mean_ssi"] = np.nan
    out["unit_image_orientation_orthogonal_weighted_ssi"] = np.nan
    out["n_unit_image_orientation_matched_units"] = 0
    out["n_unit_image_orientation_oblique_units"] = 0
    out["n_unit_image_orientation_orthogonal_units"] = 0
    baseline_ssi = stabilized_baseline["ssi"] if stabilized_baseline is not None else None
    baseline_expected = stabilized_baseline.get("expected") if stabilized_baseline is not None else None

    image_indices = out["image_index"].astype(int).to_numpy()
    row_numbers = np.arange(n_rows)
    for image_idx in sorted(pd.unique(out["image_index"].astype(int))):
        rows = row_numbers[image_indices == int(image_idx)]
        if rows.size == 0:
            continue
        axis = float(image_axis[rows[0]])
        if not math.isfinite(axis):
            continue
        delta = axis_delta_deg(pref, axis)
        valid = np.isfinite(delta) & np.isfinite(pref)
        masks = {
            "match": valid & (delta <= float(match_max_deg)),
            "orthogonal": valid & (delta >= float(orthogonal_min_deg)),
            "oblique": valid & (delta > float(match_max_deg)) & (delta < float(orthogonal_min_deg)),
        }
        for label, mask in masks.items():
            out.loc[rows, f"unit_image_orientation_{label}_mean_ssi"] = mean_bits_for_mask(ssi, mask, row_indices=rows)
            out.loc[rows, f"unit_image_orientation_{label}_weighted_ssi"] = weighted_bits_for_mask(
                ssi,
                expected,
                mask,
                row_indices=rows,
            )
            out.loc[rows, f"n_unit_image_orientation_{label}ed_units" if label == "match" else f"n_unit_image_orientation_{label}_units"] = int(
                np.count_nonzero(mask)
            )
            if stabilized_baseline is not None:
                baseline_row = baseline_rows_for_image_indices([int(image_idx)], stabilized_baseline)
                mean_baseline = mean_bits_for_mask(baseline_ssi, mask, row_indices=baseline_row)
                weighted_baseline = weighted_bits_for_mask(
                    baseline_ssi,
                    baseline_expected,
                    mask,
                    row_indices=baseline_row,
                )
                out.loc[rows, f"unit_image_orientation_{label}_mean_stabilized_ssi"] = float(mean_baseline[0])
                out.loc[rows, f"unit_image_orientation_{label}_weighted_stabilized_ssi"] = float(weighted_baseline[0])
                out.loc[rows, f"unit_image_orientation_{label}_mean_ssi_delta_vs_stabilized"] = (
                    out.loc[rows, f"unit_image_orientation_{label}_mean_ssi"] - float(mean_baseline[0])
                )
                out.loc[rows, f"unit_image_orientation_{label}_weighted_ssi_delta_vs_stabilized"] = (
                    out.loc[rows, f"unit_image_orientation_{label}_weighted_ssi"] - float(weighted_baseline[0])
                )
    out["unit_image_orientation_match_minus_orthogonal_weighted_ssi"] = (
        out["unit_image_orientation_match_weighted_ssi"] - out["unit_image_orientation_orthogonal_weighted_ssi"]
    )
    out["unit_image_orientation_match_minus_orthogonal_mean_ssi"] = (
        out["unit_image_orientation_match_mean_ssi"] - out["unit_image_orientation_orthogonal_mean_ssi"]
    )
    if stabilized_baseline is not None:
        out["unit_image_orientation_match_minus_orthogonal_weighted_stabilized_ssi"] = (
            out["unit_image_orientation_match_weighted_stabilized_ssi"]
            - out["unit_image_orientation_orthogonal_weighted_stabilized_ssi"]
        )
        out["unit_image_orientation_match_minus_orthogonal_mean_stabilized_ssi"] = (
            out["unit_image_orientation_match_mean_stabilized_ssi"]
            - out["unit_image_orientation_orthogonal_mean_stabilized_ssi"]
        )
        out["unit_image_orientation_match_minus_orthogonal_weighted_ssi_delta_vs_stabilized"] = (
            out["unit_image_orientation_match_minus_orthogonal_weighted_ssi"]
            - out["unit_image_orientation_match_minus_orthogonal_weighted_stabilized_ssi"]
        )
        out["unit_image_orientation_match_minus_orthogonal_mean_ssi_delta_vs_stabilized"] = (
            out["unit_image_orientation_match_minus_orthogonal_mean_ssi"]
            - out["unit_image_orientation_match_minus_orthogonal_mean_stabilized_ssi"]
        )
    return out


def add_trace_image_axis_columns(
    movie: pd.DataFrame,
    *,
    image_axis_col: str,
    trace_axis_col: str,
    match_max_deg: float,
    orthogonal_min_deg: float,
    min_anisotropy: float,
) -> pd.DataFrame:
    out = movie.copy()
    if image_axis_col not in out.columns or trace_axis_col not in out.columns:
        out["trace_image_axis_delta_deg"] = np.nan
        out["trace_image_axis_class"] = "unavailable"
        return out
    image_axis = pd.to_numeric(out[image_axis_col], errors="coerce").to_numpy(dtype=float)
    trace_axis = pd.to_numeric(out[trace_axis_col], errors="coerce").to_numpy(dtype=float)
    delta = axis_delta_deg(trace_axis, image_axis)
    anis = pd.to_numeric(out.get("rendered_cov_anisotropy", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(dtype=float)
    labels = np.full(out.shape[0], "oblique", dtype=object)
    labels[(~np.isfinite(delta)) | (~np.isfinite(trace_axis)) | (~np.isfinite(image_axis))] = "unavailable"
    labels[np.isfinite(anis) & (anis < float(min_anisotropy))] = "low_anisotropy"
    usable = (labels == "oblique") & np.isfinite(delta)
    labels[usable & (delta <= float(match_max_deg))] = "along_contour_axis"
    labels[usable & (delta >= float(orthogonal_min_deg))] = "across_contour_axis"
    out["trace_image_axis_delta_deg"] = delta
    out["trace_image_axis_class"] = labels
    return out


def prepare_movie_table(
    movie: pd.DataFrame,
    image: pd.DataFrame,
    trace: pd.DataFrame,
    unit: pd.DataFrame,
    ssi: np.ndarray,
    expected: np.ndarray | None,
    population: np.ndarray,
    args: argparse.Namespace,
    stabilized_baseline: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    bin_rows: list[dict[str, Any]] = []
    trace = trace.copy()
    trace["has_microsaccade"] = microsaccade_count(trace) > 0
    trace, rows = add_quantile_bin(
        trace,
        "rendered_path_length_arcmin",
        "trace_path_length_bin",
        n_bins=args.n_trace_bins,
        table_name="trace",
    )
    bin_rows.extend(rows)
    trace, rows = add_quantile_bin(
        trace,
        "rendered_diffusion_constant_arcmin2_s",
        "trace_diffusion_bin",
        n_bins=args.n_diffusion_bins,
        table_name="trace",
    )
    bin_rows.extend(rows)
    trace, rows = add_quantile_bin(
        trace,
        "rendered_cov_anisotropy",
        "trace_anisotropy_bin",
        n_bins=args.n_trace_bins,
        table_name="trace",
    )
    bin_rows.extend(rows)

    image = image.copy()
    image["image_contour_class"] = classify_image_contours(image)
    for source_col, out_col in [
        ("image_orientation_coherence", "image_orientation_coherence_bin"),
        ("image_patch_rms_contrast", "image_contrast_bin"),
        ("image_oriented_8plus_power_proxy", "image_oriented_8plus_power_bin"),
    ]:
        image, rows = add_quantile_bin(image, source_col, out_col, n_bins=args.n_image_bins, table_name="image")
        bin_rows.extend(rows)

    out = movie.copy()
    out["analysis_row_index"] = np.arange(out.shape[0], dtype=int)
    out["population_ssi"] = np.asarray(population, dtype=np.float32)
    out["has_microsaccade"] = microsaccade_count(out) > 0

    trace_join_cols = [
        "trace_bank_index",
        "has_microsaccade",
        "trace_path_length_bin",
        "trace_diffusion_bin",
        "trace_anisotropy_bin",
        "rendered_cov_anisotropy",
        "rendered_cov_axis_ratio",
        "rendered_cov_orientation_deg",
        "rendered_bcea68_arcmin2",
        "rendered_path_speed_arcmin_s",
        "rendered_speed_p95_arcmin_s",
    ]
    trace_join_cols = [col for col in trace_join_cols if col in trace.columns]
    out = out.merge(
        trace[trace_join_cols].drop_duplicates("trace_bank_index"),
        left_on="trace_index",
        right_on="trace_bank_index",
        how="left",
        suffixes=("", "_trace_table"),
        validate="many_to_one",
    )
    if "has_microsaccade_trace_table" in out.columns:
        out["has_microsaccade"] = out["has_microsaccade_trace_table"].fillna(out["has_microsaccade"]).astype(bool)
        out = out.drop(columns=["has_microsaccade_trace_table"])
    out = out.drop(columns=[col for col in ["trace_bank_index"] if col in out.columns])

    image_join_cols = [
        "image_index",
        "image_contour_class",
        "image_orientation_coherence_bin",
        "image_contrast_bin",
        "image_oriented_8plus_power_bin",
    ]
    image_join_cols = [col for col in image_join_cols if col in image.columns]
    out = out.merge(
        image[image_join_cols].drop_duplicates("image_index"),
        on="image_index",
        how="left",
        suffixes=("", "_image_table"),
        validate="many_to_one",
    )

    out, registry, plot_groups = add_unit_group_columns(out, ssi, expected, unit)
    out, registry = add_stabilized_baseline_columns(out, unit, registry, stabilized_baseline)
    if args.image_axis_col not in out.columns:
        fallback = "image_gradient_axis_deg"
        if fallback in out.columns:
            args.image_axis_col = fallback
        else:
            raise ValueError(f"Image axis column {args.image_axis_col!r} is missing and no fallback is available.")
    out = add_orientation_match_columns(
        out,
        ssi,
        expected,
        unit,
        image_axis_col=args.image_axis_col,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=float(args.orthogonal_min_deg),
        stabilized_baseline=stabilized_baseline,
    )
    out = add_trace_image_axis_columns(
        out,
        image_axis_col=args.image_axis_col,
        trace_axis_col=args.trace_axis_col,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=float(args.orthogonal_min_deg),
        min_anisotropy=float(args.trace_axis_min_anisotropy),
    )
    return out, image, trace, bin_rows, registry, plot_groups


def aggregate_metric(frame: pd.DataFrame, group_cols: list[str], metric: str, *, analysis: str) -> list[dict[str, Any]]:
    if metric not in frame.columns:
        return []
    use = frame[group_cols + [metric, "image_index", "trace_index"]].copy()
    use[metric] = pd.to_numeric(use[metric], errors="coerce")
    use = use[use[metric].notna()]
    if use.empty:
        return []
    rows: list[dict[str, Any]] = []
    for keys, sub in use.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec: dict[str, Any] = {
            "analysis": analysis,
            "metric": metric,
            "n_movies": int(sub.shape[0]),
            "n_images": int(sub["image_index"].nunique()) if "image_index" in sub.columns else 0,
            "n_traces": int(sub["trace_index"].nunique()) if "trace_index" in sub.columns else 0,
            "mean": float(sub[metric].mean()),
            "sem": sem(sub[metric]),
            "median": float(sub[metric].median()),
            "q25": float(sub[metric].quantile(0.25)),
            "q75": float(sub[metric].quantile(0.75)),
        }
        for col, value in zip(group_cols, keys):
            rec[col] = value
        rows.append(rec)
    return rows


def aggregate_many(frame: pd.DataFrame, specs: list[tuple[str, list[str], list[str]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis, group_cols, metrics in specs:
        if not all(col in frame.columns for col in group_cols):
            continue
        for metric in metrics:
            rows.extend(aggregate_metric(frame, group_cols, metric, analysis=analysis))
    return pd.DataFrame(rows)


def summarize_by_entity(movie: pd.DataFrame, entity_col: str, entity_table: pd.DataFrame, entity_index_col: str, metric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entity_index, sub in movie.groupby(entity_col, sort=True):
        rec: dict[str, Any] = {
            entity_index_col: int(entity_index),
            "n_movies": int(sub.shape[0]),
            "population_ssi_mean": float(sub["population_ssi"].mean()),
            "population_ssi_sem": sem(sub["population_ssi"]),
            "population_ssi_median": float(sub["population_ssi"].median()),
        }
        for col in metric_cols:
            if col in sub.columns:
                rec[f"{col}_mean"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        rows.append(rec)
    summary = pd.DataFrame(rows)
    return entity_table.merge(summary, on=entity_index_col, how="left", validate="one_to_one")


def unit_summary_table(
    ssi: np.ndarray,
    expected: np.ndarray | None,
    movie: pd.DataFrame,
    unit: pd.DataFrame,
    stabilized_baseline: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = unit.copy()
    out["unit_ssi_mean_all_movies"] = np.nanmean(ssi, axis=0)
    out["unit_ssi_median_all_movies"] = np.nanmedian(ssi, axis=0)
    if expected is not None:
        exp = np.asarray(expected, dtype=np.float64)
        out["unit_expected_spikes_all_movies"] = np.sum(exp, axis=0)
    if stabilized_baseline is not None:
        baseline_ssi = np.asarray(stabilized_baseline["ssi"], dtype=np.float64)
        out["unit_stabilized_ssi_mean_selected_images"] = np.nanmean(baseline_ssi, axis=0)
        out["unit_stabilized_ssi_median_selected_images"] = np.nanmedian(baseline_ssi, axis=0)
        out["unit_ssi_mean_delta_all_movies_vs_stabilized"] = (
            out["unit_ssi_mean_all_movies"] - out["unit_stabilized_ssi_mean_selected_images"]
        )
        baseline_expected = stabilized_baseline.get("expected")
        if baseline_expected is not None:
            out["unit_stabilized_expected_spikes_selected_images"] = np.sum(np.asarray(baseline_expected, dtype=np.float64), axis=0)
    masks = {
        "no_microsaccade": ~movie["has_microsaccade"].astype(bool).to_numpy(),
        "microsaccade": movie["has_microsaccade"].astype(bool).to_numpy(),
        "strong_contour_images": movie["image_contour_class"].astype(str).eq("strong").to_numpy(),
    }
    for label, mask in masks.items():
        if np.any(mask):
            out[f"unit_ssi_mean_{label}"] = np.nanmean(ssi[mask], axis=0)
        else:
            out[f"unit_ssi_mean_{label}"] = np.nan
    out["unit_ssi_delta_microsaccade_minus_no_microsaccade"] = (
        out["unit_ssi_mean_microsaccade"] - out["unit_ssi_mean_no_microsaccade"]
    )
    return out


def metric_columns_for_summaries(movie: pd.DataFrame, registry: list[dict[str, Any]]) -> list[str]:
    metrics = ["population_ssi"]
    for col in ["stabilized_population_ssi", "population_ssi_delta_vs_stabilized"]:
        if col in movie.columns:
            metrics.append(col)
    for row in registry:
        for key in ("weighted_ssi_column", "weighted_stabilized_ssi_column", "weighted_ssi_delta_vs_stabilized_column"):
            col = str(row.get(key, ""))
            if col in movie.columns:
                metrics.append(col)
    for col in [
        "unit_image_orientation_match_weighted_ssi",
        "unit_image_orientation_oblique_weighted_ssi",
        "unit_image_orientation_orthogonal_weighted_ssi",
        "unit_image_orientation_match_minus_orthogonal_weighted_ssi",
        "unit_image_orientation_match_weighted_stabilized_ssi",
        "unit_image_orientation_oblique_weighted_stabilized_ssi",
        "unit_image_orientation_orthogonal_weighted_stabilized_ssi",
        "unit_image_orientation_match_minus_orthogonal_weighted_stabilized_ssi",
        "unit_image_orientation_match_weighted_ssi_delta_vs_stabilized",
        "unit_image_orientation_oblique_weighted_ssi_delta_vs_stabilized",
        "unit_image_orientation_orthogonal_weighted_ssi_delta_vs_stabilized",
        "unit_image_orientation_match_minus_orthogonal_weighted_ssi_delta_vs_stabilized",
    ]:
        if col in movie.columns:
            metrics.append(col)
    return list(dict.fromkeys(metrics))


def qc_rows(
    ssi: np.ndarray,
    expected: np.ndarray | None,
    population: np.ndarray,
    movie: pd.DataFrame,
    image: pd.DataFrame,
    trace: pd.DataFrame,
    unit: pd.DataFrame,
    stabilized_baseline: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        [
            {"metric": "n_movies", "value": int(movie.shape[0])},
            {"metric": "n_images", "value": int(image.shape[0])},
            {"metric": "n_traces", "value": int(trace.shape[0])},
            {"metric": "n_units", "value": int(unit.shape[0])},
            {"metric": "ssi_matrix_rows", "value": int(ssi.shape[0])},
            {"metric": "ssi_matrix_cols", "value": int(ssi.shape[1])},
            {"metric": "expected_spikes_available", "value": bool(expected is not None)},
            {"metric": "movie_index_unique", "value": int(movie["movie_index"].nunique()) if "movie_index" in movie.columns else 0},
            {"metric": "movie_index_monotonic", "value": bool(movie["movie_index"].is_monotonic_increasing) if "movie_index" in movie.columns else False},
            {
                "metric": "matrix_row_index_matches_row_order",
                "value": bool((movie["matrix_row_index"].to_numpy() == np.arange(movie.shape[0])).all())
                if "matrix_row_index" in movie.columns
                else False,
            },
            {"metric": "population_ssi_min", "value": float(np.nanmin(population))},
            {"metric": "population_ssi_median", "value": float(np.nanmedian(population))},
            {"metric": "population_ssi_max", "value": float(np.nanmax(population))},
            {"metric": "stabilized_baseline_available", "value": bool(stabilized_baseline is not None)},
            {"metric": "n_microsaccade_traces", "value": int(trace["has_microsaccade"].sum()) if "has_microsaccade" in trace.columns else 0},
            {"metric": "n_microsaccade_movies", "value": int(movie["has_microsaccade"].sum()) if "has_microsaccade" in movie.columns else 0},
            {
                "metric": "n_strong_contour_images",
                "value": int(image["image_contour_class"].astype(str).eq("strong").sum()) if "image_contour_class" in image.columns else 0,
            },
        ]
    )
    for col in ["rendered_path_length_arcmin", "rendered_diffusion_constant_arcmin2_s", "rendered_cov_anisotropy"]:
        if col in trace.columns:
            vals = pd.to_numeric(trace[col], errors="coerce").dropna()
            rows.extend(
                [
                    {"metric": f"{col}_min", "value": float(vals.min())},
                    {"metric": f"{col}_median", "value": float(vals.median())},
                    {"metric": f"{col}_max", "value": float(vals.max())},
                ]
            )
    if stabilized_baseline is not None:
        baseline_population = stabilized_baseline.get("population")
        if baseline_population is not None:
            rows.extend(
                [
                    {"metric": "stabilized_population_ssi_min", "value": float(np.nanmin(baseline_population))},
                    {"metric": "stabilized_population_ssi_median", "value": float(np.nanmedian(baseline_population))},
                    {"metric": "stabilized_population_ssi_max", "value": float(np.nanmax(baseline_population))},
                ]
            )
        if "population_ssi_delta_vs_stabilized" in movie.columns:
            vals = pd.to_numeric(movie["population_ssi_delta_vs_stabilized"], errors="coerce").dropna()
            rows.extend(
                [
                    {"metric": "population_ssi_delta_vs_stabilized_min", "value": float(vals.min())},
                    {"metric": "population_ssi_delta_vs_stabilized_median", "value": float(vals.median())},
                    {"metric": "population_ssi_delta_vs_stabilized_max", "value": float(vals.max())},
                ]
            )
    return rows


def bin_x(frame: pd.DataFrame, bin_col: str, x_col: str) -> pd.Series:
    return frame.groupby(bin_col, sort=True)[x_col].median()


def trace_microsaccade_path_context_from_frame(
    frame: pd.DataFrame,
    *,
    source_label: str,
    source_path: Path | None = None,
) -> pd.DataFrame:
    if "rendered_path_length_arcmin" not in frame.columns:
        return pd.DataFrame()
    cols = ["rendered_path_length_arcmin"]
    if "trace_index" in frame.columns:
        cols.append("trace_index")
    if "trace_bank_index" in frame.columns:
        cols.append("trace_bank_index")
    for col in ("has_microsaccade", "rendered_n_microsaccade_events", "n_microsaccade_events", "source_n_microsaccade_events"):
        if col in frame.columns and col not in cols:
            cols.append(col)
    work = frame[cols].copy()
    if "has_microsaccade" not in work.columns:
        work["has_microsaccade"] = microsaccade_count(work) > 0
    if "trace_index" in work.columns:
        work = work.drop_duplicates("trace_index")
    elif "trace_bank_index" in work.columns:
        work = work.drop_duplicates("trace_bank_index")
    work["rendered_path_length_arcmin"] = pd.to_numeric(work["rendered_path_length_arcmin"], errors="coerce")
    work["has_microsaccade"] = work["has_microsaccade"].fillna(False).astype(bool)
    work = work.dropna(subset=["rendered_path_length_arcmin"])
    if work.empty:
        return pd.DataFrame()
    rows = []
    for has_microsaccade, label, display_label in [
        (False, "no_microsaccade", "drift-only"),
        (True, "microsaccade", "microsaccade"),
    ]:
        sub = work.loc[work["has_microsaccade"].eq(has_microsaccade)]
        values = sub["rendered_path_length_arcmin"].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        rows.append(
            {
                "trace_path_context": label,
                "display_label": display_label,
                "context_source_label": source_label,
                "context_source_path": str(source_path) if source_path is not None else "",
                "has_microsaccade": bool(has_microsaccade),
                "n_traces": int(values.size),
                "low_arcmin": float(np.nanmin(values)),
                "q25_arcmin": float(np.nanpercentile(values, 25.0)),
                "q40_arcmin": float(np.nanpercentile(values, 40.0)),
                "median_arcmin": float(np.nanmedian(values)),
                "q60_arcmin": float(np.nanpercentile(values, 60.0)),
                "q75_arcmin": float(np.nanpercentile(values, 75.0)),
                "high_arcmin": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(rows)


def trace_microsaccade_path_context_from_movie(movie: pd.DataFrame) -> pd.DataFrame:
    return trace_microsaccade_path_context_from_frame(movie, source_label="ssi_matrix_traces")


def load_trace_path_context_reference(movie: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.trace_path_context_reference_csv) if args.trace_path_context_reference_csv is not None else None
    if path is not None and path.exists():
        reference = pd.read_csv(path)
        context = trace_microsaccade_path_context_from_frame(
            reference,
            source_label="large_fixation_sample_pathle350arcmin",
            source_path=path,
        )
        if not context.empty:
            return context
    return trace_microsaccade_path_context_from_movie(movie)


def add_trace_path_context_bands(ax: Any, context: pd.DataFrame | None, *, include_legend: bool = True) -> None:
    if context is None or context.empty:
        return
    styles = {
        "no_microsaccade": {
            "color": "#8c8c8c",
            "alpha": 0.24,
            "line_alpha": 0.62,
            "linestyle": "-",
            "y0": 0.940,
            "y1": 0.982,
        },
        "microsaccade": {
            "color": "#5f5f5f",
            "alpha": 0.20,
            "line_alpha": 0.70,
            "linestyle": "--",
            "y0": 0.888,
            "y1": 0.930,
        },
    }
    strip_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for row in context.sort_values("median_arcmin").itertuples(index=False):
        key = str(getattr(row, "trace_path_context", "trace_path"))
        style = styles.get(
            key,
            {"color": "#7f7f7f", "alpha": 0.20, "line_alpha": 0.60, "linestyle": "-", "y0": 0.912, "y1": 0.970},
        )
        low = float(getattr(row, "q25_arcmin", getattr(row, "low_arcmin", np.nan)))
        high = float(getattr(row, "q75_arcmin", getattr(row, "high_arcmin", np.nan)))
        median = float(getattr(row, "median_arcmin", np.nan))
        if not (math.isfinite(low) and math.isfinite(high) and high > low):
            continue
        display = str(getattr(row, "display_label", key)).replace("_", " ")
        n_traces = int(getattr(row, "n_traces", 0))
        y0 = float(style["y0"])
        y1 = float(style["y1"])
        rect = patches.Rectangle(
            (low, y0),
            high - low,
            y1 - y0,
            transform=strip_transform,
            facecolor=style["color"],
            edgecolor="none",
            alpha=style["alpha"],
            zorder=3,
            label=f"{display} ref q25-q75 (n={n_traces})" if include_legend else None,
        )
        ax.add_patch(rect)
        if math.isfinite(median):
            ax.plot(
                [median, median],
                [y0, y1],
                transform=strip_transform,
                color=style["color"],
                alpha=style["line_alpha"],
                linewidth=1.1,
                linestyle=style["linestyle"],
                zorder=4,
            )


def plot_phase1_qc(image: pd.DataFrame, trace: pd.DataFrame, unit: pd.DataFrame, fig_dir: Path) -> dict[str, str]:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    trace_ms = trace["has_microsaccade"].astype(bool).to_numpy() if "has_microsaccade" in trace.columns else np.zeros(trace.shape[0], bool)

    ax = axes[0, 0]
    for mask, color, label in [(~trace_ms, NO_MS_COLOR, "no microsaccade"), (trace_ms, MS_COLOR, "microsaccade")]:
        vals = pd.to_numeric(trace.loc[mask, "rendered_path_length_arcmin"], errors="coerce").dropna()
        ax.hist(vals, bins=35, histtype="step", density=True, lw=1.8, color=color, label=label)
    ax.set_xlabel("path length (arcmin)")
    ax.set_ylabel("density")
    ax.set_title("Trace path length")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for mask, color, label in [(~trace_ms, NO_MS_COLOR, "no microsaccade"), (trace_ms, MS_COLOR, "microsaccade")]:
        vals = pd.to_numeric(trace.loc[mask, "rendered_diffusion_constant_arcmin2_s"], errors="coerce").dropna()
        vals = vals[vals >= 0]
        ax.hist(np.log10(vals + EPS), bins=35, histtype="step", density=True, lw=1.8, color=color, label=label)
    ax.set_xlabel("log10 diffusion constant")
    ax.set_ylabel("density")
    ax.set_title("Trace diffusion")

    ax = axes[0, 2]
    vals = pd.to_numeric(trace.get("rendered_cov_anisotropy", pd.Series(dtype=float)), errors="coerce").dropna()
    ax.hist(vals, bins=35, color="#7f7f7f", alpha=0.8)
    ax.set_xlabel("covariance anisotropy")
    ax.set_ylabel("count")
    ax.set_title("Trace covariance shape")

    ax = axes[1, 0]
    vals = pd.to_numeric(image.get("image_orientation_coherence", pd.Series(dtype=float)), errors="coerce").dropna()
    ax.hist(vals, bins=25, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("orientation coherence")
    ax.set_ylabel("count")
    ax.set_title("Image contour coherence")

    ax = axes[1, 1]
    vals = pd.to_numeric(image.get("image_patch_rms_contrast", pd.Series(dtype=float)), errors="coerce").dropna()
    ax.hist(vals, bins=25, color="#f28e2b", alpha=0.85)
    ax.set_xlabel("RMS contrast")
    ax.set_ylabel("count")
    ax.set_title("Image contrast")

    ax = axes[1, 2]
    counts = unit.get("sf_group", pd.Series("unknown", index=unit.index)).astype(str).value_counts()
    counts = counts.reindex(ordered_groups("sf_group", counts.index.to_list()))
    ax.bar(np.arange(len(counts)), counts.to_numpy(), color=[color_for_label(group, idx) for idx, group in enumerate(counts.index)])
    ax.set_xticks(np.arange(len(counts)), counts.index, rotation=25, ha="right")
    ax.set_ylabel("units")
    ax.set_title("Unit SF groups")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, fig_dir, "phase1_feature_qc_distributions")


def plot_lines_by_trace_bin(
    movie: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    *,
    title: str,
    ylabel: str,
    fig_dir: Path,
    stem: str,
    filter_mask: np.ndarray | None = None,
    trace_path_context: pd.DataFrame | None = None,
) -> dict[str, str]:
    use = movie.copy() if filter_mask is None else movie.loc[filter_mask].copy()
    if use.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.text(0.5, 0.5, "No rows available", ha="center", va="center")
        return save_figure(fig, fig_dir, stem)
    x = bin_x(use, "trace_path_length_bin", "rendered_path_length_arcmin")
    context = trace_path_context if trace_path_context is not None else trace_microsaccade_path_context_from_movie(use)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    add_trace_path_context_bands(ax, context)
    for idx, (col, label) in enumerate(zip(columns, labels)):
        if col not in use.columns:
            continue
        means = use.groupby("trace_path_length_bin", sort=True)[col].mean()
        errs = use.groupby("trace_path_length_bin", sort=True)[col].apply(sem)
        common = [key for key in x.index if key in means.index]
        ax.errorbar(
            x.loc[common].to_numpy(dtype=float),
            means.loc[common].to_numpy(dtype=float),
            yerr=errs.loc[common].to_numpy(dtype=float),
            marker="o",
            lw=1.8,
            capsize=2,
            color=color_for_label(label, idx),
            label=label,
        )
    ax.set_xlabel("trace path length bin median (arcmin)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, fig_dir, stem)


def labels_for_columns(columns: list[str], prefix: str) -> list[str]:
    labels = []
    for col in columns:
        text = col
        text = text.removeprefix(prefix + "_")
        text = text.removesuffix("_weighted_ssi")
        text = text.replace("_", " ")
        labels.append(text)
    return labels


def contour_matched_definition_text(unit: pd.DataFrame, sf_groups: list[str]) -> str:
    if "sf_group_definition" in unit.columns:
        definitions = unit.loc[unit["sf_group"].astype(str).isin(sf_groups), "sf_group_definition"].dropna().astype(str).unique()
        if len(definitions):
            definition = str(definitions[0])
            if "low_sf <= 0.05" in definition and "high_sf >= 0.5" in definition:
                return "low SF <= 0.05 cpd vs high SF >= 0.5 cpd"
            return definition
    return "low SF vs high SF"


def build_real_trace_sf_contour_matched_unit_curves(
    movie: pd.DataFrame,
    ssi: np.ndarray,
    unit: pd.DataFrame,
    *,
    image_axis_col: str,
    sf_groups: list[str],
    match_max_deg: float,
    orthogonal_min_deg: float,
    min_orientation_selectivity: float,
    min_matched_images_per_unit: int,
    contour_relation: str = "matched",
    stabilized_baseline: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_movie = {"image_index", "trace_path_length_bin", "rendered_path_length_arcmin", image_axis_col}
    missing_movie = sorted(required_movie.difference(movie.columns))
    if missing_movie:
        raise ValueError(f"Movie table is missing columns needed for contour-matched curves: {missing_movie}")
    required_unit = {"unit_index", "unit_label", "sf_group", "analysis_preferred_orientation_deg", "prior_orientation_selectivity_index"}
    missing_unit = sorted(required_unit.difference(unit.columns))
    if missing_unit:
        raise ValueError(f"Unit table is missing columns needed for contour-matched curves: {missing_unit}")

    image_axis = (
        movie[["image_index", image_axis_col]]
        .drop_duplicates("image_index")
        .sort_values("image_index")
        .reset_index(drop=True)
    )
    image_axis[image_axis_col] = pd.to_numeric(image_axis[image_axis_col], errors="coerce")
    trace_bins = (
        movie[["trace_path_length_bin", "rendered_path_length_arcmin"]]
        .dropna(subset=["trace_path_length_bin"])
        .groupby("trace_path_length_bin", sort=True)["rendered_path_length_arcmin"]
        .median()
        .sort_values()
    )
    if trace_bins.empty:
        raise ValueError("No trace path-length bins are available.")
    reference_bin = str(trace_bins.index[0])

    movie_bins = movie["trace_path_length_bin"].astype(str).to_numpy()
    movie_images = movie["image_index"].astype(int).to_numpy()
    row_index = np.arange(movie.shape[0], dtype=int)
    baseline_ssi = stabilized_baseline["ssi"] if stabilized_baseline is not None else None
    selection_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    relation_key = sanitize(contour_relation)
    if relation_key in {"match", "matched", "aligned", "contour_matched"}:
        relation_key = "matched"
        relation_label = "matched"
        criterion_text = f"delta <= {match_max_deg:g} deg"
    elif relation_key in {"orthogonal", "contour_orthogonal"}:
        relation_key = "orthogonal"
        relation_label = "orthogonal"
        criterion_text = f"delta >= {orthogonal_min_deg:g} deg"
    elif relation_key in {"all", "all_orientation", "all_oriented", "orientation_tuned", "oriented"}:
        relation_key = "all_orientation"
        relation_label = "all orientation-tuned"
        criterion_text = f"all image windows; OSI >= {min_orientation_selectivity:g}"
    else:
        raise ValueError(f"Unsupported contour_relation {contour_relation!r}; expected matched, orthogonal, or all_orientation.")

    use_units = unit[unit["sf_group"].astype(str).isin(sf_groups)].copy()
    sf_order = {group: idx for idx, group in enumerate(sf_groups)}
    use_units["_sf_order"] = use_units["sf_group"].astype(str).map(sf_order).fillna(len(sf_order))
    use_units = use_units.sort_values(["_sf_order", "unit_index"], kind="mergesort")

    for unit_row in use_units.itertuples(index=False):
        unit_index = int(unit_row.unit_index)
        if unit_index < 0 or unit_index >= int(ssi.shape[1]):
            continue
        sf_group = str(unit_row.sf_group)
        pref = float(unit_row.analysis_preferred_orientation_deg)
        osi = float(unit_row.prior_orientation_selectivity_index)
        orientation_ok = math.isfinite(pref) and math.isfinite(osi) and osi >= float(min_orientation_selectivity)
        if orientation_ok:
            deltas = axis_delta_deg(image_axis[image_axis_col].to_numpy(dtype=float), pref)
            if relation_key == "matched":
                matched_image_mask = np.isfinite(deltas) & (deltas <= float(match_max_deg))
            elif relation_key == "orthogonal":
                matched_image_mask = np.isfinite(deltas) & (deltas >= float(orthogonal_min_deg))
            else:
                matched_image_mask = np.isfinite(deltas)
        else:
            deltas = np.full(image_axis.shape[0], np.nan)
            matched_image_mask = np.zeros(image_axis.shape[0], dtype=bool)
        matched_images = image_axis.loc[matched_image_mask, "image_index"].astype(int).to_numpy()
        selection_rows.append(
            {
                "unit_index": unit_index,
                "unit_label": str(unit_row.unit_label),
                "contour_relation": relation_key,
                "contour_relation_label": relation_label,
                "contour_relation_criterion": criterion_text,
                "sf_group": sf_group,
                "sf_group_label": str(getattr(unit_row, "sf_group_label", sf_group)),
                "preferred_orientation_deg": pref,
                "prior_orientation_selectivity_index": osi,
                "passes_orientation_selectivity": bool(orientation_ok),
                "n_matched_images": int(matched_images.size),
                "fraction_matched_images": float(matched_images.size / image_axis.shape[0]) if image_axis.shape[0] else float("nan"),
                "mean_delta_from_contour_deg": float(np.nanmean(deltas[matched_image_mask])) if np.any(matched_image_mask) else float("nan"),
                "median_delta_from_contour_deg": float(np.nanmedian(deltas[matched_image_mask])) if np.any(matched_image_mask) else float("nan"),
                "matched_image_indices": " ".join(str(int(idx)) for idx in matched_images),
            }
        )
        if matched_images.size < int(min_matched_images_per_unit):
            continue
        image_ok = np.isin(movie_images, matched_images)
        if stabilized_baseline is not None and matched_images.size:
            baseline_rows = baseline_rows_for_image_indices(matched_images, stabilized_baseline)
            baseline_values = np.asarray(baseline_ssi[baseline_rows, unit_index], dtype=float)
            baseline_values = baseline_values[np.isfinite(baseline_values)]
            stabilized_reference = float(np.nanmean(baseline_values)) if baseline_values.size else float("nan")
        else:
            stabilized_reference = float("nan")
        per_bin_values: dict[str, dict[str, Any]] = {}
        for bin_label, bin_median_path in trace_bins.items():
            bin_key = str(bin_label)
            rows = row_index[image_ok & (movie_bins == bin_key)]
            values = np.asarray(ssi[rows, unit_index], dtype=float) if rows.size else np.asarray([], dtype=float)
            values = values[np.isfinite(values)]
            per_bin_values[bin_key] = {
                "trace_path_length_bin": bin_key,
                "trace_path_length_bin_median_arcmin": float(bin_median_path),
                "n_unit_window_samples": int(values.size),
                "unit_contour_matched_stabilized_ssi_bits_per_spike": stabilized_reference,
                "unit_contour_matched_ssi_bits_per_spike": float(np.nanmean(values)) if values.size else float("nan"),
            }
        reference = float(per_bin_values[reference_bin]["unit_contour_matched_ssi_bits_per_spike"])
        for bin_key, record in per_bin_values.items():
            value = float(record["unit_contour_matched_ssi_bits_per_spike"])
            curve_rows.append(
                {
                    "unit_index": unit_index,
                    "unit_label": str(unit_row.unit_label),
                    "contour_relation": relation_key,
                    "contour_relation_label": relation_label,
                    "contour_relation_criterion": criterion_text,
                    "sf_group": sf_group,
                    "sf_group_label": str(getattr(unit_row, "sf_group_label", sf_group)),
                    "sf_group_definition": str(getattr(unit_row, "sf_group_definition", "")),
                    "sf_split_metric": float(getattr(unit_row, "sf_split_metric", float("nan"))),
                    "preferred_orientation_deg": pref,
                    "prior_orientation_selectivity_index": osi,
                    "n_matched_images": int(matched_images.size),
                    "mean_delta_from_contour_deg": selection_rows[-1]["mean_delta_from_contour_deg"],
                    "reference_trace_path_length_bin": reference_bin,
                    "unit_contour_matched_ssi_at_reference": reference,
                    "unit_contour_matched_ssi_delta_vs_reference": value - reference if math.isfinite(value) and math.isfinite(reference) else float("nan"),
                    "unit_contour_matched_ssi_delta_vs_stabilized": value - stabilized_reference
                    if math.isfinite(value) and math.isfinite(stabilized_reference)
                    else float("nan"),
                    **record,
                }
            )

    selection = pd.DataFrame(selection_rows)
    curves = pd.DataFrame(curve_rows)
    if curves.empty:
        raise ValueError("No real-trace contour-matched unit curves survived the support filters.")

    rows: list[dict[str, Any]] = []
    value_cols = [
        "unit_contour_matched_ssi_bits_per_spike",
        "unit_contour_matched_ssi_delta_vs_reference",
    ]
    if "unit_contour_matched_stabilized_ssi_bits_per_spike" in curves.columns:
        value_cols.append("unit_contour_matched_stabilized_ssi_bits_per_spike")
    if "unit_contour_matched_ssi_delta_vs_stabilized" in curves.columns:
        value_cols.append("unit_contour_matched_ssi_delta_vs_stabilized")
    for (sf_group, bin_label), sub in curves.groupby(["sf_group", "trace_path_length_bin"], sort=False):
        for col in value_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            rows.append(
                {
                    "sf_group": str(sf_group),
                    "sf_group_label": str(sub["sf_group_label"].iloc[0]),
                    "trace_path_length_bin": str(bin_label),
                    "trace_path_length_bin_median_arcmin": float(sub["trace_path_length_bin_median_arcmin"].iloc[0]),
                    "reference_trace_path_length_bin": reference_bin,
                    "value_name": col,
                    "n_units": int(sub.loc[finite, "unit_index"].nunique()),
                    "n_finite": int(finite.sum()),
                    "mean": float(np.nanmean(values)),
                    "sem": sem(values),
                    "median": float(np.nanmedian(values)),
                    "mean_matched_images_per_unit": float(np.nanmean(sub["n_matched_images"].to_numpy(dtype=float))),
                    "median_matched_images_per_unit": float(np.nanmedian(sub["n_matched_images"].to_numpy(dtype=float))),
                    "mean_unit_window_samples": float(np.nanmean(sub["n_unit_window_samples"].to_numpy(dtype=float))),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["value_name", "trace_path_length_bin_median_arcmin", "sf_group"]).reset_index(drop=True)
    return selection, curves, summary


def plot_real_trace_sf_contour_matched_figure(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    unit: pd.DataFrame,
    fig_dir: Path,
    *,
    sf_groups: list[str],
    match_max_deg: float,
    orthogonal_min_deg: float,
    min_orientation_selectivity: float,
    min_matched_images_per_unit: int,
    image_axis_col: str,
    trace_path_context: pd.DataFrame | None = None,
    contour_relation: str = "matched",
) -> dict[str, str]:
    relation_key = sanitize(contour_relation)
    if relation_key in {"match", "matched", "aligned", "contour_matched"}:
        relation_key = "matched"
        relation_title = "Contour-matched"
        criterion_text = f"align <= {match_max_deg:g} deg"
        output_stem = "phase2_real_trace_sf_contour_matched_low_high_scale_curves"
    elif relation_key in {"orthogonal", "contour_orthogonal"}:
        relation_key = "orthogonal"
        relation_title = "Contour-orthogonal"
        criterion_text = f"orthogonal >= {orthogonal_min_deg:g} deg"
        window_title = "Absolute SSI on orthogonal windows"
        output_stem = "phase2_real_trace_sf_contour_orthogonal_low_high_scale_curves"
    elif relation_key in {"all", "all_orientation", "all_oriented", "orientation_tuned", "oriented"}:
        relation_key = "all_orientation"
        relation_title = "All orientation-tuned"
        criterion_text = "all image windows"
        window_title = "Absolute SSI across all windows"
        output_stem = "phase2_real_trace_sf_all_orientation_units_low_high_scale_curves"
    else:
        raise ValueError(f"Unsupported contour_relation {contour_relation!r}; expected matched, orthogonal, or all_orientation.")
    if relation_key == "matched":
        window_title = "Absolute SSI on matched windows"
    has_stabilized_delta = summary["value_name"].astype(str).eq("unit_contour_matched_ssi_delta_vs_stabilized").any()
    delta_value_name = (
        "unit_contour_matched_ssi_delta_vs_stabilized"
        if has_stabilized_delta
        else "unit_contour_matched_ssi_delta_vs_reference"
    )
    delta_ylabel = (
        "SSI minus stabilized baseline (bits/spike)"
        if has_stabilized_delta
        else "SSI minus smallest trace bin (bits/spike)"
    )
    panels = [
        (
            "unit_contour_matched_ssi_bits_per_spike",
            "SSI (bits/spike)",
            window_title,
        ),
        (
            delta_value_name,
            delta_ylabel,
            "Movement modulation",
        ),
    ]
    definition_text = contour_matched_definition_text(unit, sf_groups)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True)
    for ax, (value_name, ylabel, title) in zip(axes, panels, strict=True):
        add_trace_path_context_bands(ax, trace_path_context)
        for sf_group in sf_groups:
            mean_sub = summary[
                summary["sf_group"].astype(str).eq(str(sf_group))
                & summary["value_name"].astype(str).eq(value_name)
            ].sort_values("trace_path_length_bin_median_arcmin")
            if mean_sub.empty:
                continue
            x = mean_sub["trace_path_length_bin_median_arcmin"].to_numpy(dtype=float)
            y = mean_sub["mean"].to_numpy(dtype=float)
            e = mean_sub["sem"].to_numpy(dtype=float)
            color = color_for_label(sf_group)
            label = f"{str(sf_group).replace('_', ' ')} (n={int(mean_sub['n_units'].iloc[0])})"
            ax.plot(x, y, marker="o", linewidth=2.3, markersize=4.8, color=color, label=label, zorder=4)
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.16, linewidth=0, zorder=2)
        if value_name in {"unit_contour_matched_ssi_delta_vs_reference", "unit_contour_matched_ssi_delta_vs_stabilized"}:
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("trace path length bin median (arcmin)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{relation_title} unit-window pairs: real trace scale\n"
        f"{definition_text}; axis={image_axis_col}; {criterion_text}; "
        f"OSI >= {min_orientation_selectivity:g}; min images/unit = {min_matched_images_per_unit}",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return save_figure(fig, fig_dir, output_stem)


def plot_real_trace_sf_contour_relation_overlay_figure(
    matched_summary: pd.DataFrame,
    orthogonal_summary: pd.DataFrame,
    unit: pd.DataFrame,
    fig_dir: Path,
    *,
    sf_groups: list[str],
    match_max_deg: float,
    orthogonal_min_deg: float,
    min_orientation_selectivity: float,
    min_matched_images_per_unit: int,
    image_axis_col: str,
    trace_path_context: pd.DataFrame | None = None,
) -> dict[str, str]:
    has_stabilized_delta = (
        matched_summary["value_name"].astype(str).eq("unit_contour_matched_ssi_delta_vs_stabilized").any()
        and orthogonal_summary["value_name"].astype(str).eq("unit_contour_matched_ssi_delta_vs_stabilized").any()
    )
    delta_value_name = (
        "unit_contour_matched_ssi_delta_vs_stabilized"
        if has_stabilized_delta
        else "unit_contour_matched_ssi_delta_vs_reference"
    )
    delta_ylabel = (
        "SSI minus stabilized baseline (bits/spike)"
        if has_stabilized_delta
        else "SSI minus smallest trace bin (bits/spike)"
    )
    panels = [
        (
            "unit_contour_matched_ssi_bits_per_spike",
            "SSI (bits/spike)",
            "Absolute SSI on selected windows",
        ),
        (
            delta_value_name,
            delta_ylabel,
            "Movement modulation",
        ),
    ]
    relation_specs = [
        ("aligned", matched_summary, "-", "o"),
        ("orthogonal", orthogonal_summary, "--", "s"),
    ]
    definition_text = contour_matched_definition_text(unit, sf_groups)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    for ax, (value_name, ylabel, title) in zip(axes, panels, strict=True):
        add_trace_path_context_bands(ax, trace_path_context, include_legend=False)
        for relation_label, summary, linestyle, marker in relation_specs:
            for sf_group in sf_groups:
                mean_sub = summary[
                    summary["sf_group"].astype(str).eq(str(sf_group))
                    & summary["value_name"].astype(str).eq(value_name)
                ].sort_values("trace_path_length_bin_median_arcmin")
                if mean_sub.empty:
                    continue
                x = mean_sub["trace_path_length_bin_median_arcmin"].to_numpy(dtype=float)
                y = mean_sub["mean"].to_numpy(dtype=float)
                e = mean_sub["sem"].to_numpy(dtype=float)
                color = color_for_label(sf_group)
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.3,
                    markersize=4.8,
                    color=color,
                    zorder=5,
                )
                ax.fill_between(
                    x,
                    y - e,
                    y + e,
                    color=color,
                    alpha=0.11 if relation_label == "aligned" else 0.07,
                    linewidth=0,
                    zorder=2 if relation_label == "aligned" else 1,
                )
        if value_name in {"unit_contour_matched_ssi_delta_vs_reference", "unit_contour_matched_ssi_delta_vs_stabilized"}:
            ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("trace path length bin median (arcmin)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    color_handles = [
        Line2D([0], [0], color=color_for_label(sf_group), lw=2.5, label=str(sf_group).replace("_", " "))
        for sf_group in sf_groups
    ]
    relation_handles = [
        Line2D([0], [0], color="0.25", lw=2.2, linestyle="-", marker="o", markersize=4.5, label="aligned"),
        Line2D([0], [0], color="0.25", lw=2.2, linestyle="--", marker="s", markersize=4.5, label="orthogonal"),
    ]
    reference_handles = [
        patches.Patch(facecolor="#8c8c8c", alpha=0.24, edgecolor="none", label="drift-only ref IQR"),
        patches.Patch(facecolor="#5f5f5f", alpha=0.20, edgecolor="none", label="microsaccade ref IQR"),
    ]
    fig.legend(
        handles=color_handles + relation_handles + reference_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Contour-aligned vs contour-orthogonal unit-window pairs: real trace scale\n"
        f"{definition_text}; axis={image_axis_col}; aligned <= {match_max_deg:g} deg; "
        f"orthogonal >= {orthogonal_min_deg:g} deg; OSI >= {min_orientation_selectivity:g}; "
        f"min images/unit = {min_matched_images_per_unit}",
        fontsize=11.2,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.84))
    return save_figure(fig, fig_dir, "phase2_real_trace_sf_contour_aligned_vs_orthogonal_low_high_scale_curves")


def plot_phase2_figures(
    movie: pd.DataFrame,
    plot_groups: dict[str, list[str]],
    fig_dir: Path,
    *,
    trace_path_context: pd.DataFrame | None = None,
) -> dict[str, dict[str, str]]:
    figures: dict[str, dict[str, str]] = {}
    if trace_path_context is None:
        trace_path_context = trace_microsaccade_path_context_from_movie(movie)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    add_trace_path_context_bands(ax, trace_path_context)
    for mask, color, label in [
        (~movie["has_microsaccade"].astype(bool), NO_MS_COLOR, "no microsaccade"),
        (movie["has_microsaccade"].astype(bool), MS_COLOR, "microsaccade"),
    ]:
        sub = movie.loc[mask].copy()
        x = bin_x(sub, "trace_path_length_bin", "rendered_path_length_arcmin")
        means = sub.groupby("trace_path_length_bin", sort=True)["population_ssi"].mean()
        errs = sub.groupby("trace_path_length_bin", sort=True)["population_ssi"].apply(sem)
        common = [key for key in x.index if key in means.index]
        ax.errorbar(x.loc[common], means.loc[common], yerr=errs.loc[common], marker="o", lw=1.8, capsize=2, color=color, label=label)
    ax.set_xlabel("trace path length bin median (arcmin)")
    ax.set_ylabel("population SSI (bits/spike)")
    ax.set_title("Population SSI by microsaccade contamination")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    figures["microsaccade_population_by_path"] = save_figure(fig, fig_dir, "phase2_population_by_microsaccade_trace_path")

    for prefix, title in [
        ("sf", "SF group SSI across real trace scale"),
        ("tf_pref", "TF preference SSI across real trace scale"),
        ("speed_pref_legacy", "Legacy speed-pref group SSI across real trace scale"),
        ("ramp_drop", "Ramping/drop unit SSI across real trace scale"),
    ]:
        cols = plot_groups.get(prefix, [])
        if cols:
            figures[f"{prefix}_by_trace_path"] = plot_lines_by_trace_bin(
                movie,
                cols,
                labels_for_columns(cols, prefix),
                title=title,
                ylabel="group weighted SSI (bits/spike)",
                fig_dir=fig_dir,
                stem=f"phase2_{prefix}_groups_by_trace_path",
                trace_path_context=trace_path_context,
            )

    contour_cols = ["population_ssi", "population_ssi", "population_ssi"]
    contour_classes = ["weak_or_unreliable", "reliable", "strong"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    add_trace_path_context_bands(ax, trace_path_context)
    for idx, contour_class in enumerate(contour_classes):
        sub = movie[movie["image_contour_class"].astype(str).eq(contour_class)].copy()
        if sub.empty:
            continue
        x = bin_x(sub, "trace_path_length_bin", "rendered_path_length_arcmin")
        means = sub.groupby("trace_path_length_bin", sort=True)["population_ssi"].mean()
        errs = sub.groupby("trace_path_length_bin", sort=True)["population_ssi"].apply(sem)
        common = [key for key in x.index if key in means.index]
        ax.errorbar(
            x.loc[common],
            means.loc[common],
            yerr=errs.loc[common],
            marker="o",
            lw=1.8,
            capsize=2,
            color=color_for_label(contour_class, idx),
            label=contour_class.replace("_", " "),
        )
    ax.set_xlabel("trace path length bin median (arcmin)")
    ax.set_ylabel("population SSI (bits/spike)")
    ax.set_title("Image contour class x trace scale")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    figures["image_contour_by_trace_path"] = save_figure(fig, fig_dir, "phase2_image_contour_class_by_trace_path")

    orient_cols = [
        "unit_image_orientation_match_weighted_ssi",
        "unit_image_orientation_oblique_weighted_ssi",
        "unit_image_orientation_orthogonal_weighted_ssi",
    ]
    figures["unit_image_orientation_match_by_trace_path"] = plot_lines_by_trace_bin(
        movie,
        orient_cols,
        ["unit matched to image contour", "oblique units", "orthogonal units"],
        title="Unit orientation match to image contour",
        ylabel="group weighted SSI (bits/spike)",
        fig_dir=fig_dir,
        stem="phase2_unit_image_orientation_match_by_trace_path",
        trace_path_context=trace_path_context,
    )

    strong_mask = movie["image_contour_class"].astype(str).isin(["strong", "reliable"]).to_numpy()
    figures["unit_image_orientation_match_contour_images_by_trace_path"] = plot_lines_by_trace_bin(
        movie,
        orient_cols,
        ["unit matched to image contour", "oblique units", "orthogonal units"],
        title="Unit orientation match on contour-reliable images",
        ylabel="group weighted SSI (bits/spike)",
        fig_dir=fig_dir,
        stem="phase2_unit_image_orientation_match_contour_images_by_trace_path",
        filter_mask=strong_mask,
        trace_path_context=trace_path_context,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    add_trace_path_context_bands(ax, trace_path_context)
    for idx, axis_class in enumerate(["along_contour_axis", "oblique", "across_contour_axis", "low_anisotropy"]):
        sub = movie[movie["trace_image_axis_class"].astype(str).eq(axis_class)].copy()
        if sub.empty:
            continue
        x = bin_x(sub, "trace_path_length_bin", "rendered_path_length_arcmin")
        means = sub.groupby("trace_path_length_bin", sort=True)["population_ssi"].mean()
        errs = sub.groupby("trace_path_length_bin", sort=True)["population_ssi"].apply(sem)
        common = [key for key in x.index if key in means.index]
        ax.errorbar(
            x.loc[common],
            means.loc[common],
            yerr=errs.loc[common],
            marker="o",
            lw=1.8,
            capsize=2,
            color=color_for_label(axis_class, idx),
            label=axis_class.replace("_", " "),
        )
    ax.set_xlabel("trace path length bin median (arcmin)")
    ax.set_ylabel("population SSI (bits/spike)")
    ax.set_title("Trace covariance axis relative to image contour axis")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    figures["trace_image_axis_by_trace_path"] = save_figure(fig, fig_dir, "phase2_trace_image_axis_by_trace_path")

    return figures


def write_outputs(
    out_dir: Path,
    fig_dir: Path,
    matrix_dir: Path,
    ssi: np.ndarray,
    expected: np.ndarray | None,
    population: np.ndarray,
    movie: pd.DataFrame,
    image: pd.DataFrame,
    trace: pd.DataFrame,
    unit: pd.DataFrame,
    bin_rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    plot_groups: dict[str, list[str]],
    external_sources: list[dict[str, Any]],
    args: argparse.Namespace,
    stabilized_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = metric_columns_for_summaries(movie, registry)
    trace_summary = summarize_by_entity(movie, "trace_index", trace, "trace_bank_index", metric_cols)
    image_summary = summarize_by_entity(movie, "image_index", image, "image_index", metric_cols)
    unit_summary = unit_summary_table(ssi, expected, movie, unit, stabilized_baseline=stabilized_baseline)

    axis_metrics = ["population_ssi"]
    if "population_ssi_delta_vs_stabilized" in movie.columns:
        axis_metrics.append("population_ssi_delta_vs_stabilized")

    summary_specs = [
        ("trace_path", ["trace_path_length_bin"], metric_cols),
        ("trace_path_x_microsaccade", ["trace_path_length_bin", "has_microsaccade"], metric_cols),
        ("trace_diffusion_x_microsaccade", ["trace_diffusion_bin", "has_microsaccade"], metric_cols),
        ("image_contour_x_trace_path", ["image_contour_class", "trace_path_length_bin"], metric_cols),
        ("image_contour_x_microsaccade", ["image_contour_class", "has_microsaccade"], metric_cols),
        ("trace_image_axis_x_trace_path", ["trace_image_axis_class", "trace_path_length_bin"], axis_metrics),
        (
            "trace_image_axis_x_contour_x_trace_path",
            ["image_contour_class", "trace_image_axis_class", "trace_path_length_bin"],
            axis_metrics,
        ),
        ("trace_anisotropy_x_trace_path", ["trace_anisotropy_bin", "trace_path_length_bin"], axis_metrics),
        ("image_coherence_x_trace_path", ["image_orientation_coherence_bin", "trace_path_length_bin"], metric_cols),
    ]
    condition_summary = aggregate_many(movie, summary_specs)
    contour_match_sf_groups = parse_csv_list(str(args.contour_match_sf_groups))
    contour_selection, contour_curves, contour_summary = build_real_trace_sf_contour_matched_unit_curves(
        movie,
        ssi,
        unit,
        image_axis_col=str(args.image_axis_col),
        sf_groups=contour_match_sf_groups,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=float(args.orthogonal_min_deg),
        min_orientation_selectivity=float(args.min_contour_match_osi),
        min_matched_images_per_unit=int(args.min_contour_matched_images_per_unit),
        contour_relation="matched",
        stabilized_baseline=stabilized_baseline,
    )
    contour_orthogonal_selection, contour_orthogonal_curves, contour_orthogonal_summary = build_real_trace_sf_contour_matched_unit_curves(
        movie,
        ssi,
        unit,
        image_axis_col=str(args.image_axis_col),
        sf_groups=contour_match_sf_groups,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=float(args.orthogonal_min_deg),
        min_orientation_selectivity=float(args.min_contour_match_osi),
        min_matched_images_per_unit=int(args.min_contour_matched_images_per_unit),
        contour_relation="orthogonal",
        stabilized_baseline=stabilized_baseline,
    )
    all_orientation_selection, all_orientation_curves, all_orientation_summary = build_real_trace_sf_contour_matched_unit_curves(
        movie,
        ssi,
        unit,
        image_axis_col=str(args.image_axis_col),
        sf_groups=contour_match_sf_groups,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=float(args.orthogonal_min_deg),
        min_orientation_selectivity=float(args.min_contour_match_osi),
        min_matched_images_per_unit=int(args.min_contour_matched_images_per_unit),
        contour_relation="all_orientation",
        stabilized_baseline=stabilized_baseline,
    )

    movie.to_csv(out_dir / "phase1_movie_analysis_table.csv", index=False)
    trace_summary.to_csv(out_dir / "phase1_trace_summary_with_ssi.csv", index=False)
    image_summary.to_csv(out_dir / "phase1_image_summary_with_ssi.csv", index=False)
    unit_summary.to_csv(out_dir / "phase1_unit_summary_with_ssi.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(out_dir / "phase1_quantile_bin_definitions.csv", index=False)
    trace_path_context = load_trace_path_context_reference(movie, args)
    trace_path_context_csv = out_dir / "phase2_trace_path_context_windows.csv"
    trace_path_context.to_csv(trace_path_context_csv, index=False)
    pd.DataFrame(registry).to_csv(out_dir / "phase2_unit_group_registry.csv", index=False)
    condition_summary.to_csv(out_dir / "phase2_condition_summary_long.csv", index=False)
    pd.DataFrame(external_sources).to_csv(out_dir / "phase2_external_unit_label_sources.csv", index=False)
    contour_selection_csv = out_dir / "phase2_real_trace_sf_contour_matched_unit_selection.csv"
    contour_curves_csv = out_dir / "phase2_real_trace_sf_contour_matched_unit_curves.csv"
    contour_summary_csv = out_dir / "phase2_real_trace_sf_contour_matched_summary.csv"
    contour_orthogonal_selection_csv = out_dir / "phase2_real_trace_sf_contour_orthogonal_unit_selection.csv"
    contour_orthogonal_curves_csv = out_dir / "phase2_real_trace_sf_contour_orthogonal_unit_curves.csv"
    contour_orthogonal_summary_csv = out_dir / "phase2_real_trace_sf_contour_orthogonal_summary.csv"
    all_orientation_selection_csv = out_dir / "phase2_real_trace_sf_all_orientation_unit_selection.csv"
    all_orientation_curves_csv = out_dir / "phase2_real_trace_sf_all_orientation_unit_curves.csv"
    all_orientation_summary_csv = out_dir / "phase2_real_trace_sf_all_orientation_summary.csv"
    contour_selection.to_csv(contour_selection_csv, index=False)
    contour_curves.to_csv(contour_curves_csv, index=False)
    contour_summary.to_csv(contour_summary_csv, index=False)
    contour_orthogonal_selection.to_csv(contour_orthogonal_selection_csv, index=False)
    contour_orthogonal_curves.to_csv(contour_orthogonal_curves_csv, index=False)
    contour_orthogonal_summary.to_csv(contour_orthogonal_summary_csv, index=False)
    all_orientation_selection.to_csv(all_orientation_selection_csv, index=False)
    all_orientation_curves.to_csv(all_orientation_curves_csv, index=False)
    all_orientation_summary.to_csv(all_orientation_summary_csv, index=False)

    figures = {"phase1_qc": plot_phase1_qc(image, trace, unit, fig_dir)}
    figures.update(plot_phase2_figures(movie, plot_groups, fig_dir, trace_path_context=trace_path_context))
    figures["real_trace_sf_contour_matched_low_high"] = plot_real_trace_sf_contour_matched_figure(
        contour_curves,
        contour_summary,
        unit,
        fig_dir,
        sf_groups=contour_match_sf_groups,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=float(args.orthogonal_min_deg),
        min_orientation_selectivity=float(args.min_contour_match_osi),
        min_matched_images_per_unit=int(args.min_contour_matched_images_per_unit),
        image_axis_col=str(args.image_axis_col),
        trace_path_context=trace_path_context,
        contour_relation="matched",
    )
    figures["real_trace_sf_contour_orthogonal_low_high"] = plot_real_trace_sf_contour_matched_figure(
        contour_orthogonal_curves,
        contour_orthogonal_summary,
        unit,
        fig_dir,
        sf_groups=contour_match_sf_groups,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=float(args.orthogonal_min_deg),
        min_orientation_selectivity=float(args.min_contour_match_osi),
        min_matched_images_per_unit=int(args.min_contour_matched_images_per_unit),
        image_axis_col=str(args.image_axis_col),
        trace_path_context=trace_path_context,
        contour_relation="orthogonal",
    )
    figures["real_trace_sf_contour_aligned_vs_orthogonal_low_high"] = plot_real_trace_sf_contour_relation_overlay_figure(
        contour_summary,
        contour_orthogonal_summary,
        unit,
        fig_dir,
        sf_groups=contour_match_sf_groups,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=float(args.orthogonal_min_deg),
        min_orientation_selectivity=float(args.min_contour_match_osi),
        min_matched_images_per_unit=int(args.min_contour_matched_images_per_unit),
        image_axis_col=str(args.image_axis_col),
        trace_path_context=trace_path_context,
    )
    figures["real_trace_sf_all_orientation_low_high"] = plot_real_trace_sf_contour_matched_figure(
        all_orientation_curves,
        all_orientation_summary,
        unit,
        fig_dir,
        sf_groups=contour_match_sf_groups,
        match_max_deg=float(args.match_max_deg),
        orthogonal_min_deg=float(args.orthogonal_min_deg),
        min_orientation_selectivity=float(args.min_contour_match_osi),
        min_matched_images_per_unit=int(args.min_contour_matched_images_per_unit),
        image_axis_col=str(args.image_axis_col),
        trace_path_context=trace_path_context,
        contour_relation="all_orientation",
    )

    qc = qc_rows(ssi, expected, population, movie, image, trace, unit, stabilized_baseline=stabilized_baseline)
    pd.DataFrame(qc).to_csv(out_dir / "phase1_qc_summary.csv", index=False)

    key = {row["metric"]: row["value"] for row in qc}
    key.update(
        {
            "matrix_dir": str(matrix_dir),
            "out_dir": str(out_dir),
            "fig_dir": str(fig_dir),
            "image_axis_col": str(args.image_axis_col),
            "trace_axis_col": str(args.trace_axis_col),
            "match_max_deg": float(args.match_max_deg),
            "orthogonal_min_deg": float(args.orthogonal_min_deg),
            "trace_axis_min_anisotropy": float(args.trace_axis_min_anisotropy),
            "contour_match_sf_groups": contour_match_sf_groups,
            "min_contour_match_osi": float(args.min_contour_match_osi),
            "min_contour_matched_images_per_unit": int(args.min_contour_matched_images_per_unit),
            "n_condition_summary_rows": int(condition_summary.shape[0]),
            "n_unit_group_registry_rows": int(len(registry)),
            "n_real_trace_contour_matched_curve_rows": int(contour_curves.shape[0]),
            "n_real_trace_contour_matched_units_by_group": contour_curves.groupby("sf_group")["unit_index"].nunique().to_dict(),
            "n_real_trace_contour_orthogonal_curve_rows": int(contour_orthogonal_curves.shape[0]),
            "n_real_trace_contour_orthogonal_units_by_group": contour_orthogonal_curves.groupby("sf_group")["unit_index"].nunique().to_dict(),
            "n_real_trace_all_orientation_curve_rows": int(all_orientation_curves.shape[0]),
            "n_real_trace_all_orientation_units_by_group": all_orientation_curves.groupby("sf_group")["unit_index"].nunique().to_dict(),
            "stabilized_baseline_available": bool(stabilized_baseline is not None),
            "stabilized_baseline_source_dir": str(stabilized_baseline.get("source_dir", "")) if stabilized_baseline is not None else "",
            "real_trace_contour_matched_outputs": {
                "selection_csv": contour_selection_csv,
                "curves_csv": contour_curves_csv,
                "summary_csv": contour_summary_csv,
            },
            "real_trace_contour_orthogonal_outputs": {
                "selection_csv": contour_orthogonal_selection_csv,
                "curves_csv": contour_orthogonal_curves_csv,
                "summary_csv": contour_orthogonal_summary_csv,
            },
            "real_trace_all_orientation_outputs": {
                "selection_csv": all_orientation_selection_csv,
                "curves_csv": all_orientation_curves_csv,
                "summary_csv": all_orientation_summary_csv,
            },
            "trace_path_context_windows_csv": trace_path_context_csv,
            "trace_path_context_reference_csv": str(args.trace_path_context_reference_csv),
            "figures": figures,
            "external_unit_label_sources": external_sources,
            "notes": [
                "phase1_movie_analysis_table row order follows movie_feature_table row order and ssi_matrix row order.",
                "matrix_row_index is retained as provenance only; merged shards may contain shard-local matrix_row_index values.",
                "unit-image orientation match uses image contour axis and per-image unit preferred orientation deltas.",
                "trace-image axis classes use covariance major-axis orientation only when rendered covariance anisotropy passes the configured threshold.",
                "real-trace all-orientation, contour-matched, and contour-orthogonal low/high SF curves are unit-first: each unit is averaged over its selected image windows within each trace path bin before group means are computed.",
                "When stabilized_baseline_available is true, *_delta_vs_stabilized metrics subtract a zero-motion counterfactual baseline matched by image_index.",
            ],
        }
    )
    save_json(out_dir / "phase1_phase2_analysis_summary.json", key)
    return key


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else matrix_dir / "phase1_phase2_conditioning_v1"
    fig_dir = out_dir / "figures"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory already exists and is non-empty: {out_dir}. Use --force to overwrite files.")

    progress(f"Loading matrix dataset from {matrix_dir}")
    ssi, expected, population, movie, image, trace, unit = load_matrix_dataset(matrix_dir)
    progress(f"Loaded SSI matrix {ssi.shape}, movies={movie.shape[0]}, images={image.shape[0]}, traces={trace.shape[0]}, units={unit.shape[0]}")
    stabilized_baseline = load_stabilized_baseline(matrix_dir, image, unit)
    if stabilized_baseline is None:
        progress("No stabilized baseline found; delta_vs_stabilized columns will be omitted")
    else:
        progress(f"Loaded stabilized baseline {stabilized_baseline['ssi'].shape} from {matrix_dir}")

    progress("Joining external unit labels and deriving unit tuning bins")
    unit, external_sources, unit_bin_rows = enrich_unit_table(unit, args)

    progress("Building joined movie analysis table and unit-conditioned SSI columns")
    movie, image, trace, bin_rows, registry, plot_groups = prepare_movie_table(
        movie,
        image,
        trace,
        unit,
        ssi,
        expected,
        population,
        args,
        stabilized_baseline=stabilized_baseline,
    )
    bin_rows.extend(unit_bin_rows)

    progress("Writing Phase 1/2 tables, summaries, and figures")
    summary = write_outputs(
        out_dir,
        fig_dir,
        matrix_dir,
        ssi,
        expected,
        population,
        movie,
        image,
        trace,
        unit,
        bin_rows,
        registry,
        plot_groups,
        external_sources,
        args,
        stabilized_baseline=stabilized_baseline,
    )
    progress(f"Done. Wrote {out_dir}")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
