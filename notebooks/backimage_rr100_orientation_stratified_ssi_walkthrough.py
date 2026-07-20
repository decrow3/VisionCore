# %% [markdown]
# # BackImage RR100 orientation-stratified SSI walkthrough
#
# This is a notebook-style Python script for stepping through the logic behind
# the orientation-stratified BackImage RR100 contour-axis plot.
#
# Open it in VS Code, Jupyter, or any editor that recognizes `# %%` cells. The
# default cells are read-only: they inspect cached CSV/NPZ outputs, rebuild the
# plotted population summaries, and add intermediate plots that make the
# analysis contracts visible.
#
# Things this script checks:
#
# 1. Which cached rows feed the final figure.
# 2. Whether the contour-axis band mask is doing what we think.
# 3. Whether the final population bits/spike values are exactly reproducible
#    from raw per-fixation numerator and expected-spike columns.
# 4. Whether the effect is coming from information numerator, expected-spike
#    denominator, unit weighting, unit tuning, image statistics, or a label/frame
#    convention.
# 5. How to run a targeted tiny rotated-movie control on the exemplar source
#    rows, and optionally compare with the full rotated control once that
#    finishes.

# %%
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import Image, Markdown, display
except Exception:  # pragma: no cover - plain Python fallback
    Image = None
    Markdown = None

    def display(value: Any) -> None:
        print(value)

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass


def find_repo_root(start: Path) -> Path:
    """Walk upward until we find the VisionCore repository root."""
    start = start.resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "declan").exists() and (candidate / "experiments").exists():
            return candidate
    raise RuntimeError(f"Could not find repo root from {start}")


try:
    HERE = Path(__file__).resolve()
except NameError:
    HERE = Path.cwd().resolve()

ROOT = find_repo_root(HERE)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 220)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 170,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.90",
        "grid.linewidth": 0.8,
        "font.size": 9,
    }
)

ROOT

# %%


def in_notebook() -> bool:
    try:
        shell = get_ipython()
    except NameError:
        return False
    return shell is not None


def show_markdown(text: str) -> None:
    if Markdown is not None and in_notebook():
        display(Markdown(text))
    else:
        print(text)


def show_table(df: pd.DataFrame, n: int | None = None) -> None:
    out = df.head(n) if n is not None else df
    if in_notebook():
        display(out)
    else:
        print(out.to_string(index=False))


def show_image_if_exists(path: Path, *, width: int | None = None) -> None:
    path = Path(path)
    if not path.exists():
        print(f"Missing image: {path}")
        return
    if Image is not None and in_notebook():
        display(Image(filename=str(path), width=width))
    else:
        print(path)


def read_json_required(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_npz_required(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files if key != "cache_identity_json"}


def maybe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def scale_label(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:g}"


def alignment_label(value: str) -> str:
    return {
        "contour_aligned": "contour-aligned",
        "contour_orthogonal": "orthogonal",
    }.get(str(value), str(value))


def sf_label(value: str) -> str:
    return {"low_sf": "low SF", "high_sf": "high SF"}.get(str(value), str(value))


COLORS = {"contour_aligned": "#168a96", "contour_orthogonal": "#c06b2d"}
EPS = 1e-12
MOVIE_FEATURE_COLUMNS = [
    "balanced_manifest_index",
    "axis_balance_deg",
    "axis_balance_bin",
    "axis_balance_bin_start_deg",
    "axis_balance_bin_stop_deg",
    "energy_balance_column",
    "energy_balance_value",
    "energy_balance_bin",
    "energy_balance_quantile_bins",
    "image_patch_rms_contrast",
    "image_patch_std",
    "image_gradient_energy",
    "image_orientation_coherence",
    "image_oriented_gradient_energy",
    "image_multi_orientation_energy",
    "image_edge_density",
    "image_spectrum_anisotropy",
    "image_abs_8plus_power_proxy",
    "image_oriented_8plus_power_proxy",
    "image_high_freq_power_fraction",
    "image_power_8plus_cpd_fraction",
]

# %% [markdown]
# ## 1. Choose the run to inspect
#
# By default this points at the long balanced axis-30 run that produced the
# figure in the prompt. To inspect a rotated control run, either edit
# `RUN_ROOT` here or launch with:
#
# ```bash
# BACKIMAGE_RR100_RUN_ROOT=/path/to/rotated/run jupyter lab
# ```
#
# Useful knobs for later cells:
#
# - `TARGET_BAND`: orientation band to audit.
# - `TARGET_VIEW`: one of `across_along0`, `across_along1`, `along_across0`,
#   `along_across1`.
# - `TARGET_SF`: `low_sf` or `high_sf`.

# %%
DEFAULT_RUN_ROOT = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_v1"
)
RUN_ROOT = Path(os.environ.get("BACKIMAGE_RR100_RUN_ROOT", DEFAULT_RUN_ROOT)).expanduser().resolve()

TARGET_BAND = os.environ.get("BACKIMAGE_RR100_BAND", "near_vertical_axis90pm15")
TARGET_VIEW = os.environ.get("BACKIMAGE_RR100_VIEW", "across_along1")
TARGET_SF = os.environ.get("BACKIMAGE_RR100_SF", "high_sf")
TARGET_CONDITION_ID = os.environ.get("BACKIMAGE_RR100_CONDITION", "along1_across1")
EXAMPLE_MOVIES_PER_BAND = int(os.environ.get("BACKIMAGE_RR100_EXAMPLES_PER_BAND", "4"))
EXAMPLE_X_SCALE = float(os.environ.get("BACKIMAGE_RR100_EXAMPLE_X_SCALE", "1.0"))
SHOW_EXAMPLE_PATCHES_SETTING = os.environ.get("BACKIMAGE_RR100_SHOW_PATCHES", "auto").strip().lower()
SHOW_EXAMPLE_PATCHES = (
    in_notebook()
    if SHOW_EXAMPLE_PATCHES_SETTING == "auto"
    else SHOW_EXAMPLE_PATCHES_SETTING in {"1", "true", "yes", "y"}
)
RUN_TARGETED_ROTATION = os.environ.get("BACKIMAGE_RR100_RUN_TARGETED_ROTATION", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
FORCE_TARGETED_ROTATION = os.environ.get("BACKIMAGE_RR100_FORCE_TARGETED_ROTATION", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
TARGETED_ROTATION_DEG = int(os.environ.get("BACKIMAGE_RR100_TARGETED_ROTATION_DEG", "90"))
TARGETED_ROTATION_RUN_ROOT = Path(
    os.environ.get(
        "BACKIMAGE_RR100_TARGETED_ROTATION_RUN_ROOT",
        RUN_ROOT / f"tutorial_tiny_exemplar_rot{TARGETED_ROTATION_DEG}",
    )
).expanduser().resolve()
TARGETED_ROTATION_DEVICE = os.environ.get("BACKIMAGE_RR100_TARGETED_DEVICE", "").strip()
TARGETED_ROTATION_BATCH_SIZE = int(os.environ.get("BACKIMAGE_RR100_TARGETED_BATCH_SIZE", "8"))
TARGETED_ROTATION_N_BOOTSTRAP = int(os.environ.get("BACKIMAGE_RR100_TARGETED_N_BOOTSTRAP", "0"))

LONG_RUN_COMMANDS = RUN_ROOT / "long_run_commands.json"
COMMANDS = read_json_required(LONG_RUN_COMMANDS) if LONG_RUN_COMMANDS.exists() else {}

POPULATION_DIRS = {
    "across_along0": RUN_ROOT / "population_across_sweep_along0",
    "across_along1": RUN_ROOT / "population_across_sweep_along1",
    "along_across0": RUN_ROOT / "population_along_sweep_across0",
    "along_across1": RUN_ROOT / "population_along_sweep_across1",
}
if "population_dirs" in COMMANDS:
    POPULATION_DIRS.update({key: Path(value) for key, value in COMMANDS["population_dirs"].items()})

CONTOUR_RUN_DIR = Path(COMMANDS.get("contour_dir", RUN_ROOT / "contour_rr100_spatial_ssi_pairs27"))
ORIENTATION_DIR = Path(COMMANDS.get("orientation_stratified_dir", RUN_ROOT / "orientation_stratified_population"))
BALANCED_SOURCE_DIR = Path(COMMANDS.get("balanced_source_dir", RUN_ROOT / "balanced_source_windows"))
SELECTED_WINDOWS_CSV = BALANCED_SOURCE_DIR / "selected_windows.csv"
CACHE_NPZ = CONTOUR_RUN_DIR / "cache" / "backimage_contour_axis_rr100_spatial_ssi_cache.npz"
ORIENTATION_SUMMARY_CSV = ORIENTATION_DIR / "orientation_stratified_weighted_population_summary.csv"

VIEW_SPECS = [
    {
        "view": "across_along0",
        "label": "across scale\nalong=0",
        "x_col": "across_scale",
        "dir": POPULATION_DIRS["across_along0"],
    },
    {
        "view": "across_along1",
        "label": "across scale\nalong=1",
        "x_col": "across_scale",
        "dir": POPULATION_DIRS["across_along1"],
    },
    {
        "view": "along_across0",
        "label": "along scale\nacross=0",
        "x_col": "along_scale",
        "dir": POPULATION_DIRS["along_across0"],
    },
    {
        "view": "along_across1",
        "label": "along scale\nacross=1",
        "x_col": "along_scale",
        "dir": POPULATION_DIRS["along_across1"],
    },
]

path_check = pd.DataFrame(
    [
        {"name": "run_root", "path": RUN_ROOT, "exists": RUN_ROOT.exists()},
        {"name": "contour_run_dir", "path": CONTOUR_RUN_DIR, "exists": CONTOUR_RUN_DIR.exists()},
        {"name": "selected_windows_csv", "path": SELECTED_WINDOWS_CSV, "exists": SELECTED_WINDOWS_CSV.exists()},
        {"name": "cache_npz", "path": CACHE_NPZ, "exists": CACHE_NPZ.exists()},
        {"name": "orientation_summary_csv", "path": ORIENTATION_SUMMARY_CSV, "exists": ORIENTATION_SUMMARY_CSV.exists()},
        *[
            {
                "name": f"{spec['view']}_weighted_rows",
                "path": Path(spec["dir"]) / "per_fixation_weighted_alignment_population_ssi.csv",
                "exists": (Path(spec["dir"]) / "per_fixation_weighted_alignment_population_ssi.csv").exists(),
            }
            for spec in VIEW_SPECS
        ],
    ]
)
show_table(path_check)

# %% [markdown]
# ## 2. Look at the exact output figure
#
# This is only a reference. All later cells rebuild or interrogate pieces of
# the figure rather than trusting the PNG.

# %%
target_figure = ORIENTATION_DIR / f"backimage_rr100_orientation_stratified_population_curves_{TARGET_BAND}.png"
show_image_if_exists(target_figure, width=1100)

# %% [markdown]
# ## 3. Angle helpers copied from the plot contract
#
# The production scripts use axial orientation, so `0 == 180`. For a gaze-frame
# contour axis, the image-frame contour axis is `(-axis_deg) % 180`.

# %%


def orientation_axis_180(angle_deg: float | np.ndarray) -> np.ndarray:
    return np.asarray(angle_deg, dtype=np.float64) % 180.0


def angle_180_distance(a_deg: float | np.ndarray, b_deg: float | np.ndarray) -> np.ndarray:
    return np.abs(((np.asarray(a_deg, dtype=np.float64) - np.asarray(b_deg, dtype=np.float64) + 90.0) % 180.0) - 90.0)


def axial_alignment_score(preferred_deg: np.ndarray, contour_axis_deg: float) -> np.ndarray:
    """+1 means bar axis matches contour; -1 means bar axis is orthogonal."""
    delta = angle_180_distance(preferred_deg, contour_axis_deg)
    return np.cos(np.deg2rad(2.0 * delta))


def contour_axis_to_image_frame(axis_deg: float | np.ndarray, coordinate_frame: str) -> np.ndarray:
    axis = np.asarray(axis_deg, dtype=np.float64)
    if str(coordinate_frame) == "gaze":
        axis = -axis
    return orientation_axis_180(axis)


def band_mask(axis_deg: np.ndarray, band: str) -> np.ndarray:
    axis = np.asarray(axis_deg, dtype=np.float64) % 180.0
    if band == "near_horizontal_axis0pm15":
        return np.minimum(axis, 180.0 - axis) <= 15.0
    if band == "near_vertical_axis90pm15":
        return np.abs(axis - 90.0) <= 15.0
    if str(band).startswith("axis_bin_"):
        parts = str(band).split("_")
        if len(parts) == 4:
            start = float(parts[2])
            stop = float(parts[3])
            if stop >= 180.0:
                return (axis >= start) & (axis <= 180.0)
            return (axis >= start) & (axis < stop)
    raise ValueError(f"Unknown band {band!r}")


angle_demo = pd.DataFrame(
    {
        "a": [0, 5, 175, 90, 135],
        "b": [180, 175, 5, 0, 45],
    }
)
angle_demo["axial_distance_deg"] = angle_180_distance(angle_demo["a"], angle_demo["b"])
angle_demo["alignment_score_vs_b"] = axial_alignment_score(angle_demo["a"].to_numpy(), 0.0)
show_table(angle_demo)

# %% [markdown]
# ## 4. Load the per-fixation rows and saved orientation summary
#
# Each row is already one condition x fixation x SF group x alignment group. The
# final orientation plot does not average the `population_bits_per_spike` column
# directly. It sums:
#
# `information_numerator_bits_arbitrary_dt / expected_spikes_arbitrary_dt`
#
# after filtering the movie IDs to one contour-axis band.

# %%


def load_view_frame(spec: dict[str, Any]) -> pd.DataFrame:
    path = Path(spec["dir"]) / "per_fixation_weighted_alignment_population_ssi.csv"
    df = read_csv_required(path)
    df = df.copy()
    df["view"] = str(spec["view"])
    df["view_label"] = str(spec["label"])
    df["x_scale"] = pd.to_numeric(df[str(spec["x_col"])], errors="coerce")
    df["source_csv"] = str(path)
    return df


weighted_by_view = {str(spec["view"]): load_view_frame(spec) for spec in VIEW_SPECS}
orientation_summary = read_csv_required(ORIENTATION_SUMMARY_CSV)

overview_rows = []
for spec in VIEW_SPECS:
    df = weighted_by_view[str(spec["view"])]
    overview_rows.append(
        {
            "view": spec["view"],
            "rows": int(df.shape[0]),
            "n_movies": int(df["movie_index"].nunique()),
            "conditions": ", ".join(df["condition_id"].drop_duplicates().astype(str).tolist()),
            "sf_groups": ", ".join(sorted(df["sf_group"].dropna().astype(str).unique())),
            "alignment_groups": ", ".join(sorted(df["alignment_group"].dropna().astype(str).unique())),
            "x_col": spec["x_col"],
            "x_values": ", ".join(scale_label(v) for v in sorted(df["x_scale"].dropna().unique())),
        }
    )
show_table(pd.DataFrame(overview_rows))

show_markdown("Saved orientation summary rows:")
show_table(orientation_summary.head(12))

# %% [markdown]
# ## 5. Movie-axis inventory and band support
#
# The final figure filters by `movie_index`, using one contour axis per movie.
# This cell checks that all four view CSVs agree on that movie-axis table.

# %%


def movie_axis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "movie_index",
        "trial_id",
        "source_row",
        "session",
        "trial_idx",
        "axis_deg",
        "axis_coordinate_frame",
        "contour_axis_image_deg",
    ]
    cols = [*base_cols, *[col for col in MOVIE_FEATURE_COLUMNS if col in frame.columns]]
    keep = [col for col in cols if col in frame.columns]
    out = frame[keep].drop_duplicates("movie_index").sort_values("movie_index").reset_index(drop=True)
    return out


axis_by_view = {view: movie_axis_frame(df) for view, df in weighted_by_view.items()}
axis_reference = axis_by_view[VIEW_SPECS[0]["view"]]

axis_checks = []
for view, axes in axis_by_view.items():
    merged = axis_reference[["movie_index", "contour_axis_image_deg"]].merge(
        axes[["movie_index", "contour_axis_image_deg"]],
        on="movie_index",
        suffixes=("_ref", "_view"),
    )
    delta = angle_180_distance(
        merged["contour_axis_image_deg_ref"].to_numpy(dtype=float),
        merged["contour_axis_image_deg_view"].to_numpy(dtype=float),
    )
    axis_checks.append(
        {
            "view": view,
            "n_movies": int(axes["movie_index"].nunique()),
            "max_axis_disagreement_deg": float(np.nanmax(delta)),
        }
    )
axis_reference["near_horizontal_axis0pm15"] = band_mask(
    axis_reference["contour_axis_image_deg"].to_numpy(dtype=float),
    "near_horizontal_axis0pm15",
)
axis_reference["near_vertical_axis90pm15"] = band_mask(
    axis_reference["contour_axis_image_deg"].to_numpy(dtype=float),
    "near_vertical_axis90pm15",
)
show_table(pd.DataFrame(axis_checks))

band_rows = []
for band in [
    "near_horizontal_axis0pm15",
    "near_vertical_axis90pm15",
    "axis_bin_000_030",
    "axis_bin_030_060",
    "axis_bin_060_090",
    "axis_bin_090_120",
    "axis_bin_120_150",
    "axis_bin_150_180",
]:
    mask = band_mask(axis_reference["contour_axis_image_deg"].to_numpy(dtype=float), band)
    band_rows.append(
        {
            "band": band,
            "n_movies": int(mask.sum()),
            "fraction_movies": float(mask.mean()),
            "axis_min": float(np.nanmin(axis_reference.loc[mask, "contour_axis_image_deg"])) if mask.any() else float("nan"),
            "axis_max": float(np.nanmax(axis_reference.loc[mask, "contour_axis_image_deg"])) if mask.any() else float("nan"),
        }
    )
show_table(pd.DataFrame(band_rows))

# %%
fig, ax = plt.subplots(figsize=(8.8, 3.5))
axis = axis_reference["contour_axis_image_deg"].to_numpy(dtype=float)
ax.hist(axis, bins=np.arange(0, 181, 5), color="0.72", edgecolor="white", linewidth=0.5)
ax.axvspan(75, 105, color="#168a96", alpha=0.18, label="near vertical")
ax.axvspan(0, 15, color="#c06b2d", alpha=0.15, label="near horizontal")
ax.axvspan(165, 180, color="#c06b2d", alpha=0.15)
ax.set_xlabel("contour_axis_image_deg")
ax.set_ylabel("movies")
ax.set_title("Balanced-window contour-axis distribution")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ## 6. Rebuild the plotted summary from raw rows
#
# This is the core audit: filter movie IDs by orientation band, group by
# condition/SF/alignment, sum numerator and denominator, then divide. Bootstrap
# CIs are not needed for this exactness check.

# %%


def summarize_view_without_bootstrap(frame: pd.DataFrame, *, view: str, view_label: str, x_col: str, band: str) -> pd.DataFrame:
    axes = frame[["movie_index", "contour_axis_image_deg"]].drop_duplicates("movie_index")
    keep_movies = set(
        axes.loc[band_mask(axes["contour_axis_image_deg"].to_numpy(dtype=float), band), "movie_index"].astype(int).tolist()
    )
    filtered = frame[frame["movie_index"].astype(int).isin(keep_movies)].copy()
    group_cols = ["condition_id", "condition_index", x_col, "sf_group", "alignment_group"]
    rows: list[dict[str, Any]] = []
    for keys, sub in filtered.groupby(group_cols, sort=True):
        condition_id, condition_index, x_value, sf_group, alignment_group = keys
        numerator = float(np.nansum(sub["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=float)))
        denominator = float(np.nansum(sub["expected_spikes_arbitrary_dt"].to_numpy(dtype=float)))
        rows.append(
            {
                "band": band,
                "view": view,
                "view_label": view_label,
                "condition_id": str(condition_id),
                "condition_index": int(condition_index),
                "x_scale": float(x_value),
                "sf_group": str(sf_group),
                "alignment_group": str(alignment_group),
                "n_fixations": int(sub["movie_index"].nunique()),
                "accumulated_bits_per_spike": numerator / max(denominator, EPS),
                "expected_spikes_sum_arbitrary_dt": denominator,
                "information_numerator_sum_bits_arbitrary_dt": numerator,
                "mean_fixation_bits_per_spike": float(np.nanmean(sub["population_bits_per_spike"].to_numpy(dtype=float))),
                "median_fixation_bits_per_spike": float(np.nanmedian(sub["population_bits_per_spike"].to_numpy(dtype=float))),
                "n_rows": int(sub.shape[0]),
            }
        )
    return pd.DataFrame(rows)


rebuilt_parts = []
for spec in VIEW_SPECS:
    rebuilt_parts.append(
        summarize_view_without_bootstrap(
            weighted_by_view[str(spec["view"])],
            view=str(spec["view"]),
            view_label=str(spec["label"]),
            x_col=str(spec["x_col"]),
            band=TARGET_BAND,
        )
    )
rebuilt_target_band = pd.concat(rebuilt_parts, ignore_index=True)

compare_keys = ["band", "view", "condition_id", "condition_index", "x_scale", "sf_group", "alignment_group"]
saved_target_band = orientation_summary[orientation_summary["band"].astype(str) == TARGET_BAND].copy()
comparison = rebuilt_target_band.merge(
    saved_target_band,
    on=compare_keys,
    how="outer",
    suffixes=("_rebuilt", "_saved"),
    indicator=True,
)
for col in [
    "accumulated_bits_per_spike",
    "expected_spikes_sum_arbitrary_dt",
    "information_numerator_sum_bits_arbitrary_dt",
]:
    comparison[f"{col}_abs_diff"] = np.abs(comparison[f"{col}_rebuilt"] - comparison[f"{col}_saved"])

diff_cols = [col for col in comparison.columns if col.endswith("_abs_diff")]
show_table(
    pd.DataFrame(
        [
            {
                "rows_rebuilt": int(rebuilt_target_band.shape[0]),
                "rows_saved": int(saved_target_band.shape[0]),
                "merge_status": ", ".join(f"{k}:{v}" for k, v in comparison["_merge"].value_counts().to_dict().items()),
                **{f"max_{col}": float(np.nanmax(comparison[col])) for col in diff_cols},
            }
        ]
    )
)
show_table(comparison.sort_values("accumulated_bits_per_spike_abs_diff", ascending=False).head(8))

# %% [markdown]
# ## 7. Replot the target band from the rebuilt table
#
# If this matches the saved PNG shape, the odd result is already present in the
# raw numerator/denominator CSVs, not in the final plotting routine.

# %%


def plot_orientation_curves(summary: pd.DataFrame, *, title: str, use_saved_ci: bool = False) -> None:
    sf_groups = ["low_sf", "high_sf"]
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 5.9), sharey="row", constrained_layout=True)
    for row_idx, sf_group in enumerate(sf_groups):
        for col_idx, spec in enumerate(VIEW_SPECS):
            ax = axes[row_idx, col_idx]
            sub_view = summary[
                (summary["view"].astype(str) == str(spec["view"]))
                & (summary["sf_group"].astype(str) == sf_group)
            ].copy()
            for alignment_group in ["contour_aligned", "contour_orthogonal"]:
                sub = sub_view[sub_view["alignment_group"].astype(str) == alignment_group].sort_values(
                    ["x_scale", "condition_index"]
                )
                if sub.empty:
                    continue
                x = sub["x_scale"].to_numpy(dtype=float)
                y = sub["accumulated_bits_per_spike"].to_numpy(dtype=float)
                color = COLORS[alignment_group]
                if use_saved_ci:
                    lo_col = "accumulated_bits_per_spike_boot_ci_low"
                    hi_col = "accumulated_bits_per_spike_boot_ci_high"
                    if lo_col in sub.columns and hi_col in sub.columns:
                        lo = sub[lo_col].to_numpy(dtype=float)
                        hi = sub[hi_col].to_numpy(dtype=float)
                        finite = np.isfinite(lo) & np.isfinite(hi)
                        ax.fill_between(x[finite], lo[finite], hi[finite], color=color, alpha=0.12, linewidth=0)
                support = int(np.nanmedian(sub["n_fixations"].to_numpy(dtype=float)))
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=color,
                    label=f"{alignment_label(alignment_group)} (n={support})",
                )
            ax.axvline(1.0, color="0.62", linestyle=":", linewidth=0.9)
            ax.set_title(str(spec["label"]), fontsize=9.5)
            if col_idx == 0:
                ax.set_ylabel(f"{sf_label(sf_group)}\nbits/spike")
            if row_idx == 1:
                ax.set_xlabel("scale")
            if row_idx == 0 and col_idx == 3:
                ax.legend(frameon=False, fontsize=7.5, loc="best")
    fig.suptitle(title, fontsize=12)
    plt.show()


plot_orientation_curves(
    rebuilt_target_band,
    title=f"Rebuilt from raw per-fixation rows: {TARGET_BAND}",
)

# %%
plot_orientation_curves(
    saved_target_band,
    title=f"Saved summary with bootstrap CIs: {TARGET_BAND}",
    use_saved_ci=True,
)

# %% [markdown]
# ## 8. Numerator vs denominator decomposition
#
# Population bits/spike can move because information numerator changes,
# expected-spike denominator changes, or both. This cell separates them for one
# view and SF group.

# %%
focus_summary = rebuilt_target_band[
    (rebuilt_target_band["view"].astype(str) == TARGET_VIEW)
    & (rebuilt_target_band["sf_group"].astype(str) == TARGET_SF)
].copy()
focus_summary = focus_summary.sort_values(["alignment_group", "x_scale", "condition_index"])
show_table(focus_summary)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), constrained_layout=True)
panels = [
    ("accumulated_bits_per_spike", "bits/spike ratio"),
    ("information_numerator_sum_bits_arbitrary_dt", "summed information numerator"),
    ("expected_spikes_sum_arbitrary_dt", "summed expected spikes"),
]
for ax, (col, label) in zip(axes, panels, strict=True):
    for alignment_group in ["contour_aligned", "contour_orthogonal"]:
        sub = focus_summary[focus_summary["alignment_group"].astype(str) == alignment_group]
        ax.plot(
            sub["x_scale"].to_numpy(dtype=float),
            sub[col].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            color=COLORS[alignment_group],
            label=alignment_label(alignment_group),
        )
    ax.axvline(1.0, color="0.62", linestyle=":", linewidth=0.9)
    ax.set_xlabel("scale")
    ax.set_title(label)
axes[0].set_ylabel(sf_label(TARGET_SF))
axes[-1].legend(frameon=False)
fig.suptitle(f"{TARGET_VIEW}, {TARGET_SF}, {TARGET_BAND}: ratio ingredients")
plt.show()

# %% [markdown]
# ## 9. Per-fixation distributions behind one panel
#
# The final curve is an accumulated ratio over fixations. This cell shows the
# per-fixation ratios directly, alongside the accumulated curve. Large tails can
# be real, but they are worth seeing.

# %%


def target_filtered_rows(view: str, sf_group: str, band: str) -> pd.DataFrame:
    frame = weighted_by_view[view].copy()
    axes = frame[["movie_index", "contour_axis_image_deg"]].drop_duplicates("movie_index")
    keep_movies = set(
        axes.loc[band_mask(axes["contour_axis_image_deg"].to_numpy(dtype=float), band), "movie_index"].astype(int)
    )
    out = frame[
        frame["movie_index"].astype(int).isin(keep_movies)
        & (frame["sf_group"].astype(str) == sf_group)
    ].copy()
    return out


focus_rows = target_filtered_rows(TARGET_VIEW, TARGET_SF, TARGET_BAND)
show_table(
    focus_rows.groupby(["condition_id", "x_scale", "alignment_group"], sort=True)
    .agg(
        n_rows=("population_bits_per_spike", "size"),
        n_finite=("population_bits_per_spike", lambda x: int(np.isfinite(pd.to_numeric(x, errors="coerce")).sum())),
        median=("population_bits_per_spike", "median"),
        p05=("population_bits_per_spike", lambda x: float(np.nanpercentile(pd.to_numeric(x, errors="coerce"), 5))),
        p95=("population_bits_per_spike", lambda x: float(np.nanpercentile(pd.to_numeric(x, errors="coerce"), 95))),
        expected_spikes_sum=("expected_spikes_arbitrary_dt", "sum"),
    )
    .reset_index()
)

# %%
conditions = focus_summary.sort_values(["x_scale", "condition_index"])["condition_id"].drop_duplicates().tolist()
x_positions = {condition: idx for idx, condition in enumerate(conditions)}
fig, ax = plt.subplots(figsize=(10.5, 4.0))
rng = np.random.default_rng(2)
for alignment_group, offset in [("contour_aligned", -0.16), ("contour_orthogonal", 0.16)]:
    sub = focus_rows[focus_rows["alignment_group"].astype(str) == alignment_group].copy()
    for condition_id, cond_sub in sub.groupby("condition_id", sort=False):
        x0 = x_positions[str(condition_id)] + offset
        y = cond_sub["population_bits_per_spike"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue
        jitter = rng.normal(0.0, 0.025, size=y.size)
        ax.scatter(
            np.full(y.size, x0) + jitter,
            y,
            s=9,
            alpha=0.22,
            color=COLORS[alignment_group],
            linewidth=0,
        )
    summary_sub = focus_summary[focus_summary["alignment_group"].astype(str) == alignment_group]
    ax.plot(
        [x_positions[str(c)] + offset for c in summary_sub["condition_id"]],
        summary_sub["accumulated_bits_per_spike"],
        marker="D",
        linewidth=2.0,
        markersize=4,
        color=COLORS[alignment_group],
        label=f"{alignment_label(alignment_group)} accumulated",
    )
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels([scale_label(focus_summary.loc[focus_summary["condition_id"] == c, "x_scale"].iloc[0]) for c in conditions])
ax.set_xlabel("scale")
ax.set_ylabel("per-fixation population bits/spike")
ax.set_title(f"{TARGET_VIEW}, {TARGET_SF}, {TARGET_BAND}: fixation-level spread")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# ## 10. Axis frame convention audit
#
# This checks the transformation from the cached gaze-frame `axis_deg` to
# `contour_axis_image_deg`. If the strange plot is actually a frame convention
# problem, it should show up here or in the alternative-band cells below.

# %%
axis_frame = axis_reference.copy()
axis_frame["recomputed_contour_axis_image_deg"] = contour_axis_to_image_frame(
    axis_frame["axis_deg"].to_numpy(dtype=float),
    "gaze",
)
axis_frame["axis_recompute_error_deg"] = angle_180_distance(
    axis_frame["contour_axis_image_deg"].to_numpy(dtype=float),
    axis_frame["recomputed_contour_axis_image_deg"].to_numpy(dtype=float),
)
show_table(
    pd.DataFrame(
        [
            {
                "max_recompute_error_deg": float(np.nanmax(axis_frame["axis_recompute_error_deg"])),
                "median_recompute_error_deg": float(np.nanmedian(axis_frame["axis_recompute_error_deg"])),
                "n_nonzero_errors": int((axis_frame["axis_recompute_error_deg"] > 1e-9).sum()),
            }
        ]
    )
)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
axes[0].scatter(axis_frame["axis_deg"], axis_frame["contour_axis_image_deg"], s=12, alpha=0.6)
axes[0].set_xlabel("axis_deg from movie inventory")
axes[0].set_ylabel("contour_axis_image_deg in population CSV")
axes[0].set_title("Gaze-to-image conversion")

axes[1].scatter(axis_frame["contour_axis_image_deg"], axis_frame["axis_recompute_error_deg"], s=12, alpha=0.6)
axes[1].set_xlabel("contour_axis_image_deg")
axes[1].set_ylabel("axial recompute error (deg)")
axes[1].set_title("Conversion residual")
plt.show()

# %% [markdown]
# ## 11. Unit tuning and alignment weights
#
# The weighted split is continuous:
#
# - `signed_alignment = cos(2 * distance(unit_pref, contour_axis))`
# - contour-aligned weight is `max(signed_alignment, 0)`
# - orthogonal weight is `max(-signed_alignment, 0)`
#
# For near-vertical contours, contour-aligned high-SF units are mostly
# vertical-preferring, and orthogonal high-SF units are mostly horizontal-
# preferring. This cell makes that visible.

# %%


def sf_groups_csv_from_commands(commands: dict[str, Any]) -> Path | None:
    plot_cmds = commands.get("commands", {}).get("population_plots", [])
    if not plot_cmds:
        return None
    first = plot_cmds[0]
    for idx, token in enumerate(first[:-1]):
        if str(token) == "--sf-groups-csv":
            return Path(first[idx + 1])
    return None


SF_GROUPS_CSV = sf_groups_csv_from_commands(COMMANDS)
if SF_GROUPS_CSV is None:
    SF_GROUPS_CSV = ROOT / (
        "outputs/active_sensing_movie_information/"
        "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
        "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
        "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
    )

units = read_csv_required(SF_GROUPS_CSV).copy()
units["preferred_orientation_image_deg"] = orientation_axis_180(
    pd.to_numeric(units["prior_preferred_orientation_deg"], errors="coerce").to_numpy(dtype=float)
)
units["prior_orientation_selectivity_index"] = pd.to_numeric(
    units["prior_orientation_selectivity_index"],
    errors="coerce",
)
show_table(
    units.groupby("sf_group", sort=True)
    .agg(
        n_units=("unit_index", "nunique"),
        median_sf=("sf_split_metric", "median"),
        median_osi=("prior_orientation_selectivity_index", "median"),
        orientation_min=("preferred_orientation_image_deg", "min"),
        orientation_max=("preferred_orientation_image_deg", "max"),
    )
    .reset_index()
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
for ax, sf_group in zip(axes, ["low_sf", "high_sf"], strict=True):
    sub = units[units["sf_group"].astype(str) == sf_group].copy()
    ax.hist(
        sub["preferred_orientation_image_deg"].to_numpy(dtype=float),
        bins=np.arange(0, 181, 10),
        weights=np.ones(sub.shape[0]),
        color="#4c6f91" if sf_group == "low_sf" else "#7b5ea7",
        alpha=0.78,
        edgecolor="white",
    )
    ax.axvspan(75, 105, color="#168a96", alpha=0.16, label="near-vertical contour-aligned target")
    ax.axvspan(0, 15, color="#c06b2d", alpha=0.15, label="near-vertical orthogonal target")
    ax.axvspan(165, 180, color="#c06b2d", alpha=0.15)
    ax.set_title(sf_label(sf_group))
    ax.set_xlabel("preferred_orientation_image_deg")
    ax.set_ylabel("units")
axes[1].legend(frameon=False, fontsize=8)
plt.show()

# %%
selection_by_view = {}
for spec in VIEW_SPECS:
    path = Path(spec["dir"]) / "per_fixation_weighted_alignment_selection.csv"
    selection_by_view[str(spec["view"])] = read_csv_required(path).assign(view=str(spec["view"]))

selection_focus = selection_by_view[TARGET_VIEW].copy()
selection_axes = selection_focus[["movie_index", "contour_axis_image_deg"]].drop_duplicates("movie_index")
keep_target_movies = set(
    selection_axes.loc[
        band_mask(selection_axes["contour_axis_image_deg"].to_numpy(dtype=float), TARGET_BAND),
        "movie_index",
    ].astype(int)
)
selection_focus = selection_focus[
    selection_focus["movie_index"].astype(int).isin(keep_target_movies)
    & (selection_focus["sf_group"].astype(str) == TARGET_SF)
].copy()

show_table(
    selection_focus.groupby(["sf_group", "alignment_group"], sort=True)
    .agg(
        n_fixations=("movie_index", "nunique"),
        mean_n_units=("n_units", "mean"),
        mean_weight_sum=("weight_sum", "mean"),
        median_weight_sum=("weight_sum", "median"),
        mean_effective_n=("effective_n_units", "mean"),
        mean_target_delta=("mean_orientation_target_delta_deg", "mean"),
        median_target_delta=("median_orientation_target_delta_deg", "median"),
    )
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
for alignment_group in ["contour_aligned", "contour_orthogonal"]:
    sub = selection_focus[selection_focus["alignment_group"].astype(str) == alignment_group]
    color = COLORS[alignment_group]
    axes[0].scatter(sub["contour_axis_image_deg"], sub["weight_sum"], s=11, alpha=0.45, color=color, label=alignment_label(alignment_group))
    axes[1].scatter(sub["contour_axis_image_deg"], sub["effective_n_units"], s=11, alpha=0.45, color=color)
    axes[2].scatter(sub["contour_axis_image_deg"], sub["mean_orientation_target_delta_deg"], s=11, alpha=0.45, color=color)
axes[0].set_ylabel("weight_sum")
axes[1].set_ylabel("effective_n_units")
axes[2].set_ylabel("weighted mean target delta (deg)")
for ax in axes:
    ax.set_xlabel("contour_axis_image_deg")
axes[0].legend(frameon=False)
fig.suptitle(f"Selection/weight support: {TARGET_VIEW}, {TARGET_SF}, {TARGET_BAND}")
plt.show()

# %% [markdown]
# ## 12. Recompute individual CSV rows from the NPZ cache
#
# This is a direct formula check for the weighted per-fixation row:
#
# `sum(unit_bits * unit_expected_spikes * alignment_weight) / sum(unit_expected_spikes * alignment_weight)`

# %%
stats = load_npz_required(CACHE_NPZ)
show_table(
    pd.DataFrame(
        [
            {
                "key": key,
                "shape": str(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in stats.items()
            if key not in {"mean_rate_map"}
        ]
    )
)


def parse_weighted_unit_text(text: Any) -> tuple[np.ndarray, np.ndarray]:
    if text is None or (isinstance(text, float) and not np.isfinite(text)):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=np.float64)
    unit_indices: list[int] = []
    weights: list[float] = []
    for token in str(text).split():
        if ":" not in token:
            continue
        unit_text, weight_text = token.split(":", 1)
        unit_indices.append(int(unit_text.strip().lstrip("u")))
        weights.append(float(weight_text))
    return np.asarray(unit_indices, dtype=int), np.asarray(weights, dtype=np.float64)


def recompute_weighted_row(row: pd.Series, stats: dict[str, np.ndarray]) -> dict[str, Any]:
    metric_key = str(row.get("ssi_metric_cache_key", "unit_time_resolved_bits_per_movie"))
    bits = np.asarray(stats[metric_key], dtype=np.float64)
    spikes = np.asarray(stats["unit_expected_spikes_per_movie"], dtype=np.float64)
    rates = np.asarray(stats["unit_mean_rate_per_movie"], dtype=np.float64)
    cidx = int(row["condition_index"])
    midx = int(row["movie_index"])
    unit_idx, weights = parse_weighted_unit_text(row.get("unit_weights", ""))
    if unit_idx.size == 0:
        return {
            "n_units": 0,
            "numerator": 0.0,
            "denominator": 0.0,
            "population_bits_per_spike": float("nan"),
            "contributions": pd.DataFrame(),
        }
    unit_bits = bits[cidx, midx, unit_idx]
    unit_spikes = spikes[cidx, midx, unit_idx]
    unit_rates = rates[cidx, midx, unit_idx]
    weighted_spikes = unit_spikes * weights
    contributions = pd.DataFrame(
        {
            "unit_index": unit_idx,
            "alignment_weight": weights,
            "unit_bits_per_spike": unit_bits,
            "unit_expected_spikes": unit_spikes,
            "weighted_expected_spikes": weighted_spikes,
            "unit_mean_rate": unit_rates,
            "numerator_contribution": unit_bits * weighted_spikes,
        }
    )
    numerator = float(np.nansum(contributions["numerator_contribution"]))
    denominator = float(np.nansum(contributions["weighted_expected_spikes"]))
    return {
        "n_units": int(unit_idx.size),
        "numerator": numerator,
        "denominator": denominator,
        "population_bits_per_spike": numerator / max(denominator, EPS),
        "contributions": contributions,
    }


sample_candidates = focus_rows[
    (focus_rows["condition_id"].astype(str) == TARGET_CONDITION_ID)
    & (focus_rows["alignment_group"].astype(str) == "contour_orthogonal")
].copy()
if sample_candidates.empty:
    sample_candidates = focus_rows.copy()
sample_row = sample_candidates.sort_values("expected_spikes_arbitrary_dt", ascending=False).iloc[0]
sample_rebuilt = recompute_weighted_row(sample_row, stats)

row_check = pd.DataFrame(
    [
        {
            "view": TARGET_VIEW,
            "condition_id": sample_row["condition_id"],
            "movie_index": int(sample_row["movie_index"]),
            "sf_group": sample_row["sf_group"],
            "alignment_group": sample_row["alignment_group"],
            "csv_numerator": float(sample_row["information_numerator_bits_arbitrary_dt"]),
            "recomputed_numerator": sample_rebuilt["numerator"],
            "csv_denominator": float(sample_row["expected_spikes_arbitrary_dt"]),
            "recomputed_denominator": sample_rebuilt["denominator"],
            "csv_bits_per_spike": float(sample_row["population_bits_per_spike"]),
            "recomputed_bits_per_spike": sample_rebuilt["population_bits_per_spike"],
            "abs_ratio_error": abs(float(sample_row["population_bits_per_spike"]) - sample_rebuilt["population_bits_per_spike"]),
        }
    ]
)
show_table(row_check)
show_table(sample_rebuilt["contributions"].sort_values("numerator_contribution", ascending=False))

# %%
contrib = sample_rebuilt["contributions"].sort_values("numerator_contribution", ascending=False)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
axes[0].bar(contrib["unit_index"].astype(str), contrib["alignment_weight"], color="0.55")
axes[0].set_title("alignment weight")
axes[1].bar(contrib["unit_index"].astype(str), contrib["unit_bits_per_spike"], color="#168a96")
axes[1].set_title("unit bits/spike")
axes[2].bar(contrib["unit_index"].astype(str), contrib["numerator_contribution"], color="#c06b2d")
axes[2].set_title("numerator contribution")
for ax in axes:
    ax.set_xlabel("unit")
    ax.tick_params(axis="x", labelrotation=90)
fig.suptitle(
    f"One recomputed row: movie {int(sample_row['movie_index'])}, {sample_row['condition_id']}, "
    f"{sample_row['sf_group']}, {sample_row['alignment_group']}"
)
plt.show()

# %% [markdown]
# ## 13. Random row consistency check
#
# The previous cell checks one row in detail. This one checks a random batch of
# rows from the same view/band/SF.

# %%
rng = np.random.default_rng(11)
sample_n = min(40, int(focus_rows.shape[0]))
random_rows = focus_rows.sample(n=sample_n, random_state=11).reset_index(drop=True)
random_checks = []
for _, row in random_rows.iterrows():
    rebuilt = recompute_weighted_row(row, stats)
    random_checks.append(
        {
            "condition_id": row["condition_id"],
            "movie_index": int(row["movie_index"]),
            "sf_group": row["sf_group"],
            "alignment_group": row["alignment_group"],
            "numerator_error": abs(float(row["information_numerator_bits_arbitrary_dt"]) - rebuilt["numerator"]),
            "denominator_error": abs(float(row["expected_spikes_arbitrary_dt"]) - rebuilt["denominator"]),
            "ratio_error": abs(float(row["population_bits_per_spike"]) - rebuilt["population_bits_per_spike"]),
        }
    )
random_check_df = pd.DataFrame(random_checks)
show_table(random_check_df.sort_values("ratio_error", ascending=False).head(12))
show_table(
    pd.DataFrame(
        [
            {
                "n_checked": int(random_check_df.shape[0]),
                "max_numerator_error": float(random_check_df["numerator_error"].max()),
                "max_denominator_error": float(random_check_df["denominator_error"].max()),
                "max_ratio_error": float(random_check_df["ratio_error"].max()),
            }
        ]
    )
)

# %% [markdown]
# ## 14. Image-statistic checks inside the target band
#
# If near-vertical windows are systematically different images, these feature
# summaries should move. The balanced manifest intentionally controlled some of
# this, but the final dominant bands are narrower than the 30-degree bins.

# %%
feature_cols = [
    "image_patch_rms_contrast",
    "image_gradient_energy",
    "image_orientation_coherence",
    "image_oriented_gradient_energy",
    "image_edge_density",
    "image_high_freq_power_fraction",
    "image_power_8plus_cpd_fraction",
]
available_features = [col for col in feature_cols if col in axis_reference.columns]
feature_frame = axis_reference.copy()
for band in ["near_horizontal_axis0pm15", "near_vertical_axis90pm15"]:
    feature_frame[band] = band_mask(feature_frame["contour_axis_image_deg"].to_numpy(dtype=float), band)
feature_frame["dominant_band"] = np.select(
    [feature_frame["near_horizontal_axis0pm15"], feature_frame["near_vertical_axis90pm15"]],
    ["near_horizontal", "near_vertical"],
    default="other",
)

if available_features:
    summary_features = (
        feature_frame[feature_frame["dominant_band"].isin(["near_horizontal", "near_vertical"])]
        .groupby("dominant_band", sort=True)[available_features]
        .agg(["mean", "median", "std"])
    )
    show_table(summary_features)
else:
    print("No image feature columns were present in the per-fixation rows.")

# %%
if available_features:
    ncols = min(3, len(available_features))
    nrows = int(math.ceil(len(available_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, col in zip(axes, available_features, strict=False):
        data = [
            feature_frame.loc[feature_frame["dominant_band"] == "near_horizontal", col].dropna().to_numpy(dtype=float),
            feature_frame.loc[feature_frame["dominant_band"] == "near_vertical", col].dropna().to_numpy(dtype=float),
            feature_frame.loc[feature_frame["dominant_band"] == "other", col].dropna().to_numpy(dtype=float),
        ]
        ax.boxplot(data, labels=["horiz", "vert", "other"], showfliers=False)
        ax.set_title(col)
    for ax in axes[len(available_features) :]:
        ax.axis("off")
    fig.suptitle("Image feature distributions by contour-axis band")
    plt.show()

# %% [markdown]
# ## 15. Coarse orientation-bin view
#
# The near-vertical panel combines `75-105 deg`. These cells show how the
# aligned-minus-orthogonal contrast changes over all 30-degree bins.

# %%


def contrast_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    idx_cols = ["band", "view", "condition_id", "condition_index", "x_scale", "sf_group"]
    pivot = summary.pivot_table(
        index=idx_cols,
        columns="alignment_group",
        values="accumulated_bits_per_spike",
        aggfunc="first",
    ).reset_index()
    if "contour_aligned" in pivot.columns and "contour_orthogonal" in pivot.columns:
        pivot["aligned_minus_orthogonal"] = pivot["contour_aligned"] - pivot["contour_orthogonal"]
        pivot["orthogonal_minus_aligned"] = pivot["contour_orthogonal"] - pivot["contour_aligned"]
        pivot["orthogonal_over_aligned"] = pivot["contour_orthogonal"] / np.maximum(pivot["contour_aligned"], EPS)
    return pivot


contrast = contrast_from_summary(orientation_summary)
coarse = contrast[
    contrast["band"].astype(str).str.startswith("axis_bin_")
    & (contrast["view"].astype(str) == TARGET_VIEW)
    & (contrast["sf_group"].astype(str) == TARGET_SF)
].copy()
show_table(coarse.sort_values(["band", "x_scale"]))

# %%
if not coarse.empty:
    bands = sorted(coarse["band"].astype(str).unique())
    x_values = sorted(coarse["x_scale"].dropna().unique())
    mat = np.full((len(bands), len(x_values)), np.nan, dtype=np.float64)
    for i, band in enumerate(bands):
        for j, x in enumerate(x_values):
            sub = coarse[(coarse["band"].astype(str) == band) & np.isclose(coarse["x_scale"], x)]
            if not sub.empty:
                mat[i, j] = float(sub["orthogonal_minus_aligned"].iloc[0])
    vmax = max(float(np.nanpercentile(np.abs(mat), 98)), 1e-6)
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(bands)))
    ax.set_yticklabels([band.replace("axis_bin_", "").replace("_", "-") for band in bands])
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([scale_label(x) for x in x_values])
    ax.set_xlabel("scale")
    ax.set_ylabel("contour-axis bin (deg)")
    ax.set_title(f"Orthogonal minus aligned bits/spike: {TARGET_VIEW}, {TARGET_SF}")
    fig.colorbar(im, ax=ax, label="orthogonal - aligned")
    plt.show()

# %% [markdown]
# ## 16. Dominant-band high-SF discrepancy in image-frame coordinates
#
# The two dominant-band plots should be read together:
#
# - near-horizontal contour axes: `contour_orthogonal` targets image-vertical
#   structure.
# - near-vertical contour axes: `contour_aligned` targets image-vertical
#   structure.
#
# So if high SF is large for near-horizontal/orthogonal and
# near-vertical/aligned, the common explanation is not the contour-relative
# label. It is an image-frame vertical-vs-horizontal asymmetry in the high-SF
# weighted pool.

# %%
DOMINANT_BANDS = ["near_horizontal_axis0pm15", "near_vertical_axis90pm15"]
DOMINANT_BAND_LABEL = {
    "near_horizontal_axis0pm15": "near-horizontal contours",
    "near_vertical_axis90pm15": "near-vertical contours",
}


def image_vertical_minus_horizontal_contrast(summary: pd.DataFrame, *, sf_group: str = "high_sf") -> pd.DataFrame:
    dominant = summary[
        summary["band"].astype(str).isin(DOMINANT_BANDS)
        & (summary["sf_group"].astype(str) == sf_group)
    ].copy()
    out = contrast_from_summary(dominant)
    out["dominant_band_label"] = out["band"].map(DOMINANT_BAND_LABEL).fillna(out["band"])
    out["image_vertical_bits_per_spike"] = np.where(
        out["band"].astype(str) == "near_horizontal_axis0pm15",
        out["contour_orthogonal"],
        out["contour_aligned"],
    )
    out["image_horizontal_bits_per_spike"] = np.where(
        out["band"].astype(str) == "near_horizontal_axis0pm15",
        out["contour_aligned"],
        out["contour_orthogonal"],
    )
    out["image_vertical_minus_horizontal"] = (
        out["image_vertical_bits_per_spike"] - out["image_horizontal_bits_per_spike"]
    )
    out["image_vertical_over_horizontal"] = out["image_vertical_bits_per_spike"] / np.maximum(
        out["image_horizontal_bits_per_spike"],
        EPS,
    )
    return out


dominant_high_sf = image_vertical_minus_horizontal_contrast(orientation_summary, sf_group="high_sf")
show_table(
    dominant_high_sf[
        (dominant_high_sf["view"].astype(str) == TARGET_VIEW)
        & np.isclose(dominant_high_sf["x_scale"].to_numpy(dtype=float), 1.0)
    ][
        [
            "dominant_band_label",
            "view",
            "condition_id",
            "x_scale",
            "image_vertical_bits_per_spike",
            "image_horizontal_bits_per_spike",
            "image_vertical_minus_horizontal",
            "image_vertical_over_horizontal",
        ]
    ].sort_values(["dominant_band_label", "condition_id"])
)

# %%
fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.6), sharey=True, constrained_layout=True)
band_styles = {
    "near_horizontal_axis0pm15": {"color": "#8c5a2b", "linestyle": "-", "marker": "o"},
    "near_vertical_axis90pm15": {"color": "#236b8e", "linestyle": "-", "marker": "s"},
}
for ax, spec in zip(axes, VIEW_SPECS, strict=True):
    view = str(spec["view"])
    sub_view = dominant_high_sf[dominant_high_sf["view"].astype(str) == view].copy()
    for band in DOMINANT_BANDS:
        sub = sub_view[sub_view["band"].astype(str) == band].sort_values(["x_scale", "condition_index"])
        if sub.empty:
            continue
        style = band_styles[band]
        ax.plot(
            sub["x_scale"].to_numpy(dtype=float),
            sub["image_vertical_minus_horizontal"].to_numpy(dtype=float),
            linewidth=2.0,
            label=DOMINANT_BAND_LABEL[band],
            **style,
        )
    ax.axhline(0.0, color="0.35", linewidth=1.0)
    ax.axvline(1.0, color="0.62", linestyle=":", linewidth=0.9)
    ax.set_title(str(spec["label"]), fontsize=9.5)
    ax.set_xlabel("scale")
axes[0].set_ylabel("high SF bits/spike\nimage vertical - image horizontal")
axes[-1].legend(frameon=False, fontsize=8)
fig.suptitle("High-SF discrepancy reframed as image-frame target orientation")
plt.show()

# %%
dominant_x1 = dominant_high_sf[np.isclose(dominant_high_sf["x_scale"].to_numpy(dtype=float), 1.0)].copy()
show_table(
    dominant_x1.groupby(["band", "dominant_band_label"], sort=True)
    .agg(
        n_views=("view", "nunique"),
        mean_vertical_minus_horizontal=("image_vertical_minus_horizontal", "mean"),
        min_vertical_minus_horizontal=("image_vertical_minus_horizontal", "min"),
        max_vertical_minus_horizontal=("image_vertical_minus_horizontal", "max"),
        mean_vertical_over_horizontal=("image_vertical_over_horizontal", "mean"),
    )
    .reset_index()
)

# %% [markdown]
# ## 17. Pick exemplar fixations and recreate the effect from a tiny subset
#
# This is an intentionally small, visual sanity subset. It is not a replacement
# for the aggregate bootstrap. We pick a few near-horizontal and near-vertical
# contour windows with a positive high-SF image-vertical advantage at
# `EXAMPLE_X_SCALE`, display their local patches, then rerun the same
# numerator/denominator aggregation using only those source rows.

# %%
selected_windows = read_csv_required(SELECTED_WINDOWS_CSV) if SELECTED_WINDOWS_CSV.exists() else pd.DataFrame()


def alignment_axis_image_deg_for_rows(frame: pd.DataFrame) -> np.ndarray:
    contour_axis = frame["contour_axis_image_deg"].to_numpy(dtype=float) % 180.0
    aligned = frame["alignment_group"].astype(str).to_numpy() == "contour_aligned"
    return np.where(aligned, contour_axis, (contour_axis + 90.0) % 180.0)


def image_orientation_pool_for_axes(axis_deg: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_deg, dtype=np.float64) % 180.0
    vertical_dist = angle_180_distance(axis, 90.0)
    horizontal_dist = angle_180_distance(axis, 0.0)
    return np.where(vertical_dist <= horizontal_dist, "image_vertical", "image_horizontal")


def image_orientation_rows(
    frame: pd.DataFrame,
    *,
    sf_group: str = "high_sf",
    source_rows: set[int] | None = None,
    x_scale: float | None = None,
) -> pd.DataFrame:
    out = frame[frame["sf_group"].astype(str) == str(sf_group)].copy()
    if source_rows is not None:
        out = out[out["source_row"].astype(int).isin({int(v) for v in source_rows})].copy()
    if x_scale is not None:
        out = out[np.isclose(out["x_scale"].to_numpy(dtype=float), float(x_scale))].copy()
    if out.empty:
        return out
    out["alignment_axis_image_deg"] = alignment_axis_image_deg_for_rows(out)
    out["image_orientation_pool"] = image_orientation_pool_for_axes(out["alignment_axis_image_deg"].to_numpy(dtype=float))
    out["dominant_band"] = np.select(
        [
            band_mask(out["contour_axis_image_deg"].to_numpy(dtype=float), "near_horizontal_axis0pm15"),
            band_mask(out["contour_axis_image_deg"].to_numpy(dtype=float), "near_vertical_axis90pm15"),
        ],
        ["near_horizontal_axis0pm15", "near_vertical_axis90pm15"],
        default="other",
    )
    out["dominant_band_label"] = out["dominant_band"].map(DOMINANT_BAND_LABEL).fillna(out["dominant_band"])
    return out


def per_movie_image_orientation_contrast(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    id_cols = [
        "view",
        "condition_id",
        "condition_index",
        "x_scale",
        "movie_index",
        "trial_id",
        "source_row",
        "session",
        "trial_idx",
        "contour_axis_image_deg",
        "dominant_band",
        "dominant_band_label",
    ]
    value_cols = [
        "population_bits_per_spike",
        "information_numerator_bits_arbitrary_dt",
        "expected_spikes_arbitrary_dt",
    ]
    available_id_cols = [col for col in id_cols if col in rows.columns]
    tidy = rows[available_id_cols + ["image_orientation_pool", *value_cols]].copy()
    pivot = tidy.pivot_table(
        index=available_id_cols,
        columns="image_orientation_pool",
        values=value_cols,
        aggfunc="first",
    )
    pivot.columns = [f"{value}_{pool}" for value, pool in pivot.columns]
    pivot = pivot.reset_index()
    v_col = "population_bits_per_spike_image_vertical"
    h_col = "population_bits_per_spike_image_horizontal"
    if v_col in pivot.columns and h_col in pivot.columns:
        pivot["image_vertical_minus_horizontal"] = pivot[v_col] - pivot[h_col]
        pivot["image_vertical_over_horizontal"] = pivot[v_col] / np.maximum(pivot[h_col], EPS)
    return pivot


def summarize_image_orientation_curves(rows: pd.DataFrame, *, extra_group_cols: list[str] | None = None) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    extra_group_cols = list(extra_group_cols or [])
    group_cols = [
        *extra_group_cols,
        "view",
        "condition_id",
        "condition_index",
        "x_scale",
        "image_orientation_pool",
    ]
    grouped = (
        rows.groupby(group_cols, sort=True)[
            ["information_numerator_bits_arbitrary_dt", "expected_spikes_arbitrary_dt", "movie_index"]
        ]
        .agg(
            information_numerator_bits_arbitrary_dt=("information_numerator_bits_arbitrary_dt", "sum"),
            expected_spikes_arbitrary_dt=("expected_spikes_arbitrary_dt", "sum"),
            n_fixations=("movie_index", "nunique"),
        )
        .reset_index()
    )
    grouped["accumulated_bits_per_spike"] = grouped["information_numerator_bits_arbitrary_dt"] / np.maximum(
        grouped["expected_spikes_arbitrary_dt"],
        EPS,
    )
    idx_cols = [*extra_group_cols, "view", "condition_id", "condition_index", "x_scale"]
    pivot = grouped.pivot_table(
        index=idx_cols,
        columns="image_orientation_pool",
        values=["accumulated_bits_per_spike", "n_fixations"],
        aggfunc="first",
    ).reset_index()
    pivot.columns = [f"{value}_{pool}" if pool else str(value) for value, pool in pivot.columns]
    if {
        "accumulated_bits_per_spike_image_vertical",
        "accumulated_bits_per_spike_image_horizontal",
    }.issubset(pivot.columns):
        pivot["image_vertical_minus_horizontal"] = (
            pivot["accumulated_bits_per_spike_image_vertical"]
            - pivot["accumulated_bits_per_spike_image_horizontal"]
        )
        pivot["image_vertical_over_horizontal"] = pivot["accumulated_bits_per_spike_image_vertical"] / np.maximum(
            pivot["accumulated_bits_per_spike_image_horizontal"],
            EPS,
        )
    return pivot


example_rows = image_orientation_rows(
    weighted_by_view[TARGET_VIEW],
    sf_group="high_sf",
    x_scale=EXAMPLE_X_SCALE,
)
example_per_movie = per_movie_image_orientation_contrast(example_rows)
example_per_movie = example_per_movie[
    example_per_movie["dominant_band"].astype(str).isin(DOMINANT_BANDS)
    & np.isfinite(example_per_movie["image_vertical_minus_horizontal"].to_numpy(dtype=float))
].copy()

if not selected_windows.empty and not example_per_movie.empty:
    manifest_cols = [
        col
        for col in [
            "source_row",
            "balanced_manifest_index",
            "image_index",
            "image_patch_center_x_px",
            "image_patch_center_y_px",
            "image_patch_radius_px",
            "image_patch_rms_contrast",
            "image_orientation_coherence",
            "image_oriented_gradient_energy",
            "image_power_8plus_cpd_fraction",
            "image_high_freq_power_fraction",
        ]
        if col in selected_windows.columns
    ]
    example_per_movie = example_per_movie.merge(
        selected_windows[manifest_cols].drop_duplicates("source_row"),
        on="source_row",
        how="left",
    )

selected_example_parts: list[pd.DataFrame] = []
for band in DOMINANT_BANDS:
    sub = example_per_movie[example_per_movie["dominant_band"].astype(str) == band].copy()
    sub = sub.sort_values(
        ["image_vertical_minus_horizontal", "image_vertical_over_horizontal"],
        ascending=[False, False],
    )
    selected_example_parts.append(sub.head(max(1, int(EXAMPLE_MOVIES_PER_BAND))).copy())
selected_examples = pd.concat(selected_example_parts, ignore_index=True) if selected_example_parts else pd.DataFrame()
selected_examples["example_band"] = selected_examples.get("dominant_band", pd.Series(dtype=str))
selected_examples["example_band_label"] = selected_examples.get("dominant_band_label", pd.Series(dtype=str))

show_table(
    selected_examples[
        [
            col
            for col in [
                "example_band_label",
                "source_row",
                "movie_index",
                "trial_idx",
                "contour_axis_image_deg",
                "image_vertical_minus_horizontal",
                "image_vertical_over_horizontal",
                "population_bits_per_spike_image_vertical",
                "population_bits_per_spike_image_horizontal",
                "image_patch_rms_contrast",
                "image_orientation_coherence",
                "image_power_8plus_cpd_fraction",
            ]
            if col in selected_examples.columns
        ]
    ]
)

# %%


def load_example_patch(row: pd.Series, *, thumbnail_size_px: int = 176) -> np.ndarray | None:
    try:
        from declan.fixation_statistics_by_stimulus.image_features import _backimage_canvas
        from declan.fixation_statistics_by_stimulus.run_backimage_twin_drift_geometry import _clip_patch

        canvas, _ppd, _shape = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
        cx = float(row["image_patch_center_x_px"])
        cy = float(row["image_patch_center_y_px"])
        return _clip_patch(canvas, (cx, cy), int(thumbnail_size_px))
    except Exception as exc:
        print(f"Could not load BackImage patch for source_row={row.get('source_row')}: {type(exc).__name__}: {exc}")
        return None


def plot_example_patch_gallery(examples: pd.DataFrame) -> None:
    if examples.empty:
        print("No selected examples to display.")
        return
    if not SHOW_EXAMPLE_PATCHES:
        print("Patch gallery skipped. Set BACKIMAGE_RR100_SHOW_PATCHES=1 or run interactively to render thumbnails.")
        return
    n = int(examples.shape[0])
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.35 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, (_, row) in zip(axes, examples.iterrows(), strict=False):
        patch = load_example_patch(row)
        if patch is None:
            ax.axis("off")
            continue
        ax.imshow(patch, cmap="gray", interpolation="nearest")
        cy = (patch.shape[0] - 1) / 2.0
        cx = (patch.shape[1] - 1) / 2.0
        theta = np.deg2rad(float(row["contour_axis_image_deg"]))
        length = 0.42 * min(patch.shape)
        dx = math.cos(theta) * length
        dy = math.sin(theta) * length
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color="#00bcd4", linewidth=2.0)
        ax.scatter([cx], [cy], s=22, color="#ffca28", edgecolor="black", linewidth=0.4)
        ax.set_title(
            f"{row['example_band_label']}\n"
            f"src {int(row['source_row'])}, V-H {float(row['image_vertical_minus_horizontal']):+.3f}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Selected local BackImage patches; cyan line is contour axis", fontsize=11)
    plt.show()


plot_example_patch_gallery(selected_examples)

# %%
example_source_rows = {int(v) for v in selected_examples.get("source_row", pd.Series(dtype=int)).dropna().to_list()}
example_source_to_band = selected_examples[["source_row", "example_band", "example_band_label"]].drop_duplicates("source_row")

subset_parts = []
for spec in VIEW_SPECS:
    rows = image_orientation_rows(
        weighted_by_view[str(spec["view"])],
        sf_group="high_sf",
        source_rows=example_source_rows,
    )
    if rows.empty:
        continue
    rows = rows.merge(example_source_to_band, on="source_row", how="inner")
    subset_parts.append(rows)
subset_rows = pd.concat(subset_parts, ignore_index=True) if subset_parts else pd.DataFrame()
subset_curves = summarize_image_orientation_curves(
    subset_rows,
    extra_group_cols=["example_band", "example_band_label"],
)

full_image_orientation_rows = []
for spec in VIEW_SPECS:
    full_image_orientation_rows.append(image_orientation_rows(weighted_by_view[str(spec["view"])], sf_group="high_sf"))
full_image_orientation_rows_df = pd.concat(full_image_orientation_rows, ignore_index=True)
full_dominant_rows = full_image_orientation_rows_df[
    full_image_orientation_rows_df["dominant_band"].astype(str).isin(DOMINANT_BANDS)
].copy()
full_dominant_rows["example_band"] = full_dominant_rows["dominant_band"]
full_dominant_rows["example_band_label"] = full_dominant_rows["dominant_band_label"]
full_dominant_curves = summarize_image_orientation_curves(
    full_dominant_rows,
    extra_group_cols=["example_band", "example_band_label"],
)

subset_x1 = subset_curves[np.isclose(subset_curves["x_scale"].to_numpy(dtype=float), EXAMPLE_X_SCALE)].copy()
full_x1 = full_dominant_curves[np.isclose(full_dominant_curves["x_scale"].to_numpy(dtype=float), EXAMPLE_X_SCALE)].copy()
subset_x1["source"] = "tiny_example_subset"
full_x1["source"] = "full_dominant_band"
show_table(
    pd.concat([full_x1, subset_x1], ignore_index=True)[
        [
            col
            for col in [
                "source",
                "example_band_label",
                "view",
                "condition_id",
                "x_scale",
                "n_fixations_image_vertical",
                "n_fixations_image_horizontal",
                "accumulated_bits_per_spike_image_vertical",
                "accumulated_bits_per_spike_image_horizontal",
                "image_vertical_minus_horizontal",
                "image_vertical_over_horizontal",
            ]
            if col in subset_x1.columns or col in full_x1.columns
        ]
    ].sort_values(["view", "example_band_label", "source"])
)

# %%
if not subset_curves.empty:
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.7), sharey=True, constrained_layout=True)
    for ax, spec in zip(axes, VIEW_SPECS, strict=True):
        view = str(spec["view"])
        for band in DOMINANT_BANDS:
            full_sub = full_dominant_curves[
                (full_dominant_curves["view"].astype(str) == view)
                & (full_dominant_curves["example_band"].astype(str) == band)
            ].sort_values(["x_scale", "condition_index"])
            tiny_sub = subset_curves[
                (subset_curves["view"].astype(str) == view)
                & (subset_curves["example_band"].astype(str) == band)
            ].sort_values(["x_scale", "condition_index"])
            if not full_sub.empty:
                ax.plot(
                    full_sub["x_scale"],
                    full_sub["image_vertical_minus_horizontal"],
                    color=band_styles[band]["color"],
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.35,
                    label=f"full {DOMINANT_BAND_LABEL[band]}" if view == VIEW_SPECS[0]["view"] else None,
                )
            if not tiny_sub.empty:
                ax.plot(
                    tiny_sub["x_scale"],
                    tiny_sub["image_vertical_minus_horizontal"],
                    color=band_styles[band]["color"],
                    marker=band_styles[band]["marker"],
                    linestyle="--",
                    linewidth=2.0,
                    label=f"tiny {DOMINANT_BAND_LABEL[band]}" if view == VIEW_SPECS[0]["view"] else None,
                )
        ax.axhline(0.0, color="0.35", linewidth=1.0)
        ax.axvline(1.0, color="0.62", linestyle=":", linewidth=0.9)
        ax.set_title(str(spec["label"]), fontsize=9.5)
        ax.set_xlabel("scale")
    axes[0].set_ylabel("high SF bits/spike\nimage vertical - image horizontal")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Tiny exemplar subset ({EXAMPLE_MOVIES_PER_BAND} per band) vs full dominant-band effect")
    plt.show()

# %% [markdown]
# ## 18. Alternative label/frame stress tests
#
# These do not change the underlying data. They ask whether the surprising
# story depends on a naming convention:
#
# - What if we read the same target rows with aligned/orthogonal swapped?
# - What if we accidentally banded by raw `axis_deg` instead of image-frame
#   contour axis?

# %%
swapped = rebuilt_target_band.copy()
swapped["alignment_group"] = swapped["alignment_group"].replace(
    {"contour_aligned": "contour_orthogonal", "contour_orthogonal": "contour_aligned"}
)
plot_orientation_curves(
    swapped,
    title=f"Same numbers with alignment labels intentionally swapped: {TARGET_BAND}",
)

# %%


def summarize_with_raw_axis_band(frame: pd.DataFrame, *, view: str, view_label: str, x_col: str, band: str) -> pd.DataFrame:
    axes = frame[["movie_index", "axis_deg"]].drop_duplicates("movie_index")
    raw_axis_mod = orientation_axis_180(axes["axis_deg"].to_numpy(dtype=float))
    keep_movies = set(axes.loc[band_mask(raw_axis_mod, band), "movie_index"].astype(int).tolist())
    filtered = frame[frame["movie_index"].astype(int).isin(keep_movies)].copy()
    group_cols = ["condition_id", "condition_index", x_col, "sf_group", "alignment_group"]
    rows: list[dict[str, Any]] = []
    for keys, sub in filtered.groupby(group_cols, sort=True):
        condition_id, condition_index, x_value, sf_group, alignment_group = keys
        numerator = float(np.nansum(sub["information_numerator_bits_arbitrary_dt"].to_numpy(dtype=float)))
        denominator = float(np.nansum(sub["expected_spikes_arbitrary_dt"].to_numpy(dtype=float)))
        rows.append(
            {
                "band": f"{band}_raw_axis_deg",
                "view": view,
                "view_label": view_label,
                "condition_id": str(condition_id),
                "condition_index": int(condition_index),
                "x_scale": float(x_value),
                "sf_group": str(sf_group),
                "alignment_group": str(alignment_group),
                "n_fixations": int(sub["movie_index"].nunique()),
                "accumulated_bits_per_spike": numerator / max(denominator, EPS),
                "expected_spikes_sum_arbitrary_dt": denominator,
                "information_numerator_sum_bits_arbitrary_dt": numerator,
            }
        )
    return pd.DataFrame(rows)


raw_axis_parts = []
for spec in VIEW_SPECS:
    raw_axis_parts.append(
        summarize_with_raw_axis_band(
            weighted_by_view[str(spec["view"])],
            view=str(spec["view"]),
            view_label=str(spec["label"]),
            x_col=str(spec["x_col"]),
            band=TARGET_BAND,
        )
    )
raw_axis_summary = pd.concat(raw_axis_parts, ignore_index=True)
plot_orientation_curves(
    raw_axis_summary,
    title=f"Stress test: banded by raw axis_deg instead of image-frame contour axis: {TARGET_BAND}",
)

# %% [markdown]
# ## 19. A compact checklist of red flags
#
# This cell does not decide whether the result is genuine. It summarizes the
# parts that would be suspicious if nonzero or extremely imbalanced.

# %%
target_axis_mask = band_mask(axis_reference["contour_axis_image_deg"].to_numpy(dtype=float), TARGET_BAND)
target_movies = set(axis_reference.loc[target_axis_mask, "movie_index"].astype(int))
focus_selection = selection_by_view[TARGET_VIEW]
focus_selection = focus_selection[
    focus_selection["movie_index"].astype(int).isin(target_movies)
    & (focus_selection["sf_group"].astype(str) == TARGET_SF)
].copy()
weight_balance = focus_selection.pivot_table(
    index="movie_index",
    columns="alignment_group",
    values="weight_sum",
    aggfunc="first",
)
effective_balance = focus_selection.pivot_table(
    index="movie_index",
    columns="alignment_group",
    values="effective_n_units",
    aggfunc="first",
)

red_flags = pd.DataFrame(
    [
        {
            "check": "saved summary exactly rebuilt from per-fixation CSV",
            "value": float(np.nanmax(comparison["accumulated_bits_per_spike_abs_diff"])),
            "status": "ok" if float(np.nanmax(comparison["accumulated_bits_per_spike_abs_diff"])) < 1e-12 else "inspect",
        },
        {
            "check": "axis frame recompute max error deg",
            "value": float(np.nanmax(axis_frame["axis_recompute_error_deg"])),
            "status": "ok" if float(np.nanmax(axis_frame["axis_recompute_error_deg"])) < 1e-9 else "inspect",
        },
        {
            "check": "target band n movies",
            "value": int(len(target_movies)),
            "status": "ok" if len(target_movies) >= 20 else "small support",
        },
        {
            "check": f"{TARGET_VIEW} {TARGET_SF} mean aligned weight_sum",
            "value": float(np.nanmean(weight_balance.get("contour_aligned", pd.Series(dtype=float)).to_numpy(dtype=float))),
            "status": "context",
        },
        {
            "check": f"{TARGET_VIEW} {TARGET_SF} mean orthogonal weight_sum",
            "value": float(np.nanmean(weight_balance.get("contour_orthogonal", pd.Series(dtype=float)).to_numpy(dtype=float))),
            "status": "context",
        },
        {
            "check": f"{TARGET_VIEW} {TARGET_SF} mean aligned effective n",
            "value": float(np.nanmean(effective_balance.get("contour_aligned", pd.Series(dtype=float)).to_numpy(dtype=float))),
            "status": "context",
        },
        {
            "check": f"{TARGET_VIEW} {TARGET_SF} mean orthogonal effective n",
            "value": float(np.nanmean(effective_balance.get("contour_orthogonal", pd.Series(dtype=float)).to_numpy(dtype=float))),
            "status": "context",
        },
        {
            "check": "random row max ratio recompute error",
            "value": float(random_check_df["ratio_error"].max()),
            "status": "ok" if float(random_check_df["ratio_error"].max()) < 5e-4 else "inspect rounded unit_weights",
        },
    ]
)
show_table(red_flags)

# %% [markdown]
# ## 20. Targeted tiny rotated-movie run
#
# The previous cell recreates the high-SF discrepancy from a very small set of
# exemplar fixations. This cell uses that same small source-row set to make a
# cheap 90-degree movie-rotation control, without waiting for the full 576-movie
# rot90 run.
#
# By default the cell only prepares the tiny manifest and prints the commands.
# To actually run the scorer from this notebook/script, launch with:
#
# ```bash
# BACKIMAGE_RR100_RUN_TARGETED_ROTATION=1 python notebooks/backimage_rr100_orientation_stratified_ssi_walkthrough.py
# ```
#
# The output root defaults to:
#
# `RUN_ROOT / f"tutorial_tiny_exemplar_rot{TARGETED_ROTATION_DEG}"`
#
# Useful knobs:
#
# - `BACKIMAGE_RR100_TARGETED_DEVICE`: override the device, for example
#   `cuda:1` or `cpu`.
# - `BACKIMAGE_RR100_FORCE_TARGETED_ROTATION=1`: recompute even if cached tiny
#   outputs already exist.
# - `BACKIMAGE_RR100_TARGETED_N_BOOTSTRAP`: bootstrap count for the tiny posthoc
#   summaries. The default is `0` because the per-fixation rows are the target.

# %%


def json_ready_payload(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready_payload(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready_payload(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_arg(command: list[Any], flag: str, default: str | None = None) -> str | None:
    tokens = [str(token) for token in command]
    for idx, token in enumerate(tokens[:-1]):
        if token == str(flag):
            return tokens[idx + 1]
    return default


def contour_command_from_manifest(commands: dict[str, Any]) -> list[str]:
    command = commands.get("commands", {}).get("contour", [])
    return [str(token) for token in command] if command else []


def condition_pairs_from_frame(frame: pd.DataFrame) -> str:
    cond = (
        frame[["condition_index", "condition_id", "along_scale", "across_scale"]]
        .drop_duplicates("condition_index")
        .sort_values("condition_index")
    )
    return ",".join(f"{float(row.along_scale):g}:{float(row.across_scale):g}" for row in cond.itertuples(index=False))


def prepare_targeted_rotation_source_dir(
    *,
    examples: pd.DataFrame,
    selected_source: pd.DataFrame,
    source_dir: Path,
) -> Path:
    if examples.empty:
        raise ValueError("No selected examples are available for the targeted rotation run.")
    if selected_source.empty:
        raise ValueError("selected_windows.csv is required to build the targeted rotation manifest.")

    source_ids = [int(v) for v in examples["source_row"].dropna().astype(int).drop_duplicates().to_list()]
    subset = selected_source[selected_source["source_row"].astype(int).isin(source_ids)].copy()
    missing = sorted(set(source_ids).difference(set(subset["source_row"].astype(int).to_list())))
    if missing:
        raise ValueError(f"Missing selected_windows rows for exemplar source_row values: {missing}")

    order = pd.DataFrame({"source_row": source_ids, "_example_order": np.arange(len(source_ids), dtype=int)})
    subset = subset.merge(order, on="source_row", how="left").sort_values("_example_order").drop(columns=["_example_order"])
    source_dir.mkdir(parents=True, exist_ok=True)
    selected_path = source_dir / "selected_windows.csv"
    subset.to_csv(selected_path, index=False)

    metadata_path = BALANCED_SOURCE_DIR / "run_metadata.json"
    metadata = read_json_required(metadata_path) if metadata_path.exists() else {"config": {}}
    metadata = dict(metadata)
    metadata["analysis"] = "backimage_contour_axis_tiny_exemplar_manifest"
    metadata["parent_selected_windows_csv"] = SELECTED_WINDOWS_CSV
    metadata["selected_windows_csv"] = selected_path
    metadata["n_selected_windows"] = int(subset.shape[0])
    metadata["source_rows"] = source_ids
    cfg = dict(metadata.get("config", {}))
    cfg["out_dir"] = str(source_dir)
    cfg["input"] = str(SELECTED_WINDOWS_CSV)
    cfg["target_per_bin"] = int(subset.shape[0])
    metadata["config"] = cfg
    write_json_payload(source_dir / "run_metadata.json", metadata)
    return selected_path


def targeted_population_commands(contour_dir: Path, out_root: Path, *, n_bootstrap: int) -> list[list[str]]:
    specs = [
        ("population_across_sweep_along0", "across", "--fixed-along-scale", "0"),
        ("population_across_sweep_along1", "across", "--fixed-along-scale", "1"),
        ("population_along_sweep_across0", "along", "--fixed-across-scale", "0"),
        ("population_along_sweep_across1", "along", "--fixed-across-scale", "1"),
    ]
    commands_out: list[list[str]] = []
    for out_name, sweep_axis, fixed_flag, fixed_value in specs:
        commands_out.append(
            [
                sys.executable,
                "declan/active_sensing_movie_information/plot_backimage_rr100_sf_contour_alignment_population_ssi.py",
                "--contour-run-dir",
                str(contour_dir),
                "--sf-groups-csv",
                str(SF_GROUPS_CSV),
                "--out-dir",
                str(out_root / out_name),
                "--ssi-metric",
                "time_resolved",
                "--sf-groups",
                "low_sf,high_sf",
                "--sweep-axis",
                sweep_axis,
                fixed_flag,
                fixed_value,
                "--n-bootstrap",
                str(int(n_bootstrap)),
            ]
        )
    return commands_out


def build_targeted_rotation_commands(
    *,
    out_root: Path,
    source_dir: Path,
    selected_path: Path,
    rotation_deg: int,
) -> tuple[list[str], list[list[str]]]:
    contour_dir = out_root / "contour_rr100_spatial_ssi_pairs27"
    manifest_contour = contour_command_from_manifest(COMMANDS)
    device = TARGETED_ROTATION_DEVICE or command_arg(manifest_contour, "--device", "cuda:0") or "cuda:0"
    top_units = command_arg(manifest_contour, "--top-units", "12") or "12"
    condition_pairs = str(COMMANDS.get("condition_pairs") or condition_pairs_from_frame(weighted_by_view[TARGET_VIEW]))
    contour_cmd = [
        sys.executable,
        "declan/active_sensing_movie_information/run_backimage_contour_axis_rr100_spatial_ssi.py",
        "--axis-run-dir",
        str(source_dir),
        "--selected-windows-csv",
        str(selected_path),
        "--trial-source-mode",
        "selected_windows",
        "--out-dir",
        str(contour_dir),
        "--sweep-mode",
        "pairs",
        "--condition-pairs",
        condition_pairs,
        "--max-trials",
        "0",
        "--primary-ssi-metric",
        "time_resolved",
        "--device",
        str(device),
        "--batch-size",
        str(int(TARGETED_ROTATION_BATCH_SIZE)),
        "--top-units",
        str(top_units),
        "--stimulus-rotation-deg",
        str(int(rotation_deg)),
        "--no-write-zscore-plot",
    ]
    return contour_cmd, targeted_population_commands(
        contour_dir,
        out_root,
        n_bootstrap=int(TARGETED_ROTATION_N_BOOTSTRAP),
    )


def targeted_rotation_population_dirs(out_root: Path) -> dict[str, Path]:
    return {
        "across_along0": out_root / "population_across_sweep_along0",
        "across_along1": out_root / "population_across_sweep_along1",
        "along_across0": out_root / "population_along_sweep_across0",
        "along_across1": out_root / "population_along_sweep_across1",
    }


def targeted_rotation_ready(out_root: Path) -> bool:
    return all(
        (path / "per_fixation_weighted_alignment_population_ssi.csv").exists()
        for path in targeted_rotation_population_dirs(out_root).values()
    )


def run_command_checked(command: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    print(" ".join(str(token) for token in command), flush=True)
    subprocess.run([str(token) for token in command], cwd=str(ROOT), env=env, check=True)


tiny_source_dir = TARGETED_ROTATION_RUN_ROOT / "selected_example_windows"
tiny_selected_windows_csv: Path | None = None
targeted_contour_cmd: list[str] = []
targeted_population_cmds: list[list[str]] = []
if not selected_examples.empty and not selected_windows.empty:
    tiny_selected_windows_csv = prepare_targeted_rotation_source_dir(
        examples=selected_examples,
        selected_source=selected_windows,
        source_dir=tiny_source_dir,
    )
    targeted_contour_cmd, targeted_population_cmds = build_targeted_rotation_commands(
        out_root=TARGETED_ROTATION_RUN_ROOT,
        source_dir=tiny_source_dir,
        selected_path=tiny_selected_windows_csv,
        rotation_deg=int(TARGETED_ROTATION_DEG),
    )
    write_json_payload(
        TARGETED_ROTATION_RUN_ROOT / "targeted_rotation_commands.json",
        {
            "analysis": "backimage_rr100_orientation_tiny_exemplar_rotation",
            "out_root": TARGETED_ROTATION_RUN_ROOT,
            "source_dir": tiny_source_dir,
            "selected_windows_csv": tiny_selected_windows_csv,
            "source_rows": sorted(example_source_rows),
            "stimulus_rotation_deg": int(TARGETED_ROTATION_DEG),
            "commands": {
                "contour": targeted_contour_cmd,
                "population_plots": targeted_population_cmds,
            },
        },
    )

if tiny_selected_windows_csv is None:
    print("Targeted rotation manifest was not prepared because selected examples or selected_windows are missing.")
elif RUN_TARGETED_ROTATION:
    contour_cache = TARGETED_ROTATION_RUN_ROOT / "contour_rr100_spatial_ssi_pairs27/cache/backimage_contour_axis_rr100_spatial_ssi_cache.npz"
    if FORCE_TARGETED_ROTATION or not contour_cache.exists():
        run_command_checked(targeted_contour_cmd)
    else:
        print(f"Using existing tiny contour cache: {contour_cache}")
    for cmd in targeted_population_cmds:
        out_dir = Path(command_arg(cmd, "--out-dir", "") or "")
        per_fix_path = out_dir / "per_fixation_weighted_alignment_population_ssi.csv"
        if FORCE_TARGETED_ROTATION or not per_fix_path.exists():
            run_command_checked(cmd)
        else:
            print(f"Using existing tiny population rows: {per_fix_path}")
elif targeted_rotation_ready(TARGETED_ROTATION_RUN_ROOT):
    print(
        f"Using existing tiny targeted rotated outputs: {TARGETED_ROTATION_RUN_ROOT}\n"
        "Set BACKIMAGE_RR100_FORCE_TARGETED_ROTATION=1 with "
        "BACKIMAGE_RR100_RUN_TARGETED_ROTATION=1 to recompute them."
    )
else:
    print(
        "Prepared a tiny rotated-movie manifest but did not run the scorer. "
        "Set BACKIMAGE_RR100_RUN_TARGETED_ROTATION=1 to run it."
    )
    print(f"Tiny selected_windows.csv: {tiny_selected_windows_csv}")
    print("Contour command:")
    print(" ".join(str(token) for token in targeted_contour_cmd))
    print("Population commands:")
    for cmd in targeted_population_cmds:
        print(" ".join(str(token) for token in cmd))

# %%
if targeted_rotation_ready(TARGETED_ROTATION_RUN_ROOT):
    tiny_rotated_frames_by_view: dict[str, pd.DataFrame] = {}
    tiny_population_dirs = targeted_rotation_population_dirs(TARGETED_ROTATION_RUN_ROOT)
    for spec in VIEW_SPECS:
        tiny_spec = dict(spec)
        tiny_spec["dir"] = tiny_population_dirs[str(spec["view"])]
        tiny_rotated_frames_by_view[str(spec["view"])] = load_view_frame(tiny_spec)

    tiny_rotated_subset_parts = []
    for spec in VIEW_SPECS:
        rows = image_orientation_rows(
            tiny_rotated_frames_by_view[str(spec["view"])],
            sf_group="high_sf",
            source_rows=example_source_rows,
        )
        if rows.empty:
            continue
        rows = rows.merge(example_source_to_band, on="source_row", how="inner")
        tiny_rotated_subset_parts.append(rows)
    tiny_rotated_subset_rows = (
        pd.concat(tiny_rotated_subset_parts, ignore_index=True) if tiny_rotated_subset_parts else pd.DataFrame()
    )
    tiny_rotated_curves = summarize_image_orientation_curves(
        tiny_rotated_subset_rows,
        extra_group_cols=["example_band", "example_band_label"],
    )
    original_subset_for_tiny_compare = subset_curves.copy()
    original_subset_for_tiny_compare["run_label"] = "original"
    tiny_rotated_for_compare = tiny_rotated_curves.copy()
    tiny_rotated_for_compare["run_label"] = f"rotated_{TARGETED_ROTATION_DEG}deg_tiny"
    tiny_targeted_compare = pd.concat([original_subset_for_tiny_compare, tiny_rotated_for_compare], ignore_index=True)

    show_table(
        tiny_targeted_compare[
            np.isclose(tiny_targeted_compare["x_scale"].to_numpy(dtype=float), EXAMPLE_X_SCALE)
            & (tiny_targeted_compare["view"].astype(str) == TARGET_VIEW)
        ][
            [
                col
                for col in [
                    "run_label",
                    "example_band_label",
                    "view",
                    "condition_id",
                    "x_scale",
                    "n_fixations_image_vertical",
                    "n_fixations_image_horizontal",
                    "accumulated_bits_per_spike_image_vertical",
                    "accumulated_bits_per_spike_image_horizontal",
                    "image_vertical_minus_horizontal",
                    "image_vertical_over_horizontal",
                ]
                if col in tiny_targeted_compare.columns
            ]
        ].sort_values(["example_band_label", "run_label"])
    )

    if not tiny_targeted_compare.empty:
        fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.7), sharey=True, constrained_layout=True)
        run_styles = {"original": "-", f"rotated_{TARGETED_ROTATION_DEG}deg_tiny": "--"}
        for ax, spec in zip(axes, VIEW_SPECS, strict=True):
            view = str(spec["view"])
            for band in DOMINANT_BANDS:
                for run_label in ["original", f"rotated_{TARGETED_ROTATION_DEG}deg_tiny"]:
                    sub = tiny_targeted_compare[
                        (tiny_targeted_compare["view"].astype(str) == view)
                        & (tiny_targeted_compare["example_band"].astype(str) == band)
                        & (tiny_targeted_compare["run_label"].astype(str) == run_label)
                    ].sort_values(["x_scale", "condition_index"])
                    if sub.empty:
                        continue
                    ax.plot(
                        sub["x_scale"],
                        sub["image_vertical_minus_horizontal"],
                        color=band_styles[band]["color"],
                        linestyle=run_styles[run_label],
                        marker=band_styles[band]["marker"],
                        linewidth=2.0,
                        alpha=0.9,
                        label=(
                            f"{run_label} {DOMINANT_BAND_LABEL[band]}"
                            if view == VIEW_SPECS[0]["view"]
                            else None
                        ),
                    )
            ax.axhline(0.0, color="0.35", linewidth=1.0)
            ax.axvline(1.0, color="0.62", linestyle=":", linewidth=0.9)
            ax.set_title(str(spec["label"]), fontsize=9.5)
            ax.set_xlabel("scale")
        axes[0].set_ylabel("tiny subset high SF\nimage vertical - image horizontal")
        axes[0].legend(frameon=False, fontsize=7.0)
        fig.suptitle(f"Targeted tiny source rows under original vs {TARGETED_ROTATION_DEG}-degree movie rotation")
        plt.show()
else:
    print(f"Tiny targeted rotated outputs are not ready yet: {TARGETED_ROTATION_RUN_ROOT}")

# %% [markdown]
# ## 21. Optional full rotated-run comparison
#
# Once the 90-degree movie-rotation run finishes, set `ROTATED_RUN_ROOT` below
# or launch with:
#
# ```bash
# BACKIMAGE_RR100_ROTATED_RUN_ROOT=/path/to/rotated/run jupyter lab
# ```
#
# This cell compares saved orientation summaries only, so it is quick.

# %%
ROTATED_RUN_ROOT_TEXT = os.environ.get("BACKIMAGE_RR100_ROTATED_RUN_ROOT", "").strip()
DEFAULT_ROTATED_RUN_ROOT = RUN_ROOT.with_name(RUN_ROOT.name.replace("_v1", "_rot90_v1"))
ROTATED_RUN_ROOT = (
    Path(ROTATED_RUN_ROOT_TEXT).expanduser().resolve()
    if ROTATED_RUN_ROOT_TEXT
    else (DEFAULT_ROTATED_RUN_ROOT if DEFAULT_ROTATED_RUN_ROOT.exists() else None)
)


def orientation_summary_path_for_run(run_root: Path) -> Path:
    commands_path = run_root / "long_run_commands.json"
    if commands_path.exists():
        commands = read_json_required(commands_path)
        orientation_dir = Path(commands.get("orientation_stratified_dir", run_root / "orientation_stratified_population"))
    else:
        orientation_dir = run_root / "orientation_stratified_population"
    return orientation_dir / "orientation_stratified_weighted_population_summary.csv"


def load_orientation_summary_for_run(run_root: Path) -> pd.DataFrame:
    summary_path = orientation_summary_path_for_run(run_root)
    out = read_csv_required(summary_path).copy()
    out["run_root"] = str(run_root)
    return out


if ROTATED_RUN_ROOT is None:
    print("Set BACKIMAGE_RR100_ROTATED_RUN_ROOT or create the default *_rot90_v1 run to compare a rotated control.")
elif not ROTATED_RUN_ROOT.exists():
    print(f"Rotated run root does not exist yet: {ROTATED_RUN_ROOT}")
elif not orientation_summary_path_for_run(ROTATED_RUN_ROOT).exists():
    print(
        "Rotated run root exists, but posthoc population/orientation summaries are not ready yet:\n"
        f"  root: {ROTATED_RUN_ROOT}\n"
        f"  missing: {orientation_summary_path_for_run(ROTATED_RUN_ROOT)}"
    )
else:
    original = orientation_summary.copy()
    original["run_label"] = "original"
    rotated = load_orientation_summary_for_run(ROTATED_RUN_ROOT)
    rotated["run_label"] = "rotated"
    both = pd.concat([original, rotated], ignore_index=True)
    sub = both[
        (both["band"].astype(str) == TARGET_BAND)
        & (both["view"].astype(str) == TARGET_VIEW)
        & (both["sf_group"].astype(str) == TARGET_SF)
    ].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (run_label, alignment_group), g in sub.groupby(["run_label", "alignment_group"], sort=True):
        linestyle = "-" if run_label == "original" else "--"
        ax.plot(
            g.sort_values("x_scale")["x_scale"],
            g.sort_values("x_scale")["accumulated_bits_per_spike"],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            color=COLORS.get(str(alignment_group), "0.4"),
            label=f"{run_label} {alignment_label(alignment_group)}",
        )
    ax.axvline(1.0, color="0.62", linestyle=":", linewidth=0.9)
    ax.set_xlabel("scale")
    ax.set_ylabel("bits/spike")
    ax.set_title(f"Original vs rotated: {TARGET_VIEW}, {TARGET_SF}, {TARGET_BAND}")
    ax.legend(frameon=False)
    plt.show()

    rotated_frames_by_view: dict[str, pd.DataFrame] = {}
    rotated_commands_path = ROTATED_RUN_ROOT / "long_run_commands.json"
    rotated_commands = read_json_required(rotated_commands_path) if rotated_commands_path.exists() else {}
    rotated_population_dirs = {
        "across_along0": ROTATED_RUN_ROOT / "population_across_sweep_along0",
        "across_along1": ROTATED_RUN_ROOT / "population_across_sweep_along1",
        "along_across0": ROTATED_RUN_ROOT / "population_along_sweep_across0",
        "along_across1": ROTATED_RUN_ROOT / "population_along_sweep_across1",
    }
    if "population_dirs" in rotated_commands:
        rotated_population_dirs.update({key: Path(value) for key, value in rotated_commands["population_dirs"].items()})
    for spec in VIEW_SPECS:
        rotated_spec = dict(spec)
        rotated_spec["dir"] = rotated_population_dirs[str(spec["view"])]
        rotated_frames_by_view[str(spec["view"])] = load_view_frame(rotated_spec)

    rotated_subset_parts = []
    for spec in VIEW_SPECS:
        rows = image_orientation_rows(
            rotated_frames_by_view[str(spec["view"])],
            sf_group="high_sf",
            source_rows=example_source_rows,
        )
        if rows.empty:
            continue
        rows = rows.merge(example_source_to_band, on="source_row", how="inner")
        rotated_subset_parts.append(rows)
    rotated_subset_rows = pd.concat(rotated_subset_parts, ignore_index=True) if rotated_subset_parts else pd.DataFrame()
    rotated_subset_curves = summarize_image_orientation_curves(
        rotated_subset_rows,
        extra_group_cols=["example_band", "example_band_label"],
    )
    original_subset_for_compare = subset_curves.copy()
    original_subset_for_compare["run_label"] = "original"
    rotated_subset_for_compare = rotated_subset_curves.copy()
    rotated_subset_for_compare["run_label"] = "rotated_movie"
    tiny_rot_compare = pd.concat([original_subset_for_compare, rotated_subset_for_compare], ignore_index=True)
    show_table(
        tiny_rot_compare[
            np.isclose(tiny_rot_compare["x_scale"].to_numpy(dtype=float), EXAMPLE_X_SCALE)
            & (tiny_rot_compare["view"].astype(str) == TARGET_VIEW)
        ][
            [
                col
                for col in [
                    "run_label",
                    "example_band_label",
                    "view",
                    "condition_id",
                    "x_scale",
                    "n_fixations_image_vertical",
                    "n_fixations_image_horizontal",
                    "accumulated_bits_per_spike_image_vertical",
                    "accumulated_bits_per_spike_image_horizontal",
                    "image_vertical_minus_horizontal",
                    "image_vertical_over_horizontal",
                ]
                if col in tiny_rot_compare.columns
            ]
        ].sort_values(["example_band_label", "run_label"])
    )

    if not tiny_rot_compare.empty:
        fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.7), sharey=True, constrained_layout=True)
        run_styles = {"original": "-", "rotated_movie": "--"}
        for ax, spec in zip(axes, VIEW_SPECS, strict=True):
            view = str(spec["view"])
            for band in DOMINANT_BANDS:
                for run_label in ["original", "rotated_movie"]:
                    sub = tiny_rot_compare[
                        (tiny_rot_compare["view"].astype(str) == view)
                        & (tiny_rot_compare["example_band"].astype(str) == band)
                        & (tiny_rot_compare["run_label"].astype(str) == run_label)
                    ].sort_values(["x_scale", "condition_index"])
                    if sub.empty:
                        continue
                    ax.plot(
                        sub["x_scale"],
                        sub["image_vertical_minus_horizontal"],
                        color=band_styles[band]["color"],
                        linestyle=run_styles[run_label],
                        marker=band_styles[band]["marker"],
                        linewidth=2.0,
                        alpha=0.9,
                        label=(
                            f"{run_label} {DOMINANT_BAND_LABEL[band]}"
                            if view == VIEW_SPECS[0]["view"]
                            else None
                        ),
                    )
            ax.axhline(0.0, color="0.35", linewidth=1.0)
            ax.axvline(1.0, color="0.62", linestyle=":", linewidth=0.9)
            ax.set_title(str(spec["label"]), fontsize=9.5)
            ax.set_xlabel("scale")
        axes[0].set_ylabel("tiny subset high SF\nimage vertical - image horizontal")
        axes[0].legend(frameon=False, fontsize=7.0)
        fig.suptitle("Same exemplar source rows under original vs 90-degree rotated movie run")
        plt.show()

# %% [markdown]
# ## 22. Notes while interpreting
#
# Suggested reading order when something looks wrong:
#
# 1. Cell 6: if the saved and rebuilt summary differ, the final plotter is at
#    fault.
# 2. Cell 12/13: if NPZ row recomputation differs, the weighted per-fixation CSV
#    is stale or the `unit_weights` text precision is too low for exact replay.
# 3. Cell 10: if the axis conversion check fails, the orientation bands are in
#    the wrong coordinate frame.
# 4. Cell 8/9: if numerator and denominator tell different stories, do not
#    interpret the ratio alone.
# 5. Cell 11/14/15: if the effect tracks unit preference, weight support, or
#    image-statistic bands, the result may be genuine but conditional.
