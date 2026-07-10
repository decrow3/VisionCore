# %% [markdown]
# # Vernier endpoint-history, last-frame readout tutorial
#
# This is a notebook-style Python script. Open it in VS Code, Jupyter, or any
# editor that recognizes `# %%` cells to step through the method one piece at a
# time.
#
# This version mirrors the newest endpoint-history feature model contract:
#
# 1. Build motion histories whose final retinal endpoint is the same.
# 2. Keep the history before that endpoint different across conditions.
# 3. Run the twin on the full history.
# 4. Decode the Vernier finite-difference signal from only the terminal response
#    frame/window.
#
# In code, the trajectory contract is:
#
# ```text
# endpoint_trace[t] = trace[t] - trace[-1]
# ```
#
# The heaviest digital-twin cells are controlled by run flags. The tutorial
# still documents the exact runner commands and reads cached terminal-frame
# outputs when available.

# %%
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        if (candidate / "pyproject.toml").exists() and (candidate / "declan").exists():
            return candidate
    raise RuntimeError(f"Could not find repo root from {start}")


try:
    HERE = Path(__file__).resolve()
except NameError:
    HERE = Path.cwd().resolve()

ROOT = find_repo_root(HERE)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 160)

ROOT

# %%
from declan.vernier_active_sensing.run_endpoint_history_last_frame_readout import (
    DEFAULT_CONDITIONS,
    DEFAULT_OUT_DIR,
    ENDPOINT_CONDITIONS,
    build_endpoint_trace,
    endpoint_condition_rng,
)
from declan.vernier_active_sensing.trajectories import load_eye_traces, subsample_traces, valid_trace

try:
    from IPython.display import Image, Markdown, display
except Exception:  # pragma: no cover - only for plain script execution
    Image = None
    Markdown = None

    def display(value: Any) -> None:
        print(value)


def show_markdown(text: str) -> None:
    try:
        shell = get_ipython()
    except NameError:
        shell = None
    if Markdown is None or shell is None:
        print(text)
    else:
        display(Markdown(text))


def show_table(df: pd.DataFrame, n: int | None = None) -> None:
    if n is not None:
        df = df.head(n)
    display(df)


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    numeric_cols = {
        "fd_step_arcmin",
        "n",
        "final_fisher",
        "final_dprime2",
        "final_threshold_proxy",
        "mean_final_fisher",
        "median_final_fisher",
        "mean_final_threshold_proxy",
        "median_final_threshold_proxy",
        "endpoint_x_deg",
        "endpoint_y_deg",
        "endpoint_norm_deg",
        "history_rms_deg",
        "history_path_length_deg",
        "history_max_radius_deg",
        "terminal_frames",
        "history_frames",
        "n_timebins",
        "n_units",
        "mean_fisher_delta",
        "median_fisher_delta",
        "mean_threshold_ratio",
        "median_threshold_ratio",
        "p_condition_beats_baseline",
        "unit_index",
        "n_units",
        "mean_unit_log2_ssi_vs_static",
        "sem_unit_log2_ssi_vs_static",
        "population_log2_ssi_vs_static",
        "static_map_median",
        "static_map_high_percentile",
        "static_map_low_percentile",
        "positive_strength",
        "negative_strength",
        "polarity_score_positive_minus_negative",
        "static_ssi_bits_per_spike_mean",
        "zero_x_ssi_bits_per_spike_mean",
        "unit_ssi_bits_per_spike_mean",
        "delta_ssi_vs_static",
        "delta_ssi_vs_0x",
        "log2_ratio_vs_static",
        "log2_ratio_vs_0x",
        "log2_ratio_vs_static_floor",
        "log2_ratio_vs_0x_floor",
        "denominator_floor_bits",
        "budget_proxy_mean",
        "delta_budget_proxy_vs_static",
        "delta_budget_proxy_vs_0x",
        "sum_unit_ssi_bits_per_spike",
        "sum_delta_ssi_vs_static",
        "sum_delta_ssi_vs_0x",
        "sum_unit_budget_proxy",
        "sum_delta_budget_proxy_vs_static",
        "sum_delta_budget_proxy_vs_0x",
        "population_ssi_bits_per_spike_mean",
        "delta_population_ssi_vs_static",
        "delta_population_ssi_vs_0x",
        "total_rate_mean",
        "static_ssi_min_bits",
        "geometric_mean_ratio_vs_static",
        "geometric_mean_ratio_vs_static_floor",
        "arithmetic_mean_ratio_vs_static",
        "sum_static_ssi_bits_per_spike",
        "sum_current_ssi_bits_per_spike",
        "min_static_ssi_bits",
        "geometric_mean_ssi_vs_static",
    }
    for col in set(df.columns).intersection(numeric_cols):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def show_image_if_exists(path: Path) -> None:
    if not path.exists():
        print(f"Missing image: {path}")
        return
    if Image is None:
        print(path)
    else:
        display(Image(filename=str(path)))

# %% [markdown]
# ## Configuration
#
# `RUN_ENDPOINT_MODEL` is intentionally false by default. Flip it when you want
# to launch the twin from the notebook. The command below writes the canonical
# tutorial outputs under `outputs/vernier_endpoint_history_last_frame_tutorial`.

# %%
RUN_ENDPOINT_MODEL = False
RUN_LAST_FRAME_SSI = False
RUN_RR100_ENDPOINT_SCALE_GRID = False
RUN_ALONG0_UNIT_SSI_DIAGNOSTIC = True
FORCE_RERUN = False

RUN_DIR = ROOT / DEFAULT_OUT_DIR
SSI_RUN_DIR = RUN_DIR / "ssi_last_frame_maps"
RR100_SCALE_GRID_DIR = RUN_DIR / "rr100_endpoint_history_scale_grid"
UNIT_SSI_ALONG0_DIR = RR100_SCALE_GRID_DIR / "unit_ssi_along0_diagnostics"
EYE_TRACES_PATH = ROOT / "scripts" / "temporal_decoding" / "data" / "eye_traces.npz"

CONDITIONS = list(DEFAULT_CONDITIONS)
HISTORY_FRAMES = 32
TERMINAL_FRAMES = 1
N_TRACES = 4
TRACE_INDEX_TO_PLOT = 0
FD_STEPS_ARCMIN = [0.25]
SEED = 0
BATCH_SIZE = 16
POPULATION = "rr100_medoid"
UNIT_SSI_N_TRACES = 16
UNIT_SSI_BATCH_SIZE = 64
UNIT_SSI_TOP_UNITS = 12
DENOMINATOR_FLOOR_BITS = 0.01

RUN_DIR.mkdir(parents=True, exist_ok=True)

endpoint_runner_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.run_endpoint_history_last_frame_readout",
    "--out-dir",
    str(RUN_DIR),
    "--eye-traces-path",
    str(EYE_TRACES_PATH),
    "--conditions",
    ",".join(CONDITIONS),
    "--fd-steps-arcmin",
    ",".join(str(v) for v in FD_STEPS_ARCMIN),
    "--history-frames",
    str(HISTORY_FRAMES),
    "--terminal-frames",
    str(TERMINAL_FRAMES),
    "--n-traces",
    str(N_TRACES),
    "--batch-size",
    str(BATCH_SIZE),
    "--population",
    POPULATION,
]

last_frame_ssi_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.run_endpoint_history_last_frame_ssi",
    "--out-dir",
    str(SSI_RUN_DIR),
    "--eye-traces-path",
    str(EYE_TRACES_PATH),
    "--conditions",
    ",".join(CONDITIONS),
    "--populations",
    "full756,rr100_medoid",
    "--trace-index",
    str(TRACE_INDEX_TO_PLOT),
    "--history-frames",
    str(HISTORY_FRAMES),
    "--fd-step-arcmin",
    str(FD_STEPS_ARCMIN[0]),
    "--batch-size",
    str(BATCH_SIZE),
    "--map-kind-for-ssi",
    "zero",
    "--collapse",
    "mean",
]

rr100_endpoint_scale_grid_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.run_rr100_endpoint_history_scale_grid",
    "--out-dir",
    str(RR100_SCALE_GRID_DIR),
    "--eye-traces-path",
    str(EYE_TRACES_PATH),
    "--n-traces",
    str(UNIT_SSI_N_TRACES),
    "--history-frames",
    str(HISTORY_FRAMES),
    "--fd-step-arcmin",
    str(FD_STEPS_ARCMIN[0]),
    "--seed",
    str(SEED),
    "--batch-size",
    str(UNIT_SSI_BATCH_SIZE),
]

along0_unit_ssi_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.plot_rr100_endpoint_along0_unit_ssi",
    "--out-dir",
    str(UNIT_SSI_ALONG0_DIR),
    "--summary-csv",
    str(RR100_SCALE_GRID_DIR / "rr100_endpoint_history_scale_grid_summary.csv"),
    "--eye-traces-path",
    str(EYE_TRACES_PATH),
    "--n-traces",
    str(UNIT_SSI_N_TRACES),
    "--history-frames",
    str(HISTORY_FRAMES),
    "--fd-step-arcmin",
    str(FD_STEPS_ARCMIN[0]),
    "--seed",
    str(SEED),
    "--batch-size",
    str(UNIT_SSI_BATCH_SIZE),
    "--top-units",
    str(UNIT_SSI_TOP_UNITS),
]

along0_polarity_group_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.plot_rr100_along0_polarity_group_averages",
    "--mode",
    "endpoint",
    "--endpoint-dir",
    str(UNIT_SSI_ALONG0_DIR),
    "--fd-step-arcmin",
    str(FD_STEPS_ARCMIN[0]),
]

along0_filtered_polarity_group_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.plot_rr100_along0_polarity_group_averages",
    "--mode",
    "endpoint",
    "--endpoint-dir",
    str(UNIT_SSI_ALONG0_DIR),
    "--fd-step-arcmin",
    str(FD_STEPS_ARCMIN[0]),
    "--min-static-ssi-bits",
    str(DENOMINATOR_FLOOR_BITS),
    "--include-all-group",
]

along0_denominator_diagnostic_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.plot_rr100_along0_denominator_diagnostics",
    "--mode",
    "endpoint",
    "--endpoint-dir",
    str(UNIT_SSI_ALONG0_DIR),
    "--fd-step-arcmin",
    str(FD_STEPS_ARCMIN[0]),
    "--denominator-floor-bits",
    str(DENOMINATOR_FLOOR_BITS),
]

print(f"ROOT: {ROOT}")
print(f"RUN_DIR: {RUN_DIR}")
print("Endpoint-history runner command:")
print(" ".join(endpoint_runner_command))
print("Last-frame SSI command:")
print(" ".join(last_frame_ssi_command))
print("RR100 endpoint scale-grid command:")
print(" ".join(rr100_endpoint_scale_grid_command))
print("Along=0 unit SSI diagnostic command:")
print(" ".join(along0_unit_ssi_command))
print("Along=0 polarity-group command:")
print(" ".join(along0_polarity_group_command))
print("Along=0 filtered polarity-group command:")
print(" ".join(along0_filtered_polarity_group_command))
print("Along=0 denominator diagnostic command:")
print(" ".join(along0_denominator_diagnostic_command))

# %% [markdown]
# ## What Changed Relative To The Earlier Vernier Tutorial?
#
# The earlier Vernier tutorial compares motion conditions with their native
# retinal positions. That is useful, but endpoint position and prior trajectory
# history can be entangled.
#
# Here every condition is shifted so its final sample is exactly the shared
# endpoint. The static condition is all endpoint; the motion conditions reach the
# same endpoint after different histories. The readout then sees only the final
# response frame/window, so any difference across conditions has to come from how
# the temporal model's state was prepared by the preceding history.

# %%
condition_contract = pd.DataFrame(
    [
        {
            "condition": condition,
            "label": spec["label"],
            "source_condition": spec["source_condition"],
            "interpretation": spec["interpretation"],
        }
        for condition, spec in ENDPOINT_CONDITIONS.items()
        if condition in CONDITIONS
    ]
)
show_table(condition_contract)

# %% [markdown]
# ## Endpoint-Aligned Histories
#
# This cell constructs the exact histories that the runner sends into the twin.
# The endpoint check should be numerically zero for every condition, while the
# path-length and RMS columns show that the histories before that endpoint still
# differ.

# %%
trace_set = subsample_traces(load_eye_traces(EYE_TRACES_PATH), N_TRACES, SEED)
base_trace = valid_trace(trace_set, TRACE_INDEX_TO_PLOT, max_frames=HISTORY_FRAMES)
trace_args = SimpleNamespace(history_frames=HISTORY_FRAMES, frame_rate_hz=120.0)

endpoint_trace_rows: list[dict[str, Any]] = []
endpoint_traces: dict[str, np.ndarray] = {}
for condition in CONDITIONS:
    rng = endpoint_condition_rng(SEED, condition, TRACE_INDEX_TO_PLOT)
    endpoint_trace, meta = build_endpoint_trace(
        base_trace,
        condition=condition,
        trace_set=trace_set,
        rng=rng,
        args=trace_args,
    )
    endpoint_traces[condition] = endpoint_trace
    endpoint_trace_rows.append(
        {
            "condition": condition,
            "label": ENDPOINT_CONDITIONS[condition]["label"],
            "source_condition": ENDPOINT_CONDITIONS[condition]["source_condition"],
            "history_frames": endpoint_trace.shape[0],
            **meta,
        }
    )

endpoint_trace_df = pd.DataFrame(endpoint_trace_rows)
endpoint_trace_preview_path = RUN_DIR / "endpoint_history_trace_preview.csv"
endpoint_trace_df.to_csv(endpoint_trace_preview_path, index=False)

max_endpoint_norm = float(endpoint_trace_df["endpoint_norm_deg"].abs().max())
if max_endpoint_norm > 1e-7:
    raise AssertionError(f"Endpoint alignment failed: max endpoint norm {max_endpoint_norm:g} deg")

show_table(
    endpoint_trace_df[
        [
            "condition",
            "source_condition",
            "endpoint_x_deg",
            "endpoint_y_deg",
            "endpoint_norm_deg",
            "history_rms_deg",
            "history_path_length_deg",
            "history_max_radius_deg",
        ]
    ]
)
print(f"Saved trace preview table: {endpoint_trace_preview_path}")

# %%
def plot_endpoint_histories(endpoint_traces: dict[str, np.ndarray], trace_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150, constrained_layout=True)
    ax_xy, ax_bar = axes
    for condition, trace in endpoint_traces.items():
        label = ENDPOINT_CONDITIONS[condition]["label"]
        arcmin = np.asarray(trace, dtype=np.float64) * 60.0
        ax_xy.plot(arcmin[:, 0], arcmin[:, 1], marker="o", markersize=2.2, linewidth=1.25, label=label)
        ax_xy.scatter([0.0], [0.0], s=18, color="black", zorder=5)
    ax_xy.axhline(0.0, color="#999999", linewidth=0.6)
    ax_xy.axvline(0.0, color="#999999", linewidth=0.6)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("horizontal position relative to endpoint (arcmin)")
    ax_xy.set_ylabel("vertical position relative to endpoint (arcmin)")
    ax_xy.set_title("Different histories, same final endpoint")
    ax_xy.legend(frameon=False, fontsize=7)

    labels = [ENDPOINT_CONDITIONS[c]["label"] for c in CONDITIONS if c in set(trace_df["condition"])]
    rows_by_condition = {str(row["condition"]): row for _, row in trace_df.iterrows()}
    rms = [float(rows_by_condition[c]["history_rms_deg"]) * 60.0 for c in CONDITIONS if c in rows_by_condition]
    path_len = [float(rows_by_condition[c]["history_path_length_deg"]) * 60.0 for c in CONDITIONS if c in rows_by_condition]
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    ax_bar.bar(x - width / 2.0, rms, width=width, label="RMS radius")
    ax_bar.bar(x + width / 2.0, path_len, width=width, label="path length")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=25, ha="right")
    ax_bar.set_ylabel("arcmin")
    ax_bar.set_title("History differs before endpoint")
    ax_bar.legend(frameon=False)
    ax_bar.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


trace_preview_png = RUN_DIR / "endpoint_history_trace_preview.png"
plot_endpoint_histories(endpoint_traces, endpoint_trace_df, trace_preview_png)
show_image_if_exists(trace_preview_png)

# %% [markdown]
# ## Optional Twin Run
#
# The runner uses the histories above, computes `+offset` and `-offset` responses,
# keeps only `rates[-terminal_frames:]`, and then scores the terminal-window
# Poisson Fisher information. With `TERMINAL_FRAMES = 1`, every cached rate array
# should have a length of one frame.

# %%
summary_path = RUN_DIR / "endpoint_history_last_frame_summary.csv"
if RUN_ENDPOINT_MODEL and (FORCE_RERUN or not summary_path.exists()):
    subprocess.run(endpoint_runner_command, cwd=ROOT, check=True)
elif RUN_ENDPOINT_MODEL:
    print(f"Using existing cached endpoint-history run: {RUN_DIR}")
else:
    print("Model run disabled. To run it, set RUN_ENDPOINT_MODEL = True and execute this cell.")
    print(" ".join(endpoint_runner_command))

# %% [markdown]
# ## Cached Terminal-Frame Results
#
# The canonical run directory is preferred. If it does not yet contain model
# outputs, the tutorial falls back to a local smoke run when present, so the
# tables below still show the expected file contracts.

# %%
SMOKE_RUN_DIR = ROOT / "outputs" / "vernier_endpoint_history_last_frame_model_smoke"


def choose_cached_run(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if (candidate / "endpoint_history_last_frame_summary.csv").exists():
            return candidate
    return candidates[0]


CACHED_RUN_DIR = choose_cached_run([RUN_DIR, SMOKE_RUN_DIR])
print(f"Reading cached run directory: {CACHED_RUN_DIR}")

manifest = read_json_optional(CACHED_RUN_DIR / "vernier_endpoint_history_last_frame_manifest.json")
trials_df = read_csv_optional(CACHED_RUN_DIR / "endpoint_history_last_frame_trials.csv")
summary_df = read_csv_optional(CACHED_RUN_DIR / "endpoint_history_last_frame_summary.csv")
trace_metrics_df = read_csv_optional(CACHED_RUN_DIR / "endpoint_history_trace_metrics.csv")
contrasts_df = read_csv_optional(CACHED_RUN_DIR / "endpoint_history_last_frame_contrast_summary.csv")

if manifest:
    show_table(pd.DataFrame([manifest.get("assay", {})]))
else:
    print("No model manifest found yet.")

if not summary_df.empty:
    show_table(summary_df)
else:
    print("No endpoint-history summary CSV found yet.")

# %%
if not trace_metrics_df.empty:
    endpoint_cols = [
        "condition",
        "trace_index",
        "endpoint_norm_deg",
        "history_rms_deg",
        "history_path_length_deg",
        "readout_time_contract",
        "endpoint_alignment",
    ]
    show_table(trace_metrics_df[[c for c in endpoint_cols if c in trace_metrics_df.columns]])
    print("Max cached endpoint norm:", trace_metrics_df["endpoint_norm_deg"].abs().max())
else:
    print("No cached trace metrics found yet.")

if not contrasts_df.empty:
    show_table(contrasts_df)

# %%
def inspect_rate_caches(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "cache").glob("endpoint_terminal_rates_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            lengths = np.asarray(data["lengths"], dtype=np.int64)
            rows.append(
                {
                    "cache": path.name,
                    "condition": str(data["condition"][0]) if "condition" in data else "",
                    "terminal_frames": int(np.asarray(data["terminal_frames"]).ravel()[0])
                    if "terminal_frames" in data
                    else np.nan,
                    "unique_lengths": ",".join(str(int(v)) for v in sorted(set(lengths.tolist()))),
                    "plus_shape": tuple(np.asarray(data["plus"]).shape),
                    "minus_shape": tuple(np.asarray(data["minus"]).shape),
                    "readout_time_contract": str(data["readout_time_contract"][0])
                    if "readout_time_contract" in data
                    else "",
                    "endpoint_alignment": str(data["endpoint_alignment"][0]) if "endpoint_alignment" in data else "",
                }
            )
    return pd.DataFrame(rows)


cache_audit_df = inspect_rate_caches(CACHED_RUN_DIR)
if not cache_audit_df.empty:
    show_table(cache_audit_df)
    bad_lengths = cache_audit_df[cache_audit_df["unique_lengths"] != str(TERMINAL_FRAMES)]
    if not bad_lengths.empty:
        raise AssertionError("At least one cached rate file is not terminal-frame only.")
else:
    print("No terminal-rate caches found yet.")

# %%
show_image_if_exists(CACHED_RUN_DIR / "endpoint_history_last_frame_fisher.png")

# %% [markdown]
# ## Last-Frame SSI From The Terminal Map
#
# SSI is computed from the same endpoint-aligned histories, but it uses the
# terminal spatial activation map before spatial collapse. The default SSI map is
# the zero-offset Vernier stimulus, matching the earlier final-history SSI audit.
#
# SSI itself normalizes each unit's spatial map by that unit's own mean response,
# so it measures spatial concentration rather than raw gain. Static-normalized
# ratios such as `SSI / static` are a separate plotting normalization. Those
# ratios are useful diagnostics, but they are sensitive to tiny static SSI
# denominators and should be checked against absolute SSI and spike-weighted
# budget summaries.

# %%
ssi_summary_path = SSI_RUN_DIR / "vernier_endpoint_history_last_frame_ssi_summary.csv"
if RUN_LAST_FRAME_SSI and (FORCE_RERUN or not ssi_summary_path.exists()):
    subprocess.run(last_frame_ssi_command, cwd=ROOT, check=True)
elif RUN_LAST_FRAME_SSI:
    print(f"Using existing cached last-frame SSI run: {SSI_RUN_DIR}")
else:
    print("Last-frame SSI run disabled. To run it, set RUN_LAST_FRAME_SSI = True and execute this cell.")
    print(" ".join(last_frame_ssi_command))

# %%
ssi_summary_df = read_csv_optional(ssi_summary_path)
if not ssi_summary_df.empty:
    show_table(
        ssi_summary_df[
            [
                "condition",
                "population_key",
                "last_frame_ssi_bits_per_spike",
                "last_frame_ssi_bits_per_spike_vs_full",
                "last_frame_total_rate",
                "endpoint_norm_deg",
            ]
        ]
    )
else:
    print(f"No last-frame SSI summary found yet: {ssi_summary_path}")

# %%
show_image_if_exists(SSI_RUN_DIR / "vernier_endpoint_history_last_frame_ssi_bars.png")
show_image_if_exists(SSI_RUN_DIR / "vernier_endpoint_last_frame_activation_gallery_rr100_medoid_zero_mean.png")

# %% [markdown]
# ## RR100 Endpoint Scale Grid And Along-0 Unit Diagnostics
#
# The endpoint scale grid sweeps the across-contour motion scale while holding
# the final endpoint fixed. The unit diagnostic then takes the `along = 0` SSI
# line, highlights the most extreme individual units and the largest
# leave-one-unit-out influences, and writes activation maps for every highlighted
# unit at every point on the plotted line. Each activation-map tile is the
# trace-mean terminal finite-difference midpoint map, `0.5 * (plus + minus)`,
# matching the map used for the SSI diagnostic.
#
# The unit line plots average `log2(SSI(scale) / SSI(static))`, which is a
# geometric mean of unit-wise ratios. This avoids arithmetic-ratio averaging, but
# it does not remove denominator inflation from units with almost no static
# spatial structure. The filtered polarity plot keeps only units above the
# predeclared static-SSI floor, and the denominator diagnostic compares fold
# changes with absolute bits/spike and rate-weighted budget proxies.

# %%
rr100_scale_summary_path = RR100_SCALE_GRID_DIR / "rr100_endpoint_history_scale_grid_summary.csv"
if RUN_RR100_ENDPOINT_SCALE_GRID and (FORCE_RERUN or not rr100_scale_summary_path.exists()):
    subprocess.run(rr100_endpoint_scale_grid_command, cwd=ROOT, check=True)
elif RUN_RR100_ENDPOINT_SCALE_GRID:
    print(f"Using existing cached RR100 endpoint scale grid: {RR100_SCALE_GRID_DIR}")
else:
    print("RR100 endpoint scale-grid run disabled. To run it, set RUN_RR100_ENDPOINT_SCALE_GRID = True.")
    print(" ".join(rr100_endpoint_scale_grid_command))

show_image_if_exists(RR100_SCALE_GRID_DIR / "rr100_endpoint_history_last_frame_scale_grid_rows_two_baselines.png")

# %%
unit_diag_manifest_path = UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_unit_ssi_manifest.json"
unit_map_manifest_path = (
    UNIT_SSI_ALONG0_DIR
    / "highlighted_unit_activation_maps"
    / "rr100_endpoint_along0_highlighted_unit_map_manifest.csv"
)
unit_diag_required = [
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_unit_ssi_lines_top_influence.png",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_unit_ssi_lines_top_influence_with_activation_rows.png",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_unit_ssi_leave_one_out.png",
    unit_map_manifest_path,
]
if RUN_ALONG0_UNIT_SSI_DIAGNOSTIC and (FORCE_RERUN or not all(path.exists() for path in unit_diag_required)):
    subprocess.run(along0_unit_ssi_command, cwd=ROOT, check=True)
elif RUN_ALONG0_UNIT_SSI_DIAGNOSTIC:
    print(f"Using existing cached along=0 unit SSI diagnostic: {UNIT_SSI_ALONG0_DIR}")
else:
    print("Along=0 unit SSI diagnostic disabled. To run it, set RUN_ALONG0_UNIT_SSI_DIAGNOSTIC = True.")
    print(" ".join(along0_unit_ssi_command))

polarity_group_required = [
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_polarity_group_averages.png",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_polarity_unit_table.csv",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_polarity_group_summary.csv",
]
if RUN_ALONG0_UNIT_SSI_DIAGNOSTIC and (FORCE_RERUN or not all(path.exists() for path in polarity_group_required)):
    subprocess.run(along0_polarity_group_command, cwd=ROOT, check=True)
elif RUN_ALONG0_UNIT_SSI_DIAGNOSTIC:
    print(f"Using existing cached along=0 polarity-group diagnostic: {UNIT_SSI_ALONG0_DIR}")

filtered_polarity_group_required = [
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_polarity_static_ssi_ge_0p01_group_averages.png",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_polarity_static_ssi_ge_0p01_unit_table.csv",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_polarity_static_ssi_ge_0p01_group_summary.csv",
]
if RUN_ALONG0_UNIT_SSI_DIAGNOSTIC and (FORCE_RERUN or not all(path.exists() for path in filtered_polarity_group_required)):
    subprocess.run(along0_filtered_polarity_group_command, cwd=ROOT, check=True)
elif RUN_ALONG0_UNIT_SSI_DIAGNOSTIC:
    print(f"Using existing cached along=0 filtered polarity-group diagnostic: {UNIT_SSI_ALONG0_DIR}")

denominator_diagnostic_required = [
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_denominator_diagnostics.png",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_denominator_static_floor_sweep.png",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_denominator_diagnostic_units.csv",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_denominator_diagnostic_groups.csv",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_denominator_diagnostic_summary.csv",
    UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_denominator_static_floor_sweep.csv",
]
if RUN_ALONG0_UNIT_SSI_DIAGNOSTIC and (FORCE_RERUN or not all(path.exists() for path in denominator_diagnostic_required)):
    subprocess.run(along0_denominator_diagnostic_command, cwd=ROOT, check=True)
elif RUN_ALONG0_UNIT_SSI_DIAGNOSTIC:
    print(f"Using existing cached along=0 denominator diagnostic: {UNIT_SSI_ALONG0_DIR}")

show_image_if_exists(UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_unit_ssi_lines_top_influence.png")
show_image_if_exists(UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_unit_ssi_lines_top_influence_with_activation_rows.png")
show_image_if_exists(UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_polarity_group_averages.png")
show_image_if_exists(UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_polarity_static_ssi_ge_0p01_group_averages.png")
show_image_if_exists(UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_denominator_diagnostics.png")
show_image_if_exists(UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_denominator_static_floor_sweep.png")
show_image_if_exists(UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_unit_ssi_leave_one_out.png")

# %%
unit_top_df = read_csv_optional(UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_unit_ssi_top_units.csv")
if not unit_top_df.empty:
    show_table(
        unit_top_df[
            [
                "unit_index",
                "max_abs_leave_one_out_population_ratio_delta",
                "max_abs_log2_unit_ssi_vs_static",
                "static_unit_ssi_bits_per_spike_mean",
                "static_unit_mean_rate_mean",
            ]
        ],
        n=UNIT_SSI_TOP_UNITS,
    )
else:
    print(f"No along=0 top-unit table found yet: {UNIT_SSI_ALONG0_DIR}")

polarity_group_summary_df = read_csv_optional(UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_polarity_group_summary.csv")
if not polarity_group_summary_df.empty:
    show_table(polarity_group_summary_df)

denominator_summary_df = read_csv_optional(UNIT_SSI_ALONG0_DIR / "rr100_endpoint_along0_denominator_diagnostic_summary.csv")
if not denominator_summary_df.empty:
    show_table(
        denominator_summary_df[
            [
                "across_scale",
                "population_ssi_bits_per_spike_mean",
                "delta_population_ssi_vs_static",
                "budget_proxy_mean",
                "delta_budget_proxy_vs_static",
                "mean_unit_log2_ratio_vs_static",
                "mean_unit_log2_ratio_vs_static_floor",
                "sum_unit_delta_ssi_vs_static",
            ]
        ]
    )

# %% [markdown]
# Reading the SSI diagnostic figures: the unfiltered unit-wise fold-change view
# is a denominator diagnostic. Large positive log ratios from units with tiny
# static SSI should be checked against absolute bits/spike, absolute changes from
# static or 0x, and the rate-weighted budget proxy. The filtered polarity figure
# keeps only units above the static-SSI floor and is better for retained-unit
# fold-change summaries; absolute SSI and budget quantities remain primary.

unit_map_manifest_df = read_csv_optional(unit_map_manifest_path)
if not unit_map_manifest_df.empty:
    unit_sheet_df = (
        unit_map_manifest_df[["unit_index", "unit_sheet_png"]]
        .drop_duplicates()
        .sort_values("unit_index")
        .reset_index(drop=True)
    )
    print(f"Highlighted unit map sheets: {len(unit_sheet_df)}")
    show_table(unit_sheet_df)
else:
    unit_sheet_df = pd.DataFrame()
    print(f"No highlighted-unit activation-map manifest found yet: {unit_map_manifest_path}")

# %%
# Display a few sheets inline; the manifest above points to all highlighted
# units and every individual unit/scale PNG.
if not unit_sheet_df.empty:
    preview_units: list[int] = []
    if not unit_top_df.empty:
        candidate_units = list(unit_top_df["unit_index"].head(2))
        candidate_units.extend(
            list(unit_top_df.sort_values("max_abs_log2_unit_ssi_vs_static", ascending=False)["unit_index"].head(2))
        )
        for unit in candidate_units:
            unit_int = int(unit)
            if unit_int not in preview_units:
                preview_units.append(unit_int)
    if not preview_units:
        preview_units = [int(unit) for unit in unit_sheet_df["unit_index"].head(4)]
    for unit in preview_units[:4]:
        matches = unit_sheet_df[unit_sheet_df["unit_index"].astype(int).eq(int(unit))]
        if not matches.empty:
            show_image_if_exists(Path(matches.iloc[0]["unit_sheet_png"]))

# %% [markdown]
# ## Takeaway
#
# This tutorial version separates the two contracts that matter for comparison
# with the endpoint-history feature model:
#
# - The final retinal endpoint is fixed by subtracting the terminal eye position
#   from every history.
# - The Vernier readout uses only the terminal response frame/window.
#
# That means static, real FEM, shuffled, phase-cloud, horizontal, and vertical
# histories are all tested at the same final endpoint while preserving different
# temporal histories before that endpoint.
