# %% [markdown]
# # Aggregate feature-information model walkthrough
#
# This is a notebook-style Python script. Open it in VS Code, Jupyter, or any
# editor that recognizes `# %%` cells to step through the aggregate feature
# information model one piece at a time.
#
# The purpose is not just to reproduce the panel. The purpose is to make the
# analysis legible:
#
# 1. What visual variable is decoded?
# 2. What model responses are compared?
# 3. What does the stabilized baseline mean?
# 4. Where does the trajectory enter the calculation?
# 5. What does the ridge decoder assume?
# 6. How is decoder error converted into information units?
# 7. What changes when the trajectory sample is hidden?
#
# The default cells are read-only. They inspect existing cached outputs, render
# intermediate diagnostic plots, and document the runner commands. Heavy model
# recomputation is gated behind explicit booleans.

# %%
from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

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

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

ROOT

# %%
try:
    from IPython.display import Image, Markdown, display
except Exception:  # pragma: no cover - plain Python fallback
    Image = None
    Markdown = None

    def display(value: Any) -> None:
        print(value)


def show_markdown(text: str) -> None:
    """Render Markdown/LaTeX in notebooks, with readable text in plain Python."""
    try:
        shell = get_ipython()
    except NameError:
        shell = None
    if Markdown is None or shell is None:
        print(text)
    else:
        display(Markdown(text))


def show_table(df: pd.DataFrame, n: int | None = None) -> None:
    """Notebook-friendly table display with a plain-Python fallback."""
    if n is not None:
        df = df.head(n)
    display(df)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json_required(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def show_image_if_exists(path: Path) -> None:
    if not path.exists():
        print(f"Missing image: {path}")
        return
    if Image is None:
        print(path)
        return
    try:
        shell = get_ipython()
    except NameError:
        shell = None
    display_module = str(getattr(display, "__module__", ""))
    if shell is None and display_module.startswith("IPython."):
        print(path)
    else:
        display(Image(filename=str(path)))


def scale_value(scale_id: str) -> float:
    text = str(scale_id).replace("rel_", "").replace("x", "").replace("p", ".")
    if text == "static":
        return 0.0
    return float(text)


def scale_label(scale: float) -> str:
    if abs(float(scale) - round(float(scale))) < 1e-9:
        return f"{int(round(float(scale)))}x"
    return f"{float(scale):g}x"


def command_text(command: list[str]) -> str:
    return shlex.join([str(part) for part in command])


def validate_point_inside_ci(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        value = row.get(value_col)
        low = row.get("info_diag_ci95_low")
        high = row.get("info_diag_ci95_high")
        if pd.isna(value) or pd.isna(low) or pd.isna(high):
            continue
        value = float(value)
        low = float(low)
        high = float(high)
        if math.isfinite(value) and math.isfinite(low) and math.isfinite(high) and not (low <= value <= high):
            rows.append(
                {
                    "row": int(idx),
                    "family": row.get("family", ""),
                    "observer": row.get("observer", ""),
                    "scale_id": row.get("scale_id", ""),
                    "value": value,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    }
)

# %% [markdown]
# ## Cached inputs used by the walkthrough
#
# The tutorial reads two production caches:
#
# - the strict source-trial grouped static-plus-motion information cache, whose
#   promoted row is the static-plus-delta (`delta_mean`) readout;
# - the same-axis pose-unaware hidden-sample proxy cache.
#
# These caches are currently wired into the selected Figure 4 feature-information
# panel, but the model and walkthrough are intentionally named independently of
# any final panel placement.

# %%
BASE_OUTPUT = ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"

RUN_DIR = (
    BASE_OUTPUT
    / "backimage_aggregate_fem_information_n384_pyramid_k16_tworeadout_rel025-2_power_seed0_k8_v1"
)
INFO_DIR = RUN_DIR / "incremental_staticmean_plus_motion_info_decode_bootstrap_b50_source_trial_validated_20260630"

POSE_RUN_DIR = BASE_OUTPUT / "backimage_aggregate_fem_information_pose_unaware_production_n384_empirical_k8_seed0"
POSE_INFO_DIR = POSE_RUN_DIR / "pose_unaware_staticmean_plus_motion_info_source_trial_b50_20260630"

ATLAS = ROOT / "declan/figure4_active_sensing_atlas"

WALKTHROUGH_OUT_DIR = ROOT / "outputs/notebook_aggregate_feature_information_model"
WALKTHROUGH_OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_SOURCE_TRIAL_RECOMPUTE = False
RUN_POSE_UNAWARE_RECOMPUTE = False
RUN_SELECTED_FIGURE_REBUILD = False
RUN_PDF_EXPORT = False

print(f"ROOT: {ROOT}")
print(f"RUN_DIR: {RUN_DIR}")
print(f"INFO_DIR: {INFO_DIR}")
print(f"POSE_RUN_DIR: {POSE_RUN_DIR}")
print(f"POSE_INFO_DIR: {POSE_INFO_DIR}")

# %% [markdown]
# ## Optional runner commands
#
# These commands document exactly how the relevant caches are produced. They do
# not execute unless the booleans above are changed.

# %%
source_trial_recompute_command = [
    sys.executable,
    "-m",
    "declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion",
    "--run-dir",
    str(RUN_DIR),
    "--out-dir",
    str(INFO_DIR),
    "--summaries",
    "delta_mean",
    "--families",
    "empirical,brownian,rotated",
    "--scale-ids",
    "rel_0p25x,rel_0p5x,rel_1x,rel_1p5x,rel_2x",
    "--latent-names",
    "pyramid_local_field",
    "--pca-k-list",
    "16",
    "--ridge-alpha-mode",
    "fixed",
    "--fixed-ridge-alpha",
    "10",
    "--decode-group-mode",
    "source_trial",
    "--n-bootstrap",
    "50",
    "--information-ci-mode",
    "decode_bootstrap",
    "--seed",
    "0",
]

pose_unaware_recompute_command = [
    sys.executable,
    "-m",
    "declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_pose_unaware_motion",
    "--run-dir",
    str(POSE_RUN_DIR),
    "--out-dir",
    str(POSE_INFO_DIR),
    "--summaries",
    "delta_mean",
    "--families",
    "empirical",
    "--scale-ids",
    "rel_0p25x,rel_0p5x,rel_1x,rel_1p5x,rel_2x",
    "--latent-names",
    "pyramid_local_field",
    "--pca-k-list",
    "16",
    "--ridge-alpha-mode",
    "fixed",
    "--fixed-ridge-alpha",
    "10",
    "--outer-folds",
    "5",
    "--inner-folds",
    "3",
    "--decode-group-mode",
    "source_trial",
    "--n-bootstrap",
    "50",
    "--information-ci-mode",
    "decode_bootstrap",
    "--seed",
    "0",
]

selected_figure_rebuild_command = [
    sys.executable,
    "-m",
    "declan.figure4_active_sensing_atlas.scripts.build_selected_figure4_v5_compact_layout",
]

pdf_export_command = [
    sys.executable,
    "scripts/export_percent_walkthrough_pdf.py",
    "notebooks/aggregate_feature_information_model_walkthrough.py",
    "--output",
    str(WALKTHROUGH_OUT_DIR / "aggregate_feature_information_model_walkthrough.pdf"),
    "--stdout-mode",
    "capped",
    "--max-stdout-lines",
    "24",
]

for label, command in [
    ("source-trial information recompute", source_trial_recompute_command),
    ("pose-unaware proxy recompute", pose_unaware_recompute_command),
    ("selected Figure 4 rebuild", selected_figure_rebuild_command),
    ("walkthrough PDF export", pdf_export_command),
]:
    print(f"\n# {label}\n{command_text(command)}")

if RUN_SOURCE_TRIAL_RECOMPUTE:
    subprocess.run(source_trial_recompute_command, cwd=ROOT, check=True)
if RUN_POSE_UNAWARE_RECOMPUTE:
    subprocess.run(pose_unaware_recompute_command, cwd=ROOT, check=True)
if RUN_SELECTED_FIGURE_REBUILD:
    subprocess.run(selected_figure_rebuild_command, cwd=ROOT, check=True)
if RUN_PDF_EXPORT:
    subprocess.run(pdf_export_command, cwd=ROOT, check=True)

# %% [markdown]
# ## Load the cached tables
#
# The strict source-trial cache gives the main motion-rendered information gain.
# The pose-unaware cache gives the same-axis hidden-trajectory proxy.

# %%
images = read_csv_required(RUN_DIR / "analysis_images.csv")
motion_meta = read_csv_required(RUN_DIR / "aggregate_motion_metadata.csv")
trace_meta = read_csv_required(RUN_DIR / "trace_bank_metadata.csv")
decode_summary = read_csv_required(INFO_DIR / "incremental_decode_summary.csv")
gain = read_csv_required(INFO_DIR / "incremental_gain_vs_static.csv")
gain_contrasts = read_csv_required(INFO_DIR / "incremental_gain_contrasts.csv")
pose_proxy = read_csv_required(POSE_INFO_DIR / "pose_unaware_train_mean_test_samples_proxy.csv")
pose_scores = read_csv_required(POSE_INFO_DIR / "pose_unaware_train_mean_test_samples_decode_scores.csv")

info_meta = read_json_required(INFO_DIR / "run_metadata.json")
pose_meta = read_json_required(POSE_INFO_DIR / "run_metadata.json")

load_summary = pd.DataFrame(
    [
        {"table": "analysis_images", "rows": len(images), "columns": len(images.columns)},
        {"table": "aggregate_motion_metadata", "rows": len(motion_meta), "columns": len(motion_meta.columns)},
        {"table": "trace_bank_metadata", "rows": len(trace_meta), "columns": len(trace_meta.columns)},
        {"table": "incremental_gain_vs_static", "rows": len(gain), "columns": len(gain.columns)},
        {"table": "pose_unaware_proxy", "rows": len(pose_proxy), "columns": len(pose_proxy.columns)},
    ]
)
show_table(load_summary)

# %% [markdown]
# ## One-sentence claim
#
# The aggregate feature information model is there to make a specific model
# claim:
#
# > In the V1 twin, a static response plus the trajectory-rendered response
# > change carries more recoverable feature evidence than the stabilized static
# > response alone.
#
# There is a necessary observer caveat. The trajectory is used to render the
# response movie. It is not handed to the aggregate ridge decoder as an explicit
# input. So the clean wording is **trajectory-rendered / pose-available through
# rendering**, not a full biological proof that a downstream observer can use the
# signal without estimating pose. The pose-unaware proxy near the end asks what
# happens when the trajectory sample is hidden.

# %% [markdown]
# ## Step 1: define the sample
#
# A cached row is approximately
#
# $$
# i = (\text{ROI window}, \text{source row}, \text{source trial},
#      \text{motion family}, \text{motion scale}, \text{response summary},
#      \text{feature target}).
# $$
#
# The important split decision is source-trial grouping. Crops from the same
# original session/trial should not be split across train and test.

# %%
sample_table = images.copy()
sample_table["source_trial"] = sample_table["session"].astype(str) + "::trial_" + sample_table["trial_idx"].astype(int).astype(str)
trial_counts = sample_table.groupby("source_trial").size().sort_values(ascending=False)
session_counts = sample_table.groupby("session").size().sort_values(ascending=False)

sample_audit = pd.DataFrame(
    [
        {"quantity": "selected windows/images", "value": int(len(sample_table))},
        {"quantity": "unique source trials", "value": int(trial_counts.size)},
        {"quantity": "recording sessions", "value": int(session_counts.size)},
        {"quantity": "source trials with >1 crop", "value": int((trial_counts > 1).sum())},
        {"quantity": "max crops from one source trial", "value": int(trial_counts.max())},
        {"quantity": "primary decode grouping", "value": info_meta.get("decode_group_mode")},
        {"quantity": "primary decode groups", "value": info_meta.get("n_decode_groups")},
        {"quantity": "outer folds", "value": info_meta.get("outer_folds")},
    ]
)
show_table(sample_audit)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
axes[0].hist(trial_counts.to_numpy(), bins=np.arange(1, trial_counts.max() + 2) - 0.5, color="#2a5c8a")
axes[0].set_xlabel("crops per source trial")
axes[0].set_ylabel("number of source trials")
axes[0].set_title("Why source-trial grouping matters")

session_counts.sort_values().plot(kind="barh", ax=axes[1], color="#6f7d85")
axes[1].set_xlabel("selected windows")
axes[1].set_ylabel("session")
axes[1].set_title("Session contribution")
plt.tight_layout()

# %% [markdown]
# ## Step 2: render the retinal movie
#
# For a static baseline, the image is held fixed. For a motion condition, the
# image is sampled along a trajectory:
#
# $$
# R_i^\tau(t) = F_\theta\left(I_i(x + \tau_x(t), y + \tau_y(t))\right),
# $$
#
# where \(F_\theta\) is the deterministic V1 twin and \(\tau(t)\) is a measured
# or control eye trace. In this aggregate analysis, real backimage windows and
# real trace samples are pooled into a trajectory family; the specific
# image-trace pairing is not the claim.

# %%
motion = motion_meta[motion_meta["family"].ne("static")].copy()
motion["scale_value"] = motion["scale_id"].map(scale_value)

qc = (
    motion.groupby(["family", "scale_id", "scale_value"], as_index=False)
    .agg(
        requested_rms_deg=("requested_rms_deg", "mean"),
        effective_rms_deg=("effective_rms_deg", "mean"),
        clipped_fraction=("rms_clipped_high", "mean"),
        p95_speed_deg_s=("speed_p95_deg_s", "median"),
        n=("response_id", "size"),
    )
    .sort_values(["family", "scale_value"])
)
show_table(qc.head(12))

fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.2))
colors = {"empirical": "#1f4f7a", "brownian": "#8c9499", "rotated": "#8a6bb0", "ou": "#b76e2b"}
for family, block in qc.groupby("family"):
    block = block.sort_values("scale_value")
    axes[0].plot(
        block["scale_value"],
        block["effective_rms_deg"],
        marker="o",
        label=family,
        color=colors.get(family, None),
    )
    axes[1].plot(
        block["scale_value"],
        block["clipped_fraction"],
        marker="o",
        label=family,
        color=colors.get(family, None),
    )
for ax in axes:
    ax.set_xticks(sorted(qc["scale_value"].unique()), [scale_label(v) for v in sorted(qc["scale_value"].unique())])
    ax.set_xlabel("requested motion scale")
axes[0].set_ylabel("effective RMS displacement (deg)")
axes[1].set_ylabel("fraction clipped")
axes[0].set_title("Motion scale is realized")
axes[1].set_title("Render clipping audit")
axes[0].legend(frameon=False, fontsize=8)
plt.tight_layout()

# %% [markdown]
# ## Step 3: choose the decoded visual variable
#
# The target is a local image-feature vector, not the eye trace and not the raw
# population response:
#
# $$
# \Phi_i \in \mathbb{R}^k.
# $$
#
# In production, the raw local pyramid feature vector is reduced inside each
# training fold. The promoted row uses `pyramid_local_field`, \(k=16\).
# The plot below is a global audit of the same cached target array, only for
# intuition; it is not a replacement for train-fold PCA in the decoder.

# %%
latent_arrays = np.load(POSE_RUN_DIR / "latent_feature_arrays.npz")
target_raw = np.asarray(latent_arrays["pyramid_local_field"], dtype=np.float64)
target_centered = target_raw - np.nanmean(target_raw, axis=0, keepdims=True)

# This SVD is only a tutorial audit. The production decoder fits target PCA
# inside each training fold to avoid leakage.
u, s, vt = np.linalg.svd(target_centered, full_matrices=False)
target_var = (s**2) / max(1, target_centered.shape[0] - 1)
target_evr = target_var / np.sum(target_var)

target_summary = pd.DataFrame(
    [
        {"quantity": "raw target rows", "value": target_raw.shape[0]},
        {"quantity": "raw target dimensions", "value": target_raw.shape[1]},
        {"quantity": "promoted PCA dimensions", "value": int(info_meta.get("pca_k_list", [16])[0])},
        {"quantity": "global variance in first 16 PCs", "value": float(np.sum(target_evr[:16]))},
    ]
)
show_table(target_summary)

fig, ax = plt.subplots(figsize=(6.8, 3.0))
x = np.arange(1, 33)
ax.bar(x, target_evr[:32], color="#2a5c8a", alpha=0.75, label="per-PC variance")
ax.plot(x, np.cumsum(target_evr[:32]), color="#222222", marker="o", ms=3, label="cumulative")
ax.axvline(16, color="#b23a48", lw=1.2, ls="--", label="promoted k=16")
ax.set_xlabel("global target PC")
ax.set_ylabel("fraction of target variance")
ax.set_title("Pyramid target compression audit")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()

# %% [markdown]
# ## Step 4: mean versus delta_mean
#
# The naming is easy to blur, so this walkthrough keeps three response summaries
# separate:
#
# $$
# S_i^0 = \frac{1}{T}\sum_t R_i^0(t)
# $$
#
# is the stabilized/static mean response. Because the movie is stabilized, this
# is the image response held at one pose.
#
# $$
# \bar{S}_i^\tau = \frac{1}{T}\sum_t R_i^\tau(t)
# $$
#
# is the mean response over the rendered motion movie. It still contains the
# image's ordinary static drive plus whatever the trajectory changed.
#
# $$
# \Delta S_i^\tau = \bar{S}_i^\tau - S_i^0
# $$
#
# is `delta_mean`: the mean motion-induced response change relative to the
# stabilized image response.
#
# The promoted feature matrix is **static plus delta**, not motion alone:
#
# $$
# X_i^\text{static} = z_\text{train}(S_i^0),
# $$
#
# $$
# X_i^\text{static+delta} =
# \left[z_\text{train}(S_i^0),\;
#       z_\text{train}(\Delta S_i^\tau)\right].
# $$
#
# A different diagnostic variant would be **static plus mean**,
# \([S_i^0,\bar{S}_i^\tau]\). In an unregularized infinite-data linear decoder,
# \([S_i^0,\bar{S}_i^\tau]\) and \([S_i^0,\Delta S_i^\tau]\) span the same space.
# In the actual pipeline they are not interchangeable, because inputs are
# train-fold z-scored, data are finite, and ridge regularization changes the
# geometry. We promote `delta_mean` because it asks the cleaner question:
# **does the motion-induced change add feature evidence beyond the static
# image anchor?**

# %%
responses = np.load(POSE_RUN_DIR / "response_summary_arrays.npz")
sample_responses = np.load(POSE_RUN_DIR / "response_sample_summary_arrays.npz")

static_mean = np.asarray(responses["mean__static__static"], dtype=np.float64)
scale_ids = ["rel_0p25x", "rel_0p5x", "rel_1x", "rel_1p5x", "rel_2x"]

summary_rows: list[dict[str, Any]] = []
for scale_id in scale_ids:
    motion_mean = np.asarray(responses[f"mean__empirical__{scale_id}"], dtype=np.float64)
    delta_mean = np.asarray(responses[f"delta_mean__empirical__{scale_id}"], dtype=np.float64)
    delta_samples = np.asarray(sample_responses[f"delta_mean__empirical__{scale_id}"], dtype=np.float64)
    reconstructed_delta = motion_mean - static_mean
    delta_error = np.abs(delta_mean - reconstructed_delta)
    sample_scatter = np.sqrt(np.nanmean((delta_samples - delta_mean[:, None, :]) ** 2, axis=(1, 2)))
    summary_rows.append(
        {
            "scale_id": scale_id,
            "scale": scale_value(scale_id),
            "static_mean_l2_median": float(np.median(np.linalg.norm(static_mean, axis=1))),
            "motion_mean_l2_median": float(np.median(np.linalg.norm(motion_mean, axis=1))),
            "delta_mean_l2_median": float(np.median(np.linalg.norm(delta_mean, axis=1))),
            "delta_equals_motion_minus_static_max_abs": float(np.nanmax(delta_error)),
            "hidden_sample_scatter_median": float(np.median(sample_scatter)),
            "feature_dim_static": int(static_mean.shape[1]),
            "feature_dim_static_plus_delta": int(static_mean.shape[1] + delta_mean.shape[1]),
            "trace_samples_per_image": int(delta_samples.shape[1]),
        }
    )
response_audit = pd.DataFrame(summary_rows)
show_table(response_audit)

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
axes[0].plot(response_audit["scale"], response_audit["static_mean_l2_median"], marker="o", color="#555555", label="static mean S0")
axes[0].plot(response_audit["scale"], response_audit["motion_mean_l2_median"], marker="s", color="#1f4f7a", label="motion mean S_tau")
axes[0].plot(response_audit["scale"], response_audit["delta_mean_l2_median"], marker="^", color="#b23a48", label="delta mean Delta S")
axes[0].set_ylabel("median response-summary L2 norm")
axes[0].set_title("Mean response versus motion-induced change")
axes[0].legend(frameon=False, fontsize=8)

axes[1].plot(response_audit["scale"], response_audit["hidden_sample_scatter_median"], marker="o", color="#b23a48")
axes[1].set_ylabel("median hidden-sample scatter")
axes[1].set_title("Hidden trajectory samples add nuisance scatter")
for ax in axes:
    ax.set_xticks(response_audit["scale"], [scale_label(v) for v in response_audit["scale"]])
    ax.set_xlabel("motion scale")
plt.tight_layout()

# %% [markdown]
# ## Step 5: fit a matched ridge decoder
#
# For each condition \(c\), fit a linear decoder from response summaries to the
# target:
#
# $$
# \hat{\Phi}_i^c = W_c X_i^c + b_c,
# $$
#
# $$
# (\hat{W}_c,\hat{b}_c)
# =
# \arg\min_{W,b}
# \sum_{i \in D_\text{train}}
# \|\Phi_i - (W X_i^c + b)\|_2^2
# + \lambda \|W\|_F^2.
# $$
#
# Decision points:
#
# - folds are grouped by `source_trial`;
# - target PCA and input z-scoring are fit on the train fold;
# - \(\lambda\), implemented as ridge `alpha`, is fixed at 10.0;
# - the promoted motion model is `static + delta_mean`, not `delta_mean` alone;
# - `static + mean` is a diagnostic parameterization, not the promoted wording.

# %%
decode_audit = decode_summary[
    ["model", "family", "scale_id", "mean_neg_mse", "r2", "chosen_alpha_median", "target_dim", "feature_dim", "n_images", "decode_group_mode", "n_decode_groups"]
].copy()
decode_audit["readout_interpretation"] = np.where(
    decode_audit["model"].eq("static_plus_motion"),
    "static + delta_mean",
    "static mean only",
)
decode_audit = decode_audit.sort_values(["model", "family", "scale_id"])
show_table(
    decode_audit[
        [
            "model",
            "readout_interpretation",
            "family",
            "scale_id",
            "mean_neg_mse",
            "chosen_alpha_median",
            "target_dim",
            "feature_dim",
            "decode_group_mode",
            "n_decode_groups",
        ]
    ].head(12)
)

decision_points = pd.DataFrame(
    [
        {
            "decision": "baseline",
            "choice": "stabilized/static mean response",
            "reason": "asks whether motion adds feature evidence beyond the same image held still",
        },
        {
            "decision": "delta summary",
            "choice": "delta_mean",
            "reason": "tests whether the trajectory-induced change adds feature evidence around the static anchor",
        },
        {
            "decision": "mean summary diagnostic",
            "choice": "static + mean = [S0, S_tau]",
            "reason": "useful audit, but not the promoted interpretation because the movie mean still carries ordinary static image drive",
        },
        {
            "decision": "target",
            "choice": "pyramid_local_field, k=16",
            "reason": "V1-adjacent local feature target with manageable held-out dimensionality",
        },
        {
            "decision": "split",
            "choice": "source_trial grouped CV",
            "reason": "prevents crops from the same source trial from entering both train and test",
        },
        {
            "decision": "regularization",
            "choice": "fixed ridge alpha=10",
            "reason": "keeps static and motion decoders on matched capacity",
        },
        {
            "decision": "information score",
            "choice": "diagonal Gaussian residual bits",
            "reason": "converts held-out residual shrinkage into feature-information units",
        },
        {
            "decision": "intervals",
            "choice": "point-centered decode-bootstrap CIs",
            "reason": "reruns the fold-level decode pipeline under grouped resampling and keeps point estimates inside CIs",
        },
        {
            "decision": "pose audit",
            "choice": "same-axis pose-unaware hidden-sample proxy",
            "reason": "tests whether motion remains helpful when the trace sample is hidden",
        },
    ]
)
show_table(decision_points)

# %% [markdown]
# ## Step 6: convert decoder residuals into information
#
# Let \(e_i^0\) be the held-out residual under the static decoder and \(e_i^c\)
# the held-out residual under condition \(c\). The headline score is
#
# $$
# \Delta I_\text{diag}(c)
# =
# \frac{1}{2}
# \sum_{j=1}^{k}
# \log_2
# \frac{\operatorname{Var}(e^0_j)}
#      {\operatorname{Var}(e^c_j)}.
# $$
#
# Positive values mean the condition leaves less target uncertainty than the
# stabilized baseline under the matched decoder. Negative values mean it leaves
# more uncertainty than the baseline.

# %%
ci_checks = pd.DataFrame(
    [
        {
            "csv": "incremental_gain_vs_static",
            "rows": len(gain),
            "bad_ci_rows": len(validate_point_inside_ci(gain, "incremental_gain_info_diag_bits")),
            "ci_method": ", ".join(sorted(gain["information_ci_method"].dropna().unique())),
        },
        {
            "csv": "pose_unaware_train_mean_test_samples_proxy",
            "rows": len(pose_proxy),
            "bad_ci_rows": len(validate_point_inside_ci(pose_proxy, "incremental_gain_info_diag_bits")),
            "ci_method": ", ".join(sorted(pose_proxy["information_ci_method"].dropna().unique())),
        },
    ]
)
show_table(ci_checks)

# %% [markdown]
# ## Step 7: the main aggregate feature-information result
#
# The promoted curve compares \([S^0,\Delta S^\tau]\) against \([S^0]\).
# Brownian and rotated traces are shown as controls, but the main panel claim is
# the empirical recorded-drift static-plus-delta gain over static under the
# source-trial grouped information metric.

# %%
plot_gain = gain.copy()
plot_gain["scale"] = plot_gain["scale_id"].map(scale_value)
plot_gain = plot_gain.sort_values(["family", "scale"])

fig, ax = plt.subplots(figsize=(6.8, 3.6))
labels = {"empirical": "recorded drift", "brownian": "random drift", "rotated": "rotated drift"}
markers = {"empirical": "o", "brownian": "s", "rotated": "^"}
for family, block in plot_gain.groupby("family"):
    block = block.sort_values("scale")
    y = block["incremental_gain_info_diag_bits"].to_numpy(dtype=float)
    lo = block["info_diag_ci95_low"].to_numpy(dtype=float)
    hi = block["info_diag_ci95_high"].to_numpy(dtype=float)
    ax.errorbar(
        block["scale"],
        y,
        yerr=np.vstack([y - lo, hi - y]),
        marker=markers.get(family, "o"),
        lw=2.0 if family == "empirical" else 1.4,
        color=colors.get(family, None),
        alpha=1.0 if family == "empirical" else 0.75,
        capsize=0,
        label=labels.get(family, family),
    )
ax.axhline(0, color="#222222", lw=0.9)
ax.set_xticks(sorted(plot_gain["scale"].unique()), [scale_label(v) for v in sorted(plot_gain["scale"].unique())])
ax.set_xlabel("motion scale")
ax.set_ylabel("information gain over stabilized (bits)")
ax.set_title("Source-trial grouped static-plus-delta decoder")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()

empirical_table = plot_gain[plot_gain["family"].eq("empirical")][
    ["scale_id", "incremental_gain_info_diag_bits", "info_diag_ci95_low", "info_diag_ci95_high", "incremental_gain_neg_mse"]
].copy()
show_table(empirical_table)

# %% [markdown]
# ## Step 8: what does pose-unaware mean here?
#
# The main curve is not a full pose-aware oracle, because the aggregate ridge
# decoder does not receive \(\tau\) as an explicit input. But the response was
# rendered from a known trajectory family. The pose-unaware proxy asks a stricter
# question:
#
# 1. train the static-plus-delta decoder on the image-mean motion-change summary;
# 2. test on individual held-out trajectory samples from the same image class;
# 3. do not tell the decoder which trajectory sample produced the response.
#
# The practical contrast is
#
# $$
# \Delta I_\text{hidden sample}
# =
# I(\Phi;\;S^0,\Delta S^\tau_\text{sample})
# -
# I(\Phi;\;S^0),
# $$
#
# evaluated with the same residual-information axis. The known-eye proxy uses
# the image-mean \(\Delta S^\tau\) summary; the hidden-sample proxy replaces
# that averaged change with sample-level \(\Delta S^\tau_\text{sample}\)
# responses.

# %%
pose_plot = pose_proxy.copy()
pose_plot["scale"] = pose_plot["scale_id"].map(scale_value)
pose_plot = pose_plot.sort_values(["observer", "scale"])

observer_labels = {
    "known_eye_train_mean_test_mean_proxy": "known-eye mean proxy",
    "pose_unaware_train_mean_test_hidden_samples": "hidden trajectory samples",
    "hidden_sample_minus_known_eye_penalty": "hidden-minus-known penalty",
}
observer_colors = {
    "known_eye_train_mean_test_mean_proxy": "#1f4f7a",
    "pose_unaware_train_mean_test_hidden_samples": "#b23a48",
    "hidden_sample_minus_known_eye_penalty": "#222222",
}
observer_linestyles = {
    "known_eye_train_mean_test_mean_proxy": "-",
    "pose_unaware_train_mean_test_hidden_samples": "--",
    "hidden_sample_minus_known_eye_penalty": ":",
}

fig, ax = plt.subplots(figsize=(6.8, 3.6))
for observer, block in pose_plot.groupby("observer"):
    block = block.sort_values("scale")
    y = block["incremental_gain_info_diag_bits"].to_numpy(dtype=float)
    lo = block["info_diag_ci95_low"].to_numpy(dtype=float)
    hi = block["info_diag_ci95_high"].to_numpy(dtype=float)
    ax.errorbar(
        block["scale"],
        y,
        yerr=np.vstack([y - lo, hi - y]),
        marker="o",
        lw=2.0,
        linestyle=observer_linestyles.get(observer, "-"),
        color=observer_colors.get(observer, None),
        capsize=0,
        label=observer_labels.get(observer, observer),
    )
ax.axhline(0, color="#222222", lw=0.9)
ax.set_xticks(sorted(pose_plot["scale"].unique()), [scale_label(v) for v in sorted(pose_plot["scale"].unique())])
ax.set_xlabel("motion scale")
ax.set_ylabel("information gain over stabilized (bits)")
ax.set_title("Same-axis pose-unaware proxy")
ax.legend(frameon=True, fontsize=8)
ax.get_legend().get_frame().set_alpha(0.9)
plt.tight_layout()

pose_table = pose_plot[
    [
        "observer",
        "scale_id",
        "incremental_gain_info_diag_bits",
        "info_diag_ci95_low",
        "info_diag_ci95_high",
        "incremental_gain_neg_mse",
    ]
].copy()
show_table(pose_table)

# %% [markdown]
# ## Step 9: combine the interpretation
#
# The current numeric summary:
#
# - recorded drift is positive over static at every plotted scale;
# - hidden trajectory samples have negative point estimates over static at every
#   plotted scale, though their diagonal-bit CIs cross zero;
# - the hidden-minus-known penalty is strongly negative at every scale.
#
# So the clean interpretation is:
#
# > In this V1-twin model, adding the trajectory-rendered response change
# > \(\Delta S^\tau\) to the stabilized response \(S^0\) adds feature evidence
# > over \(S^0\) alone. If the trajectory sample is hidden, the same
# > motion-induced structure becomes nuisance variability.

# %%
empirical = plot_gain[plot_gain["family"].eq("empirical")].copy()
hidden = pose_plot[pose_plot["observer"].eq("pose_unaware_train_mean_test_hidden_samples")].copy()
known = pose_plot[pose_plot["observer"].eq("known_eye_train_mean_test_mean_proxy")].copy()
penalty = pose_plot[pose_plot["observer"].eq("hidden_sample_minus_known_eye_penalty")].copy()

merged = (
    empirical[["scale_id", "scale", "incremental_gain_info_diag_bits", "info_diag_ci95_low", "info_diag_ci95_high"]]
    .rename(
        columns={
            "incremental_gain_info_diag_bits": "recorded_drift_bits",
            "info_diag_ci95_low": "recorded_drift_ci_low",
            "info_diag_ci95_high": "recorded_drift_ci_high",
        }
    )
    .merge(
        hidden[["scale_id", "incremental_gain_info_diag_bits", "info_diag_ci95_low", "info_diag_ci95_high"]].rename(
            columns={
                "incremental_gain_info_diag_bits": "pose_unaware_bits",
                "info_diag_ci95_low": "pose_unaware_ci_low",
                "info_diag_ci95_high": "pose_unaware_ci_high",
            }
        ),
        on="scale_id",
    )
    .merge(
        penalty[["scale_id", "incremental_gain_info_diag_bits", "info_diag_ci95_low", "info_diag_ci95_high"]].rename(
            columns={
                "incremental_gain_info_diag_bits": "hidden_minus_known_bits",
                "info_diag_ci95_low": "hidden_minus_known_ci_low",
                "info_diag_ci95_high": "hidden_minus_known_ci_high",
            }
        ),
        on="scale_id",
    )
    .sort_values("scale")
)

show_table(merged)

fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(merged))
width = 0.26
ax.bar(x - width, merged["recorded_drift_bits"], width=width, color="#1f4f7a", label="recorded drift")
ax.bar(x, merged["pose_unaware_bits"], width=width, color="#b23a48", label="hidden trajectory")
ax.bar(x + width, merged["hidden_minus_known_bits"], width=width, color="#555555", label="hidden-minus-known")
ax.axhline(0, color="#222222", lw=0.9)
ax.set_xticks(x, [scale_label(v) for v in merged["scale"]])
ax.set_xlabel("motion scale")
ax.set_ylabel("diagonal information gain (bits)")
ax.set_title("Interpretive bracket for the aggregate feature-information model")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()

# %% [markdown]
# ## Step 10: reusable model outputs
#
# The aggregate model outputs are intentionally reusable. A current paper draft
# can route them into any panel slot, but the model artifact itself is defined by
# these cache paths and metadata, not by that final layout decision.

# %%
model_outputs = pd.DataFrame(
    [
        {
            "artifact": "motion-rendered static-plus-delta information",
            "path": str(INFO_DIR / "incremental_gain_vs_static.csv"),
            "rows": len(gain),
            "decode_group_mode": info_meta.get("decode_group_mode"),
            "ci_method": info_meta.get("information_axis", {}).get("ci_method"),
        },
        {
            "artifact": "pose-unaware hidden-sample proxy",
            "path": str(POSE_INFO_DIR / "pose_unaware_train_mean_test_samples_proxy.csv"),
            "rows": len(pose_proxy),
            "decode_group_mode": pose_meta.get("decode_group_mode"),
            "ci_method": pose_meta.get("information_axis", {}).get("ci_method"),
        },
    ]
)
show_table(model_outputs)

# %% [markdown]
# ## What this tutorial justifies, and what it does not
#
# Justified by this analysis:
#
# - model-rendered motion changes can increase held-out feature evidence when
#   added to a stabilized response;
# - the result is measured on source-trial grouped held-out decoding;
# - the promoted unit is diagonal Gaussian residual-information gain in bits;
# - the same-axis pose-unaware proxy shows that hidden trajectory samples are a
#   serious nuisance.
#
# Not justified by this analysis alone:
#
# - total population information;
# - a claim that the animal has explicit access to the eye trace;
# - a full covariance-aware pose-hidden observer;
# - a natural-scale optimum for fixational motion;
# - a unique claim that recorded drift beats every plausible control at every
#   scale.
#
# The short paper wording should be:
#
# > Adding motion-rendered V1-twin response changes to stabilized responses
# > carries more recoverable local feature evidence than stabilized responses
# > alone; this benefit is observer-dependent, as a same-axis hidden-trajectory
# > proxy turns the motion response change into nuisance variability.
