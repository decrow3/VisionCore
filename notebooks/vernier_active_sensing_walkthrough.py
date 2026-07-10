# %% [markdown]
# # Vernier active-sensing walkthrough
#
# This is a notebook-style Python script. Open it in VS Code, Jupyter, or any
# editor that recognizes `# %%` cells to step through the Vernier analysis one
# piece at a time.
#
# The goal is not just to reproduce figures. The goal is to make the method
# legible:
#
# 1. What is the latent task variable?
# 2. What stimulus is rendered?
# 3. What eye trajectories are compared?
# 4. What does each readout assume the observer knows?
# 5. Where does the covariance penalty enter?
# 6. Which outputs are cached and which are recomputed?
#
# The default cells do not run the digital twin. They render pixels, inspect eye
# traces, and read existing cached Vernier outputs when available. Model-running
# cells are gated by explicit booleans.

# %%
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
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
from declan.vernier_active_sensing.metrics import (
    expected_counts,
    poisson_fisher_counts,
    pose_blind_diagonal_fisher,
)
from declan.vernier_active_sensing.stimulus import (
    RenderGeometry,
    VernierSpec,
    central_retina_frame,
    pixel_audit,
    render_world,
    sample_retina_movie,
)
from declan.vernier_active_sensing.trajectories import (
    TraceSet,
    condition_trace,
    valid_trace,
)
from scripts.temporal_decoding.stimulus_hires import N_LAGS as MODEL_HISTORY_FRAMES


try:
    from IPython.display import Image, Markdown, display
except Exception:  # pragma: no cover - only for plain script execution
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


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in df.columns:
        if col in {
            "fd_step_arcmin",
            "n",
            "mean_final_fisher",
            "median_final_fisher",
            "mean_final_threshold_proxy",
            "median_final_threshold_proxy",
            "final_fisher",
            "final_dprime2",
            "final_threshold_proxy",
            "mean_threshold_ratio",
            "median_threshold_ratio",
            "p_condition_beats_baseline",
            "pose_sigma_arcmin",
            "compact_k",
            "compact_alpha",
            "cov_shrinkage",
            "accuracy",
            "known_accuracy",
            "zero_accuracy",
            "joint_accuracy",
            "mean_gap_closure_vs_zero_known",
            "mean_margin_gap_closure_vs_zero_known",
            "trajectory_weight_sigma_arcmin",
            "mean_trajectory_weight_neff",
            "mean_trajectory_weight_true",
            "mean_posterior_neff_true",
            "mean_joint_score",
            "mean_known_eye_score",
            "mean_zero_eye_score",
            "across_scale",
            "along_scale",
            "median_nearest_rms_dist_arcmin",
            "mean_nearest_rms_dist_arcmin",
            "median_d_traj_over_d_sign",
            "mean_d_traj_over_d_sign",
            "fraction_d_traj_gt_d_sign",
            "fraction_d_traj_gt_10x_d_sign",
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
        }:
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


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def nearest_available_step(df: pd.DataFrame, desired: float) -> float | None:
    if df.empty or "fd_step_arcmin" not in df:
        return None
    vals = np.sort(pd.to_numeric(df["fd_step_arcmin"], errors="coerce").dropna().unique())
    if vals.size == 0:
        return None
    return float(vals[np.argmin(np.abs(vals - float(desired)))])


def condition_label(condition: str) -> str:
    labels = {
        "static_center": "static center",
        "static_repeated_phase": "static repeated phase",
        "static_phase_cloud_matched_positions": "phase cloud",
        "real_fem": "real FEM",
        "order_shuffled_positions": "order shuffled",
        "axis_horizontal": "horizontal only (across contour)",
        "axis_vertical": "vertical only (along contour)",
        "brownian_iso_1x": "Brownian iso 1x",
        "brownian_across_0x": "Brownian across 0x",
        "brownian_across_0p125x": "Brownian across 0.125x",
        "brownian_across_0p25x": "Brownian across 0.25x",
        "brownian_across_0p5x": "Brownian across 0.5x",
        "brownian_across_1x": "Brownian across 1x",
        "brownian_across_2x": "Brownian across 2x",
        "brownian_across_3x": "Brownian across 3x",
        "brownian_along_0x": "Brownian along 0x",
        "brownian_along_0p125x": "Brownian along 0.125x",
        "brownian_along_0p25x": "Brownian along 0.25x",
        "brownian_along_0p5x": "Brownian along 0.5x",
        "brownian_along_1x": "Brownian along 1x",
        "brownian_along_2x": "Brownian along 2x",
        "brownian_along_3x": "Brownian along 3x",
        "brownian_phase_cloud": "Brownian phase cloud",
        "brownian_order_shuffled": "Brownian order shuffled",
    }
    return labels.get(str(condition), str(condition).replace("_", " "))


def readout_label(readout: str) -> str:
    """Human-readable observer labels for cached readout names."""
    labels = {
        "pose_aware_diagonal_poisson": "known-trace Fisher",
        "pose_blind_diagonal_count_plus_marginal": "hidden-trace Fisher",
        "pose_blind_full_cov_optimal": "hidden-trace full covariance",
        "pose_blind_full_cov_optimal_unit_subset": "hidden-trace full covariance subset",
    }
    return labels.get(str(readout), str(readout).replace("_", " "))


COLORS = {
    "static_center": "#4d4d4d",
    "static_repeated_phase": "#8c6d31",
    "static_phase_cloud_matched_positions": "#1f77b4",
    "real_fem": "#d62728",
    "order_shuffled_positions": "#9467bd",
    "axis_horizontal": "#2ca02c",
    "axis_vertical": "#ff7f0e",
    "brownian_iso_1x": "#4d4d4d",
    "brownian_across_0x": "#9ecae1",
    "brownian_across_0p25x": "#6baed6",
    "brownian_across_0p5x": "#3182bd",
    "brownian_across_2x": "#08519c",
    "brownian_along_0x": "#fdd0a2",
    "brownian_along_0p25x": "#fdae6b",
    "brownian_along_0p5x": "#f16913",
    "brownian_along_2x": "#a63603",
    "brownian_phase_cloud": "#6f4e9b",
    "brownian_order_shuffled": "#b279a2",
    "scaled_real_0": "#7f7f7f",
    "scaled_real_0.125": "#5ab4ac",
    "scaled_real_0.25": "#3288bd",
    "scaled_real_0.5": "#17becf",
    "scaled_real_0.75": "#66c2a5",
    "scaled_real_1.5": "#bcbd22",
    "scaled_real_2": "#fdae61",
    "scaled_real_3": "#f46d43",
}

# %% [markdown]
# ## The story in plain English
#
# *Skip this section if you already know Vernier acuity and Fisher information.*
#
# ### What is Vernier acuity?
#
# A Vernier stimulus is two short vertical bars stacked end-to-end with a small
# gap between them. The bottom bar can be shifted slightly left or right relative
# to the top bar. Vernier acuity is the ability to detect this shift. Humans are
# strikingly sensitive to it — thresholds can be just a few arcseconds, far
# below the width of a single photoreceptor cone.
#
# ### Why do fixational eye movements matter?
#
# Even when you try to hold your gaze perfectly still, your eye makes small
# involuntary movements called fixational eye movements (FEM). These cause the
# retinal image to drift continuously — a "fixed" stimulus sweeps across your
# photoreceptors over time.
#
# This creates a fundamental ambiguity: a neuron cannot tell whether it fired
# because the Vernier bottom bar shifted right, or because the eye shifted left
# by the same amount. Both produce the same retinal image.
#
# ### The central question
#
# In this notebook, Vernier is the clean teaching case for **pose confusion**:
# two bars make the offset/eye-position ambiguity unusually transparent. That
# does not mean we know in advance that the stimulus is too poor for joint
# decoding. The right second-pass question is whether the available bar-edge
# structure, temporal continuity, and V1 response geometry provide enough
# constraint for a hidden-trace observer to recover useful Vernier evidence.
#
# The known-trace / hidden-trace contrast is therefore an upper-bound and
# nuisance-cost diagnostic, while the trajectory-table and joint-observer
# sections should be read as genuine tests of whether Vernier image structure can
# contribute to latent-pose marginalization.
#
# The active-sensing lesson is therefore not simply "more movement is better."
# During precise observations, the useful strategy may be to **reduce motion
# across the critical contour** while allowing safer motion along it.
#
# For this vertical Vernier stimulus:
#
# - **Across-contour motion** is horizontal. It changes the apparent left-right
#   offset and therefore directly competes with the task variable.
# - **Along-contour motion** is vertical. It moves along the bar and should be
#   less confusable with the Vernier offset.
#
# The key framing is therefore: real drift need not get longer along the contour;
# the important active-sensing move may be that drift gets **smaller across** the
# contour when precision matters.
#
# ### How we measure it
#
# We run a digital twin model of marmoset V1 neurons under different eye
# trajectory conditions and compute how precisely an ideal observer could read
# out the Vernier sign (+δ vs −δ) from the population response.
#
# Two metrics travel together through the walkthrough:
#
# - **Fisher information**: task-specific information about the Vernier offset.
#   It asks whether the response changes along the `+delta` versus `-delta`
#   direction, after accounting for count noise. The first demonstration uses
#   the known-trace upper bound, where the eye trace is supplied.
# - **General spatial spiking information (SSI)**: task-agnostic spatial
#   organization. It asks whether the response map is spatially concentrated,
#   regardless of whether that concentration helps Vernier discrimination.
#
# The first key comparison is whether SSI and known-trace Fisher move together.
# SSI says "the response is spatially informative"; Fisher says "the response is
# informative about this offset under this observer."
#
# The critical manipulation is whether the observer knows the current eye
# position:
#
# - **Known-trace**: observer knows exactly where the eye was at each frame.
#   This is an upper bound because the nuisance state is supplied for free. The
#   code calls this `pose_aware`.
# - **Hidden-trace**: observer is not given the eye-trajectory label. The code
#   calls this `pose_blind`, but the biological interpretation is narrower than
#   "realistic": a real animal may have extraretinal signals, priors, temporal
#   continuity, or learned joint image-pose structure.
#
# Comparing these two scores tells us how much position uncertainty costs. In
# Vernier, that cost is the point: the motion-induced signal is only usable if
# the latent eye trajectory can be resolved, marginalized, or otherwise
# controlled. Whether the Vernier movie contains enough structure for that is a
# testable question, not a premise.

# %% [markdown]
# ## Recommended live-demo order
#
# 1. Run setup and configuration.
# 2. Run the Vernier schematic.
# 3. Run the synthetic anisotropic Brownian FEM traces based on backimage
#    fixation statistics.
# 4. Step through the paired Fisher-versus-SSI toy calculations.
# 5. Run the synthetic along/across diffusion plots.
# 6. Only then load the cached real-twin Vernier results as an end-stage audit.
# 7. Use known-trace versus hidden-trace as an upper-bound/nuisance-cost
#    diagnostic.
# 8. Run joint/trajectory-table observer extensions as positive tests of whether
#    the Vernier movie contains useful image structure for pose marginalization.
# 9. Then compare with natural-image stimuli, where richer structure gives a
#    separate and likely stronger joint image-pose test.

# %%
# Simple Vernier schematic — pixel-array approach so the dark background is
# baked into the image itself (avoids set_axis_off() erasing facecolor).

def _make_vernier_image(offset_px: int, H: int = 180, W: int = 180) -> np.ndarray:
    """Return an H×W×3 float32 RGB image of the Vernier stimulus."""
    bw, bh, gap = 22, 58, 16
    cx, cy = W // 2, H // 2
    img = np.full((H, W, 3), 0.23, dtype=np.float32)  # dark grey background
    # top bar — always centred
    img[cy - gap // 2 - bh : cy - gap // 2, cx - bw // 2 : cx + bw // 2] = 1.0
    # bottom bar — shifted right by offset_px
    cb0 = max(0, cx - bw // 2 + offset_px)
    cb1 = min(W, cx + bw // 2 + offset_px)
    if cb0 < cb1:
        img[cy + gap // 2 : cy + gap // 2 + bh, cb0:cb1] = 1.0
    return img

_VSCH = 180  # canvas height/width in pixels
_VSCX, _VSCY = _VSCH // 2, _VSCH // 2  # centre pixel
_VSBH, _VSGAP = 58, 16                  # bar height and gap (pixels)

fig_schematic, axes_s = plt.subplots(1, 3, figsize=(8, 3.5), dpi=140)
for ax, _opx, _lbl in zip(
    axes_s,
    [-18, 0, 18],
    ["−δ\n(bottom bar shifted left)", "0\n(bars aligned)", "+δ\n(bottom bar shifted right)"],
):
    ax.imshow(_make_vernier_image(_opx, _VSCH, _VSCH), interpolation="nearest", aspect="auto")
    if _opx != 0:
        # Arrow from bar centre to shifted position, at the vertical midpoint of the bottom bar
        _ay = _VSCY + _VSGAP // 2 + _VSBH // 2
        ax.annotate(
            "",
            xy=(_VSCX + _opx, _ay),
            xytext=(_VSCX, _ay),
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=2.0, mutation_scale=16),
        )
    ax.set_title(_lbl, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for _sp in ax.spines.values():
        _sp.set_visible(False)

fig_schematic.suptitle(
    "The Vernier task: detect the direction of the bottom-bar shift\n"
    "The offset δ is typically less than 1 arcmin — much smaller than shown here",
    fontsize=9,
)
fig_schematic.tight_layout()
fig_schematic

# %% [markdown]
# ## The objective in one sentence
#
# The first pass asks how much the response tells an observer about a small
# Vernier offset when the eye trajectory is supplied. This gives the clean
# known-trace Fisher signal to compare with general SSI. The hidden-pose nuisance
# penalty is introduced later, in the **Known-trace versus hidden-trace** section.
#
# In the local finite-difference version, the task variable is the Vernier
# offset `theta`, measured in arcmin. For responses `mu(theta)`, the local
# derivative is approximated with symmetric offsets:
#
# ```text
# f_prime ~= [mu(+delta) - mu(-delta)] / (2 delta)
# ```
#
# The known-trace upper-bound metric is:
#
# ```text
# J_pop = f_prime.T Sigma^{-1} f_prime
# dprime2(+delta vs -delta) = Delta_mu.T Sigma^{-1} Delta_mu
# ```
#
# In this code, `delta` is the half-step stored as `fd_step_arcmin`, so the
# plus-minus separation is `2 * delta`. If `Sigma` is fixed across the two
# hypotheses, then `dprime2 ~= (2 * delta)^2 * J_pop`.
#
# The key comparison before the hidden-trace section is therefore:
#
# ```text
# known-trace Fisher = Vernier-aligned signal, with the trace supplied
# general SSI       = spatial organization, not tied to the Vernier offset
# ```

# %%
show_markdown(
    r"""
### Same objective, formatted

Local derivative from symmetric offsets:

$$
f'(\theta_0, \tau)
\approx
\frac{\mu(\theta_0 + \delta, \tau) - \mu(\theta_0 - \delta, \tau)}
{2\delta}
$$

Known-trace Fisher:

$$
J_{\mathrm{known}}
=
f'^{\top}\Sigma_{\mathrm{count}}^{-1} f'
$$

Spatial spiking information, shown alongside Fisher:

$$
I_{\mathrm{SSI}}(t)
=
\sum_u
\frac{\bar r_u(t)}{\sum_v \bar r_v(t)}
\left[
\frac{1}{|\mathcal X|}
\sum_{x\in\mathcal X}
\frac{r_u(x,t)}{\bar r_u(t)}
\log_2
\frac{r_u(x,t)}{\bar r_u(t)}
\right]
$$

The ratio \(r_u(x,t) / \bar r_u(t)\) is the internal SSI normalization. It is
appropriate: SSI asks how spatially concentrated a unit's map is relative to its
own mean rate. This is separate from later fold-change plots such as
\(\log_2 I_{\mathrm{SSI}}(\mathrm{scale}) / I_{\mathrm{SSI}}(\mathrm{static})\).
Those fold-change curves are diagnostics. Averaging unit-wise log ratios is a
geometric mean in ratio space, but it can still be dominated by tiny static SSI
denominators. Interpret absolute bits/spike, absolute changes from baseline,
and spike-weighted SSI budgets before interpreting fold-change curves.

Fisher is offset-specific. SSI is general spatial organization. The hidden-pose
covariance term is introduced later, after this known-trace comparison is clear.
"""
)

# %% [markdown]
# ## Why use this objective?
#
# A raw modulation objective answers a different question: "Does the eye
# movement change the response?" That is not enough for Vernier acuity, because
# a movement can create large response changes that do not help distinguish
# `+delta` from `-delta`.
#
# The known-trace Fisher/d' objective is useful here because it asks for signal
# aligned with the task variable after accounting for count-noise variance:
#
# ```text
# useful Vernier signal = response change along the +delta vs -delta axis
# count-noise burden    = Sigma_count
# known-trace score     = signal.T Sigma_count^{-1} signal
# ```
#
# So the linear-Gaussian form is not included as a claim that the brain is
# literally linear. It is the constrained discriminability metric: the
# inverse covariance term is the formal noise penalty. The later hidden-trace
# section adds the trajectory nuisance covariance. When a later analysis uses a
# flexible decoder or ridge regression without response-noise covariance, it is
# answering a less constrained question and should not be interpreted as the same
# objective.

# %% [markdown]
# ### Fisher information in plain English
#
# Fisher information answers: "How precisely can I estimate a parameter from
# these noisy measurements?"
#
# In our case: the parameter is the Vernier offset θ, and the measurements are
# V1 spike counts. High Fisher means the two offset conditions (+δ and −δ) are
# easy to tell apart — the response distributions barely overlap. Low Fisher
# means there is too much uninformative variance for the signal to stand out.
#
# The `Σ^{-1}` in the formula is the covariance penalty. Response directions that
# are highly variable but unrelated to the Vernier offset inflate the denominator
# and reduce the effective Fisher score.

# %%
# Toy illustration — Fisher information as signal vs noise
fig_fisher, axes_f = plt.subplots(1, 2, figsize=(9, 3.5), dpi=140)
_x = np.linspace(-5, 5, 600)

for ax, (_sigma, _title) in zip(
    axes_f,
    [
        (0.7, "Low noise → easy discrimination\n(high Fisher information)"),
        (2.0, "High noise or nuisance variance\n→ hard discrimination (low Fisher)"),
    ],
):
    _dp = np.exp(-0.5 * ((_x - 1.0) / _sigma) ** 2) / (_sigma * np.sqrt(2 * np.pi))
    _dm = np.exp(-0.5 * ((_x + 1.0) / _sigma) ** 2) / (_sigma * np.sqrt(2 * np.pi))
    ax.fill_between(_x, 0, _dp, alpha=0.45, color="#d62728", label="response to +δ")
    ax.fill_between(_x, 0, _dm, alpha=0.45, color="#1f77b4", label="response to −δ")
    ax.fill_between(_x, 0, np.minimum(_dp, _dm), alpha=0.55, color="#555555", label="overlap = errors")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_title(_title, fontsize=9)
    ax.set_xlabel("population response (1D projection)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks([])

axes_f[0].set_ylabel("probability density")
fig_fisher.suptitle(
    "Same signal amplitude (Δμ = 2), different noise — grey overlap drives error rate",
    fontsize=9,
    y=1.04,
)
fig_fisher.tight_layout()
fig_fisher

# %% [markdown]
# ## Assumptions made explicit
#
# These are the assumptions this walkthrough can justify from the code and from
# the attached lineage note. Anything not justified is marked as clarification
# needed rather than filled in by guesswork.

# %%
assumptions = pd.DataFrame(
    [
        {
            "assumption": "The task latent is Vernier offset theta.",
            "where_in_code": "VernierSpec.offset_arcmin and +/- fd_step_arcmin pairs",
            "reasoning_status": "Explicit in the runner and observer names.",
        },
        {
            "assumption": "The digital twin response is treated as the response mean mu(theta, tau).",
            "where_in_code": "compute_vernier_rates returns deterministic rates for a rendered movie",
            "reasoning_status": "Explicit implementation choice.",
        },
        {
            "assumption": "Small-offset Fisher is estimated by symmetric finite difference.",
            "where_in_code": "finite_difference_derivative(plus, minus, step_arcmin)",
            "reasoning_status": "Explicit implementation choice.",
        },
        {
            "assumption": "Known-trace scoring knows the eye trace and sums diagonal count Fisher across time.",
            "where_in_code": "poisson_fisher_counts for each trace",
            "reasoning_status": "Explicit implementation choice.",
        },
        {
            "assumption": "Hidden-trace scoring hides the trace and treats pose-marginal response spread as nuisance.",
            "where_in_code": "pose_blind_diagonal_fisher and pose_blind_full_covariance_fisher",
            "reasoning_status": "Explicit implementation choice.",
        },
        {
            "assumption": "Threshold proxy is 1 / sqrt(cumulative Fisher).",
            "where_in_code": "InformationResult.threshold_proxy",
            "reasoning_status": "Metric proxy only; not a fitted behavioral threshold.",
        },
        {
            "assumption": "General SSI is a task-agnostic spatial organization metric, not Vernier discriminability.",
            "where_in_code": "spatial_ssi_single_frame_np / spatial_ssi_timecourse_np",
            "reasoning_status": "Explicit formula matched to repository SSI helpers.",
        },
        {
            "assumption": "Early tutorial traces are synthetic anisotropic Brownian FEMs scaled from reviewed backimage fixation windows.",
            "where_in_code": "BACKIMAGE_FIXATION_WINDOWS_PATH and anisotropic_brownian_trace",
            "reasoning_status": "Pedagogical control; not a claim that real drift is exactly Brownian.",
        },
        {
            "assumption": "Default bar width, gap, length, contrast, and max spatial collapse are analysis defaults.",
            "where_in_code": "run_vernier_active_sensing.parse_args defaults",
            "reasoning_status": "Clarification needed for any behavioral-calibration claim.",
        },
        {
            "assumption": "Full-covariance unit subsets are diagnostic when the full population is too large.",
            "where_in_code": "select_pose_hidden_covariance_units top_abs_fd subset",
            "reasoning_status": "Computational/numerical guardrail, not a biological unit-selection claim.",
        },
    ]
)
show_table(assumptions)

# %% [markdown]
# ## Configuration
#
# `RUN_DIR` points at existing cached outputs. The preferred default is the
# scale/pose sweep because it includes known-trace, hidden-trace diagonal,
# pose-uncertainty, full-covariance, and compact-aware readouts.
#
# The booleans below are intentionally false by default. Set them true only
# when you want this notebook to launch the runner.

# %%
RUN_DIR_CANDIDATES = [
    ROOT / "outputs/vernier_active_sensing_scale_pose_sweep_gpu0",
    ROOT / "outputs/vernier_active_sensing_first_pass",
    ROOT / "outputs/vernier_joint_geometry_enumerated_gpu0_fixed",
    ROOT / "outputs/vernier_active_sensing_model_smoke_fixed",
]
RUN_DIR = first_existing(RUN_DIR_CANDIDATES)

JOINT_RUN_DIR = first_existing(
    [
        ROOT / "outputs/vernier_joint_geometry_enumerated_gpu0_fixed",
        ROOT / "outputs/vernier_joint_geometry_smoke_enumerated_fixed",
        ROOT / "outputs/vernier_joint_geometry_smoke",
    ]
)
TRAJECTORY_TABLE_RUN_DIR = first_existing(
    [
        ROOT / "outputs/vernier_trajectory_table_observer_poisson_all_v1",
        ROOT / "outputs/vernier_trajectory_table_observer_cross_prior_poisson_v1",
        ROOT / "outputs/vernier_trajectory_table_observer_smoke_poisson",
    ]
)
NOISY_TRAJECTORY_RUN_DIR = first_existing(
    [
        ROOT / "outputs/notebook_vernier_walkthrough/rr100_noisy_trajectory_observer",
    ]
)
HELDOUT_TRAJECTORY_RUN_DIR = first_existing(
    [
        ROOT / "outputs/notebook_vernier_walkthrough/rr100_heldout_trajectory_observer_along1",
        ROOT / "outputs/notebook_vernier_walkthrough/rr100_heldout_trajectory_observer_smoke",
    ]
)
CATALOG_MISMATCH_RUN_DIR = first_existing(
    [
        ROOT / "outputs/notebook_vernier_walkthrough/rr100_catalog_mismatch_diagnostic_along1",
    ]
)

WALKTHROUGH_OUT_DIR = ROOT / "outputs/notebook_vernier_walkthrough"
WALKTHROUGH_OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKIMAGE_FIXATION_WINDOWS_PATH = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)

FD_STEPS_FOR_PIXEL_AUDIT = [0.125, 0.25, 0.5, 1.0]
FD_STEP_TO_PLOT = 0.25
DEMO_TRACE_COUNT = 4
DEMO_MAX_FRAMES = 128
SEED = 0

SYNTHETIC_DIFFUSION_SCALES = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0]
SYNTHETIC_TRACE_COUNT = 4
SYNTHETIC_REFERENCE_RMS_QUANTILE = 0.5
SYNTHETIC_REFERENCE_MAX_RMS_DEG = 0.25
SYNTHETIC_REFERENCE_MAX_SPEED_P95_DPS = 30.0
INTOY_RUCCI_2020_SUSTAINED_FIXATION_D_ARCMIN2_S = 17.5
INTOY_RUCCI_2020_SUSTAINED_FIXATION_D_SEM_ARCMIN2_S = 2.2
INTOY_RUCCI_2020_FREEVIEWING_D_ARCMIN2_S = 26.2
INTOY_RUCCI_2020_FREEVIEWING_D_SEM_ARCMIN2_S = 2.6

RUN_RENDER_AUDIT = False
RUN_TINY_MODEL_SMOKE = False
RUN_CACHE_RECOMPUTE = False

# --- Population view ---
# "full"    : all 756 canonical channels (default, no extra loading).
# "reduced" : redundancy-resolved V1-RR post-activation pooling.
#             Requires population spec files under outputs/redundancy_resolved_v1_twin/.
RR100_MOVIE_MEDOID_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
RR192_VERSION = "V1-RR_complete_0p65_moviesplit0p75_pair0p60_rec4_blockjkP0p50n5L0p50n4_merge2nd1.01"

POPULATION_MODE = "full"
POPULATION_VERSION_NAME = RR100_MOVIE_MEDOID_VERSION

# --- SSI comparison across conditions ---
# Requires loading and running the digital twin. Safe to leave False for read-only
# walkthrough sessions.
RUN_SSI_FROM_MODEL = True
SSI_CONDITIONS = [
    "static_center",
    "real_fem",
    "order_shuffled_positions",
    "static_phase_cloud_matched_positions",
    "axis_horizontal",
    "axis_vertical",
]
SSI_TRACE_IDX = 0     # which demo trace (from the eye-trace pool above) to use
SSI_MAX_FRAMES = 40   # frames per condition — fewer = faster; 40 ≈ 333 ms at 120 Hz
SSI_DEVICE = None     # None → auto-detect (cuda if available)
SSI_BATCH_SIZE = 32

# --- SSI comparison across V1-RR population views ---
# Reuses the same full 756-channel Vernier spatial movies, then applies each
# post-activation population view. Spatial movies and summary metrics are cached
# under WALKTHROUGH_OUT_DIR so plot-only reruns stay light.
RUN_SSI_POPULATION_COMPARISON = True
SSI_POPULATION_FORCE_RECOMPUTE = True
SSI_POPULATION_COMPARISON_SPECS = [
    {"key": "full756", "label": "full 756", "version": None},
    {"key": "rr100_medoid", "label": "RR100 movie-medoid", "version": RR100_MOVIE_MEDOID_VERSION},
    {"key": "rr192", "label": "RR192 mean", "version": RR192_VERSION},
]
SSI_POPULATION_CACHE_DIR = WALKTHROUGH_OUT_DIR / "ssi_population_comparison"
SSI_POPULATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- SSI final-history activation-map diagnostic ---
# This is the "not quite instantaneous" view: render exactly one model-history
# worth of Vernier frames, run the temporal model on those lag windows, then score
# only the final spatial activation map. It uses history for context but avoids
# averaging a long sequence of history-dependent activation maps.
RUN_SSI_FINAL_HISTORY_MAP_DIAGNOSTIC = True
SSI_FINAL_HISTORY_FORCE_RECOMPUTE = False
SSI_FINAL_HISTORY_FRAMES = int(MODEL_HISTORY_FRAMES)
SSI_FINAL_HISTORY_CONDITIONS = SSI_CONDITIONS
SSI_FINAL_HISTORY_POPULATION_KEYS = ["full756", "rr100_medoid"]
SSI_FINAL_HISTORY_FD_STEP_ARCMIN = float(FD_STEP_TO_PLOT)
SSI_FINAL_HISTORY_BIN_SECONDS = 1.0 / 120.0
SSI_FINAL_HISTORY_CACHE_DIR = WALKTHROUGH_OUT_DIR / "ssi_final_history_map"
SSI_FINAL_HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- RR100 real-trace along=0 unit SSI diagnostic ---
# This mirrors the endpoint-history tutorial's unit diagnostic, but uses the
# original real-trace scale grid and averages SSI/maps over all response frames.
RUN_RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIAGNOSTIC = True
RR100_REAL_TRACE_SCALE_GRID_DIR = WALKTHROUGH_OUT_DIR / "rr100_real_trace_scale_grid"
RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR = RR100_REAL_TRACE_SCALE_GRID_DIR / "unit_ssi_along0_diagnostics"
RR100_REAL_TRACE_UNIT_SSI_N_TRACES = 16
RR100_REAL_TRACE_UNIT_SSI_MAX_FRAMES = 60
RR100_REAL_TRACE_UNIT_SSI_BATCH_SIZE = 64
RR100_REAL_TRACE_UNIT_SSI_TOP_UNITS = 12
RR100_REAL_TRACE_DENOMINATOR_FLOOR_BITS = 0.01

print(f"ROOT: {ROOT}")
print(f"BACKIMAGE_FIXATION_WINDOWS_PATH: {BACKIMAGE_FIXATION_WINDOWS_PATH}")
print(f"RUN_DIR: {RUN_DIR}")
print(f"JOINT_RUN_DIR: {JOINT_RUN_DIR}")
print(f"TRAJECTORY_TABLE_RUN_DIR: {TRAJECTORY_TABLE_RUN_DIR}")
print(f"NOISY_TRAJECTORY_RUN_DIR: {NOISY_TRAJECTORY_RUN_DIR}")
print(f"HELDOUT_TRAJECTORY_RUN_DIR: {HELDOUT_TRAJECTORY_RUN_DIR}")
print(f"CATALOG_MISMATCH_RUN_DIR: {CATALOG_MISMATCH_RUN_DIR}")

# %% [markdown]
# ## Optional runner commands
#
# These cells document the exact commands but do not execute unless you flip the
# corresponding boolean. This keeps the walkthrough safe to step through during
# explanation sessions.

# %%
render_audit_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.run_vernier_active_sensing",
    "--skip-model",
    "--fd-steps-arcmin",
    ",".join(str(v) for v in FD_STEPS_FOR_PIXEL_AUDIT),
    "--out-dir",
    str(WALKTHROUGH_OUT_DIR / "render_audit_run"),
]

tiny_model_smoke_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.run_vernier_active_sensing",
    "--out-dir",
    str(WALKTHROUGH_OUT_DIR / "tiny_model_smoke"),
    "--n-traces",
    "2",
    "--max-frames",
    "5",
    "--fd-steps-arcmin",
    "0.5",
    "--conditions",
    "static_center,real_fem,order_shuffled_positions",
    "--device",
    "cpu",
    "--batch-size",
    "2",
]

cache_recompute_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.run_vernier_active_sensing",
    "--out-dir",
    str(RUN_DIR) if RUN_DIR is not None else str(WALKTHROUGH_OUT_DIR),
    "--recompute-from-cache",
    "--run-full-cov-pose-blind",
    "--run-compact-aware-pose-blind",
]

rr100_noisy_trajectory_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.run_rr100_noisy_trajectory_observer",
    "--source-dir",
    str(WALKTHROUGH_OUT_DIR / "rr100_real_trace_scale_grid"),
    "--out-dir",
    str(WALKTHROUGH_OUT_DIR / "rr100_noisy_trajectory_observer"),
    "--trajectory-sigmas-arcmin",
    "0,0.125,0.25,0.5,1,2,inf",
]

rr100_real_trace_along0_unit_ssi_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.plot_rr100_real_trace_along0_unit_ssi",
    "--out-dir",
    str(RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR),
    "--summary-csv",
    str(RR100_REAL_TRACE_SCALE_GRID_DIR / "rr100_real_trace_scale_grid_summary.csv"),
    "--n-traces",
    str(RR100_REAL_TRACE_UNIT_SSI_N_TRACES),
    "--max-frames",
    str(RR100_REAL_TRACE_UNIT_SSI_MAX_FRAMES),
    "--fd-step-arcmin",
    str(FD_STEP_TO_PLOT),
    "--seed",
    str(SEED),
    "--batch-size",
    str(RR100_REAL_TRACE_UNIT_SSI_BATCH_SIZE),
    "--top-units",
    str(RR100_REAL_TRACE_UNIT_SSI_TOP_UNITS),
]

rr100_real_trace_along0_polarity_group_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.plot_rr100_along0_polarity_group_averages",
    "--mode",
    "real_trace",
    "--real-trace-dir",
    str(RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR),
    "--fd-step-arcmin",
    str(FD_STEP_TO_PLOT),
    "--real-trace-max-frames",
    str(RR100_REAL_TRACE_UNIT_SSI_MAX_FRAMES),
]

rr100_real_trace_along0_filtered_polarity_group_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.plot_rr100_along0_polarity_group_averages",
    "--mode",
    "real_trace",
    "--real-trace-dir",
    str(RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR),
    "--fd-step-arcmin",
    str(FD_STEP_TO_PLOT),
    "--real-trace-max-frames",
    str(RR100_REAL_TRACE_UNIT_SSI_MAX_FRAMES),
    "--min-static-ssi-bits",
    str(RR100_REAL_TRACE_DENOMINATOR_FLOOR_BITS),
    "--include-all-group",
]

rr100_real_trace_along0_denominator_diagnostic_command = [
    sys.executable,
    "-m",
    "declan.vernier_active_sensing.plot_rr100_along0_denominator_diagnostics",
    "--mode",
    "real_trace",
    "--real-trace-dir",
    str(RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR),
    "--fd-step-arcmin",
    str(FD_STEP_TO_PLOT),
    "--real-trace-max-frames",
    str(RR100_REAL_TRACE_UNIT_SSI_MAX_FRAMES),
    "--denominator-floor-bits",
    str(RR100_REAL_TRACE_DENOMINATOR_FLOOR_BITS),
]

print("Render-only command:")
print(" ".join(render_audit_command))
print("\nTiny model smoke command:")
print(" ".join(tiny_model_smoke_command))
print("\nCache recompute command:")
print(" ".join(cache_recompute_command))
print("\nRR100 noisy trajectory observer command:")
print(" ".join(rr100_noisy_trajectory_command))
print("\nRR100 real-trace along=0 unit SSI diagnostic command:")
print(" ".join(rr100_real_trace_along0_unit_ssi_command))
print("\nRR100 real-trace along=0 polarity-group command:")
print(" ".join(rr100_real_trace_along0_polarity_group_command))
print("\nRR100 real-trace along=0 filtered polarity-group command:")
print(" ".join(rr100_real_trace_along0_filtered_polarity_group_command))
print("\nRR100 real-trace along=0 denominator diagnostic command:")
print(" ".join(rr100_real_trace_along0_denominator_diagnostic_command))

# %%
if RUN_RENDER_AUDIT:
    subprocess.run(render_audit_command, cwd=ROOT, check=True)

if RUN_TINY_MODEL_SMOKE:
    subprocess.run(tiny_model_smoke_command, cwd=ROOT, check=True)

if RUN_CACHE_RECOMPUTE and RUN_DIR is not None:
    subprocess.run(cache_recompute_command, cwd=ROOT, check=True)

# %% [markdown]
# ## RR100 Real-Trace Along-0 Unit SSI Diagnostic
#
# This is the original-method counterpart to the endpoint-history unit plot. It
# uses the original real-trace anisotropic scale grid, recomputes only the
# along=0 spatial maps needed for highlighted units, and orders the highlighted
# legend and activation-map rows by the unit's y-value at across=1.
#
# Important interpretation detail: these unit curves are mean log2 ratios, so
# they are geometric-mean fold changes in ratio space. They are not an absolute
# information budget. Units with near-zero static SSI can still create large
# fold changes after tiny absolute SSI increases. The denominator diagnostic and
# filtered polarity plot below therefore report absolute SSI changes, a
# spike-weighted budget proxy, and retained-unit curves after excluding units
# below the static SSI floor `RR100_REAL_TRACE_DENOMINATOR_FLOOR_BITS`.

# %%
rr100_real_trace_unit_manifest_path = (
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_unit_ssi_manifest.json"
)
rr100_real_trace_unit_map_manifest_path = (
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR
    / "highlighted_unit_activation_maps"
    / "rr100_real_trace_along0_highlighted_unit_map_manifest.csv"
)
rr100_real_trace_unit_required = [
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR
    / "rr100_real_trace_along0_unit_ssi_lines_top_influence_with_activation_rows.png",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_unit_ssi_lines_top_influence.png",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_unit_ssi_leave_one_out.png",
    rr100_real_trace_unit_map_manifest_path,
]
if RUN_RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIAGNOSTIC and not all(
    path.exists() for path in rr100_real_trace_unit_required
):
    subprocess.run(rr100_real_trace_along0_unit_ssi_command, cwd=ROOT, check=True)
elif RUN_RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIAGNOSTIC:
    print(f"Using existing cached RR100 real-trace along=0 unit SSI diagnostic: {RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR}")
else:
    print("RR100 real-trace along=0 unit SSI diagnostic disabled.")
    print(" ".join(rr100_real_trace_along0_unit_ssi_command))

rr100_real_trace_polarity_required = [
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_polarity_group_averages.png",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_polarity_unit_table.csv",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_polarity_group_summary.csv",
]
if RUN_RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIAGNOSTIC and not all(
    path.exists() for path in rr100_real_trace_polarity_required
):
    subprocess.run(rr100_real_trace_along0_polarity_group_command, cwd=ROOT, check=True)
elif RUN_RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIAGNOSTIC:
    print(f"Using existing cached RR100 real-trace polarity-group diagnostic: {RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR}")

rr100_real_trace_filtered_polarity_required = [
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_polarity_static_ssi_ge_0p01_group_averages.png",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_polarity_static_ssi_ge_0p01_unit_table.csv",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_polarity_static_ssi_ge_0p01_group_summary.csv",
]
if RUN_RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIAGNOSTIC and not all(
    path.exists() for path in rr100_real_trace_filtered_polarity_required
):
    subprocess.run(rr100_real_trace_along0_filtered_polarity_group_command, cwd=ROOT, check=True)
elif RUN_RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIAGNOSTIC:
    print(f"Using existing cached RR100 real-trace filtered polarity-group diagnostic: {RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR}")

rr100_real_trace_denominator_required = [
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_denominator_diagnostics.png",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_denominator_static_floor_sweep.png",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_denominator_diagnostic_units.csv",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_denominator_diagnostic_groups.csv",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_denominator_diagnostic_summary.csv",
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_denominator_static_floor_sweep.csv",
]
if RUN_RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIAGNOSTIC and not all(
    path.exists() for path in rr100_real_trace_denominator_required
):
    subprocess.run(rr100_real_trace_along0_denominator_diagnostic_command, cwd=ROOT, check=True)
elif RUN_RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIAGNOSTIC:
    print(f"Using existing cached RR100 real-trace denominator diagnostic: {RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR}")

show_image_if_exists(
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR
    / "rr100_real_trace_along0_unit_ssi_lines_top_influence_with_activation_rows.png"
)
show_image_if_exists(RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_polarity_group_averages.png")
show_image_if_exists(
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_polarity_static_ssi_ge_0p01_group_averages.png"
)
show_image_if_exists(RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_denominator_diagnostics.png")
show_image_if_exists(RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_denominator_static_floor_sweep.png")
show_image_if_exists(RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_unit_ssi_leave_one_out.png")

rr100_real_trace_unit_top_df = read_csv_optional(
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_unit_ssi_top_units.csv"
)
if not rr100_real_trace_unit_top_df.empty:
    show_table(
        rr100_real_trace_unit_top_df[
            [
                "unit_index",
                "max_abs_leave_one_out_population_ratio_delta",
                "max_abs_log2_unit_ssi_vs_static",
                "static_unit_ssi_bits_per_spike_mean",
                "static_unit_mean_rate_mean",
            ]
        ],
        n=RR100_REAL_TRACE_UNIT_SSI_TOP_UNITS,
    )
else:
    print(f"No real-trace along=0 top-unit table found yet: {RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR}")

rr100_real_trace_polarity_summary_df = read_csv_optional(
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_polarity_group_summary.csv"
)
if not rr100_real_trace_polarity_summary_df.empty:
    show_table(rr100_real_trace_polarity_summary_df)

rr100_real_trace_denominator_summary_df = read_csv_optional(
    RR100_REAL_TRACE_ALONG0_UNIT_SSI_DIR / "rr100_real_trace_along0_denominator_diagnostic_summary.csv"
)
if not rr100_real_trace_denominator_summary_df.empty:
    show_table(
        rr100_real_trace_denominator_summary_df[
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
# is useful for finding denominator artifacts, not for claiming population
# information gains. If the large positive fold changes sit at tiny static SSI,
# check the absolute SSI table, the rate-weighted budget proxy, and the
# static-SSI threshold sweep. The filtered polarity-group figure is the cleaner
# retained-unit fold-change summary; the absolute/budget panels remain the
# primary evidence for whether spatial information actually increased.

# %% [markdown]
# ## Population view
#
# The **canonical twin** outputs 756 channel activation maps, one per recorded unit.
# The **redundancy-resolved twin** (V1-RR) applies a post-activation population
# view derived from multi-stimulus redundancy checks. The current default
# reduced view is the RR100 movie-medoid candidate; RR192 is kept as a
# comparison view below. The two single-population options are:
#
# - `POPULATION_MODE = "full"` — all 756 channels, identical to the standard
#   Vernier runner output. No extra files needed.
# - `POPULATION_MODE = "reduced"` — the population named by
#   `POPULATION_VERSION_NAME`. Requires the population spec files produced by
#   the redundancy-resolution scripts.
#
# For Fisher analysis the pooling must happen **before** spatial collapse, so
# running the SSI or a new Fisher sweep with the reduced population requires
# re-running the model or reusing cached spatial maps (see the SSI sections below).
#
# This section loads and describes the population spec without running the model.

# %%
_population_view = None
_pop_label = "full 756"

if POPULATION_MODE == "reduced":
    try:
        from declan.redundancy_resolved_v1_population import load_population_view
        _population_view = load_population_view(version_name=POPULATION_VERSION_NAME)
        _pop_label = f"V1-RR {_population_view.n_units}"
        print(f"Loaded population view: {_population_view.name}")
        print(f"  {_population_view.n_units} representatives from {_population_view.input_channels} channels")
        _mem = _population_view.membership
        _cluster_mem = _population_view.cluster_membership
        if _cluster_mem is None:
            _cluster_mem = _mem
        group_sizes = (_cluster_mem > 0).sum(axis=1)
        n_singletons = int((group_sizes == 1).sum())
        n_grouped = int((group_sizes > 1).sum())
        print(f"  singletons (size=1): {n_singletons}")
        print(f"  groups (size>1):     {n_grouped}")
        print(f"  max group size:      {int(group_sizes.max())}")
        print(f"  pooling row-sum range: [{float(_mem.sum(axis=1).min()):.3f}, {float(_mem.sum(axis=1).max()):.3f}]")
        print(f"  cluster row-sum range: [{float(_cluster_mem.sum(axis=1).min()):.3f}, {float(_cluster_mem.sum(axis=1).max()):.3f}]")
    except FileNotFoundError as exc:
        print(f"Reduced population spec not found: {exc}")
        print("Falling back to full population.")
        POPULATION_MODE = "full"
elif POPULATION_MODE == "full":
    print(f"Population mode: full 756 channels (no pooling applied)")

# %%
if POPULATION_MODE == "reduced" and _population_view is not None:
    _mem = _population_view.membership
    _cluster_mem = _population_view.cluster_membership
    if _cluster_mem is None:
        _cluster_mem = _mem
    group_sizes = (_cluster_mem > 0).sum(axis=1)
    fig_pop, axes_pop = plt.subplots(1, 2, figsize=(9, 3.5), dpi=140)
    axes_pop[0].hist(group_sizes, bins=range(1, int(group_sizes.max()) + 2), color="#4c78a8", alpha=0.85)
    axes_pop[0].set_xlabel("channels per representative")
    axes_pop[0].set_ylabel("count")
    axes_pop[0].set_title(f"Group size distribution ({_population_view.n_units} reps)")
    axes_pop[0].spines[["top", "right"]].set_visible(False)

    # Cluster-membership matrix sparsity — show which channels belong to each representative
    n_show = min(40, _population_view.n_units)
    axes_pop[1].imshow(_cluster_mem[:n_show], aspect="auto", cmap="Blues", interpolation="nearest")
    axes_pop[1].set_xlabel("input channel index (0–755)")
    axes_pop[1].set_ylabel("representative index")
    axes_pop[1].set_title(f"Cluster membership (first {n_show} reps)")

    fig_pop.suptitle(
        f"Redundancy-resolved population: {_population_view.name}",
        fontsize=9,
        y=1.04,
    )
    fig_pop.tight_layout()
    fig_pop
else:
    print("Full-population mode — membership plot skipped.")

# %% [markdown]
# ## Stimulus geometry
#
# The renderer draws a high-resolution world image and samples a model-retina
# movie from it. The Vernier stimulus is two bars separated by a vertical gap.
# The lower bar is shifted by `offset_arcmin`.
#
# Clarification needed: the code exposes the default dimensions, but this
# walkthrough does not infer why those exact numbers were chosen. Treat them as
# current analysis defaults unless a separate behavioral or stimulus-design
# note says otherwise.

# %%
geometry = RenderGeometry()
canonical_spec = VernierSpec(
    offset_arcmin=0.0,
    bar_width_arcmin=2.0,
    gap_arcmin=4.0,
    bar_length_arcmin=12.0,
    contrast=0.5,
    polarity="bright",
)

geometry_table = pd.DataFrame(
    [
        {
            "quantity": "world_ppd",
            "value": geometry.world_ppd,
            "meaning": "high-resolution drawing grid, pixels per degree",
        },
        {
            "quantity": "world_pixel_arcmin",
            "value": geometry.world_pixel_arcmin,
            "meaning": "arcmin per high-resolution world pixel",
        },
        {
            "quantity": "retina_ppd",
            "value": geometry.retina_ppd,
            "meaning": "model retina sampling grid, pixels per degree",
        },
        {
            "quantity": "model_pixel_arcmin",
            "value": geometry.model_pixel_arcmin,
            "meaning": "arcmin per sampled retina pixel",
        },
        {
            "quantity": "retina_size",
            "value": str(geometry.retina_size),
            "meaning": "height x width passed into the model stimulus path",
        },
    ]
)
show_table(geometry_table)

# %%
step = FD_STEP_TO_PLOT
frame_zero = central_retina_frame(canonical_spec, geometry)
frame_plus = central_retina_frame(canonical_spec.with_offset(+step), geometry)
frame_minus = central_retina_frame(canonical_spec.with_offset(-step), geometry)
frame_diff = frame_plus - frame_minus
frame_derivative = frame_diff / (2.0 * step)

fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=140)
images = [
    (frame_minus, f"-{step:g} arcmin", "gray"),
    (frame_zero, "0 arcmin", "gray"),
    (frame_plus, f"+{step:g} arcmin", "gray"),
    (frame_derivative, "finite-difference derivative", "coolwarm"),
]
for ax, (frame, title, cmap) in zip(axes, images, strict=True):
    if cmap == "coolwarm":
        vmax = float(np.nanmax(np.abs(frame))) or 1.0
        im = ax.imshow(frame, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(frame, cmap=cmap, vmin=0, vmax=geometry.max_raw)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
fig.suptitle("Rendered Vernier stimulus and local offset derivative", y=1.04)
fig

# %% [markdown]
# ### What you're seeing above
#
# The first three panels show the Vernier stimulus at three offsets: −δ (bottom
# bar shifted left), zero (bars aligned), and +δ (bottom bar shifted right). The
# offset values here are just `fd_step_arcmin` — small by design to stay in the
# "local" regime where the finite-difference approximation is valid.
#
# The rightmost panel is the **pixel-level derivative**: which retinal locations
# change intensity when the offset increases. Only a thin vertical strip near the
# bottom-bar edge is affected. Everything else is identical across the three
# stimuli.
#
# **Why this matters**: the discriminability signal lives at that thin edge.
# Eye movements that sweep that edge across different neuron positions sample the
# stimulus more thoroughly over time — but only if the observer can keep track of
# where the edge was. When pose is hidden, those movements also add variance.

# %%
audit = pixel_audit(
    canonical_spec,
    fd_steps_arcmin=FD_STEPS_FOR_PIXEL_AUDIT,
    geometry=geometry,
    device="cpu",
)
pixel_df = pd.DataFrame(audit["fd_rows"])
pixel_cols = [
    "fd_step_arcmin",
    "max_abs_plus_minus_diff",
    "l2_plus_minus_diff",
    "pixel_fisher_per_arcmin2_diag",
    "centroid_sensitivity_x_px_per_arcmin",
    "centroid_sensitivity_y_px_per_arcmin",
]
show_table(pixel_df[pixel_cols])

# %% [markdown]
# ## Synthetic Brownian FEM traces for demonstrations
#
# The tutorial should not lean on the real Vernier cache until the end. For the
# teaching plots, we use synthetic Brownian fixational traces whose scale is
# anchored to real **backimage** fixation windows.
#
# Assumption made here: the synthetic traces are a controlled demonstration of
# along/across diffusion, not a claim that biological drift is exactly Brownian.
# The Brownian choice is useful because its diffusion constants are explicit:
# scaling the increment variance on one axis changes that axis' diffusion while
# the other axis can be held at 1x.
#
# Unit convention for the synthetic generator:
#
# \[
# E[(\Delta x)^2 + (\Delta y)^2] = 2(D_{\mathrm{across}} + D_{\mathrm{along}})\Delta t.
# \]
#
# Intoy & Rucci (2020; Nat. Commun. 11:795, doi:10.1038/s41467-020-14616-2)
# report a scalar Brownian diffusion constant using
# \(\sigma^2(t) \propto 4Dt\), where \(\sigma^2(t)\) is total 2D displacement
# variance. To stay comparable, we use the existing backimage
# `diffusion_constant_deg2_s` column, which is computed as the slope of the MSD
# curve divided by 4. We do **not** use lag-1 MSD as the scale; lag-1 steps can
# be dominated by sample-scale noise or short-lag correlations and are not the
# paper's estimator.
#
# The backimage summary does not store axiswise diffusion slopes. For the
# synthetic sweep, we therefore define 1x as an equal-axis Brownian baseline:
# D_across = D_along = D_scalar. This makes the across and along curves meet at
# a true equal-diffusion 1x condition. The observed covariance fractions are
# retained only as diagnostics, not as the generator's 1x diffusion scale.

# %%
backimage_windows = read_csv_optional(BACKIMAGE_FIXATION_WINDOWS_PATH)
backimage_numeric_cols = [
    "rms_radius_deg",
    "path_length_deg",
    "duration_s",
    "anisotropy",
    "speed_p95_deg_s",
    "max_radius_deg",
    "cov_xx_deg2",
    "cov_yy_deg2",
    "msd_lag1_deg2",
    "n_samples",
    "diffusion_constant_deg2_s",
]
for col in backimage_numeric_cols:
    if col in backimage_windows:
        backimage_windows[col] = pd.to_numeric(backimage_windows[col], errors="coerce")

if backimage_windows.empty:
    reference_windows = pd.DataFrame()
else:
    reference_mask = np.ones(len(backimage_windows), dtype=bool)
    if "image_feature_ok" in backimage_windows:
        reference_mask &= backimage_windows["image_feature_ok"].fillna(False).astype(bool).to_numpy()
    if "rms_radius_deg" in backimage_windows:
        reference_mask &= (backimage_windows["rms_radius_deg"] <= SYNTHETIC_REFERENCE_MAX_RMS_DEG).fillna(False).to_numpy()
    if "speed_p95_deg_s" in backimage_windows:
        reference_mask &= (backimage_windows["speed_p95_deg_s"] <= SYNTHETIC_REFERENCE_MAX_SPEED_P95_DPS).fillna(False).to_numpy()
    reference_windows = backimage_windows.loc[reference_mask].copy()


def estimate_axis_diffusion_constants_arcmin2_s(windows: pd.DataFrame) -> dict[str, float]:
    """Estimate Brownian axis diffusion constants from slope-based backimage D.

    Intoy & Rucci (2020) use sigma^2(t) ~= 4Dt for total 2D displacement
    variance. The reviewed backimage table already stores that scalar D in
    deg^2/s via an MSD-slope fit. We convert it to arcmin^2/s and use positive
    slope rows for a paper-comparable Brownian scale. The synthetic 1x
    baseline sets both axes equal to that scalar D; the covariance split is
    returned only as a diagnostic.
    """
    required = ["diffusion_constant_deg2_s", "cov_xx_deg2", "cov_yy_deg2"]
    empty_result = {
        "diffusion_estimate_rows": 0.0,
        "positive_diffusion_rows": 0.0,
        "sample_interval_s": float("nan"),
        "lag1_implied_scalar_D_arcmin2_s": float("nan"),
        "scalar_D_all_mean_arcmin2_s": float("nan"),
        "scalar_D_positive_mean_arcmin2_s": float("nan"),
        "scalar_D_positive_median_arcmin2_s": float("nan"),
        "D_across_arcmin2_s": float("nan"),
        "D_along_arcmin2_s": float("nan"),
        "D_across_covariance_split_arcmin2_s": float("nan"),
        "D_along_covariance_split_arcmin2_s": float("nan"),
        "axis_fraction_across": float("nan"),
        "axis_fraction_along": float("nan"),
    }
    if windows.empty or any(col not in windows for col in required):
        return empty_result

    optional = [col for col in ["msd_lag1_deg2", "duration_s", "n_samples"] if col in windows]
    work = windows[[*required, *optional]].apply(pd.to_numeric, errors="coerce")
    scalar_d_arcmin2_s = work["diffusion_constant_deg2_s"] * 60.0**2
    var_sum = work["cov_xx_deg2"] + work["cov_yy_deg2"]
    valid_finite = (
        np.isfinite(scalar_d_arcmin2_s)
        & np.isfinite(work["cov_xx_deg2"])
        & np.isfinite(work["cov_yy_deg2"])
        & (var_sum > 0.0)
    )
    if not bool(valid_finite.any()):
        return empty_result

    valid_positive = valid_finite & (scalar_d_arcmin2_s > 0.0)
    rows_for_scale = valid_positive if bool(valid_positive.any()) else valid_finite
    valid_work = work.loc[rows_for_scale].copy()
    valid_var_sum = var_sum.loc[rows_for_scale].astype(float)
    valid_scalar_d = scalar_d_arcmin2_s.loc[rows_for_scale].astype(float)
    frac_across = valid_work["cov_xx_deg2"].astype(float) / valid_var_sum
    frac_along = valid_work["cov_yy_deg2"].astype(float) / valid_var_sum
    scalar_d_reference_arcmin2_s = float(np.nanmean(valid_scalar_d))
    d_axis_sum_arcmin2_s = 2.0 * valid_scalar_d
    d_across_covariance_split_arcmin2_s = d_axis_sum_arcmin2_s * frac_across
    d_along_covariance_split_arcmin2_s = d_axis_sum_arcmin2_s * frac_along

    if all(col in work for col in ["msd_lag1_deg2", "duration_s", "n_samples"]):
        dt_s = work["duration_s"] / work["n_samples"]
        lag1_ok = np.isfinite(work["msd_lag1_deg2"]) & np.isfinite(dt_s) & (work["msd_lag1_deg2"] > 0.0) & (dt_s > 0.0)
        lag1_implied_scalar_d = (work.loc[lag1_ok, "msd_lag1_deg2"] * 60.0**2) / (4.0 * dt_s.loc[lag1_ok])
        lag1_implied_scalar_d_value = float(np.nanmedian(lag1_implied_scalar_d)) if bool(lag1_ok.any()) else float("nan")
        sample_interval_s = float(np.nanmedian(dt_s.loc[np.isfinite(dt_s) & (dt_s > 0.0)]))
    else:
        lag1_implied_scalar_d_value = float("nan")
        sample_interval_s = float("nan")

    return {
        "diffusion_estimate_rows": float(valid_finite.sum()),
        "positive_diffusion_rows": float(valid_positive.sum()),
        "sample_interval_s": sample_interval_s,
        "lag1_implied_scalar_D_arcmin2_s": lag1_implied_scalar_d_value,
        "scalar_D_all_mean_arcmin2_s": float(np.nanmean(scalar_d_arcmin2_s.loc[valid_finite])),
        "scalar_D_positive_mean_arcmin2_s": float(np.nanmean(scalar_d_arcmin2_s.loc[valid_positive])) if bool(valid_positive.any()) else float("nan"),
        "scalar_D_positive_median_arcmin2_s": float(np.nanmedian(scalar_d_arcmin2_s.loc[valid_positive])) if bool(valid_positive.any()) else float("nan"),
        "D_across_arcmin2_s": scalar_d_reference_arcmin2_s,
        "D_along_arcmin2_s": scalar_d_reference_arcmin2_s,
        "D_across_covariance_split_arcmin2_s": float(np.nanmean(d_across_covariance_split_arcmin2_s)),
        "D_along_covariance_split_arcmin2_s": float(np.nanmean(d_along_covariance_split_arcmin2_s)),
        "axis_fraction_across": float(np.nanmean(frac_across)),
        "axis_fraction_along": float(np.nanmean(frac_along)),
    }


if reference_windows.empty:
    target_x_std_deg = 0.03
    target_y_std_deg = 0.03
    reference_duration_s = DEMO_MAX_FRAMES / 120.0
    reference_source = "fallback defaults; backimage fixation-window file unavailable or filtered empty"
else:
    q = float(SYNTHETIC_REFERENCE_RMS_QUANTILE)
    target_x_std_deg = float(np.sqrt(np.nanquantile(reference_windows["cov_xx_deg2"], q)))
    target_y_std_deg = float(np.sqrt(np.nanquantile(reference_windows["cov_yy_deg2"], q)))
    reference_duration_s = float(np.nanmedian(reference_windows["duration_s"]))
    reference_source = str(BACKIMAGE_FIXATION_WINDOWS_PATH)

axis_diffusion_estimate = estimate_axis_diffusion_constants_arcmin2_s(reference_windows)
reference_d_across_arcmin2_s = axis_diffusion_estimate["D_across_arcmin2_s"]
reference_d_along_arcmin2_s = axis_diffusion_estimate["D_along_arcmin2_s"]

backimage_reference_table = pd.DataFrame(
    [
        {
            "quantity": "source rows",
            "value": len(backimage_windows),
            "meaning": "all reviewed backimage fixation windows",
        },
        {
            "quantity": "reference rows",
            "value": len(reference_windows),
            "meaning": "after image-ok, RMS, and speed filters",
        },
        {
            "quantity": "diffusion estimate rows",
            "value": int(axis_diffusion_estimate["diffusion_estimate_rows"]),
            "meaning": "reference rows with finite slope-based D and xy covariance",
        },
        {
            "quantity": "positive diffusion rows",
            "value": int(axis_diffusion_estimate["positive_diffusion_rows"]),
            "meaning": "rows with positive MSD-slope D; used for the Brownian 1x scale",
        },
        {
            "quantity": "target across std",
            "value": target_x_std_deg * 60.0,
            "meaning": "arcmin; horizontal axis for vertical Vernier",
        },
        {
            "quantity": "target along std",
            "value": target_y_std_deg * 60.0,
            "meaning": "arcmin; vertical axis for vertical Vernier",
        },
        {
            "quantity": "reference duration",
            "value": reference_duration_s,
            "meaning": "seconds per backimage fixation window",
        },
        {
            "quantity": "sample interval",
            "value": axis_diffusion_estimate["sample_interval_s"],
            "meaning": "seconds; median duration_s / n_samples",
        },
        {
            "quantity": "scalar D, all slope rows",
            "value": axis_diffusion_estimate["scalar_D_all_mean_arcmin2_s"],
            "meaning": "arcmin^2/s; mean of backimage diffusion_constant_deg2_s, including clipped zeros",
        },
        {
            "quantity": "scalar D, positive slope rows",
            "value": axis_diffusion_estimate["scalar_D_positive_mean_arcmin2_s"],
            "meaning": "arcmin^2/s; mean paper-style D used for the synthetic 1x scale",
        },
        {
            "quantity": "scalar D, positive median",
            "value": axis_diffusion_estimate["scalar_D_positive_median_arcmin2_s"],
            "meaning": "arcmin^2/s; diagnostic showing the window-level D distribution is skewed",
        },
        {
            "quantity": "1x D across",
            "value": reference_d_across_arcmin2_s,
            "meaning": "arcmin^2/s; equal-axis 1x baseline set to scalar D",
        },
        {
            "quantity": "1x D along",
            "value": reference_d_along_arcmin2_s,
            "meaning": "arcmin^2/s; equal-axis 1x baseline set to scalar D",
        },
        {
            "quantity": "covariance-split D across",
            "value": axis_diffusion_estimate["D_across_covariance_split_arcmin2_s"],
            "meaning": "arcmin^2/s; diagnostic only, not used for synthetic 1x",
        },
        {
            "quantity": "covariance-split D along",
            "value": axis_diffusion_estimate["D_along_covariance_split_arcmin2_s"],
            "meaning": "arcmin^2/s; diagnostic only, not used for synthetic 1x",
        },
        {
            "quantity": "lag-1 implied scalar D",
            "value": axis_diffusion_estimate["lag1_implied_scalar_D_arcmin2_s"],
            "meaning": "arcmin^2/s; diagnostic only, not used because it is not the Intoy/Rucci estimator",
        },
        {
            "quantity": "Intoy/Rucci sustained fixation D",
            "value": f"{INTOY_RUCCI_2020_SUSTAINED_FIXATION_D_ARCMIN2_S} ± {INTOY_RUCCI_2020_SUSTAINED_FIXATION_D_SEM_ARCMIN2_S}",
            "meaning": "arcmin^2/s; paper-reported mean ± SEM across observers",
        },
        {
            "quantity": "Intoy/Rucci free-viewing D",
            "value": f"{INTOY_RUCCI_2020_FREEVIEWING_D_ARCMIN2_S} ± {INTOY_RUCCI_2020_FREEVIEWING_D_SEM_ARCMIN2_S}",
            "meaning": "arcmin^2/s; paper-reported mean ± SEM across observers",
        },
        {
            "quantity": "1x equality convention",
            "value": "D_across = D_along = D_scalar",
            "meaning": "keeps the diffusion sweep anchored to a true equal-axis 1x point",
        },
        {
            "quantity": "source",
            "value": reference_source,
            "meaning": "where the synthetic Brownian scale came from",
        },
    ]
)
show_table(backimage_reference_table)

# %%
condition_explanations = pd.DataFrame(
    [
        {
            "condition": "static_center",
            "what_changes": "All time bins use nominal zero phase.",
            "what_it_controls": "No retinal motion.",
        },
        {
            "condition": "brownian_iso_1x",
            "what_changes": "Synthetic Brownian x and y increments use the same backimage-derived scalar D at 1x.",
            "what_it_controls": "Equal-axis baseline synthetic drift before varying across or along diffusion.",
        },
        {
            "condition": "brownian_across_sx",
            "what_changes": "Horizontal/across Brownian diffusion is scaled by s; vertical/along stays 1x.",
            "what_it_controls": "Whether reducing across-contour diffusion protects Vernier-relevant information.",
        },
        {
            "condition": "brownian_along_sx",
            "what_changes": "Vertical/along Brownian diffusion is scaled by s; horizontal/across stays 1x.",
            "what_it_controls": "Whether changing along-contour diffusion has the same effect as across-contour diffusion.",
        },
        {
            "condition": "brownian_phase_cloud",
            "what_changes": "Each frame draws an iid retinal position from the pooled Brownian 1x position cloud.",
            "what_it_controls": "Spatial phase distribution without trace continuity or trace identity.",
        },
        {
            "condition": "brownian_order_shuffled",
            "what_changes": "Each Brownian 1x trace keeps the same positions but permutes their temporal order.",
            "what_it_controls": "Position cloud matched within trace, temporal order removed.",
        },
    ]
)
show_table(condition_explanations)

# %%
def trace_rms_deg(trace: np.ndarray) -> float:
    centered = np.asarray(trace, dtype=np.float64) - np.mean(trace, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def trace_path_length_deg(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trace, axis=0), axis=1)))


def centered_brownian_trace(n_frames: int, rng: np.random.Generator) -> np.ndarray:
    increments = rng.normal(size=(int(n_frames), 2))
    trace = np.cumsum(increments, axis=0)
    trace -= np.mean(trace, axis=0, keepdims=True)
    return trace.astype(np.float32)


def scale_axis_to_std(values: np.ndarray, target_std: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - float(np.mean(values))
    if float(target_std) <= 0.0:
        return np.zeros_like(centered, dtype=np.float32)
    current = float(np.std(centered))
    if current <= 1e-12:
        return np.zeros_like(centered, dtype=np.float32)
    return (centered * (float(target_std) / current)).astype(np.float32)


def anisotropic_brownian_trace(
    base_unit_trace: np.ndarray,
    *,
    d_across: float,
    d_along: float,
    target_across_std_deg: float,
    target_along_std_deg: float,
) -> np.ndarray:
    """Scale a centered Brownian path to requested along/across diffusion constants.

    For a Brownian process, multiplying position amplitude by sqrt(D) corresponds
    to multiplying increment variance, and therefore diffusion constant, by D.
    """
    out = np.zeros_like(base_unit_trace, dtype=np.float32)
    out[:, 0] = scale_axis_to_std(
        base_unit_trace[:, 0],
        float(target_across_std_deg) * math.sqrt(max(float(d_across), 0.0)),
    )
    out[:, 1] = scale_axis_to_std(
        base_unit_trace[:, 1],
        float(target_along_std_deg) * math.sqrt(max(float(d_along), 0.0)),
    )
    return out.astype(np.float32)


def diffusion_condition_name(axis: str, scale: float) -> str:
    if axis == "iso":
        return "brownian_iso_1x"
    text = f"{float(scale):g}".replace(".", "p")
    return f"brownian_{axis}_{text}x"


def shuffled_trace(trace: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return the same retinal positions in a random temporal order."""
    idx = np.arange(trace.shape[0])
    rng.shuffle(idx)
    out = np.asarray(trace, dtype=np.float32)[idx].copy()
    out -= np.mean(out, axis=0, keepdims=True)
    return out.astype(np.float32)


def phase_cloud_trace(source_traces: list[np.ndarray], n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Draw iid retinal positions from the pooled synthetic phase cloud."""
    cloud = np.concatenate([np.asarray(trace, dtype=np.float32) for trace in source_traces], axis=0)
    idx = rng.integers(0, cloud.shape[0], size=int(n_frames))
    out = cloud[idx].astype(np.float32, copy=True)
    out -= np.mean(out, axis=0, keepdims=True)
    return out


rng = np.random.default_rng(SEED)
synthetic_unit_traces = [
    centered_brownian_trace(DEMO_MAX_FRAMES, rng)
    for _ in range(int(SYNTHETIC_TRACE_COUNT))
]

synthetic_specs: list[dict[str, Any]] = [
    {
        "condition": "static_center",
        "d_across": 0.0,
        "d_along": 0.0,
        "curve": "static",
        "varied_axis": "none",
        "diffusion_constant": 0.0,
    },
    {
        "condition": "brownian_iso_1x",
        "d_across": 1.0,
        "d_along": 1.0,
        "curve": "both axes 1x",
        "varied_axis": "none",
        "diffusion_constant": 1.0,
    },
    {
        "condition": "brownian_phase_cloud",
        "d_across": 1.0,
        "d_along": 1.0,
        "curve": "trajectory controls",
        "varied_axis": "phase cloud",
        "diffusion_constant": 1.0,
    },
    {
        "condition": "brownian_order_shuffled",
        "d_across": 1.0,
        "d_along": 1.0,
        "curve": "trajectory controls",
        "varied_axis": "temporal order",
        "diffusion_constant": 1.0,
    },
]
for scale in SYNTHETIC_DIFFUSION_SCALES:
    synthetic_specs.append(
        {
            "condition": diffusion_condition_name("across", scale),
            "d_across": float(scale),
            "d_along": 1.0,
            "curve": "vary across; along held 1x",
            "varied_axis": "across contour",
            "diffusion_constant": float(scale),
        }
    )
    synthetic_specs.append(
        {
            "condition": diffusion_condition_name("along", scale),
            "d_across": 1.0,
            "d_along": float(scale),
            "curve": "vary along; across held 1x",
            "varied_axis": "along contour",
            "diffusion_constant": float(scale),
        }
    )

for spec_row in synthetic_specs:
    spec_row["D_across_arcmin2_s"] = reference_d_across_arcmin2_s * float(spec_row["d_across"])
    spec_row["D_along_arcmin2_s"] = reference_d_along_arcmin2_s * float(spec_row["d_along"])

synthetic_diffusion_constants_df = pd.DataFrame(synthetic_specs).rename(
    columns={"d_across": "D_across", "d_along": "D_along"}
)
show_table(
    synthetic_diffusion_constants_df[
        [
            "condition",
            "curve",
            "varied_axis",
            "D_across",
            "D_along",
            "D_across_arcmin2_s",
            "D_along_arcmin2_s",
        ]
    ],
    n=32,
)

# %% [markdown]
# The `arcmin^2/s` columns are the reported Brownian diffusion constants implied
# by the backimage scalar D and the relative scale factors. The 1x point is
# equal across axes by construction. The trace plots still use the empirical
# backimage fixation-cloud scale for visual anchoring. For `phase_cloud` and
# `order_shuffled`, the position scale is 1x, but temporal continuity is
# intentionally broken, so do not read their path lengths or jumps as biological
# diffusion.

# %%
synthetic_trace_bank: dict[str, list[np.ndarray]] = {}
for spec_row in synthetic_specs:
    condition = str(spec_row["condition"])
    if condition == "static_center":
        traces = [np.zeros((DEMO_MAX_FRAMES, 2), dtype=np.float32) for _ in synthetic_unit_traces]
    elif condition == "brownian_phase_cloud":
        iso_traces = [
            anisotropic_brownian_trace(
                unit_trace,
                d_across=1.0,
                d_along=1.0,
                target_across_std_deg=target_x_std_deg,
                target_along_std_deg=target_y_std_deg,
            )
            for unit_trace in synthetic_unit_traces
        ]
        traces = [phase_cloud_trace(iso_traces, DEMO_MAX_FRAMES, rng) for _ in synthetic_unit_traces]
    elif condition == "brownian_order_shuffled":
        iso_traces = [
            anisotropic_brownian_trace(
                unit_trace,
                d_across=1.0,
                d_along=1.0,
                target_across_std_deg=target_x_std_deg,
                target_along_std_deg=target_y_std_deg,
            )
            for unit_trace in synthetic_unit_traces
        ]
        traces = [shuffled_trace(trace, rng) for trace in iso_traces]
    else:
        traces = [
            anisotropic_brownian_trace(
                unit_trace,
                d_across=float(spec_row["d_across"]),
                d_along=float(spec_row["d_along"]),
                target_across_std_deg=target_x_std_deg,
                target_along_std_deg=target_y_std_deg,
            )
            for unit_trace in synthetic_unit_traces
        ]
    synthetic_trace_bank[condition] = traces

demo_conditions = [
    "static_center",
    "brownian_iso_1x",
    "brownian_phase_cloud",
    "brownian_order_shuffled",
    "brownian_across_0x",
    "brownian_across_0p25x",
    "brownian_across_0p5x",
    "brownian_across_2x",
    "brownian_along_0x",
    "brownian_along_0p25x",
    "brownian_along_0p5x",
    "brownian_along_2x",
]
demo_conditions = [cond for cond in demo_conditions if cond in synthetic_trace_bank]
conditioned: dict[str, np.ndarray] = {condition: synthetic_trace_bank[condition][0] for condition in demo_conditions}

synthetic_trace_array = np.stack(synthetic_trace_bank["brownian_iso_1x"], axis=0)
trace_set = TraceSet(
    traces=synthetic_trace_array.astype(np.float32),
    durations=np.full(synthetic_trace_array.shape[0], synthetic_trace_array.shape[1], dtype=np.int32),
)
trace_set_all = trace_set
base_trace = valid_trace(trace_set, 0, max_frames=DEMO_MAX_FRAMES)

condition_meta: list[dict[str, Any]] = []
spec_lookup = {row["condition"]: row for row in synthetic_specs}
for condition in demo_conditions:
    trace = conditioned[condition]
    spec_row = spec_lookup[condition]
    condition_meta.append(
        {
            "condition": condition,
            "D_across": float(spec_row["d_across"]),
            "D_along": float(spec_row["d_along"]),
            "D_across_arcmin2_s": float(spec_row["D_across_arcmin2_s"]),
            "D_along_arcmin2_s": float(spec_row["D_along_arcmin2_s"]),
            "x_std_arcmin": float(np.std(trace[:, 0]) * 60.0),
            "y_std_arcmin": float(np.std(trace[:, 1]) * 60.0),
            "rms_radius_arcmin": trace_rms_deg(trace) * 60.0,
            "path_length_arcmin": trace_path_length_deg(trace) * 60.0,
        }
    )

print(f"Synthetic trace bank: {len(synthetic_unit_traces)} paired traces x {DEMO_MAX_FRAMES} frames")
show_table(pd.DataFrame(condition_meta))

# %% [markdown]
# `brownian_phase_cloud` and `brownian_order_shuffled` can have very large
# apparent path lengths because they intentionally break temporal continuity.
# Read their path length as a control diagnostic, not as a plausible biological
# drift trajectory.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
time_ms = np.arange(base_trace.shape[0]) / 120.0 * 1000.0
for condition, trace in conditioned.items():
    label = condition_label(condition)
    color = COLORS.get(condition)
    axes[0].plot(time_ms, trace[:, 0] * 60.0, label=label, color=color, linewidth=1.2)
    axes[1].plot(trace[:, 0] * 60.0, trace[:, 1] * 60.0, label=label, color=color, linewidth=1.2)
axes[0].set_xlabel("time (ms)")
axes[0].set_ylabel("x phase (arcmin)")
axes[0].set_title("Across-contour retinal phase over time")
axes[1].set_xlabel("x phase (arcmin)")
axes[1].set_ylabel("y phase (arcmin)")
axes[1].set_title("Synthetic Brownian trajectories")
axes[1].axis("equal")
axes[0].spines[["top", "right"]].set_visible(False)
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig

# %% [markdown]
# ### What the conditions control
#
# Each condition applies a different anisotropic scaling to the same underlying
# Brownian innovations. Comparisons are interpretable because the random walk is
# paired: changing `D_across` or `D_along` changes the diffusion constant, not
# the identity of the random trace.
#
# The most important contrast pairs are:
#
# | Pair | What it isolates |
# |---|---|
# | Brownian iso 1x vs static center | Any effect of synthetic drift at all |
# | Brownian phase cloud vs Brownian iso 1x | Same broad phase distribution without trace continuity |
# | Brownian order shuffled vs Brownian iso 1x | Same positions within each trace, temporal order removed |
# | across sweep, along held 1x | Whether shrinking across-contour diffusion protects Vernier information |
# | along sweep, across held 1x | Whether along-contour diffusion has the same cost |
# | Fisher vs general SSI | Whether spatial organization tracks task-specific discriminability |

# %% [markdown]
# ## Synthetic Fisher and general SSI side by side
#
# Before touching cached twin outputs, we can compute the two quantities on the
# rendered stimulus movie itself. This is a **teaching proxy**, not the V1 model:
#
# - Known-eye pixel Fisher asks how much the retinal image changes with Vernier
#   offset when the trace is supplied.
# - Pixel SSI asks how spatially concentrated the retinal movie is.
#
# The useful lesson is the relationship, not the absolute units. A condition can
# create spatially structured images without changing the Vernier-aligned Fisher
# signal in the same way. We postpone hidden-pose nuisance until the later
# **Known-trace versus hidden-trace** cell.

# %%
show_markdown(
    r"""
### Stimulus-level proxy equations

For each synthetic trace \(\tau(t)\), render two movies:

$$
R_+(t,x,y) = R(\theta_0+\delta,\tau(t)), \quad
R_-(t,x,y) = R(\theta_0-\delta,\tau(t)).
$$

The local offset derivative is:

$$
\partial_\theta R(t,x,y)
\approx
\frac{R_+(t,x,y)-R_-(t,x,y)}{2\delta}.
$$

The known-trace stimulus-level Fisher proxy is diagonal and Poisson-like:

$$
J_{\mathrm{pixel}}(t)
=
\sum_{x,y}
\frac{\left[\partial_\theta R(t,x,y)\right]^2}
{\bar R(t,x,y)+\epsilon}.
$$

The general SSI proxy is computed from the same mean movie
\(\bar R=(R_+ + R_-)/2\), but ignores the sign of the Vernier offset.
"""
)

# %%
def spatial_ssi_single_frame_np(rate_maps: np.ndarray, eps: float = 1e-8) -> dict[str, np.ndarray | float]:
    """Spatial SSI for one frame, matching the repository's population formula."""
    y = np.asarray(rate_maps, dtype=np.float64)
    if y.ndim == 2:
        y = y[None, :, :]
    if y.ndim != 3:
        raise ValueError(f"Expected (unit, H, W) or (H, W), got {y.shape}")
    y = np.maximum(y, 0.0)
    flat = y.reshape(y.shape[0], -1)
    rbar = flat.mean(axis=1)
    gain = flat / (rbar[:, None] + eps)
    unit_bits = np.mean(gain * np.log2(gain + eps), axis=1)
    weights = rbar / max(float(np.sum(rbar)), eps)
    return {
        "unit_bits_per_spike": unit_bits,
        "unit_mean_rate": rbar,
        "population_bits_per_spike": float(np.sum(weights * unit_bits)),
        "population_bits_per_frame_proxy": float(np.sum(rbar * unit_bits)),
    }


def spatial_ssi_timecourse_np(rate_movie: np.ndarray, eps: float = 1e-8) -> dict[str, np.ndarray]:
    """Spatial SSI timecourse for (T, H, W) or (T, unit, H, W)."""
    y = np.asarray(rate_movie, dtype=np.float64)
    if y.ndim == 3:
        y = y[:, None, :, :]
    if y.ndim != 4:
        raise ValueError(f"Expected (T, H, W) or (T, unit, H, W), got {y.shape}")
    bits_per_spike = []
    bits_per_frame_proxy = []
    for t in range(y.shape[0]):
        frame = spatial_ssi_single_frame_np(y[t], eps=eps)
        bits_per_spike.append(float(frame["population_bits_per_spike"]))
        bits_per_frame_proxy.append(float(frame["population_bits_per_frame_proxy"]))
    return {
        "bits_per_spike": np.asarray(bits_per_spike, dtype=np.float64),
        "bits_per_frame_proxy": np.asarray(bits_per_frame_proxy, dtype=np.float64),
        "cumulative_bits_proxy": np.cumsum(np.asarray(bits_per_frame_proxy, dtype=np.float64)),
    }


def retinal_movie_np(world_image: Any, trace_deg: np.ndarray, *, geometry: RenderGeometry) -> np.ndarray:
    movie = sample_retina_movie(world_image, np.asarray(trace_deg, dtype=np.float32), geometry=geometry, device="cpu")
    return movie[0, 0].detach().cpu().numpy().astype(np.float32)


def stimulus_plus_minus_movies(
    trace_deg: np.ndarray,
    *,
    fd_step: float,
    spec: VernierSpec,
    geometry: RenderGeometry,
    plus_world: Any | None = None,
    minus_world: Any | None = None,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if plus_world is None:
        plus_world = render_world(spec.with_offset(+float(fd_step)), geometry=geometry, device="cpu")
    if minus_world is None:
        minus_world = render_world(spec.with_offset(-float(fd_step)), geometry=geometry, device="cpu")
    plus = retinal_movie_np(plus_world, trace_deg, geometry=geometry) / float(geometry.max_raw)
    minus = retinal_movie_np(minus_world, trace_deg, geometry=geometry) / float(geometry.max_raw)
    return plus, minus


def stimulus_fisher_ssi_curves_from_movies(
    plus: np.ndarray,
    minus: np.ndarray,
    *,
    fd_step: float,
    epsilon: float = 1e-6,
) -> dict[str, np.ndarray]:
    mean_movie = np.maximum((plus + minus) / 2.0, 0.0)
    derivative = (plus - minus) / (2.0 * float(fd_step))
    diag = np.maximum(mean_movie, float(epsilon))
    fisher_per_frame = np.sum((derivative * derivative) / diag, axis=(1, 2))
    dprime2_per_frame = np.sum(((plus - minus) * (plus - minus)) / diag, axis=(1, 2))
    ssi = spatial_ssi_timecourse_np(mean_movie, eps=epsilon)
    return {
        "fisher_per_frame": fisher_per_frame,
        "cumulative_fisher": np.cumsum(fisher_per_frame),
        "dprime2_per_frame": dprime2_per_frame,
        "cumulative_dprime2": np.cumsum(dprime2_per_frame),
        "ssi_bits_per_spike": ssi["bits_per_spike"],
        "ssi_bits_per_frame_proxy": ssi["bits_per_frame_proxy"],
        "cumulative_ssi_bits_proxy": ssi["cumulative_bits_proxy"],
    }


def stimulus_fisher_ssi_curves(
    trace_deg: np.ndarray,
    *,
    fd_step: float,
    spec: VernierSpec,
    geometry: RenderGeometry,
    plus_world: Any | None = None,
    minus_world: Any | None = None,
    epsilon: float = 1e-6,
) -> dict[str, np.ndarray]:
    plus, minus = stimulus_plus_minus_movies(
        trace_deg,
        fd_step=fd_step,
        spec=spec,
        geometry=geometry,
        plus_world=plus_world,
        minus_world=minus_world,
    )
    return stimulus_fisher_ssi_curves_from_movies(plus, minus, fd_step=fd_step, epsilon=epsilon)


def summarize_synthetic_condition_metrics(
    traces_by_condition: dict[str, list[np.ndarray]],
    specs: list[dict[str, Any]],
    *,
    fd_step: float,
    spec: VernierSpec,
    geometry: RenderGeometry,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    spec_lookup = {row["condition"]: row for row in specs}
    plus_world = render_world(spec.with_offset(+float(fd_step)), geometry=geometry, device="cpu")
    minus_world = render_world(spec.with_offset(-float(fd_step)), geometry=geometry, device="cpu")
    for condition, traces in traces_by_condition.items():
        plus_minus = [
            stimulus_plus_minus_movies(
                trace,
                fd_step=fd_step,
                spec=spec,
                geometry=geometry,
                plus_world=plus_world,
                minus_world=minus_world,
            )
            for trace in traces
        ]
        per_trace = [
            stimulus_fisher_ssi_curves_from_movies(plus, minus, fd_step=fd_step)
            for plus, minus in plus_minus
        ]
        mean_curves = {
            key: np.mean(np.stack([item[key] for item in per_trace], axis=0), axis=0)
            for key in per_trace[0]
        }
        plus_trials = [plus.reshape(plus.shape[0], -1) for plus, _minus in plus_minus]
        minus_trials = [minus.reshape(minus.shape[0], -1) for _plus, minus in plus_minus]
        pose_hidden = pose_blind_diagonal_fisher(
            plus_trials,
            minus_trials,
            step_arcmin=float(fd_step),
            bin_seconds=1.0,
            epsilon=1e-6,
        )
        mean_curves["pose_hidden_cumulative_fisher"] = np.asarray(pose_hidden["cumulative_fisher"], dtype=np.float64)
        mean_curves["pose_hidden_fisher_per_frame"] = np.asarray(pose_hidden["fisher_per_bin"], dtype=np.float64)
        mean_curves["pose_hidden_cumulative_dprime2"] = np.asarray(pose_hidden["cumulative_dprime2"], dtype=np.float64)
        curves[condition] = mean_curves
        row_spec = spec_lookup.get(condition, {})
        rows.append(
            {
                "condition": condition,
                "curve": row_spec.get("curve", ""),
                "varied_axis": row_spec.get("varied_axis", ""),
                "diffusion_constant": row_spec.get("diffusion_constant", np.nan),
                "D_across": row_spec.get("d_across", np.nan),
                "D_along": row_spec.get("d_along", np.nan),
                "D_across_arcmin2_s": row_spec.get("D_across_arcmin2_s", np.nan),
                "D_along_arcmin2_s": row_spec.get("D_along_arcmin2_s", np.nan),
                "final_pixel_fisher": float(mean_curves["cumulative_fisher"][-1]),
                "final_pose_hidden_pixel_fisher": float(mean_curves["pose_hidden_cumulative_fisher"][-1]),
                "final_pixel_dprime2": float(mean_curves["cumulative_dprime2"][-1]),
                "hidden_to_known_fisher_ratio": float(
                    mean_curves["pose_hidden_cumulative_fisher"][-1]
                    / max(float(mean_curves["cumulative_fisher"][-1]), 1e-12)
                ),
                "mean_general_ssi_bits_per_spike": float(np.mean(mean_curves["ssi_bits_per_spike"])),
                "final_general_ssi_bits_proxy": float(mean_curves["cumulative_ssi_bits_proxy"][-1]),
            }
        )
    return pd.DataFrame(rows), curves


synthetic_metric_df, synthetic_metric_curves = summarize_synthetic_condition_metrics(
    synthetic_trace_bank,
    synthetic_specs,
    fd_step=FD_STEP_TO_PLOT,
    spec=canonical_spec,
    geometry=geometry,
)

synthetic_metric_display_df = synthetic_metric_df[
    [
        "condition",
        "D_across",
        "D_along",
        "D_across_arcmin2_s",
        "D_along_arcmin2_s",
        "final_pixel_fisher",
        "final_pose_hidden_pixel_fisher",
        "final_pixel_dprime2",
        "mean_general_ssi_bits_per_spike",
        "final_general_ssi_bits_proxy",
    ]
].rename(
    columns={
        "final_pixel_fisher": "known_trace_pixel_fisher",
        "final_pose_hidden_pixel_fisher": "hidden_trace_pixel_fisher",
        "final_pixel_dprime2": "known_trace_pixel_dprime2",
    }
)
show_table(
    synthetic_metric_display_df.sort_values(["D_across", "D_along", "condition"]),
    n=24,
)

# %%
def normalize_to_condition(df: pd.DataFrame, column: str, baseline_condition: str = "brownian_iso_1x") -> pd.Series:
    baseline_rows = df.loc[df["condition"] == baseline_condition, column]
    baseline = float(baseline_rows.iloc[0]) if not baseline_rows.empty else float("nan")
    if not np.isfinite(baseline) or abs(baseline) <= 1e-12:
        return pd.Series(np.nan, index=df.index)
    return df[column] / baseline


plot_metric_df = synthetic_metric_df.copy()
plot_metric_df["pixel_fisher_vs_iso"] = normalize_to_condition(plot_metric_df, "final_pixel_fisher")
plot_metric_df["pose_hidden_pixel_fisher_vs_iso"] = normalize_to_condition(plot_metric_df, "final_pose_hidden_pixel_fisher")
plot_metric_df["ssi_bits_per_spike_vs_iso"] = normalize_to_condition(plot_metric_df, "mean_general_ssi_bits_per_spike")
plot_metric_df["ssi_bits_proxy_vs_iso"] = normalize_to_condition(plot_metric_df, "final_general_ssi_bits_proxy")

# %% [markdown]
# ### Early condition-by-condition comparison
#
# This is the table to linger on during the first explanation. For each motion
# condition, compare two normalized columns:
#
# - `known_trace_fisher_vs_iso`: task-aligned Vernier signal when the trace is
#   supplied.
# - `general_ssi_vs_iso`: spatial organization of the movie, regardless of
#   whether it helps Vernier.
# - `D_across_arcmin2_s` and `D_along_arcmin2_s`: the real-unit diffusion
#   constants implied by the backimage 1x estimate and the synthetic scale.
#
# The key comparison is not "which metric is bigger" because the units differ.
# The key comparison is whether a condition changes Fisher and SSI in the same
# direction. If SSI stays flat while known-trace Fisher changes, spatial
# organization is not the same thing as Vernier-specific signal.

# %%
control_conditions = [
    "static_center",
    "brownian_iso_1x",
    "brownian_phase_cloud",
    "brownian_order_shuffled",
]
across_conditions = [
    diffusion_condition_name("across", scale)
    for scale in SYNTHETIC_DIFFUSION_SCALES
]
along_conditions = [
    diffusion_condition_name("along", scale)
    for scale in SYNTHETIC_DIFFUSION_SCALES
]
ordered_metric_conditions = [
    condition
    for condition in [*control_conditions, *across_conditions, *along_conditions]
    if condition in set(plot_metric_df["condition"])
]

early_comparison_df = plot_metric_df[plot_metric_df["condition"].isin(ordered_metric_conditions)].copy()
early_comparison_df["condition"] = pd.Categorical(
    early_comparison_df["condition"],
    categories=ordered_metric_conditions,
    ordered=True,
)
early_comparison_df = early_comparison_df.sort_values("condition")
early_comparison_df["condition_label"] = early_comparison_df["condition"].astype(str).map(condition_label)
early_comparison_df["group"] = np.select(
    [
        early_comparison_df["condition"].astype(str).isin(control_conditions),
        early_comparison_df["condition"].astype(str).isin(across_conditions),
        early_comparison_df["condition"].astype(str).isin(along_conditions),
    ],
    ["controls", "across sweep", "along sweep"],
    default="other",
)
early_comparison_df["known_trace_pixel_fisher"] = early_comparison_df["final_pixel_fisher"]
early_comparison_df["known_trace_fisher_vs_iso"] = early_comparison_df["pixel_fisher_vs_iso"]
early_comparison_df["general_ssi_vs_iso"] = early_comparison_df["ssi_bits_per_spike_vs_iso"]
show_table(
    early_comparison_df[
        [
            "group",
            "condition_label",
            "diffusion_constant",
            "D_across_arcmin2_s",
            "D_along_arcmin2_s",
            "known_trace_pixel_fisher",
            "known_trace_fisher_vs_iso",
            "general_ssi_vs_iso",
        ]
    ]
)

# %%
def plot_control_fisher_ssi(df: pd.DataFrame, conditions: list[str]) -> plt.Figure:
    rows = df[df["condition"].isin(conditions)].copy()
    rows["condition"] = pd.Categorical(rows["condition"], categories=conditions, ordered=True)
    rows = rows.sort_values("condition")
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=140)
    ax.bar(
        x - width / 2,
        rows["pixel_fisher_vs_iso"],
        width=width,
        color="#4c78a8",
        label="known-trace Fisher",
    )
    ax.bar(
        x + width / 2,
        rows["ssi_bits_per_spike_vs_iso"],
        width=width,
        color="#f58518",
        label="general SSI",
    )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(c) for c in rows["condition"].astype(str)], rotation=25, ha="right")
    ax.set_ylabel("normalized to Brownian iso 1x")
    ax.set_title("Controls: known-trace Fisher vs general SSI")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_synthetic_diffusion_sweeps(
    df: pd.DataFrame,
    *,
    fisher_mode: str = "pose_hidden",
) -> plt.Figure:
    fisher_modes = {
        "pose_hidden": {
            "column": "pose_hidden_pixel_fisher_vs_iso",
            "ylabel": "hidden-trace Fisher / iso 1x",
            "title": "Hidden-trace Fisher",
            "suptitle": "Synthetic anisotropic Brownian traces based on backimage fixation scale (hidden-trace)",
        },
        "pose_aware": {
            "column": "pixel_fisher_vs_iso",
            "ylabel": "known-trace Fisher / iso 1x",
            "title": "Known-trace Fisher",
            "suptitle": "Synthetic anisotropic Brownian traces based on backimage fixation scale (known-trace upper bound)",
        },
    }
    if fisher_mode not in fisher_modes:
        raise ValueError(f"Unknown fisher_mode={fisher_mode!r}; expected one of {sorted(fisher_modes)}")
    fisher_cfg = fisher_modes[fisher_mode]
    fisher_col = str(fisher_cfg["column"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=140, sharex=True)
    styles = {
        "vary across; along held 1x": ("#1f77b4", "o"),
        "vary along; across held 1x": ("#ff7f0e", "s"),
    }
    for curve, rows in df[df["curve"].isin(styles)].groupby("curve", sort=False):
        rows = rows.sort_values("diffusion_constant")
        color, marker = styles[curve]
        axes[0].plot(
            rows["diffusion_constant"],
            rows[fisher_col],
            marker=marker,
            color=color,
            linewidth=2.2,
            label=curve,
        )
        axes[1].plot(
            rows["diffusion_constant"],
            rows["ssi_bits_per_spike_vs_iso"],
            marker=marker,
            color=color,
            linewidth=2.2,
            label=curve,
        )
    for ax in axes:
        ax.axvline(1.0, color="#333333", linestyle=":", linewidth=1.0, alpha=0.65)
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.set_xlabel("relative diffusion constant on varied axis")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(str(fisher_cfg["ylabel"]))
    axes[1].set_ylabel("general SSI / iso 1x")
    axes[0].set_title(str(fisher_cfg["title"]))
    axes[1].set_title("General SSI")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(str(fisher_cfg["suptitle"]), y=1.03)
    fig.tight_layout()
    return fig


fig_control_compare = plot_control_fisher_ssi(plot_metric_df, control_conditions)
fig_control_compare

# %% [markdown]
# ## Known-trace, hidden-trace, and SSI are three different objects
#
# The synthetic section deliberately plots these side by side, but they should
# not be collapsed into one generic Vernier metric.
#
# **Known-trace Fisher** supplies the trajectory and averages the conditional
# Fisher over traces:
#
# $$
# J_{\mathrm{known}} = E_{\tau}\left[f_{\theta}(\tau)^T \Sigma_{\mathrm{count}}(\tau)^{-1} f_{\theta}(\tau)\right].
# $$
#
# **Hidden-trace Fisher** pools over possible traces. The observer pays a
# pose-marginal covariance penalty:
#
# $$
# J_{\mathrm{hidden}} = \bar f_{\theta}^T \left(\Sigma_{\mathrm{count}}+\Sigma_{\mathrm{pose}}\right)^{-1}\bar f_{\theta}, \qquad \Sigma_{\mathrm{pose}}=\mathrm{Cov}_{\tau}\left[\mu(\theta_0,\tau)\right].
# $$
#
# **General SSI** is not a Vernier observer. It asks whether the response is
# spatially organized, regardless of whether that organization separates
# `+delta` from `-delta`.
#
# The labels below therefore mean:
#
# - **Known-trace Fisher**: upper-bound Vernier signal when the trace is supplied.
# - **Hidden-trace Fisher**: constrained discriminability after pose-marginal
#   covariance enters the denominator.
# - **General SSI**: task-agnostic spatial organization.

# %% [markdown]
# The next anisotropic diffusion figure cells use the same SSI panel but change
# the Fisher panel. One figure shows **known-trace Fisher**, the supplied-
# trajectory upper bound. The other shows **hidden-trace Fisher**, where
# trajectory labels are not supplied and pose-marginal covariance enters the
# denominator.

# %%
fig_synth_diffusion_pose_aware = plot_synthetic_diffusion_sweeps(plot_metric_df, fisher_mode="pose_aware")
fig_synth_diffusion_pose_aware

fig_synth_diffusion = plot_synthetic_diffusion_sweeps(plot_metric_df)
fig_synth_diffusion_pose_hidden = fig_synth_diffusion
fig_synth_diffusion

# %% [markdown]
# ```text
# How to read the synthetic comparison:
#
# Known-trace Fisher: upper-bound Vernier signal when the trace is supplied.
#
# Hidden-trace Fisher: does the movie separate +delta from -delta after the
# trace label is hidden and pose-marginal covariance becomes nuisance?
#
# General SSI: is the movie spatially concentrated or broadly spatially
# informative, independent of Vernier sign?
#
# The active-sensing comparison is whether changing along- or across-contour
# diffusion changes hidden-trace Fisher and general SSI in the same way.
# ```

# %% [markdown]
# ### The pose confusion problem
#
# The trajectory plots above show retinal phase (where on the retina the
# stimulus is currently falling). The key insight is that the same retinal
# phase can arise from two different combinations of (Vernier offset, eye
# position):
#
# - `+δ offset` at eye position p looks the same as `−δ offset` at eye position
#   p + 2δ.
#
# When pose is **hidden**, the observer must average over all possible eye
# positions and cannot exploit this to their advantage. The result: response
# variance that comes from pose variability inflates the noise floor, reducing
# discriminability.
#
# When pose is **known**, responses can be interpreted conditional on the
# trajectory, so covariance over traces no longer enters the nuisance
# covariance.

# %%
# Pose confusion schematic — reuses _make_vernier_image from the Vernier schematic cell
fig_pose, axes_p = plt.subplots(1, 3, figsize=(10, 4.0), dpi=140)
_scenarios = [
    {
        "title": "Scenario A\n+δ offset, eye at centre\n(retinal edge at +δ)",
        "offset": 0.32,
        "eye_shift": 0.0,
        "color": "#d62728",
    },
    {
        "title": "Scenario B\n−δ offset, eye shifted right\n(retinal edge ALSO at +δ)\n← identical retinal image!",
        "offset": -0.32,
        "eye_shift": 0.64,
        "color": "#1f77b4",
    },
    {
        "title": "True −δ, eye at centre\n(retinal edge at −δ)\n← distinguishable",
        "offset": -0.32,
        "eye_shift": 0.0,
        "color": "#888888",
    },
]
for ax, s in zip(axes_p, _scenarios):
    # Build retinal image from pixel array — avoids set_axis_off() killing the facecolor
    _ret_offset_px = int(round((s["offset"] - s["eye_shift"]) * 56))  # scale to pixels
    _pose_img = _make_vernier_image(_ret_offset_px, _VSCH, _VSCH)
    ax.imshow(_pose_img, interpolation="nearest", aspect="auto")
    # Dashed vertical line at the retinal edge position
    _edge_col = _VSCX + _ret_offset_px
    ax.axvline(_edge_col, color=s["color"], linestyle="--", linewidth=2.0, alpha=0.85)
    _title_color = "#444444" if s["color"] == "#888888" else s["color"]
    ax.set_title(s["title"], fontsize=8, color=_title_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for _sp in ax.spines.values():
        _sp.set_visible(False)

fig_pose.suptitle(
    "Pose confusion: Scenarios A and B cast identical retinal images.\n"
    "Without the eye-position label, a hidden-trace observer cannot tell them apart.",
    fontsize=9,
)
fig_pose.tight_layout()
fig_pose

# %% [markdown]
# ## Readout logic
#
# ### Readout name decoder
#
# The output rows use long names that encode exactly what the ideal observer
# assumes. Here is a plain-English translation of each piece:
#
# - `pose_aware`: historical code name for the known-trace upper bound; the
#   observer knows the exact eye position at every frame.
# - `pose_blind`: historical code name for hidden-trace scoring; the observer is
#   not given the eye-trajectory label.
# - `diagonal`: neurons are treated as independent, ignoring correlated noise.
# - `full_cov`: full population covariance matrix is used, capturing correlated
#   noise.
# - `poisson`: spike-count noise follows Poisson statistics, so variance equals
#   mean.
# - `marginal` or `count_plus_marginal`: pose-marginal response spread is added
#   to the noise floor.
# - `unit_subset`: the run used a smaller set of neurons for numerical
#   feasibility.
# - `compact`: diagnostic for whether discounting suspected nuisance subspaces
#   changes the score.
# - `pose_uncertain_sigma_N`: intermediate observer with Gaussian uncertainty
#   about pose.
#
# **The most important pair**: `pose_aware_diagonal_poisson` vs
# `pose_blind_diagonal_count_plus_marginal`. The second row name is historical
# code vocabulary. In this walkthrough, we call the first **known-trace Fisher**
# and the second **hidden-trace Fisher**. If real FEM helps even when the
# trajectory label is hidden, it will show up here. If hidden-trace Fisher is
# close to known-trace Fisher, the position ambiguity costs little for the given
# condition.
#
# `pose_aware_diagonal_poisson`
#
# - The observer knows which retinal phase/trace produced the response.
# - For each trace, the code computes a diagonal count-noise Fisher curve.
# - Time bins are accumulated as if they are conditionally independent blocks.
#
# `pose_blind_diagonal_count_plus_marginal` (hidden-trace diagonal)
#
# - The observer does not know the trace label.
# - Responses are pooled over possible eye traces.
# - The diagonal variance is count noise plus pose-marginal response covariance
#   over traces.
#
# `pose_blind_full_cov_optimal` or `pose_blind_full_cov_optimal_unit_subset`
# (hidden-trace full covariance)
#
# - The observer still does not know trace labels.
# - The covariance is full population covariance, with count-noise floor and
#   shrinkage.
# - If the row says `unit_subset`, this is a numerical diagnostic over selected
#   units, not the full population.
#
# `pose_uncertain_diagonal_sigma...`
#
# - Interpolates between known-trace and hidden-trace diagonal scoring by
#   weighting traces according to pose distance.
#
# `pose_blind_compact_*`
#
# - Diagnostics asking whether removing or discounting compact nuisance
#   subspaces changes the hidden-trace Fisher score.

# %% [markdown]
# ## End-stage audit: load cached real-twin Vernier outputs
#
# Everything above was synthetic and controlled. From here on, the notebook reads
# existing cached Vernier outputs from the digital twin. These cache-backed plots
# should be treated as the real analysis audit, not as the first teaching move.
#
# The main output tables are:
#
# - `information_summary.csv`: per-trace and aggregate readout rows.
# - `condition_reliability_summary.csv`: condition/readout summaries across
#   traces.
# - `paired_baseline_contrast_summary.csv`: trace-paired condition vs baseline
#   contrasts.

# %%
if RUN_DIR is None:
    rel_df = pd.DataFrame()
    info_df = pd.DataFrame()
    contrast_df = pd.DataFrame()
    manifest = {}
    print("No cached Vernier run directory found. Rendering and eye-trace cells still work.")
else:
    info_df = read_csv_optional(RUN_DIR / "information_summary.csv")
    rel_df = read_csv_optional(RUN_DIR / "condition_reliability_summary.csv")
    contrast_df = read_csv_optional(RUN_DIR / "paired_baseline_contrast_summary.csv")
    manifest = read_json_optional(RUN_DIR / "vernier_active_sensing_manifest.json")

print(f"information_summary rows: {len(info_df)}")
print(f"condition_reliability_summary rows: {len(rel_df)}")
print(f"paired_baseline_contrast_summary rows: {len(contrast_df)}")

if manifest:
    manifest_args = manifest.get("args", {})
    show_table(
        pd.DataFrame(
            [
                {"key": "conditions", "value": manifest_args.get("conditions")},
                {"key": "fd_steps_arcmin", "value": manifest_args.get("fd_steps_arcmin")},
                {"key": "n_traces", "value": manifest_args.get("n_traces")},
                {"key": "max_frames", "value": manifest_args.get("max_frames")},
                {"key": "inference_mode", "value": manifest_args.get("inference_mode")},
                {"key": "spatial_collapse", "value": manifest_args.get("spatial_collapse")},
                {"key": "run_full_cov_pose_blind", "value": manifest_args.get("run_full_cov_pose_blind")},
                {"key": "run_compact_aware_pose_blind", "value": manifest_args.get("run_compact_aware_pose_blind")},
            ]
        )
    )

# %%
if JOINT_RUN_DIR is None:
    joint_summary = pd.DataFrame()
else:
    joint_summary = read_csv_optional(JOINT_RUN_DIR / "joint_geometry_observer_summary.csv")

if TRAJECTORY_TABLE_RUN_DIR is None:
    table_summary = pd.DataFrame()
else:
    table_summary = read_csv_optional(TRAJECTORY_TABLE_RUN_DIR / "trajectory_table_observer_summary.csv")

# %% [markdown]
# ## Claim boundaries before looking at results
#
# Keep these boundaries next to the first result plots. They prevent the
# dashboard from silently turning a diagnostic pattern into a behavior claim.

# %%
claim_boundaries = pd.DataFrame(
    [
        {
            "result": "Known-trace Fisher improves",
            "safe_claim": "FEMs can create Vernier-aligned response modulation in the twin when eye trajectory is supplied.",
            "do_not_claim_yet": "This signal is usable when eye trajectory is hidden.",
        },
        {
            "result": "Hidden-trace Fisher drops",
            "safe_claim": "Hidden eye position becomes nuisance covariance under this observer.",
            "do_not_claim_yet": "Vernier is established as the main joint-inference success story before the stricter tests are run.",
        },
        {
            "result": "Across-contour reduction helps",
            "safe_claim": "For this vertical Vernier stimulus, reducing horizontal/across motion can reduce the pose nuisance.",
            "do_not_claim_yet": "Along-contour movements get longer.",
        },
        {
            "result": "Vernier joint observer struggles",
            "safe_claim": "Under the tested observer and catalog, Vernier did not yet show robust hidden-trace rescue.",
            "do_not_claim_yet": "Negative results from one catalog or likelihood scale rule out useful Vernier image-structure contribution in general.",
        },
        {
            "result": "Vernier joint observer improves under stricter tests",
            "safe_claim": "The bar-edge movie contains image/temporal structure that can contribute to pose marginalization under the tested observer.",
            "do_not_claim_yet": "The same mechanism is proven for all priors, likelihood calibrations, and trajectory catalogs.",
        },
        {
            "result": "SSI changes",
            "safe_claim": "Spatial response organization changes.",
            "do_not_claim_yet": "Vernier discriminability changes.",
        },
    ]
)
show_table(claim_boundaries)

# %%
def metric_value(
    df: pd.DataFrame,
    *,
    readout: str,
    condition: str,
    fd_step: float,
    column: str = "mean_final_fisher",
) -> float:
    if df.empty:
        return float("nan")
    rows = df[
        (df["readout"] == readout)
        & (df["condition"] == condition)
        & np.isclose(pd.to_numeric(df["fd_step_arcmin"], errors="coerce"), float(fd_step))
    ]
    if rows.empty or column not in rows:
        return float("nan")
    return float(rows[column].iloc[0])


def axis_diffusion_points(
    df: pd.DataFrame,
    *,
    fd_step: float,
    readout: str = "pose_aware_diagonal_poisson",
    column: str = "mean_final_fisher",
    normalize_to_static: bool = True,
) -> pd.DataFrame:
    """Return the cached 0/1 axis-diffusion proxy.

    The Vernier caches currently have binary axis controls:
    - `axis_vertical`: along-contour real, across-contour removed.
    - `axis_horizontal`: across-contour real, along-contour removed.
    - `real_fem`: along and across both at their real 1x scale.

    This gives two requested one-axis sweeps, each with the other axis held at
    1x, but only at diffusion constants 0 and 1. A dense curve needs an
    anisotropic model rerun.
    """
    static = metric_value(df, readout=readout, condition="static_center", fd_step=fd_step, column=column)
    denom = static if normalize_to_static and np.isfinite(static) and static > 0 else 1.0
    rows = [
        {
            "curve": "vary across; along held 1x",
            "varied_axis": "across contour",
            "diffusion_constant": 0.0,
            "condition": "axis_vertical",
            "value": metric_value(df, readout=readout, condition="axis_vertical", fd_step=fd_step, column=column) / denom,
        },
        {
            "curve": "vary across; along held 1x",
            "varied_axis": "across contour",
            "diffusion_constant": 1.0,
            "condition": "real_fem",
            "value": metric_value(df, readout=readout, condition="real_fem", fd_step=fd_step, column=column) / denom,
        },
        {
            "curve": "vary along; across held 1x",
            "varied_axis": "along contour",
            "diffusion_constant": 0.0,
            "condition": "axis_horizontal",
            "value": metric_value(df, readout=readout, condition="axis_horizontal", fd_step=fd_step, column=column) / denom,
        },
        {
            "curve": "vary along; across held 1x",
            "varied_axis": "along contour",
            "diffusion_constant": 1.0,
            "condition": "real_fem",
            "value": metric_value(df, readout=readout, condition="real_fem", fd_step=fd_step, column=column) / denom,
        },
    ]
    return pd.DataFrame(rows)


def plot_axis_diffusion_proxy(
    df: pd.DataFrame,
    *,
    fd_step: float,
    readout: str = "pose_aware_diagonal_poisson",
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    own_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=140)
    else:
        fig = ax.figure
    points = axis_diffusion_points(df, fd_step=fd_step, readout=readout)
    if points["value"].notna().sum() == 0:
        ax.text(0.5, 0.5, "No axis-control rows found", ha="center", va="center")
        return fig if own_fig else None
    styles = {
        "vary across; along held 1x": ("#ff7f0e", "o"),
        "vary along; across held 1x": ("#2ca02c", "s"),
    }
    for curve, rows in points.groupby("curve", sort=False):
        rows = rows.sort_values("diffusion_constant")
        color, marker = styles.get(curve, ("#777777", "o"))
        ax.plot(
            rows["diffusion_constant"],
            rows["value"],
            marker=marker,
            linewidth=2.2,
            color=color,
            label=curve,
        )
    ax.axvline(1.0, color="#333333", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("relative diffusion constant on varied axis")
    ax.set_ylabel("Fisher / static-center Fisher")
    ax.set_title("Cached 0/1 axis-control proxy, not dense diffusion sweep")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig if own_fig else None


def trajectory_table_setting_label(df: pd.DataFrame) -> str:
    """Compact label for trajectory-table catalog/evaluation settings."""
    if df.empty:
        return ""
    parts: list[str] = []
    if "trajectory_table_mode" in df:
        modes = sorted(map(str, pd.Series(df["trajectory_table_mode"]).dropna().unique()))
        if len(modes) == 1:
            parts.append(modes[0])
    for col, label in [
        ("trajectory_table_include_self", "include_self"),
        ("trajectory_table_leave_one_out", "leave_one_out"),
    ]:
        if col in df:
            vals = sorted(map(str, pd.Series(df[col]).dropna().unique()))
            if len(vals) == 1:
                parts.append(f"{label}={vals[0]}")
    return "; ".join(parts)


def plot_key_result_dashboard(
    rel_df: pd.DataFrame,
    table_summary: pd.DataFrame,
    *,
    fd_step: float,
) -> plt.Figure | None:
    if rel_df.empty:
        print("No reliability table loaded; dashboard skipped.")
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), dpi=140)

    # Result 1: known-trace Fisher, normalized to static.
    conditions = ["static_center", "static_phase_cloud_matched_positions", "real_fem", "order_shuffled_positions"]
    static = metric_value(rel_df, readout="pose_aware_diagonal_poisson", condition="static_center", fd_step=fd_step)
    denom = static if np.isfinite(static) and static > 0 else 1.0
    vals = [
        metric_value(rel_df, readout="pose_aware_diagonal_poisson", condition=condition, fd_step=fd_step) / denom
        for condition in conditions
    ]
    axes[0, 0].bar(np.arange(len(conditions)), vals, color=[COLORS.get(c, "#777777") for c in conditions], alpha=0.9)
    axes[0, 0].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0, 0].set_xticks(np.arange(len(conditions)))
    axes[0, 0].set_xticklabels([condition_label(c) for c in conditions], rotation=25, ha="right")
    axes[0, 0].set_ylabel("Fisher / static")
    axes[0, 0].set_title("1. Known-eye upper-bound signal")

    # Result 2: known-trace versus hidden-trace.
    conditions = ["static_center", "real_fem", "static_phase_cloud_matched_positions", "order_shuffled_positions"]
    x = np.arange(len(conditions))
    width = 0.36
    aware = [
        metric_value(rel_df, readout="pose_aware_diagonal_poisson", condition=condition, fd_step=fd_step) / denom
        for condition in conditions
    ]
    hidden = [
        metric_value(rel_df, readout="pose_blind_diagonal_count_plus_marginal", condition=condition, fd_step=fd_step) / denom
        for condition in conditions
    ]
    axes[0, 1].bar(x - width / 2, aware, width=width, label="known-trace", color="#54a24b", alpha=0.9)
    axes[0, 1].bar(x + width / 2, hidden, width=width, label="hidden-trace", color="#f58518", alpha=0.9)
    axes[0, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([condition_label(c) for c in conditions], rotation=25, ha="right")
    axes[0, 1].set_ylabel("Fisher / static")
    axes[0, 1].set_title("2. Hidden-trace nuisance penalty")
    axes[0, 1].legend(frameon=False, fontsize=8)

    # Result 3: along/across diffusion proxy.
    plot_axis_diffusion_proxy(rel_df, fd_step=fd_step, ax=axes[1, 0])
    axes[1, 0].set_title("3. Cached 0/1 axis-control proxy")

    # Result 4: Vernier trajectory-table diagnostic if available.
    if not table_summary.empty and "joint_accuracy" in table_summary:
        rows = table_summary[np.isclose(pd.to_numeric(table_summary["fd_step_arcmin"], errors="coerce"), float(fd_step))].copy()
        conditions = [c for c in ["static_center", "real_fem", "order_shuffled_positions", "axis_horizontal", "axis_vertical"] if c in set(rows["condition"])]
        x = np.arange(len(conditions))
        zero = [float(rows.loc[rows["condition"] == c, "zero_accuracy"].iloc[0]) for c in conditions]
        joint = [float(rows.loc[rows["condition"] == c, "joint_accuracy"].iloc[0]) for c in conditions]
        known = [float(rows.loc[rows["condition"] == c, "known_accuracy"].iloc[0]) for c in conditions]
        width = 0.25
        axes[1, 1].bar(x - width, zero, width=width, label="zero-eye", color="#999999")
        axes[1, 1].bar(x, joint, width=width, label="joint", color="#4c78a8")
        axes[1, 1].bar(x + width, known, width=width, label="known-trace", color="#54a24b")
        axes[1, 1].axhline(0.5, color="#333333", linestyle="--", linewidth=1.0)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([condition_label(c) for c in conditions], rotation=25, ha="right")
        axes[1, 1].set_ylabel("accuracy")
        axes[1, 1].legend(frameon=False, fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No trajectory-table summary loaded", ha="center", va="center")
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    _setting = trajectory_table_setting_label(rows) if "rows" in locals() and not rows.empty else ""
    axes[1, 1].set_title("4. Trajectory-table diagnostic" + (f"\n{_setting}" if _setting else ""))

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Vernier diagnostic dashboard, fd={fd_step:g} arcmin", y=1.02)
    fig.tight_layout()
    return fig


plot_step = nearest_available_step(rel_df, FD_STEP_TO_PLOT)
fig_dashboard = plot_key_result_dashboard(rel_df, table_summary, fd_step=plot_step) if plot_step is not None else None
fig_dashboard

# %% [markdown]
# ```text
# Result 1
# Question: Can motion create Vernier-aligned signal under a known-trace upper bound?
# Readout: Known-trace diagonal Poisson Fisher.
# Takeaway: A bar above static means model V1 contains more offset signal under that motion condition, but only when eye trajectory is supplied.
#
# Result 2
# Question: What happens to that signal when the eye trajectory label is hidden?
# Readout: Known-trace versus hidden-trace diagonal Fisher.
# Takeaway: A large drop means the modulation is mostly an upper-bound signal, not automatically usable evidence.
#
# Result 3
# Question: Is the active-sensing move longer along-contour drift, or smaller across-contour drift?
# Readout: Cached axis controls, normalized to static-center Fisher.
# Takeaway: The key hypothesis is across-contour shrink during precise observation, not necessarily longer along-contour motion.
#
# Result 4
# Question: Can the Vernier stimulus itself support joint trajectory marginalization?
# Readout: Trajectory-table likelihood-ratio observer.
# Takeaway: Treat this as a positive second-pass test. A robust result should survive leave-one-out/cross-prior catalogs and likelihood calibration.
# ```

# %% [markdown]
# ## Known-trace condition comparison
#
# This is the simplest score to understand: each trace is known, so movement is
# not a hidden nuisance variable. The score asks whether the response change
# caused by the Vernier offset is large relative to diagonal count noise.
#
# **What to look for**: bars above static_center mean that condition provides
# more Vernier information than no movement at all under the known-trace upper
# bound. This is not the final active-sensing claim; it only says the motion can
# generate offset-aligned modulation if the trajectory nuisance is supplied for
# free.
#
# **Threshold proxy** (second plot): `1 / sqrt(Fisher)`. Lower bars mean better
# acuity. The reference level is static_center; conditions with shorter bars
# could, in principle, allow finer discrimination.

# %%
def plot_condition_bars(
    df: pd.DataFrame,
    *,
    readout: str,
    fd_step: float,
    conditions: list[str],
    value_col: str = "mean_final_fisher",
    title: str = "",
) -> plt.Figure | None:
    if df.empty:
        print("No reliability dataframe available.")
        return None
    sub = df[
        (df["readout"] == readout)
        & np.isclose(pd.to_numeric(df["fd_step_arcmin"], errors="coerce"), float(fd_step))
        & df["condition"].isin(conditions)
    ].copy()
    if sub.empty:
        print(f"No rows for readout={readout!r}, fd_step={fd_step}")
        return None
    sub["condition"] = pd.Categorical(sub["condition"], categories=conditions, ordered=True)
    sub = sub.sort_values("condition")
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(sub)), 3.6), dpi=140)
    ax.bar(
        np.arange(len(sub)),
        sub[value_col],
        color=[COLORS.get(c, "#777777") for c in sub["condition"].astype(str)],
        alpha=0.9,
    )
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels([condition_label(c) for c in sub["condition"].astype(str)], rotation=35, ha="right")
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(title or f"{readout}, fd={fd_step:g} arcmin")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


plot_step = nearest_available_step(rel_df, FD_STEP_TO_PLOT)
primary_conditions = [
    "static_center",
    "static_phase_cloud_matched_positions",
    "real_fem",
    "order_shuffled_positions",
    "axis_horizontal",
    "axis_vertical",
    "scaled_real_0.5",
    "scaled_real_1.5",
]
fig = (
    plot_condition_bars(
        rel_df,
        readout="pose_aware_diagonal_poisson",
        fd_step=plot_step,
        conditions=primary_conditions,
        value_col="mean_final_fisher",
        title="Known-trace diagonal Poisson Fisher",
    )
    if plot_step is not None
    else None
)
fig

# %%
fig = (
    plot_condition_bars(
        rel_df,
        readout="pose_aware_diagonal_poisson",
        fd_step=plot_step,
        conditions=primary_conditions,
        value_col="mean_final_threshold_proxy",
        title="Known-trace threshold proxy, lower is better",
    )
    if plot_step is not None
    else None
)
fig

# %% [markdown]
# ```text
# Question:
# Does a motion condition increase Vernier-relevant signal when the eye trace is supplied?
#
# Readout:
# Known-trace diagonal Poisson Fisher. The eye trace is supplied to the observer.
#
# Takeaway:
# This is the upper-bound signal story. It does not by itself say whether the
# signal is usable when the trajectory `tau` is hidden.
# ```

# %% [markdown]
# ## Hidden-trace nuisance equation
#
# The hidden-trace observer first averages responses over possible eye traces:
#
# $$
# \bar\mu_{\pm}=E_{\tau}\left[\mu(\pm\delta,\tau)\right], \qquad \Delta\bar\mu=\bar\mu_{+}-\bar\mu_{-}.
# $$
#
# The covariance used by the hidden-trace readout contains both count noise and
# pose-marginal response covariance:
#
# $$
# \Sigma_{\mathrm{hidden}}=E_{\tau}\left[\Sigma_{\mathrm{count}}(\tau)\right]+\mathrm{Cov}_{\tau}\left[\mu(\theta_0,\tau)\right].
# $$
#
# The corresponding local discriminability is:
#
# $$
# d_{\mathrm{hidden}}^{\prime 2}=\Delta\bar\mu^T\Sigma_{\mathrm{hidden}}^{-1}\Delta\bar\mu.
# $$
#
# This is the heart of the Vernier lesson: motion-induced signal only becomes
# useful under hidden pose if it survives this pose-marginal covariance penalty.

# %% [markdown]
# ## Known-trace versus hidden-trace
#
# This is the central nuisance test. If movement creates response modulation
# that is not aligned with the Vernier signal axis, hiding pose should reduce
# constrained discriminability.
#
# **What to look for**: for each condition, compare the bar heights across
# readouts. A large drop from known-trace Fisher to hidden-trace Fisher means
# the eye-position ambiguity costs a lot: the movement-induced response variance
# is mostly nuisance from the decoder's perspective. In Vernier, a large gap is
# expected and should be read as the pose-confusion lesson, not as a rescued
# active-sensing result.
#
# The hidden-trace full-covariance bars are stricter still: they also account for
# correlated noise across neurons. A further drop there suggests off-diagonal
# covariance matters and the diagonal approximation was over-optimistic.

# %%
def plot_readout_comparison(
    df: pd.DataFrame,
    *,
    fd_step: float,
    conditions: list[str],
    readouts: list[str],
    value_col: str = "mean_final_fisher",
) -> plt.Figure | None:
    if df.empty:
        print("No reliability dataframe available.")
        return None
    sub = df[
        np.isclose(pd.to_numeric(df["fd_step_arcmin"], errors="coerce"), float(fd_step))
        & df["condition"].isin(conditions)
        & df["readout"].isin(readouts)
    ].copy()
    if sub.empty:
        print(f"No rows for fd_step={fd_step} and readouts={readouts}")
        return None
    cond_order = [c for c in conditions if c in set(sub["condition"])]
    readout_order = [r for r in readouts if r in set(sub["readout"])]
    width = 0.8 / max(len(readout_order), 1)
    x = np.arange(len(cond_order))
    fig, ax = plt.subplots(figsize=(max(8, 0.95 * len(cond_order)), 4.0), dpi=140)
    for idx, readout in enumerate(readout_order):
        vals = []
        for condition in cond_order:
            rows = sub[(sub["condition"] == condition) & (sub["readout"] == readout)]
            vals.append(float(rows[value_col].iloc[0]) if not rows.empty else np.nan)
        ax.bar(x + (idx - (len(readout_order) - 1) / 2) * width, vals, width=width, label=readout_label(readout))
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(c) for c in cond_order], rotation=35, ha="right")
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(f"Readout comparison at fd={fd_step:g} arcmin")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


comparison_readouts = [
    "pose_aware_diagonal_poisson",
    "pose_blind_diagonal_count_plus_marginal",
    "pose_blind_full_cov_optimal",
    "pose_blind_full_cov_optimal_unit_subset",
]
fig = (
    plot_readout_comparison(
        rel_df,
        fd_step=plot_step,
        conditions=primary_conditions,
        readouts=comparison_readouts,
        value_col="mean_final_fisher",
    )
    if plot_step is not None
    else None
)
fig

# %% [markdown]
# ```text
# Question:
# Does motion-induced response modulation remain useful when eye pose is hidden?
#
# Readout:
# Known-trace Fisher conditions on the supplied trace. Hidden-trace Fisher pools
# over possible eye traces and pays a pose-marginal covariance penalty.
#
# Takeaway:
# A large known-trace > hidden-trace gap means the motion creates signal, but
# much of it is entangled with eye-position uncertainty. Vernier is useful
# because it exposes this failure mode cleanly.
# ```

# %% [markdown]
# ## Pose uncertainty sweep
#
# This section is easy to misread, so here is the plain-English version first.
# `sigma` is the assumed uncertainty in the observer's eye-position estimate,
# measured in arcmin.
#
# - `sigma = 0`: the observer knows the eye position exactly. This is close to
#   the known-trace upper bound.
# - small `sigma`: the observer has a noisy but useful eye-position estimate.
# - large `sigma`: the observer averages over many possible positions. This
#   approaches the hidden-trace problem.
#
# The plot below is normalized to each condition's `sigma=0` Fisher, so the
# question is not "which condition has the largest absolute Fisher?" The question
# is: **how quickly does usable Fisher disappear as pose becomes uncertain?**
#
# This is a diagnostic bridge, not a headline result. The implementation is
# diagonal and does not replace the full hidden-trace covariance analysis.

# %%
pose_uncertainty_legend = pd.DataFrame(
    [
        {
            "sigma_arcmin": "0",
            "observer_assumption": "exact eye-position estimate",
            "interpretation": "known-trace upper-bound end of the continuum",
        },
        {
            "sigma_arcmin": "small",
            "observer_assumption": "nearby poses are plausible; distant poses are unlikely",
            "interpretation": "tests how much pose precision is needed",
        },
        {
            "sigma_arcmin": "large",
            "observer_assumption": "many eye positions are plausible",
            "interpretation": "moves toward hidden-trace marginalization",
        },
    ]
)
show_table(pose_uncertainty_legend)

# %%
def plot_pose_uncertainty(
    df: pd.DataFrame,
    *,
    fd_step: float,
    conditions: list[str],
    normalize_to_sigma0: bool = True,
) -> plt.Figure | None:
    if df.empty:
        print("No reliability dataframe available.")
        return None
    sub = df[
        np.isclose(pd.to_numeric(df["fd_step_arcmin"], errors="coerce"), float(fd_step))
        & df["condition"].isin(conditions)
        & df["readout"].astype(str).str.startswith("pose_uncertain_diagonal_sigma")
    ].copy()
    if sub.empty:
        print("No pose-uncertainty rows in this run.")
        return None
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=140)
    for condition in conditions:
        rows = sub[sub["condition"] == condition].sort_values("pose_sigma_arcmin")
        if rows.empty:
            continue
        y = pd.to_numeric(rows["mean_final_fisher"], errors="coerce").to_numpy(dtype=float)
        if normalize_to_sigma0:
            sigma = pd.to_numeric(rows["pose_sigma_arcmin"], errors="coerce").to_numpy(dtype=float)
            zero_idx = int(np.nanargmin(np.abs(sigma))) if np.isfinite(sigma).any() else 0
            denom = y[zero_idx] if np.isfinite(y[zero_idx]) and y[zero_idx] > 0 else 1.0
            y = y / denom
        ax.plot(
            rows["pose_sigma_arcmin"],
            y,
            marker="o",
            label=condition_label(condition),
            color=COLORS.get(condition),
        )
    ax.set_xlabel("assumed eye-position uncertainty sigma (arcmin)")
    ax.set_ylabel("Fisher / sigma=0 Fisher" if normalize_to_sigma0 else "mean final Fisher")
    ax.set_title(f"How much pose precision is needed? fd={fd_step:g} arcmin")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


fig = (
    plot_pose_uncertainty(
        rel_df,
        fd_step=plot_step,
        conditions=["static_center", "real_fem", "scaled_real_0.5", "axis_horizontal", "axis_vertical"],
    )
    if plot_step is not None
    else None
)
fig

# %% [markdown]
# ## Along/across diffusion constants
#
# This is the reframed active-sensing question:
#
# - The hypothesis is **not** that along-contour motion becomes longer.
# - The hypothesis is that **across-contour diffusion is reduced** during precise
#   observations, because across-contour motion directly mimics Vernier offset.
#
# The cached Vernier run contains a binary axis sweep rather than a dense
# anisotropic diffusion grid. For this vertical Vernier stimulus:
#
# - `axis_vertical` = along-contour real motion with across-contour motion
#   removed. This is the point `D_across = 0`, with `D_along = 1`.
# - `axis_horizontal` = across-contour real motion with along-contour motion
#   removed. This is the point `D_along = 0`, with `D_across = 1`.
# - `real_fem` = both axes at their real 1x diffusion.
#
# So the plot below shows the two requested curves at the currently cached
# diffusion constants 0 and 1. A true multi-point curve would require rerunning
# the model with anisotropic traces on the intended grid:
#
# ```text
# D_across in {0, .125, .25, .5, 1, 2, 3}, D_along = 1
# D_along  in {0, .125, .25, .5, 1, 2, 3}, D_across = 1
# ```

# %%
if plot_step is not None:
    axis_diffusion_table = axis_diffusion_points(rel_df, fd_step=plot_step)
    show_table(axis_diffusion_table)
    fig = plot_axis_diffusion_proxy(rel_df, fd_step=plot_step)
else:
    axis_diffusion_table = pd.DataFrame()
    fig = None
fig

# %% [markdown]
# ```text
# Question:
# Does reducing across-contour diffusion preserve or improve Vernier information
# when along-contour diffusion is held at its real 1x value?
#
# Readout:
# Cached known-trace Fisher axis controls, normalized to static-center Fisher.
#
# Takeaway:
# This is the active-sensing shrinkage framing: precision can come from reducing
# across-contour nuisance, not from making along-contour paths longer.
# ```

# %% [markdown]
# ## Motion-amplitude sweep
#
# The scale sweep asks whether reducing or expanding the real FEM path improves
# information about fine Vernier position. Scale-matched phase-cloud and
# order-shuffled controls are important because they separate amplitude from
# temporal order.
#
# **What to look for**: does the "scaled real" curve peak below scale = 1,
# suggesting reduced drift can be better for precision? The matched phase-cloud
# and shuffled-order curves show what amplitude alone would give without
# temporal structure. If "scaled real" rises above the matched controls,
# temporal order adds something beyond mere amplitude.

# %%
def parse_scale_condition(condition: str) -> tuple[str, float] | None:
    condition = str(condition)
    patterns = [
        ("scaled_real_", "scaled real"),
        ("static_phase_cloud_matched_scaled_", "matched phase cloud"),
        ("order_shuffled_scaled_", "order shuffled"),
    ]
    if condition == "real_fem":
        return "scaled real", 1.0
    for prefix, family in patterns:
        if condition.startswith(prefix):
            try:
                return family, float(condition[len(prefix) :])
            except ValueError:
                return None
    return None


def plot_scale_sweep(
    df: pd.DataFrame,
    *,
    fd_step: float,
    readout: str = "pose_aware_diagonal_poisson",
) -> plt.Figure | None:
    if df.empty:
        print("No reliability dataframe available.")
        return None
    sub = df[
        (df["readout"] == readout)
        & np.isclose(pd.to_numeric(df["fd_step_arcmin"], errors="coerce"), float(fd_step))
    ].copy()
    parsed = sub["condition"].map(parse_scale_condition)
    sub["family"] = [item[0] if item else None for item in parsed]
    sub["scale"] = [item[1] if item else np.nan for item in parsed]
    sub = sub.dropna(subset=["family", "scale"])
    if sub.empty:
        print("No scale-sweep rows in this run.")
        return None
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=140)
    palette = {
        "scaled real": "#d62728",
        "matched phase cloud": "#1f77b4",
        "order shuffled": "#9467bd",
    }
    for family, rows in sub.groupby("family"):
        rows = rows.sort_values("scale")
        ax.plot(rows["scale"], rows["mean_final_fisher"], marker="o", label=family, color=palette.get(family))
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("motion scale around trace mean")
    ax.set_ylabel("mean final Fisher")
    ax.set_title(f"Scale sweep: {readout}, fd={fd_step:g} arcmin")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


fig = plot_scale_sweep(rel_df, fd_step=plot_step) if plot_step is not None else None
fig

# %% [markdown]
# ```text
# Question:
# Does uniformly reducing or expanding the real trajectory help Vernier
# discriminability?
#
# Readout:
# Known-trace Fisher for isotropically scaled real traces and matched controls.
#
# Takeaway:
# This is an amplitude control. It is useful, but it is not the same as the
# along/across diffusion question above because both axes are scaled together.
# ```

# %% [markdown]
# ### Optional anisotropic diffusion rerun scaffold
#
# The plot above uses the cached 0/1 axis controls. To get the dense curves you
# pictured, the production run should add anisotropic conditions with one axis
# held at 1x while the other is swept:
#
# ```text
# along sweep:  D_along = [0, 0.125, 0.25, 0.5, 1, 2, 3], D_across = 1
# across sweep: D_across = [0, 0.125, 0.25, 0.5, 1, 2, 3], D_along = 1
# ```
#
# Because trace amplitude scale `s` implies approximate diffusion scale `s^2`,
# decide before the rerun whether the condition names should encode amplitude
# scale or diffusion scale. For the tutorial plot, diffusion scale is easier to
# explain.

# %%
ANISOTROPIC_DIFFUSION_SCALES = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0]
anisotropic_sweep_plan = pd.DataFrame(
    [
        {
            "curve": "vary along; across held 1x",
            "D_along": scale,
            "D_across": 1.0,
            "amplitude_scale_along_if_D_encoded": math.sqrt(scale),
            "amplitude_scale_across_if_D_encoded": 1.0,
        }
        for scale in ANISOTROPIC_DIFFUSION_SCALES
    ]
    + [
        {
            "curve": "vary across; along held 1x",
            "D_along": 1.0,
            "D_across": scale,
            "amplitude_scale_along_if_D_encoded": 1.0,
            "amplitude_scale_across_if_D_encoded": math.sqrt(scale),
        }
        for scale in ANISOTROPIC_DIFFUSION_SCALES
    ]
)
show_table(anisotropic_sweep_plan)

# %% [markdown]
# ## Trace-paired contrasts
#
# Contrast rows compare a condition with a baseline on paired traces. The
# `threshold_ratio` is:
#
# ```text
# threshold_ratio = sqrt(baseline_Fisher / condition_Fisher)
# ```
#
# Values below 1 mean the condition would need a smaller offset to reach the
# same Fisher proxy, under that readout.

# %%
if not contrast_df.empty:
    contrast_focus = contrast_df[
        (contrast_df["readout"] == "pose_aware_diagonal_poisson")
        & contrast_df["condition"].isin(["real_fem", "scaled_real_0.5", "scaled_real_1.5", "axis_horizontal", "axis_vertical"])
    ].sort_values(["fd_step_arcmin", "condition", "baseline_condition"])
    show_table(
        contrast_focus[
            [
                "condition",
                "baseline_condition",
                "fd_step_arcmin",
                "n",
                "mean_fisher_delta",
                "mean_threshold_ratio",
                "p_condition_beats_baseline",
            ]
        ],
        n=30,
    )
else:
    print("No paired contrast summary available.")

# %%
def plot_threshold_contrasts(df: pd.DataFrame, *, fd_step: float) -> plt.Figure | None:
    if df.empty:
        print("No contrast dataframe available.")
        return None
    sub = df[
        (df["readout"] == "pose_aware_diagonal_poisson")
        & np.isclose(pd.to_numeric(df["fd_step_arcmin"], errors="coerce"), float(fd_step))
    ].copy()
    keep_pairs = [
        ("real_fem", "static_center"),
        ("real_fem", "static_phase_cloud_matched_positions"),
        ("scaled_real_0.5", "static_phase_cloud_matched_scaled_0.5"),
        ("scaled_real_1.5", "static_phase_cloud_matched_scaled_1.5"),
        ("axis_horizontal", "static_center"),
        ("axis_vertical", "static_center"),
    ]
    sub = sub[
        [
            (row.condition, row.baseline_condition) in keep_pairs
            for row in sub.itertuples(index=False)
        ]
    ]
    if sub.empty:
        print("No selected threshold contrast rows found.")
        return None
    labels = [
        f"{condition_label(row.condition)}\nvs {condition_label(row.baseline_condition)}"
        for row in sub.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(sub)), 4.0), dpi=140)
    x = np.arange(len(sub))
    ax.bar(x, sub["mean_threshold_ratio"], color=[COLORS.get(c, "#777777") for c in sub["condition"]], alpha=0.9)
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("mean threshold ratio")
    ax.set_title(f"Trace-paired threshold proxy contrasts, fd={fd_step:g} arcmin")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


fig = plot_threshold_contrasts(contrast_df, fd_step=plot_step) if plot_step is not None else None
fig

# %% [markdown]
# ## Cumulative Fisher from rate caches
#
# The summary tables give final values. The cache files let us reconstruct
# curves over time for selected conditions. This cell recomputes only the chosen
# conditions, so it stays light enough for interactive explanation.
#
# **What to look for**: conditions with steeper slopes accumulate Vernier
# information faster over time. A curve that plateaus early means most of the
# signal is extracted in the first few frames. Shaded regions are ±1 SEM across
# traces — narrow bands mean the result is consistent across trials.

# %%
def cache_path_for(run_dir: Path, condition: str, fd_step: float) -> Path | None:
    pattern = f"rates_{condition}_fd{float(fd_step):.4f}arcmin.npz"
    path = run_dir / "cache" / pattern
    if path.exists():
        return path
    matches = sorted((run_dir / "cache").glob(f"rates_{condition}_fd*arcmin.npz"))
    if not matches:
        return None
    steps = []
    for match in matches:
        found = re.search(r"_fd([0-9.]+)arcmin", match.name)
        steps.append(float(found.group(1)) if found else math.inf)
    return matches[int(np.argmin(np.abs(np.asarray(steps) - float(fd_step))))]


def load_rate_cache_trials(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], float, str]:
    with np.load(path, allow_pickle=True) as npz:
        plus = np.asarray(npz["plus"], dtype=np.float32)
        minus = np.asarray(npz["minus"], dtype=np.float32)
        lengths = np.asarray(npz["lengths"], dtype=np.int32)
        fd_step = float(np.asarray(npz["fd_step_arcmin"])[0])
        condition = str(npz["condition"][0])
    plus_trials = [plus[i, : int(lengths[i])] for i in range(plus.shape[0])]
    minus_trials = [minus[i, : int(lengths[i])] for i in range(minus.shape[0])]
    return plus_trials, minus_trials, fd_step, condition


def cumulative_pose_aware_curves(
    plus_trials: list[np.ndarray],
    minus_trials: list[np.ndarray],
    *,
    fd_step: float,
    bin_seconds: float = 1.0 / 120.0,
    phi: float = 1.0,
) -> np.ndarray:
    curves = []
    for plus, minus in zip(plus_trials, minus_trials, strict=True):
        t = min(plus.shape[0], minus.shape[0])
        result = poisson_fisher_counts(
            expected_counts(plus[:t], bin_seconds),
            expected_counts(minus[:t], bin_seconds),
            step_arcmin=fd_step,
            phi=phi,
        )
        curves.append(result.cumulative_fisher)
    t_min = min(curve.shape[0] for curve in curves)
    return np.stack([curve[:t_min] for curve in curves], axis=0)


def plot_cumulative_from_caches(
    run_dir: Path,
    *,
    conditions: list[str],
    fd_step: float,
) -> plt.Figure | None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    any_rows = False
    for condition in conditions:
        path = cache_path_for(run_dir, condition, fd_step)
        if path is None:
            continue
        plus_trials, minus_trials, actual_step, _condition = load_rate_cache_trials(path)
        curves = cumulative_pose_aware_curves(plus_trials, minus_trials, fd_step=actual_step)
        x = np.arange(curves.shape[1])
        mean = np.nanmean(curves, axis=0)
        sem = np.nanstd(curves, axis=0, ddof=1) / max(math.sqrt(curves.shape[0]), 1.0)
        ax.plot(x, mean, label=condition_label(condition), color=COLORS.get(condition), linewidth=2.0)
        ax.fill_between(x, mean - sem, mean + sem, color=COLORS.get(condition), alpha=0.15, linewidth=0)
        any_rows = True
    if not any_rows:
        plt.close(fig)
        print("No matching cache files found.")
        return None
    ax.set_xlabel("time bin")
    ax.set_ylabel("cumulative Fisher")
    ax.set_title(f"Known-trace cumulative Fisher from caches, fd={fd_step:g} arcmin")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


fig = (
    plot_cumulative_from_caches(
        RUN_DIR,
        conditions=["static_center", "real_fem", "scaled_real_0.5", "axis_horizontal", "axis_vertical"],
        fd_step=plot_step,
    )
    if RUN_DIR is not None and plot_step is not None
    else None
)
fig

# %% [markdown]
# ## Hidden-trace diagonal curve from caches
#
# This recomputes the diagonal hidden-trace curve for a selected condition. The
# difference from the known-trace curve is the extra pose-marginal covariance in
# the denominator.
#
# **What to look for**: the gap between the two curves quantifies the cost of
# not knowing the eye position. A large gap means pose uncertainty is expensive
# for that condition. If the curves nearly overlap, the movement-induced variance
# is small relative to count noise and hiding pose barely hurts.

# %%
def plot_pose_aware_blind_curves_for_condition(
    run_dir: Path,
    *,
    condition: str,
    fd_step: float,
) -> plt.Figure | None:
    path = cache_path_for(run_dir, condition, fd_step)
    if path is None:
        print(f"No cache for {condition}, fd={fd_step}")
        return None
    plus_trials, minus_trials, actual_step, _condition = load_rate_cache_trials(path)
    aware = cumulative_pose_aware_curves(plus_trials, minus_trials, fd_step=actual_step)
    blind = pose_blind_diagonal_fisher(
        plus_trials,
        minus_trials,
        step_arcmin=actual_step,
        bin_seconds=1.0 / 120.0,
        phi=1.0,
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    x = np.arange(aware.shape[1])
    mean = np.nanmean(aware, axis=0)
    sem = np.nanstd(aware, axis=0, ddof=1) / max(math.sqrt(aware.shape[0]), 1.0)
    ax.plot(x, mean, label="known-trace mean", color="#4c78a8", linewidth=2.0)
    ax.fill_between(x, mean - sem, mean + sem, color="#4c78a8", alpha=0.15, linewidth=0)
    ax.plot(
        np.arange(len(blind["cumulative_fisher"])),
        blind["cumulative_fisher"],
        label="hidden-trace diagonal",
        color="#f58518",
        linewidth=2.0,
    )
    ax.set_xlabel("time bin")
    ax.set_ylabel("cumulative Fisher")
    ax.set_title(f"{condition_label(condition)}: known-trace vs hidden-trace")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


fig = (
    plot_pose_aware_blind_curves_for_condition(RUN_DIR, condition="real_fem", fd_step=plot_step)
    if RUN_DIR is not None and plot_step is not None
    else None
)
fig

# %% [markdown]
# ## Full-covariance integrity check
#
# The attached lineage note highlights a key provenance question: was a contrast
# evaluated under a full-covariance constrained metric, or only under diagonal
# approximations?
#
# Use this cell to see which conditions have a full-covariance row in the loaded
# run. If only `pose_blind_full_cov_optimal_unit_subset` appears, the row is a
# subset diagnostic because the runner capped the number of units.

# %%
if rel_df.empty:
    print("No reliability table loaded.")
else:
    full_cov_rows = rel_df[rel_df["readout"].astype(str).str.contains("full_cov", na=False)].copy()
    if full_cov_rows.empty:
        print("No full-covariance hidden-trace rows in this run.")
        print("To generate them, rerun or recompute with --run-full-cov-pose-blind.")
    else:
        cols = [
            "readout",
            "condition",
            "fd_step_arcmin",
            "mean_final_fisher",
            "cov_shrinkage",
            "unit_subset",
            "n_units_original",
            "n_units_used",
        ]
        show_table(full_cov_rows[cols].sort_values(["fd_step_arcmin", "condition"]), n=30)

# %% [markdown]
# ## Compact-aware controls
#
# These rows ask whether hidden-trace performance changes when suspected nuisance
# subspaces are projected out or precision is discounted. This is a diagnostic,
# not a claim that the brain performs exactly this projection.

# %%
def plot_compact_alpha(df: pd.DataFrame, *, fd_step: float, condition: str, k: int = 2) -> plt.Figure | None:
    if df.empty:
        print("No reliability dataframe available.")
        return None
    sub = df[
        np.isclose(pd.to_numeric(df["fd_step_arcmin"], errors="coerce"), float(fd_step))
        & (df["condition"] == condition)
        & (pd.to_numeric(df["compact_k"], errors="coerce") == int(k))
        & (df["compact_mode"] == "soft_discount")
    ].copy()
    if sub.empty:
        print("No compact soft-discount rows for this condition/k.")
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=140)
    for source, rows in sub.groupby("subspace_source"):
        rows = rows.sort_values("compact_alpha")
        ax.plot(rows["compact_alpha"], rows["mean_final_fisher"], marker="o", label=str(source))
    ax.set_xlabel("alpha: retained precision in nuisance subspace")
    ax.set_ylabel("mean final Fisher")
    ax.set_title(f"Compact-aware soft discount, {condition_label(condition)}, k={k}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


fig = plot_compact_alpha(rel_df, fd_step=plot_step, condition="real_fem", k=2) if plot_step is not None else None
fig

# %% [markdown]
# ## Vernier joint observer diagnostic
#
# The joint-geometry observer is a later diagnostic. It changes the question
# from "how much local Fisher exists under known or hidden pose?" to "can a
# simple observer recover the Vernier sign while marginalizing or fitting eye
# trajectory state?"
#
# For the second pass, the working expectation should be that Vernier may contain
# enough image structure to help: the bar edges, their temporal displacement, and
# V1 response geometry might constrain the latent trajectory even though the
# stimulus is minimal. The test is whether that contribution survives stricter
# observer settings, not whether an optimistic catalog can be made to work once.
#
# Any high joint-observer accuracy here should be interpreted relative to the
# trajectory catalog, prior, likelihood scale, and evaluation setting used by
# the cache. A convincing Vernier joint-decoding result should survive
# leave-one-out or cross-prior trajectory catalogs and should be accompanied by
# posterior diagnostics such as true-trace rank and posterior effective count.

# %%
if JOINT_RUN_DIR is None:
    joint_summary = pd.DataFrame()
    print("No joint geometry run directory found.")
else:
    joint_summary = read_csv_optional(JOINT_RUN_DIR / "joint_geometry_observer_summary.csv")
    print(f"JOINT_RUN_DIR: {JOINT_RUN_DIR}")
    print(f"joint summary rows: {len(joint_summary)}")
    if not joint_summary.empty:
        show_table(joint_summary.head(12))

# %%
def plot_joint_accuracy(df: pd.DataFrame, *, fd_step: float) -> plt.Figure | None:
    if df.empty:
        print("No joint summary available.")
        return None
    sub = df[
        np.isclose(pd.to_numeric(df["fd_step_arcmin"], errors="coerce"), float(fd_step))
        & (df["joint_control"] == "correct_chart")
    ].copy()
    if sub.empty:
        print("No correct-chart joint rows for this fd step.")
        return None
    sub = sub.sort_values("condition")
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(sub)), 4.0), dpi=140)
    x = np.arange(len(sub))
    width = 0.25
    ax.bar(x - width, sub["zero_accuracy"], width=width, label="zero eye", color="#999999")
    ax.bar(x, sub["accuracy"], width=width, label="joint", color="#4c78a8")
    ax.bar(x + width, sub["known_accuracy"], width=width, label="known-trace", color="#54a24b")
    ax.axhline(0.5, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(c) for c in sub["condition"]], rotation=35, ha="right")
    ax.set_ylabel("classification accuracy")
    ax.set_title(f"Joint geometry observer, fd={fd_step:g} arcmin")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


joint_step = nearest_available_step(joint_summary, FD_STEP_TO_PLOT)
fig = plot_joint_accuracy(joint_summary, fd_step=joint_step) if joint_step is not None else None
fig

# %% [markdown]
# ## Second-pass Vernier joint-decoding tests
#
# The tests below import the strongest lessons from the backimage observer work
# while keeping Vernier as the stimulus. They are designed to give Vernier the
# chance to succeed, while separating real image-structure contribution from
# catalog leakage or overconfident scoring.
#
# ```text
# 1. Include-self trajectory table:
#    optimistic empirical-catalog upper bound.
#
# 2. Leave-one-out trajectory table:
#    true trace removed from the nuisance catalog.
#
# 3. Cross-prior catalog:
#    score one motion condition using a different trajectory prior condition.
#
# 4. Likelihood-scale / posterior-temperature calibration:
#    choose the scale on heldout traces, then report heldout Vernier accuracy,
#    margin closure, true-trace rank, and posterior N_eff.
#
# 5. Prior-family checks:
#    uniform empirical catalog, Brownian prior, AR(1) prior, and known-start
#    prior. Known-start is less strict, but useful for asking whether a small
#    amount of extraretinal pose information would rescue the observer.
#
# 6. Dense anisotropic diffusion:
#    D_across in {0, .125, .25, .5, 1, 2, 3} with D_along=1, and the matched
#    along sweep, using effective RMS/clipping audits.
# ```
#
# Evidence for meaningful Vernier image-structure contribution would look like:
# hidden-trace or joint performance above zero-eye after leave-one-out/cross-prior
# controls, posterior mass that is more concentrated on useful trajectory states
# than chance, and a stable dependence on across-contour diffusion after matching
# the trace scale and temporal waveform.

# %% [markdown]
# ## Exact trajectory-table observer diagnostic
#
# The trajectory-table observer uses cached exact responses for an empirical
# trajectory catalog and evaluates a Vernier likelihood ratio after
# marginalizing over trajectory identity.
#
# This is closest to the "pose hidden as nuisance" story, but it is a different
# estimator from the local Fisher calculations above. Keep its interpretation
# separate from Fisher, and use it to test whether the Vernier movie itself
# supplies enough structure for trajectory marginalization.
#
# Any high trajectory-table accuracy here should be interpreted relative to the
# exact cached response table and its `include_self` / `leave_one_out` setting;
# it becomes much more meaningful if it survives leave-one-out, cross-prior
# catalogs, likelihood-scale calibration, and prior-family checks.

# %%
if TRAJECTORY_TABLE_RUN_DIR is None:
    table_summary = pd.DataFrame()
    print("No trajectory-table observer directory found.")
else:
    table_summary = read_csv_optional(TRAJECTORY_TABLE_RUN_DIR / "trajectory_table_observer_summary.csv")
    print(f"TRAJECTORY_TABLE_RUN_DIR: {TRAJECTORY_TABLE_RUN_DIR}")
    print(f"trajectory-table summary rows: {len(table_summary)}")
    if not table_summary.empty:
        show_table(table_summary.head(12))

# %%
def plot_trajectory_table_accuracy(df: pd.DataFrame, *, fd_step: float) -> plt.Figure | None:
    if df.empty:
        print("No trajectory-table summary available.")
        return None
    sub = df[np.isclose(pd.to_numeric(df["fd_step_arcmin"], errors="coerce"), float(fd_step))].copy()
    if sub.empty:
        print("No trajectory-table rows for this fd step.")
        return None
    sub = sub.sort_values("condition")
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(sub)), 4.0), dpi=140)
    x = np.arange(len(sub))
    width = 0.25
    ax.bar(x - width, sub["zero_accuracy"], width=width, label="zero eye", color="#999999")
    ax.bar(x, sub["joint_accuracy"], width=width, label="marginal trajectory", color="#4c78a8")
    ax.bar(x + width, sub["known_accuracy"], width=width, label="known-trace", color="#54a24b")
    ax.axhline(0.5, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(c) for c in sub["condition"]], rotation=35, ha="right")
    ax.set_ylabel("classification accuracy")
    setting = trajectory_table_setting_label(sub)
    ax.set_title(f"Trajectory-table observer, fd={fd_step:g} arcmin" + (f"\n{setting}" if setting else ""))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


table_step = nearest_available_step(table_summary, FD_STEP_TO_PLOT)
fig = plot_trajectory_table_accuracy(table_summary, fd_step=table_step) if table_step is not None else None
fig

# %% [markdown]
# ## Noisy retinal-trajectory observer diagnostic
#
# This include-self pilot is an endpoint sanity check for a finite-precision
# trajectory cue. The observer receives a trajectory cue with uncertainty
# `sigma_e` and computes a Vernier likelihood ratio after marginalizing over the
# empirical trajectory table:
#
# ```text
# log p(r | s, hat_tau_i)
#   = log sum_j p(r | s, tau_j) p(tau_j | hat_tau_i).
# ```
#
# Because the true trajectory is retained in this include-self catalog, the
# endpoints are teaching anchors:
#
# - `sigma_e = 0`: the prior collapses to the anchored/true trajectory, so this
#   recovers the known-trajectory endpoint when self is included.
# - `sigma_e = inf`: the prior is uniform over the retained catalog, so this
#   recovers the unknown-trajectory empirical marginal.
#
# In the pilot below, the trajectory catalog is the saved RR100 scaled-real-trace
# grid. No synthetic traces are generated in this analysis. In a held-out catalog
# where the true trajectory is excluded, `sigma_e = 0` no longer recovers the
# known-trajectory endpoint.

# %%
if NOISY_TRAJECTORY_RUN_DIR is None:
    noisy_summary = pd.DataFrame()
    print("No noisy retinal-trajectory observer directory found.")
else:
    noisy_summary = read_csv_optional(
        NOISY_TRAJECTORY_RUN_DIR / "rr100_noisy_trajectory_observer_summary.csv"
    )
    print(f"NOISY_TRAJECTORY_RUN_DIR: {NOISY_TRAJECTORY_RUN_DIR}")
    print(f"noisy trajectory summary rows: {len(noisy_summary)}")
    if not noisy_summary.empty:
        show_table(noisy_summary.head(12))

# %%
def _sigma_plot_x(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = np.empty_like(vals, dtype=float)
    for idx, value in enumerate(vals):
        if value == 0:
            x[idx] = -2.0
        elif np.isfinite(value):
            x[idx] = math.log10(max(float(value), 1e-12))
        else:
            x[idx] = 1.0
    return x


def _sigma_plot_ticks() -> tuple[list[float], list[str]]:
    return (
        [-2.0, math.log10(0.125), math.log10(0.25), math.log10(0.5), 0.0, math.log10(2.0), 1.0],
        ["0", "0.125", "0.25", "0.5", "1", "2", "inf"],
    )


def plot_noisy_trajectory_sigma_sweep(df: pd.DataFrame) -> plt.Figure | None:
    if df.empty:
        print("No noisy trajectory summary available.")
        return None
    conditions = [
        "real_aniso_across_0_along_1",
        "real_aniso_across_0p25_along_1",
        "real_aniso_across_1_along_1",
        "real_aniso_across_2_along_1",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=140, constrained_layout=True)
    for condition in conditions:
        sub = df[df["condition"].eq(condition)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("trajectory_weight_sigma_arcmin")
        x = _sigma_plot_x(sub["trajectory_weight_sigma_arcmin"])
        label = str(sub.iloc[0].get("label", condition_label(condition)))
        axes[0].plot(x, sub["joint_accuracy"], marker="o", label=label)
        axes[1].plot(x, sub["mean_trajectory_weight_neff"], marker="o", label=label)
    ticks, labels = _sigma_plot_ticks()
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel("trajectory cue sigma_e (arcmin)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("Vernier sign accuracy")
    axes[0].set_title("Noisy trajectory marginal")
    axes[1].set_ylabel("prior effective trajectory count")
    axes[1].set_title("How many traces does the cue allow?")
    axes[1].legend(frameon=False, fontsize=8)
    return fig


fig = plot_noisy_trajectory_sigma_sweep(noisy_summary)
fig

# %% [markdown]
# The important readout is the pair of curves: accuracy says whether Vernier sign
# survives marginalization, and `N_eff` says how diffuse the trajectory prior has
# become. The margin-closure column is still useful as a continuity diagnostic,
# but it is not the lead plot because it can become unstable when the zero-known
# denominator is small.

# %%
def _static_baseline_mask(df: pd.DataFrame) -> pd.Series:
    if "is_static_baseline" in df:
        return df["is_static_baseline"].astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})
    if "condition" in df:
        return df["condition"].astype(str).eq("static_center")
    return pd.Series(False, index=df.index)


def _display_sigma_value(value: float) -> str:
    return "inf" if not np.isfinite(float(value)) else f"{float(value):g}"


def plot_noisy_trajectory_static_relative_heatmaps(df: pd.DataFrame) -> plt.Figure | None:
    if df.empty:
        print("No noisy trajectory summary available.")
        return None
    required = {"trajectory_weight_sigma_arcmin", "across_scale", "along_scale", "mean_joint_score"}
    if not required.issubset(df.columns):
        print("Noisy trajectory summary lacks static-relative heatmap columns.")
        return None
    static = df[_static_baseline_mask(df)].copy()
    grid = df[~_static_baseline_mask(df)].copy()
    if static.empty or grid.empty:
        print("Need both static and scale-grid rows for static-relative heatmaps.")
        return None
    sigmas = [0.0, 0.25, 0.5, 1.0, 2.0, float("inf")]
    scales = sorted(pd.to_numeric(grid["across_scale"], errors="coerce").dropna().unique())
    available_sigmas = sorted(pd.to_numeric(df["trajectory_weight_sigma_arcmin"], errors="coerce").dropna().unique())
    sigmas = [sigma for sigma in sigmas if any(np.isclose(available_sigmas, sigma)) or not np.isfinite(sigma)]
    if not scales or not sigmas:
        return None

    fig, axes = plt.subplots(
        1,
        len(sigmas),
        figsize=(3.0 * len(sigmas), 3.2),
        dpi=140,
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(-1)
    im = None
    for ax, sigma in zip(axes_arr, sigmas, strict=True):
        if np.isfinite(sigma):
            static_sub = static[np.isclose(static["trajectory_weight_sigma_arcmin"], sigma)]
            sub = grid[np.isclose(grid["trajectory_weight_sigma_arcmin"], sigma)]
        else:
            static_sub = static[np.isinf(static["trajectory_weight_sigma_arcmin"])]
            sub = grid[np.isinf(grid["trajectory_weight_sigma_arcmin"])]
        values = np.full((len(scales), len(scales)), np.nan, dtype=float)
        if not static_sub.empty:
            static_score = float(static_sub.iloc[0]["mean_joint_score"])
            for y, along in enumerate(scales):
                for x, across in enumerate(scales):
                    cell = sub[
                        np.isclose(sub["across_scale"], across)
                        & np.isclose(sub["along_scale"], along)
                    ]
                    if not cell.empty and abs(static_score) > 1e-12:
                        values[y, x] = float(cell.iloc[0]["mean_joint_score"]) / static_score
        im = ax.imshow(values, origin="lower", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.5)
        ax.set_title(f"sigma={_display_sigma_value(sigma)}")
        ax.set_xticks(np.arange(len(scales)))
        ax.set_yticks(np.arange(len(scales)))
        ax.set_xticklabels([f"{scale:g}" for scale in scales], rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels([f"{scale:g}" for scale in scales], fontsize=6)
        ax.set_xlabel("across scale")
        for yy in range(values.shape[0]):
            for xx in range(values.shape[1]):
                if np.isfinite(values[yy, xx]):
                    ax.text(xx, yy, f"{values[yy, xx]:.2g}", ha="center", va="center", fontsize=4.8, color="white")
    axes_arr[0].set_ylabel("along scale")
    if im is not None:
        fig.colorbar(im, ax=axes_arr.tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("Mean Vernier LLR margin relative to static", y=1.04)
    return fig


fig = plot_noisy_trajectory_static_relative_heatmaps(noisy_summary)
fig

# %%
if NOISY_TRAJECTORY_RUN_DIR is not None:
    show_image_if_exists(
        NOISY_TRAJECTORY_RUN_DIR / "rr100_noisy_trajectory_observer_static_relative_heatmaps.png"
    )

# %% [markdown]
# ## Held-out empirical trajectory observer diagnostic
#
# The include-self noisy-trajectory pilot is useful for endpoint checks, but it
# is still a small catalog analysis. The stricter stepping-stone analysis uses
# more real traces and splits trajectory identities into disjoint observation
# and nuisance-prior sets.
#
# In this version:
#
# - `known_*` columns are the pose-aware endpoint: the observer is scored against
#   the same trajectory that generated the response.
# - finite `sigma_e` curves are trajectory-cue-conditioned local catalog
#   marginals over held-out nuisance trajectories.
# - `sigma_e = inf` is the pose-unaware empirical Monte Carlo marginal over
#   held-out nuisance trajectories.
#
# Because the true observation trajectory is deliberately absent from the prior
# set, `sigma_e = 0` is a nearest-held-out-trajectory limit, not the pose-aware
# endpoint. So this sigma sweep interpolates from nearest held-out catalog match
# to uniform held-out catalog marginal, not from pose-aware to pose-unaware.
# This is why the plots overlay pose-aware separately.

# %%
if HELDOUT_TRAJECTORY_RUN_DIR is None:
    heldout_summary = pd.DataFrame()
    print("No held-out trajectory observer directory found.")
else:
    heldout_summary = read_csv_optional(
        HELDOUT_TRAJECTORY_RUN_DIR / "rr100_heldout_trajectory_observer_summary.csv"
    )
    print(f"HELDOUT_TRAJECTORY_RUN_DIR: {HELDOUT_TRAJECTORY_RUN_DIR}")
    print(f"held-out trajectory summary rows: {len(heldout_summary)}")
    if not heldout_summary.empty:
        show_table(heldout_summary.head(18))

# %%
if HELDOUT_TRAJECTORY_RUN_DIR is not None:
    show_image_if_exists(
        HELDOUT_TRAJECTORY_RUN_DIR / "rr100_heldout_along1_across_by_sigma.png"
    )
    show_image_if_exists(
        HELDOUT_TRAJECTORY_RUN_DIR / "rr100_heldout_uniform_k_convergence.png"
    )

# %% [markdown]
# ## Held-out catalog density diagnostic
#
# Before interpreting the held-out catalog observer, check whether the catalog is
# dense enough in response space. The diagnostic compares same-sign trajectory
# mismatch to same-trajectory Vernier sign distance:
#
# ```text
# D_traj = ||mu(s_i, tau_i) - mu(s_i, tau_NN(i))||^2_Sigma^-1
# D_sign = ||mu(+, tau_i) - mu(-, tau_i)||^2_Sigma^-1
# ```
#
# If `D_traj >> D_sign`, the nearest held-out trajectory is already a much worse
# response match than the Vernier sign difference. In that regime a finite
# catalog marginal should fail unless the catalog is densified or the likelihood
# includes an interpolation/noise term.

# %%
if CATALOG_MISMATCH_RUN_DIR is None:
    catalog_mismatch_summary = pd.DataFrame()
    print("No held-out catalog mismatch diagnostic directory found.")
else:
    catalog_mismatch_summary = read_csv_optional(
        CATALOG_MISMATCH_RUN_DIR / "rr100_catalog_mismatch_summary.csv"
    )
    print(f"CATALOG_MISMATCH_RUN_DIR: {CATALOG_MISMATCH_RUN_DIR}")
    print(f"catalog mismatch summary rows: {len(catalog_mismatch_summary)}")
    if not catalog_mismatch_summary.empty:
        show_table(catalog_mismatch_summary.head(12))

# %%
if CATALOG_MISMATCH_RUN_DIR is not None:
    show_image_if_exists(
        CATALOG_MISMATCH_RUN_DIR / "rr100_catalog_mismatch_diagnostic.png"
    )

# %% [markdown]
# ## Vernier tutorial landing
#
# The clean tutorial result is the known-versus-hidden trajectory Fisher story.
# When the eye trajectory is supplied, reduced across-contour motion can increase
# Vernier information. When the eye trajectory is hidden, trajectory-induced
# response variance overwhelms the small Vernier offset signal. This is the
# useful Vernier lesson.
#
# The leave-one-trajectory-out catalog observer is our best finite-catalog
# attempt at a generative trajectory marginal. It is worth including as a
# cautionary diagnostic, but not as the main solution. It fails for an
# interpretable reason: nearest held-out trajectory mismatch is larger than the
# Vernier sign signal. Therefore the sigma axis in that analysis is a catalog
# weighting scale, not a calibrated bridge from pose-aware to pose-unaware.
#
# For the manuscript/tutorial, use this as the stopping point:
#
# 1. Report pose-aware and pose-unaware Fisher for the real-trace scale sweep.
# 2. Show that the known-trace benefit comes mainly from reducing across-contour
#    motion, not from adding along-contour motion.
# 3. Include the leave-one-out catalog observer only as a principled negative
#    control showing why sparse whole-trajectory lookup is inadequate.
# 4. Leave robust joint inference to the feature-decoder / natural-image branch.

# %% [markdown]
# ## Bridge to natural-image active sensing
#
# Vernier is deliberately minimal. That is why it is so good for teaching pose
# confusion: offset and eye position can mimic each other. Minimal does not mean
# useless, though. The second-pass tests above ask whether bar-edge structure and
# temporal continuity are enough to support meaningful hidden-trace decoding.
#
# Natural-image stimuli provide a richer comparison class:
#
# ```text
# Does image structure plus V1 response geometry let an observer recover useful
# information while marginalizing over latent eye trajectory?
# ```
#
# In that setting, local edges, textures, and image-specific response patterns
# can also constrain `tau`, often with more redundancy than two Vernier bars.
# Vernier supplies the clean pose-confusion case; natural images test whether the
# same observer logic scales to richer image structure.

# %% [markdown]
# ## Supplement: spatial organization, not Vernier discriminability
#
# Fisher information (above) asks about Vernier discriminability — it is derived
# from the response **derivative** with respect to offset. SSI asks a different
# question: how much does the V1 population response vary across the spatial
# positions of the output feature map?
#
# **SSI formula** (bits per spike, per unit):
#
# ```text
# gain(u, x, y)  = rate(u, x, y) / mean_rate(u)       [spatial gain map]
# unit_bits(u)   = mean_xy[ gain * log2(gain) ]        [spatial selectivity]
# pop_bits       = sum_u [ w(u) * unit_bits(u) ]       [rate-weighted mean]
#   where w(u) = mean_rate(u) / sum_u mean_rate(u)
# ```
#
# A unit with uniform rate across the output feature map has SSI = 0.
# A unit that fires strongly at one spatial position and nowhere else has high SSI.
#
# **In the Vernier context**: SSI measures how concentrated the V1 response is
# at particular spatial positions in the output feature map, under each motion
# condition. Eye movement sweeps the Vernier edge across neurons; SSI changes if
# that sweep concentrates or disperses responses across the readout grid.
#
# **Important distinction from Fisher**: SSI is not penalised by nuisance variance
# and does not assess Vernier discriminability. A high-SSI condition might still
# have low Fisher if the spatially-concentrated response is not aligned with the
# Vernier offset axis.
#
# Set `RUN_SSI_FROM_MODEL = True` in the configuration cell to enable this section.
# It requires loading the digital twin and re-running selected conditions without
# spatial collapse.

# %%
# SSI helper functions (inline, no validate-script import needed)
def _ssi_single_frame(rate_maps: np.ndarray, eps: float = 1e-8) -> dict:
    """Spatial Spiking Information from a (unit, H, W) rate map."""
    y = np.asarray(rate_maps, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"Expected (unit, H, W), got {y.shape}")
    flat = y.reshape(y.shape[0], -1)
    rbar = flat.mean(axis=1)
    gain = flat / (rbar[:, None] + eps)
    unit_bits = np.mean(gain * np.log2(gain + eps), axis=1)
    weights = rbar / max(float(rbar.sum()), eps)
    return {
        "unit_bits_per_spike": unit_bits,
        "unit_mean_rate": rbar,
        "population_bits_per_spike": float(np.sum(weights * unit_bits)),
    }


def _ssi_timecourse(rate_movie: np.ndarray) -> np.ndarray:
    """SSI (pop bits/spike) at each time step; expects (T, C, H, W)."""
    y = np.asarray(rate_movie, dtype=np.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected (T, C, H, W), got {y.shape}")
    return np.array([_ssi_single_frame(y[t])["population_bits_per_spike"] for t in range(y.shape[0])], dtype=np.float32)


# %%
_ssi_results: dict[str, np.ndarray] = {}
_ssi_final_vals: dict[str, float] = {}
_ssi_pop = None
_ssi_model = None
_ssi_readout = None
_ssi_base_trace = None
_ssi_device_str = ""

if not RUN_SSI_FROM_MODEL:
    print("SSI section disabled (RUN_SSI_FROM_MODEL = False).")
    print("Set it to True in the configuration cell to run the model and compute SSI.")
else:
    # ---- Load model --------------------------------------------------------
    print("Loading model and readout...")
    from declan.vernier_active_sensing.forward import build_vernier_movie, load_model_and_readout
    from scripts.temporal_decoding.rate_computation import compute_trial_rates

    _ssi_device = SSI_DEVICE
    _ssi_model, _ssi_readout = load_model_and_readout(device=_ssi_device)
    _ssi_device_str = str(next(_ssi_model.model.parameters()).device)
    print(f"Model on: {_ssi_device_str}")

    # ---- Population view for SSI -------------------------------------------
    if POPULATION_MODE == "reduced" and _population_view is not None:
        from declan.redundancy_resolved_v1_population import apply_population_view as _apply_pop
        _ssi_pop = _population_view
        print(f"SSI population: {_pop_label} ({_ssi_pop.n_units} units)")
    else:
        _ssi_pop = None
        print("SSI population: full 756 channels")

    # ---- Get the demo trace for this SSI run --------------------------------
    _ssi_base_trace = valid_trace(trace_set, SSI_TRACE_IDX, max_frames=SSI_MAX_FRAMES)
    print(f"Demo trace shape: {_ssi_base_trace.shape}")

    # ---- Compute spatial rate maps per condition ----------------------------
    _rng_ssi = np.random.default_rng(SEED)
    _ssi_results: dict[str, np.ndarray] = {}

    for _cond in SSI_CONDITIONS:
        _trace, _ = condition_trace(
            _ssi_base_trace,
            condition=_cond,
            trace_set=trace_set,
            rng=_rng_ssi,
        )
        _stim = build_vernier_movie(
            canonical_spec,
            _trace[:SSI_MAX_FRAMES],
            device=_ssi_device_str,
        )
        # return_spatial=True gives (T, N, H, W) — needed for SSI
        _spatial = compute_trial_rates(
            _ssi_model,
            _ssi_readout,
            _stim,
            batch_size=SSI_BATCH_SIZE,
            spatial_collapse="max",      # ignored when return_spatial=True
            return_spatial=True,
        )
        # Apply population pooling if using redundancy twin
        if _ssi_pop is not None:
            _spatial = _apply_pop(_spatial, _ssi_pop)

        _ssi_results[_cond] = _spatial.astype(np.float32)
        print(f"  {_cond}: spatial rate map shape {_spatial.shape}")

    print("SSI computation done.")

# %%
if RUN_SSI_FROM_MODEL and _ssi_results:
    # ---- SSI timecourses ---------------------------------------------------
    fig_ssi_tc, ax_ssi_tc = plt.subplots(figsize=(8.5, 4.0), dpi=140)
    for _cond, _spatial in _ssi_results.items():
        _tc = _ssi_timecourse(_spatial)
        _ssi_final_vals[_cond] = float(np.mean(_tc))
        ax_ssi_tc.plot(
            np.arange(len(_tc)) / 120.0 * 1000.0,
            _tc,
            label=condition_label(_cond),
            color=COLORS.get(_cond),
            linewidth=1.8,
        )
    ax_ssi_tc.set_xlabel("time (ms)")
    ax_ssi_tc.set_ylabel("population SSI (bits / spike)")
    ax_ssi_tc.set_title(
        f"Spatial Spiking Information timecourse — {_pop_label}\n"
        "(higher = response more concentrated in space at that moment)"
    )
    ax_ssi_tc.legend(frameon=False, fontsize=8)
    ax_ssi_tc.spines[["top", "right"]].set_visible(False)
    fig_ssi_tc.tight_layout()
    fig_ssi_tc

# %%
if RUN_SSI_FROM_MODEL and _ssi_results:
    # ---- Mean SSI bar chart ------------------------------------------------
    _conds_sorted = SSI_CONDITIONS
    _ssi_means = [_ssi_final_vals.get(c, float("nan")) for c in _conds_sorted]

    fig_ssi_bar, ax_ssi_bar = plt.subplots(figsize=(max(6, 0.8 * len(_conds_sorted)), 3.8), dpi=140)
    ax_ssi_bar.bar(
        np.arange(len(_conds_sorted)),
        _ssi_means,
        color=[COLORS.get(c, "#777777") for c in _conds_sorted],
        alpha=0.9,
    )
    ax_ssi_bar.set_xticks(np.arange(len(_conds_sorted)))
    ax_ssi_bar.set_xticklabels([condition_label(c) for c in _conds_sorted], rotation=35, ha="right")
    ax_ssi_bar.set_ylabel("mean SSI (bits / spike)")
    ax_ssi_bar.set_title(
        f"Mean Spatial Spiking Information by condition — {_pop_label}\n"
        "Note: SSI is not Vernier discriminability — see interpretation note below"
    )
    ax_ssi_bar.spines[["top", "right"]].set_visible(False)
    fig_ssi_bar.tight_layout()
    fig_ssi_bar

# %%
if RUN_SSI_FROM_MODEL and _ssi_results:
    # ---- Real-cache Fisher + freshly computed SSI comparison ---------------
    _fisher_ssi_rows = []
    for _cond in SSI_CONDITIONS:
        _fisher = metric_value(
            rel_df,
            readout="pose_aware_diagonal_poisson",
            condition=_cond,
            fd_step=FD_STEP_TO_PLOT,
            column="mean_final_fisher",
        )
        _fisher_hidden = metric_value(
            rel_df,
            readout="pose_blind_diagonal_count_plus_marginal",
            condition=_cond,
            fd_step=FD_STEP_TO_PLOT,
            column="mean_final_fisher",
        )
        _fisher_ssi_rows.append(
            {
                "condition": _cond,
                "known_eye_fisher": _fisher,
                "pose_hidden_fisher": _fisher_hidden,
                "mean_general_ssi_bits_per_spike": _ssi_final_vals.get(_cond, float("nan")),
            }
        )
    _real_fisher_ssi_df = pd.DataFrame(_fisher_ssi_rows)
    show_table(_real_fisher_ssi_df)

    fig_real_cmp, ax_real_cmp = plt.subplots(figsize=(5.2, 4.2), dpi=140)
    for _, _row in _real_fisher_ssi_df.iterrows():
        ax_real_cmp.scatter(
            _row["known_eye_fisher"],
            _row["mean_general_ssi_bits_per_spike"],
            s=55,
            color=COLORS.get(_row["condition"], "#777777"),
        )
        ax_real_cmp.text(
            _row["known_eye_fisher"],
            _row["mean_general_ssi_bits_per_spike"],
            " " + condition_label(_row["condition"]),
            fontsize=7,
            va="center",
        )
    ax_real_cmp.set_xlabel("cached known-trace Fisher")
    ax_real_cmp.set_ylabel("fresh general SSI (bits / spike)")
    ax_real_cmp.set_title("Real-cache audit: Fisher versus general SSI")
    ax_real_cmp.spines[["top", "right"]].set_visible(False)
    fig_real_cmp.tight_layout()
    fig_real_cmp

# %%
if RUN_SSI_FROM_MODEL and POPULATION_MODE == "reduced" and _population_view is not None:
    # ---- Full vs reduced SSI comparison ------------------------------------
    # Recompute SSI for the full population on the same spatial maps
    # by re-running the model without population pooling.
    print("Computing full-population SSI for comparison...")
    _ssi_full_means = {}
    _rng_ssi2 = np.random.default_rng(SEED)
    for _cond in SSI_CONDITIONS:
        _trace, _ = condition_trace(
            _ssi_base_trace,
            condition=_cond,
            trace_set=trace_set,
            rng=_rng_ssi2,
        )
        _stim = build_vernier_movie(
            canonical_spec,
            _trace[:SSI_MAX_FRAMES],
            device=_ssi_device_str,
        )
        _spatial_full = compute_trial_rates(
            _ssi_model,
            _ssi_readout,
            _stim,
            batch_size=SSI_BATCH_SIZE,
            return_spatial=True,
        ).astype(np.float32)
        _tc_full = _ssi_timecourse(_spatial_full)
        _ssi_full_means[_cond] = float(np.mean(_tc_full))
        print(f"  {_cond}: full mean SSI = {_ssi_full_means[_cond]:.4f}")

    _x = np.arange(len(_conds_sorted))
    _w = 0.38
    fig_ssi_cmp, ax_ssi_cmp = plt.subplots(figsize=(max(7, 0.9 * len(_conds_sorted)), 4.0), dpi=140)
    ax_ssi_cmp.bar(_x - _w / 2, [_ssi_full_means.get(c, float("nan")) for c in _conds_sorted],
                   width=_w, label="full 756", color="#4c78a8", alpha=0.85)
    ax_ssi_cmp.bar(_x + _w / 2, [_ssi_final_vals.get(c, float("nan")) for c in _conds_sorted],
                   width=_w, label=_pop_label, color="#f58518", alpha=0.85)
    ax_ssi_cmp.set_xticks(_x)
    ax_ssi_cmp.set_xticklabels([condition_label(c) for c in _conds_sorted], rotation=35, ha="right")
    ax_ssi_cmp.set_ylabel("mean SSI (bits / spike)")
    ax_ssi_cmp.set_title("SSI comparison: full vs redundancy-resolved population")
    ax_ssi_cmp.legend(frameon=False)
    ax_ssi_cmp.spines[["top", "right"]].set_visible(False)
    fig_ssi_cmp.tight_layout()
    fig_ssi_cmp

# %% [markdown]
# ## V1-RR100 SSI audit in the Vernier task
#
# This section tests the new redundancy-resolved population views on the Vernier
# SSI calculation. It computes one full 756-channel spatial response movie for
# each condition, caches that movie, and then applies several post-activation
# population views to the same movie:
#
# - full 756-channel canonical twin
# - RR100 movie-medoid
# - RR192 mean-pooled comparison set
#
# This keeps the comparison paired at the movie level. If the full spatial movie
# cache exists, these cells do not need to run the model again. Set
# `RUN_SSI_POPULATION_COMPARISON = True` in the configuration cell to enable.

# %%
def _slug_for_cache(value: object, max_len: int = 120) -> str:
    text = str(value)
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return (slug or "unnamed")[:max_len]


def _population_summary_label(spec: dict[str, Any], view: Any) -> str:
    label = str(spec.get("label", "population"))
    if getattr(view, "membership", None) is None:
        return label
    return f"{label} ({int(view.n_units)} reps)"


def _load_ssi_population_views() -> list[dict[str, Any]]:
    from declan.redundancy_resolved_v1_population import full_population_view, load_population_view

    views: list[dict[str, Any]] = []
    for spec in SSI_POPULATION_COMPARISON_SPECS:
        version = spec.get("version")
        if version is None:
            view = full_population_view(756, name=str(spec.get("label", "full 756")))
        else:
            view = load_population_view(version_name=str(version))
        views.append(
            {
                "key": str(spec["key"]),
                "label": _population_summary_label(spec, view),
                "version": view.name,
                "view": view,
            }
        )
    return views


def _ssi_frame_total_rate(rate_movie: np.ndarray) -> np.ndarray:
    y = np.asarray(rate_movie, dtype=np.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected (T, C, H, W), got {y.shape}")
    return y.reshape(y.shape[0], y.shape[1], -1).mean(axis=2).sum(axis=1)


_ssi_popcomp_views: list[dict[str, Any]] = []
_ssi_popcomp_summary = pd.DataFrame()
_ssi_popcomp_timecourse_df = pd.DataFrame()

if not RUN_SSI_POPULATION_COMPARISON:
    print("V1-RR SSI population comparison disabled (RUN_SSI_POPULATION_COMPARISON = False).")
else:
    _ssi_popcomp_views = _load_ssi_population_views()
    _view_rows = []
    for row in _ssi_popcomp_views:
        view = row["view"]
        membership = getattr(view, "membership", None)
        cluster_membership = getattr(view, "cluster_membership", None)
        if cluster_membership is None:
            group_sizes = np.ones(int(view.n_units), dtype=int)
        else:
            group_sizes = (np.asarray(cluster_membership) > 0).sum(axis=1)
        _view_rows.append(
            {
                "population_key": row["key"],
                "label": row["label"],
                "version": row["version"],
                "n_units": int(view.n_units),
                "pooling": "identity" if membership is None else str(view.meta.get("pooling_mode", "unknown")),
                "n_singletons": int((group_sizes == 1).sum()),
                "n_groups": int((group_sizes > 1).sum()),
                "largest_group": int(group_sizes.max()),
            }
        )
    _ssi_popcomp_view_table = pd.DataFrame(_view_rows)
    show_table(_ssi_popcomp_view_table)

# %%
if RUN_SSI_POPULATION_COMPARISON:
    from declan.redundancy_resolved_v1_population import apply_population_view

    _summary_rows: list[dict[str, Any]] = []
    _timecourse_rows: list[dict[str, Any]] = []
    _model_for_ssi = None
    _readout_for_ssi = None
    _ssi_device_str2 = ""
    _base_trace_for_ssi = valid_trace(trace_set, SSI_TRACE_IDX, max_frames=SSI_MAX_FRAMES)

    def _ensure_vernier_model_loaded() -> tuple[Any, Any, str]:
        global _model_for_ssi, _readout_for_ssi, _ssi_device_str2
        if _model_for_ssi is None or _readout_for_ssi is None:
            print("Loading model and readout for missing SSI spatial-map caches...")
            from declan.vernier_active_sensing.forward import load_model_and_readout

            _model_for_ssi, _readout_for_ssi = load_model_and_readout(device=SSI_DEVICE)
            _ssi_device_str2 = str(next(_model_for_ssi.model.parameters()).device)
            print(f"Model on: {_ssi_device_str2}")
        return _model_for_ssi, _readout_for_ssi, _ssi_device_str2

    for _cond_idx, _cond in enumerate(SSI_CONDITIONS):
        _cache_path = (
            SSI_POPULATION_CACHE_DIR
            / f"full_spatial_{_slug_for_cache(_cond)}_trace{int(SSI_TRACE_IDX)}_frames{int(SSI_MAX_FRAMES)}.npz"
        )
        if _cache_path.exists() and not SSI_POPULATION_FORCE_RECOMPUTE:
            with np.load(_cache_path) as _data:
                _spatial_full = np.asarray(_data["spatial"], dtype=np.float32)
            print(f"Loaded cached full spatial movie for {_cond}: {_spatial_full.shape}")
        else:
            from declan.vernier_active_sensing.forward import build_vernier_movie
            from scripts.temporal_decoding.rate_computation import compute_trial_rates

            _model_for_ssi, _readout_for_ssi, _ssi_device_str2 = _ensure_vernier_model_loaded()
            _rng_cond = np.random.default_rng(SEED + 1009 * int(_cond_idx))
            _trace, _ = condition_trace(
                _base_trace_for_ssi,
                condition=_cond,
                trace_set=trace_set,
                rng=_rng_cond,
            )
            _stim = build_vernier_movie(
                canonical_spec,
                _trace[:SSI_MAX_FRAMES],
                device=_ssi_device_str2,
            )
            _spatial_full = compute_trial_rates(
                _model_for_ssi,
                _readout_for_ssi,
                _stim,
                batch_size=SSI_BATCH_SIZE,
                return_spatial=True,
            ).astype(np.float32)
            np.savez_compressed(
                _cache_path,
                spatial=_spatial_full,
                condition=np.asarray([_cond]),
                trace_index=np.asarray([int(SSI_TRACE_IDX)], dtype=np.int32),
                max_frames=np.asarray([int(SSI_MAX_FRAMES)], dtype=np.int32),
            )
            print(f"Computed and cached full spatial movie for {_cond}: {_spatial_full.shape}")

        for _view_row in _ssi_popcomp_views:
            _view = _view_row["view"]
            _spatial_pop = (
                _spatial_full
                if getattr(_view, "membership", None) is None
                else apply_population_view(_spatial_full, _view)
            )
            _tc = _ssi_timecourse(_spatial_pop)
            _total_rate = _ssi_frame_total_rate(_spatial_pop)
            _bits_frame_proxy = _tc * _total_rate
            _summary_rows.append(
                {
                    "condition": _cond,
                    "condition_label": condition_label(_cond),
                    "population_key": _view_row["key"],
                    "population_label": _view_row["label"],
                    "population_version": _view_row["version"],
                    "n_units": int(_spatial_pop.shape[1]),
                    "n_time": int(_spatial_pop.shape[0]),
                    "mean_ssi_bits_per_spike": float(np.nanmean(_tc)),
                    "median_ssi_bits_per_spike": float(np.nanmedian(_tc)),
                    "final_ssi_bits_per_spike": float(_tc[-1]),
                    "std_ssi_bits_per_spike": float(np.nanstd(_tc)),
                    "mean_total_rate": float(np.nanmean(_total_rate)),
                    "final_total_rate": float(_total_rate[-1]),
                    "mean_ssi_bits_frame_proxy": float(np.nanmean(_bits_frame_proxy)),
                    "cumulative_ssi_bits_proxy": float(np.nansum(_bits_frame_proxy)),
                }
            )
            for _t, (_ssi_value, _rate_value, _bits_proxy_value) in enumerate(
                zip(_tc, _total_rate, _bits_frame_proxy, strict=True)
            ):
                _timecourse_rows.append(
                    {
                        "condition": _cond,
                        "condition_label": condition_label(_cond),
                        "population_key": _view_row["key"],
                        "population_label": _view_row["label"],
                        "time_bin": int(_t),
                        "time_ms": float(_t / 120.0 * 1000.0),
                        "ssi_bits_per_spike": float(_ssi_value),
                        "total_rate": float(_rate_value),
                        "ssi_bits_frame_proxy": float(_bits_proxy_value),
                    }
                )

    _ssi_popcomp_summary = pd.DataFrame(_summary_rows)
    _ssi_popcomp_timecourse_df = pd.DataFrame(_timecourse_rows)

    _full_baseline = _ssi_popcomp_summary[_ssi_popcomp_summary["population_key"] == "full756"][
        [
            "condition",
            "mean_ssi_bits_per_spike",
            "mean_ssi_bits_frame_proxy",
            "mean_total_rate",
            "cumulative_ssi_bits_proxy",
        ]
    ].rename(
        columns={
            "mean_ssi_bits_per_spike": "full_mean_ssi_bits_per_spike",
            "mean_ssi_bits_frame_proxy": "full_mean_ssi_bits_frame_proxy",
            "mean_total_rate": "full_mean_total_rate",
            "cumulative_ssi_bits_proxy": "full_cumulative_ssi_bits_proxy",
        }
    )
    _ssi_popcomp_summary = _ssi_popcomp_summary.merge(_full_baseline, on="condition", how="left")
    for _num, _den, _out in [
        ("mean_ssi_bits_per_spike", "full_mean_ssi_bits_per_spike", "mean_ssi_bits_per_spike_vs_full"),
        ("mean_ssi_bits_frame_proxy", "full_mean_ssi_bits_frame_proxy", "mean_ssi_bits_frame_proxy_vs_full"),
        ("mean_total_rate", "full_mean_total_rate", "mean_total_rate_vs_full"),
        ("cumulative_ssi_bits_proxy", "full_cumulative_ssi_bits_proxy", "cumulative_ssi_bits_proxy_vs_full"),
    ]:
        _ssi_popcomp_summary[_out] = _ssi_popcomp_summary[_num] / _ssi_popcomp_summary[_den].replace(0.0, np.nan)

    _summary_path = SSI_POPULATION_CACHE_DIR / "vernier_ssi_population_comparison_summary.csv"
    _timecourse_path = SSI_POPULATION_CACHE_DIR / "vernier_ssi_population_comparison_timecourses.csv"
    _ssi_popcomp_summary.to_csv(_summary_path, index=False)
    _ssi_popcomp_timecourse_df.to_csv(_timecourse_path, index=False)
    print(f"Saved SSI population summary: {_summary_path}")
    print(f"Saved SSI population timecourses: {_timecourse_path}")
    show_table(
        _ssi_popcomp_summary[
            [
                "condition",
                "population_label",
                "n_units",
                "mean_ssi_bits_per_spike",
                "mean_ssi_bits_per_spike_vs_full",
                "mean_ssi_bits_frame_proxy",
                "mean_ssi_bits_frame_proxy_vs_full",
                "mean_total_rate_vs_full",
            ]
        ]
    )

# %%
if RUN_SSI_POPULATION_COMPARISON and not _ssi_popcomp_timecourse_df.empty:
    _pop_labels = list(dict.fromkeys(_ssi_popcomp_timecourse_df["population_label"].tolist()))
    fig_rr_ssi_tc, axes_rr_ssi_tc = plt.subplots(
        len(_pop_labels),
        1,
        figsize=(8.5, max(2.4 * len(_pop_labels), 3.2)),
        dpi=140,
        sharex=True,
        constrained_layout=True,
    )
    axes_arr = np.atleast_1d(axes_rr_ssi_tc)
    for ax, _pop_label in zip(axes_arr, _pop_labels, strict=True):
        _pop_rows = _ssi_popcomp_timecourse_df[_ssi_popcomp_timecourse_df["population_label"] == _pop_label]
        for _cond in SSI_CONDITIONS:
            _rows = _pop_rows[_pop_rows["condition"] == _cond]
            if _rows.empty:
                continue
            ax.plot(
                _rows["time_ms"],
                _rows["ssi_bits_per_spike"],
                label=condition_label(_cond),
                color=COLORS.get(_cond),
                linewidth=1.7,
            )
        ax.set_ylabel("SSI bits/spike")
        ax.set_title(_pop_label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes_arr[-1].set_xlabel("time (ms)")
    axes_arr[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig_rr_ssi_tc.suptitle("Vernier SSI timecourses across population views", y=1.01)
    fig_rr_ssi_tc.savefig(SSI_POPULATION_CACHE_DIR / "vernier_ssi_population_timecourses.png", bbox_inches="tight")
    fig_rr_ssi_tc

# %%
if RUN_SSI_POPULATION_COMPARISON and not _ssi_popcomp_summary.empty:
    _plot_summary = _ssi_popcomp_summary.copy()
    _condition_order = [c for c in SSI_CONDITIONS if c in set(_plot_summary["condition"])]
    _population_order = list(dict.fromkeys(_plot_summary["population_label"].tolist()))
    _x = np.arange(len(_condition_order))
    _w = min(0.18, 0.75 / max(len(_population_order), 1))
    fig_rr_ssi_bars, axes_rr_ssi_bars = plt.subplots(1, 2, figsize=(14.0, 4.4), dpi=140, constrained_layout=True)
    _pop_colors = {
        "full 756": "#4c78a8",
        "RR100 movie-medoid (100 reps)": "#e45756",
        "RR192 mean (192 reps)": "#f58518",
    }
    for _j, _pop_label in enumerate(_population_order):
        _rows = _plot_summary[_plot_summary["population_label"] == _pop_label].set_index("condition")
        _offset = (_j - (len(_population_order) - 1) / 2.0) * _w
        axes_rr_ssi_bars[0].bar(
            _x + _offset,
            [_rows.loc[c, "mean_ssi_bits_per_spike"] if c in _rows.index else np.nan for c in _condition_order],
            width=_w,
            label=_pop_label,
            color=_pop_colors.get(_pop_label, None),
            alpha=0.88,
        )
        axes_rr_ssi_bars[1].bar(
            _x + _offset,
            [_rows.loc[c, "mean_ssi_bits_per_spike_vs_full"] if c in _rows.index else np.nan for c in _condition_order],
            width=_w,
            label=_pop_label,
            color=_pop_colors.get(_pop_label, None),
            alpha=0.88,
        )
    for ax, ylabel, title in [
        (axes_rr_ssi_bars[0], "mean SSI (bits/spike)", "absolute SSI"),
        (axes_rr_ssi_bars[1], "mean SSI / full 756", "normalized to full population"),
    ]:
        ax.set_xticks(_x)
        ax.set_xticklabels([condition_label(c) for c in _condition_order], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.5) if "normalized" in title else None
        ax.spines[["top", "right"]].set_visible(False)
    axes_rr_ssi_bars[0].legend(frameon=False, fontsize=8)
    fig_rr_ssi_bars.suptitle("Vernier SSI across full and redundancy-resolved V1 populations", y=1.03)
    fig_rr_ssi_bars.savefig(SSI_POPULATION_CACHE_DIR / "vernier_ssi_population_bars.png", bbox_inches="tight")
    fig_rr_ssi_bars

# %%
if RUN_SSI_POPULATION_COMPARISON and not _ssi_popcomp_summary.empty:
    fig_rr_ssi_proxy, axes_rr_ssi_proxy = plt.subplots(1, 2, figsize=(13.5, 4.2), dpi=140, constrained_layout=True)
    _population_order = list(dict.fromkeys(_ssi_popcomp_summary["population_label"].tolist()))
    _condition_order = [c for c in SSI_CONDITIONS if c in set(_ssi_popcomp_summary["condition"])]
    for _pop_label in _population_order:
        _rows = _ssi_popcomp_summary[_ssi_popcomp_summary["population_label"] == _pop_label].set_index("condition")
        axes_rr_ssi_proxy[0].plot(
            [condition_label(c) for c in _condition_order],
            [_rows.loc[c, "mean_ssi_bits_frame_proxy_vs_full"] if c in _rows.index else np.nan for c in _condition_order],
            marker="o",
            label=_pop_label,
        )
        axes_rr_ssi_proxy[1].plot(
            [condition_label(c) for c in _condition_order],
            [_rows.loc[c, "mean_total_rate_vs_full"] if c in _rows.index else np.nan for c in _condition_order],
            marker="o",
            label=_pop_label,
        )
    for ax, ylabel, title in [
        (axes_rr_ssi_proxy[0], "rate-weighted SSI proxy / full", "SSI bits/frame proxy"),
        (axes_rr_ssi_proxy[1], "total mean rate / full", "population rate scale"),
    ]:
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.spines[["top", "right"]].set_visible(False)
    axes_rr_ssi_proxy[0].legend(frameon=False, fontsize=8)
    fig_rr_ssi_proxy.suptitle("Amplitude-sensitive SSI summaries after population reduction", y=1.03)
    fig_rr_ssi_proxy.savefig(SSI_POPULATION_CACHE_DIR / "vernier_ssi_population_rate_weighted_proxy.png", bbox_inches="tight")
    fig_rr_ssi_proxy

# %%
if RUN_SSI_POPULATION_COMPARISON and not _ssi_popcomp_summary.empty:
    _fisher_context_rows = []
    for _cond in SSI_CONDITIONS:
        _fisher_context_rows.append(
            {
                "condition": _cond,
                "cached_full_known_trace_fisher": metric_value(
                    rel_df,
                    readout="pose_aware_diagonal_poisson",
                    condition=_cond,
                    fd_step=FD_STEP_TO_PLOT,
                    column="mean_final_fisher",
                ),
                "cached_full_hidden_trace_fisher": metric_value(
                    rel_df,
                    readout="pose_blind_diagonal_count_plus_marginal",
                    condition=_cond,
                    fd_step=FD_STEP_TO_PLOT,
                    column="mean_final_fisher",
                ),
            }
        )
    _fisher_context = pd.DataFrame(_fisher_context_rows)
    _ssi_with_fisher_context = _ssi_popcomp_summary.merge(_fisher_context, on="condition", how="left")
    show_table(
        _ssi_with_fisher_context[
            [
                "condition",
                "population_label",
                "mean_ssi_bits_per_spike",
                "mean_ssi_bits_per_spike_vs_full",
                "cached_full_known_trace_fisher",
                "cached_full_hidden_trace_fisher",
            ]
        ]
    )

    fig_rr_ssi_fisher, axes_rr_ssi_fisher = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=140, constrained_layout=True)
    for _pop_label, _rows in _ssi_with_fisher_context.groupby("population_label", sort=False):
        axes_rr_ssi_fisher[0].scatter(
            _rows["cached_full_known_trace_fisher"],
            _rows["mean_ssi_bits_per_spike"],
            s=48,
            label=_pop_label,
            alpha=0.9,
        )
        axes_rr_ssi_fisher[1].scatter(
            _rows["cached_full_hidden_trace_fisher"],
            _rows["mean_ssi_bits_per_spike"],
            s=48,
            label=_pop_label,
            alpha=0.9,
        )
    axes_rr_ssi_fisher[0].set_xlabel("cached full known-trace Fisher")
    axes_rr_ssi_fisher[1].set_xlabel("cached full hidden-trace Fisher")
    for ax in axes_rr_ssi_fisher:
        ax.set_ylabel("mean SSI (bits/spike)")
        ax.spines[["top", "right"]].set_visible(False)
    axes_rr_ssi_fisher[0].legend(frameon=False, fontsize=8)
    fig_rr_ssi_fisher.suptitle(
        "SSI population comparison with cached full-twin Fisher as task context",
        y=1.03,
    )
    fig_rr_ssi_fisher.savefig(SSI_POPULATION_CACHE_DIR / "vernier_ssi_population_fisher_context.png", bbox_inches="tight")
    fig_rr_ssi_fisher

# %% [markdown]
# ## V1-RR100 SSI from a final history-filled activation map
#
# The previous SSI cells compute a spatial activation **movie** and summarize SSI
# across many response frames. That is useful, but every frame after the first
# already has temporal history in it, so the timecourse can read like a long
# sequence of history-dependent maps.
#
# This diagnostic asks a slightly different question:
#
# 1. Render exactly one model-history window of Vernier stimulus frames
#    (`N_LAGS = 32` here).
# 2. Feed that movie through the standard lag-window model path.
# 3. Keep only the **final** spatial activation map, whose lag cube is filled by
#    those 32 stimulus frames.
# 4. Compute SSI from that single map for full 756 and RR100.
#
# This is not truly instantaneous, because the final activation still depends on
# the filled temporal history. But it avoids averaging over a longer sequence of
# history-bearing activation maps.

# %%
_ssi_final_history_summary = pd.DataFrame()

if not RUN_SSI_FINAL_HISTORY_MAP_DIAGNOSTIC:
    print("Final-history SSI diagnostic disabled (RUN_SSI_FINAL_HISTORY_MAP_DIAGNOSTIC = False).")
else:
    from declan.redundancy_resolved_v1_population import apply_population_view
    from declan.vernier_active_sensing.forward import build_vernier_movie, load_model_and_readout
    from scripts.temporal_decoding.rate_computation import compute_trial_rates

    _history_frames = int(SSI_FINAL_HISTORY_FRAMES)
    if _history_frames != int(MODEL_HISTORY_FRAMES):
        print(
            f"Warning: SSI_FINAL_HISTORY_FRAMES={_history_frames}, "
            f"but model N_LAGS={int(MODEL_HISTORY_FRAMES)}."
        )

    _all_final_views = _ssi_popcomp_views if _ssi_popcomp_views else _load_ssi_population_views()
    _wanted_keys = set(str(k) for k in SSI_FINAL_HISTORY_POPULATION_KEYS)
    _final_views = [row for row in _all_final_views if row["key"] in _wanted_keys]
    if not _final_views:
        raise RuntimeError(f"No population views matched SSI_FINAL_HISTORY_POPULATION_KEYS={SSI_FINAL_HISTORY_POPULATION_KEYS}")

    print("Final-history SSI population views:")
    show_table(
        pd.DataFrame(
            [
                {
                    "population_key": row["key"],
                    "population_label": row["label"],
                    "version": row["version"],
                    "n_units": int(row["view"].n_units),
                }
                for row in _final_views
            ]
        )
    )

    _final_model = None
    _final_readout = None
    _final_device_str = ""

    def _ensure_final_history_model_loaded() -> tuple[Any, Any, str]:
        global _final_model, _final_readout, _final_device_str
        if _final_model is None or _final_readout is None:
            print("Loading model and readout for final-history SSI diagnostic...")
            _final_model, _final_readout = load_model_and_readout(device=SSI_DEVICE)
            _final_device_str = str(next(_final_model.model.parameters()).device)
            print(f"Model on: {_final_device_str}")
        return _final_model, _final_readout, _final_device_str

    def _single_map_total_rate(rate_map: np.ndarray) -> float:
        y = np.asarray(rate_map, dtype=np.float32)
        if y.ndim != 3:
            raise ValueError(f"Expected (C, H, W), got {y.shape}")
        return float(y.reshape(y.shape[0], -1).mean(axis=1).sum())

    def _single_map_diagonal_poisson_fisher(
        plus_map: np.ndarray,
        minus_map: np.ndarray,
        *,
        fd_step_arcmin: float,
        bin_seconds: float,
    ) -> dict[str, float]:
        plus = np.asarray(plus_map, dtype=np.float64).reshape(1, -1)
        minus = np.asarray(minus_map, dtype=np.float64).reshape(1, -1)
        info = poisson_fisher_counts(
            expected_counts(plus, float(bin_seconds)),
            expected_counts(minus, float(bin_seconds)),
            step_arcmin=float(fd_step_arcmin),
        )
        return {
            "fisher": float(info.cumulative_fisher[-1]),
            "dprime2": float(info.cumulative_dprime2[-1]),
            "threshold_proxy": float(info.threshold_proxy[-1]),
            "spike_count": float(info.spike_count[-1]),
        }

    _final_rows: list[dict[str, Any]] = []
    _base_trace_final = valid_trace(trace_set, SSI_TRACE_IDX, max_frames=_history_frames)
    _fd_step_final = float(SSI_FINAL_HISTORY_FD_STEP_ARCMIN)
    _bin_seconds_final = float(SSI_FINAL_HISTORY_BIN_SECONDS)

    for _cond_idx, _cond in enumerate(SSI_FINAL_HISTORY_CONDITIONS):
        _cache_path = (
            SSI_FINAL_HISTORY_CACHE_DIR
            / (
                f"final_history_full_map_{_slug_for_cache(_cond)}"
                f"_trace{int(SSI_TRACE_IDX)}_frames{int(_history_frames)}"
                f"_fd{_fd_step_final:.4f}arcmin.npz"
            )
        )
        _final_spatial_zero = None
        _final_spatial_plus = None
        _final_spatial_minus = None
        if _cache_path.exists() and not SSI_FINAL_HISTORY_FORCE_RECOMPUTE:
            with np.load(_cache_path) as _data:
                if all(key in _data for key in ["final_spatial_zero", "final_spatial_plus", "final_spatial_minus"]):
                    _final_spatial_zero = np.asarray(_data["final_spatial_zero"], dtype=np.float32)
                    _final_spatial_plus = np.asarray(_data["final_spatial_plus"], dtype=np.float32)
                    _final_spatial_minus = np.asarray(_data["final_spatial_minus"], dtype=np.float32)
                elif "final_spatial" in _data:
                    _final_spatial_zero = np.asarray(_data["final_spatial"], dtype=np.float32)
            if _final_spatial_plus is not None and _final_spatial_minus is not None:
                print(f"Loaded final-history full maps for {_cond}: {_final_spatial_zero.shape}")

        if _final_spatial_zero is None or _final_spatial_plus is None or _final_spatial_minus is None:
            _final_model, _final_readout, _final_device_str = _ensure_final_history_model_loaded()
            _rng_cond = np.random.default_rng(SEED + 2003 * int(_cond_idx))
            _trace, _ = condition_trace(
                _base_trace_final,
                condition=_cond,
                trace_set=trace_set,
                rng=_rng_cond,
            )
            _trace = np.asarray(_trace[:_history_frames], dtype=np.float32)
            if _trace.shape[0] != _history_frames:
                raise RuntimeError(
                    f"Expected {_history_frames} frames for {_cond}, got {_trace.shape[0]}"
                )

            def _compute_final_spatial_map(_spec: VernierSpec) -> tuple[np.ndarray, tuple[int, ...]]:
                _stim = build_vernier_movie(
                    _spec,
                    _trace,
                    n_lags=int(MODEL_HISTORY_FRAMES),
                    device=_final_device_str,
                )
                if int(_stim.shape[0]) != _history_frames:
                    raise RuntimeError(
                        f"Expected {_history_frames} lag windows for {_cond}, got {_stim.shape[0]}"
                    )
                _spatial_movie = compute_trial_rates(
                    _final_model,
                    _final_readout,
                    _stim,
                    batch_size=SSI_BATCH_SIZE,
                    return_spatial=True,
                ).astype(np.float32)
                return np.asarray(_spatial_movie[-1], dtype=np.float32), tuple(int(v) for v in _spatial_movie.shape)

            _final_spatial_zero, _zero_movie_shape = _compute_final_spatial_map(canonical_spec)
            _final_spatial_plus, _plus_movie_shape = _compute_final_spatial_map(
                canonical_spec.with_offset(+_fd_step_final)
            )
            _final_spatial_minus, _minus_movie_shape = _compute_final_spatial_map(
                canonical_spec.with_offset(-_fd_step_final)
            )
            np.savez_compressed(
                _cache_path,
                final_spatial_zero=_final_spatial_zero,
                final_spatial_plus=_final_spatial_plus,
                final_spatial_minus=_final_spatial_minus,
                condition=np.asarray([_cond]),
                trace_index=np.asarray([int(SSI_TRACE_IDX)], dtype=np.int32),
                history_frames=np.asarray([int(_history_frames)], dtype=np.int32),
                model_history_frames=np.asarray([int(MODEL_HISTORY_FRAMES)], dtype=np.int32),
                fd_step_arcmin=np.asarray([_fd_step_final], dtype=np.float32),
                bin_seconds=np.asarray([_bin_seconds_final], dtype=np.float32),
                zero_movie_shape=np.asarray(_zero_movie_shape, dtype=np.int32),
                plus_movie_shape=np.asarray(_plus_movie_shape, dtype=np.int32),
                minus_movie_shape=np.asarray(_minus_movie_shape, dtype=np.int32),
            )
            print(f"Computed final-history full maps for {_cond}: {_final_spatial_zero.shape}")

        for _view_row in _final_views:
            _view = _view_row["view"]
            _final_pop_zero = (
                _final_spatial_zero
                if getattr(_view, "membership", None) is None
                else apply_population_view(_final_spatial_zero, _view)
            )
            _final_pop_plus = (
                _final_spatial_plus
                if getattr(_view, "membership", None) is None
                else apply_population_view(_final_spatial_plus, _view)
            )
            _final_pop_minus = (
                _final_spatial_minus
                if getattr(_view, "membership", None) is None
                else apply_population_view(_final_spatial_minus, _view)
            )
            _ssi = _ssi_single_frame(_final_pop_zero)
            _total_rate = _single_map_total_rate(_final_pop_zero)
            _final_fisher = _single_map_diagonal_poisson_fisher(
                _final_pop_plus,
                _final_pop_minus,
                fd_step_arcmin=_fd_step_final,
                bin_seconds=_bin_seconds_final,
            )
            _final_rows.append(
                {
                    "condition": _cond,
                    "condition_label": condition_label(_cond),
                    "population_key": _view_row["key"],
                    "population_label": _view_row["label"],
                    "population_version": _view_row["version"],
                    "n_units": int(_final_pop_zero.shape[0]),
                    "history_frames": int(_history_frames),
                    "model_history_frames": int(MODEL_HISTORY_FRAMES),
                    "history_ms": float(_history_frames / 120.0 * 1000.0),
                    "readout_time_bin": int(_history_frames - 1),
                    "readout_time_ms": float((_history_frames - 1) / 120.0 * 1000.0),
                    "fd_step_arcmin": float(_fd_step_final),
                    "bin_seconds": float(_bin_seconds_final),
                    "fisher_response_vector": "final_spatial_map_flattened",
                    "final_history_ssi_bits_per_spike": float(_ssi["population_bits_per_spike"]),
                    "final_history_total_rate": _total_rate,
                    "final_history_ssi_bits_frame_proxy": float(_ssi["population_bits_per_spike"] * _total_rate),
                    "final_history_fisher": float(_final_fisher["fisher"]),
                    "final_history_dprime2": float(_final_fisher["dprime2"]),
                    "final_history_threshold_proxy": float(_final_fisher["threshold_proxy"]),
                    "final_history_spike_count": float(_final_fisher["spike_count"]),
                }
            )

    _ssi_final_history_summary = pd.DataFrame(_final_rows)
    _full_final = _ssi_final_history_summary[_ssi_final_history_summary["population_key"] == "full756"][
        [
            "condition",
            "final_history_ssi_bits_per_spike",
            "final_history_total_rate",
            "final_history_ssi_bits_frame_proxy",
            "final_history_fisher",
            "final_history_dprime2",
            "final_history_spike_count",
        ]
    ].rename(
        columns={
            "final_history_ssi_bits_per_spike": "full_final_history_ssi_bits_per_spike",
            "final_history_total_rate": "full_final_history_total_rate",
            "final_history_ssi_bits_frame_proxy": "full_final_history_ssi_bits_frame_proxy",
            "final_history_fisher": "full_final_history_fisher",
            "final_history_dprime2": "full_final_history_dprime2",
            "final_history_spike_count": "full_final_history_spike_count",
        }
    )
    _ssi_final_history_summary = _ssi_final_history_summary.merge(_full_final, on="condition", how="left")
    for _num, _den, _out in [
        (
            "final_history_ssi_bits_per_spike",
            "full_final_history_ssi_bits_per_spike",
            "final_history_ssi_bits_per_spike_vs_full",
        ),
        ("final_history_total_rate", "full_final_history_total_rate", "final_history_total_rate_vs_full"),
        (
            "final_history_ssi_bits_frame_proxy",
            "full_final_history_ssi_bits_frame_proxy",
            "final_history_ssi_bits_frame_proxy_vs_full",
        ),
        ("final_history_fisher", "full_final_history_fisher", "final_history_fisher_vs_full"),
        ("final_history_dprime2", "full_final_history_dprime2", "final_history_dprime2_vs_full"),
        ("final_history_spike_count", "full_final_history_spike_count", "final_history_spike_count_vs_full"),
    ]:
        _ssi_final_history_summary[_out] = (
            _ssi_final_history_summary[_num] / _ssi_final_history_summary[_den].replace(0.0, np.nan)
        )

    _final_summary_path = SSI_FINAL_HISTORY_CACHE_DIR / "vernier_ssi_final_history_map_summary.csv"
    _ssi_final_history_summary.to_csv(_final_summary_path, index=False)
    print(f"Saved final-history SSI summary: {_final_summary_path}")
    show_table(
        _ssi_final_history_summary[
            [
                "condition",
                "population_label",
                "n_units",
                "history_frames",
                "final_history_ssi_bits_per_spike",
                "final_history_ssi_bits_per_spike_vs_full",
                "final_history_fisher",
                "final_history_fisher_vs_full",
                "final_history_total_rate_vs_full",
            ]
        ]
    )

# %%
if RUN_SSI_FINAL_HISTORY_MAP_DIAGNOSTIC and not _ssi_final_history_summary.empty:
    _condition_order = [c for c in SSI_FINAL_HISTORY_CONDITIONS if c in set(_ssi_final_history_summary["condition"])]
    _population_order = list(dict.fromkeys(_ssi_final_history_summary["population_label"].tolist()))
    _x = np.arange(len(_condition_order))
    _w = min(0.34, 0.75 / max(len(_population_order), 1))
    _pop_colors = {
        "full 756": "#4c78a8",
        "RR100 movie-medoid (100 reps)": "#e45756",
        "RR192 mean (192 reps)": "#f58518",
    }
    fig_final_hist, axes_final_hist = plt.subplots(1, 4, figsize=(18.0, 4.5), dpi=140, constrained_layout=True)
    for _j, _pop_label in enumerate(_population_order):
        _rows = _ssi_final_history_summary[_ssi_final_history_summary["population_label"] == _pop_label].set_index("condition")
        _offset = (_j - (len(_population_order) - 1) / 2.0) * _w
        axes_final_hist[0].bar(
            _x + _offset,
            [
                _rows.loc[c, "final_history_ssi_bits_per_spike"] if c in _rows.index else np.nan
                for c in _condition_order
            ],
            width=_w,
            label=_pop_label,
            color=_pop_colors.get(_pop_label, None),
            alpha=0.88,
        )
        axes_final_hist[1].bar(
            _x + _offset,
            [
                _rows.loc[c, "final_history_ssi_bits_per_spike_vs_full"] if c in _rows.index else np.nan
                for c in _condition_order
            ],
            width=_w,
            label=_pop_label,
            color=_pop_colors.get(_pop_label, None),
            alpha=0.88,
        )
        axes_final_hist[2].bar(
            _x + _offset,
            [
                _rows.loc[c, "final_history_fisher"] if c in _rows.index else np.nan
                for c in _condition_order
            ],
            width=_w,
            label=_pop_label,
            color=_pop_colors.get(_pop_label, None),
            alpha=0.88,
        )
        axes_final_hist[3].bar(
            _x + _offset,
            [
                _rows.loc[c, "final_history_fisher_vs_full"] if c in _rows.index else np.nan
                for c in _condition_order
            ],
            width=_w,
            label=_pop_label,
            color=_pop_colors.get(_pop_label, None),
            alpha=0.88,
        )
    for ax, ylabel, title in [
        (axes_final_hist[0], "final-map SSI (bits/spike)", "single final map"),
        (axes_final_hist[1], "SSI / full 756", "normalized SSI"),
        (axes_final_hist[2], "single-bin Fisher", "final +δ vs −δ map"),
        (axes_final_hist[3], "Fisher / full 756", "normalized Fisher"),
    ]:
        ax.set_xticks(_x)
        ax.set_xticklabels([condition_label(c) for c in _condition_order], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if title.startswith("normalized"):
            ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes_final_hist[0].legend(frameon=False, fontsize=8)
    fig_final_hist.suptitle(
        f"Vernier SSI from one final activation map after {int(SSI_FINAL_HISTORY_FRAMES)} history frames",
        y=1.03,
    )
    fig_final_hist.savefig(SSI_FINAL_HISTORY_CACHE_DIR / "vernier_ssi_final_history_map_bars.png", bbox_inches="tight")
    fig_final_hist

# %%
if RUN_SSI_FINAL_HISTORY_MAP_DIAGNOSTIC and not _ssi_final_history_summary.empty:
    _condition_order = [c for c in SSI_FINAL_HISTORY_CONDITIONS if c in set(_ssi_final_history_summary["condition"])]
    _rr_final = _ssi_final_history_summary[
        _ssi_final_history_summary["population_key"] == "rr100_medoid"
    ].set_index("condition")
    _retention_metrics = [
        ("final_history_ssi_bits_per_spike_vs_full", "SSI bits/spike", "#4c78a8"),
        ("final_history_total_rate_vs_full", "total rate", "#72b7b2"),
        ("final_history_ssi_bits_frame_proxy_vs_full", "SSI x rate", "#f58518"),
        ("final_history_fisher_vs_full", "Fisher", "#e45756"),
    ]
    fig_final_retention, ax_final_retention = plt.subplots(figsize=(8.8, 4.2), dpi=150, constrained_layout=True)
    for _metric, _label, _color in _retention_metrics:
        ax_final_retention.plot(
            np.arange(len(_condition_order)),
            [_rr_final.loc[c, _metric] if c in _rr_final.index else np.nan for c in _condition_order],
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=_label,
            color=_color,
        )
    ax_final_retention.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.45)
    ax_final_retention.set_xticks(np.arange(len(_condition_order)))
    ax_final_retention.set_xticklabels([condition_label(c) for c in _condition_order], rotation=30, ha="right")
    ax_final_retention.set_ylabel("RR100 / full 756")
    ax_final_retention.set_title("Final history readout: RR100 retention")
    ax_final_retention.set_ylim(0, 1.05)
    ax_final_retention.spines[["top", "right"]].set_visible(False)
    ax_final_retention.legend(frameon=False, ncols=2, fontsize=8)
    fig_final_retention.savefig(
        SSI_FINAL_HISTORY_CACHE_DIR / "vernier_ssi_final_history_rr100_retention.png",
        bbox_inches="tight",
    )
    fig_final_retention

# %%
if RUN_SSI_FINAL_HISTORY_MAP_DIAGNOSTIC and not _ssi_final_history_summary.empty:
    _axis_conditions = ["axis_horizontal", "axis_vertical"]
    _axis_labels = ["across-only", "along-only"]
    _axis_pop_order = ["full756", "rr100_medoid"]
    _axis_pop_labels = {
        "full756": "full 756",
        "rr100_medoid": "RR100",
    }
    _axis_colors = {
        "full756": "#4c78a8",
        "rr100_medoid": "#e45756",
    }
    _axis_metrics = [
        ("final_history_ssi_bits_per_spike", "final-map SSI (bits/spike)", "SSI"),
        ("final_history_fisher", "single-bin Fisher", "Fisher"),
    ]
    fig_final_axis, axes_final_axis = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=150, constrained_layout=True)
    _x_axis = np.arange(len(_axis_conditions))
    _w_axis = 0.32
    for _ax, (_metric, _ylabel, _title) in zip(axes_final_axis, _axis_metrics):
        _axis_metric_values: list[float] = []
        _axis_ratio_lines: list[str] = []
        for _j, _pop_key in enumerate(_axis_pop_order):
            _rows = _ssi_final_history_summary[
                _ssi_final_history_summary["population_key"] == _pop_key
            ].set_index("condition")
            _values = [
                _rows.loc[c, _metric] if c in _rows.index else np.nan
                for c in _axis_conditions
            ]
            _axis_metric_values.extend([float(v) for v in _values if np.isfinite(v)])
            _offset = (_j - 0.5) * _w_axis
            _ax.bar(
                _x_axis + _offset,
                _values,
                width=_w_axis,
                color=_axis_colors[_pop_key],
                alpha=0.88,
                label=_axis_pop_labels[_pop_key],
            )
            if np.all(np.isfinite(_values)) and _values[1] != 0:
                _axis_ratio_lines.append(
                    f"{_axis_pop_labels[_pop_key]} across/along = {_values[0] / _values[1]:.2f}x"
                )
        _ax.set_xticks(_x_axis)
        _ax.set_xticklabels(_axis_labels)
        _ax.set_ylabel(_ylabel)
        _ax.set_title(_title)
        if _axis_metric_values:
            _ax.set_ylim(0, max(_axis_metric_values) * 1.28)
        if _axis_ratio_lines:
            _ax.text(
                0.03,
                0.96,
                "\n".join(_axis_ratio_lines),
                transform=_ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#222222",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
            )
        _ax.spines[["top", "right"]].set_visible(False)
    axes_final_axis[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig_final_axis.suptitle(
        f"Axis-only final readout after {int(SSI_FINAL_HISTORY_FRAMES)} history frames",
        y=1.04,
    )
    fig_final_axis.savefig(
        SSI_FINAL_HISTORY_CACHE_DIR / "vernier_ssi_final_history_axis_focus.png",
        bbox_inches="tight",
    )
    fig_final_axis

# %% [markdown]
# ### SSI interpretation note
#
# **What SSI captures**: how spatially localised the V1 population response is at
# each time step. High SSI means the response is concentrated in a small region of
# the output feature map; low SSI means it is spread across many positions.
#
# **What SSI does not capture**:
# - Whether the spatial pattern is informative about the Vernier sign (+δ vs −δ).
#   A highly localised response to the wrong edge location is high SSI but low
#   Fisher.
# - The Vernier discriminability. Fisher information and d' are the right metrics
#   for that question (computed above from the rate caches).
#
# **Expected behaviour**: static fixation produces a steady, localised response at
# the Vernier edge position → higher SSI. Real FEM sweeps the edge across many
# positions over time → the instantaneous SSI at each frame may be higher (the
# edge is always at some position) but the time-averaged SSI may behave
# differently depending on how much temporal averaging the readout applies.
#
# **Full vs reduced population**: the redundancy twin counts each redundant group
# as one representative. If redundant channels tend to have similar spatial
# profiles (which is the basis for grouping them), removing duplicates should not
# strongly change SSI. A large full-vs-reduced SSI gap would suggest the grouping
# conflates units with meaningfully different spatial tuning.

# %% [markdown]
# ## Audit details
#
# These tables are useful for provenance and debugging, but they come after the
# main teaching plots so they do not interrupt the known-trace, hidden-trace, and
# SSI comparisons.

# %%
if not rel_df.empty:
    readout_counts = (
        rel_df.groupby("readout", dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", "readout"], ascending=[False, True])
    )
    show_table(readout_counts, n=40)

# %% [markdown]
# ## Interpretation checklist
#
# Use this checklist before turning any plot into a claim.
#
# 1. Is the target variable the Vernier offset, not an auxiliary image feature?
# 2. Is the readout known-trace, hidden-trace diagonal, hidden-trace full covariance,
#    or a trajectory-table observer?
# 3. If hidden-trace, does the score include pose-marginal nuisance covariance?
# 4. If full covariance, was it the full population or a unit-subset diagnostic?
# 5. Is the baseline phase-cloud, shuffled order, static center, or scale matched?
# 6. Does a lower threshold proxy come from higher Vernier-aligned signal, or
#    from a readout that is not penalizing nuisance?
# 7. Are we treating Vernier as a pose-confusion diagnostic, or natural images as
#    the joint-inference success target?
# 8. Are we making a twin-internal claim, an animal-behavior claim, or the bridge
#    between the two?
#
# Clarification needed before manuscript-level language:
#
# - Why the exact default Vernier dimensions were chosen.
# - Whether `spatial_collapse=max` is a historical implementation default or a
#   principled readout choice for this analysis.
# - Which full-covariance run should be treated as the canonical along/across or
#   scale contrast if multiple output directories disagree.

# %%
checklist = pd.DataFrame(
    [
        {
            "claim_type": "Twin-internal",
            "safe_wording": "Under readout X, condition A has higher/lower Vernier Fisher than condition B.",
            "needs_extra_evidence": "None beyond the loaded run and its provenance.",
        },
        {
            "claim_type": "Behavior bridge",
            "safe_wording": "Animal drift geometry is compared with a pre-specified twin objective.",
            "needs_extra_evidence": "Separate behavior analysis and a pre-committed objective/readout.",
        },
        {
            "claim_type": "Brain readout",
            "safe_wording": "A noise-limited constrained observer would be penalized by nuisance covariance.",
            "needs_extra_evidence": "Do not claim the brain literally implements this linear readout without additional evidence.",
        },
        {
            "claim_type": "Full covariance",
            "safe_wording": "Full-covariance or unit-subset full-covariance diagnostic, depending on row name.",
            "needs_extra_evidence": "Canonical full-population run if the row is unit-subset only.",
        },
        {
            "claim_type": "Stimulus scope",
            "safe_wording": "Vernier demonstrates pose confusion and known-trace upper bounds; natural images are the target for robust joint image-pose inference.",
            "needs_extra_evidence": "Natural-image joint observer results, not Vernier alone.",
        },
    ]
)
show_table(checklist)

# %%
