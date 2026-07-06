"""Step-through validation for a redundancy-resolved V1 reduced twin population.

Run this file in an editor that understands ``# %%`` cells.  The workflow is
kept separate from ``notebooks/spatial_spiking_information_walkthrough.ipynb``
so the notebook can remain a full-twin SSI walkthrough while this script vets
the reduced population as a plug-in population view.
"""

# %%
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from IPython.display import display
except Exception:  # pragma: no cover - interactive convenience only.
    def display(obj):
        print(obj)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from declan.redundancy_resolved_v1_population import (
    apply_population_view,
    cubes_to_visioncore_stim,
    load_population_bundle,
    load_population_view,
)
from jake.twininfo.common import extract_fixrsvp_eye_traces
from jake.twininfo.retinal_examples import model_lag_cubes_from_image_trace, select_trace_examples
from jake.twininfo.stimuli import load_natural_images
from spatial_info import compute_rate_map_batched

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)

# %% Knobs
POPULATION_VERSION_NAME = "V1-RR_complete_0p65_moviesplit0p75_pair0p60_rec4_blockjkP0p50n5L0p50n4_merge2nd1.01"
POPULATION_SHORT_NAME = "V1-RR192"


def _safe_slug(value: object, max_len: int = 96) -> str:
    text = str(value)
    text = Path(text).stem if "/" in text else text
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_len] or "unnamed"


POPULATION_TAG = _safe_slug(POPULATION_VERSION_NAME)
OUT_DIR = ROOT / "outputs" / "redundancy_resolved_v1_twin" / f"validation_{POPULATION_TAG}"
CACHE_PATH = OUT_DIR / "condition_comparison.npz"
MOVIE_GROUP_QUALITY_CSV = OUT_DIR / "movie_group_quality.csv"
FRAME_GROUP_QUALITY_CSV = OUT_DIR / "frame_group_quality_from_movie_audit.csv"
RECONSTRUCTION_CSV = OUT_DIR / "pool_expand_reconstruction.csv"
FIG_DIR = OUT_DIR / "figures"

DEVICE = None  # None -> cuda if available, else cpu. Override with "cpu", "cuda:0", etc.
BATCH_SIZE = 8
RNG_SEED = 7
T_MAX = 40
FRAME_INDEX = T_MAX - 1
IMAGE_INDICES = [24]

FORCE_RECOMPUTE = False
FORCE_MOVIE_AUDIT_RECOMPUTE = False
SAVE_FIGURES = True

REQUESTED_CONDITIONS = [
    ("static", "static", "static"),
    ("empirical 0.5x", "empirical", "rel_0p5x"),
    ("empirical 1x", "empirical", "rel_1x"),
    ("empirical 2x", "empirical", "rel_2x"),
    ("brownian 1x", "brownian", "rel_1x"),
    ("rotated 1x", "rotated", "rel_1x"),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
print("population:", POPULATION_VERSION_NAME)
print("cache:", CACHE_PATH.relative_to(ROOT))

# %% Small analysis helpers
def _scale_trace_about_mean(trace: np.ndarray, scale: float) -> np.ndarray:
    tr = np.asarray(trace, dtype=np.float32)
    center = tr.mean(axis=0, keepdims=True)
    return (center + float(scale) * (tr - center)).astype(np.float32)


def _rotate_trace_about_mean(trace: np.ndarray, angle_rad: float = np.pi / 2) -> np.ndarray:
    tr = np.asarray(trace, dtype=np.float32)
    center = tr.mean(axis=0, keepdims=True)
    centered = tr - center
    rot = np.asarray(
        [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]],
        dtype=np.float32,
    )
    return (centered @ rot.T + center).astype(np.float32)


def _brownian_trace_matched_to_empirical(trace: np.ndarray, seed: int) -> np.ndarray:
    tr = np.asarray(trace, dtype=np.float32)
    center = tr.mean(axis=0, keepdims=True)
    steps = np.diff(tr, axis=0).astype(np.float64)
    if steps.shape[0] == 0:
        return np.repeat(center, tr.shape[0], axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    mu = steps.mean(axis=0)
    cov = np.cov(steps.T) if steps.shape[0] > 1 else np.eye(2) * 1e-10
    cov = np.asarray(cov, dtype=np.float64) + np.eye(2) * 1e-10
    ctrl_steps = rng.multivariate_normal(mu, cov, size=steps.shape[0]).astype(np.float32)
    ctrl = np.vstack([np.zeros((1, 2), dtype=np.float32), np.cumsum(ctrl_steps, axis=0)])
    ctrl -= ctrl.mean(axis=0, keepdims=True)
    target_rms = np.sqrt(np.mean(np.sum((tr - center) ** 2, axis=1)))
    ctrl_rms = np.sqrt(np.mean(np.sum(ctrl * ctrl, axis=1)))
    if ctrl_rms > 1e-8:
        ctrl *= float(target_rms / ctrl_rms)
    return (ctrl + center).astype(np.float32)


def condition_traces_from_source(trace: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    tr = np.asarray(trace, dtype=np.float32)
    center = tr.mean(axis=0, keepdims=True)
    return {
        "static": np.repeat(center, tr.shape[0], axis=0).astype(np.float32),
        "empirical 0.5x": _scale_trace_about_mean(tr, 0.5),
        "empirical 1x": tr.copy(),
        "empirical 2x": _scale_trace_about_mean(tr, 2.0),
        "brownian 1x": _brownian_trace_matched_to_empirical(tr, seed=seed + 101),
        "rotated 1x": _rotate_trace_about_mean(tr, angle_rad=np.pi / 2),
    }


def trace_qc_table(condition_traces: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for label, trace in condition_traces.items():
        centered = trace - trace.mean(axis=0, keepdims=True)
        steps = np.diff(trace, axis=0)
        step_amp = np.linalg.norm(steps, axis=1) if steps.size else np.zeros(0)
        rows.append(
            {
                "condition": label,
                "rms_displacement_deg": float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))),
                "path_length_deg": float(step_amp.sum()) if step_amp.size else 0.0,
                "step_rms_deg": float(np.sqrt(np.mean(step_amp * step_amp))) if step_amp.size else 0.0,
                "step_p95_deg": float(np.percentile(step_amp, 95)) if step_amp.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def spatial_ssi_single_frame(rate_maps: np.ndarray, eps: float = 1e-8) -> dict[str, np.ndarray | float]:
    y = np.asarray(rate_maps, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"Expected (unit, height, width), got {y.shape}")
    if np.any(y < 0):
        raise ValueError("Rate maps must be non-negative.")
    n_units, height, width = y.shape
    flat = y.reshape(n_units, height * width)
    rbar = flat.mean(axis=1)
    gain = flat / (rbar[:, None] + eps)
    unit_bits = np.mean(gain * np.log2(gain + eps), axis=1)
    weights = rbar / np.maximum(rbar.sum(), eps)
    return {
        "unit_bits_per_spike": unit_bits,
        "unit_mean_rate": rbar,
        "population_bits_per_spike": float(np.sum(weights * unit_bits)),
    }


def corr_1d(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom < eps:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def group_centroid_quality_from_unit_array(
    unit_array: np.ndarray,
    cluster_membership: np.ndarray,
    *,
    image_index: int,
    condition: str,
    metric_space: str,
) -> pd.DataFrame:
    """Member-to-centroid and pairwise member-member correlations, channel first: C x ..."""
    x = np.asarray(unit_array, dtype=np.float32)
    mem = np.asarray(cluster_membership, dtype=np.float32)
    if x.ndim < 2:
        raise ValueError(f"Expected channel-first array with at least 2 dims, got {x.shape}")
    if x.shape[0] != mem.shape[1]:
        raise ValueError(f"unit_array has {x.shape[0]} channels, membership expects {mem.shape[1]}")

    rows = []
    for rep_idx, weights in enumerate(mem):
        members = np.flatnonzero(weights > 0)
        if len(members) <= 1:
            continue
        centroid = np.tensordot(weights[members], x[members], axes=(0, 0))
        corrs = np.asarray([corr_1d(x[m], centroid) for m in members], dtype=np.float32)
        stds = np.asarray([float(np.nanstd(x[m])) for m in members], dtype=np.float32)
        finite_corrs = corrs[np.isfinite(corrs)]
        worst_pos = int(np.nanargmin(corrs)) if finite_corrs.size else 0

        # Pairwise member-member correlations — catches bad merges that member-centroid
        # can miss when two dissimilar members both correlate moderately with their mean.
        vecs = [np.asarray(x[m], dtype=np.float64).ravel() for m in members]
        pairwise = [
            corr_1d(vecs[i], vecs[j])
            for i in range(len(members))
            for j in range(i + 1, len(members))
        ]
        pairwise = np.asarray(pairwise, dtype=np.float32)
        finite_pairwise = pairwise[np.isfinite(pairwise)]
        worst_pair_pos = int(np.nanargmin(pairwise)) if finite_pairwise.size else 0
        if finite_pairwise.size:
            # recover which (i, j) pair corresponds to worst_pair_pos
            pair_idx = 0
            worst_pair = (int(members[0]), int(members[1]))
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if pair_idx == worst_pair_pos:
                        worst_pair = (int(members[i]), int(members[j]))
                    pair_idx += 1
        else:
            worst_pair = (-1, -1)

        rows.append(
            {
                "image_index": int(image_index),
                "condition": str(condition),
                "metric_space": str(metric_space),
                "rep_idx": int(rep_idx),
                "n_members": int(len(members)),
                "min_member_centroid_corr": float(np.nanmin(corrs)) if finite_corrs.size else float("nan"),
                "p05_member_centroid_corr": float(np.nanpercentile(corrs, 5)) if finite_corrs.size else float("nan"),
                "median_member_centroid_corr": float(np.nanmedian(corrs)) if finite_corrs.size else float("nan"),
                "worst_member": int(members[worst_pos]),
                "worst_member_corr": float(corrs[worst_pos]),
                "min_pairwise_member_corr": float(np.nanmin(finite_pairwise)) if finite_pairwise.size else float("nan"),
                "median_pairwise_member_corr": float(np.nanmedian(finite_pairwise)) if finite_pairwise.size else float("nan"),
                "worst_pair_ch_a": int(worst_pair[0]),
                "worst_pair_ch_b": int(worst_pair[1]),
                "worst_pair_corr": float(pairwise[worst_pair_pos]) if finite_pairwise.size else float("nan"),
                "min_member_std": float(np.nanmin(stds)),
                "median_member_std": float(np.nanmedian(stds)),
                "centroid_std": float(np.nanstd(centroid)),
                "members": ",".join(str(int(m)) for m in members),
            }
        )
    return pd.DataFrame(rows)


def expand_population_to_full_channels(reduced: np.ndarray, cluster_membership: np.ndarray) -> np.ndarray:
    """Expand reduced rates back to full channels by assigning each channel its representative."""
    y = np.asarray(reduced, dtype=np.float32)
    mem = np.asarray(cluster_membership, dtype=np.float32)
    if y.ndim == 3:
        y = y[np.newaxis]
        squeeze_time = True
    elif y.ndim == 4:
        squeeze_time = False
    else:
        raise ValueError(f"Expected reduced shape (T, R, H, W) or (R, H, W), got {y.shape}")
    if y.shape[1] != mem.shape[0]:
        raise ValueError(f"Reduced movie has {y.shape[1]} reps, membership has {mem.shape[0]}")
    nonzero_per_channel = np.sum(mem > 0, axis=0)
    if not np.all(nonzero_per_channel == 1):
        raise ValueError("Membership must assign each input channel to exactly one representative.")
    channel_to_rep = np.argmax(mem > 0, axis=0)
    expanded = y[:, channel_to_rep]
    return expanded[0] if squeeze_time else expanded


def spatial_ssi_timecourse(rate_movie: np.ndarray) -> np.ndarray:
    y = np.asarray(rate_movie, dtype=np.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected (T, C, H, W), got {y.shape}")
    return np.asarray(
        [spatial_ssi_single_frame(y[t])["population_bits_per_spike"] for t in range(y.shape[0])],
        dtype=np.float32,
    )


def pool_expand_reconstruction_metrics(
    full_movie: np.ndarray,
    reduced_movie: np.ndarray,
    cluster_membership: np.ndarray,
    *,
    image_index: int,
    condition: str,
) -> dict[str, float | int | str]:
    full = np.asarray(full_movie, dtype=np.float32)
    mem = np.asarray(cluster_membership, dtype=np.float32)
    # channels with label -2 are excluded from the population and have no representative;
    # restrict both full and cluster membership to the 1-rep-per-channel subset before expanding
    represented_mask = np.sum(mem > 0, axis=0) == 1
    full = full[:, represented_mask]
    mem = mem[:, represented_mask]
    expanded = expand_population_to_full_channels(reduced_movie, mem).astype(np.float32)
    if expanded.shape != full.shape:
        raise ValueError(f"Expanded movie shape {expanded.shape} does not match full movie {full.shape}")

    residual = expanded - full
    full_std = float(np.nanstd(full))
    per_channel_corr = np.asarray([corr_1d(full[:, ch], expanded[:, ch]) for ch in range(full.shape[1])], dtype=np.float32)
    full_ssi = spatial_ssi_timecourse(full)
    expanded_ssi = spatial_ssi_timecourse(expanded)
    reduced_ssi = spatial_ssi_timecourse(np.asarray(reduced_movie, dtype=np.float32))
    return {
        "image_index": int(image_index),
        "condition": str(condition),
        "n_time": int(full.shape[0]),
        "n_represented_channels": int(full.shape[1]),
        "n_representatives": int(np.asarray(reduced_movie).shape[1]),
        "global_rate_corr": corr_1d(full, expanded),
        "rmse": float(np.sqrt(np.nanmean(residual * residual))),
        "nrmse_by_full_std": float(np.sqrt(np.nanmean(residual * residual)) / max(full_std, 1e-8)),
        "median_channel_corr": float(np.nanmedian(per_channel_corr)),
        "p05_channel_corr": float(np.nanpercentile(per_channel_corr, 5)),
        "min_channel_corr": float(np.nanmin(per_channel_corr)),
        "mean_full_ssi": float(np.nanmean(full_ssi)),
        "mean_expanded_ssi": float(np.nanmean(expanded_ssi)),
        "mean_reduced_population_ssi": float(np.nanmean(reduced_ssi)),
        "mean_abs_expanded_minus_full_ssi": float(np.nanmean(np.abs(expanded_ssi - full_ssi))),
        "max_abs_expanded_minus_full_ssi": float(np.nanmax(np.abs(expanded_ssi - full_ssi))),
        # This is an effect size, not a reconstruction error: reduction changes the
        # population weighting by counting each representative once.
        "mean_reduced_minus_full_ssi": float(np.nanmean(reduced_ssi - full_ssi)),
        "mean_abs_reduced_minus_full_ssi": float(np.nanmean(np.abs(reduced_ssi - full_ssi))),
        "ssi_timecourse_corr_expanded_full": corr_1d(full_ssi, expanded_ssi),
        "ssi_timecourse_corr_reduced_full": corr_1d(full_ssi, reduced_ssi),
    }


# %% Load the canonical twin and reduced population view
population_view = load_population_view(version_name=POPULATION_VERSION_NAME)
membership = population_view.membership
assert membership is not None
cluster_membership = population_view.cluster_membership
assert cluster_membership is not None

bundle = load_population_bundle(population=population_view, device=DEVICE)
print(f"loaded bundle on {bundle.device}")
print(f"canonical channels: {len(bundle.unit_rows)}")
print(f"{population_view.name}: {population_view.n_units} representatives from {population_view.input_channels} channels")
print("pooling membership row-sum range:", float(membership.sum(axis=1).min()), float(membership.sum(axis=1).max()))
print(
    "cluster membership row-sum range:",
    float(cluster_membership.sum(axis=1).min()),
    float(cluster_membership.sum(axis=1).max()),
)

# %% Inspect the population spec and largest groups
representatives = pd.DataFrame(population_view.meta["representatives"])
representatives["member_preview"] = representatives["members"].map(lambda xs: ",".join(map(str, xs[:8])))
representatives.sort_values(["n_members", "mean_ccnorm"], ascending=[False, False]).head(20)[
    ["rep_idx", "kind", "n_members", "mean_ccnorm", "sessions", "member_preview"]
]

# %% Select a source fixation trace
eye_traces, durations = extract_fixrsvp_eye_traces(bundle.model, min_fix_dur=int(T_MAX))
examples = select_trace_examples(eye_traces, durations, t_max=int(T_MAX), n_each=1, seed=int(RNG_SEED), stride=8)
source_example = next((ex for ex in examples if ex.kind == "fixation"), examples[0])
condition_traces = condition_traces_from_source(source_example.trace, seed=RNG_SEED)
trace_qc_table(condition_traces).style.format(precision=4)

# %%
fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
for label, trace in condition_traces.items():
    ax.plot(trace[:, 0], trace[:, 1], lw=1.0, alpha=0.85, label=label)
ax.set_aspect("equal")
ax.set_xlabel("x eye position (deg)")
ax.set_ylabel("y eye position (deg)")
ax.set_title(f"Condition traces from {source_example.example_id}")
ax.legend(fontsize=7)
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / "condition_traces.png")

# %% Compute or load full and reduced-population snapshots/traces
def compute_condition_cache(cache_path: Path) -> dict[str, np.ndarray | list[str]]:
    condition_labels = list(condition_traces)
    full_frame_maps = []
    reduced_frame_maps = []
    full_mean_traces = []
    reduced_mean_traces = []
    image_indices = []

    for image_index in IMAGE_INDICES:
        (_spec, image) = load_natural_images(1, indices=(int(image_index),))[0]
        for label in condition_labels:
            trace = condition_traces[label]
            cubes = model_lag_cubes_from_image_trace(
                image,
                trace,
                t_max=int(T_MAX),
                crop_center_offset_px=(0.0, 0.0),
            )
            stim = cubes_to_visioncore_stim(cubes)
            with torch.no_grad():
                full_rate = compute_rate_map_batched(bundle.model, bundle.readout, stim, batch_size=int(BATCH_SIZE))
            full_np = full_rate.detach().cpu().numpy().astype(np.float32)
            if full_np.shape[1] != population_view.input_channels:
                raise ValueError(f"Expected {population_view.input_channels} channels, got {full_np.shape[1]}")
            reduced_np = apply_population_view(full_np, population_view).astype(np.float32)
            frame_index = int(np.clip(FRAME_INDEX, 0, full_np.shape[0] - 1))

            full_frame_maps.append(full_np[frame_index])
            reduced_frame_maps.append(reduced_np[frame_index])
            full_mean_traces.append(full_np.mean(axis=(2, 3)))
            reduced_mean_traces.append(reduced_np.mean(axis=(2, 3)))
            image_indices.append(int(image_index))
            print(f"image {image_index} | {label}: full={full_np.shape}, reduced={reduced_np.shape}")

    out = {
        "condition_labels": np.asarray(condition_labels * len(IMAGE_INDICES)),
        "image_indices": np.asarray(image_indices, dtype=np.int32),
        "full_frame_maps": np.stack(full_frame_maps, axis=0).astype(np.float32),
        "reduced_frame_maps": np.stack(reduced_frame_maps, axis=0).astype(np.float32),
        "full_mean_traces": np.stack(full_mean_traces, axis=0).astype(np.float32),
        "reduced_mean_traces": np.stack(reduced_mean_traces, axis=0).astype(np.float32),
        "frame_index": np.asarray(FRAME_INDEX, dtype=np.int32),
        "t_max": np.asarray(T_MAX, dtype=np.int32),
        "population_name": np.asarray(population_view.name),
        "population_n_units": np.asarray(population_view.n_units, dtype=np.int32),
        "population_version_name": np.asarray(POPULATION_VERSION_NAME),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    return out


def load_condition_cache(cache_path: Path) -> dict[str, np.ndarray]:
    d = np.load(cache_path, allow_pickle=True)
    return {key: np.asarray(d[key]) for key in d.files}


if CACHE_PATH.exists() and not FORCE_RECOMPUTE:
    data = load_condition_cache(CACHE_PATH)
    print("loaded cache")
else:
    data = compute_condition_cache(CACHE_PATH)

print("full frame maps:", data["full_frame_maps"].shape)
print(f"{POPULATION_SHORT_NAME} frame maps:", data["reduced_frame_maps"].shape)

# %% Full tested-movie audit: group coherence and pool-expand reconstruction
def compute_movie_audit_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movie_quality_parts = []
    frame_quality_parts = []
    reconstruction_rows = []

    for image_index in IMAGE_INDICES:
        (_spec, image) = load_natural_images(1, indices=(int(image_index),))[0]
        for label in condition_traces:
            trace = condition_traces[label]
            cubes = model_lag_cubes_from_image_trace(
                image,
                trace,
                t_max=int(T_MAX),
                crop_center_offset_px=(0.0, 0.0),
            )
            stim = cubes_to_visioncore_stim(cubes)
            with torch.no_grad():
                full_rate = compute_rate_map_batched(bundle.model, bundle.readout, stim, batch_size=int(BATCH_SIZE))
            full_np = full_rate.detach().cpu().numpy().astype(np.float32)
            reduced_np = apply_population_view(full_np, population_view).astype(np.float32)
            frame_index = int(np.clip(FRAME_INDEX, 0, full_np.shape[0] - 1))

            # Movie quality flattens (time, H, W) per member.
            movie_q = group_centroid_quality_from_unit_array(
                np.moveaxis(full_np, 1, 0),
                cluster_membership,
                image_index=int(image_index),
                condition=label,
                metric_space="movie_t_h_w",
            )
            frame_q = group_centroid_quality_from_unit_array(
                full_np[frame_index],
                cluster_membership,
                image_index=int(image_index),
                condition=label,
                metric_space=f"frame_{frame_index}",
            )
            recon = pool_expand_reconstruction_metrics(
                full_np,
                reduced_np,
                cluster_membership,
                image_index=int(image_index),
                condition=label,
            )
            movie_quality_parts.append(movie_q)
            frame_quality_parts.append(frame_q)
            reconstruction_rows.append(recon)
            print(
                f"movie audit image {image_index} | {label}: "
                f"worst movie group={movie_q['min_member_centroid_corr'].min():.3f}, "
                f"global recon corr={recon['global_rate_corr']:.3f}"
            )

    movie_quality = pd.concat(movie_quality_parts, ignore_index=True)
    frame_quality = pd.concat(frame_quality_parts, ignore_index=True)
    reconstruction = pd.DataFrame(reconstruction_rows)
    movie_quality.to_csv(MOVIE_GROUP_QUALITY_CSV, index=False)
    frame_quality.to_csv(FRAME_GROUP_QUALITY_CSV, index=False)
    reconstruction.to_csv(RECONSTRUCTION_CSV, index=False)
    return movie_quality, frame_quality, reconstruction


if (
    MOVIE_GROUP_QUALITY_CSV.exists()
    and FRAME_GROUP_QUALITY_CSV.exists()
    and RECONSTRUCTION_CSV.exists()
    and not FORCE_MOVIE_AUDIT_RECOMPUTE
):
    movie_group_quality = pd.read_csv(MOVIE_GROUP_QUALITY_CSV)
    frame_group_quality_from_audit = pd.read_csv(FRAME_GROUP_QUALITY_CSV)
    reconstruction_summary = pd.read_csv(RECONSTRUCTION_CSV)
    print("loaded movie audit tables")
else:
    movie_group_quality, frame_group_quality_from_audit, reconstruction_summary = compute_movie_audit_tables()

print("movie group quality:", movie_group_quality.shape)
print("frame group quality:", frame_group_quality_from_audit.shape)
print("reconstruction:", reconstruction_summary.shape)

if "mean_reduced_minus_full_ssi" not in reconstruction_summary and "mean_reduced_population_ssi" in reconstruction_summary:
    reconstruction_summary["mean_reduced_minus_full_ssi"] = (
        reconstruction_summary["mean_reduced_population_ssi"] - reconstruction_summary["mean_full_ssi"]
    )

# %% Population-level SSI comparison
rows = []
for i, label in enumerate(data["condition_labels"]):
    full = spatial_ssi_single_frame(data["full_frame_maps"][i])
    reduced = spatial_ssi_single_frame(data["reduced_frame_maps"][i])
    rows.append(
        {
            "image_index": int(data["image_indices"][i]),
            "condition": str(label),
            "full_pop_ssi": full["population_bits_per_spike"],
            "reduced_pop_ssi": reduced["population_bits_per_spike"],
            "delta_reduced_minus_full": reduced["population_bits_per_spike"] - full["population_bits_per_spike"],
            "full_median_unit_ssi": float(np.median(full["unit_bits_per_spike"])),
            "reduced_median_unit_ssi": float(np.median(reduced["unit_bits_per_spike"])),
        }
    )
ssi_summary = pd.DataFrame(rows)
ssi_summary.style.format(precision=5)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
x = np.arange(len(ssi_summary))
axes[0].plot(x, ssi_summary["full_pop_ssi"], marker="o", label="full 756", color="tab:blue")
axes[0].plot(x, ssi_summary["reduced_pop_ssi"], marker="o", label=POPULATION_SHORT_NAME, color="tab:orange")
axes[0].set_xticks(x)
axes[0].set_xticklabels(ssi_summary["condition"].str.replace(" ", "\n"), fontsize=8)
axes[0].set_ylabel("population SSI (bits/spike)")
axes[0].legend(fontsize=8)

colors = np.where(ssi_summary["delta_reduced_minus_full"].to_numpy() >= 0, "tab:green", "tab:red")
axes[1].bar(x, ssi_summary["delta_reduced_minus_full"], color=colors, alpha=0.85)
axes[1].axhline(0, color="0.2", lw=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(ssi_summary["condition"].str.replace(" ", "\n"), fontsize=8)
axes[1].set_ylabel(f"{POPULATION_SHORT_NAME} - full SSI")
fig.suptitle("Population SSI effect of counting each representative once")
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / "population_ssi_full_vs_reduced.png")

# %% Group centroid quality on the tested activation maps
def group_centroid_quality(frame_maps: np.ndarray, cluster_membership: np.ndarray) -> pd.DataFrame:
    return group_centroid_quality_from_unit_array(
        frame_maps,
        cluster_membership,
        image_index=-1,
        condition="",
        metric_space="frame",
    ).drop(columns=["image_index", "condition", "metric_space"])


quality_tables = []
for i, label in enumerate(data["condition_labels"]):
    q = group_centroid_quality(data["full_frame_maps"][i], cluster_membership)
    q["condition"] = str(label)
    q["image_index"] = int(data["image_indices"][i])
    quality_tables.append(q)
group_quality = pd.concat(quality_tables, ignore_index=True)
group_quality_summary = (
    group_quality.groupby(["image_index", "condition"])
    .agg(
        worst_min_centroid_corr=("min_member_centroid_corr", "min"),
        median_min_centroid_corr=("min_member_centroid_corr", "median"),
        n_groups=("rep_idx", "size"),
    )
    .reset_index()
)
group_quality_summary.style.format(precision=4)

# %% Movie-vs-frame audit summaries and split-to-singletons estimates
movie_quality_summary = (
    movie_group_quality.groupby(["image_index", "condition"])
    .agg(
        worst_movie_min_centroid_corr=("min_member_centroid_corr", "min"),
        median_movie_min_centroid_corr=("min_member_centroid_corr", "median"),
        groups_below_0p60=("min_member_centroid_corr", lambda x: int(np.sum(np.asarray(x) < 0.60))),
        groups_below_0p75=("min_member_centroid_corr", lambda x: int(np.sum(np.asarray(x) < 0.75))),
        n_groups=("rep_idx", "size"),
    )
    .reset_index()
)
display(movie_quality_summary.style.format(precision=4))
display(reconstruction_summary.style.format(precision=5))

print(
    "Interpretation: full-vs-expanded metrics test merge reconstruction. "
    "Full-vs-reduced SSI is an effect size of changing the population weighting, "
    "not a conservation criterion."
)

frame_vs_movie = frame_group_quality_from_audit.merge(
    movie_group_quality,
    on=["image_index", "condition", "rep_idx", "n_members"],
    suffixes=("_frame", "_movie"),
)
frame_vs_movie["frame_minus_movie_min_corr"] = (
    frame_vs_movie["min_member_centroid_corr_frame"]
    - frame_vs_movie["min_member_centroid_corr_movie"]
)
display(
    frame_vs_movie.sort_values("min_member_centroid_corr_frame")
    .head(25)[
        [
            "image_index",
            "condition",
            "rep_idx",
            "n_members",
            "min_member_centroid_corr_frame",
            "min_member_centroid_corr_movie",
            "worst_member_frame",
            "worst_member_movie",
            "min_member_std_frame",
            "centroid_std_frame",
        ]
    ]
    .style.format(precision=4)
)


def split_to_singletons_estimate(group_table: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    worst_by_rep = (
        group_table.groupby("rep_idx")
        .agg(
            n_members=("n_members", "max"),
            worst_min_member_centroid_corr=("min_member_centroid_corr", "min"),
        )
        .reset_index()
    )
    failing = worst_by_rep[worst_by_rep["worst_min_member_centroid_corr"] < float(threshold)]
    extra_reps = int(np.sum(failing["n_members"].to_numpy(dtype=int) - 1))
    return {
        "threshold": float(threshold),
        "n_failing_groups": int(len(failing)),
        "extra_reps_if_split_to_singletons": extra_reps,
        "estimated_n_representatives": int(population_view.n_units + extra_reps),
        "largest_failing_group": int(failing["n_members"].max()) if len(failing) else 0,
        "worst_group_corr": float(failing["worst_min_member_centroid_corr"].min()) if len(failing) else np.nan,
    }


resplit_estimates = pd.DataFrame(
    [
        split_to_singletons_estimate(movie_group_quality, 0.60),
        split_to_singletons_estimate(movie_group_quality, 0.75),
    ]
)
display(resplit_estimates.style.format(precision=4))

# %% Quality metric dashboard plots
def plot_quality_metric_dashboard(
    movie_group_quality: pd.DataFrame,
    frame_group_quality: pd.DataFrame,
    frame_vs_movie: pd.DataFrame,
    reconstruction_summary: pd.DataFrame,
    resplit_estimates: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), constrained_layout=True)
    axes = axes.ravel()

    bins = np.linspace(-0.2, 1.0, 49)
    axes[0].hist(
        movie_group_quality["min_member_centroid_corr"],
        bins=bins,
        histtype="stepfilled",
        alpha=0.45,
        label="centroid corr (movie)",
        color="tab:blue",
    )
    axes[0].hist(
        frame_group_quality["min_member_centroid_corr"],
        bins=bins,
        histtype="step",
        linewidth=2.0,
        label="centroid corr (frame)",
        color="tab:orange",
    )
    if "min_pairwise_member_corr" in movie_group_quality.columns:
        axes[0].hist(
            movie_group_quality["min_pairwise_member_corr"].dropna(),
            bins=bins,
            histtype="step",
            linewidth=1.5,
            linestyle="dashed",
            label="pairwise corr (movie)",
            color="tab:purple",
        )
    for thr in (0.60, 0.75):
        axes[0].axvline(thr, color="0.25", lw=0.9, ls="--" if thr == 0.75 else ":")
    axes[0].set_xlabel("correlation")
    axes[0].set_ylabel("groups x conditions")
    axes[0].set_title("Group coherence distribution")
    axes[0].legend(frameon=False, fontsize=7)

    # Pairwise vs centroid scatter — reveals hidden bad merges
    has_pairwise_movie = "min_pairwise_member_corr" in frame_vs_movie.columns
    if has_pairwise_movie:
        axes[1].scatter(
            frame_vs_movie["min_pairwise_member_corr_movie"]
            if "min_pairwise_member_corr_movie" in frame_vs_movie.columns
            else frame_vs_movie.get("min_pairwise_member_corr", np.full(len(frame_vs_movie), np.nan)),
            frame_vs_movie["min_member_centroid_corr_frame"],
            s=np.clip(frame_vs_movie["n_members"] * 3, 8, 140),
            c=frame_vs_movie["n_members"],
            cmap="viridis",
            alpha=0.55,
            linewidths=0,
        )
        axes[1].plot([-0.2, 1.0], [-0.2, 1.0], color="0.25", lw=0.8, ls="--")
        axes[1].axhline(0.75, color="0.55", lw=0.8, ls=":")
        axes[1].axvline(0.75, color="0.55", lw=0.8, ls=":")
        axes[1].set_xlabel("min pairwise member corr (movie)")
        axes[1].set_ylabel("min centroid corr (frame)")
        axes[1].set_title("Pairwise vs centroid corr\n(upper-left = hidden bad merge)")
    else:
        axes[1].scatter(
            frame_vs_movie["min_member_centroid_corr_movie"],
            frame_vs_movie["min_member_centroid_corr_frame"],
            s=np.clip(frame_vs_movie["n_members"] * 3, 8, 140),
            c=frame_vs_movie["n_members"],
            cmap="viridis",
            alpha=0.55,
            linewidths=0,
        )
        axes[1].plot([-0.2, 1.0], [-0.2, 1.0], color="0.25", lw=0.8, ls="--")
        axes[1].axhline(0.75, color="0.55", lw=0.8, ls=":")
        axes[1].axvline(0.75, color="0.55", lw=0.8, ls=":")
        axes[1].set_xlabel("movie min centroid corr")
        axes[1].set_ylabel("single-frame min centroid corr")
        axes[1].set_title("Frame artifact check")

    cond_summary = (
        movie_group_quality.groupby("condition")
        .agg(
            worst=("min_member_centroid_corr", "min"),
            median=("min_member_centroid_corr", "median"),
            below_0p75=("min_member_centroid_corr", lambda x: int(np.sum(np.asarray(x) < 0.75))),
        )
        .reindex(list(dict.fromkeys(movie_group_quality["condition"])))
        .reset_index()
    )
    x = np.arange(len(cond_summary))
    axes[2].bar(x, cond_summary["median"], color="tab:blue", alpha=0.55, label="median")
    axes[2].scatter(x, cond_summary["worst"], color="tab:red", s=34, zorder=3, label="worst")
    axes[2].axhline(0.75, color="0.3", lw=0.9, ls="--")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(cond_summary["condition"].str.replace(" ", "\n"), fontsize=8)
    axes[2].set_ylabel("movie min member-centroid corr")
    axes[2].set_title("Movie coherence by condition")
    axes[2].legend(frameon=False, fontsize=8)
    for xi, n_bad in zip(x, cond_summary["below_0p75"]):
        axes[2].text(xi, 0.03, f"{int(n_bad)}<.75", ha="center", va="bottom", fontsize=7, rotation=90)

    recon = reconstruction_summary.copy()
    xr = np.arange(len(recon))
    axes[3].plot(xr, recon["global_rate_corr"], marker="o", label="global rate corr", color="tab:green")
    axes[3].plot(xr, recon["median_channel_corr"], marker="o", label="median channel corr", color="tab:purple")
    if "p05_channel_corr" in recon:
        axes[3].plot(xr, recon["p05_channel_corr"], marker="o", label="p05 channel corr", color="tab:red", alpha=0.75)
    axes[3].set_xticks(xr)
    axes[3].set_xticklabels(recon["condition"].astype(str).str.replace(" ", "\n"), fontsize=8)
    axes[3].set_ylim(-0.05, 1.02)
    axes[3].set_ylabel("correlation")
    axes[3].set_title("Pool-expand reconstruction correlations")
    axes[3].legend(frameon=False, fontsize=8)

    ax4 = axes[4]
    ax4.bar(xr - 0.18, recon["nrmse_by_full_std"], width=0.36, color="tab:gray", alpha=0.75, label="NRMSE")
    ax4.set_ylabel("NRMSE by full std")
    ax4.set_xticks(xr)
    ax4.set_xticklabels(recon["condition"].astype(str).str.replace(" ", "\n"), fontsize=8)
    ax4b = ax4.twinx()
    ax4b.plot(
        xr,
        recon["mean_abs_expanded_minus_full_ssi"],
        color="tab:orange",
        marker="o",
        label="expanded-full SSI abs",
    )
    ax4b.set_ylabel("pool-expanded SSI abs error")
    ax4.set_title("Reconstruction error scale")
    lines, labels = ax4.get_legend_handles_labels()
    lines_b, labels_b = ax4b.get_legend_handles_labels()
    ax4.legend(lines + lines_b, labels + labels_b, frameon=False, fontsize=8, loc="upper left")

    if "mean_reduced_minus_full_ssi" in recon:
        axes[5].bar(
            xr,
            recon["mean_reduced_minus_full_ssi"],
            color=np.where(recon["mean_reduced_minus_full_ssi"] >= 0, "tab:green", "tab:red"),
            alpha=0.75,
        )
        axes[5].axhline(0, color="0.2", lw=0.8)
        axes[5].set_xticks(xr)
        axes[5].set_xticklabels(recon["condition"].astype(str).str.replace(" ", "\n"), fontsize=8)
        axes[5].set_ylabel("reduced - full SSI")
        axes[5].set_title("Reduced-population SSI effect")
    else:
        axes[5].axis("off")

    fig.suptitle(
        f"{POPULATION_SHORT_NAME} quality metrics: merge reconstruction vs reduced-population effect",
        fontsize=12,
    )
    return fig


def plot_worst_group_quality_table(
    frame_vs_movie: pd.DataFrame,
    *,
    n: int = 20,
) -> plt.Figure:
    pairwise_col = "min_pairwise_member_corr_frame"
    has_pairwise = pairwise_col in frame_vs_movie.columns
    cols = [
        "condition",
        "rep_idx",
        "n_members",
        "min_member_centroid_corr_frame",
        "min_member_centroid_corr_movie",
        *(([pairwise_col]) if has_pairwise else []),
        "worst_member_frame",
        "worst_member_movie",
        "centroid_std_frame",
    ]
    # Sort by pairwise corr first when available — it catches bad merges centroid misses
    sort_cols = ([pairwise_col, "min_member_centroid_corr_frame"] if has_pairwise
                 else ["min_member_centroid_corr_frame", "min_member_centroid_corr_movie"])
    table_df = (
        frame_vs_movie.sort_values(sort_cols)
        .head(n)[[c for c in cols if c in frame_vs_movie.columns]]
        .copy()
    )
    fmt_cols = ["min_member_centroid_corr_frame", "min_member_centroid_corr_movie",
                "centroid_std_frame", pairwise_col]
    for col in fmt_cols:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda v: f"{float(v):.3f}")
    table_df["condition"] = table_df["condition"].astype(str)

    col_labels = [
        "condition", "rep", "n",
        "frame\ncentroid", "movie\ncentroid",
        *(["min\npairwise"] if has_pairwise else []),
        "worst\nframe ch", "worst\nmovie ch", "centroid\nstd",
    ]
    # keep only labels for columns that survived the filter above
    col_labels = col_labels[: len(table_df.columns)]

    fig, ax = plt.subplots(figsize=(14.5, 0.44 * (len(table_df) + 2.5)), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    # colour-code centroid corr columns (indices 3, 4) and pairwise col (5 if present)
    corr_col_indices = {3, 4} | ({5} if has_pairwise else set())
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        elif col in corr_col_indices:
            try:
                value = float(cell.get_text().get_text())
            except ValueError:
                value = 1.0
            if value < 0.60:
                cell.set_facecolor("#f4b6b6")
            elif value < 0.75:
                cell.set_facecolor("#f7ddb0")
    title = ("Worst group-quality rows: sorted by min pairwise member corr (then centroid corr)"
             if has_pairwise else
             "Worst group-quality rows: low frame/movie member-centroid correlations")
    ax.set_title(title, pad=12)
    return fig


fig_quality_dashboard = plot_quality_metric_dashboard(
    movie_group_quality,
    frame_group_quality_from_audit,
    frame_vs_movie,
    reconstruction_summary,
    resplit_estimates,
)
fig_worst_quality_table = plot_worst_group_quality_table(frame_vs_movie, n=20)
if SAVE_FIGURES:
    fig_quality_dashboard.savefig(FIG_DIR / "quality_metric_dashboard.png")
    fig_worst_quality_table.savefig(FIG_DIR / "worst_group_quality_table.png")

# %%
fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
for condition, sub in group_quality.groupby("condition"):
    ax.scatter(
        sub["n_members"],
        sub["min_member_centroid_corr"],
        s=12,
        alpha=0.45,
        label=condition,
    )
ax.axhline(0.75, color="0.3", lw=0.8, ls="--")
ax.set_xlabel("group size")
ax.set_ylabel("min member-centroid corr")
ax.set_title("Group centroid quality on tested held-out maps")
ax.legend(fontsize=7, ncol=2)
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / "group_centroid_quality_scatter.png")

# %%
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), constrained_layout=True)
axes[0].scatter(
    frame_vs_movie["min_member_centroid_corr_movie"],
    frame_vs_movie["min_member_centroid_corr_frame"],
    s=np.clip(frame_vs_movie["n_members"] * 3, 8, 120),
    alpha=0.45,
)
axes[0].plot([-0.2, 1.0], [-0.2, 1.0], color="0.3", lw=0.8, ls="--")
axes[0].axhline(0.75, color="0.5", lw=0.8, ls=":")
axes[0].axvline(0.75, color="0.5", lw=0.8, ls=":")
axes[0].set_xlabel("movie min member-centroid corr")
axes[0].set_ylabel("single-frame min member-centroid corr")
axes[0].set_title("Frame vs full tested movie")

bins = np.linspace(-0.2, 1.0, 49)
axes[1].hist(movie_group_quality["min_member_centroid_corr"], bins=bins, alpha=0.75, label="movie")
axes[1].hist(frame_group_quality_from_audit["min_member_centroid_corr"], bins=bins, alpha=0.55, label="frame")
axes[1].axvline(0.75, color="0.3", lw=0.9, ls="--")
axes[1].set_xlabel("min member-centroid corr")
axes[1].set_ylabel("groups x conditions")
axes[1].set_title("Quality distribution")
axes[1].legend(frameon=False, fontsize=8)

axes[2].scatter(
    movie_group_quality["n_members"],
    movie_group_quality["min_member_centroid_corr"],
    s=12,
    alpha=0.45,
)
axes[2].axhline(0.75, color="0.3", lw=0.9, ls="--")
axes[2].set_xlabel("group size")
axes[2].set_ylabel("movie min member-centroid corr")
axes[2].set_title("Movie quality vs group size")
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / "frame_vs_movie_group_quality.png")

# %% Choose groups to inspect: largest plus worst-on-tested-case groups
largest_groups = (
    representatives.query("n_members > 1")
    .sort_values("n_members", ascending=False)["rep_idx"]
    .head(4)
    .astype(int)
    .tolist()
)
worst_groups = (
    frame_vs_movie.sort_values(["min_member_centroid_corr_frame", "min_member_centroid_corr_movie"])
    .drop_duplicates("rep_idx")["rep_idx"]
    .head(4)
    .astype(int)
    .tolist()
)
worst_movie_groups = (
    movie_group_quality.sort_values("min_member_centroid_corr")
    .drop_duplicates("rep_idx")["rep_idx"]
    .head(4)
    .astype(int)
    .tolist()
)
GROUPS_TO_PLOT = list(dict.fromkeys(largest_groups + worst_groups + worst_movie_groups))
print("largest groups:", largest_groups)
print("worst frame groups:", worst_groups)
print("worst movie groups:", worst_movie_groups)
print("plotting:", GROUPS_TO_PLOT)

# %% Activation maps: selected groups, members, and centroid
def plot_group_activation_maps(
    frame_maps: np.ndarray,
    rr_maps: np.ndarray,
    groups: list[int],
    *,
    condition_label: str,
    max_members: int = 6,
):
    n_rows = len(groups)
    n_cols = max_members + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.45 * n_cols, 1.45 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for row, rep_idx in enumerate(groups):
        weights = cluster_membership[rep_idx]
        members = np.flatnonzero(weights > 0)
        shown = members[:max_members]
        arrays = [frame_maps[m] for m in shown] + [rr_maps[rep_idx]]
        pooled = np.concatenate([a.ravel() for a in arrays])
        vmin, vmax = np.percentile(pooled, [1, 99])
        for col, ax in enumerate(axes[row]):
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(shown):
                ch = int(shown[col])
                ax.imshow(frame_maps[ch], cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(f"ch {ch}", fontsize=7)
            elif col == max_members:
                ax.imshow(rr_maps[rep_idx], cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(f"RR {rep_idx}", fontsize=7)
            else:
                ax.set_axis_off()
        axes[row, 0].set_ylabel(f"rep {rep_idx}\nn={len(members)}", rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(f"Group activation maps: {condition_label}", y=1.02)
    return fig


COND_TO_PLOT = "empirical 1x"
row_idx = int(np.flatnonzero(data["condition_labels"] == COND_TO_PLOT)[0])
fig = plot_group_activation_maps(
    data["full_frame_maps"][row_idx],
    data["reduced_frame_maps"][row_idx],
    GROUPS_TO_PLOT,
    condition_label=COND_TO_PLOT,
)
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / "selected_group_activation_maps.png")

# %% Overlapping temporal mean-rate traces for selected groups
def plot_group_trace_overlays(
    full_mean_traces: np.ndarray,
    rr_mean_traces: np.ndarray,
    groups: list[int],
    *,
    condition_label: str,
):
    n = len(groups)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 2.4 * rows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    t = np.arange(full_mean_traces.shape[0])
    for ax in axes:
        ax.set_axis_off()
    for ax, rep_idx in zip(axes, groups):
        ax.set_axis_on()
        members = np.flatnonzero(cluster_membership[rep_idx] > 0)
        for ch in members:
            ax.plot(t, full_mean_traces[:, ch], color="0.65", lw=0.8, alpha=0.45)
        ax.plot(t, rr_mean_traces[:, rep_idx], color="tab:orange", lw=2.0, label="RR centroid")
        ax.set_title(f"rep {rep_idx}, n={len(members)}", fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("mean rate")
    fig.suptitle(f"Member traces and {POPULATION_SHORT_NAME} centroid: {condition_label}", y=1.02)
    return fig


fig = plot_group_trace_overlays(
    data["full_mean_traces"][row_idx],
    data["reduced_mean_traces"][row_idx],
    GROUPS_TO_PLOT,
    condition_label=COND_TO_PLOT,
)
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / "selected_group_trace_overlays.png")

# %% Quick per-representative QC table for the selected condition
selected_quality = group_quality[
    (group_quality["condition"] == COND_TO_PLOT)
    & (group_quality["image_index"] == int(data["image_indices"][row_idx]))
].sort_values("min_member_centroid_corr")
selected_quality.head(25)[
    [
        "rep_idx",
        "n_members",
        "min_member_centroid_corr",
        "median_member_centroid_corr",
        "worst_member",
        "min_member_std",
        "centroid_std",
    ]
].style.format(precision=4)

# %%
