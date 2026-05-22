"""backimage_long_sweep_20s.py

Efficient long-duration (default ~20s) BackImage hybrid-eye-trace sweeps.

Key differences vs natimg_digitaltwin_spatialinfo_declan.py:
- Avoids constructing the full lag-embedded stimulus tensor for the entire sequence.
- Computes model responses and spatial SSI metrics in temporal batches.
- Supports long sequences (e.g., 2400 frames @ 120 Hz) without OOM.
- Saves one result pickle per image (resume-friendly).

This script only depends on:
- model + outputs (to build readout + sessions)
- cached backimage fixation results + image cache
- cached fixrsvp fixation pool

Run:
  /home/declan/VisionCore/.venv/bin/python scripts/backimage_long_sweep_20s.py --help
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import pickle
import time
from typing import Any

# Allow `python scripts/backimage_long_sweep_20s.py` to import the `scripts` package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from scripts.spatial_info_cache_declan import (
    load_backimage_fixation_results,
    load_backimage_image_cache,
    load_fixrsvp_fixation_pool,
)

class PopulationReadout(nn.Module):
    def __init__(self, feat_weights: torch.Tensor, biases: torch.Tensor, space_weights: torch.Tensor):
        super().__init__()
        self.features = nn.Conv2d(
            feat_weights.shape[1], feat_weights.shape[0], kernel_size=1, bias=False
        )
        self.features.weight = nn.Parameter(feat_weights, requires_grad=False)
        self.bias = nn.Parameter(biases, requires_grad=False)
        self.space_weights = nn.Parameter(space_weights[:, None, :, :], requires_grad=False)
        self.n_units = int(space_weights.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        space = F.conv2d(feat, self.space_weights, groups=self.n_units, padding="valid")
        return space + self.bias[None, :, None, None]


def get_spatial_readout(model, outputs: list[dict[str, Any]]) -> PopulationReadout:
    """Combine dataset-specific readouts into a single population readout."""
    sessions = [outputs[i]["sess"] for i in range(len(outputs))]

    model_dataset_idx = [i for i, name in enumerate(model.names) if name in sessions]
    cids2use = [
        np.where(outputs[sessions.index(model.names[i])]["ccnorm"]["ccnorm"] > 0.5)[0]
        for i in model_dataset_idx
    ]

    feat_weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    space_weights: list[torch.Tensor] = []

    for i, model_readout_idx in enumerate(model_dataset_idx):
        readout = model.model.readouts[model_readout_idx]
        feat_weight = readout.features.weight.detach().cpu()
        bias = readout.bias.detach().cpu()
        space_weight = readout.compute_gaussian_mask(14, 14, model.device).detach().cpu()

        feat_weight = feat_weight[cids2use[i]]
        bias = bias[cids2use[i]]
        space_weight = space_weight[cids2use[i]]

        feat_weights.append(feat_weight)
        biases.append(bias)
        space_weights.append(space_weight)

    feat_weights_cat = torch.cat(feat_weights, dim=0)
    biases_cat = torch.cat(biases, dim=0)
    space_weights_cat = torch.cat(space_weights, dim=0)
    return PopulationReadout(feat_weights_cat, biases_cat, space_weights_cat)


def eye_deg_to_norm(eye_deg: torch.Tensor, ppd: float, img_size: tuple[int, int]):
    """Convert eye position in degrees (relative to image center) -> grid_sample [-1,1] coords."""
    H, W = img_size
    eye_deg = eye_deg.to(dtype=torch.float32)
    x_pix = eye_deg[:, 0] * float(ppd)
    y_pix = eye_deg[:, 1] * float(ppd)
    x_norm = 2.0 * x_pix / (W - 1)
    y_norm = -2.0 * y_pix / (H - 1)
    return torch.stack((x_norm, y_norm), dim=-1)


def shift_movie_with_eye(
    movie: torch.Tensor,
    eye_xy: torch.Tensor,
    out_size=(100, 100),
    center=(0.0, 0.0),
    mode="bilinear",
    padding_mode="border",
    scale_factor=1.0,
    align_corners=True,
):
    """Resample gaze-contingent movie using grid_sample (copied from mcfarland_sim)."""
    if movie.dim() == 3:
        movie = movie.unsqueeze(1)
        squeeze_C = True
    elif movie.dim() == 4:
        squeeze_C = False
    else:
        raise ValueError("movie must have shape (T,H,W) or (T,C,H,W)")

    T, C, H, W = movie.shape
    device = movie.device
    dtype = movie.dtype
    eye_xy = eye_xy.to(device=device, dtype=dtype)

    outH, outW = out_size
    cx, cy = center

    x_extent = (outW / W) * float(scale_factor)
    y_extent = (outH / H) * float(scale_factor)

    ys = torch.linspace(-y_extent, y_extent, outH, device=device, dtype=dtype)
    xs = torch.linspace(-x_extent, x_extent, outW, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((grid_x + cx, grid_y + cy), dim=-1).unsqueeze(0)

    grid = base_grid - eye_xy.view(T, 1, 1, 2)

    out = F.grid_sample(
        movie,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    if squeeze_C:
        out = out[:, 0]
    return out


DEFAULT_PPD = 37.50476617
DEFAULT_FPS = 120.0

DEFAULT_FIXRSVP_POOL_CACHE = Path("../declan/fixrsvp_fixation_pool.pkl")
DEFAULT_BACKIMAGE_RESULTS_CACHE = Path("../declan/backimage_fixation_results.pkl")
DEFAULT_BACKIMAGE_IMAGE_CACHE = Path("../declan/backimage_image_cache.pkl")


def _load_pickle_if_exists(path: Path | None):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_device() -> torch.device:
    try:
        import importlib

        mod = importlib.import_module("DataYatesV1")
        get_free_device = getattr(mod, "get_free_device")
        dev = get_free_device()
        return torch.device(dev) if not isinstance(dev, torch.device) else dev
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_name(image_file: str) -> str:
    # keep filenames short-ish and filesystem-safe
    return image_file.replace("/", "_")


def _ensure_deg_units(xy: np.ndarray, ppd: float) -> np.ndarray:
    """Convert pixels->deg heuristically if magnitudes look like pixels."""
    xy = np.asarray(xy, dtype=np.float32)
    med_amp = np.nanmedian(np.hypot(xy[:, 0], xy[:, 1]))
    if np.isfinite(med_amp) and med_amp > 5.0:
        return xy / float(ppd)
    return xy


def rescale_fixations_only(trace: np.ndarray, saccade_mask: np.ndarray, eye_scale: float) -> np.ndarray:
    """Rescale only fixational jitter, leaving saccade frames untouched."""
    trace = np.asarray(trace, dtype=np.float32)
    saccade_mask = np.asarray(saccade_mask, dtype=bool)
    out = trace.copy()

    fix_idx = np.where(~saccade_mask)[0]
    if fix_idx.size == 0:
        return out

    # Split contiguous fixation runs
    split_points = np.where(np.diff(fix_idx) > 1)[0]
    runs = np.split(fix_idx, split_points + 1)

    for run in runs:
        if run.size < 2:
            continue
        seg = out[run]
        mu = np.nanmean(seg, axis=0, keepdims=True)
        out[run] = mu + (seg - mu) * float(eye_scale)

    return out


def create_hybrid_eye_trace(
    *,
    fixation_pool: list[np.ndarray],
    saccade_targets: np.ndarray,
    n_saccades: int,
    total_duration: int,
    saccade_duration: int = 6,
    micro_bridge_frames: int = 6,
    max_trim_frames: int = 20,
    min_fix_frames: int = 1,
    min_saccade_sep_deg: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Create a hybrid eye trace of exact length total_duration.

    Fixations are stitched from RSVP fixation bouts (zero-meaned) and re-centered.
    Saccades are linear interpolations to targets.

    Returns
    -------
    trace : (T,2) float32 in degrees
    saccade_mask : (T,) bool
    saccade_times : list[int] start indices for each saccade
    """
    rng = np.random.default_rng(seed)

    saccade_targets = np.asarray(saccade_targets, dtype=np.float32)
    if saccade_targets.ndim != 2 or saccade_targets.shape[1] != 2:
        raise ValueError(f"saccade_targets must be (N,2); got {saccade_targets.shape}")

    bout_lengths = np.array([len(b) for b in fixation_pool], dtype=np.int32)
    if bout_lengths.size == 0:
        raise ValueError("fixation_pool is empty")

    # Cap saccades to what can fit given min fixation frames.
    # Need (n_saccades + 1) fixations and n_saccades saccades.
    # Each fixation has at least min_fix_frames.
    max_sacc = int((total_duration - min_fix_frames) // (saccade_duration + min_fix_frames))
    n_saccades = int(np.clip(n_saccades, 0, max_sacc))

    # initial center
    current_center = saccade_targets[rng.integers(0, len(saccade_targets))].astype(np.float32)

    def trim_bout(bout: np.ndarray, target_len: int) -> np.ndarray | None:
        L = len(bout)
        if L == target_len:
            return bout
        if L > target_len and (L - target_len) <= max_trim_frames:
            start = rng.integers(0, L - target_len + 1)
            return bout[start : start + target_len]
        return None

    def compose_fixation_sequence(target_len: int, center: np.ndarray) -> np.ndarray:
        residual = int(target_len)
        parts: list[np.ndarray] = []
        prev_last = None

        while residual > 0:
            avail_for_seg = residual if prev_last is None else max(residual - micro_bridge_frames, 1)

            length_diff = np.abs(bout_lengths - avail_for_seg)
            ranked = np.argsort(length_diff)
            seg = None

            # Try near-length bouts with trimming.
            k = min(50, len(ranked))
            for idx in rng.permutation(ranked[:k]):
                bout = fixation_pool[int(idx)]
                if len(bout) == avail_for_seg:
                    seg = bout
                    break
                seg_try = trim_bout(bout, avail_for_seg)
                if seg_try is not None:
                    seg = seg_try
                    break

            # Last resort: clamp a random bout.
            if seg is None:
                bout = fixation_pool[rng.integers(0, len(fixation_pool))]
                if len(bout) >= avail_for_seg:
                    start = rng.integers(0, len(bout) - avail_for_seg + 1)
                    seg = bout[start : start + avail_for_seg]
                else:
                    seg = bout

            seg = np.asarray(seg, dtype=np.float32)
            jitter = seg - seg.mean(axis=0, keepdims=True)
            seg_centered = center[None, :] + jitter

            if prev_last is not None:
                bridge_len = min(micro_bridge_frames, residual)
                bridge = np.linspace(prev_last, seg_centered[0], num=bridge_len, dtype=np.float32)
                parts.append(bridge)
                residual -= bridge_len

            take = min(len(seg_centered), residual)
            parts.append(seg_centered[:take])
            residual -= take
            prev_last = parts[-1][-1]

        return np.vstack(parts)

    # allocate
    trace_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    saccade_times: list[int] = []

    T = int(total_duration)
    saccade_frames_total = int(n_saccades) * int(saccade_duration)
    fixation_frames_total = T - saccade_frames_total

    # distribute fix frames across (n_saccades+1) fixations with a minimum
    n_fix = n_saccades + 1
    base = fixation_frames_total // n_fix
    rem = fixation_frames_total - base * n_fix
    fix_lens = [max(min_fix_frames, base + (1 if i < rem else 0)) for i in range(n_fix)]

    # fix_lens might overshoot due to min_fix_frames; correct by trimming from the end
    over = sum(fix_lens) - fixation_frames_total
    i = n_fix - 1
    while over > 0 and i >= 0:
        take = min(over, max(0, fix_lens[i] - min_fix_frames))
        fix_lens[i] -= take
        over -= take
        i -= 1

    frame_idx = 0
    for i_fix in range(n_fix):
        flen = int(fix_lens[i_fix])
        if flen > 0:
            seg = compose_fixation_sequence(flen, current_center)
            trace_parts.append(seg)
            mask_parts.append(np.zeros(flen, dtype=bool))
            frame_idx += flen

        if i_fix < n_saccades:
            # saccade to next target
            if float(min_saccade_sep_deg) <= 0.0:
                tgt = saccade_targets[rng.integers(0, len(saccade_targets))].astype(np.float32)
            else:
                # try a limited number of random draws to enforce minimum separation
                K = 50
                tgt = None
                for _ in range(K):
                    cand = saccade_targets[rng.integers(0, len(saccade_targets))].astype(np.float32)
                    if np.hypot(*(cand - current_center)) >= float(min_saccade_sep_deg):
                        tgt = cand
                        break
                if tgt is None:
                    # fallback: choose the farthest available target
                    dists = np.hypot(saccade_targets[:, 0] - current_center[0], saccade_targets[:, 1] - current_center[1])
                    idx = int(np.argmax(dists))
                    tgt = saccade_targets[idx].astype(np.float32)

            saccade_times.append(frame_idx)

            s = int(saccade_duration)
            if s > 0:
                start = trace_parts[-1][-1] if trace_parts else current_center
                sac = np.linspace(start, tgt, num=s, dtype=np.float32)
                trace_parts.append(sac)
                mask_parts.append(np.ones(s, dtype=bool))
                frame_idx += s
            current_center = tgt

    trace = np.vstack(trace_parts) if trace_parts else np.zeros((0, 2), dtype=np.float32)
    saccade_mask = np.concatenate(mask_parts) if mask_parts else np.zeros((0,), dtype=bool)

    # Clamp/pad to exact length
    if trace.shape[0] > T:
        trace = trace[:T]
        saccade_mask = saccade_mask[:T]
    elif trace.shape[0] < T:
        pad = T - trace.shape[0]
        trace = np.vstack([trace, np.repeat(trace[-1][None, :], pad, axis=0)]) if trace.shape[0] else np.zeros((T, 2), dtype=np.float32)
        saccade_mask = np.concatenate([saccade_mask, np.zeros(pad, dtype=bool)])

    return trace.astype(np.float32), saccade_mask.astype(bool), saccade_times


def _build_eye_movie(
    *,
    full_stack: np.ndarray,
    eyepos_deg: np.ndarray,
    ppd: float,
    out_size: tuple[int, int],
    scale_factor: float,
    n_lags: int,
    device: torch.device,
    window_radius_deg: float | None = None,
) -> torch.Tensor:
    """Return eye_movie (T + n_lags, H, W) on device."""
    eyepos_deg = np.asarray(eyepos_deg, dtype=np.float32)
    eyepos = torch.from_numpy(eyepos_deg).to(device)

    # NOTE: eye_deg_to_norm expects (x,y) but codebase flips; match spatial_info.make_counterfactual_stim
    eye_norm = eye_deg_to_norm(torch.fliplr(eyepos), float(ppd), full_stack.shape[1:3])

    # shift_movie_with_eye expects frames aligned with eye trace length
    frames = torch.from_numpy(full_stack[: eyepos.shape[0] + n_lags]).float().to(device)

    # pad beginning with first n_lags eye positions
    eye_path = torch.cat([eye_norm[:n_lags], eye_norm], dim=0)

    # If a per-frame window radius in degrees is specified, override out_size
    if window_radius_deg is not None:
        r_px = int(round(float(window_radius_deg) * float(ppd)))
        # make odd-sized window (diameter = 2*r + 1)
        out_h = out_w = max(1, 2 * r_px + 1)
        out_size_use = (out_h, out_w)
    else:
        out_size_use = out_size

    eye_movie = shift_movie_with_eye(
        frames,
        eye_path,
        out_size=out_size_use,
        center=(0.0, 0.0),
        scale_factor=float(scale_factor),
        mode="bilinear",
        padding_mode="border",
    )
    return eye_movie


def _save_eye_movie_mp4(path: Path, eye_movie: torch.Tensor, n_lags: int, fps: float = 30.0):
    """Save `eye_movie` (T_total, H, W) -> mp4 using matplotlib's FFMpegWriter.

    This produces a more portable MP4 when system `ffmpeg` is available.
    If that fails, fall back to an animated GIF (Pillow) and finally `.npy`.
    """
    try:
        arr = eye_movie.detach().cpu().numpy()
    except Exception:
        arr = eye_movie.cpu().numpy()

    # drop initial lag padding
    if arr.shape[0] > n_lags:
        frames = arr[n_lags:]
    else:
        frames = arr

    if frames.size == 0:
        return

    # robust normalization
    vmin = float(np.percentile(frames, 1.0))
    vmax = float(np.percentile(frames, 99.0))
    if not np.isfinite(vmin):
        vmin = float(frames.min())
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    # scale to 0-255 uint8 for display
    frames_u8 = np.clip((frames - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)

    path_parent = Path(path).parent
    path_parent.mkdir(parents=True, exist_ok=True)

    # Try matplotlib + FFMpegWriter
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")

        # display first frame
        im = ax.imshow(frames_u8[0], cmap="gray", vmin=0, vmax=255)

        writer = FFMpegWriter(fps=float(fps), codec="libx264", bitrate=8000)
        with writer.saving(fig, str(path), dpi=100):
            for f in frames_u8:
                im.set_data(f)
                writer.grab_frame()

        plt.close(fig)
        return
    except Exception:
        # Fall back to GIF/.npy if ffmpeg or writer not available
        try:
            plt.close("all")
        except Exception:
            pass

    # Fallback: animated GIF via Pillow (no ffmpeg required)
    try:
        from PIL import Image

        pil_frames = []
        for f in frames_u8:
            if f.ndim == 2:
                img = Image.fromarray(f, mode="L").convert("P", palette=Image.ADAPTIVE)
            else:
                img = Image.fromarray(f.astype(np.uint8))
            pil_frames.append(img)

        gif_path = Path(path).with_suffix(".gif")
        duration_ms = int(1000.0 / float(fps)) if fps > 0 else 100
        if len(pil_frames) == 1:
            pil_frames[0].save(str(gif_path))
        else:
            pil_frames[0].save(
                str(gif_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
        return
    except Exception:
        # GIF write failed; fall through to raw numpy save
        pass

    # Last resort: write raw numpy array for offline conversion
    try:
        npy_path = Path(path).with_suffix(".npy")
        np.save(str(npy_path), frames_u8)
    except Exception:
        return


def _iter_lagged_batches(eye_movie: torch.Tensor, *, n_lags: int, batch_size: int):
    """Yield lagged stimulus batches (B,1,n_lags,H,W) and corresponding time indices."""
    # eye_movie: (T_total, H, W)
    T_total = int(eye_movie.shape[0])
    out_T = T_total - n_lags + 1
    device = eye_movie.device

    lags = torch.arange(n_lags - 1, -1, -1, device=device)  # current -> past ordering

    for t_start in range(0, out_T, batch_size):
        t_end = min(t_start + batch_size, out_T)
        t_idx = torch.arange(t_start, t_end, device=device)
        idx = t_idx[:, None] + lags[None, :]
        stim = eye_movie[idx]  # (B, n_lags, H, W)
        stim = stim.unsqueeze(1)  # (B,1,n_lags,H,W)
        yield stim, (t_start, t_end)


def compute_spatial_info_streaming(
    *,
    model,
    readout,
    eye_movie: torch.Tensor,
    n_lags: int,
    batch_size: int,
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    """Compute mean(ispike_t), mean(irate_t), mean(I_tn) without storing full y."""
    model.model.eval()
    readout.eval()

    sum_ispike = 0.0
    sum_irate = 0.0
    sum_I_tn = 0.0
    n_t = 0
    n_I = 0

    device = next(model.model.parameters()).device

    with torch.no_grad():
        for stim, _ in _iter_lagged_batches(eye_movie, n_lags=n_lags, batch_size=batch_size):
            # normalize like existing code
            stim = (stim - 127.0) / 255.0
            stim = stim.to(device)

            x = model.model.core_forward(stim, None)
            y = readout(x[:, :, -1])
            y = model.model.activation(y)
            y = y.cpu()  # free GPU quickly

            # y: (B,N,H,W)
            B, N, H, W = y.shape
            P = H * W

            r = y.reshape(B, N, P)
            rbar = r.mean(dim=2)  # (B,N)
            g = r / (rbar[..., None] + eps)
            logg = torch.log2(g + eps)
            I_tn = (g * logg).mean(dim=2)  # (B,N)

            spikes_bn = rbar  # dt=1.0 to match existing scripts
            bits_t = (spikes_bn * I_tn).sum(dim=1)  # (B,)
            spikes_t = spikes_bn.sum(dim=1)  # (B,)
            ispike_t = bits_t / (spikes_t + eps)

            irate_t = (rbar * I_tn).sum(dim=1)  # (B,)

            sum_ispike += float(ispike_t.sum().item())
            sum_irate += float(irate_t.sum().item())
            sum_I_tn += float(I_tn.sum().item())
            n_t += int(B)
            n_I += int(I_tn.numel())

    mean_ispike = sum_ispike / max(1, n_t)
    mean_irate = sum_irate / max(1, n_t)
    mean_I_tn = sum_I_tn / max(1, n_I)
    return mean_ispike, mean_irate, mean_I_tn


@dataclass
class SweepConfig:
    duration_sec: float = 20.0
    fps: float = DEFAULT_FPS
    n_lags: int = 32
    out_size: tuple[int, int] = (151, 151)
    stim_scale: float = 1.0
    ppd: float = DEFAULT_PPD
    saccade_duration_frames: int = 6
    min_fix_frames: int = 1
    batch_size: int = 16
    saccade_rates_hz: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    eye_scales: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    window_radius_deg: float = 1.0
    trials_per_condition: int = 3
    videos_per_image: int = 3
    min_saccade_sep_deg: float = 0.0


def run_long_sweep_all_images(
    *,
    model,
    outputs: list[dict[str, Any]],
    config: SweepConfig,
    save_dir: Path,
    resume: bool,
    max_images: int | None,
    fixrsvp_pool_cache: Path | None = DEFAULT_FIXRSVP_POOL_CACHE,
    backimage_results_cache: Path | None = DEFAULT_BACKIMAGE_RESULTS_CACHE,
    backimage_image_cache: Path | None = DEFAULT_BACKIMAGE_IMAGE_CACHE,
    no_cache_build: bool = False,
    save_videos_dir: Path | None = None,
    max_videos: int = 10,
) -> dict[str, Any]:
    device = next(model.model.parameters()).device

    readout = get_spatial_readout(model, outputs).to(device)
    sessions = [outputs[i]["sess"] for i in range(len(outputs))]

    # Prefer loading precomputed caches directly to avoid repeated dataset loads.
    fixation_pool = _load_pickle_if_exists(fixrsvp_pool_cache)
    if fixation_pool is None:
        if no_cache_build:
            raise FileNotFoundError(
                f"Missing fixation_pool cache at {fixrsvp_pool_cache}. Run once without --no-cache-build to create it."
            )
        fixation_pool = load_fixrsvp_fixation_pool(
            model=model,
            sessions=sessions,
            min_fix_frames=20,
            amp_thresh_deg=1.0,
            force_recompute=False,
        )

    backimage_results = _load_pickle_if_exists(backimage_results_cache)
    image_cache = _load_pickle_if_exists(backimage_image_cache)

    if backimage_results is None or image_cache is None:
        if no_cache_build:
            raise FileNotFoundError(
                "Missing BackImage caches. Expected both:\n"
                f"  - {backimage_results_cache}\n"
                f"  - {backimage_image_cache}\n"
                "Run once without --no-cache-build to create them (slow), then rerun."
            )
        backimage_results = load_backimage_fixation_results(model=model, sessions=sessions)
        image_cache = load_backimage_image_cache(model=model, sessions=sessions, results=backimage_results)

    # Sort images by n_trials
    items = sorted(backimage_results.items(), key=lambda kv: -int(kv[1].get("n_trials", 0)))
    if max_images is not None:
        items = items[: int(max_images)]

    save_dir.mkdir(parents=True, exist_ok=True)

    # prepare optional videos directory
    if save_videos_dir is not None:
        save_videos_dir = Path(save_videos_dir)
        save_videos_dir.mkdir(parents=True, exist_ok=True)
    saved_videos = 0
    rng = np.random.default_rng()

    T = int(round(config.duration_sec * config.fps))

    summary = {"completed": 0, "skipped": 0, "failed": 0, "save_dir": str(save_dir), "T": T}

    for image_file, meta in items:
        videos_saved_for_image = 0
        out_path = save_dir / f"sweep20s_{_safe_name(image_file)}.pkl"
        if resume and out_path.exists():
            summary["skipped"] += 1
            continue

        try:
            fixation_targets = _ensure_deg_units(np.asarray(meta["eyepos"], dtype=np.float32), config.ppd)
            img = image_cache.get(image_file)
            if img is None:
                raise ValueError(f"Missing pixels for {image_file!r} in image_cache")

            # Debug: image size in pixels and degrees
            H_img, W_img = img.shape
            half_w_deg = (W_img - 1) / (2.0 * float(config.ppd))
            half_h_deg = (H_img - 1) / (2.0 * float(config.ppd))
            print(f"DEBUG image={image_file} size_px=({H_img},{W_img}) ppd={config.ppd:.3f} half_deg=(w={half_w_deg:.3f},h={half_h_deg:.3f})")

            # Debug: saccade targets distances from center (deg, px)
            if fixation_targets.size:
                r_deg = np.hypot(fixation_targets[:, 0], fixation_targets[:, 1])
                r_px = r_deg * float(config.ppd)
                print(f"DEBUG saccade_targets: n={len(r_deg)} r_deg min/max/mean={r_deg.min():.3f}/{r_deg.max():.3f}/{r_deg.mean():.3f} deg; r_px min/max/mean={r_px.min():.1f}/{r_px.max():.1f}/{r_px.mean():.1f} px")

                # Remove targets that fall outside the image bounds (off-screen)
                in_w = np.abs(fixation_targets[:, 0]) <= half_w_deg
                in_h = np.abs(fixation_targets[:, 1]) <= half_h_deg
                keep_mask = in_w & in_h
                n_keep = int(np.count_nonzero(keep_mask))
                n_total = len(keep_mask)
                if n_keep < n_total:
                    print(f"DEBUG removing {n_total-n_keep}/{n_total} off-screen saccade targets for {image_file}")
                    fixation_targets = fixation_targets[keep_mask]
                # If no targets remain, replace with a single center target (0,0)
                if fixation_targets.size == 0:
                    print(f"WARNING: no on-screen saccade targets remain for {image_file}; substituting center target [0,0]")
                    fixation_targets = np.asarray([[0.0, 0.0]], dtype=np.float32)

                # Plot and save a 2D histogram of saccade positions (degrees)
                try:
                    plot_dir = save_videos_dir if save_videos_dir is not None else save_dir
                    if plot_dir is not None:
                        plot_dir = Path(plot_dir)
                        plot_dir.mkdir(parents=True, exist_ok=True)
                        x = fixation_targets[:, 0]
                        y = fixation_targets[:, 1]
                        # Use image-centric bounds (degrees)
                        xlim = (-half_w_deg, half_w_deg)
                        ylim = (-half_h_deg, half_h_deg)
                        # reasonable bin count for visualization
                        bins = 200
                        fig, ax = plt.subplots(figsize=(6, 5))
                        h = ax.hist2d(x, y, bins=bins, range=[xlim, ylim], cmap="viridis")
                        ax.set_xlabel("x (deg)")
                        ax.set_ylabel("y (deg)")
                        ax.set_title(f"Saccade positions: {image_file}")
                        cbar = fig.colorbar(h[3], ax=ax)
                        cbar.set_label("counts")
                        hist_path = plot_dir / f"{_safe_name(image_file)}_saccade2d_hist.png"
                        fig.savefig(hist_path, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                except Exception as e:
                    print(f"WARNING: failed to write saccade 2D histogram for {image_file}: {e}")

            # Build static stack (T + n_lags) frames
            stim_len = T + config.n_lags
            full_stack = np.repeat(img[None, :, :].astype(np.float32), stim_len, axis=0)

            results = {
                "image_file": image_file,
                "n_trials": int(meta.get("n_trials", -1)),
                "n_sessions": int(meta.get("n_sessions", -1)),
                "img_hw": tuple(img.shape),
                "ppd": float(config.ppd),
                "duration_sec": float(config.duration_sec),
                "fps": float(config.fps),
                "T_frames": int(T),
                "n_lags": int(config.n_lags),
                "out_size": tuple(config.out_size),
                "stim_scale": float(config.stim_scale),
                "saccade_duration_frames": int(config.saccade_duration_frames),
                "min_fix_frames": int(config.min_fix_frames),
                "batch_size": int(config.batch_size),
                "saccade_rates_hz": list(config.saccade_rates_hz),
                "eye_scales": list(config.eye_scales),
                "i_spikes": [],
                "i_rates": [],
                "I_t": [],
                "trial_seeds": [],
                "n_saccades_realized": [],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            trials = int(getattr(config, "trials_per_condition", 1))
            # determine high-rate threshold (75th percentile) to prioritize saving those videos
            try:
                rates_arr = np.asarray(config.saccade_rates_hz, dtype=float)
                rate_threshold = float(np.percentile(rates_arr, 75.0))
            except Exception:
                rate_threshold = float(max(config.saccade_rates_hz))
            for sacc_rate in config.saccade_rates_hz:
                n_sacc_target = int(round(float(sacc_rate) * float(config.duration_sec)))
                # containers per saccade-rate
                sp_trials: list[list[float]] = []
                rt_trials: list[list[float]] = []
                It_trials: list[list[float]] = []
                saccs_realized: list[int] = []

                for t_i in range(trials):
                    seed = int(rng.integers(0, 2 ** 31 - 1))
                    # persist seed so this trial can be exactly reproduced later
                    if "trial_seeds" not in results:
                        results["trial_seeds"] = []
                    # ensure we have a per-saccade-rate container
                    if len(results["trial_seeds"]) <= len(sp_trials):
                        # pad up to current saccade-rate index with empty lists
                        while len(results["trial_seeds"]) <= len(sp_trials):
                            results["trial_seeds"].append([])
                    results["trial_seeds"][len(sp_trials)].append(int(seed))
                    base_trace, base_mask, sacc_times = create_hybrid_eye_trace(
                        fixation_pool=fixation_pool,
                        saccade_targets=fixation_targets,
                        n_saccades=n_sacc_target,
                        total_duration=T,
                        saccade_duration=config.saccade_duration_frames,
                        min_fix_frames=config.min_fix_frames,
                        min_saccade_sep_deg=float(getattr(config, "min_saccade_sep_deg", 0.0)),
                        seed=seed,
                    )

                    row_sp: list[float] = []
                    row_rt: list[float] = []
                    row_It: list[float] = []

                    for eye_scale in config.eye_scales:
                        trace = rescale_fixations_only(base_trace, base_mask, float(eye_scale))

                        # Build eye_movie on GPU, then stream through model
                        eye_movie = _build_eye_movie(
                            full_stack=full_stack,
                            eyepos_deg=trace,
                            ppd=config.ppd,
                            out_size=config.out_size,
                            scale_factor=config.stim_scale,
                            n_lags=config.n_lags,
                            device=device,
                            window_radius_deg=float(config.window_radius_deg) if getattr(config, "window_radius_deg", None) is not None else None,
                        )

                        # Optionally save a few example videos per-image (resume-friendly).
                        # Prioritize higher saccade rates so we collect examples at fast rates.
                        should_save_video = False
                        if save_videos_dir is not None and saved_videos < int(max_videos) and videos_saved_for_image < int(getattr(config, "videos_per_image", 1)):
                            # Always allow at least one video per image; otherwise prefer high rates
                            if videos_saved_for_image == 0 or float(sacc_rate) >= rate_threshold:
                                should_save_video = True

                        if should_save_video:
                            try:
                                vid_name = f"{_safe_name(image_file)}_s{str(sacc_rate).replace('.', 'p')}_t{t_i}_e{str(eye_scale).replace('.', 'p')}.mp4"
                                _save_eye_movie_mp4(save_videos_dir / vid_name, eye_movie, n_lags=config.n_lags, fps=min(30.0, float(config.fps)))
                                saved_videos += 1
                                videos_saved_for_image += 1
                            except Exception:
                                pass

                        ispike_mean, irate_mean, It_mean = compute_spatial_info_streaming(
                            model=model,
                            readout=readout,
                            eye_movie=eye_movie,
                            n_lags=config.n_lags,
                        batch_size=config.batch_size,
                        )

                        row_sp.append(float(ispike_mean))
                        row_rt.append(float(irate_mean))
                        row_It.append(float(It_mean))

                        # free per-condition GPU memory
                        del eye_movie
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    sp_trials.append(row_sp)
                    rt_trials.append(row_rt)
                    It_trials.append(row_It)
                    saccs_realized.append(int(len(sacc_times)))

                results["i_spikes"].append(sp_trials)
                results["i_rates"].append(rt_trials)
                results["I_t"].append(It_trials)
                results["n_saccades_realized"].append(saccs_realized)

            results["i_spikes"] = np.asarray(results["i_spikes"], dtype=np.float32)
            results["i_rates"] = np.asarray(results["i_rates"], dtype=np.float32)
            results["I_t"] = np.asarray(results["I_t"], dtype=np.float32)
            results["n_saccades_realized"] = np.asarray(results["n_saccades_realized"], dtype=np.int32)
            # Persist trial RNG seeds for exact reproducibility: shape (n_rates, trials)
            try:
                results["trial_seeds"] = np.asarray(results.get("trial_seeds", []), dtype=np.int64)
            except Exception:
                # If conversion fails for some reason, store as object array
                results["trial_seeds"] = np.asarray(results.get("trial_seeds", []), dtype=object)

            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(results, f)
            tmp.replace(out_path)

            summary["completed"] += 1

        except Exception as e:
            summary["failed"] += 1
            (save_dir / f"sweep20s_{_safe_name(image_file)}.ERROR.txt").write_text(str(e))

    # Save a run summary json
    (save_dir / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    return summary


def _parse_floats_csv(s: str) -> tuple[float, ...]:
    xs = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        xs.append(float(part))
    return tuple(xs)


def main():
    ap = argparse.ArgumentParser(description="Long (~20s) BackImage sweep with streaming SSI.")
    ap.add_argument("--outputs", type=str, required=True, help="Path to dill/pickle outputs list used to build the spatial readout")
    ap.add_argument("--save-dir", type=str, default="/home/declan/VisionCore/declan/overnight_backimage_long_sweeps_20s", help="Directory to write per-image results")
    ap.add_argument("--resume", action="store_true", help="Skip images with existing output pickles")
    ap.add_argument("--max-images", type=int, default=None, help="Limit number of images (debug)")

    ap.add_argument(
        "--no-cache-build",
        action="store_true",
        help="Fail fast if required caches are missing (prevents accidental multi-hour dataset scans)",
    )
    ap.add_argument(
        "--fixrsvp-pool-cache",
        type=str,
        default=str(DEFAULT_FIXRSVP_POOL_CACHE),
        help="Path to fixation_pool pickle (default: ../declan/fixrsvp_fixation_pool.pkl)",
    )
    ap.add_argument(
        "--backimage-results-cache",
        type=str,
        default=str(DEFAULT_BACKIMAGE_RESULTS_CACHE),
        help="Path to backimage_fixation_results pickle (default: ../declan/backimage_fixation_results.pkl)",
    )
    ap.add_argument(
        "--backimage-image-cache",
        type=str,
        default=str(DEFAULT_BACKIMAGE_IMAGE_CACHE),
        help="Path to backimage_image_cache pickle (default: ../declan/backimage_image_cache.pkl)",
    )

    ap.add_argument("--duration-sec", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=float(DEFAULT_FPS))
    ap.add_argument("--n-lags", type=int, default=32)
    ap.add_argument("--out-size", type=str, default="151,151")
    ap.add_argument("--stim-scale", type=float, default=1.0)
    ap.add_argument("--ppd", type=float, default=float(DEFAULT_PPD))

    ap.add_argument("--saccade-rates", type=str, default="0,0.25,0.5,1,2,4,8,16")
    ap.add_argument("--eye-scales", type=str, default="0,0.5,1,2")
    ap.add_argument("--saccade-duration-frames", type=int, default=6)
    ap.add_argument("--min-fix-frames", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--save-videos-dir",
        type=str,
        default=None,
        help="Optional directory to write a few gaze-corrected mp4 examples",
    )
    ap.add_argument(
        "--max-videos",
        type=int,
        default=10,
        help="Maximum number of example videos to save (default 10)",
    )
    ap.add_argument(
        "--window-radius-deg",
        type=float,
        default=1.0,
        help="Radius (deg) of per-frame gaze window to sample around center of gaze (default 1.0)",
    )
    ap.add_argument(
        "--trials-per-condition",
        type=int,
        default=3,
        help="Number of randomized trials per saccade-rate condition (default 3)",
    )
    ap.add_argument(
        "--videos-per-image",
        type=int,
        default=3,
        help="Number of example videos to save per image (default 3)",
    )
    ap.add_argument(
        "--min-saccade-sep-deg",
        type=float,
        default=0.0,
        help="Minimum angular separation (deg) between successive saccade targets when sampling (default 0.0)",
    )

    args = ap.parse_args()

    # Load model
    from utils import get_model_and_dataset_configs

    model, _ = get_model_and_dataset_configs()
    device = _get_device()
    model = model.to(device)

    # Load outputs
    import dill

    outputs_path = Path(args.outputs)
    if not outputs_path.exists():
        raise FileNotFoundError(f"Outputs file not found: {outputs_path}")
    with open(outputs_path, "rb") as f:
        outputs = dill.load(f)

    out_size = tuple(int(x) for x in args.out_size.split(","))

    cfg = SweepConfig(
        duration_sec=float(args.duration_sec),
        fps=float(args.fps),
        n_lags=int(args.n_lags),
        out_size=(int(out_size[0]), int(out_size[1])),
        stim_scale=float(args.stim_scale),
        ppd=float(args.ppd),
        saccade_duration_frames=int(args.saccade_duration_frames),
        min_fix_frames=int(args.min_fix_frames),
        saccade_rates_hz=_parse_floats_csv(args.saccade_rates),
        eye_scales=_parse_floats_csv(args.eye_scales),
        window_radius_deg=float(args.window_radius_deg),
        trials_per_condition=int(args.trials_per_condition),
        videos_per_image=int(args.videos_per_image),
        min_saccade_sep_deg=float(args.min_saccade_sep_deg),
    )

    save_dir = Path(args.save_dir)
    summary = run_long_sweep_all_images(
        model=model,
        outputs=outputs,
        config=cfg,
        save_dir=save_dir,
        resume=bool(args.resume),
        max_images=args.max_images,
        fixrsvp_pool_cache=Path(args.fixrsvp_pool_cache) if args.fixrsvp_pool_cache else None,
        backimage_results_cache=Path(args.backimage_results_cache) if args.backimage_results_cache else None,
        backimage_image_cache=Path(args.backimage_image_cache) if args.backimage_image_cache else None,
        no_cache_build=bool(args.no_cache_build),
        save_videos_dir=Path(args.save_videos_dir) if args.save_videos_dir else None,
        max_videos=int(args.max_videos),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
