"""Model-facing Vernier movie construction and digital-twin forward passes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dill
import numpy as np
import torch

from scripts.temporal_decoding.rate_computation import _collapse_spatial, compute_trial_rates
from scripts.temporal_decoding.stimulus_hires import N_LAGS
from scripts.spatial_info import embed_time_lags

from .stimulus import RenderGeometry, VernierSpec, render_world, sample_retina_movie


ROOT = Path(__file__).resolve().parents[2]
PKL_PATH = ROOT / "scripts" / "mcfarland_outputs_mono.pkl"
STIMULUS_NORMALIZATION = "pixelnorm_renderer_raw_scaled_to_u8_minus_127_div_255"


def renderer_raw_to_model_pixelnorm(raw: torch.Tensor, *, max_raw: float) -> torch.Tensor:
    """Convert Vernier renderer-local raw luminance to dataset pixelnorm units.

    ``RenderGeometry`` uses a local audited range of ``0..max_raw`` with neutral
    gray at roughly ``max_raw / 2``. The twin expects the dataset pixelnorm
    convention over raw 8-bit values, so first map the renderer-local range onto
    ``0..255`` and then apply ``(raw_u8 - 127) / 255``.
    """
    scale = 255.0 / float(max_raw)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"max_raw must be positive and finite, got {max_raw!r}")
    raw_u8 = raw.to(dtype=torch.float32) * scale
    return (raw_u8 - 127.0) / 255.0


def load_model_and_readout(device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from scripts.utils import get_model_and_dataset_configs
    from scripts.spatial_info import get_spatial_readout

    model, _ = get_model_and_dataset_configs(mode="standard")
    model.model.eval()
    if hasattr(model.model, "convnet") and hasattr(model.model.convnet, "use_checkpointing"):
        model.model.convnet.use_checkpointing = False
    model = model.to(device)
    with PKL_PATH.open("rb") as handle:
        outputs = dill.load(handle)
    readout = get_spatial_readout(model, outputs).to(device)
    readout.eval()
    return model, readout


def build_vernier_movie(
    spec: VernierSpec,
    eye_trace_deg: np.ndarray,
    *,
    geometry: RenderGeometry | None = None,
    n_lags: int = N_LAGS,
    device: str = "cpu",
) -> torch.Tensor:
    """Return lag-embedded model stimulus shaped ``(T, 1, n_lags, H, W)``.

    Model-bound stimuli are returned in dataset pixelnorm units, not
    display-normalized ``[0, 1]`` units.
    Keep this path synchronized with ``compute_vernier_rates_continuous`` and
    lag-diagnostic finite-difference paths when changing normalization.
    See ``docs/digital_twin_stimulus_normalization.md``.
    """
    geom = geometry or RenderGeometry()
    eye = torch.as_tensor(eye_trace_deg, dtype=torch.float32, device=device)
    world = render_world(spec, geom, device=device)
    padded_eye = torch.cat([eye[:1].expand(max(int(n_lags) - 1, 0), -1), eye], dim=0)
    movie = sample_retina_movie(world, padded_eye, geometry=geom, device=device)[0, 0]
    stim = embed_time_lags(movie.detach().cpu(), n_lags=int(n_lags))
    return renderer_raw_to_model_pixelnorm(stim, max_raw=float(geom.max_raw))


def compute_vernier_rates(
    model: Any,
    readout: Any,
    spec: VernierSpec,
    eye_trace_deg: np.ndarray,
    *,
    inference_mode: str = "framewise",
    geometry: RenderGeometry | None = None,
    n_lags: int = N_LAGS,
    batch_size: int = 32,
    spatial_collapse: str = "max",
    device: str | None = None,
) -> np.ndarray:
    """Run one Vernier movie through the twin.

    ``framewise`` uses the validated lag-window path from the E-optotype cache.
    ``continuous`` feeds the sampled movie through the recurrent module once.
    """
    if device is None:
        device = str(next(model.model.parameters()).device)
    if inference_mode == "framewise":
        stim = build_vernier_movie(spec, eye_trace_deg, geometry=geometry, n_lags=n_lags, device=device)
        return compute_trial_rates(model, readout, stim, batch_size=int(batch_size), spatial_collapse=spatial_collapse)
    if inference_mode == "continuous":
        return compute_vernier_rates_continuous(
            model,
            readout,
            spec,
            eye_trace_deg,
            geometry=geometry,
            prepad_frames=int(n_lags),
            spatial_collapse=spatial_collapse,
            device=device,
        )
    raise ValueError(f"Unsupported inference_mode: {inference_mode}")


def compute_vernier_rates_continuous(
    model: Any,
    readout: Any,
    spec: VernierSpec,
    eye_trace_deg: np.ndarray,
    *,
    geometry: RenderGeometry | None = None,
    prepad_frames: int = N_LAGS,
    spatial_collapse: str = "max",
    device: str | None = None,
) -> np.ndarray:
    """Continuous ConvGRU movie inference, matching ``eoptotype_continuous_pass``.

    The recurrent frontend receives the same pixelnorm stimulus convention used
    for training/evaluation.
    See ``docs/digital_twin_stimulus_normalization.md``.
    """
    if device is None:
        device = str(next(model.model.parameters()).device)
    geom = geometry or RenderGeometry()
    eye = torch.as_tensor(eye_trace_deg, dtype=torch.float32, device=device)
    padded_eye = torch.cat([eye[:1].expand(int(prepad_frames), -1), eye], dim=0)
    world = render_world(spec, geom, device=device)
    movie_raw = sample_retina_movie(world, padded_eye, geometry=geom, device=device)[0, 0]
    movie = renderer_raw_to_model_pixelnorm(movie_raw, max_raw=float(geom.max_raw))
    x = movie.unsqueeze(0).unsqueeze(0).to(device)
    device_obj = next(model.model.parameters()).device
    use_amp = device_obj.type == "cuda"
    model.model.eval()
    readout.eval()
    with torch.inference_mode(), torch.autocast(device_type=device_obj.type, dtype=torch.bfloat16, enabled=use_amp):
        x_front = model.model.frontend(x)
        x_conv = model.model.convnet(x_front)
        x_recurrent = model.model.recurrent(x_conv)
        feats = x_recurrent[0].permute(1, 0, 2, 3).contiguous()
        y = readout(feats)
        rates = _collapse_spatial(model.model.activation(y), method=spatial_collapse).float().detach().cpu().numpy()
    return rates[int(prepad_frames) :].astype(np.float32)
