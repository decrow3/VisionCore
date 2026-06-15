"""High-resolution Vernier stimulus rendering and pixel-level audits.

The renderer deliberately mirrors the existing E-optotype high-resolution path:
draw in a high-PPD world canvas, then sample the model retina from that world.
It keeps the Vernier-specific provenance in one place so the model-facing code
can treat each rendered image as an audited source artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from scripts.temporal_decoding.stimulus_hires import (
    RETINA_PPD,
    RETINA_SIZE,
    WORLD_PPD,
    WORLD_SIZE,
    _grid_sample,
)


RAW_WHITE = 127.0


@dataclass(frozen=True)
class RenderGeometry:
    """Spatial geometry shared by the world renderer and model retina."""

    world_ppd: float = WORLD_PPD
    world_size: tuple[int, int] = WORLD_SIZE
    retina_ppd: float = RETINA_PPD
    retina_size: tuple[int, int] = RETINA_SIZE
    background_raw: float = RAW_WHITE * 0.5
    max_raw: float = RAW_WHITE

    @property
    def model_pixel_arcmin(self) -> float:
        return 60.0 / float(self.retina_ppd)

    @property
    def world_pixel_arcmin(self) -> float:
        return 60.0 / float(self.world_ppd)


@dataclass(frozen=True)
class VernierSpec:
    """Vernier stimulus parameters in visual-angle units."""

    offset_arcmin: float = 0.0
    bar_width_arcmin: float = 2.0
    gap_arcmin: float = 4.0
    bar_length_arcmin: float = 12.0
    contrast: float = 0.5
    polarity: str = "bright"
    center_x_arcmin: float = 0.0
    center_y_arcmin: float = 0.0
    edge_softness_world_px: float = 0.75
    orientation_deg: float = 0.0

    def with_offset(self, offset_arcmin: float) -> "VernierSpec":
        return VernierSpec(**{**asdict(self), "offset_arcmin": float(offset_arcmin)})


def arcmin_to_deg(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) / 60.0 if isinstance(value, np.ndarray) else float(value) / 60.0


def deg_to_arcmin(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * 60.0 if isinstance(value, np.ndarray) else float(value) * 60.0


def _bar_mask(
    x_arcmin: torch.Tensor,
    y_arcmin: torch.Tensor,
    *,
    center_x: float,
    y0: float,
    y1: float,
    width: float,
    softness: float,
) -> torch.Tensor:
    """Soft rectangular bar mask in [0, 1]."""
    k = 1.0 / max(float(softness), 1e-6)
    x_left = center_x - width / 2.0
    x_right = center_x + width / 2.0
    return (
        torch.sigmoid(k * (x_arcmin - x_left))
        * torch.sigmoid(k * (x_right - x_arcmin))
        * torch.sigmoid(k * (y_arcmin - y0))
        * torch.sigmoid(k * (y1 - y_arcmin))
    )


class VernierRenderer(nn.Module):
    """Render a two-segment vertical Vernier stimulus on a high-res world canvas."""

    def __init__(self, geometry: RenderGeometry | None = None, device: str = "cpu"):
        super().__init__()
        self.geometry = geometry or RenderGeometry()
        self.device_name = str(device)
        h, w = self.geometry.world_size
        xs = (torch.arange(w, dtype=torch.float32) - (w - 1) / 2.0) / float(self.geometry.world_ppd) * 60.0
        ys = ((h - 1) / 2.0 - torch.arange(h, dtype=torch.float32)) / float(self.geometry.world_ppd) * 60.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("x_arcmin", xx)
        self.register_buffer("y_arcmin", yy)

    def forward(self, spec: VernierSpec) -> torch.Tensor:
        """Return raw-luminance image shaped ``(1, 1, H, W)`` in [0, 127]."""
        if spec.polarity not in {"bright", "dark"}:
            raise ValueError(f"polarity must be 'bright' or 'dark', got {spec.polarity!r}")

        geom = self.geometry
        half_gap = float(spec.gap_arcmin) / 2.0
        length = float(spec.bar_length_arcmin)
        width = float(spec.bar_width_arcmin)
        softness_arcmin = float(spec.edge_softness_world_px) * float(geom.world_pixel_arcmin)
        cx = float(spec.center_x_arcmin)
        cy = float(spec.center_y_arcmin)
        theta = np.deg2rad(float(spec.orientation_deg))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        x0 = self.x_arcmin - cx
        y0 = self.y_arcmin - cy
        x_local = x0 * cos_t + y0 * sin_t + cx
        y_local = -x0 * sin_t + y0 * cos_t + cy

        upper = _bar_mask(
            x_local,
            y_local,
            center_x=cx,
            y0=cy + half_gap,
            y1=cy + half_gap + length,
            width=width,
            softness=softness_arcmin,
        )
        lower = _bar_mask(
            x_local,
            y_local,
            center_x=cx + float(spec.offset_arcmin),
            y0=cy - half_gap - length,
            y1=cy - half_gap,
            width=width,
            softness=softness_arcmin,
        )
        mask = torch.clamp(upper + lower, 0.0, 1.0)
        signed_contrast = float(spec.contrast) if spec.polarity == "bright" else -float(spec.contrast)
        image = float(geom.background_raw) + signed_contrast * float(geom.background_raw) * mask
        image = torch.clamp(image, 0.0, float(geom.max_raw))
        return image.unsqueeze(0).unsqueeze(0)


class VernierRetina(nn.Module):
    """Sample a Vernier world image along a retinal phase/eye-position trace."""

    def __init__(self, geometry: RenderGeometry | None = None):
        super().__init__()
        self.geometry = geometry or RenderGeometry()
        world_h, world_w = self.geometry.world_size
        retina_h, retina_w = self.geometry.retina_size
        ppd_ratio = float(self.geometry.world_ppd) / float(self.geometry.retina_ppd)
        xs_pix_ret = torch.linspace(-(retina_w / 2.0) + 0.5, (retina_w / 2.0) - 0.5, retina_w)
        ys_pix_ret = torch.linspace(-(retina_h / 2.0) + 0.5, (retina_h / 2.0) - 0.5, retina_h)
        xs_norm = xs_pix_ret * ppd_ratio * (2.0 / world_w)
        ys_norm = ys_pix_ret * ppd_ratio * (2.0 / world_h)
        grid_y, grid_x = torch.meshgrid(ys_norm, xs_norm, indexing="ij")
        self.register_buffer("base_grid_flat", torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1))
        self.pixels = int(retina_h * retina_w)

    def forward(self, world_image: torch.Tensor, eye_trace_deg: torch.Tensor) -> torch.Tensor:
        """Return retinal movie shaped ``(1, 1, T, H_ret, W_ret)``."""
        geom = self.geometry
        t = int(eye_trace_deg.shape[0])
        dev = world_image.device
        shift_x = eye_trace_deg[:, 0] * float(geom.world_ppd) * (2.0 / float(geom.world_size[1]))
        shift_y = -eye_trace_deg[:, 1] * float(geom.world_ppd) * (2.0 / float(geom.world_size[0]))
        shifts = torch.stack([shift_x, shift_y], dim=1).to(dev)
        grid = (self.base_grid_flat.to(dev).unsqueeze(1) + shifts.unsqueeze(0)).unsqueeze(0)
        sampled = _grid_sample(world_image, grid, fill_value=float(geom.background_raw))
        retina_h, retina_w = geom.retina_size
        sampled = sampled.view(1, 1, retina_h, retina_w, t)
        return sampled.permute(0, 1, 4, 2, 3)


def render_world(spec: VernierSpec, geometry: RenderGeometry | None = None, device: str = "cpu") -> torch.Tensor:
    renderer = VernierRenderer(geometry=geometry, device=device).to(device)
    renderer.eval()
    with torch.no_grad():
        return renderer(spec)


def sample_retina_movie(
    world_image: torch.Tensor,
    eye_trace_deg: np.ndarray | torch.Tensor,
    *,
    geometry: RenderGeometry | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Sample ``world_image`` into a raw-luminance retinal movie."""
    retina = VernierRetina(geometry=geometry).to(device)
    if not isinstance(eye_trace_deg, torch.Tensor):
        eye_trace_deg = torch.as_tensor(eye_trace_deg, dtype=torch.float32)
    with torch.no_grad():
        return retina(world_image.to(device), eye_trace_deg.to(device))


def central_retina_frame(
    spec: VernierSpec,
    geometry: RenderGeometry | None = None,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Render and sample one centered retinal frame in model raw-luminance units."""
    geom = geometry or RenderGeometry()
    world = render_world(spec, geom, device=device)
    movie = sample_retina_movie(world, np.zeros((1, 2), dtype=np.float32), geometry=geom, device=device)
    return movie[0, 0, 0].detach().cpu().numpy().astype(np.float32)


def pixel_audit(
    spec: VernierSpec,
    *,
    fd_steps_arcmin: list[float],
    geometry: RenderGeometry | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Return rendering and pixel-level finite-difference diagnostics."""
    geom = geometry or RenderGeometry()
    zero = central_retina_frame(spec.with_offset(0.0), geom, device=device)
    rows: list[dict[str, Any]] = []
    for step in fd_steps_arcmin:
        plus = central_retina_frame(spec.with_offset(float(step)), geom, device=device)
        minus = central_retina_frame(spec.with_offset(-float(step)), geom, device=device)
        diff = plus - minus
        deriv = diff / (2.0 * float(step))
        sigma = np.maximum((plus + minus) / 2.0, 1e-3)
        fisher = float(np.sum((deriv * deriv) / sigma))
        contrast_template = np.abs(zero - float(geom.background_raw))
        x_template = _x_template(zero.shape)
        template_fisher = _template_fisher(deriv, sigma, contrast_template)
        x_template_fisher = _template_fisher(deriv, sigma, contrast_template * x_template)
        centroid_plus = luminance_centroid(plus, geom.background_raw)
        centroid_minus = luminance_centroid(minus, geom.background_raw)
        centroid_dx = float(centroid_plus[0] - centroid_minus[0])
        centroid_dy = float(centroid_plus[1] - centroid_minus[1])
        rows.append(
            {
                "fd_step_arcmin": float(step),
                "max_abs_plus_minus_diff": float(np.max(np.abs(diff))),
                "l2_plus_minus_diff": float(np.sqrt(np.sum(diff * diff))),
                "pixel_fisher_per_arcmin2_diag": fisher,
                "template_fisher_per_arcmin2_contrast": template_fisher,
                "template_fisher_per_arcmin2_x_weighted": x_template_fisher,
                "total_luminance_plus": float(np.sum(plus)),
                "total_luminance_minus": float(np.sum(minus)),
                "total_luminance_abs_delta": float(abs(np.sum(plus) - np.sum(minus))),
                "centroid_x_plus_px": centroid_plus[0],
                "centroid_y_plus_px": centroid_plus[1],
                "centroid_x_minus_px": centroid_minus[0],
                "centroid_y_minus_px": centroid_minus[1],
                "centroid_dx_px": centroid_dx,
                "centroid_dy_px": centroid_dy,
                "centroid_sensitivity_x_px_per_arcmin": centroid_dx / (2.0 * float(step)),
                "centroid_sensitivity_y_px_per_arcmin": centroid_dy / (2.0 * float(step)),
            }
        )
    return {
        "geometry": asdict(geom),
        "spec": asdict(spec),
        "zero_frame_min": float(np.min(zero)),
        "zero_frame_max": float(np.max(zero)),
        "zero_frame_mean": float(np.mean(zero)),
        "model_pixel_arcmin": float(geom.model_pixel_arcmin),
        "world_pixel_arcmin": float(geom.world_pixel_arcmin),
        "fd_rows": rows,
    }


def luminance_centroid(frame: np.ndarray, background_raw: float) -> tuple[float, float]:
    """Centroid of absolute contrast energy relative to the background."""
    arr = np.asarray(frame, dtype=np.float64)
    weights = np.abs(arr - float(background_raw))
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan"), float("nan")
    h, w = arr.shape
    yy, xx = np.mgrid[:h, :w]
    return float(np.sum(xx * weights) / total), float(np.sum(yy * weights) / total)


def _x_template(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    _yy, xx = np.mgrid[:h, :w]
    x = xx.astype(np.float64) - (w - 1) / 2.0
    scale = float(np.max(np.abs(x))) or 1.0
    return x / scale


def _template_fisher(deriv: np.ndarray, sigma: np.ndarray, template: np.ndarray) -> float:
    d = np.asarray(deriv, dtype=np.float64).ravel()
    s = np.maximum(np.asarray(sigma, dtype=np.float64).ravel(), 1e-8)
    t = np.asarray(template, dtype=np.float64).ravel()
    norm = float(np.sqrt(np.sum((t * t) / s)))
    if norm <= 1e-12:
        return float("nan")
    t = t / norm
    return float(np.sum(d * t / s) ** 2)


def scaled_render_geometry(geometry: RenderGeometry, factor: float) -> RenderGeometry:
    factor = float(factor)
    if factor <= 0:
        raise ValueError(f"Resolution factor must be positive, got {factor}")
    world_h, world_w = geometry.world_size
    return replace(
        geometry,
        world_ppd=float(geometry.world_ppd) * factor,
        world_size=(max(8, int(round(world_h * factor))), max(8, int(round(world_w * factor)))),
    )


def renderer_resolution_sweep(
    spec: VernierSpec,
    *,
    fd_steps_arcmin: list[float],
    factors: list[float],
    geometry: RenderGeometry | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    base_geom = geometry or RenderGeometry()
    rows: list[dict[str, Any]] = []
    for factor in factors:
        geom = scaled_render_geometry(base_geom, float(factor))
        audit = pixel_audit(spec, fd_steps_arcmin=fd_steps_arcmin, geometry=geom, device=device)
        for row in audit["fd_rows"]:
            rows.append(
                {
                    "resolution_factor": float(factor),
                    "world_ppd": float(geom.world_ppd),
                    "world_size_h": int(geom.world_size[0]),
                    "world_size_w": int(geom.world_size[1]),
                    **row,
                }
            )
    return rows


def save_pixel_audit_artifacts(
    out_dir: Path,
    spec: VernierSpec,
    *,
    fd_steps_arcmin: list[float],
    geometry: RenderGeometry | None = None,
    device: str = "cpu",
    resolution_factors: list[float] | None = None,
) -> dict[str, Any]:
    """Save stimulus PNGs, difference PNGs, line profiles, and return audit dict."""
    geom = geometry or RenderGeometry()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = pixel_audit(spec, fd_steps_arcmin=fd_steps_arcmin, geometry=geom, device=device)
    factors = [float(v) for v in (resolution_factors or []) if float(v) > 0.0]
    if factors:
        audit["resolution_sweep_rows"] = renderer_resolution_sweep(
            spec,
            fd_steps_arcmin=fd_steps_arcmin,
            factors=factors,
            geometry=geom,
            device=device,
        )
    offsets = sorted({0.0, *fd_steps_arcmin, *[-float(s) for s in fd_steps_arcmin]})
    frames = {offset: central_retina_frame(spec.with_offset(offset), geom, device=device) for offset in offsets}
    for offset, frame in frames.items():
        _save_frame(out_dir / f"vernier_offset_{offset:+.4f}arcmin.png", frame, geom)
    for step in fd_steps_arcmin:
        diff = frames[float(step)] - frames[-float(step)]
        _save_frame(out_dir / f"vernier_diff_pm_{float(step):.4f}arcmin.png", diff, geom, diverging=True)
        _save_profile(out_dir / f"vernier_profiles_pm_{float(step):.4f}arcmin.png", frames[float(step)], frames[-float(step)])
    return audit


def _save_frame(path: Path, frame: np.ndarray, geom: RenderGeometry, *, diverging: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=150)
    if diverging:
        lim = float(np.max(np.abs(frame))) or 1.0
        im = ax.imshow(frame, cmap="coolwarm", vmin=-lim, vmax=lim)
    else:
        im = ax.imshow(frame, cmap="gray", vmin=0.0, vmax=float(geom.max_raw))
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.1)
    fig.savefig(path)
    plt.close(fig)


def _save_profile(path: Path, plus: np.ndarray, minus: np.ndarray) -> None:
    h = int(plus.shape[0])
    row = h // 2
    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=150)
    ax.plot(plus[row], label="+delta")
    ax.plot(minus[row], label="-delta")
    ax.plot(plus[row] - minus[row], label="difference")
    ax.set_xlabel("retina x pixel")
    ax.set_ylabel("raw luminance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
