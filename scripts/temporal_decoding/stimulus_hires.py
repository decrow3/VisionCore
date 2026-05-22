"""
High-resolution E optotype stimulus generation via world→retina pipeline.

Renders the Tumbling E at 120 ppd in a 512×512 world canvas, then resamples
through DifferentiableRetina to produce retinal movies at the model's native
37.5 ppd. This preserves sub-pixel orientation information that is destroyed by
direct rendering onto the 37.5 ppd canvas (where the E gap is <1 pixel at
LogMAR 0.0 and below).

At 120 ppd the E has:
    LogMAR  0.0:  letter=12.5px, gap=2.5px   (well-resolved)
    LogMAR -0.1:  letter=11.1px, gap=2.2px
    LogMAR -0.2:  letter=9.9px,  gap=2.0px
    LogMAR -0.3:  letter=8.9px,  gap=1.8px

vs. at 37.5 ppd (old pipeline):
    LogMAR  0.0:  letter=3.13px, gap=0.63px  (sub-pixel gap)
    LogMAR -0.02: gap=0.60px → all orientations IDENTICAL after uint8 rounding

Compatibility: output format matches make_counterfactual_stim →
    (T_valid, 1, n_lags, H_out, W_out) float32 tensor
so compute_trial_rates() can be called unchanged.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_info import embed_time_lags

# ─── Constants ────────────────────────────────────────────────────────────────

WORLD_PPD = 120.0           # pixels per degree in world canvas
WORLD_SIZE = (512, 512)     # world canvas pixels (4.27° × 4.27°)
RETINA_PPD = 37.50476617    # pixels per degree for model input
RETINA_SIZE = (101, 101)    # must match model's expected spatial input
N_LAGS = 32
TEMPLATE_RES = 1024
BLUR_SIGMA = 1.0


# ─── Custom grid sampler (inline to avoid import side-effects) ─────────────────

def _grid_sample(image: torch.Tensor, grid: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Bilinear grid sample with configurable out-of-bounds fill.

    Default behaviour matches the original implementation: fill with 0.0 outside the image.
    """
    B, C, H, W = image.shape
    _, P, T_dim, _ = grid.shape  # (B, P, T, 2) or (B, H_out, W_out, 2)

    x = grid[..., 0]
    y = grid[..., 1]

    x_pix = (x + 1) * W * 0.5 - 0.5
    y_pix = (y + 1) * H * 0.5 - 0.5

    x0 = torch.floor(x_pix).long()
    x1 = x0 + 1
    y0 = torch.floor(y_pix).long()
    y1 = y0 + 1

    x0_c = torch.clamp(x0, 0, W - 1)
    x1_c = torch.clamp(x1, 0, W - 1)
    y0_c = torch.clamp(y0, 0, H - 1)
    y1_c = torch.clamp(y1, 0, H - 1)

    image_flat = image.view(B, C, -1)

    def get_pixel(ix, iy):
        lin = iy * W + ix
        lin_exp = lin.unsqueeze(1).expand(-1, C, -1, -1).reshape(B, C, -1)
        return torch.gather(image_flat, 2, lin_exp).reshape(B, C, *grid.shape[1:3])

    Ia = get_pixel(x0_c, y0_c)
    Ib = get_pixel(x0_c, y1_c)
    Ic = get_pixel(x1_c, y0_c)
    Id = get_pixel(x1_c, y1_c)

    wa = ((x1 - x_pix) * (y1 - y_pix)).unsqueeze(1)
    wb = ((x1 - x_pix) * (y_pix - y0)).unsqueeze(1)
    wc = ((x_pix - x0) * (y1 - y_pix)).unsqueeze(1)
    wd = ((x_pix - x0) * (y_pix - y0)).unsqueeze(1)

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    mask = ((x_pix >= 0) & (x_pix < W - 1) &
            (y_pix >= 0) & (y_pix < H - 1)).unsqueeze(1).to(out.dtype)

    fill = torch.as_tensor(fill_value, dtype=out.dtype, device=out.device).view(1, 1, 1, 1)
    return out * mask + fill * (1.0 - mask)


# ─── Stage 1: High-resolution E world image ───────────────────────────────────

class HiResERenderer(nn.Module):
    """
    Renders a Tumbling E at high resolution (120 ppd, 512×512).

    Output: (1, 1, H_world, W_world) float32 tensor, values in [0, 1]
    where 1 = E foreground, 0 = background.
    """

    def __init__(
        self,
        ppd: float = WORLD_PPD,
        canvas_size: tuple = WORLD_SIZE,
        template_res: int = TEMPLATE_RES,
        blur_sigma: float = BLUR_SIGMA,
        device: str = 'cpu',
    ):
        super().__init__()
        self.ppd = ppd
        self.canvas_size = canvas_size
        self.device = device
        self.blur_sigma = blur_sigma

        H, W = canvas_size
        self.extent = [-W / (2 * ppd), W / (2 * ppd),
                       -H / (2 * ppd), H / (2 * ppd)]

        self.register_buffer('template', self._make_template(template_res))

    def _make_template(self, res: int) -> torch.Tensor:
        xx = torch.linspace(-1, 1, res)
        yy = torch.linspace(-1, 1, res)
        y, x = torch.meshgrid(yy, xx, indexing='ij')
        k = 200.0

        def box(x0, x1, y0, y1):
            return (torch.sigmoid(k * (x - x0)) * torch.sigmoid(k * (x1 - x)) *
                    torch.sigmoid(k * (y - y0)) * torch.sigmoid(k * (y1 - y)))

        shape = (box(-1.0, -0.6, -1.0, 1.0) +  # vertical stroke
                 box(-0.6, 1.0, 0.6, 1.0) +    # top stroke
                 box(-0.6, 1.0, -0.2, 0.2) +   # middle stroke
                 box(-0.6, 1.0, -1.0, -0.6))   # bottom stroke
        shape = torch.clamp(shape, 0, 1)
        return shape.unsqueeze(0).unsqueeze(0)  # (1, 1, res, res)

    def get_affine_matrix(
        self,
        orientation_deg: float,
        logmar: float,
        center_offset_deg: tuple = (0.0, 0.0),
    ) -> torch.Tensor:
        """Build inverse affine matrix for grid_sample."""
        H, W = self.canvas_size
        angle = torch.tensor(orientation_deg * (np.pi / 180.0))
        c, s = torch.cos(angle), torch.sin(angle)
        R_inv = torch.stack([
            torch.stack([c, s]),
            torch.stack([-s, c]),
        ])  # (2, 2)

        size_pix = (5.0 * (10.0 ** logmar / 60.0)) * self.ppd
        sx_inv = W / (size_pix + 1e-8)
        sy_inv = H / (size_pix + 1e-8)
        S_inv = torch.diag(torch.tensor([sx_inv, sy_inv]))

        A = S_inv @ R_inv  # (2, 2)

        # Center offset: shift E center by (dx_deg, dy_deg) in world output space.
        #
        # PyTorch affine_grid uses the inverse transform:
        #   p_source = A @ p_target + b
        # To place the E center (at template origin 0) at world position (dx_out, dy_out):
        #   0 = A @ [dx_out, dy_out]^T + b  →  b = -A @ [dx_out, dy_out]^T
        #
        # Note: dy is negated because image y-axis is flipped vs. spatial y.
        dx_out_norm = center_offset_deg[0] * self.ppd * (2.0 / W)
        dy_out_norm = -center_offset_deg[1] * self.ppd * (2.0 / H)
        offset_out = torch.tensor([dx_out_norm, dy_out_norm], dtype=A.dtype)
        b = -(A @ offset_out).unsqueeze(1)  # (2, 1)

        return torch.cat([A, b], dim=1).unsqueeze(0)  # (1, 2, 3)

    def forward(
        self,
        orientation_deg: float,
        logmar: float,
        center_offset_deg: tuple = (0.0, 0.0),
    ) -> torch.Tensor:
        """
        Returns (1, 1, H_world, W_world) float32 in [0, 1].
        """
        dev = self.template.device
        affine = self.get_affine_matrix(
            orientation_deg, logmar, center_offset_deg
        ).to(dev)
        H, W = self.canvas_size
        grid = F.affine_grid(affine, (1, 1, H, W), align_corners=False).to(dev)
        world = _grid_sample(self.template, grid)
        if self.blur_sigma > 0:
            world = TF.gaussian_blur(world, kernel_size=7, sigma=self.blur_sigma)
        return world


# ─── Stage 2: Retinal sampling along eye trace ────────────────────────────────

class HiResRetina(nn.Module):
    """
    Samples the high-res world image along an eye trace trajectory.

    Converts eye positions (degrees at RETINA_PPD scale) to shifts in the
    WORLD_PPD canvas, then performs a single batched grid_sample over all
    (space, time) pairs.

    Output: (1, 1, T, H_ret, W_ret)
    """

    def __init__(
        self,
        world_ppd: float = WORLD_PPD,
        retina_ppd: float = RETINA_PPD,
        world_canvas_size: tuple = WORLD_SIZE,
        retina_size: tuple = RETINA_SIZE,
    ):
        super().__init__()
        self.world_ppd = world_ppd
        self.retina_ppd = retina_ppd
        self.world_h, self.world_w = world_canvas_size
        self.retina_h, self.retina_w = retina_size
        self.n_pixels = retina_h = retina_size[0]
        retina_w = retina_size[1]

        # Pre-compute flattened base retinal grid in world-normalized coords.
        #
        # Each retina pixel at position p_ret (in retina pixels, centered at 0):
        #   deg = p_ret / retina_ppd            (convert retina pixels → degrees)
        #   world_pix = deg * world_ppd         (convert degrees → world pixels)
        #   norm = world_pix / (world_dim / 2)  (convert world pixels → [-1, 1])
        #
        # Combined: norm = p_ret * (world_ppd / retina_ppd) * (2 / world_dim)
        ppd_ratio = world_ppd / retina_ppd  # = 120 / 37.5 ≈ 3.2

        xs_pix_ret = torch.linspace(
            -(retina_w / 2) + 0.5, (retina_w / 2) - 0.5, retina_w
        )  # retina pixel coords, e.g. [-50, ..., 50] for W=101
        ys_pix_ret = torch.linspace(
            -(retina_h / 2) + 0.5, (retina_h / 2) - 0.5, retina_h
        )

        # World normalized coords: ±(W_ret/2 / retina_ppd * world_ppd) / (world_w/2)
        xs_norm = xs_pix_ret * ppd_ratio * (2.0 / world_canvas_size[1])
        ys_norm = ys_pix_ret * ppd_ratio * (2.0 / world_canvas_size[0])

        grid_y, grid_x = torch.meshgrid(ys_norm, xs_norm, indexing='ij')
        # Flatten: P = H_ret * W_ret pixels, each described by (x, y)
        self.register_buffer(
            'base_grid_flat',
            torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        )  # (P, 2)
        self.P = retina_h * retina_w

    def forward(
        self,
        world_image: torch.Tensor,
        eye_trace_deg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            world_image: (1, 1, H_world, W_world) float32
            eye_trace_deg: (T, 2) eye positions in degrees

        Returns:
            retinal_movie: (1, 1, T, H_ret, W_ret)
        """
        T = eye_trace_deg.shape[0]
        P = self.P
        dev = world_image.device

        # Convert eye positions (degrees) to world-normalized shifts
        # deg → world_pix = deg * world_ppd; world_pix → norm = pix / (dim/2)
        shift_x = eye_trace_deg[:, 0] * self.world_ppd * (2.0 / self.world_w)
        shift_y = -eye_trace_deg[:, 1] * self.world_ppd * (2.0 / self.world_h)
        # Note: y is negated because image y-axis is flipped (top=0 in pixel coords)
        shifts = torch.stack([shift_x, shift_y], dim=1).to(dev)  # (T, 2)

        # Build (B=1, P, T, 2) spatiotemporal sampling grid
        base = self.base_grid_flat.unsqueeze(1)  # (P, 1, 2)
        shifts_t = shifts.unsqueeze(0)            # (1, T, 2)
        grid = (base + shifts_t).unsqueeze(0)     # (1, P, T, 2)

        # Sample (fill outside the world canvas with background gray instead of black)
        output = _grid_sample(world_image, grid, fill_value=127.0)   # (1, 1, P, T)

        # Reshape to (1, 1, T, H_ret, W_ret)
        output = output.view(1, 1, self.retina_h, self.retina_w, T)
        output = output.permute(0, 1, 4, 2, 3)    # (1, 1, T, H_ret, W_ret)
        return output


# ─── Counterfactual stimulus builder (high-res path) ──────────────────────────

def hires_counterfactual_stim(
    orientation_deg: float,
    logmar: float,
    eyepos: np.ndarray,
    condition: str = 'real',
    null_trace: np.ndarray = None,
    center_offset_deg: tuple = (0.0, 0.0),
    n_lags: int = N_LAGS,
    retina_size: tuple = RETINA_SIZE,
    world_size: tuple = WORLD_SIZE,
    world_ppd: float = WORLD_PPD,
    retina_ppd: float = RETINA_PPD,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Build the eye-shifted stimulus sequence using the high-res world→retina pipeline.

    Replaces build_counterfactual_stim / make_counterfactual_stim for the
    hyperacuity regime (LogMAR ≤ 0.2) where direct rendering at 37.5 ppd
    loses sub-pixel orientation information.

    Args:
        orientation_deg: E orientation (0=opens right, 90=down, 180=left, 270=up)
        logmar: letter size in LogMAR units
        eyepos: (T, 2) float32 eye positions in degrees for this trial
        condition: 'real', 'stabilized', 'scaled_0.5', 'scaled_2.0', 'matched_null'
        null_trace: (T, 2) phase-randomized trace (required for matched_null)
        center_offset_deg: (dx, dy) subpixel offset for phase-robustness testing
        n_lags: model temporal history frames
        retina_size: (H, W) output retinal patch size (must match model input)
        world_size: (H, W) high-res world canvas
        world_ppd: pixels per degree in world canvas (120 recommended)
        retina_ppd: model's native ppd (37.5)
        device: torch device

    Returns:
        (T_valid, 1, n_lags, H_ret, W_ret) float32 tensor — same format as
        make_counterfactual_stim output, compatible with compute_trial_rates()
    """
    # Resolve effective eye trace for this condition
    eye_t = torch.from_numpy(eyepos).float()

    if condition == 'real':
        ep = eye_t
    elif condition == 'stabilized':
        mean = eye_t.mean(0, keepdim=True)
        ep = mean.expand_as(eye_t)
    elif condition.startswith('scaled_'):
        scale = float(condition.split('_')[1])
        mean = eye_t.mean(0, keepdim=True)
        ep = mean + (eye_t - mean) * scale
    elif condition == 'matched_null':
        assert null_trace is not None, "matched_null requires null_trace"
        ep = torch.from_numpy(null_trace).float()
    else:
        raise ValueError(f"Unknown condition: {condition}")

    T = ep.shape[0]

    # Stage 1: Render high-res world image
    renderer = HiResERenderer(
        ppd=world_ppd,
        canvas_size=world_size,
        device=device,
    ).to(device)
    renderer.eval()

    with torch.no_grad():
        world_img = renderer(orientation_deg, logmar, center_offset_deg)
        # world_img: (1, 1, H_world, W_world) in [0, 1] (E=1, bg=0)
        # Convert to mean-gray convention: E=0, bg=127 float32 in [0, 127]
        world_gray = 127.0 * (1.0 - world_img)  # (1, 1, H, W)

        # Stage 2: Sample retinal patch along eye trace
        retina = HiResRetina(
            world_ppd=world_ppd,
            retina_ppd=retina_ppd,
            world_canvas_size=world_size,
            retina_size=retina_size,
        ).to(device)
        retina.eval()

        ep_dev = ep.to(device)
        # Pad beginning with n_lags copies of the first frame's eye position
        ep_padded = torch.cat([ep_dev[:1].expand(n_lags, -1), ep_dev], dim=0)

        movie = retina(world_gray, ep_padded)    # (1, 1, T+n_lags, H_ret, W_ret)
        movie = movie[0, 0]                      # (T+n_lags, H_ret, W_ret)

    # Embed time lags: output (T, 1, n_lags, H_ret, W_ret)
    eye_stim = embed_time_lags(movie.cpu(), n_lags=n_lags)

    # Convert to float range that matches model training convention (scaled ~0–1)
    # The model was trained on stimuli in [0, 255] uint8 space divided by 127
    eye_stim = eye_stim / 127.0

    return eye_stim


def letter_size_pixels_hires(logmar: float) -> float:
    """Letter height in pixels at WORLD_PPD."""
    return 5.0 * (10.0 ** logmar / 60.0) * WORLD_PPD


def gap_size_pixels_hires(logmar: float) -> float:
    """Critical gap (E opening) in pixels at WORLD_PPD."""
    return letter_size_pixels_hires(logmar) / 5.0


# ─── Validation helpers ───────────────────────────────────────────────────────

def check_orientation_discriminability(
    logmar_values: list,
    center_offsets: list = None,
    verbose: bool = True,
) -> dict:
    """
    Verify that the 4 E orientations produce distinct retinal images at each LogMAR.

    For each LogMAR and each phase offset, reports:
        max_pixel_diff: max pixel difference between ori=0 and ori=90 retinal patches
        gap_px_world:   gap size in pixels at WORLD_PPD
        orientations_distinct: True if max_pixel_diff > 0

    Args:
        logmar_values: list of LogMAR values to check
        center_offsets: list of (dx, dy) subpixel offsets to test (default: [(0,0)])
        verbose: print table

    Returns:
        dict keyed by (logmar, offset) → result dict
    """
    if center_offsets is None:
        center_offsets = [(0.0, 0.0)]

    # Use stabilized (zero-eye-movement) traces to test raw stimulus discriminability
    T_test = 5
    eye_fixed = np.zeros((T_test, 2), dtype=np.float32)

    results = {}
    for lm in logmar_values:
        for offset in center_offsets:
            frames = {}
            for ori in [0, 90, 180, 270]:
                stim = hires_counterfactual_stim(
                    ori, lm, eye_fixed,
                    condition='stabilized',
                    center_offset_deg=offset,
                    n_lags=1,   # minimal lag for efficiency
                    device='cpu',
                )
                # Take the last time step, lag 0
                frames[ori] = stim[-1, 0, 0].numpy()  # (H_ret, W_ret)

            max_diffs = {
                'ori0_vs_90': float(np.abs(frames[0] - frames[90]).max()),
                'ori0_vs_180': float(np.abs(frames[0] - frames[180]).max()),
                'ori0_vs_270': float(np.abs(frames[0] - frames[270]).max()),
            }
            gap_px = gap_size_pixels_hires(lm)
            distinct = all(v > 1e-6 for v in max_diffs.values())

            results[(lm, offset)] = {
                'max_diffs': max_diffs,
                'gap_px_world': gap_px,
                'distinct': distinct,
            }

            if verbose:
                sym = '✓' if distinct else '✗'
                print(
                    f"  {sym} LM={lm:+.2f}, offset={offset}: "
                    f"gap={gap_px:.2f}px@120ppd, "
                    f"max_diff(0v90)={max_diffs['ori0_vs_90']:.4f}"
                )
    return results


# ─── Optional debug movie export ─────────────────────────────────────────────

def save_stimulus_mp4(
    out_path: str | Path,
    stim: torch.Tensor | np.ndarray,
    *,
    fps: int = 60,
    lag_index: int = 0,
    trim_first_frame: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = 'gray',
) -> None:
    """Save an MP4 from a stimulus tensor for quick debugging.

    Accepts either:
      - stim: (T, 1, n_lags, H, W) (as returned by hires_counterfactual_stim)
      - movie: (T, H, W)

    Notes:
      - hires_counterfactual_stim pads the eye trace by n_lags, so the returned
        tensor typically has one extra frame vs the original trial length.
        With trim_first_frame=True (default), we drop that first frame.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(stim, torch.Tensor):
        arr = stim.detach().to('cpu').float().numpy()
    else:
        arr = np.asarray(stim)

    if arr.ndim == 5:
        # (T, 1, n_lags, H, W) → (T, H, W)
        movie = arr[:, 0, lag_index]
    elif arr.ndim == 4:
        # (T, 1, H, W) → (T, H, W)
        movie = arr[:, 0]
    elif arr.ndim == 3:
        movie = arr
    else:
        raise ValueError(f"Unsupported stim shape for video export: {arr.shape}")

    if trim_first_frame and movie.shape[0] > 1:
        movie = movie[1:]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(movie[0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.axis('off')

    writer = animation.FFMpegWriter(fps=fps, codec='libx264', bitrate=6000)
    with writer.saving(fig, str(out_path), dpi=150):
        for t in range(movie.shape[0]):
            im.set_data(movie[t])
            writer.grab_frame()

    plt.close(fig)


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='HiRes E-optotype world→retina stimulus validation')
    parser.add_argument('--save_debug_movie', action='store_true', help='Also export a small debug MP4 of the retinal stimulus')
    parser.add_argument('--debug_movie_out', type=str, default=None, help='Output MP4 path (default: scripts/temporal_decoding/figures/debug_hires_stim.mp4)')
    parser.add_argument('--debug_movie_orientation_deg', type=float, default=0.0)
    parser.add_argument('--debug_movie_logmar', type=float, default=0.0)
    parser.add_argument('--debug_movie_condition', type=str, default='stabilized')
    parser.add_argument('--debug_movie_T', type=int, default=120)
    parser.add_argument('--debug_movie_n_lags', type=int, default=N_LAGS)
    parser.add_argument('--debug_movie_fps', type=int, default=60)
    parser.add_argument('--debug_movie_device', type=str, default='cpu')
    parser.add_argument('--debug_movie_keep_first_frame', action='store_true', help='Do not drop the first (padding-related) frame')

    args = parser.parse_args()

    print("=== HiRes Stimulus Pipeline Validation ===")
    print()

    # 1. Check discriminability across LogMAR range
    print("Orientation discriminability (stabilized, no eye movement):")
    logmar_test = [0.4, 0.2, 0.1, 0.0, -0.05, -0.1, -0.2, -0.3]
    disc_results = check_orientation_discriminability(logmar_test, verbose=True)

    # Compare with low-res pipeline
    print()
    print("Low-res (37.5 ppd) vs high-res (120 ppd) comparison:")
    print(f"{'LogMAR':>8} | {'gap@37.5ppd':>12} | {'gap@120ppd':>12} | {'lo-res max_diff':>15} | {'hi-res max_diff':>15}")
    print("-" * 75)
    from stimulus import e_optotype_stack, letter_size_pixels

    for lm in logmar_test:
        # Low-res
        lo0 = e_optotype_stack(0, lm)[0].astype(float)
        lo90 = e_optotype_stack(90, lm)[0].astype(float)
        lo_diff = np.abs(lo0 - lo90).max()
        gap_lo = letter_size_pixels(lm) / 5.0
        gap_hi = gap_size_pixels_hires(lm)
        hi_diff = disc_results[(lm, (0.0, 0.0))]['max_diffs']['ori0_vs_90']
        print(f"  {lm:+.2f}   | {gap_lo:12.3f} | {gap_hi:12.3f} | {lo_diff:15.2f} | {hi_diff:15.4f}")

    # 2. Visualize retinal patches at LogMAR 0.0 and -0.1
    print()
    print("Generating visualization...")
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    fig.suptitle('HiRes Retinal Patches: 4 Orientations', fontsize=13, fontweight='bold')

    T_viz = 1
    eye_fixed_viz = np.zeros((T_viz, 2), dtype=np.float32)

    for row, lm in enumerate([0.0, -0.1, -0.2]):
        for col, ori in enumerate([0, 90, 180, 270]):
            stim = hires_counterfactual_stim(
                ori, lm, eye_fixed_viz,
                condition='stabilized',
                n_lags=1, device='cpu',
            )
            patch = stim[0, 0, 0].numpy()  # (H_ret, W_ret)
            ax = axes[row, col]
            ax.imshow(patch, cmap='gray', interpolation='nearest')
            gap = gap_size_pixels_hires(lm)
            ax.set_title(f'LM={lm:+.1f}, ori={ori}°\ngap@world={gap:.1f}px', fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'figures', 'debug_hires_retinal.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Saved: {out_path}")

    if args.save_debug_movie:
        eye_fixed_movie = np.zeros((args.debug_movie_T, 2), dtype=np.float32)
        stim = hires_counterfactual_stim(
            args.debug_movie_orientation_deg,
            args.debug_movie_logmar,
            eye_fixed_movie,
            condition=args.debug_movie_condition,
            n_lags=args.debug_movie_n_lags,
            device=args.debug_movie_device,
        )

        mp4_out = args.debug_movie_out
        if mp4_out is None:
            mp4_out = os.path.join(os.path.dirname(__file__), 'figures', 'debug_hires_stim.mp4')

        save_stimulus_mp4(
            mp4_out,
            stim,
            fps=args.debug_movie_fps,
            lag_index=0,
            trim_first_frame=not args.debug_movie_keep_first_frame,
            vmin=0.0,
            vmax=1.0,
        )
        print(f"Saved: {mp4_out}")

    print()
    print("Validation complete.")
