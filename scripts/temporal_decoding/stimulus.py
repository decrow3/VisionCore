"""
Task 1.1: E Optotype Stimulus Generator Module

Generates Tumbling E stimuli at arbitrary orientations and LogMAR values,
compatible with the counterfactual stimulus pipeline (make_counterfactual_stim).

Output format: (N_frames, H, W) uint8 numpy array matching get_fixrsvp_stack() output.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Constants
PPD = 37.50476617
CANVAS_SIZE = 600
BACKGROUND = 127


def _differentiable_grid_sample(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Bilinear grid sample with zero padding outside boundaries.
    Copied from check_fixrsvp_model_fisherinfo.py to avoid importing that script
    (which runs expensive model-loading code at import time).
    """
    B, C, H, W = image.shape
    _, H_out, W_out, _ = grid.shape

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
        return torch.gather(image_flat, 2, lin_exp).reshape(B, C, H_out, W_out)

    Ia = get_pixel(x0_c, y0_c)
    Ib = get_pixel(x0_c, y1_c)
    Ic = get_pixel(x1_c, y0_c)
    Id = get_pixel(x1_c, y1_c)

    wa = ((x1 - x_pix) * (y1 - y_pix)).unsqueeze(1)
    wb = ((x1 - x_pix) * (y_pix - y0)).unsqueeze(1)
    wc = ((x_pix - x0) * (y1 - y_pix)).unsqueeze(1)
    wd = ((x_pix - x0) * (y_pix - y0)).unsqueeze(1)

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    mask = ((x_pix >= 0) & (x_pix < W - 1) & (y_pix >= 0) & (y_pix < H - 1)).unsqueeze(1)
    return out * mask.float()


class EOptotypeRenderer(nn.Module):
    """
    Renders a Tumbling E optotype at arbitrary orientation and LogMAR size.

    The E template (in normalized coords) has:
      - Vertical stroke: x in [-1, -0.6], full height
      - Top stroke: x in [-0.6, 1], y in [0.6, 1]
      - Middle stroke: x in [-0.6, 1], y in [-0.2, 0.2]
      - Bottom stroke: x in [-0.6, 1], y in [-1, -0.6]
    At orientation=0 the E opens to the right.
    """

    def __init__(
        self,
        ppd: float = PPD,
        canvas_size: int = CANVAS_SIZE,
        template_res: int = 1024,
        blur_sigma: float = 0.5,
        device: str = 'cpu',
    ):
        super().__init__()
        self.ppd = ppd
        self.canvas_size = canvas_size
        self.blur_sigma = blur_sigma
        self.device = device

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

    def _get_affine_matrix(self, orientation_deg: float, logmar: float) -> torch.Tensor:
        H = W = self.canvas_size
        angle = torch.tensor(orientation_deg * (np.pi / 180.0))
        c, s = torch.cos(angle), torch.sin(angle)

        # Inverse rotation matrix (2x2)
        R_inv = torch.stack([
            torch.stack([c, s]),
            torch.stack([-s, c]),
        ])  # (2, 2)

        size_pix = (5.0 * (10.0 ** logmar / 60.0)) * self.ppd
        sx_inv = W / (size_pix + 1e-8)
        sy_inv = H / (size_pix + 1e-8)
        S_inv = torch.diag(torch.tensor([sx_inv, sy_inv]))

        A = S_inv @ R_inv  # (2, 2)
        b = torch.zeros(2, 1)  # centered at (0, 0)
        return torch.cat([A, b], dim=1).unsqueeze(0)  # (1, 2, 3)

    def forward(self, orientation_deg: float, logmar: float) -> torch.Tensor:
        """
        Render the E optotype.

        Returns:
            torch.Tensor of shape (1, 1, H, W), values in [0, 1]
            Value 1 = E foreground, Value 0 = background
        """
        affine = self._get_affine_matrix(orientation_deg, logmar).to(self.device)
        H = W = self.canvas_size
        grid = F.affine_grid(affine, (1, 1, H, W), align_corners=False).to(self.device)
        world = _differentiable_grid_sample(self.template.to(self.device), grid)

        if self.blur_sigma > 0:
            world = TF.gaussian_blur(world, kernel_size=7, sigma=self.blur_sigma)

        return world


def e_optotype_stack(
    orientation_deg: float,
    logmar: float,
    n_frames: int = 540,
    ppd: float = PPD,
    canvas_size: int = CANVAS_SIZE,
    background: int = BACKGROUND,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Generate a Tumbling E optotype stimulus stack compatible with make_counterfactual_stim.

    Args:
        orientation_deg: E orientation in degrees (0=opens right, 90=opens down, 180=opens left, 270=opens up)
        logmar: letter size in LogMAR units. At 0.0, critical gap = ~0.625 px at 37.5 ppd.
        n_frames: number of frames (static stimulus repeated)
        ppd: pixels per degree
        canvas_size: canvas width/height in pixels (must match eye trace pipeline, default 600)
        background: background pixel value (127 = mean gray)
        device: torch device for rendering ('cpu' or 'cuda')

    Returns:
        np.ndarray of shape (n_frames, canvas_size, canvas_size), dtype=uint8
        Dark E (0) on gray background (127), matching get_fixrsvp_stack() convention.
    """
    renderer = EOptotypeRenderer(ppd=ppd, canvas_size=canvas_size, device=device)

    with torch.no_grad():
        world = renderer(orientation_deg, logmar)  # (1, 1, H, W), [0, 1]

    world_np = world[0, 0].cpu().numpy()  # (H, W)

    # Dark E on gray background: foreground (1.0) → 0, background (0.0) → background
    frame = np.round(background * (1.0 - world_np)).clip(0, 255).astype(np.uint8)

    # Repeat for all frames
    stack = np.broadcast_to(frame[np.newaxis], (n_frames, canvas_size, canvas_size)).copy()
    return stack


def letter_size_pixels(logmar: float, ppd: float = PPD) -> float:
    """Return the full letter height/width in pixels for a given LogMAR value."""
    return 5.0 * (10.0 ** logmar / 60.0) * ppd


def gap_size_pixels(logmar: float, ppd: float = PPD) -> float:
    """Return the critical gap size in pixels (1/5 of letter height)."""
    return letter_size_pixels(logmar, ppd) / 5.0


def visualize_e_optotypes(
    logmar_values=None,
    orientations=None,
    figsize=(14, 8),
    crop_factor=0.1,
):
    """
    Show E optotypes at several LogMAR values and orientations.

    Args:
        crop_factor: fraction of canvas to show around center (0.1 = center 10%)
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if logmar_values is None:
        logmar_values = [-0.2, 0.0, 0.2, 0.5, 0.8, 1.0]
    if orientations is None:
        orientations = [0, 90, 180, 270]

    fig, axes = plt.subplots(len(orientations), len(logmar_values), figsize=figsize)

    H = W = CANVAS_SIZE
    c = int(CANVAS_SIZE * (0.5 - crop_factor / 2))
    d = int(CANVAS_SIZE * crop_factor)

    for i, ori in enumerate(orientations):
        for j, lm in enumerate(logmar_values):
            stack = e_optotype_stack(ori, lm, n_frames=1)
            ax = axes[i, j]
            crop = stack[0, c:c + d, c:c + d]
            ax.imshow(crop, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.axis('off')
            if i == 0:
                gap_px = gap_size_pixels(lm)
                ax.set_title(f'LogMAR {lm:.1f}\n({gap_px:.2f} px gap)', fontsize=7)
            if j == 0:
                ax.set_ylabel(f'{ori}°', fontsize=8, rotation=0, labelpad=20)

    plt.suptitle('Tumbling E Optotypes (center crop)', fontsize=12)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Testing E optotype generation...")
    print(f"PPD={PPD}, canvas={CANVAS_SIZE}x{CANVAS_SIZE}")

    for logmar in [0.0, 0.5, 1.0]:
        gap_px = gap_size_pixels(logmar)
        for ori in [0, 90, 180, 270]:
            stack = e_optotype_stack(ori, logmar)
            assert stack.shape == (540, CANVAS_SIZE, CANVAS_SIZE), f"Shape mismatch: {stack.shape}"
            assert stack.dtype == np.uint8
        print(f"  LogMAR={logmar:.1f}: gap={gap_px:.3f}px, letter={letter_size_pixels(logmar):.2f}px  OK")

    # Verify that larger LogMAR → more dark pixels
    counts = {}
    for lm in [0.0, 0.5, 1.0]:
        s = e_optotype_stack(0, lm, n_frames=1)
        counts[lm] = int((s[0] < 64).sum())
    assert counts[1.0] > counts[0.5] > counts[0.0], \
        f"Expected monotonic dark pixels: {counts}"
    print(f"\nDark pixel counts (threshold<64): {counts}")
    print("Monotonic with LogMAR: OK")

    fig = visualize_e_optotypes()
    out_path = 'test_e_optotypes.png'
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to {out_path}")
    print("All tests passed!")
