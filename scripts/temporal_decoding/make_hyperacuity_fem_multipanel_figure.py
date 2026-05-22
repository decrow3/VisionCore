"""Make a polished descriptive multipanel figure for FEM hyperacuity results.

This script is intentionally self-contained and reads only cached artifacts:
- Decoder-controls results: scripts/temporal_decoding/data/results/decoder_controls_lm-*.npz
- Uniform all-hires neurometric sweep: scripts/temporal_decoding/data/results/neurometric_allhires_fresh.npz
- Eye traces: scripts/temporal_decoding/data/eye_traces.npz

Outputs a single multi-panel figure emphasizing three points:
1. FEM produces visibly different retinal input over a fixation.
2. Real FEM beats stabilization in the hyperacuity regime at long windows.
3. The benefit emerges only in the hard regime and vanishes again near chance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DecoderControls:
    logmar: float
    windows: np.ndarray
    d1_real_mean: np.ndarray
    d1_real_std: np.ndarray
    d1_stabilized_mean: np.ndarray
    d1_stabilized_std: np.ndarray


@dataclass(frozen=True)
class BoundaryProbe:
    logmars: np.ndarray
    real_mean: np.ndarray
    real_std: np.ndarray
    stabilized_mean: np.ndarray
    stabilized_std: np.ndarray


@dataclass(frozen=True)
class NeurometricSweep:
    logmars: np.ndarray
    real_A_mean: np.ndarray
    real_A_std: np.ndarray
    stabilized_A_mean: np.ndarray
    stabilized_A_std: np.ndarray
    real_C_mean: np.ndarray
    real_C_std: np.ndarray
    stabilized_C_mean: np.ndarray
    stabilized_C_std: np.ndarray


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_decoder_controls_file(path: Path) -> DecoderControls:
    d = np.load(path, allow_pickle=True)
    return DecoderControls(
        logmar=float(np.atleast_1d(d["logmar"])[0]),
        windows=np.asarray(d["windows"], dtype=int),
        d1_real_mean=np.asarray(d["D1_real_mean"], dtype=float),
        d1_real_std=np.asarray(d["D1_real_std"], dtype=float),
        d1_stabilized_mean=np.asarray(d["D1_stabilized_mean"], dtype=float),
        d1_stabilized_std=np.asarray(d["D1_stabilized_std"], dtype=float),
    )


def _load_decoder_controls_dir(results_dir: Path) -> list[DecoderControls]:
    items: list[DecoderControls] = []
    for p in sorted(results_dir.glob("decoder_controls_lm*.npz")):
        if "mlp_only" in p.name:
            continue
        items.append(_load_decoder_controls_file(p))
    items.sort(key=lambda x: x.logmar)
    return items


def _pick_window_index(windows: np.ndarray, window: int) -> int:
    matches = np.where(np.asarray(windows) == int(window))[0]
    if len(matches) != 1:
        raise ValueError(f"Window {window} not found uniquely in {windows}")
    return int(matches[0])


def _load_eye_traces(eye_traces_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(eye_traces_path, allow_pickle=True)
    traces = np.asarray(d["traces"], dtype=np.float32)  # (n_traces, T, 2)
    durations = np.asarray(d["durations"], dtype=np.int32)
    return traces, durations


def _load_boundary_probe(path: Path) -> BoundaryProbe:
    d = np.load(path, allow_pickle=True)
    return BoundaryProbe(
        logmars=np.asarray(d["logmar_values"], dtype=float),
        real_mean=np.asarray(d["acc_real_A"], dtype=float),
        real_std=np.asarray(d["std_real_A"], dtype=float),
        stabilized_mean=np.asarray(d["acc_stabilized_A"], dtype=float),
        stabilized_std=np.asarray(d["std_stabilized_A"], dtype=float),
    )


def _load_neurometric_sweep(path: Path) -> NeurometricSweep:
    d = np.load(path, allow_pickle=True)
    return NeurometricSweep(
        logmars=np.asarray(d["logmar_values"], dtype=float),
        real_A_mean=np.asarray(d["acc_real_A"], dtype=float),
        real_A_std=np.asarray(d["std_real_A"], dtype=float),
        stabilized_A_mean=np.asarray(d["acc_stabilized_A"], dtype=float),
        stabilized_A_std=np.asarray(d["std_stabilized_A"], dtype=float),
        real_C_mean=np.asarray(d["acc_real_C"], dtype=float),
        real_C_std=np.asarray(d["std_real_C"], dtype=float),
        stabilized_C_mean=np.asarray(d["acc_stabilized_C"], dtype=float),
        stabilized_C_std=np.asarray(d["std_stabilized_C"], dtype=float),
    )


def _normalize_img(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    arr = np.clip(arr, 0, 127)
    arr = 127.0 - arr
    return arr / 127.0


def _display_img(img: np.ndarray) -> np.ndarray:
    arr = _normalize_img(img)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax <= amin + 1e-8:
        return np.zeros_like(arr)
    arr = (arr - amin) / (amax - amin)
    return np.clip(arr, 0.0, 1.0)


def _center_crop(img: np.ndarray, crop_size: int = 33) -> np.ndarray:
    arr = np.asarray(img)
    h, w = arr.shape[-2], arr.shape[-1]
    crop = int(min(crop_size, h, w))
    y0 = max((h - crop) // 2, 0)
    x0 = max((w - crop) // 2, 0)
    return arr[y0:y0 + crop, x0:x0 + crop]


def _find_idx(arr: np.ndarray, value: float) -> int | None:
    idxs = np.where(np.isclose(np.asarray(arr, dtype=float), float(value)))[0]
    if len(idxs) == 0:
        return None
    return int(idxs[0])


def _estimate_crossover(logmars: np.ndarray, advantage: np.ndarray) -> float | None:
    x = np.asarray(logmars, dtype=float)
    y = np.asarray(advantage, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    for i in range(len(x) - 1):
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if y0 == 0.0:
            return float(x[i])
        if y0 * y1 < 0.0:
            t = -y0 / (y1 - y0)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None


def _render_hires_snapshots(
    orientation_deg: float,
    logmar: float,
    eye_trace_deg: np.ndarray,
    condition: str,
    valid_frame_indices: list[int],
    n_lags: int = 32,
) -> list[np.ndarray]:
    """Render a small set of retinal frames for Panel A.

    Uses the hi-res world→retina pipeline so the qualitative appearance matches
    the hyperacuity regime used elsewhere.

    Returns a list of (H, W) float arrays in [0, 127] (mean-gray convention).
    """
    import torch

    from scripts.temporal_decoding.stimulus_hires import hires_counterfactual_stim

    stim = hires_counterfactual_stim(
        orientation_deg=float(orientation_deg),
        logmar=float(logmar),
        eyepos=np.asarray(eye_trace_deg, dtype=np.float32),
        condition=str(condition),
        n_lags=int(n_lags),
        device="cpu",
    )
    # stim: (T_valid, 1, n_lags, H, W)
    frames: list[np.ndarray] = []
    for idx in valid_frame_indices:
        idx = int(idx)
        if idx < 0 or idx >= stim.shape[0]:
            raise ValueError(f"Frame idx {idx} out of range for T_valid={stim.shape[0]}")
        frame = stim[idx, 0, -1].detach().cpu().numpy()
        frames.append(np.asarray(frame, dtype=np.float32))

    # ensure no torch tensors leak
    assert all(not isinstance(f, torch.Tensor) for f in frames)
    return frames


def make_figure(
    decoder_controls: list[DecoderControls],
    neurometric_npz_path: Path,
    eye_traces_path: Path,
    out_path_png: Path,
    out_path_pdf: Path,
    example_logmar: float = -0.40,
    example_orientation: float = 0.0,
    example_trace_index: int = 0,
    example_n_lags: int = 32,
    example_frame_indices: tuple[int, ...] = (0, 15, 30, 45),
    long_window: int = 60,
    verbose: bool = False,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec

    # Typography / export settings: vector-friendly, clean, journal-style defaults.
    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
        }
    )

    # Colors: neutral for stabilized, saturated warm tone for real FEM.
    color_real = "#c44e2b"
    color_real_fill = "#f1c2b3"
    color_stab = "#384047"
    color_stab_fill = "#cfd4d8"
    color_adv = "#1f6aa5"
    color_chance = "#9a9a9a"
    color_help = "#e9f3ea"
    color_hurt = "#f8e9e7"
    panel_bg = "#fbfbf9"

    # Panels B-E data
    logmars = np.array([dc.logmar for dc in decoder_controls], dtype=float)
    keep = logmars < 0
    decoder_controls = [dc for dc, k in zip(decoder_controls, keep) if k]
    logmars = np.array([dc.logmar for dc in decoder_controls], dtype=float)

    if len(decoder_controls) == 0:
        raise RuntimeError("No decoder_controls_lm*.npz files found with logmar < 0")

    w_idx = _pick_window_index(decoder_controls[0].windows, long_window)
    windows = decoder_controls[0].windows

    real_long = np.array([dc.d1_real_mean[w_idx] for dc in decoder_controls])
    real_long_std = np.array([dc.d1_real_std[w_idx] for dc in decoder_controls])
    stab_long = np.array([dc.d1_stabilized_mean[w_idx] for dc in decoder_controls])
    stab_long_std = np.array([dc.d1_stabilized_std[w_idx] for dc in decoder_controls])

    # Example file for the window sweep panel
    example_matches = [dc for dc in decoder_controls if np.isclose(dc.logmar, example_logmar)]
    if len(example_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one decoder-controls file at logmar={example_logmar}, got {len(example_matches)}"
        )
    ex = example_matches[0]

    # Panel A: stimulus snapshots
    traces, durations = _load_eye_traces(eye_traces_path)
    if example_trace_index < 0 or example_trace_index >= traces.shape[0]:
        raise ValueError(f"example_trace_index out of range: {example_trace_index}")
    T = int(durations[example_trace_index])
    eye = traces[example_trace_index, :T]

    real_frames = _render_hires_snapshots(
        orientation_deg=example_orientation,
        logmar=example_logmar,
        eye_trace_deg=eye,
        condition="real",
        valid_frame_indices=list(example_frame_indices),
        n_lags=example_n_lags,
    )
    stab_frames = _render_hires_snapshots(
        orientation_deg=example_orientation,
        logmar=example_logmar,
        eye_trace_deg=eye,
        condition="stabilized",
        valid_frame_indices=list(example_frame_indices),
        n_lags=example_n_lags,
    )

    sweep = _load_neurometric_sweep(neurometric_npz_path)
    sweep_keep = np.asarray(sweep.logmars) < 0
    sweep_logmars = np.asarray(sweep.logmars[sweep_keep], dtype=float)
    sweep_real_A = np.asarray(sweep.real_A_mean[sweep_keep], dtype=float)
    sweep_real_A_std = np.asarray(sweep.real_A_std[sweep_keep], dtype=float)
    sweep_stab_A = np.asarray(sweep.stabilized_A_mean[sweep_keep], dtype=float)
    sweep_stab_A_std = np.asarray(sweep.stabilized_A_std[sweep_keep], dtype=float)
    sweep_advantage = sweep_real_A - sweep_stab_A
    crossover_est = _estimate_crossover(sweep_logmars, sweep_advantage)

    fig = plt.figure(figsize=(9.4, 5.9), constrained_layout=False)
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        width_ratios=[1.2, 1.0, 1.0],
        height_ratios=[1.1, 1.0],
        wspace=0.32,
        hspace=0.34,
    )

    # Panel A spans two columns for a stronger visual opening.
    axA = fig.add_subplot(gs[0, :2])
    axA.set_facecolor(panel_bg)
    axA.set_axis_off()

    n_cols = len(real_frames)
    left = 0.08
    bottom = 0.12
    grid_w = 0.84
    grid_h = 0.72
    pad_x = 0.012
    pad_y = 0.04
    cell_w = (grid_w - (n_cols - 1) * pad_x) / n_cols
    cell_h = (grid_h - 1 * pad_y) / 2

    if verbose:
        print("Panel A: Retinal snapshot min/max values:")
        for c in range(n_cols):
            rmin, rmax = np.min(real_frames[c]), np.max(real_frames[c])
            smin, smax = np.min(stab_frames[c]), np.max(stab_frames[c])
            print(
                f"  real[{c}]: min={rmin:.2f} max={rmax:.2f} | "
                f"stabilized[{c}]: min={smin:.2f} max={smax:.2f}"
            )

    for c in range(n_cols):
        axr = axA.inset_axes(
            (left + c * (cell_w + pad_x), bottom + cell_h + pad_y, cell_w, cell_h)
        )
        axr.imshow(_display_img(_center_crop(real_frames[c])), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axr.set_xticks([])
        axr.set_yticks([])
        for spine in axr.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#aaaaaa")
        axr.text(
            0.5,
            1.02,
            f"f={int(example_frame_indices[c])}",
            ha="center",
            va="bottom",
            transform=axr.transAxes,
            fontsize=7,
            color="#555555",
        )

        axs = axA.inset_axes((left + c * (cell_w + pad_x), bottom, cell_w, cell_h))
        axs.imshow(_display_img(_center_crop(stab_frames[c])), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axs.set_xticks([])
        axs.set_yticks([])
        for spine in axs.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#aaaaaa")

    axA.text(0.0, 1.02, "A", transform=axA.transAxes, fontweight="bold", va="bottom")
    axA.text(
        0.08,
        0.985,
        f"Zoomed retinal input over a fixation (LogMAR {example_logmar:+.2f}, ori={int(example_orientation)} degrees)",
        transform=axA.transAxes,
        va="top",
        fontsize=9.5,
    )
    axA.text(0.03, 0.67, "real FEM", transform=axA.transAxes, rotation=90, va="center", color=color_real)
    axA.text(0.03, 0.23, "stabilized", transform=axA.transAxes, rotation=90, va="center", color=color_stab)
    axA.text(
        0.08,
        0.05,
        "Real FEM sweeps the tiny letter across retinal sampling positions; stabilization removes that motion.",
        transform=axA.transAxes,
        color="#5c5c5c",
    )

    # ── Panel B: integration window sweep at the example LogMAR ─────────────
    axB = fig.add_subplot(gs[0, 2])
    axB.set_facecolor(panel_bg)
    axB.text(-0.18, 1.02, "B", transform=axB.transAxes, fontweight="bold", va="bottom")

    axB.axhline(0.25, color=color_chance, lw=1.0, ls="--", zorder=0)
    axB.fill_between(ex.windows, ex.d1_stabilized_mean, ex.d1_real_mean, color="#f4f1eb", alpha=0.7, zorder=0)
    axB.errorbar(
        ex.windows,
        ex.d1_stabilized_mean,
        yerr=ex.d1_stabilized_std,
        color=color_stab,
        marker="o",
        lw=1.5,
        ms=3,
        capsize=2,
        label="stabilized",
    )
    axB.errorbar(
        ex.windows,
        ex.d1_real_mean,
        yerr=ex.d1_real_std,
        color=color_real,
        marker="o",
        lw=1.5,
        ms=3,
        capsize=2,
        label="real FEM",
    )
    axB.set_title(f"Longer integration reveals FEM benefit\n(LogMAR {example_logmar:+.2f})")
    axB.set_xlabel("Integration window (frames)")
    axB.set_ylabel("Accuracy")
    axB.set_ylim(0.2, 1.02)
    axB.set_xscale("log")
    axB.set_xticks(ex.windows)
    axB.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    axB.grid(True, axis="y", alpha=0.22)
    if np.any(np.asarray(ex.windows) == long_window):
        ew_idx = _pick_window_index(ex.windows, long_window)
        delta_ex = float(ex.d1_real_mean[ew_idx] - ex.d1_stabilized_mean[ew_idx])
        axB.text(
            0.50,
            0.08,
            f"Delta at W={long_window}: {delta_ex:+.3f}",
            transform=axB.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d3d3d3", lw=0.6),
        )
    axB.legend(frameon=False, loc="upper left")

    # ── Panel C: full all-hires sweep across LogMAR ─────────────────────────
    axC = fig.add_subplot(gs[1, 0])
    axC.set_facecolor(panel_bg)
    axC.text(-0.18, 1.02, "C", transform=axC.transAxes, fontweight="bold", va="bottom")

    axC.axhline(0.25, color=color_chance, lw=1.0, ls="--", zorder=0)
    axC.errorbar(
        sweep_logmars,
        sweep_stab_A,
        yerr=sweep_stab_A_std,
        color=color_stab,
        marker="o",
        lw=1.5,
        ms=3,
        capsize=2,
        label="stabilized (Model A)",
    )
    axC.errorbar(
        sweep_logmars,
        sweep_real_A,
        yerr=sweep_real_A_std,
        color=color_real,
        marker="o",
        lw=1.5,
        ms=3,
        capsize=2,
        label="real FEM (Model A)",
    )

    axC.axvspan(min(sweep_logmars) - 0.02, -0.35, color="#eeeeee", zorder=-1, alpha=0.8)
    axC.axvline(example_logmar, color="#b6b6b6", lw=0.9, ls=":")
    axC.text(
        0.02,
        0.06,
        "shaded: hyperacuity regime",
        transform=axC.transAxes,
        color="#666666",
    )

    axC.set_title("Full all-hires sweep across size")
    axC.set_xlabel("LogMAR")
    axC.set_ylabel("Accuracy")
    axC.set_ylim(0.2, 1.02)
    axC.grid(True, axis="y", alpha=0.22)
    axC.legend(frameon=False, loc="upper left")

    # ── Panel D: FEM advantage curve from the same full sweep ───────────────
    axD = fig.add_subplot(gs[1, 1])
    axD.set_facecolor(panel_bg)
    axD.text(-0.18, 1.02, "D", transform=axD.transAxes, fontweight="bold", va="bottom")

    axD.axhspan(0.0, max(np.max(sweep_advantage) + 0.01, 0.01), color=color_help, zorder=0)
    axD.axhspan(min(np.min(sweep_advantage) - 0.01, -0.01), 0.0, color=color_hurt, zorder=0)
    axD.axhline(0.0, color="#666666", lw=1.0)
    axD.plot(sweep_logmars, sweep_advantage, color=color_adv, marker="o", lw=1.8, ms=4)
    axD.fill_between(sweep_logmars, sweep_advantage, 0.0, color="#cfe2f2", alpha=0.6)
    axD.axvline(example_logmar, color="#b6b6b6", lw=0.9, ls=":")
    if crossover_est is not None:
        axD.axvline(crossover_est, color="#6b6b6b", lw=1.0, ls="--")
        axD.text(
            crossover_est,
            0.98,
            f"crossover ~ {crossover_est:+.2f}",
            rotation=90,
            va="top",
            ha="right",
            transform=axD.get_xaxis_transform(),
            color="#5f5f5f",
        )
    axD.set_title("FEM advantage = real - stabilized")
    axD.set_xlabel("LogMAR")
    axD.set_ylabel("Delta accuracy")
    axD.grid(True, axis="y", alpha=0.22)
    axD.text(0.04, 0.90, "FEM helps", transform=axD.transAxes, color="#4d6b4e")
    axD.text(0.04, 0.06, "FEM hurts", transform=axD.transAxes, color="#8b5950")

    # ── Panel E: tail of the same full sweep near the chance boundary ───────
    axE = fig.add_subplot(gs[1, 2])
    axE.set_facecolor(panel_bg)
    axE.text(-0.18, 1.02, "E", transform=axE.transAxes, fontweight="bold", va="bottom")

    axE.axhline(0.25, color=color_chance, lw=1.0, ls="--", zorder=0)
    tail_keep = sweep_logmars <= -0.35
    axE.errorbar(
        sweep_logmars[tail_keep],
        sweep_stab_A[tail_keep],
        yerr=sweep_stab_A_std[tail_keep],
        color=color_stab,
        marker="o",
        lw=1.5,
        ms=3,
        capsize=2,
        label="stabilized (A)",
    )
    axE.errorbar(
        sweep_logmars[tail_keep],
        sweep_real_A[tail_keep],
        yerr=sweep_real_A_std[tail_keep],
        color=color_real,
        marker="o",
        lw=1.5,
        ms=3,
        capsize=2,
        label="real FEM (A)",
    )
    axE.set_title("Same sweep near the chance boundary")
    axE.set_xlabel("LogMAR")
    axE.set_ylabel("Accuracy")
    axE.set_ylim(0.2, 0.85)
    axE.grid(True, axis="y", alpha=0.22)
    axE.legend(frameon=False, loc="upper right")
    idx55 = _find_idx(sweep_logmars, -0.55)
    if idx55 is not None:
        axE.text(
            0.04,
            0.08,
            "Both conditions collapse to chance by -0.55",
            transform=axE.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d3d3d3", lw=0.6),
        )

    fig.suptitle("FEMs improve E-optotype discriminability in the hyperacuity regime", y=0.99, fontsize=12)
    fig.text(
        0.5,
        0.955,
        "A uniform all-hires sweep shows the same story without stitching: FEM hurts near threshold, helps in hyperacuity, then both conditions fail near the rendering limit.",
        ha="center",
        color="#555555",
        fontsize=8.5,
    )

    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    out_path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_png, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Nature-style multi-panel figure for hyperacuity FEM results."
    )
    parser.add_argument(
        "--example_logmar",
        type=float,
        default=-0.40,
        help="LogMAR for the window-sweep panel (default: -0.40)",
    )
    parser.add_argument(
        "--example_orientation",
        type=float,
        default=0.0,
        help="Orientation (deg) for stimulus snapshots (default: 0)",
    )
    parser.add_argument(
        "--example_trace_index",
        type=int,
        default=0,
        help="Which eye trace index to use for snapshots (default: 0)",
    )
    parser.add_argument(
        "--example_frames",
        type=str,
        default="0,15,30,45",
        help="Comma-separated T_valid indices for snapshots (default: 0,15,30,45)",
    )
    parser.add_argument(
        "--long_window",
        type=int,
        default=60,
        help="Window (frames) used for the long-window LogMAR curve (default: 60)",
    )
    parser.add_argument(
        "--out_stem",
        type=str,
        default="fig_hyperacuity_fem_multipanel",
        help="Output filename stem (default: fig_hyperacuity_fem_multipanel)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print snapshot and boundary diagnostics while building the figure.",
    )
    args = parser.parse_args()

    root = _repo_root()
    results_dir = root / "scripts" / "temporal_decoding" / "data" / "results"
    figures_dir = root / "scripts" / "temporal_decoding" / "figures"

    neurometric_npz = results_dir / "neurometric_allhires_fresh.npz"
    eye_traces = root / "scripts" / "temporal_decoding" / "data" / "eye_traces.npz"

    if not neurometric_npz.exists():
        raise FileNotFoundError(neurometric_npz)
    if not eye_traces.exists():
        raise FileNotFoundError(eye_traces)

    decoder_controls = _load_decoder_controls_dir(results_dir)

    frame_indices = tuple(int(x.strip()) for x in args.example_frames.split(",") if x.strip())

    out_png = figures_dir / f"{args.out_stem}.png"
    out_pdf = figures_dir / f"{args.out_stem}.pdf"

    make_figure(
        decoder_controls=decoder_controls,
        neurometric_npz_path=neurometric_npz,
        eye_traces_path=eye_traces,
        out_path_png=out_png,
        out_path_pdf=out_pdf,
        example_logmar=float(args.example_logmar),
        example_orientation=float(args.example_orientation),
        example_trace_index=int(args.example_trace_index),
        example_frame_indices=frame_indices,
        long_window=int(args.long_window),
        verbose=bool(args.verbose),
    )

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
