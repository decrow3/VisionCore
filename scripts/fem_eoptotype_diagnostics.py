"""FEM E-optotype diagnostic pipeline.

Generates retinal movies for a Tumbling-E across a set of sizes, under:
  - stabilized (no motion)
  - fem (real fixation trace)
  - matched_null (optional; shuffled trace)

Then computes simple retinal stats and (optionally) model spatial activations.

This script is intentionally self-contained and borrows conventions from:
  - declan/FEMs_Eoptotype_checks.md
  - scripts/fixrsvp_digitaltwin_spatialinfo_declan.py

Notes
-----
* Eye trace source defaults to declan/fixrsvp_fixation_pool.pkl (list of (T,2) deg traces).
* E rendering + world→retina sampling uses scripts/temporal_decoding/stimulus_hires.py.
* Model normalization follows the repo's `pixelnorm`: (x - 127) / 255.
"""

from __future__ import annotations

import argparse
import sys
import pickle
from pathlib import Path

import numpy as np
import torch


def _pick_best_cuda_device() -> str:
    """Pick the CUDA device with the most free memory.

    Uses PyTorch's CUDA memory query. Indices respect CUDA_VISIBLE_DEVICES.
    """
    if not torch.cuda.is_available():
        return "cpu"

    n = int(torch.cuda.device_count())
    if n <= 1:
        return "cuda:0"

    best_i = 0
    best_free = -1
    for i in range(n):
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except TypeError:
            # Older torch: mem_get_info() doesn't accept a device arg
            with torch.cuda.device(i):
                free, _total = torch.cuda.mem_get_info()

        if int(free) > int(best_free):
            best_free = int(free)
            best_i = int(i)

    return f"cuda:{best_i}"


def _parse_float_list(s: str) -> list[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def _gap_px_to_logmar(gap_px: float, ppd: float) -> float:
    # gap_deg = gap_px / ppd ; gap_arcmin = gap_deg * 60
    # logMAR = log10(gap_arcmin)
    return float(np.log10((gap_px * 60.0) / ppd))


def _load_fixation_pool(pool_path: Path) -> list[np.ndarray]:
    with pool_path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list) or not obj:
        raise ValueError(f"Expected a non-empty list in {pool_path}, got {type(obj)}")
    # validate shape
    for i in range(min(5, len(obj))):
        a = np.asarray(obj[i])
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f"Pool element {i} expected shape (T,2), got {a.shape}")
    return obj


def _select_eye_trace(pool: list[np.ndarray], n_frames: int, index: int | None) -> np.ndarray:
    lengths = np.array([p.shape[0] for p in pool], dtype=int)

    if index is not None:
        if index < 0 or index >= len(pool):
            raise ValueError(f"eye_trace_index out of range: {index} (0..{len(pool)-1})")
        trace = np.asarray(pool[index], dtype=np.float32)
    else:
        # deterministic: choose longest trace that can satisfy n_frames, else global max
        ok = np.where(lengths >= n_frames)[0]
        pick = int(ok[np.argmax(lengths[ok])]) if ok.size else int(np.argmax(lengths))
        trace = np.asarray(pool[pick], dtype=np.float32)

    if trace.shape[0] >= n_frames:
        return trace[:n_frames]
    # If requested longer than available, pad by repeating the final sample.
    pad = np.repeat(trace[-1:, :], n_frames - trace.shape[0], axis=0)
    return np.concatenate([trace, pad], axis=0)


def _make_matched_null(trace: np.ndarray, seed: int) -> np.ndarray:
    """Matched-null eye trace.

    Current implementation is a full time permutation.
    This is *not* FEM-like (it destroys temporal smoothness) and can inflate
    frame-to-frame energy. Prefer omitting matched_null for the first diagnostic,
    or swapping to a smooth null (phase randomization / chunk shuffle).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(trace.shape[0])
    return trace[perm]


def _crop_eye_stim_to_n_frames(eye_stim: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Crop lag-embedded stimulus to exactly n_frames.

    `hires_counterfactual_stim()` may return T+1 frames due to padding + lag embedding.
    For consistent comparisons, we drop the first frame when possible and then
    take exactly `n_frames` frames.
    """
    T = int(eye_stim.shape[0])
    if T < n_frames:
        raise ValueError(f"eye_stim has only {T} frames, but n_frames={n_frames} was requested")

    if T >= n_frames + 1:
        return eye_stim[1 : 1 + n_frames]

    return eye_stim[:n_frames]


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _save_png(path: Path, img: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def _save_mp4(path: Path, frames: np.ndarray, fps: int = 30, vmin: float | None = None, vmax: float | None = None) -> None:
    """Save (T,H,W) grayscale frames to mp4 via matplotlib+ffmpeg."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    path.parent.mkdir(parents=True, exist_ok=True)

    T = int(frames.shape[0])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")

    writer = FFMpegWriter(fps=fps, metadata={"artist": "VisionCore"}, bitrate=3000)
    with writer.saving(fig, str(path), dpi=120):
        for t in range(T):
            ax.clear()
            ax.axis("off")
            ax.imshow(frames[t], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            writer.grab_frame()

    plt.close(fig)


def _save_activation_grid_mp4(
    path: Path,
    maps: np.ndarray,
    fps: int = 15,
    vmin: float = -6.0,
    vmax: float = 6.0,
    pad_value: float = 0.0,
) -> None:
    """Save activation maps (T,N,H,W) as a tiled grid mp4.

    Matches the visual style used by `scripts/spatial_info.make_movie()`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    try:
        from torchvision.utils import make_grid
    except Exception as e:
        raise RuntimeError("torchvision is required to save activation grid movies") from e

    path.parent.mkdir(parents=True, exist_ok=True)

    y = torch.from_numpy(maps)
    if y.ndim != 4:
        raise ValueError(f"Expected maps with shape (T,N,H,W), got {tuple(y.shape)}")

    # z-score each unit over (time, space)
    mu = y.mean(dim=(0, 2, 3), keepdim=True)
    std = y.std(dim=(0, 2, 3), keepdim=True)
    y = (y - mu) / (std + 1e-8)

    T, N, _H, _W = y.shape
    nrow = int(np.ceil(np.sqrt(N)))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")

    writer = FFMpegWriter(fps=fps, metadata={"artist": "VisionCore"}, bitrate=3000)
    with writer.saving(fig, str(path), dpi=120):
        for t in range(T):
            ax.clear()
            ax.axis("off")

            frames = y[t].unsqueeze(1)  # (N, 1, H, W)
            grid = make_grid(frames, nrow=nrow, normalize=False, padding=1, pad_value=float(pad_value))
            ax.imshow(grid[0].numpy(), cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            writer.grab_frame()

    plt.close(fig)


def _retina_stats(movie: np.ndarray) -> dict[str, np.ndarray]:
    # movie: (T,H,W) float32
    mean_img = movie.mean(axis=0)
    std_img = movie.std(axis=0)

    d = movie[1:] - movie[:-1]
    delta_energy = np.sqrt(np.sum(d.astype(np.float64) ** 2, axis=(1, 2))).astype(np.float32)

    y_center = movie.shape[1] // 2
    xt_slice = movie[:, y_center, :].T  # (x, T)
    xt_slice = xt_slice.T  # (T, x)

    return {
        "retina_mean": mean_img.astype(np.float32),
        "retina_std": std_img.astype(np.float32),
        "retina_delta_energy": delta_energy,
        "retina_xt_slice": xt_slice.astype(np.float32),
    }


def _com_and_width(maps: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Compute CoM and 2nd moment widths for maps.

    Args:
        maps: (T, N, H, W) non-negative

    Returns:
        com_traj: (T, N, 2) with (x_com, y_com) in pixel coordinates
        width_traj: (T, N, 2) with (var_x, var_y) in pixel^2
    """
    T, N, H, W = maps.shape
    x = np.arange(W, dtype=np.float32)[None, None, None, :]
    y = np.arange(H, dtype=np.float32)[None, None, :, None]

    wsum = maps.sum(axis=(2, 3), keepdims=True) + eps
    x_com = (maps * x).sum(axis=(2, 3), keepdims=True) / wsum
    y_com = (maps * y).sum(axis=(2, 3), keepdims=True) / wsum

    var_x = (maps * (x - x_com) ** 2).sum(axis=(2, 3), keepdims=True) / wsum
    var_y = (maps * (y - y_com) ** 2).sum(axis=(2, 3), keepdims=True) / wsum

    com = np.concatenate([x_com, y_com], axis=2).squeeze(-1)  # (T,N,2,1)->(T,N,2)
    width = np.concatenate([var_x, var_y], axis=2).squeeze(-1)

    return com.astype(np.float32), width.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_dir",
        type=str,
        default="declan/E_diagnostics",
        help="Output directory root.",
    )

    parser.add_argument(
        "--gap_px_list",
        type=str,
        default="12,9,8,6.5,5,4,3.2,2.5,2.0,1.6",
        help="Comma-separated E gap sizes in *world* pixels (at --world_ppd).",
    )

    parser.add_argument("--orientation_deg", type=float, default=0.0)

    parser.add_argument("--n_frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument(
        "--eye_pool_path",
        type=str,
        default="declan/fixrsvp_fixation_pool.pkl",
        help="Pickle containing list of fixation eye traces (deg).",
    )
    parser.add_argument("--eye_trace_index", type=int, default=None)
    parser.add_argument("--eye_seed", type=int, default=0)

    parser.add_argument("--include_matched_null", action="store_true")

    # hires world→retina sampling parameters
    # NOTE: `world_ppd` controls physical size + rendering resolution.
    #       `retina_ppd`/`retina_size` controls the sampled movie grid (one per run).
    parser.add_argument("--world_ppd", type=float, default=240.0)
    parser.add_argument("--world_size", type=int, default=512)
    parser.add_argument("--retina_ppd", type=float, default=37.50476617)
    parser.add_argument("--retina_size", type=int, default=101)
    parser.add_argument("--n_lags", type=int, default=32)

    # model activations
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--model_index", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda|cpu; default auto")

    parser.add_argument(
        "--save_neural_maps",
        action="store_true",
        help="If set and a model is loaded, save neural maps and stats.",
    )
    parser.add_argument(
        "--max_units",
        type=int,
        default=128,
        help="Max units to keep for neural maps (readout feature maps).",
    )
    parser.add_argument(
        "--dataset_idx",
        type=int,
        default=0,
        help="Dataset index to choose a readout head from (when saving neural maps).",
    )

    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Import hires generator lazily (keeps CLI fast; avoids side-effect imports at module import time)
    this_dir = Path(__file__).resolve().parent
    td_dir = this_dir / "temporal_decoding"
    if str(td_dir) not in sys.path:
        sys.path.insert(0, str(td_dir))
    from stimulus_hires import hires_counterfactual_stim  # type: ignore

    print(
        "Settings:\n"
        f"  world_ppd={args.world_ppd} (render/size convention)\n"
        f"  retina_ppd={args.retina_ppd}, retina_size={args.retina_size} (sampled movie grid; one per run)\n"
        f"  n_frames={args.n_frames}, n_lags={args.n_lags}\n"
        "Tip: run twice for two diagnostics (e.g. retina_ppd=240 for human inspection, then retina_ppd=37.5 for model).\n"
    )

    # Load eye trace
    pool = _load_fixation_pool(Path(args.eye_pool_path))
    fem_trace = _select_eye_trace(pool, n_frames=args.n_frames, index=args.eye_trace_index)
    fem_trace = fem_trace - fem_trace.mean(axis=0, keepdims=True)  # keep centered

    null_trace = None
    if args.include_matched_null:
        print("WARNING: --include_matched_null uses a full time-permutation null (not FEM-like).")
        null_trace = _make_matched_null(fem_trace, seed=args.eye_seed)

    # Optional model load
    model = None
    device = args.device
    if device is None:
        device = _pick_best_cuda_device() if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            try:
                free, total = torch.cuda.mem_get_info(int(device.split(":", 1)[1]))
                print(f"Auto-selected {device} (free {free / (1024**3):.2f} / {total / (1024**3):.2f} GiB)")
            except Exception:
                print(f"Auto-selected {device}")

    if args.checkpoint_path or args.model_type:
        from eval.eval_stack_multidataset import load_model

        ckpt_dir = args.checkpoint_dir or "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/checkpoints"
        model, _info = load_model(
            model_type=args.model_type,
            model_index=args.model_index,
            checkpoint_path=args.checkpoint_path,
            checkpoint_dir=ckpt_dir,
            device=device,
            verbose=True,
        )
        model.model.eval()

    gap_list = _parse_float_list(args.gap_px_list)
    if not gap_list:
        raise ValueError("gap_px_list is empty")

    # Trace/FOV sanity check (worst-case letter size: max gap in list)
    worst_gap_px = float(np.max(np.asarray(gap_list, dtype=np.float32)))
    worst_logmar = _gap_px_to_logmar(worst_gap_px, ppd=float(args.world_ppd))

    # Letter height = 5 * gap, so half-letter in degrees is (2.5 * gap_px) / world_ppd.
    half_letter_deg = (2.5 * worst_gap_px) / float(args.world_ppd)

    # Retinal patch half-width in degrees.
    half_fov_deg = (0.5 * float(args.retina_size)) / float(args.retina_ppd)

    # Eye trace excursion after mean-centering.
    exc_x = float(np.max(np.abs(fem_trace[:, 0])))
    exc_y = float(np.max(np.abs(fem_trace[:, 1])))
    exc_deg = max(exc_x, exc_y)

    # Small border margin (in pixels) to avoid grazing the edge.
    margin_px = 2.0
    margin_deg = margin_px / float(args.retina_ppd)

    required_half_fov_deg = exc_deg + half_letter_deg + margin_deg
    if required_half_fov_deg > half_fov_deg:
        required_retina_size = int(np.ceil(2.0 * required_half_fov_deg * float(args.retina_ppd)))
        print(
            "WARNING: eye trace may clip the worst-case E within the sampled retinal patch.\n"
            f"  worst_gap_px={worst_gap_px:g} (logMAR={worst_logmar:+.2f}), half_letter_deg={half_letter_deg:.4f}\n"
            f"  eye_excursion_deg: max(|x|)={exc_x:.4f}, max(|y|)={exc_y:.4f} (using {exc_deg:.4f})\n"
            f"  patch_half_fov_deg={half_fov_deg:.4f} (retina_size={args.retina_size}, retina_ppd={args.retina_ppd})\n"
            f"  suggested: --retina_size >= {required_retina_size} (keeps a ~{margin_px:g}px margin)\n"
        )

    for i, gap_px in enumerate(gap_list):
        logmar = _gap_px_to_logmar(gap_px, ppd=float(args.world_ppd))
        size_dir = out_root / f"size_{i:02d}_gap_{gap_px:g}_logmar_{logmar:+.2f}"
        size_dir.mkdir(parents=True, exist_ok=True)

        # Build stimuli (embedded lags) for each condition
        eye_stim_fem = hires_counterfactual_stim(
            orientation_deg=float(args.orientation_deg),
            logmar=float(logmar),
            eyepos=fem_trace,
            condition="real",
            null_trace=null_trace,
            n_lags=int(args.n_lags),
            retina_size=(int(args.retina_size), int(args.retina_size)),
            world_size=(int(args.world_size), int(args.world_size)),
            world_ppd=float(args.world_ppd),
            retina_ppd=float(args.retina_ppd),
            device="cpu",  # keep stimulus generation on CPU for determinism
        )

        eye_stim_stab = hires_counterfactual_stim(
            orientation_deg=float(args.orientation_deg),
            logmar=float(logmar),
            eyepos=fem_trace,
            condition="stabilized",
            null_trace=null_trace,
            n_lags=int(args.n_lags),
            retina_size=(int(args.retina_size), int(args.retina_size)),
            world_size=(int(args.world_size), int(args.world_size)),
            world_ppd=float(args.world_ppd),
            retina_ppd=float(args.retina_ppd),
            device="cpu",
        )

        eye_stim_null = None
        if args.include_matched_null:
            eye_stim_null = hires_counterfactual_stim(
                orientation_deg=float(args.orientation_deg),
                logmar=float(logmar),
                eyepos=fem_trace,
                condition="matched_null",
                null_trace=null_trace,
                n_lags=int(args.n_lags),
                retina_size=(int(args.retina_size), int(args.retina_size)),
                world_size=(int(args.world_size), int(args.world_size)),
                world_ppd=float(args.world_ppd),
                retina_ppd=float(args.retina_ppd),
                device="cpu",
            )

        # Force consistent duration (avoid warm-up / off-by-one issues)
        eye_stim_fem = _crop_eye_stim_to_n_frames(eye_stim_fem, n_frames=int(args.n_frames))
        eye_stim_stab = _crop_eye_stim_to_n_frames(eye_stim_stab, n_frames=int(args.n_frames))
        if eye_stim_null is not None:
            eye_stim_null = _crop_eye_stim_to_n_frames(eye_stim_null, n_frames=int(args.n_frames))

        # Retinal movies (current frame from lag-embedded tensor)
        # stimulus_hires returns eye_stim scaled by /127.0, with background near 1 and E near 0.
        retina_fem = (eye_stim_fem[:, 0, 0].numpy() * 127.0).astype(np.float32)
        retina_stab = (eye_stim_stab[:, 0, 0].numpy() * 127.0).astype(np.float32)
        retina_null = (eye_stim_null[:, 0, 0].numpy() * 127.0).astype(np.float32) if eye_stim_null is not None else None

        # Save mp4s
        try:
            _save_mp4(size_dir / "retina_stabilized.mp4", retina_stab, fps=int(args.fps), vmin=0, vmax=127)
            _save_mp4(size_dir / "retina_fem.mp4", retina_fem, fps=int(args.fps), vmin=0, vmax=127)
            if retina_null is not None:
                _save_mp4(size_dir / "retina_matched_null.mp4", retina_null, fps=int(args.fps), vmin=0, vmax=127)
        except Exception as e:
            print(f"WARNING: failed to write mp4s (ffmpeg missing?): {e}")

        # Retinal stats (per condition)
        cond_movies: dict[str, np.ndarray] = {
            "stabilized": retina_stab,
            "fem": retina_fem,
        }
        if retina_null is not None:
            cond_movies["matched_null"] = retina_null

        for cond, mov in cond_movies.items():
            stats = _retina_stats(mov)
            _save_npy(size_dir / f"retina_mean_{cond}.npy", stats["retina_mean"])
            _save_npy(size_dir / f"retina_std_{cond}.npy", stats["retina_std"])
            _save_npy(size_dir / f"retina_delta_energy_{cond}.npy", stats["retina_delta_energy"])
            _save_npy(size_dir / f"retina_xt_slice_{cond}.npy", stats["retina_xt_slice"])

            _save_png(size_dir / f"retinal_mean_{cond}.png", stats["retina_mean"], vmin=0, vmax=127)
            _save_png(size_dir / f"retinal_std_{cond}.png", stats["retina_std"])

        # Optional neural maps
        if model is None or not args.save_neural_maps:
            continue

        # Convert to model input convention:
        # eye_stim_* is (T,1,n_lags,H,W) in [0,1] with background~1.
        # Reconstruct gray values (0..127) and apply pixelnorm: (x-127)/255.
        def to_model_stim(eye_stim: torch.Tensor) -> torch.Tensor:
            x = eye_stim * 127.0
            return (x - 127.0) / 255.0

        stim_fem = to_model_stim(eye_stim_fem).to(device)
        stim_stab = to_model_stim(eye_stim_stab).to(device)

        with torch.no_grad():
            # Core features (T, C, S, Hf, Wf)
            core_fem = model.model.core_forward(stim_fem, behavior=None)
            core_stab = model.model.core_forward(stim_stab, behavior=None)

            feats_fem = core_fem[:, :, -1]  # (T, C, Hf, Wf)
            feats_stab = core_stab[:, :, -1]

            # Readout feature maps: (T, n_units, Hf, Wf)
            readout = model.model.readouts[int(args.dataset_idx)]
            maps_fem = readout.features(feats_fem)
            maps_stab = readout.features(feats_stab)
            if getattr(readout, "bias", None) is not None:
                maps_fem = maps_fem + readout.bias[None, :, None, None]
                maps_stab = maps_stab + readout.bias[None, :, None, None]

            # Activation → nonnegative
            maps_fem = model.model.activation(maps_fem)
            maps_stab = model.model.activation(maps_stab)

            # Keep at most max_units
            n_units = maps_fem.shape[1]
            keep = min(int(args.max_units), int(n_units))
            maps_fem = maps_fem[:, :keep]
            maps_stab = maps_stab[:, :keep]

        maps_fem_np = maps_fem.detach().cpu().numpy().astype(np.float32)
        maps_stab_np = maps_stab.detach().cpu().numpy().astype(np.float32)

        _save_npy(size_dir / "neural_maps_fem.npy", maps_fem_np)
        _save_npy(size_dir / "neural_maps_stabilized.npy", maps_stab_np)

        mean_map = maps_fem_np.mean(axis=0)
        std_map = maps_fem_np.std(axis=0)
        _save_npy(size_dir / "neural_mean_map.npy", mean_map.astype(np.float32))
        _save_npy(size_dir / "neural_std_map.npy", std_map.astype(np.float32))

        # CoM + width on FEM maps
        com_traj, width_traj = _com_and_width(maps_fem_np)
        _save_npy(size_dir / "com_traj.npy", com_traj)
        _save_npy(size_dir / "width_traj.npy", width_traj)

        # Quick visualization: avg across units
        _save_png(size_dir / "neural_mean_map_avg.png", mean_map.mean(axis=0))
        _save_png(size_dir / "neural_std_map_avg.png", std_map.mean(axis=0))

        # Activation grid movies (similar to fixrsvp spatial_info activations)
        try:
            _save_activation_grid_mp4(
                size_dir / "neural_activations_fem.mp4",
                maps_fem_np,
                fps=int(args.fps),
                vmin=-6.0,
                vmax=6.0,
            )
            _save_activation_grid_mp4(
                size_dir / "neural_activations_stabilized.mp4",
                maps_stab_np,
                fps=int(args.fps),
                vmin=-6.0,
                vmax=6.0,
            )
        except Exception as e:
            print(f"WARNING: failed to write neural activation grid mp4s: {e}")


if __name__ == "__main__":
    main()
