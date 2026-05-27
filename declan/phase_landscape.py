from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_DIR = ROOT / "scripts" / "temporal_decoding"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEMPORAL_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPORAL_DIR))

from geometry_utils import (  # noqa: E402
    ORIENTATIONS,
    compute_alpha,
    compute_signal_covariance,
    compute_translation_mimicry,
    dump_json,
    format_logmar,
    maybe_git_commit,
    orthonormal_basis,
)
from scripts.temporal_decoding.rate_computation import _collapse_spatial  # noqa: E402
from scripts.temporal_decoding.stimulus_hires import (  # noqa: E402
    BLUR_SIGMA,
    HiResERenderer,
    HiResRetina,
    RETINA_PPD,
    RETINA_SIZE,
    WORLD_PPD,
    WORLD_SIZE,
)


DEFAULT_OUTPUT_DIR = ROOT / "declan" / "results" / "phase_landscape"
EYE_TRACES_PATH = TEMPORAL_DIR / "data" / "eye_traces.npz"
PKL_PATH = ROOT / "scripts" / "mcfarland_outputs_mono.pkl"
PIXEL_DEG = 1.0 / RETINA_PPD


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(float(x)) for x in str(text).split(",") if str(x).strip())


def _load_eye_traces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["traces"].astype(np.float32), data["durations"].astype(np.int32)


def _grand_mean_eye_pos(traces: np.ndarray, durations: np.ndarray) -> np.ndarray:
    all_pos = np.concatenate([traces[i, : durations[i]] for i in range(len(durations))], axis=0)
    return all_pos.mean(axis=0).astype(np.float32)


def _eye_overlay_positions(traces: np.ndarray, durations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grand_mean = _grand_mean_eye_pos(traces, durations)
    trial_means = np.stack([traces[i, : durations[i]].mean(axis=0) for i in range(len(durations))], axis=0)
    frame_pos = np.concatenate([traces[i, : durations[i]] for i in range(len(durations))], axis=0)
    return grand_mean, trial_means - grand_mean[None, :], frame_pos - grand_mean[None, :]


def _load_model_and_readout(device: str | None = None):
    import dill
    from spatial_info import get_spatial_readout
    from utils import get_model_and_dataset_configs

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = get_model_and_dataset_configs(mode="standard")
    model.model.eval()
    if hasattr(model.model, "convnet") and hasattr(model.model.convnet, "use_checkpointing"):
        model.model.convnet.use_checkpointing = False
    model = model.to(device)

    with open(PKL_PATH, "rb") as f:
        outputs = dill.load(f)

    readout = get_spatial_readout(model, outputs).to(device)
    readout.eval()
    return model, readout


def _build_static_offset_movie(
    orientation_deg: float,
    logmar: float,
    center_offset_deg: tuple[float, float],
    n_frames: int,
    device: str,
) -> torch.Tensor:
    renderer = HiResERenderer(
        ppd=WORLD_PPD,
        canvas_size=WORLD_SIZE,
        blur_sigma=BLUR_SIGMA,
        device=device,
    ).to(device)
    retina = HiResRetina(
        world_ppd=WORLD_PPD,
        retina_ppd=RETINA_PPD,
        world_canvas_size=WORLD_SIZE,
        retina_size=RETINA_SIZE,
    ).to(device)
    renderer.eval()
    retina.eval()

    eyepos = torch.zeros((n_frames, 2), device=device, dtype=torch.float32)
    with torch.no_grad():
        world_img = renderer(orientation_deg, logmar, center_offset_deg=center_offset_deg)
        world_gray = 127.0 * (1.0 - world_img)
        movie = retina(world_gray, eyepos)[0, 0] / 127.0
    return movie.cpu()


def _compute_time_mean_rate(model, readout, movie: torch.Tensor, spatial_collapse: str = "max") -> np.ndarray:
    device = next(model.model.parameters()).device
    use_amp = device.type == "cuda"
    x = movie.unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        x_front = model.model.frontend(x)
        x_conv = model.model.convnet(x_front)
        x_recurrent = model.model.recurrent(x_conv)
        feats = x_recurrent[0].permute(1, 0, 2, 3).contiguous()
        y = readout(feats)
        rates = _collapse_spatial(model.model.activation(y), method=spatial_collapse).float().cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rates.mean(axis=0).astype(np.float32)


def _finite_difference_jacobian(rates: np.ndarray, offset_x_deg: np.ndarray, offset_y_deg: np.ndarray) -> np.ndarray:
    _, n_ori, nx, ny, n_neurons = rates.shape
    jac = np.zeros((rates.shape[0], n_ori, nx, ny, n_neurons, 2), dtype=np.float32)
    for li in range(rates.shape[0]):
        for oi in range(n_ori):
            arr = rates[li, oi]
            for ix in range(nx):
                if ix == 0:
                    dx = offset_x_deg[ix + 1] - offset_x_deg[ix]
                    grad_x = (arr[ix + 1] - arr[ix]) / dx
                elif ix == nx - 1:
                    dx = offset_x_deg[ix] - offset_x_deg[ix - 1]
                    grad_x = (arr[ix] - arr[ix - 1]) / dx
                else:
                    dx = offset_x_deg[ix + 1] - offset_x_deg[ix - 1]
                    grad_x = (arr[ix + 1] - arr[ix - 1]) / dx
                for iy in range(ny):
                    if iy == 0:
                        dy = offset_y_deg[iy + 1] - offset_y_deg[iy]
                        grad_y = (arr[ix, iy + 1] - arr[ix, iy]) / dy
                    elif iy == ny - 1:
                        dy = offset_y_deg[iy] - offset_y_deg[iy - 1]
                        grad_y = (arr[ix, iy] - arr[ix, iy - 1]) / dy
                    else:
                        dy = offset_y_deg[iy + 1] - offset_y_deg[iy - 1]
                        grad_y = (arr[ix, iy + 1] - arr[ix, iy - 1]) / dy
                    jac[li, oi, ix, iy, :, 0] = grad_x[iy]
                    jac[li, oi, ix, iy, :, 1] = grad_y
    return jac


def _pairwise_separations(class_means: np.ndarray) -> np.ndarray:
    n_ori = class_means.shape[0]
    out = np.full((n_ori, n_ori), np.nan, dtype=np.float64)
    for a in range(n_ori):
        for b in range(n_ori):
            if a == b:
                continue
            out[a, b] = float(np.linalg.norm(class_means[b] - class_means[a]))
    return out


def _save_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No phase summary rows to save")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_heatmaps(
    output_dir: Path,
    title_prefix: str,
    values: np.ndarray,
    logmars: list[float],
    offset_x_arcmin: np.ndarray,
    offset_y_arcmin: np.ndarray,
    trial_means_arcmin: np.ndarray,
    frame_pos_arcmin: np.ndarray,
    cmap: str,
    file_name: str,
) -> None:
    n_log = len(logmars)
    fig, axes = plt.subplots(1, n_log, figsize=(4.2 * n_log, 4.0), squeeze=False, constrained_layout=True)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    for ax, logmar, arr in zip(axes[0], logmars, values):
        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=[offset_x_arcmin[0], offset_x_arcmin[-1], offset_y_arcmin[0], offset_y_arcmin[-1]],
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.scatter(trial_means_arcmin[:, 0], trial_means_arcmin[:, 1], s=10, alpha=0.2, color="#111111", label="trial means")
        ax.scatter(frame_pos_arcmin[:: max(1, len(frame_pos_arcmin) // 5000), 0], frame_pos_arcmin[:: max(1, len(frame_pos_arcmin) // 5000), 1], s=1, alpha=0.05, color="#c62828", label="frame positions")
        ax.scatter([0.0], [0.0], s=50, marker="x", color="#ffffff")
        ax.set_title(f"lm={logmar:+.2f}")
        ax.set_xlabel("x offset (arcmin)")
        ax.set_ylabel("y offset (arcmin)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label=title_prefix)
    fig.savefig(output_dir / file_name, dpi=200)
    plt.close(fig)


def _plot_fixed_center_histogram(
    output_dir: Path,
    mean_sep: np.ndarray,
    logmars: list[float],
) -> None:
    fig, axes = plt.subplots(1, len(logmars), figsize=(4.0 * len(logmars), 3.4), squeeze=False, constrained_layout=True)
    center_idx_x = mean_sep.shape[1] // 2
    center_idx_y = mean_sep.shape[2] // 2
    for ax, logmar, arr in zip(axes[0], logmars, mean_sep):
        flat = arr.reshape(-1)
        ax.hist(flat[np.isfinite(flat)], bins=30, color="#90caf9", edgecolor="#1565c0")
        ax.axvline(arr[center_idx_x, center_idx_y], color="#c62828", linestyle="--", label="fixed center")
        ax.set_title(f"lm={logmar:+.2f}")
        ax.set_xlabel("mean pairwise separation")
        ax.set_ylabel("count")
        ax.legend(frameon=False)
    fig.savefig(output_dir / "phase_landscape_fixed_center_histogram.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-landscape analysis for deterministic E-optotype geometry")
    parser.add_argument("--logmars", type=str, default="-0.20,-0.35,-0.40")
    parser.add_argument("--orientations", type=str, default="0,90,180,270")
    parser.add_argument("--grid_range_pix", type=float, default=3.0)
    parser.add_argument("--grid_size", type=int, default=17)
    parser.add_argument("--n_frames", type=int, default=120)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--spatial_collapse", choices=("max", "mean"), default="max")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--svd_eps", type=float, default=1e-9)
    parser.add_argument("--ridge_scale", type=float, default=1e-6)
    parser.add_argument("--arcmin_limits", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--n_constrained_angles", type=int, default=720)
    args = parser.parse_args()

    logmars = _parse_csv_floats(args.logmars)
    orientations = _parse_csv_ints(args.orientations)
    arcmin_limits = tuple(_parse_csv_floats(args.arcmin_limits))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    traces, durations = _load_eye_traces(EYE_TRACES_PATH)
    grand_mean, trial_means_rel, frame_pos_rel = _eye_overlay_positions(traces, durations)
    trial_means_arcmin = trial_means_rel * 60.0
    frame_pos_arcmin = frame_pos_rel * 60.0

    offset_pix = np.linspace(-args.grid_range_pix, args.grid_range_pix, args.grid_size, dtype=np.float32)
    offset_deg = offset_pix / RETINA_PPD
    model, readout = _load_model_and_readout(device=args.device)
    device = next(model.model.parameters()).device.type

    rates = np.zeros((len(logmars), len(orientations), args.grid_size, args.grid_size, 756), dtype=np.float32)
    for li, logmar in enumerate(logmars):
        for oi, orientation in enumerate(orientations):
            for ix, dx in enumerate(offset_deg):
                for iy, dy in enumerate(offset_deg):
                    movie = _build_static_offset_movie(
                        orientation_deg=float(orientation),
                        logmar=float(logmar),
                        center_offset_deg=(float(dx), float(dy)),
                        n_frames=int(args.n_frames),
                        device=device,
                    )
                    rates[li, oi, ix, iy] = _compute_time_mean_rate(
                        model,
                        readout,
                        movie,
                        spatial_collapse=args.spatial_collapse,
                    )

    jac = _finite_difference_jacobian(rates, offset_deg, offset_deg)

    signal_trace = np.zeros((len(logmars), args.grid_size, args.grid_size), dtype=np.float32)
    mean_pairwise_sep = np.zeros_like(signal_trace)
    min_pairwise_sep = np.zeros_like(signal_trace)
    alpha_pooled = np.zeros_like(signal_trace)
    alpha_orientation_mean = np.zeros_like(signal_trace)
    mean_mimicry = np.zeros_like(signal_trace)
    max_mimicry = np.zeros_like(signal_trace)
    jacobian_norm = np.zeros((len(logmars), len(orientations), args.grid_size, args.grid_size), dtype=np.float32)
    jacobian_anisotropy = np.zeros_like(jacobian_norm)
    pairwise_mimicry = np.full((len(logmars), args.grid_size, args.grid_size, len(orientations), len(orientations)), np.nan, dtype=np.float32)
    summary_rows: list[dict] = []

    for li, logmar in enumerate(logmars):
        for ix in range(args.grid_size):
            for iy in range(args.grid_size):
                class_means = rates[li, :, ix, iy, :].astype(np.float64)
                signal_cov = compute_signal_covariance(class_means)
                signal_trace[li, ix, iy] = float(np.trace(signal_cov))

                pair_sep = _pairwise_separations(class_means)
                mean_pairwise_sep[li, ix, iy] = float(np.nanmean(pair_sep))
                min_pairwise_sep[li, ix, iy] = float(np.nanmin(pair_sep))

                pooled_J = np.concatenate([jac[li, oi, ix, iy] for oi in range(len(orientations))], axis=1)
                pooled_U, _ = orthonormal_basis(pooled_J, svd_eps=args.svd_eps)
                alpha_pooled[li, ix, iy] = compute_alpha(pooled_U, signal_cov)

                alpha_vals = []
                mimicry_vals = []
                for oi, orientation in enumerate(orientations):
                    J_here = jac[li, oi, ix, iy].astype(np.float64)
                    U_here, S_here = orthonormal_basis(J_here, svd_eps=args.svd_eps)
                    alpha_vals.append(compute_alpha(U_here, signal_cov))
                    jacobian_norm[li, oi, ix, iy] = float(np.sqrt(np.trace(J_here.T @ J_here)))
                    jacobian_anisotropy[li, oi, ix, iy] = float(S_here[0] / max(S_here[-1], 1e-12)) if S_here.size else np.nan
                    for oj, _ in enumerate(orientations):
                        if oi == oj:
                            continue
                        metrics = compute_translation_mimicry(
                            class_means[oi],
                            class_means[oj],
                            J_here,
                            ridge_scale=args.ridge_scale,
                            arcmin_limits=arcmin_limits,
                            svd_eps=args.svd_eps,
                            n_constrained_angles=args.n_constrained_angles,
                        )
                        pairwise_mimicry[li, ix, iy, oi, oj] = metrics["mimicry_unconstrained"]
                        mimicry_vals.append(metrics["mimicry_unconstrained"])

                alpha_orientation_mean[li, ix, iy] = float(np.mean(alpha_vals))
                mean_mimicry[li, ix, iy] = float(np.mean(mimicry_vals))
                max_mimicry[li, ix, iy] = float(np.max(mimicry_vals))
                summary_rows.append(
                    {
                        "logmar": float(logmar),
                        "offset_x_deg": float(offset_deg[ix]),
                        "offset_y_deg": float(offset_deg[iy]),
                        "offset_x_arcmin": float(offset_deg[ix] * 60.0),
                        "offset_y_arcmin": float(offset_deg[iy] * 60.0),
                        "signal_trace": float(signal_trace[li, ix, iy]),
                        "mean_pairwise_sep": float(mean_pairwise_sep[li, ix, iy]),
                        "min_pairwise_sep": float(min_pairwise_sep[li, ix, iy]),
                        "alpha_pooled": float(alpha_pooled[li, ix, iy]),
                        "alpha_orientation_mean": float(alpha_orientation_mean[li, ix, iy]),
                        "mean_mimicry": float(mean_mimicry[li, ix, iy]),
                        "max_mimicry": float(max_mimicry[li, ix, iy]),
                    }
                )

    np.savez_compressed(
        args.output_dir / "phase_landscape_metrics.npz",
        logmars=np.asarray(logmars, dtype=np.float64),
        orientations=np.asarray(orientations, dtype=np.int64),
        offset_x_pix=offset_pix,
        offset_y_pix=offset_pix,
        offset_x_deg=offset_deg,
        offset_y_deg=offset_deg,
        rates=rates,
        jacobian=jac,
        signal_trace=signal_trace,
        mean_pairwise_sep=mean_pairwise_sep,
        min_pairwise_sep=min_pairwise_sep,
        alpha_pooled=alpha_pooled,
        alpha_orientation_mean=alpha_orientation_mean,
        mean_mimicry=mean_mimicry,
        max_mimicry=max_mimicry,
        pairwise_mimicry=pairwise_mimicry,
        jacobian_norm=jacobian_norm,
        jacobian_anisotropy=jacobian_anisotropy,
        eye_trial_mean_xy=trial_means_rel,
        eye_frame_xy=frame_pos_rel,
        grand_mean_eye_pos=grand_mean,
    )
    _save_summary_csv(args.output_dir / "phase_landscape_summary.csv", summary_rows)
    dump_json(
        args.output_dir / "phase_landscape_config.json",
        {
            "script": "declan/phase_landscape.py",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": maybe_git_commit(ROOT),
            "logmars": logmars,
            "orientations": list(orientations),
            "grid_range_pix": args.grid_range_pix,
            "grid_size": args.grid_size,
            "pixel_deg": PIXEL_DEG,
            "n_frames": args.n_frames,
            "device": device,
            "spatial_collapse": args.spatial_collapse,
            "ridge_scale": args.ridge_scale,
            "arcmin_limits": list(arcmin_limits),
            "retina_ppd": RETINA_PPD,
            "world_ppd": WORLD_PPD,
            "retina_size": list(RETINA_SIZE),
            "world_size": list(WORLD_SIZE),
            "model_checkpoint": "learned_resnet_none_convgru_gaussian epoch 147",
        },
    )

    offset_arcmin = offset_deg * 60.0
    _plot_heatmaps(
        args.output_dir,
        "mean pairwise separation",
        mean_pairwise_sep,
        logmars,
        offset_arcmin,
        offset_arcmin,
        trial_means_arcmin,
        frame_pos_arcmin,
        cmap="viridis",
        file_name="phase_landscape_class_separation.png",
    )
    _plot_heatmaps(
        args.output_dir,
        "alpha pooled",
        alpha_pooled,
        logmars,
        offset_arcmin,
        offset_arcmin,
        trial_means_arcmin,
        frame_pos_arcmin,
        cmap="magma",
        file_name="phase_landscape_alpha.png",
    )
    _plot_heatmaps(
        args.output_dir,
        "mean mimicry",
        mean_mimicry,
        logmars,
        offset_arcmin,
        offset_arcmin,
        trial_means_arcmin,
        frame_pos_arcmin,
        cmap="plasma",
        file_name="phase_landscape_mimicry.png",
    )
    _plot_fixed_center_histogram(args.output_dir, mean_pairwise_sep, logmars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())