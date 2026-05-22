"""
Check whether lo-res vs hi-res stimulus generation produces comparable temporal flicker
(and detect padding/edge artifacts like black-border flicker).

This does NOT run the model; it only generates stimuli and computes simple metrics.

Example:
  python scripts/temporal_decoding/check_stimulus_flicker_dual_regime.py \
    --trace_idx 0 --logmars 0.4 0.3 0.2 0.0 --conditions real stabilized --save_movies
"""
import os
import sys
import argparse
import numpy as np

# Match the import style used by the temporal_decoding scripts: add scripts/ to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_eye_traces import load_eye_traces
from stimulus import e_optotype_stack
from rate_computation import build_counterfactual_stim, OUT_SIZE, PPD, N_LAGS
from stimulus_hires import hires_counterfactual_stim, save_stimulus_mp4


def _trim_first_frame(movie_thw: np.ndarray, keep_first: bool) -> np.ndarray:
    # Both pipelines typically produce T+1 frames due to lag padding; dropping the first makes length=T.
    if keep_first or movie_thw.shape[0] <= 1:
        return movie_thw
    return movie_thw[1:]


def flicker_rms(movie_thw: np.ndarray) -> float:
    """Global RMS of frame-to-frame pixel changes."""
    d = np.diff(movie_thw, axis=0)
    return float(np.sqrt(np.mean(d * d)))


def flicker_rms_mean_intensity(movie_thw: np.ndarray) -> float:
    """RMS of frame-to-frame changes in mean intensity (catches big padding jumps)."""
    m = movie_thw.mean(axis=(1, 2))
    dm = np.diff(m, axis=0)
    return float(np.sqrt(np.mean(dm * dm)))


def dark_fraction_stats(movie_thw: np.ndarray, dark_thresh: float = 0.05) -> dict:
    """
    Fraction of pixels near black each frame. Useful to detect lo-res padding_mode='zeros'
    artifacts (moving black border) since background should be ~1.0 after /127 normalization.
    """
    frac = (movie_thw < dark_thresh).mean(axis=(1, 2))
    return {
        "dark_frac_mean": float(frac.mean()),
        "dark_frac_std": float(frac.std()),
        "dark_frac_min": float(frac.min()),
        "dark_frac_max": float(frac.max()),
    }


def build_movies_for_one(eyepos: np.ndarray, *, ori: int, logmar: float, condition: str,
                         n_lags: int, keep_first_frame: bool, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      movie_lo: (T or T+1, H, W) float32 in ~[0,1]
      movie_hi: (T or T+1, H, W) float32 in ~[0,1]
    """
    T = eyepos.shape[0]

    # Lo-res (same path used by compute_population_rates)
    stim_stack = e_optotype_stack(
        ori, logmar,
        n_frames=T + n_lags + 2,  # ensure enough frames for padding
        ppd=PPD,
    )
    stim_lo = build_counterfactual_stim(
        stim_stack,
        eyepos,
        condition=condition,
        n_lags=n_lags,
        out_size=OUT_SIZE,
        ppd=PPD,
    ) / 127.0
    movie_lo = stim_lo[:, 0, 0].cpu().numpy()  # (T+1, H, W)
    movie_lo = _trim_first_frame(movie_lo, keep_first_frame)

    # Hi-res (same path used by compute_population_rates_hires)
    stim_hi = hires_counterfactual_stim(
        ori, logmar, eyepos,
        condition=condition,
        n_lags=n_lags,
        retina_size=OUT_SIZE,
        device=device,
    )
    movie_hi = stim_hi[:, 0, 0].cpu().numpy()
    movie_hi = _trim_first_frame(movie_hi, keep_first_frame)

    return movie_lo.astype(np.float32), movie_hi.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_idx", type=int, default=0)
    parser.add_argument("--max_T", type=int, default=240, help="truncate trace length for speed")
    parser.add_argument("--logmars", type=float, nargs="+", default=[0.4, 0.3, 0.2, 0.0, -0.1])
    parser.add_argument("--orientations", type=int, nargs="+", default=[0])
    parser.add_argument("--conditions", type=str, nargs="+", default=["real", "stabilized"])
    parser.add_argument("--n_lags", type=int, default=N_LAGS)
    parser.add_argument("--device", type=str, default="cpu", help="hi-res stimulus device (cpu/cuda:0/...)")
    parser.add_argument("--keep_first_frame", action="store_true")
    parser.add_argument("--save_movies", action="store_true")
    parser.add_argument("--save_diff_movie", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    eye_traces_path = os.path.join(data_dir, "eye_traces.npz")
    td = load_eye_traces(eye_traces_path)

    i = int(args.trace_idx)
    T_full = int(td["durations"][i])
    T = min(T_full, int(args.max_T))
    eyepos = td["traces"][i, :T].astype(np.float32)

    print(f"trace_idx={i}  T_full={T_full}  using_T={T}")
    print(f"OUT_SIZE={OUT_SIZE}  n_lags={args.n_lags}")
    print()

    for cond in args.conditions:
        for ori in args.orientations:
            for lm in args.logmars:
                movie_lo, movie_hi = build_movies_for_one(
                    eyepos,
                    ori=ori,
                    logmar=float(lm),
                    condition=cond,
                    n_lags=int(args.n_lags),
                    keep_first_frame=bool(args.keep_first_frame),
                    device=str(args.device),
                )

                # Basic flicker metrics
                lo_f = flicker_rms(movie_lo)
                hi_f = flicker_rms(movie_hi)
                lo_m = flicker_rms_mean_intensity(movie_lo)
                hi_m = flicker_rms_mean_intensity(movie_hi)
                lo_dark = dark_fraction_stats(movie_lo)
                hi_dark = dark_fraction_stats(movie_hi)

                # Cross-pipeline difference at same LogMAR
                # (If this is huge for stabilized, that’s a red flag.)
                minT = min(movie_lo.shape[0], movie_hi.shape[0])
                diff = np.abs(movie_lo[:minT] - movie_hi[:minT])
                diff_mean = float(diff.mean())
                diff_max = float(diff.max())

                print(f"[cond={cond:10s} ori={ori:3d} lm={lm:+.2f}] "
                      f"flicker_rms lo={lo_f:.6f} hi={hi_f:.6f} | "
                      f"mean_flicker lo={lo_m:.6f} hi={hi_m:.6f} | "
                      f"dark_std lo={lo_dark['dark_frac_std']:.6f} hi={hi_dark['dark_frac_std']:.6f} | "
                      f"|lo-hi| mean={diff_mean:.6f} max={diff_max:.6f}")

                if args.save_movies:
                    base = f"trace{i}_T{T}_cond{cond}_ori{ori}_lm{lm:+.2f}"
                    out_lo = os.path.join(figures_dir, f"{base}_lores.mp4")
                    out_hi = os.path.join(figures_dir, f"{base}_hires.mp4")
                    save_stimulus_mp4(out_lo, movie_lo, fps=60, trim_first_frame=False, vmin=0.0, vmax=1.0)
                    save_stimulus_mp4(out_hi, movie_hi, fps=60, trim_first_frame=False, vmin=0.0, vmax=1.0)

                    if args.save_diff_movie:
                        # Visualize where they differ; autoscale to 99th percentile for visibility
                        vmax = float(np.percentile(diff, 99.0)) if diff.size else 1.0
                        out_df = os.path.join(figures_dir, f"{base}_absdiff.mp4")
                        save_stimulus_mp4(out_df, diff, fps=60, trim_first_frame=False, vmin=0.0, vmax=max(vmax, 1e-6))

    print("\nDone.")


if __name__ == "__main__":
    main()