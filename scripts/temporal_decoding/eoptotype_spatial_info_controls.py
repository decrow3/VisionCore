"""E-optotype spatial-information controls (real FEM vs null).

Goal
----
This script reuses the repo's existing spatial-information pipeline
(`scripts/spatial_info.py`) on the Tumbling-E stimulus used by the temporal
orientation-decoding analyses.

It directly addresses the interpretational tension:
  - Decoders: stabilized > real at near-threshold LogMAR (time_mean features)
  - Prior work: FEM can increase spatial/Fisher-style information vs null

Here we compute *spatial SSI* (single-spike information over spatial bins)
from the model's spatial readout maps under:
  - real eye trace (FEM)
  - null trace (same mean position, no motion)

Inputs are taken from the temporal-decoding cache:
  - scripts/temporal_decoding/data/eye_traces.npz

Outputs
-------
Writes an .npz under scripts/temporal_decoding/data/results/ with:
  - ispike_t_real/null: population bits/spike vs time
  - irate_t_real/null: population bits/sec vs time
  - cum_bits_real/null: cumulative bits vs time (integral of bits/sec)

Notes
-----
- This is *not* a behavioral decoder; it is a representational diagnostic.
- Spatial SSI depends on having spatial rate maps (T, N, H, W) from the readout.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


EYE_TRACES_PATH = Path("scripts/temporal_decoding/data/eye_traces.npz")
RESULTS_DIR = Path("scripts/temporal_decoding/data/results")


def _parse_int_list(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def _format_logmar_for_fname(logmar: float) -> str:
    # no leading '+'
    return f"{float(logmar):.2f}".replace("+", "")


def _pick_device(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass(frozen=True)
class TraceSet:
    traces: np.ndarray  # (M, T_max, 2) float32, NaN padded
    durations: np.ndarray  # (M,) int


def load_eye_traces(path: Path = EYE_TRACES_PATH) -> TraceSet:
    d = np.load(path, allow_pickle=True)
    traces = np.asarray(d["traces"], dtype=np.float32)
    durations = np.asarray(d["durations"], dtype=int)
    if traces.ndim != 3 or traces.shape[-1] != 2:
        raise ValueError(f"Expected traces shaped (M,T,2), got {traces.shape}")
    return TraceSet(traces=traces, durations=durations)


def select_trace_indices(durations: np.ndarray, trace_idxs: list[int], n_traces: int, min_T: int) -> list[int]:
    M = int(durations.shape[0])
    if trace_idxs:
        for i in trace_idxs:
            if i < 0 or i >= M:
                raise ValueError(f"trace_idx out of range: {i} (0..{M-1})")
        keep = [int(i) for i in trace_idxs if int(durations[int(i)]) >= int(min_T)]
        if not keep:
            raise ValueError("No selected trace_idxs satisfy min_T")
        return keep

    ok = np.where(durations >= int(min_T))[0]
    if ok.size == 0:
        raise ValueError(f"No traces with duration >= {min_T}")
    n = int(min(int(n_traces), int(ok.size)))
    return [int(i) for i in ok[:n]]


def crop_counterfactual_to_T(eye_stim: torch.Tensor, T: int) -> torch.Tensor:
    """Crop lag-embedded stimulus to exactly T frames.

    `make_counterfactual_stim()` returns (T+1, 1, n_lags, H, W) when fed an eye trace of length T.
    For stable alignment, we drop the first frame when possible.
    """
    T0 = int(eye_stim.shape[0])
    if T0 < T:
        raise ValueError(f"eye_stim has only {T0} frames, but T={T} requested")
    if T0 >= T + 1:
        return eye_stim[1 : 1 + T]
    return eye_stim[:T]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logmar", type=float, default=-0.20)
    parser.add_argument("--orientation", type=int, default=0)
    parser.add_argument("--n_traces", type=int, default=10, help="Number of traces to average over (ignored if --trace_idxs is set)")
    parser.add_argument("--trace_idxs", type=str, default="", help="Comma-separated explicit trace indices")
    parser.add_argument("--min_T", type=int, default=60, help="Minimum trace duration to include")
    parser.add_argument("--max_T", type=int, default=120, help="Cap each trace to this many frames")
    parser.add_argument("--n_lags", type=int, default=32)
    parser.add_argument("--out_size", type=int, default=101, help="Retinal crop size (square)")
    parser.add_argument("--ppd", type=float, default=37.50476617)
    parser.add_argument("--stim_scale", type=float, default=1.0)
    parser.add_argument(
        "--hires_threshold",
        type=float,
        default=0.35,
        help="Use the hi-res world→retina stimulus path when logmar < this threshold (matches dual-regime decoding caches)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mode", type=str, default="standard", help="Model loading mode (passed through)")
    args = parser.parse_args()

    device = _pick_device(args.device)
    print(f"Device: {device}")

    traceset = load_eye_traces()
    trace_idxs = _parse_int_list(args.trace_idxs)
    keep = select_trace_indices(traceset.durations, trace_idxs, args.n_traces, args.min_T)
    print(f"Using {len(keep)} trace(s): {keep[:10]}{'...' if len(keep) > 10 else ''}")

    # Load model/readout via the canonical temporal-decoding helper.
    from scripts.temporal_decoding.run_analysis import load_model_and_readout
    model, readout, _outputs = load_model_and_readout(mode=args.mode, device=device)

    # Spatial-info machinery (SSI + rate maps).
    from scripts.spatial_info import spatial_ssi_population
    from scripts.spatial_info import compute_rate_map_batched

    dt = 1.0 / 120.0
    out_hw = (int(args.out_size), int(args.out_size))
    use_hires = float(args.logmar) < float(args.hires_threshold)
    if use_hires:
        print(f"Using hi-res stimulus path (logmar={args.logmar:+.2f} < {args.hires_threshold:.2f})")
        from scripts.temporal_decoding.stimulus_hires import hires_counterfactual_stim
    else:
        from scripts.temporal_decoding.stimulus import e_optotype_stack
        from scripts.spatial_info import make_counterfactual_stim

    max_T = int(args.max_T)

    # Accumulators (ragged -> NaN padded)
    ispike_real_list: list[np.ndarray] = []
    ispike_null_list: list[np.ndarray] = []
    irate_real_list: list[np.ndarray] = []
    irate_null_list: list[np.ndarray] = []
    cum_bits_real_list: list[np.ndarray] = []
    cum_bits_null_list: list[np.ndarray] = []

    for idx in keep:
        dur = int(traceset.durations[idx])
        T = int(min(dur, max_T))
        trace = np.asarray(traceset.traces[idx, :T], dtype=np.float32)

        # Ensure finite (some traces NaN padded earlier than durations in rare cases)
        finite = np.isfinite(trace[:, 0]) & np.isfinite(trace[:, 1])
        if not np.all(finite):
            first_bad = int(np.where(~finite)[0][0])
            T = int(min(T, first_bad))
            trace = trace[:T]
        if T < int(args.min_T):
            continue

        if use_hires:
            # Hi-res pipeline already produces lag-embedded retinal stim in the correct
            # numeric convention (0..1), matching temporal-decoding caches.
            eye_stim = hires_counterfactual_stim(
                orientation_deg=int(args.orientation),
                logmar=float(args.logmar),
                eyepos=trace,
                condition='real',
                n_lags=int(args.n_lags),
                retina_size=out_hw,
                device=device,
            )
            eye_stim_null = hires_counterfactual_stim(
                orientation_deg=int(args.orientation),
                logmar=float(args.logmar),
                eyepos=trace,
                condition='stabilized',
                n_lags=int(args.n_lags),
                retina_size=out_hw,
                device=device,
            )

            eye_stim = crop_counterfactual_to_T(eye_stim, T)
            eye_stim_null = crop_counterfactual_to_T(eye_stim_null, T)

            stim_real = eye_stim
            stim_null = eye_stim_null
        else:
            # Low-res pipeline: build world stack on the 37.5-ppd canvas.
            full_stack = e_optotype_stack(
                orientation_deg=int(args.orientation),
                logmar=float(args.logmar),
                n_frames=T + int(args.n_lags) + 1,
            )

            eyepos = torch.from_numpy(trace).float()
            null_eyepos = torch.zeros_like(eyepos) + eyepos.mean(dim=0, keepdim=True)

            eye_stim = make_counterfactual_stim(
                full_stack,
                eyepos,
                ppd=float(args.ppd),
                scale_factor=float(args.stim_scale),
                n_lags=int(args.n_lags),
                out_size=out_hw,
            )
            eye_stim_null = make_counterfactual_stim(
                full_stack,
                null_eyepos,
                ppd=float(args.ppd),
                scale_factor=float(args.stim_scale),
                n_lags=int(args.n_lags),
                out_size=out_hw,
            )

            eye_stim = crop_counterfactual_to_T(eye_stim, T)
            eye_stim_null = crop_counterfactual_to_T(eye_stim_null, T)

            # Match temporal-decoding normalization: uint8-ish [0,127] → [0,1]
            stim_real = eye_stim / 127.0
            stim_null = eye_stim_null / 127.0

        y_real = compute_rate_map_batched(model, readout, stim_real, batch_size=16)  # (T, N, H, W)
        y_null = compute_rate_map_batched(model, readout, stim_null, batch_size=16)

        # SSI
        ispike_t_real, irate_t_real, _I_tn_real = spatial_ssi_population(y_real, dt=dt)
        ispike_t_null, irate_t_null, _I_tn_null = spatial_ssi_population(y_null, dt=dt)

        cum_bits_real = torch.cumsum(irate_t_real, dim=0) * dt
        cum_bits_null = torch.cumsum(irate_t_null, dim=0) * dt

        ispike_real_list.append(ispike_t_real.detach().cpu().numpy())
        ispike_null_list.append(ispike_t_null.detach().cpu().numpy())
        irate_real_list.append(irate_t_real.detach().cpu().numpy())
        irate_null_list.append(irate_t_null.detach().cpu().numpy())
        cum_bits_real_list.append(cum_bits_real.detach().cpu().numpy())
        cum_bits_null_list.append(cum_bits_null.detach().cpu().numpy())

        print(
            f"trace={idx:3d} T={T:3d} | "
            f"cum_bits real={float(cum_bits_real[-1]):.3f} null={float(cum_bits_null[-1]):.3f} | "
            f"mean ispike real={float(ispike_t_real.mean()):.4f} null={float(ispike_t_null.mean()):.4f}"
        )

    if not ispike_real_list:
        raise SystemExit("No traces produced outputs (check --min_T/--max_T)")

    # Ragged -> NaN padded arrays
    def pad_to_max(xs: list[np.ndarray]) -> np.ndarray:
        L = max(int(x.shape[0]) for x in xs)
        out = np.full((len(xs), L), np.nan, dtype=np.float32)
        for i, x in enumerate(xs):
            out[i, : x.shape[0]] = x.astype(np.float32)
        return out

    ispike_real = pad_to_max(ispike_real_list)
    ispike_null = pad_to_max(ispike_null_list)
    irate_real = pad_to_max(irate_real_list)
    irate_null = pad_to_max(irate_null_list)
    cum_bits_real = pad_to_max(cum_bits_real_list)
    cum_bits_null = pad_to_max(cum_bits_null_list)

    # Aggregate curves (nanmean over traces)
    ispike_real_mu = np.nanmean(ispike_real, axis=0)
    ispike_null_mu = np.nanmean(ispike_null, axis=0)
    irate_real_mu = np.nanmean(irate_real, axis=0)
    irate_null_mu = np.nanmean(irate_null, axis=0)
    cum_bits_real_mu = np.nanmean(cum_bits_real, axis=0)
    cum_bits_null_mu = np.nanmean(cum_bits_null, axis=0)

    # Print one-line summary at final common frame
    last = int(
        np.nanmax(
            [
                np.where(np.isfinite(cum_bits_real_mu))[0].max(),
                np.where(np.isfinite(cum_bits_null_mu))[0].max(),
            ]
        )
    )
    print("\n=== Summary (nanmean over traces) ===")
    print(f"logmar={args.logmar:+.2f} ori={args.orientation} | frames={last+1}")
    print(
        f"cum_bits: real={cum_bits_real_mu[last]:.3f}  null={cum_bits_null_mu[last]:.3f}  "
        f"(Δ={cum_bits_real_mu[last]-cum_bits_null_mu[last]:+.3f})"
    )
    print(
        f"mean ispike: real={np.nanmean(ispike_real_mu):.4f}  null={np.nanmean(ispike_null_mu):.4f}  "
        f"(Δ={np.nanmean(ispike_real_mu)-np.nanmean(ispike_null_mu):+.4f})"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / (
        f"eoptotype_spatial_ssi_lm{_format_logmar_for_fname(args.logmar)}_ori{int(args.orientation)}_T{int(args.max_T)}_n{len(ispike_real_list)}.npz"
    )

    np.savez(
        out_path,
        logmar=float(args.logmar),
        orientation=int(args.orientation),
        trace_indices=np.asarray(keep, dtype=int),
        dt=float(dt),
        ispike_t_real=ispike_real,
        ispike_t_null=ispike_null,
        irate_t_real=irate_real,
        irate_t_null=irate_null,
        cum_bits_real=cum_bits_real,
        cum_bits_null=cum_bits_null,
        ispike_t_real_mean=ispike_real_mu,
        ispike_t_null_mean=ispike_null_mu,
        irate_t_real_mean=irate_real_mu,
        irate_t_null_mean=irate_null_mu,
        cum_bits_real_mean=cum_bits_real_mu,
        cum_bits_null_mean=cum_bits_null_mu,
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
