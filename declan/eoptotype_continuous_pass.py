"""Priority 3: continuous forward pass for E-optotype decoding.

This script feeds each trial as one continuous retinal movie through the model
instead of evaluating overlapping lag windows independently. It reuses the
existing D1 and D3 decoders on the resulting continuous response trajectories.

Initial scope:
- hi-res E-optotype path (intended for the hyperacuity regime)
- D1 retest for real vs stabilized
- D3 retest on continuous trajectories

Usage
-----
/home/declan/VisionCore/.venv/bin/python declan/eoptotype_continuous_pass.py \
  --logmar -0.40 \
  --windows 1,24,60 \
  --n_traces 32
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / 'declan'
TEMPORAL_DIR = ROOT / 'scripts' / 'temporal_decoding'
DATA_DIR = TEMPORAL_DIR / 'data'
EYE_TRACES_PATH = DATA_DIR / 'eye_traces.npz'
RESULTS_DIR = SCRIPT_DIR / 'continuous_pass_results'
PKL_PATH = ROOT / 'scripts' / 'mcfarland_outputs_mono.pkl'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEMPORAL_DIR))

from scripts.temporal_decoding.eoptotype_decoder_controls import (  # noqa: E402
    decode_d1_time_mean,
    decode_d3_supervised_mean_trajectory,
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


DEFAULT_WINDOWS = (1, 24, 60)
DEFAULT_ORIENTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class Curve:
    windows: list[int]
    mean: np.ndarray
    std: np.ndarray


def _parse_csv_ints(s: str) -> list[int]:
    return [int(float(x)) for x in str(s).split(',') if str(x).strip()]


def _load_eye_traces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    traces = d['traces'].astype(np.float32)
    durations = d['durations'].astype(np.int32)
    return traces, durations


def _maybe_subsample(
    traces: np.ndarray,
    durations: np.ndarray,
    n_traces: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_traces is None or n_traces <= 0 or n_traces >= traces.shape[0]:
        return traces, durations
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(traces.shape[0], size=int(n_traces), replace=False))
    return traces[idx], durations[idx]


def _load_model_and_readout(device: str | None = None):
    import dill

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from utils import get_model_and_dataset_configs
    from spatial_info import get_spatial_readout

    model, _ = get_model_and_dataset_configs(mode='standard')
    model.model.eval()
    if hasattr(model.model, 'convnet') and hasattr(model.model.convnet, 'use_checkpointing'):
        model.model.convnet.use_checkpointing = False
    model = model.to(device)

    with open(PKL_PATH, 'rb') as f:
        outputs = dill.load(f)

    readout = get_spatial_readout(model, outputs).to(device)
    readout.eval()
    return model, readout


def _resolve_eye_trace(eyepos: np.ndarray, condition: str) -> torch.Tensor:
    eye_t = torch.from_numpy(eyepos).float()
    if condition == 'real':
        return eye_t
    if condition == 'stabilized':
        mean = eye_t.mean(0, keepdim=True)
        return mean.expand_as(eye_t)
    raise ValueError(f'Unsupported condition: {condition}')


def build_hires_continuous_movie(
    orientation_deg: float,
    logmar: float,
    eyepos: np.ndarray,
    condition: str,
    prepad_frames: int,
    device: str,
) -> torch.Tensor:
    ep = _resolve_eye_trace(eyepos, condition)
    ep_padded = torch.cat([ep[:1].expand(prepad_frames, -1), ep], dim=0)

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

    with torch.no_grad():
        world_img = renderer(orientation_deg, logmar)
        world_gray = 127.0 * (1.0 - world_img)
        movie = retina(world_gray, ep_padded.to(device))[0, 0] / 127.0
    return movie.cpu()


def compute_continuous_trial_rates(
    model,
    readout,
    movie: torch.Tensor,
    spatial_collapse: str = 'max',
) -> np.ndarray:
    device = next(model.model.parameters()).device
    use_amp = device.type == 'cuda'
    x = movie.unsqueeze(0).unsqueeze(0).to(device)

    model.model.eval()
    readout.eval()

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        x_front = model.model.frontend(x)
        x_conv = model.model.convnet(x_front)
        x_recurrent = model.model.recurrent(x_conv)
        feats = x_recurrent[0].permute(1, 0, 2, 3).contiguous()
        y = readout(feats)
        rates = _collapse_spatial(model.model.activation(y), method=spatial_collapse).float().cpu().numpy()

    del x, x_front, x_conv, x_recurrent, feats, y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rates.astype(np.float32)


def compute_rates_by_condition(
    model,
    readout,
    logmar: float,
    traces: np.ndarray,
    durations: np.ndarray,
    orientations: list[int],
    conditions: list[str],
    prepad_frames: int,
    device: str,
) -> dict[str, dict[str, list[np.ndarray]]]:
    results: dict[str, dict[str, list[np.ndarray]]] = {}
    for condition in conditions:
        per_condition: dict[str, list[np.ndarray]] = {}
        print(f'\nCondition: {condition}')
        for orientation in orientations:
            sid = f'ori{orientation}'
            per_condition[sid] = []
            print(f'  Orientation {orientation}...', flush=True)
            for i in range(traces.shape[0]):
                T = int(durations[i])
                eyepos = traces[i, :T]
                movie = build_hires_continuous_movie(
                    orientation_deg=float(orientation),
                    logmar=float(logmar),
                    eyepos=eyepos,
                    condition=condition,
                    prepad_frames=int(prepad_frames),
                    device=device,
                )
                rates = compute_continuous_trial_rates(model, readout, movie, spatial_collapse='max')
                per_condition[sid].append(rates)
            lengths = np.asarray([r.shape[0] for r in per_condition[sid]], dtype=int)
            print(
                f'    lengths: min={lengths.min()} median={int(np.median(lengths))} '
                f'max={lengths.max()} mean={lengths.mean():.1f}',
                flush=True,
            )
        results[condition] = per_condition
    return results


def _compute_curve(fn, windows: list[int]) -> Curve:
    means: list[float] = []
    stds: list[float] = []
    for window in windows:
        mean_acc, std_acc, _ = fn(window)
        means.append(float(mean_acc))
        stds.append(float(std_acc))
    return Curve(windows=windows, mean=np.asarray(means, dtype=float), std=np.asarray(stds, dtype=float))


def make_figure(logmar: float, curves: dict[str, Curve]) -> Figure:
    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=160)
    for label, curve in curves.items():
        ax.errorbar(curve.windows, curve.mean, yerr=curve.std, marker='o', linewidth=2, capsize=3, label=label)
    ax.set_xlabel('Window W (output frames)')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.2, 1.02)
    ax.set_title(f'Continuous-pass decoding | LogMAR {logmar:+.2f}')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    return fig


def save_rates_cache(out_path: Path, rates_by_stim: dict[str, list[np.ndarray]]) -> None:
    all_rates = []
    all_lengths = []
    all_labels = []
    for sid in sorted(rates_by_stim.keys()):
        for r in rates_by_stim[sid]:
            all_rates.append(r)
            all_lengths.append(r.shape[0])
            all_labels.append(sid)
    t_max = max(all_lengths)
    n_units = all_rates[0].shape[1]
    rates_padded = np.full((len(all_rates), t_max, n_units), np.nan, dtype=np.float32)
    for i, r in enumerate(all_rates):
        rates_padded[i, :r.shape[0]] = r
    np.savez(
        out_path,
        rates=rates_padded,
        lengths=np.asarray(all_lengths, dtype=np.int32),
        labels=np.asarray(all_labels, dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logmar', type=float, default=-0.40)
    parser.add_argument('--windows', type=str, default=','.join(map(str, DEFAULT_WINDOWS)))
    parser.add_argument('--orientations', type=str, default='0,90,180,270')
    parser.add_argument('--conditions', type=str, default='real,stabilized')
    parser.add_argument('--eye_traces_path', type=str, default=str(EYE_TRACES_PATH))
    parser.add_argument('--n_traces', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepad_frames', type=int, default=32)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--skip_d3', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    windows = _parse_csv_ints(args.windows)
    orientations = _parse_csv_ints(args.orientations)
    conditions = [x.strip() for x in str(args.conditions).split(',') if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces, durations = _load_eye_traces(Path(args.eye_traces_path))
    traces, durations = _maybe_subsample(traces, durations, args.n_traces, args.seed)

    model, readout = _load_model_and_readout(device=args.device)
    device = next(model.model.parameters()).device.type

    print('Continuous pass E-optotype analysis')
    print(f'  logmar={args.logmar:+.2f}')
    print(f'  windows={windows}')
    print(f'  orientations={orientations}')
    print(f'  conditions={conditions}')
    print(f'  n_traces={traces.shape[0]}')
    print(f'  device={device}')

    rates_by_condition = compute_rates_by_condition(
        model=model,
        readout=readout,
        logmar=float(args.logmar),
        traces=traces,
        durations=durations,
        orientations=orientations,
        conditions=conditions,
        prepad_frames=int(args.prepad_frames),
        device=device,
    )

    curves: dict[str, Curve] = {}
    payload: dict[str, np.ndarray] = {
        'windows': np.asarray(windows, dtype=np.int32),
        'logmar': np.asarray([float(args.logmar)], dtype=float),
        'n_traces': np.asarray([int(traces.shape[0])], dtype=np.int32),
        'prepad_frames': np.asarray([int(args.prepad_frames)], dtype=np.int32),
    }

    for condition in conditions:
        rates_by_stim = rates_by_condition[condition]
        save_rates_cache(out_dir / f'continuous_rates_lm{args.logmar:+.2f}_{condition}.npz', rates_by_stim)

        curve_d1 = _compute_curve(
            lambda w, rates=rates_by_stim: decode_d1_time_mean(rates, w, n_splits=int(args.n_splits)),
            windows,
        )
        curves[f'D1 {condition}'] = curve_d1
        payload[f'D1_{condition}_mean'] = curve_d1.mean
        payload[f'D1_{condition}_std'] = curve_d1.std

        if not args.skip_d3:
            curve_d3 = _compute_curve(
                lambda w, rates=rates_by_stim: decode_d3_supervised_mean_trajectory(rates, w, n_splits=int(args.n_splits)),
                windows,
            )
            curves[f'D3 {condition}'] = curve_d3
            payload[f'D3_{condition}_mean'] = curve_d3.mean
            payload[f'D3_{condition}_std'] = curve_d3.std

    fig = make_figure(float(args.logmar), curves)
    stem = f'continuous_pass_lm{args.logmar:+.2f}_n{traces.shape[0]}'
    fig.savefig(out_dir / f'{stem}.png', bbox_inches='tight')
    plt.close(fig)
    np.savez(out_dir / f'{stem}.npz', **payload)

    print('\nSummary')
    for label, curve in curves.items():
        pairs = ', '.join(f'W={w}: {m:.3f}±{s:.3f}' for w, m, s in zip(curve.windows, curve.mean, curve.std, strict=True))
        print(f'  {label}: {pairs}')
    print(f'\nSaved figure: {out_dir / f"{stem}.png"}')
    print(f'Saved summary: {out_dir / f"{stem}.npz"}')


if __name__ == '__main__':
    main()