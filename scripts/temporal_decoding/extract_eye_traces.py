"""
Task 1.2: Eye Trace Library Extraction and Caching

Extracts all fixational eye traces from the dataset into a single cached .npz file
with metadata (session, trial, duration, RMS, path length, velocity RMS).

Usage:
    python extract_eye_traces.py [--output data/eye_traces.npz]
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_stack_multidataset import load_single_dataset

# Constants
MAX_T = 540           # max frames per trace (at 120 Hz)
MIN_DURATION = 60     # minimum valid trace length (frames)
MAX_ECCENTRICITY = 1.0  # degrees — filter to fixation periods only
FRAME_RATE = 120.0    # Hz


def compute_trace_kinematics(trace: np.ndarray, frame_rate: float = FRAME_RATE) -> dict:
    """
    Compute kinematic statistics for a single eye trace.

    Args:
        trace: (T, 2) array of eye positions in degrees (x, y). No NaN values.
        frame_rate: sampling rate in Hz

    Returns:
        dict with rms, path_length, velocity_rms
    """
    T = trace.shape[0]

    # RMS displacement from mean position
    mean_pos = trace.mean(axis=0)
    displacements = trace - mean_pos
    rms = float(np.sqrt(np.mean(displacements ** 2)))

    # Total path length in degrees
    if T > 1:
        diffs = np.diff(trace, axis=0)
        step_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        path_length = float(step_lengths.sum())
    else:
        path_length = 0.0

    # Velocity RMS in deg/s
    if T > 1:
        velocities = np.diff(trace, axis=0) * frame_rate  # (T-1, 2) deg/s
        speeds = np.sqrt((velocities ** 2).sum(axis=1))   # (T-1,)
        velocity_rms = float(np.sqrt(np.mean(speeds ** 2)))
    else:
        velocity_rms = 0.0

    return {
        'rms': rms,
        'path_length': path_length,
        'velocity_rms': velocity_rms,
    }


def extract_eye_traces(
    model,
    outputs,
    max_T: int = MAX_T,
    min_duration: int = MIN_DURATION,
    max_eccentricity: float = MAX_ECCENTRICITY,
) -> dict:
    """
    Extract fixational eye traces from all sessions.

    Args:
        model: loaded VisionCore model (with model.names)
        outputs: list of per-session output dicts (from mcfarland_outputs_mono.pkl)
        max_T: maximum trace length (NaN-padded beyond this)
        min_duration: minimum valid duration to include
        max_eccentricity: eccentricity threshold (degrees) for fixation filter

    Returns:
        dict with keys:
            traces: (N, max_T, 2) float32, NaN-padded
            durations: (N,) int
            sessions: (N,) str
            rms: (N,) float
            path_length: (N,) float
            velocity_rms: (N,) float
    """
    import torch

    sessions = [outputs[i]['sess'] for i in range(len(outputs))]

    all_traces = []
    all_durations = []
    all_sessions = []
    all_rms = []
    all_path_lengths = []
    all_velocity_rms = []

    for name in sessions:
        if name not in model.names:
            print(f"  Session {name} not in model, skipping")
            continue

        dataset_idx = model.names.index(name)

        try:
            train_data, val_data, _ = load_single_dataset(model, dataset_idx)

            # Get all fixrsvp trial indices
            inds = torch.concatenate([
                train_data.get_dataset_inds('fixrsvp'),
                val_data.get_dataset_inds('fixrsvp'),
            ], dim=0)

            dataset = train_data.shallow_copy()
            dataset.inds = inds
            inds_np = inds.detach().cpu().numpy()

            session_count = 0
            for dset_idx in np.unique(inds_np[:, 0]).astype(int):
                eyepos_all = dataset.dsets[dset_idx]['eyepos'][:].numpy()
                trial_inds_all = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()

                # Mark which samples belong to fixrsvp trials
                in_fixrsvp = np.zeros(len(trial_inds_all), dtype=bool)
                sample_idx = inds_np[inds_np[:, 0] == dset_idx, 1].astype(int)
                sample_idx = sample_idx[(sample_idx >= 0) & (sample_idx < len(in_fixrsvp))]
                in_fixrsvp[sample_idx] = True

                # Fixation filter: eccentricity < max_eccentricity
                eccentricity = np.hypot(eyepos_all[:, 0], eyepos_all[:, 1])
                fixation_mask = (eccentricity < max_eccentricity) & in_fixrsvp

                trials = np.unique(trial_inds_all[in_fixrsvp & ~np.isnan(trial_inds_all)])

                for t in trials:
                    ix = (trial_inds_all == t) & fixation_mask
                    eyepos = eyepos_all[ix]

                    if len(eyepos) < min_duration:
                        continue

                    # Clamp to max_T
                    eyepos = eyepos[:max_T]
                    T_valid = len(eyepos)

                    # NaN-padded trace
                    trace = np.full((max_T, 2), np.nan, dtype=np.float32)
                    trace[:T_valid] = eyepos

                    kinematics = compute_trace_kinematics(eyepos)

                    all_traces.append(trace)
                    all_durations.append(T_valid)
                    all_sessions.append(name)
                    all_rms.append(kinematics['rms'])
                    all_path_lengths.append(kinematics['path_length'])
                    all_velocity_rms.append(kinematics['velocity_rms'])
                    session_count += 1

            print(f"  {name}: {session_count} traces")

        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            continue

    if not all_traces:
        raise RuntimeError("No eye traces extracted. Check dataset loading.")

    return {
        'traces': np.stack(all_traces, axis=0),          # (N, max_T, 2)
        'durations': np.array(all_durations, dtype=np.int32),
        'sessions': np.array(all_sessions),
        'rms': np.array(all_rms, dtype=np.float32),
        'path_length': np.array(all_path_lengths, dtype=np.float32),
        'velocity_rms': np.array(all_velocity_rms, dtype=np.float32),
    }


def save_eye_traces(data: dict, path: str) -> None:
    """Save extracted eye traces to .npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"Saved {data['traces'].shape[0]} traces to {path}")


def load_eye_traces(path: str) -> dict:
    """Load eye traces from .npz file."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def print_summary(data: dict) -> None:
    """Print summary statistics for the extracted eye traces."""
    N = data['traces'].shape[0]
    durations = data['durations']
    rms = data['rms']
    vlrms = data['velocity_rms']
    sessions = data['sessions']

    print(f"\n=== Eye Trace Summary ===")
    print(f"Total traces: {N}")
    print(f"Unique sessions: {len(np.unique(sessions))}")
    print(f"Duration (frames at 120 Hz):")
    print(f"  min={durations.min()}, median={np.median(durations):.0f}, "
          f"max={durations.max()}, mean={durations.mean():.1f}")
    print(f"Duration (ms): min={durations.min()/120*1000:.0f}, "
          f"median={np.median(durations)/120*1000:.0f}")
    print(f"RMS displacement (deg):")
    print(f"  min={rms.min():.4f}, median={np.median(rms):.4f}, max={rms.max():.4f}")
    print(f"Velocity RMS (deg/s):")
    print(f"  min={vlrms.min():.2f}, median={np.median(vlrms):.2f}, max={vlrms.max():.2f}")
    print(f"========================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract eye traces from dataset')
    parser.add_argument('--output', default=os.path.join(
        os.path.dirname(__file__), 'data', 'eye_traces.npz'))
    parser.add_argument('--min_duration', type=int, default=MIN_DURATION)
    parser.add_argument('--max_eccentricity', type=float, default=MAX_ECCENTRICITY)
    parser.add_argument('--mode', default='standard',
                        help='Model loading mode (standard or frozencore)')
    args = parser.parse_args()

    import dill
    from types import SimpleNamespace
    from models.config_loader import load_dataset_configs

    dataset_configs_path = "experiments/dataset_configs/multi_basic_240_all.yaml"
    print(f"Loading dataset configs from {dataset_configs_path}...")
    dataset_configs = load_dataset_configs(dataset_configs_path)
    names = [cfg['session'] for cfg in dataset_configs]
    model = SimpleNamespace(names=names, dataset_configs=dataset_configs)

    pkl_path = os.path.join(os.path.dirname(__file__), '..', 'mcfarland_outputs_mono.pkl')
    print(f"Loading outputs from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        outputs = dill.load(f)

    print(f"Extracting eye traces (min_duration={args.min_duration}, "
          f"max_eccentricity={args.max_eccentricity} deg)...")
    data = extract_eye_traces(
        model, outputs,
        min_duration=args.min_duration,
        max_eccentricity=args.max_eccentricity,
    )

    print_summary(data)
    save_eye_traces(data, args.output)
