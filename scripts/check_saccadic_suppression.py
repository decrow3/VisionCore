"""
Check for emergent saccadic suppression in a digital twin V1 model.
Generates Saccade-Triggered Average (STA) PSTHs for real vs predicted spikes.
"""
# filepath: /home/declan/VisionCore/scripts/check_saccadic_suppression.py
# This script loads a trained V1 model and a free-viewing dataset, detects saccades from eye position data,
# and computes the Saccade-Triggered Average (STA) for both real and predicted neural responses. It then plots the population average and a few single-unit examples to visually assess whether the model
import os
import sys
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

data_root = os.environ.get("DATAYATESV1_ROOT", None)
if data_root:
    sys.path.insert(0, data_root)
else:
    candidate = Path(__file__).resolve().parent.parent.parent / "DataYatesV1"
    if candidate.exists():
        sys.path.insert(0, str(candidate))

# Adjust these imports based on your exact working directory
from utils import get_model_and_dataset_configs
from eval.eval_stack_multidataset import load_single_dataset
from spatial_info import (
    make_integrated_counterfactual_stim,
    embed_time_lags,
)

def get_saccade_onsets(eyepos, fps=100.0, vel_threshold_deg_sec=20.0, min_isi_ms=50, return_debug=False):
    """
    Detect saccade onsets based on eye velocity.
    eyepos: [T, 2] array of eye positions in degrees
    """
    # Calculate instantaneous velocity (degrees per second)
    vel = np.linalg.norm(np.diff(eyepos, axis=0), axis=1) * fps
    vel = np.concatenate([[0], vel]) # Pad first frame to maintain length T
    
    # Threshold to find high-velocity frames
    is_fast = vel > vel_threshold_deg_sec
    
    # Find onsets (fast frame preceded by a slow frame) without circular wrap
    onsets_mask = is_fast.copy()
    onsets_mask[1:] = is_fast[1:] & (~is_fast[:-1])
    raw_onsets = np.where(onsets_mask)[0]
    
    # Filter out onsets that are too close together (e.g., multi-peak saccades)
    min_isi_frames = int((min_isi_ms / 1000.0) * fps)
    valid_onsets = []
    for t in raw_onsets:
        if len(valid_onsets) == 0 or (t - valid_onsets[-1]) >= min_isi_frames:
            valid_onsets.append(t)

    valid_onsets = np.array(valid_onsets)

    if return_debug:
        debug = {
            'n_samples': int(len(eyepos)),
            'n_fast_frames': int(is_fast.sum()),
            'n_raw_onsets': int(len(raw_onsets)),
            'n_after_isi': int(len(valid_onsets)),
            'velocity_p50': float(np.percentile(vel, 50)),
            'velocity_p90': float(np.percentile(vel, 90)),
            'velocity_p95': float(np.percentile(vel, 95)),
            'velocity_p99': float(np.percentile(vel, 99)),
            'threshold': float(vel_threshold_deg_sec),
        }
        return valid_onsets, debug

    return valid_onsets

def compute_sta_psth(robs, pred, onsets, fps=100.0, win_pre_ms=100, win_post_ms=250):
    """
    Compute the Saccade-Triggered Average for real and predicted spikes.
    """
    win_pre_frames = int((win_pre_ms / 1000.0) * fps)
    win_post_frames = int((win_post_ms / 1000.0) * fps)
    
    psth_robs, psth_pred = [], []
    
    kept_onsets = []
    for t in onsets:
        # Ensure we don't go out of bounds
        if t - win_pre_frames >= 0 and t + win_post_frames < len(robs):
            psth_robs.append(robs[t - win_pre_frames : t + win_post_frames])
            psth_pred.append(pred[t - win_pre_frames : t + win_post_frames])
            kept_onsets.append(t)
            
    # Average across all valid saccades -> Shape: [Time, Neurons]
    if len(psth_robs) == 0:
        raise ValueError("No valid saccades found within analysis window. Try adjusting thresholds or window size.")

    sta_robs = np.stack(psth_robs).mean(axis=0)
    sta_pred = np.stack(psth_pred).mean(axis=0)
    
    time_axis = np.linspace(-win_pre_ms, win_post_ms, win_pre_frames + win_post_frames)
    
    return time_axis, sta_robs, sta_pred, len(psth_robs), np.array(kept_onsets)


def run_model_from_stim(model, stim, dataset_idx, device, behavior=None, batch_size=256):
    """
    Run model predictions from a prebuilt stimulus tensor.
    stim: [T, C, n_lags, H, W]
    behavior: optional [T, B]
    """
    preds = []
    with torch.no_grad():
        for i0 in range(0, stim.shape[0], batch_size):
            x = stim[i0:i0 + batch_size].to(device)
            b = None
            if behavior is not None:
                b = behavior[i0:i0 + batch_size].to(device)

            if getattr(model, 'is_modulator_only', False):
                out = model.model(None, dataset_idx, b)
            elif hasattr(model.model, 'spike_history'):
                out = model.model(x, dataset_idx, b, None)
            else:
                out = model.model(x, dataset_idx, b)

            if getattr(model, 'log_input', False):
                out = torch.exp(out)

            preds.append(out.detach().cpu())

    return torch.cat(preds, dim=0).numpy()


def maybe_normalize_stim(stim):
    """Normalize uint8-like stimulus to training scale if needed."""
    stim_min = float(stim.min())
    stim_max = float(stim.max())
    if stim_max > 2.0:
        return (stim - 127.0) / 255.0
    return stim


def main():
    # 1. Load Model and Data (Using your standard pipeline)
    print("Loading model and dataset configs...")
    model, dataset_configs = get_model_and_dataset_configs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 2. Pick a free-viewing-capable dataset and load it with the evaluation utility
    if isinstance(dataset_configs, dict):
        cfg_iter = list(dataset_configs.items())
    else:
        cfg_iter = list(enumerate(dataset_configs))

    dataset_idx = None
    for idx, cfg in cfg_iter:
        types = cfg.get('types', []) if isinstance(cfg, dict) else []
        if 'backimage' in types:
            dataset_idx = int(idx)
            break
    if dataset_idx is None:
        raise ValueError("No dataset with `backimage` found in dataset configs. Natural-image analysis requires backimage.")

    print(f"Loading dataset index {dataset_idx}...")
    train_data, val_data, _ = load_single_dataset(model, dataset_idx)

    # 3. Build an evaluation subset (require backimage for natural/free-viewing)
    if hasattr(val_data, 'get_dataset_inds'):
        if any(d.metadata.get('name') == 'backimage' for d in val_data.dsets):
            stim_inds = val_data.get_dataset_inds('backimage')
            stim_type = 'backimage'
        else:
            raise ValueError("Could not find `backimage` in validation dataset.")
    else:
        raise ValueError("Validation dataset does not support `get_dataset_inds`.")

    print(f"Using stimulus subset: {stim_type} ({len(stim_inds)} samples)")

    dataset = val_data.shallow_copy()
    dataset.inds = stim_inds
    dset_idx = int(dataset.inds[:, 0].unique().item())
    sample_inds = dataset.inds[:, 1].detach().cpu().numpy().astype(int)
    trial_inds_all = dataset.dsets[dset_idx].covariates['trial_inds'].detach().cpu().numpy()
    sample_trial_inds = trial_inds_all[sample_inds]

    fps = 120.0

    # 4. Collect robs/eyepos/behavior and legacy strobed predictions
    print("Collecting responses, eye traces, and legacy strobed predictions...")
    batch_size = 512
    robs_chunks = []
    eyepos_chunks = []
    pred_strobed_chunks = []
    behavior_chunks = []
    has_behavior = None
    n_lags = None
    out_size = None

    with torch.no_grad():
        for i0 in range(0, len(dataset), batch_size):
            batch = dataset[i0:i0 + batch_size]
            batch_device = {k: v.to(device) for k, v in batch.items()}
            if n_lags is None and 'stim' in batch:
                n_lags = int(batch['stim'].shape[2])
                out_size = (int(batch['stim'].shape[3]), int(batch['stim'].shape[4]))

            if getattr(model, 'is_modulator_only', False):
                output = model.model(None, dataset_idx, batch_device.get('behavior'))
            elif hasattr(model.model, 'spike_history'):
                output = model.model(batch_device['stim'], dataset_idx, batch_device.get('behavior', None), batch_device.get('history', None))
            else:
                output = model.model(batch_device['stim'], dataset_idx, batch_device.get('behavior'))

            if getattr(model, 'log_input', False):
                output = torch.exp(output)

            robs_chunks.append(batch['robs'].detach().cpu())
            pred_strobed_chunks.append(output.detach().cpu())

            if 'eyepos' in batch:
                eyepos_chunks.append(batch['eyepos'].detach().cpu())
            else:
                raise ValueError("Batch is missing `eyepos`; ensure keys_lags includes eyepos in dataset config.")

            if has_behavior is None:
                has_behavior = ('behavior' in batch)
            if has_behavior and 'behavior' in batch:
                behavior_chunks.append(batch['behavior'].detach().cpu())

    robs = torch.cat(robs_chunks, dim=0).numpy()
    pred_strobed = torch.cat(pred_strobed_chunks, dim=0).numpy()
    eyepos = torch.cat(eyepos_chunks, dim=0).numpy()
    behavior = torch.cat(behavior_chunks, dim=0) if has_behavior and len(behavior_chunks) > 0 else None

    # Ensure shape [T, 2] for eye position
    if eyepos.ndim == 3 and eyepos.shape[1] == 1:
        eyepos = eyepos[:, 0]
    elif eyepos.ndim > 2:
        eyepos = eyepos.reshape(eyepos.shape[0], -1)[:, :2]

    eyepos = np.asarray(eyepos, dtype=np.float32)
    if eyepos.ndim != 2 or eyepos.shape[1] < 2:
        raise ValueError(f"Unexpected eyepos shape after reshape: {eyepos.shape}")
    eyepos = eyepos[:, :2]

    valid_eye = np.isfinite(eyepos).all(axis=1)
    robs = robs[valid_eye]
    pred_strobed = pred_strobed[valid_eye]
    eyepos = eyepos[valid_eye]
    sample_trial_inds = sample_trial_inds[valid_eye]
    if behavior is not None:
        behavior_mask = torch.from_numpy(valid_eye)
        behavior = behavior[behavior_mask]

    if len(eyepos) == 0:
        raise ValueError("No valid eye position samples after finite-value filtering.")

    if n_lags is None or out_size is None:
        raise ValueError("Could not infer stimulus lag/spatial dimensions from dataset batch.")

    # 5. Build integrated counterfactual stimulus from true per-trial BackImage backgrounds
    print("Generating integrated predictions from true BackImage trial backgrounds...")
    ppd = 37.50476617

    try:
        exp_mod = importlib.import_module('DataYatesV1.exp')
        BackImageTrial = getattr(exp_mod, 'BackImageTrial')
        get_trial_protocols = getattr(exp_mod, 'get_trial_protocols')
    except Exception as exc:
        raise ImportError("Could not import BackImage utilities from DataYatesV1.exp") from exc

    sess_obj = dataset.dsets[dset_idx].metadata['sess']
    exp = sess_obj.exp
    protocols = get_trial_protocols(exp)
    backimage_trial_inds = np.where(np.array(protocols) == 'BackImage')[0]
    trial_obj_map = {int(iT): BackImageTrial(exp['D'][iT], exp['S']) for iT in backimage_trial_inds}

    pred_integrated = np.full_like(pred_strobed, np.nan)
    total_samples = int(len(sample_trial_inds))
    segments_processed = 0
    segments_skipped = 0

    start = 0
    while start < total_samples:
        end = start + 1
        tid = sample_trial_inds[start]
        while end < total_samples and sample_trial_inds[end] == tid:
            end += 1

        if not np.isfinite(tid):
            segments_skipped += 1
            start = end
            continue

        tid_int = int(tid)
        if tid_int not in trial_obj_map:
            segments_skipped += 1
            start = end
            continue

        seg_len = end - start

        try:
            image = trial_obj_map[tid_int].get_image()
            if image.ndim == 3:
                if image.shape[-1] in (3, 4):
                    image = image.mean(axis=-1)
                elif image.shape[0] in (3, 4):
                    image = image.mean(axis=0)
                else:
                    image = image.mean(axis=-1)

            full_stack_trial = np.repeat(image[None, :, :], seg_len + n_lags, axis=0)
            full_stack_t = torch.from_numpy(np.ascontiguousarray(full_stack_trial)).float().unsqueeze(1)
            eyepos_t = torch.from_numpy(np.ascontiguousarray(eyepos[start:end, :2])).float()

            integrated_movie = make_integrated_counterfactual_stim(
                full_stack_t,
                eyepos_t,
                ppd=ppd,
                n_lags=n_lags,
                sub_frames=10,
            )
            integrated_stim = embed_time_lags(integrated_movie, n_lags=n_lags)
            integrated_stim = maybe_normalize_stim(integrated_stim)

            behavior_seg = behavior[start:end] if behavior is not None else None
            pred_seg = run_model_from_stim(
                model,
                integrated_stim,
                dataset_idx,
                device,
                behavior=behavior_seg,
                batch_size=256,
            )

            pred_integrated[start:end] = pred_seg
            segments_processed += 1
        except Exception:
            segments_skipped += 1

        start = end

    good_pred = np.isfinite(pred_integrated).all(axis=1)
    robs = robs[good_pred]
    pred_strobed = pred_strobed[good_pred]
    pred_integrated = pred_integrated[good_pred]
    eyepos = eyepos[good_pred]
    if behavior is not None:
        behavior = behavior[torch.from_numpy(good_pred)]

    print(
        f"Integrated prediction summary: total samples={total_samples}, "
        f"segments processed={segments_processed}, segments skipped={segments_skipped}, "
        f"final kept samples={int(good_pred.sum())}"
    )
    
    # 6. Detect Saccades
    print("Detecting saccades...")
    onsets, sac_debug = get_saccade_onsets(eyepos, fps=fps, return_debug=True)
    print(f"Found {len(onsets)} saccades.")
    print(
        "Velocity stats (deg/s): "
        f"p50={sac_debug['velocity_p50']:.2f}, "
        f"p90={sac_debug['velocity_p90']:.2f}, "
        f"p95={sac_debug['velocity_p95']:.2f}, "
        f"p99={sac_debug['velocity_p99']:.2f}, "
        f"threshold={sac_debug['threshold']:.2f}"
    )
    
    # 7. Compute PSTH
    time_axis, sta_robs, sta_strobed, n_saccades, kept_onsets = compute_sta_psth(robs, pred_strobed, onsets, fps=fps)
    _, _, sta_integrated, _, _ = compute_sta_psth(robs, pred_integrated, onsets, fps=fps)
    print(f"Saccades kept after PSTH window bounds: {n_saccades}/{len(onsets)}")
    
    # 8. Plot Results
    print("Plotting...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot population average
    axes[0].plot(time_axis, sta_robs.mean(axis=1), 'k-', lw=2, label='Real (In Vivo)')
    axes[0].plot(time_axis, sta_strobed.mean(axis=1), 'r--', lw=2, label='Predicted Strobed')
    axes[0].plot(time_axis, sta_integrated.mean(axis=1), 'b-.', lw=2, label='Predicted Integrated')
    axes[0].axvline(0, color='gray', linestyle=':', alpha=0.7)
    axes[0].set_title(f"Population Saccade-Triggered Average (N={n_saccades} saccades)")
    axes[0].set_ylabel("Mean Firing Rate")
    axes[0].legend()
    
    # Plot a few random single units
    n_units_to_plot = min(5, sta_robs.shape[1])
    units_to_plot = np.random.choice(sta_robs.shape[1], n_units_to_plot, replace=False)
    for u in units_to_plot:
        axes[1].plot(time_axis, sta_robs[:, u], 'k-', alpha=0.3)
        axes[1].plot(time_axis, sta_strobed[:, u], 'r--', alpha=0.3)
        axes[1].plot(time_axis, sta_integrated[:, u], 'b-.', alpha=0.3)
        
    axes[1].axvline(0, color='gray', linestyle=':', alpha=0.7)
    axes[1].set_title("Single Unit Examples")
    axes[1].set_xlabel("Time from Saccade Onset (ms)")
    axes[1].set_ylabel("Firing Rate")
    
    plt.tight_layout()

    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"saccadic_suppression_dataset{dataset_idx}_{stim_type}_nsac{n_saccades}_strobed_vs_integrated.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()