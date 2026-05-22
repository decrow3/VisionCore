"""Import-safe FIXRSVP fixation-bout extraction.

This module exists so caching utilities can import the core fixation-pool builder
without importing notebook-style scripts that execute analysis at import time.

Public API:
- build_fixation_pool_from_fixrsvp
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from eval.eval_stack_multidataset import load_single_dataset


def build_fixation_pool_from_fixrsvp(
    model,
    sessions: List[str],
    *,
    ppd: float = 37.50476617,
    min_fix_frames: int = 20,
    amp_thresh_deg: float = 1.0,
):
    """Return contiguous fixation bouts from FIXRSVP samples.

    Parameters
    ----------
    model : VisionCore model (must have .names)
    sessions : list of session names
    ppd : pixels per degree (used only if eyepos appears to be in pixels)
    min_fix_frames : minimum contiguous fixation length to keep
    amp_thresh_deg : fixation threshold, applied in degrees

    Returns
    -------
    fixation_pool : list of np.ndarray (T,2) eye positions in degrees
    """

    fixation_pool: list[np.ndarray] = []

    for name in sessions:
        try:
            dataset_idx = model.names.index(name)
            train_data, val_data, _ = load_single_dataset(model, dataset_idx)

            inds = torch.concatenate(
                [
                    train_data.get_dataset_inds("fixrsvp"),
                    val_data.get_dataset_inds("fixrsvp"),
                ],
                dim=0,
            )
            if len(inds) == 0:
                continue

            dataset = train_data.shallow_copy()
            dataset.inds = inds

            inds_np = inds.detach().cpu().numpy()
            for dset_idx in np.unique(inds_np[:, 0]).astype(int):
                eyepos_all = dataset.dsets[dset_idx]["eyepos"][:].numpy()  # (N,2)
                trial_inds = dataset.dsets[dset_idx].covariates["trial_inds"].numpy()

                # Explicit protocol slicing: dataset.inds may not affect raw dsets access.
                in_fixrsvp = np.zeros(len(trial_inds), dtype=bool)
                sample_idx = inds_np[inds_np[:, 0] == dset_idx, 1].astype(int)
                sample_idx = sample_idx[(sample_idx >= 0) & (sample_idx < len(in_fixrsvp))]
                in_fixrsvp[sample_idx] = True

                trials = np.unique(trial_inds[in_fixrsvp & ~np.isnan(trial_inds)])

                # Unit check: if amplitude is large, treat as pixels and convert to degrees.
                median_amp = np.nanmedian(np.hypot(eyepos_all[:, 0], eyepos_all[:, 1]))
                if median_amp > 5.0:
                    print(
                        f"[{name}] eyepos appears in pixels (median={median_amp:.2f}); "
                        f"converting to degrees (/ppd={ppd:.2f})"
                    )
                    eyepos_all = eyepos_all / float(ppd)

                fixation_mask_all = (
                    np.hypot(eyepos_all[:, 0], eyepos_all[:, 1]) < float(amp_thresh_deg)
                ) & in_fixrsvp

                for t in trials:
                    trial_sample_inds = np.where((trial_inds == t) & in_fixrsvp)[0]
                    if trial_sample_inds.size == 0:
                        continue

                    fix_sample_inds = trial_sample_inds[fixation_mask_all[trial_sample_inds]]
                    if fix_sample_inds.size == 0:
                        continue

                    # Split by adjacency on original sample indices.
                    split_pts = np.where(np.diff(fix_sample_inds) != 1)[0] + 1
                    runs = np.split(fix_sample_inds, split_pts)
                    for run_inds in runs:
                        if run_inds.size >= int(min_fix_frames):
                            fixation_pool.append(eyepos_all[run_inds].astype(np.float32))

        except Exception as e:
            print(f"Failed to load FIXRSVP from session {name}: {e}")

    return fixation_pool
