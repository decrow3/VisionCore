
"""spatialinfo_cache.py

Drop-in cache utilities for Declan's digital-twin spatial-information scripts.

This file is intentionally a light refactor of logic that originally lived inline in:
- natimg_digitaltwin_spatialinfo_declan.py

Goals:
- Centralize disk caching for expensive session loops (dataset loads)
- Provide simple, reusable loaders for:
  - backimage fixation aggregation
  - fixrsvp fixation bout pool
  - backimage image pixel caching

Note: The core computation functions are included verbatim from the original script
where possible, to minimize behavioral drift.
"""

from __future__ import annotations

from pathlib import Path
import os
import pickle
import hashlib
import json
from typing import Dict, Any, List, Optional, TYPE_CHECKING

import numpy as np
import torch

# External dependencies (as in original script)
from eval.eval_stack_multidataset import load_single_dataset
import importlib

try:
    _exp_mod = importlib.import_module("DataYatesV1.exp")
    BackImageTrial = getattr(_exp_mod, "BackImageTrial")
    get_trial_protocols = getattr(_exp_mod, "get_trial_protocols")
except Exception:
    BackImageTrial = None
    get_trial_protocols = None

# spatialinfo_cache.py

# -----------------------------------------------------------------------------
# Cache utility and core function imports
# -----------------------------------------------------------------------------

def _meta_hash(meta: Dict[str, Any]) -> str:
    payload = json.dumps(meta, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def cache_load_or_compute(
    cache_path: str | Path,
    compute_fn,
    *,
    meta: Dict[str, Any],
):
    """Load from pickle if meta matches; else compute, save, and return."""
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    mhash = _meta_hash(meta)

    if cache_path.exists() and meta_path.exists():
        try:
            with open(cache_path, "rb") as f:
                obj = pickle.load(f)
            saved = json.loads(meta_path.read_text())
            if saved.get("meta_hash") == mhash:
                print(f"\u2713 Loaded cache: {cache_path} (meta {mhash})")
                return obj
            else:
                print(f"Meta mismatch for {cache_path}: cached={saved.get('meta_hash')} new={mhash}; recomputing...")
        except Exception as e:
            print(f"Warning: failed to load cache {cache_path}: {e}; recomputing...")

    obj = compute_fn()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)
        meta_path.write_text(json.dumps({"meta_hash": mhash, "meta": meta}, indent=2))
        print(f"\u2713 Saved cache: {cache_path} (meta {mhash})")
    except Exception as e:
        print(f"Warning: failed to save cache {cache_path}: {e}")
    return obj


def _import_attr(module_name: str, attr_name: str):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr_name)
    except Exception as e:
        raise ImportError(f"Failed to import {attr_name} from {module_name}: {e}")

# Optional stimulus helpers
try:
    from spatial_info import make_stimulus_stack
except Exception:
    make_stimulus_stack = None
try:
    from mcfarland_sim import get_fixrsvp_stack as _get_fixrsvp_stack
except Exception:
    _get_fixrsvp_stack = None


# -----------------------------------------------------------------------------
# Cache-backed convenience wrappers
# -----------------------------------------------------------------------------

def load_backimage_fixation_results(
    *,
    model,
    sessions: List[str],
    cache_path: str | Path = "../declan/backimage_fixation_results.pkl",
    image_file: Optional[str] = None,
    n_images: int = 27,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Cached wrapper around get_fixations_for_backimage_across_sessions(...).

    Parameters
    ----------
    model, sessions : see underlying function
    cache_path : where to store/load pickled results
    image_file, n_images : forwarded to underlying function
    force_recompute : bypass cache when True
    """
    cache_path = Path(cache_path)

    meta = {
        "fn": "get_fixations_for_backimage_across_sessions",
        "sessions": list(sessions),
        "image_file": image_file,
        "n_images": int(n_images),
    }

    if force_recompute:
        fn = _import_attr(
            "scripts.natimg_digitaltwin_spatialinfo_declan",
            "get_fixations_for_backimage_across_sessions",
        )
        out = fn(
            model=model, sessions=sessions, image_file=image_file, n_images=n_images
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
        cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text(
            json.dumps({"meta_hash": "FORCED", "meta": meta}, indent=2)
        )
        return out

    def _compute():
        fn = _import_attr(
            "scripts.natimg_digitaltwin_spatialinfo_declan",
            "get_fixations_for_backimage_across_sessions",
        )
        return fn(
            model=model, sessions=sessions, image_file=image_file, n_images=n_images
        )

    return cache_load_or_compute(
        cache_path,
        _compute,
        meta=meta,
    )


def load_fixrsvp_fixation_pool(
    *,
    model,
    sessions: List[str],
    ppd: float = 37.50476617,
    min_fix_frames: int = 20,
    amp_thresh_deg: float = 1.0,
    cache_path: str | Path = "../declan/fixrsvp_fixation_pool.pkl",
    force_recompute: bool = False,
) -> List[np.ndarray]:
    """
    Cached wrapper around build_fixation_pool_from_fixrsvp(...).

    Returns
    -------
    fixation_pool : list of (T,2) arrays in degrees
    """
    cache_path = Path(cache_path)

    meta = {
        "fn": "build_fixation_pool_from_fixrsvp",
        "sessions": list(sessions),
        "ppd": float(ppd),
        "min_fix_frames": int(min_fix_frames),
        "amp_thresh_deg": float(amp_thresh_deg),
    }

    if force_recompute:
        fn = _import_attr(
            "scripts.fixrsvp_fixation_pool_declan",
            "build_fixation_pool_from_fixrsvp",
        )
        out = fn(
            model, sessions, ppd=ppd, min_fix_frames=min_fix_frames, amp_thresh_deg=amp_thresh_deg
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
        cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text(
            json.dumps({"meta_hash": "FORCED", "meta": meta}, indent=2)
        )
        return out

    def _compute():
        fn = _import_attr(
            "scripts.fixrsvp_fixation_pool_declan",
            "build_fixation_pool_from_fixrsvp",
        )
        return fn(
            model, sessions, ppd=ppd, min_fix_frames=min_fix_frames, amp_thresh_deg=amp_thresh_deg
        )

    return cache_load_or_compute(
        cache_path,
        _compute,
        meta=meta,
    )


def load_backimage_image_cache(
    *,
    model,
    sessions: List[str],
    results: Dict[str, Any],
    cache_path: str | Path = "../declan/backimage_image_cache.pkl",
    max_sessions_to_scan: int = 3,
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load (or build) a cache of backimage image pixels keyed by image_file.

    This consolidates the 'cache images to avoid re-loading datasets' logic
    that was inline in the original script.

    Parameters
    ----------
    model, sessions : used to scan datasets for BackImage trials
    results : output of load_backimage_fixation_results(...) or similar dict whose
              keys are image_file strings
    cache_path : where to store/load pickled image cache dict
    max_sessions_to_scan : scan only sessions[:max_sessions_to_scan] as in original
    force_recompute : rebuild cache from scratch when True

    Returns
    -------
    image_cache : dict mapping image_file -> image array
    """
    cache_path = Path(cache_path)

    # Determine which image files we expect
    image_files_to_plot = [img for img, _ in sorted(results.items(), key=lambda x: -x[1]["n_trials"])]

    # Try loading cached images first (verbatim behavior)
    if BackImageTrial is None or get_trial_protocols is None:
        raise ImportError("DataYatesV1.exp BackImageTrial/get_trial_protocols not available")

    image_cache: Dict[str, np.ndarray] = {}
    if (not force_recompute) and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                image_cache = pickle.load(f)
            print(f"✓ Loaded image cache: {{cache_path}} ({{len(image_cache)}} images)")
        except Exception as e:
            print(f"Warning: failed to load image cache ({{cache_path}}): {{e}}")
            image_cache = {}

    if force_recompute:
        image_cache = {}

    missing = [img for img in image_files_to_plot if img not in image_cache]

    if len(missing) > 0:
        print(f"Building image cache for {{len(missing)}} missing images (this may load a few datasets once)...")

        # Single pass over a few sessions (verbatim behavior)
        for name in sessions[:max_sessions_to_scan]:
            try:
                dataset_idx = model.names.index(name)
                train_data, val_data, _ = load_single_dataset(model, dataset_idx)

                inds = torch.concatenate(
                    [
                        train_data.get_dataset_inds("backimage"),
                        val_data.get_dataset_inds("backimage"),
                    ],
                    dim=0,
                )

                if len(inds) == 0:
                    continue

                dataset = train_data.shallow_copy()
                dataset.inds = inds
                dset_idx = inds[:, 0].unique().item()

                sess_obj = dataset.dsets[dset_idx].metadata["sess"]
                exp = sess_obj.exp
                protocols = get_trial_protocols(exp)

                trial_inds = dataset.dsets[dset_idx].covariates["trial_inds"].numpy()
                unique_trial_inds = np.unique(trial_inds[~np.isnan(trial_inds)])

                backimage_trial_inds = np.where(np.array(protocols) == "BackImage")[0]
                backimage_trial_inds = backimage_trial_inds[np.isin(backimage_trial_inds, unique_trial_inds)]
                if len(backimage_trial_inds) == 0:
                    continue

                backimage_trials = [BackImageTrial(exp["D"][iT], exp["S"]) for iT in backimage_trial_inds]

                # Fill cache for any images we still need
                for trial_obj in backimage_trials:
                    imgf = trial_obj.image_file
                    if imgf in missing and imgf not in image_cache:
                        image_cache[imgf] = trial_obj.get_image()

                # Update missing list and early-exit if done
                missing = [img for img in image_files_to_plot if img not in image_cache]
                if len(missing) == 0:
                    break

            except Exception as e:
                print(f"  Warning: Failed to cache images from session {{name}}: {{e}}")
                continue

        # Save cache for future use (verbatim behavior)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(image_cache, f)
            print(f"✓ Saved image cache: {{cache_path}} ({{len(image_cache)}} images)")
            if len(missing) > 0:
                print(f"  Note: still missing {{len(missing)}} images (not found in sessions[:{{max_sessions_to_scan}}])")
        except Exception as e:
            print(f"Warning: failed to save image cache ({{cache_path}}): {{e}}")

    return image_cache


# -----------------------------------------------------------------------------
# Lightweight stimulus loaders
# -----------------------------------------------------------------------------

def load_static_stimulus_stack(
    *,
    frame: Optional[int] = None,
    frames_per_im: int = 540,
    num_frames: int = 540,
    stim_type: str = "fixrsvp",
):
    """Create a stimulus stack via spatial_info.make_stimulus_stack."""
    if make_stimulus_stack is None:
        raise ImportError("spatial_info.make_stimulus_stack is not available")
    return make_stimulus_stack(type=stim_type, frame=frame, frames_per_im=frames_per_im, num_frames=num_frames)


def load_fixrsvp_stack(*, frames_per_im: int) -> np.ndarray:
    """Wrapper around mcfarland_sim.get_fixrsvp_stack."""
    if _get_fixrsvp_stack is None:
        raise ImportError("mcfarland_sim.get_fixrsvp_stack is not available")
    return _get_fixrsvp_stack(frames_per_im=frames_per_im)