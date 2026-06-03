from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from VisionCore.paths import VISIONCORE_ROOT
except Exception:
    VISIONCORE_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_OUT_ROOT = VISIONCORE_ROOT / "outputs" / "twin_covariance_structure" / "shared_transformation_geometry"


def load_sessions_from_dataset_config(dataset_configs_path: Path) -> list[str]:
    cfg = yaml.safe_load(dataset_configs_path.read_text(encoding="utf-8"))
    sessions = cfg.get("sessions", [])
    return [str(s) for s in sessions]


def parse_session_name(session_name: str) -> tuple[str, str]:
    subject, date = str(session_name).split("_", 1)
    return subject, date


def harmonize_fixrsvp_arrays(data: dict[str, Any]) -> dict[str, Any]:
    robs = np.asarray(data.get("robs")) if data.get("robs") is not None else None
    eyepos = np.asarray(data.get("eyepos")) if data.get("eyepos") is not None else None
    image_ids = np.asarray(data.get("image_ids")) if data.get("image_ids") is not None else None
    stim = np.asarray(data.get("stim")) if data.get("stim") is not None else None

    nt_candidates: list[int] = []
    t_candidates: list[int] = []
    for arr in (robs, eyepos, image_ids, stim):
        if arr is None or arr.ndim < 2:
            continue
        nt_candidates.append(int(arr.shape[0]))
        t_candidates.append(int(arr.shape[1]))

    if not nt_candidates or not t_candidates:
        return dict(data)

    nt = min(nt_candidates)
    tt = min(t_candidates)

    out = dict(data)
    if robs is not None and robs.ndim >= 2:
        out["robs"] = robs[:nt, :tt, ...]
    if eyepos is not None and eyepos.ndim >= 2:
        out["eyepos"] = eyepos[:nt, :tt, ...]
    if image_ids is not None and image_ids.ndim >= 2:
        out["image_ids"] = image_ids[:nt, :tt, ...]
    if stim is not None and stim.ndim >= 2:
        out["stim"] = stim[:nt, :tt, ...]
    return out


def json_dumps_pretty(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
