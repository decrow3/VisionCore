#!/usr/bin/env python3
# %% [markdown]
# # Canonical V1 Twin Channel Redundancy: Activation Fingerprints
#
# Interactive first-pass inspection for the redundancy-resolved V1 twin project.
#
# This script deliberately uses percent cells so it can be opened as a Jupyter/
# VS Code notebook while remaining easy to diff. The core object is the full
# spatial response movie
#
#     T x C x H x W
#
# from the canonical shared twin readout on a real BackImage fixation trace.
# Channel fingerprints are then built as
#
#     C x (T * H * W)
#
# so the intentional convolutional spatial samples are preserved as samples of
# each channel, while clustering/merging candidates remain channels.

# %% Imports and configuration
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is for notebook display only.
    pd = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


BACKIMAGE_RESULTS_PATH = ROOT / "declan" / "backimage_fixation_results.pkl"
OUT_DIR = ROOT / "outputs" / "redundancy_resolved_v1_twin" / "step1_activation_fingerprints"
MCFARLAND_OUTPUT_CANDIDATES = (
    ROOT / "scripts" / "mcfarland_outputs_mono.pkl",
    ROOT / "scripts" / "mcfarland_outputs.pkl",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_MODE = "standard"

N_LAGS = 32
OUT_SIZE = (151, 151)
SCALE_FACTOR = 1.0
PPD = 37.50476617
BATCH_SIZE = 32

# Start small enough to remain pleasant in a notebook. Increase once the first
# response movie looks healthy.
IMAGE_KEY: str | None = None
IMAGE_RANK = 0
TRACE_INDEX = 0
MAX_FRAMES: int | None = 160
CENTER_EYE_TRACE = False

LOAD_CACHE_IF_AVAILABLE = True
SAVE_ACTIVATION_CACHE = True
SAVE_FIGURES = True

RANDOM_SEED = 7
N_RANDOM_CHANNELS = 12
N_TOP_PAIRS = 24
FINGERPRINT_NORMALIZATION = "zscore"  # one of: "none", "center", "zscore"
RUN_TSNE = True

OUT_DIR.mkdir(parents=True, exist_ok=True)


# %% Lightweight data containers
@dataclass(frozen=True)
class BackimageCase:
    image_key: str
    entry: dict[str, Any]
    image: np.ndarray
    eyepos: np.ndarray
    trace_index: int
    centered_eye_trace: bool


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    readout: torch.nn.Module
    outputs: list[dict[str, Any]]
    unit_rows: list[dict[str, Any]]
    device: str


# %% General helpers
def _safe_slug(value: object, max_len: int = 96) -> str:
    text = str(value)
    text = Path(text).stem if "/" in text else text
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_len] or "unnamed"


def _as_table(rows: list[dict[str, Any]]):
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _channel_label(channel: int, unit_rows: list[dict[str, Any]] | None = None) -> str:
    if unit_rows is None or channel >= len(unit_rows):
        return f"ch {channel}"
    row = unit_rows[channel]
    sess = str(row.get("session", "session?"))
    cid = row.get("source_unit_index", "?")
    return f"ch {channel}\n{sess}:{cid}"


def _zscore_1d(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(np.nanmean(x))) / (float(np.nanstd(x)) + eps)


# %% BackImage loading helpers
def _to_gray_float32(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = arr[:3]
        arr = 0.2989 * arr[0] + 0.5870 * arr[1] + 0.1140 * arr[2]
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image after conversion, got {arr.shape}")
    if np.nanmax(arr) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.float32)


def _image_search_dirs() -> list[Path]:
    dirs: list[Path] = []

    try:
        import importlib

        support = importlib.import_module("DataYatesV1.exp.support")
        get_backimage_directory = getattr(support, "get_backimage_directory", None)
        if callable(get_backimage_directory):
            dirs.append(Path(get_backimage_directory()))
    except Exception:
        pass

    fallback = Path("/home/declan/DataYatesV1/DataYatesV1/exp/SupportData/Backgrounds")
    dirs.extend([fallback, ROOT / "declan", ROOT / "data", ROOT / "datasets"])

    for raw in os.environ.get("VC_IMAGE_DIRS", "").split(":"):
        raw = raw.strip()
        if raw:
            dirs.append(Path(raw))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = str(path)
        if key not in seen and path.exists():
            seen.add(key)
            unique.append(path)
    return unique


def find_image_on_disk(filename: str, search_dirs: Iterable[Path] | None = None) -> Path | None:
    basename = Path(filename).name
    if not basename:
        return None
    for base in search_dirs or _image_search_dirs():
        if not base.is_dir():
            continue
        for root, _dirs, files in os.walk(base):
            if basename in files:
                return Path(root) / basename
    return None


def load_image_for_entry(entry: dict[str, Any], image_key: str) -> np.ndarray:
    if entry.get("image") is not None:
        return _to_gray_float32(entry["image"])

    for key in ("image_path", "path", "filepath", "file_path", "img_path"):
        raw_path = entry.get(key)
        if isinstance(raw_path, (list, tuple)):
            raw_path = raw_path[0] if raw_path else None
        if isinstance(raw_path, bytes):
            raw_path = raw_path.decode("utf-8", errors="ignore")
        if isinstance(raw_path, str) and Path(raw_path).exists():
            return _read_image_file(Path(raw_path))

    found = find_image_on_disk(image_key)
    if found is not None:
        return _read_image_file(found)

    raise FileNotFoundError(
        f"Could not resolve image for BackImage key {image_key!r}. "
        "Try setting VC_IMAGE_DIRS to the directory containing the background images."
    )


def _read_image_file(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return _to_gray_float32(np.asarray(image))
    except Exception:
        import imageio.v2 as imageio

        return _to_gray_float32(imageio.imread(path))


def normalize_eye_trace(raw: object, center: bool = False) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 2:
            arr = arr[:-1]
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[1] == 2:
            pass
        elif arr.shape[0] == 2:
            arr = arr.T
        elif arr.shape[1] > 2:
            arr = arr[:, :2]
        else:
            raise ValueError(f"Could not interpret eye trace shape {arr.shape}")
    else:
        flat = arr.reshape(-1)
        if flat.size % 2:
            flat = flat[:-1]
        arr = flat.reshape(-1, 2)

    finite_rows = np.isfinite(arr).all(axis=1)
    if not finite_rows.all():
        first_bad = int(np.where(~finite_rows)[0][0])
        arr = arr[:first_bad]
    if center and arr.size:
        arr = arr - np.nanmean(arr, axis=0, keepdims=True)
    return arr.astype(np.float32)


def split_entry_eye_traces(entry: dict[str, Any], center: bool = False) -> tuple[list[np.ndarray], str]:
    raw = entry.get("eyepos", [])
    n_trials_raw = entry.get("n_trials", 0)
    try:
        n_trials = int(n_trials_raw)
    except Exception:
        n_trials = 0

    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        traces = [normalize_eye_trace(arr[i, :, :2], center=center) for i in range(arr.shape[0])]
        return traces, "array_Tx2_per_trial"

    if arr.ndim == 2 and arr.shape[1] >= 2:
        arr = arr[:, :2].astype(np.float32)
        if n_trials > 1 and arr.shape[0] >= n_trials:
            frames_per_trial = int(arr.shape[0] // n_trials)
            usable = int(frames_per_trial * n_trials)
            if frames_per_trial >= 1 and usable > 0:
                traces_3d = arr[:usable].reshape(n_trials, frames_per_trial, 2)
                traces = [normalize_eye_trace(traces_3d[i], center=center) for i in range(n_trials)]
                return traces, "concatenated_samples_split_by_n_trials"
        return [normalize_eye_trace(arr, center=center)], "single_trace_array"

    if isinstance(raw, (list, tuple)):
        traces = [normalize_eye_trace(trace, center=center) for trace in raw]
        return traces, "list_of_traces"

    return [normalize_eye_trace(raw, center=center)], "single_trace_fallback"


def load_backimage_results(path: Path = BACKIMAGE_RESULTS_PATH) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected BackImage results dict at {path}, got {type(payload).__name__}")
    return payload


def summarize_backimage_results(results: dict[str, Any], n_trace_lengths: int = 20):
    rows: list[dict[str, Any]] = []
    for image_key, entry in results.items():
        eyepos_raw = entry.get("eyepos", []) if isinstance(entry, dict) else []
        traces, split_mode = split_entry_eye_traces(entry) if isinstance(entry, dict) else ([], "not_a_dict")
        lengths: list[int] = []
        for trace in traces[:n_trace_lengths]:
            try:
                lengths.append(int(normalize_eye_trace(trace).shape[0]))
            except Exception:
                pass
        eyepos_arr = np.asarray(eyepos_raw)
        rows.append(
            {
                "image_key": image_key,
                "n_trials_field": entry.get("n_trials", np.nan) if isinstance(entry, dict) else np.nan,
                "n_eye_traces": len(traces),
                "n_eye_samples_raw": int(eyepos_arr.shape[0]) if eyepos_arr.ndim >= 1 else 0,
                "median_trace_frames": float(np.median(lengths)) if lengths else np.nan,
                "min_trace_frames_sample": int(np.min(lengths)) if lengths else np.nan,
                "split_mode": split_mode,
                "has_cached_image": bool(isinstance(entry, dict) and entry.get("image") is not None),
            }
        )
    rows.sort(key=lambda row: (row["n_eye_traces"], row["n_trials_field"]), reverse=True)
    return _as_table(rows)


def select_backimage_case(
    results: dict[str, Any],
    image_key: str | None = IMAGE_KEY,
    image_rank: int = IMAGE_RANK,
    trace_index: int = TRACE_INDEX,
    max_frames: int | None = MAX_FRAMES,
    center_eye_trace: bool = CENTER_EYE_TRACE,
) -> BackimageCase:
    if image_key is None:
        def _rank_entry(item: tuple[str, Any]) -> tuple[int, int]:
            _key, candidate = item
            if not isinstance(candidate, dict):
                return (0, 0)
            traces, _split_mode = split_entry_eye_traces(candidate)
            try:
                n_trials = int(candidate.get("n_trials", 0))
            except Exception:
                n_trials = 0
            return (len(traces), n_trials)

        ordered = sorted(
            results.items(),
            key=_rank_entry,
            reverse=True,
        )
        image_key, entry = ordered[int(image_rank)]
    else:
        entry = results[image_key]

    if not isinstance(entry, dict):
        raise TypeError(f"BackImage entry {image_key!r} is not a dict")
    traces, split_mode = split_entry_eye_traces(entry, center=center_eye_trace)
    if not traces:
        raise ValueError(f"BackImage entry {image_key!r} has no eyepos traces")

    trace_index = int(np.clip(trace_index, 0, len(traces) - 1))
    eyepos = normalize_eye_trace(traces[trace_index], center=False)
    if max_frames is not None:
        eyepos = eyepos[: int(max_frames)]
    if eyepos.shape[0] < 2:
        raise ValueError(
            f"Selected trace is too short after cleanup: {eyepos.shape}. "
            f"Entry split mode was {split_mode!r}; try a different TRACE_INDEX or inspect this entry."
        )

    image = load_image_for_entry(entry, str(image_key))
    return BackimageCase(
        image_key=str(image_key),
        entry=entry,
        image=image,
        eyepos=eyepos,
        trace_index=trace_index,
        centered_eye_trace=center_eye_trace,
    )


def plot_backimage_case(case: BackimageCase, max_points: int = 2000):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(case.image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("BackImage")
    axes[0].axis("off")

    eye = case.eyepos[:max_points]
    axes[1].plot(eye[:, 0], label="x", linewidth=1.2)
    axes[1].plot(eye[:, 1], label="y", linewidth=1.2)
    axes[1].set_title("Eye Trace")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("deg")
    axes[1].legend(frameon=False)

    axes[2].plot(eye[:, 0], eye[:, 1], linewidth=1.0)
    axes[2].scatter(eye[0, 0], eye[0, 1], s=28, label="start")
    axes[2].scatter(eye[-1, 0], eye[-1, 1], s=28, label="end")
    axes[2].set_title("Eye Path")
    axes[2].set_xlabel("x deg")
    axes[2].set_ylabel("y deg")
    axes[2].axis("equal")
    axes[2].legend(frameon=False)
    return fig


# %% Model/readout helpers
def _load_pickle_or_dill(path: Path) -> Any:
    try:
        import dill

        with path.open("rb") as handle:
            return dill.load(handle)
    except ImportError:
        with path.open("rb") as handle:
            return pickle.load(handle)


def load_mcfarland_outputs(candidates: Iterable[Path] = MCFARLAND_OUTPUT_CANDIDATES) -> list[dict[str, Any]]:
    for path in candidates:
        if path.exists():
            outputs = _load_pickle_or_dill(path)
            if not isinstance(outputs, list):
                raise TypeError(f"Expected list in {path}, got {type(outputs).__name__}")
            return outputs
    candidates_s = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find any mcfarland outputs file among: {candidates_s}")


def build_readout_unit_rows(
    model: torch.nn.Module,
    outputs: list[dict[str, Any]],
    ccnorm_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    sessions = [str(output.get("sess", "")) for output in outputs]
    model_names = [str(name) for name in getattr(model, "names", [])]
    rows: list[dict[str, Any]] = []

    for model_readout_index, session in enumerate(model_names):
        if session not in sessions:
            continue
        output_index = sessions.index(session)
        output = outputs[output_index]
        ccnorm_payload = output.get("ccnorm", {})
        ccnorm = np.asarray(ccnorm_payload.get("ccnorm", []), dtype=np.float32)
        if ccnorm.size == 0:
            continue
        source_unit_indices = np.where(ccnorm > float(ccnorm_threshold))[0]
        for source_unit_index in source_unit_indices:
            rows.append(
                {
                    "channel": len(rows),
                    "session": session,
                    "source_unit_index": int(source_unit_index),
                    "ccnorm": float(ccnorm[source_unit_index]),
                    "model_readout_index": int(model_readout_index),
                    "mcfarland_output_index": int(output_index),
                }
            )
    return rows


def load_model_bundle(device: str = DEVICE, mode: str = MODEL_MODE) -> ModelBundle:
    from spatial_info import get_spatial_readout
    from utils import get_model_and_dataset_configs

    model, _dataset_configs = get_model_and_dataset_configs(mode=mode)
    model = model.to(device).eval()
    outputs = load_mcfarland_outputs()
    readout, unit_rows = get_spatial_readout(model, outputs, return_unit_rows=True)
    readout = readout.to(device).eval()
    return ModelBundle(model=model, readout=readout, outputs=outputs, unit_rows=unit_rows, device=device)


# %% Activation movie construction and caching
def activation_cache_path(case: BackimageCase) -> Path:
    slug = _safe_slug(case.image_key)
    frames = case.eyepos.shape[0]
    centered = "centered" if case.centered_eye_trace else "stored"
    name = (
        f"activation_movie_{slug}_trace{case.trace_index:03d}_{centered}"
        f"_frames{frames}_lag{N_LAGS}_out{OUT_SIZE[0]}x{OUT_SIZE[1]}_scale{SCALE_FACTOR:g}.npz"
    )
    return OUT_DIR / name


def build_retinal_stimulus(case: BackimageCase) -> torch.Tensor:
    from mcfarland_sim import shift_movie_with_eye
    from spatial_info import embed_time_lags
    from scripts.fixrsvp_eye_conventions import stored_eyepos_to_eye_norm

    if case.eyepos.shape[0] < N_LAGS:
        raise ValueError(f"Need at least N_LAGS={N_LAGS} eye samples, got {case.eyepos.shape[0]}")

    image = np.asarray(case.image, dtype=np.float32)
    eyepos_t = torch.from_numpy(case.eyepos).float()
    eye_norm = stored_eyepos_to_eye_norm(eyepos_t, PPD, image.shape[-2:], device=eyepos_t.device)

    movie = torch.from_numpy(image).float().unsqueeze(0).expand(case.eyepos.shape[0] + N_LAGS, -1, -1)
    eye_movie = shift_movie_with_eye(
        movie,
        torch.cat([eye_norm[:N_LAGS], eye_norm], dim=0),
        out_size=OUT_SIZE,
        scale_factor=SCALE_FACTOR,
        mode="bilinear",
    )
    stim = embed_time_lags(eye_movie, n_lags=N_LAGS)
    return (stim - 127.0) / 255.0


def compute_activation_movie(
    case: BackimageCase,
    bundle: ModelBundle,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    from spatial_info import compute_rate_map_batched

    stim = build_retinal_stimulus(case)
    with torch.no_grad():
        y = compute_rate_map_batched(bundle.model, bundle.readout, stim, batch_size=batch_size)
    y_np = y.detach().cpu().numpy().astype(np.float32) if isinstance(y, torch.Tensor) else np.asarray(y, dtype=np.float32)
    if y_np.ndim != 4:
        raise ValueError(f"Expected activation movie T x C x H x W, got {y_np.shape}")
    return y_np


def save_activation_cache(
    path: Path,
    activation_movie: np.ndarray,
    case: BackimageCase,
    bundle: ModelBundle,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unit_sessions = np.asarray([str(row.get("session", "")) for row in bundle.unit_rows])
    source_unit_indices = np.asarray([int(row.get("source_unit_index", -1)) for row in bundle.unit_rows], dtype=np.int64)
    ccnorm = np.asarray([float(row.get("ccnorm", np.nan)) for row in bundle.unit_rows], dtype=np.float32)
    metadata = {
        "analysis": "canonical_twin_channel_redundancy_step1_activation_fingerprints",
        "image_key": case.image_key,
        "trace_index": case.trace_index,
        "centered_eye_trace": case.centered_eye_trace,
        "activation_shape": list(activation_movie.shape),
        "n_lags": N_LAGS,
        "out_size": list(OUT_SIZE),
        "scale_factor": SCALE_FACTOR,
        "ppd": PPD,
        "model_mode": MODEL_MODE,
        "device": bundle.device,
    }
    np.savez_compressed(
        path,
        activation_movie=activation_movie.astype(np.float32),
        image=case.image.astype(np.float32),
        eyepos=case.eyepos.astype(np.float32),
        unit_sessions=unit_sessions,
        source_unit_indices=source_unit_indices,
        ccnorm=ccnorm,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    _save_json(path.with_suffix(".json"), metadata)
    _write_csv(path.with_name(path.stem + "_unit_table.csv"), bundle.unit_rows)


def load_activation_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        metadata_json = str(npz["metadata_json"].item()) if "metadata_json" in npz else "{}"
        return {
            "activation_movie": np.asarray(npz["activation_movie"], dtype=np.float32),
            "image": np.asarray(npz["image"], dtype=np.float32) if "image" in npz else None,
            "eyepos": np.asarray(npz["eyepos"], dtype=np.float32) if "eyepos" in npz else None,
            "metadata": json.loads(metadata_json),
        }


# %% Activation diagnostics
def activation_summary_rows(activation_movie: np.ndarray) -> list[dict[str, Any]]:
    y = np.asarray(activation_movie, dtype=np.float32)
    rows = [
        {"metric": "shape_T_C_H_W", "value": " x ".join(str(v) for v in y.shape)},
        {"metric": "finite_fraction", "value": float(np.isfinite(y).mean())},
        {"metric": "global_min", "value": float(np.nanmin(y))},
        {"metric": "global_max", "value": float(np.nanmax(y))},
        {"metric": "global_mean", "value": float(np.nanmean(y))},
        {"metric": "global_std", "value": float(np.nanstd(y))},
    ]
    channel_std = np.nanstd(y, axis=(0, 2, 3))
    rows.extend(
        [
            {"metric": "channel_std_median", "value": float(np.nanmedian(channel_std))},
            {"metric": "near_silent_channels_std_lt_1e-6", "value": int(np.sum(channel_std < 1e-6))},
        ]
    )
    return rows


def spatial_mean_traces(activation_movie: np.ndarray) -> np.ndarray:
    y = np.asarray(activation_movie, dtype=np.float32)
    return y.mean(axis=(2, 3))


def channel_variance_rank(activation_movie: np.ndarray) -> np.ndarray:
    y = np.asarray(activation_movie, dtype=np.float32)
    return np.argsort(np.nanstd(y, axis=(0, 2, 3)))[::-1]


def plot_activation_distributions(activation_movie: np.ndarray):
    y = np.asarray(activation_movie, dtype=np.float32)
    channel_mean = np.nanmean(y, axis=(0, 2, 3))
    channel_std = np.nanstd(y, axis=(0, 2, 3))
    traces = spatial_mean_traces(y)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].hist(channel_mean, bins=60, color="0.25")
    axes[0].set_title("Channel Mean")
    axes[0].set_xlabel("activation")
    axes[0].set_ylabel("channels")

    axes[1].hist(channel_std, bins=60, color="0.25")
    axes[1].set_title("Channel Std")
    axes[1].set_xlabel("activation std")

    axes[2].plot(np.nanmean(traces, axis=1), color="black", linewidth=1.2)
    axes[2].fill_between(
        np.arange(traces.shape[0]),
        np.nanpercentile(traces, 10, axis=1),
        np.nanpercentile(traces, 90, axis=1),
        color="0.7",
        alpha=0.5,
        linewidth=0,
    )
    axes[2].set_title("Population Activity")
    axes[2].set_xlabel("model frame")
    axes[2].set_ylabel("spatial-mean activation")
    return fig


def plot_channel_trace_snippets(
    activation_movie: np.ndarray,
    channels: Iterable[int],
    unit_rows: list[dict[str, Any]] | None = None,
    frame_slice: slice = slice(0, 120),
    zscore: bool = True,
):
    traces = spatial_mean_traces(activation_movie)
    channels = [int(c) for c in channels]
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    for channel in channels:
        trace = traces[frame_slice, channel]
        if zscore:
            trace = _zscore_1d(trace)
        ax.plot(np.arange(trace.shape[0]), trace, linewidth=1.0, alpha=0.85, label=_channel_label(channel, unit_rows))
    ax.set_title("Spatial-Mean Channel Trace Snippets")
    ax.set_xlabel("model frame")
    ax.set_ylabel("z-scored activation" if zscore else "activation")
    if len(channels) <= 12:
        ax.legend(frameon=False, ncol=2, fontsize=8)
    return fig


def plot_channel_map_strip(
    activation_movie: np.ndarray,
    channels: Iterable[int],
    frames: Iterable[int],
    unit_rows: list[dict[str, Any]] | None = None,
    robust: bool = True,
):
    y = np.asarray(activation_movie, dtype=np.float32)
    channels = [int(c) for c in channels]
    frames = [int(f) for f in frames if 0 <= int(f) < y.shape[0]]
    if not channels or not frames:
        raise ValueError("Need at least one channel and one valid frame")

    fig, axes = plt.subplots(
        len(channels),
        len(frames),
        figsize=(1.8 * len(frames), 1.75 * len(channels)),
        squeeze=False,
        constrained_layout=True,
    )
    for r, channel in enumerate(channels):
        vals = y[:, channel]
        if robust:
            vmin, vmax = np.nanpercentile(vals, [2, 98])
        else:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None
        for c, frame in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(y[frame, channel], cmap="magma", vmin=vmin, vmax=vmax)
            if r == 0:
                ax.set_title(f"t={frame}")
            if c == 0:
                ax.set_ylabel(_channel_label(channel, unit_rows), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    return fig


def plot_top_pair_overview(
    activation_movie: np.ndarray,
    pairs: list[dict[str, Any]],
    unit_rows: list[dict[str, Any]] | None = None,
    n_pairs: int = 8,
    frame_slice: slice = slice(0, 120),
):
    y = np.asarray(activation_movie, dtype=np.float32)
    traces = spatial_mean_traces(y)
    selected = pairs[:n_pairs]
    if not selected:
        raise ValueError("No pairs to plot")

    fig, axes = plt.subplots(len(selected), 3, figsize=(13, 2.0 * len(selected)), squeeze=False, constrained_layout=True)
    for r, pair in enumerate(selected):
        a = int(pair["channel_a"])
        b = int(pair["channel_b"])
        corr = float(pair["corr"])
        ax = axes[r, 0]
        ta = _zscore_1d(traces[frame_slice, a])
        tb = _zscore_1d(traces[frame_slice, b])
        ax.plot(ta, label=_channel_label(a, unit_rows), linewidth=1.0)
        ax.plot(tb, label=_channel_label(b, unit_rows), linewidth=1.0)
        ax.set_title(f"rank {pair.get('rank', r + 1)} corr={corr:.3f}")
        ax.set_xlabel("frame")
        ax.set_ylabel("z")
        ax.legend(frameon=False, fontsize=7)

        mean_a = np.nanmean(y[:, a], axis=0)
        mean_b = np.nanmean(y[:, b], axis=0)
        vmin = float(np.nanpercentile(np.stack([mean_a, mean_b]), 2))
        vmax = float(np.nanpercentile(np.stack([mean_a, mean_b]), 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None
        axes[r, 1].imshow(mean_a, cmap="magma", vmin=vmin, vmax=vmax)
        axes[r, 1].set_title("mean map A")
        axes[r, 2].imshow(mean_b, cmap="magma", vmin=vmin, vmax=vmax)
        axes[r, 2].set_title("mean map B")
        for c in (1, 2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    return fig


# %% Fingerprints and similarity
def make_channel_fingerprints(
    activation_movie: np.ndarray,
    normalization: str = FINGERPRINT_NORMALIZATION,
    eps: float = 1e-6,
) -> np.ndarray:
    y = np.asarray(activation_movie, dtype=np.float32)
    fingerprints = np.transpose(y, (1, 0, 2, 3)).reshape(y.shape[1], -1)
    if normalization == "none":
        return fingerprints.astype(np.float32, copy=False)
    fingerprints = fingerprints - np.nanmean(fingerprints, axis=1, keepdims=True)
    if normalization == "center":
        return fingerprints.astype(np.float32, copy=False)
    if normalization == "zscore":
        fingerprints = fingerprints / (np.nanstd(fingerprints, axis=1, keepdims=True) + eps)
        return fingerprints.astype(np.float32, copy=False)
    raise ValueError(f"Unknown fingerprint normalization {normalization!r}")


def channel_correlation_from_fingerprints(fingerprints: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(fingerprints, dtype=np.float32)
    x = x - np.nanmean(x, axis=1, keepdims=True)
    norms = np.sqrt(np.nansum(x * x, axis=1, keepdims=True))
    z = x / np.maximum(norms, eps)
    corr = z @ z.T
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def top_correlated_pairs(
    corr: np.ndarray,
    unit_rows: list[dict[str, Any]] | None = None,
    n_pairs: int = N_TOP_PAIRS,
    min_corr: float | None = None,
) -> list[dict[str, Any]]:
    corr = np.asarray(corr, dtype=np.float32)
    ii, jj = np.triu_indices(corr.shape[0], k=1)
    vals = corr[ii, jj]
    order = np.argsort(vals)[::-1]
    rows: list[dict[str, Any]] = []
    for idx in order:
        value = float(vals[idx])
        if min_corr is not None and value < min_corr:
            break
        a = int(ii[idx])
        b = int(jj[idx])
        row: dict[str, Any] = {
            "rank": len(rows) + 1,
            "channel_a": a,
            "channel_b": b,
            "corr": value,
        }
        if unit_rows is not None and a < len(unit_rows) and b < len(unit_rows):
            row.update(
                {
                    "session_a": unit_rows[a].get("session", ""),
                    "source_unit_a": unit_rows[a].get("source_unit_index", ""),
                    "ccnorm_a": unit_rows[a].get("ccnorm", np.nan),
                    "session_b": unit_rows[b].get("session", ""),
                    "source_unit_b": unit_rows[b].get("source_unit_index", ""),
                    "ccnorm_b": unit_rows[b].get("ccnorm", np.nan),
                    "same_session": unit_rows[a].get("session", "") == unit_rows[b].get("session", ""),
                }
            )
        rows.append(row)
        if len(rows) >= int(n_pairs):
            break
    return rows


def plot_correlation_heatmap(
    corr: np.ndarray,
    channel_order: Iterable[int] | None = None,
    max_channels: int = 220,
):
    corr = np.asarray(corr, dtype=np.float32)
    if channel_order is None:
        order = np.arange(corr.shape[0])
    else:
        order = np.asarray(list(channel_order), dtype=int)
    order = order[: min(max_channels, order.size)]
    sub = corr[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(sub, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"Channel Fingerprint Correlation ({len(order)} channels)")
    ax.set_xlabel("ordered channel")
    ax.set_ylabel("ordered channel")
    fig.colorbar(im, ax=ax, shrink=0.8, label="corr")
    return fig


def save_fingerprint_cache(path: Path, fingerprints: np.ndarray, corr: np.ndarray, top_pairs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fingerprints=fingerprints.astype(np.float32),
        corr=corr.astype(np.float32),
        top_pairs_json=np.asarray(json.dumps(top_pairs, sort_keys=True, default=_json_default)),
    )
    _write_csv(path.with_suffix(".top_pairs.csv"), top_pairs)


# %% Optional embedding view
def compute_channel_embedding(
    fingerprints: np.ndarray,
    run_tsne: bool = RUN_TSNE,
    seed: int = RANDOM_SEED,
    pca_dim: int = 30,
) -> tuple[np.ndarray, str]:
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:
        raise RuntimeError(f"scikit-learn is needed for embeddings: {exc}") from exc

    x = np.asarray(fingerprints, dtype=np.float32)
    n_components = min(int(pca_dim), x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(x)
    if not run_tsne:
        if scores.shape[1] < 2:
            scores = np.pad(scores, ((0, 0), (0, 2 - scores.shape[1])))
        return scores[:, :2], "PCA"

    try:
        from sklearn.manifold import TSNE
    except Exception as exc:
        raise RuntimeError(f"scikit-learn TSNE import failed: {exc}") from exc

    perplexity = min(30.0, max(5.0, (x.shape[0] - 1) / 4.0))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    return tsne.fit_transform(scores), "t-SNE on PCA fingerprints"


def plot_channel_embedding(
    embedding: np.ndarray,
    title: str,
    unit_rows: list[dict[str, Any]] | None = None,
):
    fig, ax = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
    if unit_rows is None:
        colors = np.arange(embedding.shape[0])
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=16, cmap="viridis", alpha=0.8)
        fig.colorbar(scatter, ax=ax, shrink=0.8, label="channel")
    else:
        sessions = [str(row.get("session", "")) for row in unit_rows]
        unique = {session: i for i, session in enumerate(sorted(set(sessions)))}
        colors = np.asarray([unique[session] for session in sessions])
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=16, cmap="tab20", alpha=0.82)
        fig.colorbar(scatter, ax=ax, shrink=0.8, label="session index")
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    return fig


# %% Step 1: load and summarize BackImage fixation cases
backimage_results = load_backimage_results(BACKIMAGE_RESULTS_PATH)
backimage_summary = summarize_backimage_results(backimage_results)
backimage_summary.head(12) if pd is not None else backimage_summary[:12]


# %% Step 2: select one image/trace and inspect the retinal drive
case = select_backimage_case(
    backimage_results,
    image_key=IMAGE_KEY,
    image_rank=IMAGE_RANK,
    trace_index=TRACE_INDEX,
    max_frames=MAX_FRAMES,
    center_eye_trace=CENTER_EYE_TRACE,
)
fig_case = plot_backimage_case(case)
if SAVE_FIGURES:
    fig_case.savefig(OUT_DIR / f"case_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png", dpi=160)
plt.show()


# %% Step 3: load canonical shared twin/readout and source-unit metadata
bundle = load_model_bundle(device=DEVICE, mode=MODEL_MODE)
unit_table = _as_table(bundle.unit_rows)
_write_csv(OUT_DIR / "canonical_shared_twin_unit_table.csv", bundle.unit_rows)
unit_table.head(12) if pd is not None else unit_table[:12]


# %% Step 4: compute or load the full T x C x H x W activation movie
cache_path = activation_cache_path(case)
if LOAD_CACHE_IF_AVAILABLE and cache_path.exists():
    payload = load_activation_cache(cache_path)
    activation_movie = payload["activation_movie"]
    print(f"Loaded activation movie cache: {cache_path}")
else:
    activation_movie = compute_activation_movie(case, bundle, batch_size=BATCH_SIZE)
    if SAVE_ACTIVATION_CACHE:
        save_activation_cache(cache_path, activation_movie, case, bundle)
        print(f"Saved activation movie cache: {cache_path}")

activation_summary = _as_table(activation_summary_rows(activation_movie))
activation_summary


# %% Step 5: sanity plots for the activation movie
fig_dist = plot_activation_distributions(activation_movie)
if SAVE_FIGURES:
    fig_dist.savefig(OUT_DIR / f"activation_distributions_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png", dpi=160)
plt.show()

variance_order = channel_variance_rank(activation_movie)
rng = np.random.default_rng(RANDOM_SEED)
active_pool = variance_order[: max(50, min(activation_movie.shape[1], 250))]
random_channels = rng.choice(active_pool, size=min(N_RANDOM_CHANNELS, active_pool.size), replace=False)
fig_traces = plot_channel_trace_snippets(activation_movie, random_channels, unit_rows=bundle.unit_rows)
if SAVE_FIGURES:
    fig_traces.savefig(OUT_DIR / f"random_channel_traces_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png", dpi=160)
plt.show()


# %% Step 6: inspect spatial activation maps for active channels
preview_channels = variance_order[: min(6, activation_movie.shape[1])]
preview_frames = np.linspace(0, activation_movie.shape[0] - 1, num=min(8, activation_movie.shape[0]), dtype=int)
fig_maps = plot_channel_map_strip(activation_movie, preview_channels, preview_frames, unit_rows=bundle.unit_rows)
if SAVE_FIGURES:
    fig_maps.savefig(OUT_DIR / f"active_channel_map_strip_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png", dpi=180)
plt.show()


# %% Step 7: build C x (T * H * W) channel fingerprints and similarity matrix
fingerprints = make_channel_fingerprints(activation_movie, normalization=FINGERPRINT_NORMALIZATION)
corr = channel_correlation_from_fingerprints(fingerprints)
top_pairs = top_correlated_pairs(corr, unit_rows=bundle.unit_rows, n_pairs=N_TOP_PAIRS)
top_pairs_table = _as_table(top_pairs)

fingerprint_cache_path = cache_path.with_name(cache_path.stem + f"_fingerprints_{FINGERPRINT_NORMALIZATION}.npz")
save_fingerprint_cache(fingerprint_cache_path, fingerprints, corr, top_pairs)
print(f"Saved fingerprint cache: {fingerprint_cache_path}")
top_pairs_table


# %% Step 8: visualize similarity structure and suspiciously redundant pairs
fig_corr = plot_correlation_heatmap(corr, channel_order=variance_order)
if SAVE_FIGURES:
    fig_corr.savefig(OUT_DIR / f"fingerprint_corr_heatmap_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png", dpi=170)
plt.show()

fig_pairs = plot_top_pair_overview(activation_movie, top_pairs, unit_rows=bundle.unit_rows, n_pairs=min(8, len(top_pairs)))
if SAVE_FIGURES:
    fig_pairs.savefig(OUT_DIR / f"top_correlated_pair_overview_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png", dpi=170)
plt.show()


# %% Step 9: optional PCA / t-SNE atlas for visual cluster sniffing
RUN_TSNE = True  # set to False to just run PCA instead of t-SNE
try:
    embedding, embedding_name = compute_channel_embedding(fingerprints, run_tsne=RUN_TSNE)
    fig_embed = plot_channel_embedding(embedding, embedding_name, unit_rows=bundle.unit_rows)
    if SAVE_FIGURES:
        suffix = "tsne" if RUN_TSNE else "pca"
        fig_embed.savefig(OUT_DIR / f"channel_embedding_{suffix}_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png", dpi=170)
    plt.show()
except RuntimeError as exc:
    print(f"Skipping optional embedding cell: {exc}")
