"""Generate/cache E-optotype rate files for decoder analyses.

This fills the `.npz` caches consumed by:
- scripts/temporal_decoding/eoptotype_decoder_controls.py
- scripts/temporal_decoding/integration_time_controls.py

It is intentionally *cache-first*: if a cache exists, it is reused unless `--force`
(or `--force_missing_only` is not set).

Examples
--------
# Cache hi-res rates for a lower LogMAR sweep (real + stabilized)
/home/declan/VisionCore/.venv/bin/python scripts/temporal_decoding/cache_eoptotype_rates.py \
  --logmars -0.30,-0.35,-0.40,-0.45,-0.50 \
  --conditions real,stabilized \
  --orientations 0,90,180,270

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RATES_DIR = DATA_DIR / "rates"
EYE_TRACES_PATH = DATA_DIR / "eye_traces.npz"
PKL_PATH = SCRIPT_DIR.parent / "mcfarland_outputs_mono.pkl"

HIRES_THRESHOLD_DEFAULT = 0.35


def _parse_csv_floats(s: str) -> list[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


def _parse_csv_ints(s: str) -> list[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(float(x)) for x in str(s).split(",") if str(x).strip() != ""]


def _parse_csv_strings(s: str) -> list[str]:
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def _format_logmar_for_filename(logmar: float) -> str:
    return f"{float(logmar):.2f}"


def _normalize_file_tag(tag: str | None) -> str:
    """Normalize a user-provided tag for cache filenames.

    - Empty/None -> ''
    - Non-empty -> ensure leading '_' and restrict to safe characters
    """
    if tag is None:
        return ""
    tag = str(tag).strip()
    if tag == "":
        return ""
    if not tag.startswith("_"):
        tag = "_" + tag
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. ")
    if any(ch not in allowed for ch in tag):
        raise ValueError(
            "--file_tag contains unsupported characters; use only letters, numbers, '_', '-', '.', or spaces"
        )
    return tag.replace(" ", "_")


def _cache_path(
    rates_dir: Path,
    logmar: float,
    orientation: int,
    condition: str,
    hires: bool,
    file_tag: str = "",
) -> Path:
    prefix = "rates_hires_lm" if hires else "rates_lm"
    lm_str = _format_logmar_for_filename(float(logmar))
    file_tag = _normalize_file_tag(file_tag)
    return rates_dir / f"{prefix}{lm_str}_ori{int(orientation)}_{condition}{file_tag}.npz"


def _load_model_and_readout(device: str | None = None):
    import dill
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not PKL_PATH.exists():
        raise FileNotFoundError(f"Missing model outputs pickle: {PKL_PATH}")

    sys.path.insert(0, str(SCRIPT_DIR.parent))

    from utils import get_model_and_dataset_configs
    from spatial_info import get_spatial_readout

    print(f"Loading model (device={device})...", flush=True)
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


def _load_eye_traces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    traces = d["traces"].astype(np.float32)
    durations = d["durations"].astype(int)
    return traces, durations


def _compute_and_save_one(
    *,
    model,
    readout,
    rates_dir: Path,
    logmar: float,
    orientation: int,
    condition: str,
    traces: np.ndarray,
    durations: np.ndarray,
    hires_threshold: float,
    batch_size: int,
    spatial_collapse: str,
    force: bool,
    file_tag: str,
) -> Path:
    from rate_computation import compute_population_rates, compute_population_rates_hires, save_rates

    use_hires = float(logmar) < float(hires_threshold)
    out_path = _cache_path(rates_dir, logmar, orientation, condition, hires=use_hires, file_tag=file_tag)
    if out_path.exists() and not force:
        return out_path

    pipeline = "hi-res" if use_hires else "lo-res"
    print(
        f"  Computing {pipeline}: LM={logmar:.2f} ori={orientation} cond={condition} -> {out_path.name}",
        flush=True,
    )

    if use_hires:
        result = compute_population_rates_hires(
            model,
            readout,
            float(orientation),
            float(logmar),
            traces,
            durations,
            condition=condition,
            batch_size=batch_size,
            spatial_collapse=spatial_collapse,
            stim_params={"logmar": float(logmar), "orientation": int(orientation)},
            verbose=True,
        )
    else:
        from stimulus import e_optotype_stack

        stim_stack = e_optotype_stack(int(orientation), float(logmar))
        result = compute_population_rates(
            model,
            readout,
            stim_stack,
            traces,
            durations,
            condition=condition,
            batch_size=batch_size,
            spatial_collapse=spatial_collapse,
            stim_params={"logmar": float(logmar), "orientation": int(orientation)},
            verbose=True,
        )

    save_rates(result, str(out_path))
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache E-optotype rate files")
    parser.add_argument(
        "--logmars",
        type=str,
        required=True,
        help="Comma-separated LogMAR values (e.g. '-0.30,-0.35,-0.40').",
    )
    parser.add_argument(
        "--orientations",
        type=str,
        default="0,90,180,270",
        help="Comma-separated orientations in degrees.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="real,stabilized",
        help="Comma-separated FEM conditions (e.g. 'real,stabilized,fixed_center,scaled_0.5,scaled_2.0').",
    )
    parser.add_argument(
        "--rates_dir",
        type=str,
        default=str(RATES_DIR),
        help="Where to write the cached rate files.",
    )
    parser.add_argument(
        "--file_tag",
        type=str,
        default="",
        help="Optional tag appended to cached filenames (e.g. 'full' -> *_full.npz). Useful to avoid overwriting existing caches.",
    )
    parser.add_argument(
        "--eye_traces_path",
        type=str,
        default=str(EYE_TRACES_PATH),
        help="Path to cached eye traces (.npz).",
    )
    parser.add_argument(
        "--hires_threshold",
        type=float,
        default=HIRES_THRESHOLD_DEFAULT,
        help="Use hi-res pipeline for LogMAR < threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device ('cuda' or 'cpu'); default auto.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Frames per GPU batch when running the model.",
    )
    parser.add_argument(
        "--spatial_collapse",
        type=str,
        default="max",
        choices=("max", "mean"),
        help="How to collapse spatial readout maps to scalar rates.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    parser.add_argument(
        "--n_traces",
        type=int,
        default=None,
        help="Limit number of eye traces (debug/quick run).",
    )
    args = parser.parse_args(argv)

    file_tag = _normalize_file_tag(args.file_tag)

    logmars = _parse_csv_floats(args.logmars)
    orientations = _parse_csv_ints(args.orientations)
    conditions = _parse_csv_strings(args.conditions)

    if len(logmars) == 0:
        raise ValueError("No logmars parsed")
    if len(orientations) == 0:
        raise ValueError("No orientations parsed")
    if len(conditions) == 0:
        raise ValueError("No conditions parsed")

    rates_dir = Path(args.rates_dir)
    rates_dir.mkdir(parents=True, exist_ok=True)

    traces, durations = _load_eye_traces(Path(args.eye_traces_path))

    if args.n_traces is not None:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(traces), size=min(int(args.n_traces), len(traces)), replace=False)
        traces = traces[idx]
        durations = durations[idx]
        print(f"Using {len(traces)} traces (subsample)", flush=True)
    else:
        print(f"Using {len(traces)} traces", flush=True)

    model, readout = _load_model_and_readout(args.device)

    # Print excursion stats for any scaled or fixed_center conditions before caching
    from rate_computation import print_excursion_stats
    for condition in conditions:
        if condition.startswith('scaled_') or condition == 'fixed_center':
            print_excursion_stats(traces, durations, condition)

    # Compute in a deterministic, cache-friendly order
    for logmar in logmars:
        use_hires = float(logmar) < float(args.hires_threshold)
        print(f"\n=== LogMAR {logmar:.2f} ({'hi-res' if use_hires else 'lo-res'}) ===", flush=True)
        for condition in conditions:
            for orientation in orientations:
                out_path = _compute_and_save_one(
                    model=model,
                    readout=readout,
                    rates_dir=rates_dir,
                    logmar=float(logmar),
                    orientation=int(orientation),
                    condition=str(condition),
                    traces=traces,
                    durations=durations,
                    hires_threshold=float(args.hires_threshold),
                    batch_size=int(args.batch_size),
                    spatial_collapse=str(args.spatial_collapse),
                    force=bool(args.force),
                    file_tag=file_tag,
                )
                if out_path.exists():
                    print(f"    [ok] {out_path.name}", flush=True)

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
