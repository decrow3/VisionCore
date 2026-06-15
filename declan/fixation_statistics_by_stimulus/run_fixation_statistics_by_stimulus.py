#!/usr/bin/env python3
"""Compare fixation eye-movement statistics across stimulus regimes.

Smoke example:

    uv run python -m declan.fixation_statistics_by_stimulus.run_fixation_statistics_by_stimulus \
      --sessions Allen_2022-04-13 --max-windows-per-stimulus 200 \
      --out-dir outputs/fixation_statistics_by_stimulus_smoke
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .classifier import classify_stimulus_from_windows
from .extraction import ExtractionConfig, extract_session_stimulus
from .io_utils import parse_csv_list, write_csv, write_json
from .plots import save_qc_plots
from .summaries import paired_metric_contrasts, summarize_events, summarize_windows


DEFAULT_OUT_DIR = Path("outputs") / "fixation_statistics_by_stimulus"
DEFAULT_STIMULI = ("fixrsvp", "backimage", "gaborium", "gratings")


def _session_name(session: Any) -> str:
    return str(getattr(session, "name", session))


def load_sessions(raw: str) -> list[Any]:
    from DataYatesV1 import get_complete_sessions, get_session

    if str(raw).strip().lower() in {"all", "*"}:
        return list(get_complete_sessions())
    sessions = []
    available = {_session_name(s): s for s in get_complete_sessions()}
    for name in parse_csv_list(raw):
        if name in available:
            sessions.append(available[name])
            continue
        if "_" not in name:
            raise ValueError(f"Session must be Subject_YYYY-MM-DD, got {name!r}")
        subject, date = name.split("_", 1)
        sessions.append(get_session(subject, date))
    return sessions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", default="Allen_2022-04-13", help="Comma-separated session names or 'all'.")
    parser.add_argument("--stimuli", default=",".join(DEFAULT_STIMULI), help="Comma-separated dataset stimuli.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--window-samples", type=int, default=128)
    parser.add_argument("--stride-samples", type=int, default=16)
    parser.add_argument("--min-epoch-samples", type=int, default=24)
    parser.add_argument("--min-valid-fraction", type=float, default=0.0)
    parser.add_argument("--fixation-radius-deg", type=float, default=1.0)
    parser.add_argument("--max-abs-eye-deg", type=float, default=12.0)
    parser.add_argument("--speed-z", type=float, default=6.0)
    parser.add_argument("--event-pad-samples", type=int, default=1)
    parser.add_argument("--early-s", type=float, default=0.10)
    parser.add_argument("--mid-s", type=float, default=0.30)
    parser.add_argument("--max-windows-per-stimulus", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--baseline-stimulus", default="fixrsvp")
    parser.add_argument("--include-image-features", action="store_true")
    parser.add_argument("--image-patch-radius-deg", type=float, default=1.0)
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(args.sessions)
    stimuli = parse_csv_list(args.stimuli)
    cfg = ExtractionConfig(
        dt=float(args.dt),
        window_samples=int(args.window_samples),
        stride_samples=int(args.stride_samples),
        min_epoch_samples=int(args.min_epoch_samples),
        min_valid_fraction=float(args.min_valid_fraction),
        fixation_radius_deg=float(args.fixation_radius_deg),
        max_abs_eye_deg=float(args.max_abs_eye_deg),
        speed_z=float(args.speed_z),
        event_pad_samples=int(args.event_pad_samples),
        early_s=float(args.early_s),
        mid_s=float(args.mid_s),
        max_windows_per_stimulus=int(args.max_windows_per_stimulus),
        seed=int(args.seed),
        include_image_features=bool(args.include_image_features),
        image_patch_radius_deg=float(args.image_patch_radius_deg),
    )

    window_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for session in tqdm(sessions, desc="sessions"):
        for stimulus in stimuli:
            rows, events, inventory = extract_session_stimulus(session, stimulus, cfg)
            window_rows.extend(rows)
            event_rows.extend(events)
            inventory_rows.append(inventory)

    session_summary = summarize_windows(window_rows)
    event_summary = summarize_events(event_rows, inventory_rows)
    contrast_rows = paired_metric_contrasts(
        session_summary,
        baseline=str(args.baseline_stimulus),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    classifier_rows = [] if args.skip_classifier else classify_stimulus_from_windows(window_rows, seed=int(args.seed))
    figures = [] if args.skip_plots else save_qc_plots(out_dir, window_rows)

    write_csv(out_dir / "window_features.csv", window_rows)
    write_csv(out_dir / "saccade_event_features.csv", event_rows)
    write_csv(out_dir / "session_stimulus_inventory.csv", inventory_rows)
    write_csv(out_dir / "session_phase_summary.csv", session_summary)
    write_csv(out_dir / "saccade_event_summary.csv", event_summary)
    write_csv(out_dir / "paired_metric_contrasts.csv", contrast_rows)
    write_csv(out_dir / "classifier_summary.csv", classifier_rows)
    write_json(out_dir / "run_metadata.json", {
        "sessions": [_session_name(s) for s in sessions],
        "stimuli": stimuli,
        "config": asdict(cfg),
        "event_detector": "jake.twininfo.eye_controls.detect_microsaccade_events",
        "event_detector_note": (
            "Windows are contiguous valid samples after removing operational high-speed events. "
            "This is not the full jake.detect_saccades asymmetric-Gaussian raw-DPI detector."
        ),
        "n_window_rows": len(window_rows),
        "n_event_rows": len(event_rows),
        "figures": figures,
    })
    print(f"Wrote {len(window_rows)} window rows and {len(event_rows)} event rows to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
