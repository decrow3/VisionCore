#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from VisionCore.paths import STATS_DIR


DEFAULT_LOGMARS = (-0.35, -0.30, -0.20, 0.00, 0.20, 0.40, 0.60)
DEFAULT_ORIENTATIONS = (0, 90, 180, 270)
DEFAULT_OUTPUT_ROOT = STATS_DIR / "fem_step_jacobian_regime_matrix"
DEFAULT_REGIMES = (
    ("drift_lte_1p0", ["--max-step-arcmin=1.0"]),
    ("intermediate_lte_1p5", ["--max-step-arcmin=1.5"]),
    ("all_steps", []),
    ("large_gte_2p0", ["--min-step-arcmin=2.0"]),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the expanded FEM step-Jacobian regime matrix.")
    parser.add_argument("--logmars", default=",".join(f"{x:.2f}" for x in DEFAULT_LOGMARS))
    parser.add_argument("--orientations", default=",".join(str(x) for x in DEFAULT_ORIENTATIONS))
    parser.add_argument("--max-traces", type=int, default=10)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--max-steps-per-trace", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        "scripts/fem_step_jacobian_prediction.py",
        f"--logmars={args.logmars}",
        f"--orientations={args.orientations}",
        f"--max-traces={int(args.max_traces)}",
        f"--step-stride={int(args.step_stride)}",
        f"--max-steps-per-trace={int(args.max_steps_per_trace)}",
        f"--bootstrap-samples={int(args.bootstrap_samples)}",
        f"--bootstrap-seed={int(args.bootstrap_seed)}",
    ]
    if args.skip_figures:
        common.append("--skip-figures")

    for regime_label, extra_args in DEFAULT_REGIMES:
        output_dir = output_root / regime_label
        command = common + extra_args + [f"--output-dir={output_dir}"]
        print("Running:", " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)

    summary_command = [
        sys.executable,
        "scripts/fem_step_jacobian_regime_summary.py",
        f"--output-dir={output_root / 'summary'}",
        f"--bootstrap-samples={int(args.bootstrap_samples)}",
        f"--bootstrap-seed={int(args.bootstrap_seed)}",
    ]
    for regime_label, _ in DEFAULT_REGIMES:
        summary_command.append(f"--run={regime_label}={output_root / regime_label}")
    print("Running:", " ".join(summary_command), flush=True)
    if not args.dry_run:
        subprocess.run(summary_command, check=True)


if __name__ == "__main__":
    main()