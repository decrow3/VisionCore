"""Run a known-start quadratic observer with a matched Brownian trace prior."""

from __future__ import annotations

import argparse
from pathlib import Path

from declan.backimage_trajectory_observer.analyze_continuous_joint_trajectory import (
    analyze,
    build_parser as build_analyzer_parser,
)
from declan.figure4_active_sensing_atlas.scripts.run_panel_c_knownstart_continuous_joint_observer import (
    KNOWNSTART_ANALYZER_ARGS,
    SOURCE_ROOT,
)


def _scale_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brownian-cov-scale", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--max-tables",
        type=int,
        default=64,
        help="Optional response-table limit for smoke runs; 0 runs the full cache.",
    )
    parser.add_argument("--skip-tables", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the analyzer arguments without running the analyzer.",
    )
    return parser


def analyzer_argv(args: argparse.Namespace) -> list[str]:
    scale = float(args.brownian_cov_scale)
    if scale <= 0.0:
        raise ValueError("--brownian-cov-scale must be positive")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else SOURCE_ROOT
        / f"continuous_joint_quadratic_poisson_scale_conditioned_knownstart_matched_brownian_scale{_scale_slug(scale)}_smoke64"
    )
    argv = [
        *KNOWNSTART_ANALYZER_ARGS,
        "--trajectory-process-model",
        "matched_brownian",
        "--brownian-cov-scale",
        f"{scale:g}",
        "--out-dir",
        str(out_dir),
    ]
    if int(args.max_tables) > 0:
        argv.extend(["--max-tables", str(int(args.max_tables))])
    if int(args.skip_tables) > 0:
        argv.extend(["--skip-tables", str(int(args.skip_tables))])
    argv.extend(["--progress-every", str(max(1, int(args.progress_every)))])
    return argv


def main() -> None:
    args = build_parser().parse_args()
    argv = analyzer_argv(args)
    if args.dry_run:
        print("python -m declan.backimage_trajectory_observer.analyze_continuous_joint_trajectory")
        for item in argv:
            print(item)
        return
    analyzer_args = build_analyzer_parser().parse_args(argv)
    analyze(analyzer_args)


if __name__ == "__main__":
    main()
