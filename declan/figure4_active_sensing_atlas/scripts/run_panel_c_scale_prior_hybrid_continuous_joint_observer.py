"""Run the predeclared Figure 4C scale-specific trajectory-prior candidate.

This is the source rerun of the diagnostic posterior-row hybrid: known-start
AR(1) at 0.5x/1.0x and known-start matched-Brownian covariance scale 8 at 2.0x.
"""

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


DEFAULT_OUT_DIR = SOURCE_ROOT / "continuous_joint_quadratic_poisson_scale_conditioned_knownstart_scale_prior_hybrid_predeclared_full"


SCALE_PRIOR_ANALYZER_ARGS = [
    *KNOWNSTART_ANALYZER_ARGS,
    "--trajectory-process-model-by-scale",
    "0.5:ar1,1.0:ar1,2.0:matched_brownian",
    "--brownian-cov-scale-by-scale",
    "2.0:8",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-tables",
        type=int,
        default=0,
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
    argv = [*SCALE_PRIOR_ANALYZER_ARGS, "--out-dir", str(Path(args.out_dir))]
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
