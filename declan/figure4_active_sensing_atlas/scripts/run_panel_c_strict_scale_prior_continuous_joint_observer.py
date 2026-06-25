"""Run the strict inferred-start Figure 4C scale-specific prior candidate.

This tests whether the scale-specific AR(1)/matched-Brownian trajectory prior
can improve the no-start endpoint, not just the less-strict known-start
candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from declan.backimage_trajectory_observer.analyze_continuous_joint_trajectory import (
    analyze,
    build_parser as build_analyzer_parser,
)
from declan.figure4_active_sensing_atlas.scripts.run_panel_c_promoted_continuous_joint_observer import (
    PROMOTED_ANALYZER_ARGS,
    SOURCE_ROOT,
)


DEFAULT_OUT_DIR = SOURCE_ROOT / "continuous_joint_quadratic_poisson_scale_conditioned_strict_scale_prior_predeclared_full"


STRICT_SCALE_PRIOR_ANALYZER_ARGS = [
    *PROMOTED_ANALYZER_ARGS,
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
    argv = [*STRICT_SCALE_PRIOR_ANALYZER_ARGS, "--out-dir", str(Path(args.out_dir))]
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
