"""Run the Figure 4C joint decoder with a synthetic empirical confined prior.

This mirrors the promoted inferred-start scale-specific observer, but replaces
the 2.0x matched-Brownian process prior with synthetic_empirical_confined.  In
that mode, held-out training FEM traces calibrate a synthetic confined-step
trajectory generator; the continuous decoder then fits its path prior to the
generated traces, not to empirical trajectory replay.
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


DEFAULT_OUT_DIR = (
    SOURCE_ROOT / "continuous_joint_quadratic_poisson_scale_conditioned_synthetic_empirical_confined_predeclared_full"
)

SYNTHETIC_PROCESS_BY_SCALE = "0.5:ar1,1.0:ar1,2.0:synthetic_empirical_confined"


def _replace_option(argv: list[str], option: str, value: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == option:
            skip_next = True
            continue
        out.append(item)
    out.extend([option, value])
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--synthetic-prior-samples", type=int, default=512)
    parser.add_argument("--synthetic-prior-kappa-weight-power", type=float, default=0.5)
    parser.add_argument("--synthetic-prior-seed", type=int, default=0)
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
    if int(args.synthetic_prior_samples) <= 0:
        raise ValueError("--synthetic-prior-samples must be positive")
    argv = _replace_option(
        [*PROMOTED_ANALYZER_ARGS],
        "--trajectory-process-model-by-scale",
        SYNTHETIC_PROCESS_BY_SCALE,
    )
    argv.extend(
        [
            "--synthetic-prior-samples",
            str(int(args.synthetic_prior_samples)),
            "--synthetic-prior-kappa-weight-power",
            f"{float(args.synthetic_prior_kappa_weight_power):g}",
            "--synthetic-prior-seed",
            str(int(args.synthetic_prior_seed)),
            "--out-dir",
            str(Path(args.out_dir)),
        ]
    )
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
