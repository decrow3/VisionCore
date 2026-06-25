"""Run the Figure 4C affine-x1000 continuous-joint observer/control.

This wrapper records the current feature-primary affine candidate and its
intercept-ablation control. The normal candidate uses the guarded affine
quadratic observation model with intercept ridge multiplier 1000. Setting
``--intercept-scale 0`` fits the same affine maps but removes the intercept
term during trajectory profiling and Poisson scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from declan.backimage_trajectory_observer.analyze_continuous_joint_trajectory import (
    analyze,
    build_parser as build_analyzer_parser,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1"
)
COMPACT_BASIS = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_disjoint_compact_basis_delta025_v1"
    / "image_disjoint_compact_basis_delta0p25_fold0of2.npz"
)
DEFAULT_OUT_DIR = SOURCE_ROOT / "continuous_joint_quadratic_affine_poisson_scale_conditioned_interceptx1000_full"
DEFAULT_ABLATION_OUT_DIR = (
    SOURCE_ROOT / "continuous_joint_quadratic_affine_poisson_scale_conditioned_interceptx1000_ablate_intercept_full"
)


AFFINE_ANALYZER_ARGS = [
    "--run-dir",
    str(SOURCE_ROOT),
    "--response-manifest",
    str(SOURCE_ROOT / "response_cache_manifest.csv"),
    "--compact-basis-path",
    str(COMPACT_BASIS),
    "--basis-key",
    "basis",
    "--basis-max-dim",
    "10",
    "--basis-max-dim-by-scale",
    "0.5:10,1.0:20,2.0:20",
    "--ridge",
    "0.01",
    "--ridge-by-scale",
    "0.5:0.01,1.0:0.1,2.0:0.1",
    "--continuous-score-mode",
    "quadratic_affine_poisson_profile",
    "--quadratic-intercept-ridge-multiplier",
    "1000",
    "--trajectory-sidecar-dir",
    str(SOURCE_ROOT / "continuous_joint_trajectory_sidecars"),
    "--trajectory-initial-position",
    "inferred",
    "--trajectory-initial-position-var",
    "0.0001",
    "--quadratic-optimizer-max-iter",
    "80",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for analyzer artifacts. Defaults to the full affine x1000 "
            "directory for intercept-scale 1, and the full ablation directory for intercept-scale 0."
        ),
    )
    parser.add_argument(
        "--intercept-scale",
        type=float,
        default=1.0,
        help="Non-negative affine intercept score scale; use 0 for the direct intercept ablation.",
    )
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
    intercept_scale = float(args.intercept_scale)
    if intercept_scale < 0.0:
        raise ValueError("--intercept-scale must be non-negative")
    default_out = DEFAULT_ABLATION_OUT_DIR if intercept_scale == 0.0 else DEFAULT_OUT_DIR
    out_dir = default_out if args.out_dir is None else Path(args.out_dir)
    argv = [
        *AFFINE_ANALYZER_ARGS,
        "--quadratic-affine-intercept-scale",
        f"{intercept_scale:g}",
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
