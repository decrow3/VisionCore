#!/usr/bin/env python3
"""Orchestrate a balanced long BackImage RR100 SF x contour-alignment run."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_contour_axis_rr100_sf_contour_alignment_long_axis30_n576_low0p05_high0p5_v1"
)
DEFAULT_SF_GROUPS_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
CONDITION_PAIRS = ",".join(
    [
        "0:0",
        "0:0.25",
        "0:0.5",
        "0:1",
        "0:2",
        "0.25:0",
        "0.25:0.25",
        "0.25:0.5",
        "0.25:1",
        "0.25:2",
        "0.5:0",
        "0.5:0.25",
        "0.5:1",
        "0.5:2",
        "1:0",
        "1:0.125",
        "1:0.25",
        "1:0.5",
        "1:0.75",
        "1:1",
        "1:1.5",
        "1:2",
        "1:3",
        "2:0",
        "2:0.25",
        "2:0.5",
        "2:1",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--axis-bin-width-deg", type=float, default=30.0)
    parser.add_argument("--target-per-bin", type=int, default=96)
    parser.add_argument("--energy-balance-column", type=str, default="image_oriented_gradient_energy")
    parser.add_argument("--energy-quantile-bins", type=int, default=2)
    parser.add_argument("--manifest-max-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-units", type=int, default=12)
    parser.add_argument(
        "--stimulus-rotation-deg",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="Whole-movie rotation control passed to the contour SSI runner.",
    )
    parser.add_argument("--sf-groups-csv", type=Path, default=DEFAULT_SF_GROUPS_CSV)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Build manifest and contour inventory only.")
    parser.add_argument("--force-contour", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--skip-contour", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--print-only", action="store_true", help="Write command manifest without executing stages.")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str], print_only: bool) -> None:
    print(" ".join(cmd), flush=True)
    if print_only:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def population_plot_commands(args: argparse.Namespace, contour_dir: Path, out_root: Path) -> list[list[str]]:
    specs = [
        ("population_across_sweep_along0", "across", "--fixed-along-scale", "0"),
        ("population_across_sweep_along1", "across", "--fixed-along-scale", "1"),
        ("population_along_sweep_across0", "along", "--fixed-across-scale", "0"),
        ("population_along_sweep_across1", "along", "--fixed-across-scale", "1"),
    ]
    commands: list[list[str]] = []
    for out_name, sweep_axis, fixed_flag, fixed_value in specs:
        commands.append(
            [
                sys.executable,
                "declan/active_sensing_movie_information/plot_backimage_rr100_sf_contour_alignment_population_ssi.py",
                "--contour-run-dir",
                str(contour_dir),
                "--sf-groups-csv",
                str(Path(args.sf_groups_csv)),
                "--out-dir",
                str(out_root / out_name),
                "--ssi-metric",
                "time_resolved",
                "--sf-groups",
                "low_sf,high_sf",
                "--sweep-axis",
                sweep_axis,
                fixed_flag,
                fixed_value,
                "--n-bootstrap",
                str(int(args.n_bootstrap)),
            ]
        )
    return commands


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    source_dir = out_root / "balanced_source_windows"
    contour_dir = out_root / "contour_rr100_spatial_ssi_pairs27"
    stratified_dir = out_root / "orientation_stratified_population"
    out_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    manifest_cmd = [
        sys.executable,
        "declan/active_sensing_movie_information/make_backimage_contour_axis_balanced_window_manifest.py",
        "--out-dir",
        str(source_dir),
        "--axis-bin-width-deg",
        str(float(args.axis_bin_width_deg)),
        "--target-per-bin",
        str(int(args.target_per_bin)),
        "--energy-balance-column",
        str(args.energy_balance_column),
        "--energy-quantile-bins",
        str(int(args.energy_quantile_bins)),
        "--seed",
        str(int(args.seed)),
    ]
    if int(args.manifest_max_windows) > 0:
        manifest_cmd.extend(["--max-windows", str(int(args.manifest_max_windows))])

    contour_cmd = [
        sys.executable,
        "declan/active_sensing_movie_information/run_backimage_contour_axis_rr100_spatial_ssi.py",
        "--axis-run-dir",
        str(source_dir),
        "--trial-source-mode",
        "selected_windows",
        "--out-dir",
        str(contour_dir),
        "--sweep-mode",
        "pairs",
        "--condition-pairs",
        CONDITION_PAIRS,
        "--max-trials",
        "0",
        "--primary-ssi-metric",
        "time_resolved",
        "--device",
        str(args.device),
        "--batch-size",
        str(int(args.batch_size)),
        "--top-units",
        str(int(args.top_units)),
        "--stimulus-rotation-deg",
        str(int(args.stimulus_rotation_deg)),
        "--write-zscore-plot",
    ]
    if bool(args.force_contour):
        contour_cmd.append("--force")
    if bool(args.dry_run):
        contour_cmd.append("--dry-run")

    plot_cmds = population_plot_commands(args, contour_dir, out_root)
    stratified_cmd = [
        sys.executable,
        "declan/active_sensing_movie_information/plot_backimage_rr100_orientation_stratified_population_ssi.py",
        "--out-dir",
        str(stratified_dir),
        "--across-along0-dir",
        str(out_root / "population_across_sweep_along0"),
        "--across-along1-dir",
        str(out_root / "population_across_sweep_along1"),
        "--along-across0-dir",
        str(out_root / "population_along_sweep_across0"),
        "--along-across1-dir",
        str(out_root / "population_along_sweep_across1"),
        "--band-mode",
        "dominant_and_coarse30",
        "--coarse-bin-width-deg",
        "30",
        "--min-fixations-per-band",
        "20",
        "--n-bootstrap",
        str(int(args.n_bootstrap)),
    ]

    command_manifest = {
        "analysis": "backimage_rr100_sf_contour_alignment_long",
        "out_root": out_root,
        "balanced_source_dir": source_dir,
        "contour_dir": contour_dir,
        "population_dirs": {
            "across_along0": out_root / "population_across_sweep_along0",
            "across_along1": out_root / "population_across_sweep_along1",
            "along_across0": out_root / "population_along_sweep_across0",
            "along_across1": out_root / "population_along_sweep_across1",
        },
        "orientation_stratified_dir": stratified_dir,
        "condition_pairs": CONDITION_PAIRS,
        "stimulus_rotation_deg": int(args.stimulus_rotation_deg),
        "commands": {
            "manifest": manifest_cmd,
            "contour": contour_cmd,
            "population_plots": plot_cmds,
            "orientation_stratified": stratified_cmd,
        },
        "dry_run": bool(args.dry_run),
    }
    write_json(out_root / "long_run_commands.json", command_manifest)

    if not bool(args.skip_manifest):
        run_command(manifest_cmd, cwd=ROOT, env=env, print_only=bool(args.print_only))
    if not bool(args.skip_contour):
        run_command(contour_cmd, cwd=ROOT, env=env, print_only=bool(args.print_only))
    if bool(args.dry_run) or bool(args.skip_plots):
        print(f"Wrote command manifest: {out_root / 'long_run_commands.json'}", flush=True)
        return
    for cmd in plot_cmds:
        run_command(cmd, cwd=ROOT, env=env, print_only=bool(args.print_only))
    run_command(stratified_cmd, cwd=ROOT, env=env, print_only=bool(args.print_only))
    print(f"Long run complete: {out_root}", flush=True)


if __name__ == "__main__":
    main()
