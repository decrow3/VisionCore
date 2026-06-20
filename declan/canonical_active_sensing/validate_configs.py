"""Validate canonical active-sensing JSON configs and render commands."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from ._config import argv_from_args, enforce_fresh_output_paths, load_config, print_command, section_args


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_MODULES = {
    "aggregate_fem_k16_v1.json": {
        "aggregate_fem": "declan.fixation_statistics_by_stimulus.run_backimage_aggregate_fem_information",
        "aggregate_incremental": "declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion",
    },
    "local_pairing_k16_v1.json": {
        "local_pairing_primary": "declan.fixation_statistics_by_stimulus.run_backimage_local_pairing_Iz_revisit",
        "local_pairing_sentinel": "declan.fixation_statistics_by_stimulus.run_backimage_local_pairing_Iz_revisit",
        "local_incremental_primary": "declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion",
        "local_incremental_sentinel": "declan.fixation_statistics_by_stimulus.summarize_backimage_aggregate_incremental_motion",
    },
    "joint_posterior_k16_v1.json": {
        "observer_rel0p25": "declan.fixation_statistics_by_stimulus.run_backimage_trajectory_table_observer",
        "posterior_rel0p25": "declan.backimage_trajectory_observer.analyze_feature_posterior",
    },
    "feature_adjudication_v1.json": {
        "feature_adjudication": "declan.fixation_statistics_by_stimulus.analyze_backimage_feature_decomposition_adjudication",
    },
    "figure_active_sensing_v1.json": {
        "aggregate_figure_pack": "declan.fixation_statistics_by_stimulus.make_backimage_aggregate_fem_figure_pack",
    },
}
INPUT_PATH_KEYS = (
    "input",
    "window_manifest",
    "run_dir",
    "feature_npz",
    "aggregate_run_dir",
    "aggregate_incremental_dir",
    "local_pairing_dirs",
    "joint_feature_dirs",
)


def _path_values(value: object) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, Path):
        return [value]
    if isinstance(value, (list, tuple)):
        return [Path(str(item)) for item in value if str(item)]
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]


def _validate_against_parser(module_name: str, args_map: dict[str, object]) -> None:
    module = importlib.import_module(module_name)
    build_parser = getattr(module, "build_parser", None)
    if build_parser is None:
        return
    build_parser().parse_args(argv_from_args(args_map))


def _validate_input_paths(filename: str, section: str, args_map: dict[str, object]) -> None:
    missing: list[str] = []
    for key in INPUT_PATH_KEYS:
        if key not in args_map:
            continue
        for path in _path_values(args_map.get(key)):
            if not path.exists():
                missing.append(f"{key}={path}")
    if missing:
        joined = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Missing prerequisite path(s) in {filename}:{section}\n{joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--check-inputs", action="store_true", help="Require referenced prerequisite paths to exist.")
    parser.add_argument("--check-output-freshness", action="store_true", help="Fail if configured out_dir paths are non-empty.")
    args = parser.parse_args()
    for filename, sections in CONFIG_MODULES.items():
        config_path = PACKAGE_DIR / "configs" / filename
        config = load_config(config_path)
        for section, module_name in sections.items():
            args_map = section_args(config, section)
            _validate_against_parser(module_name, args_map)
            if args.check_inputs:
                _validate_input_paths(filename, section, args_map)
            if args.check_output_freshness:
                enforce_fresh_output_paths(args_map)
            if args.print_commands:
                print_command(module_name, args_map)
            else:
                print(f"ok {filename}:{section}")


if __name__ == "__main__":
    main()
