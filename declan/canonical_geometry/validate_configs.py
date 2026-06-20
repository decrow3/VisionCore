"""Validate canonical geometry JSON configs."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from ._config import argv_from_args, enforce_fresh_output_paths, load_config, print_command, section_args


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_MODULES = {
    "raw_edge_v1.json": {
        "raw_edge_audit": "declan.fixation_statistics_by_stimulus.analyze_backimage_raw_edge_roadblock",
    },
    "figure_geometry_v1.json": {
        "geometry_figure_pack": "declan.canonical_geometry.make_geometry_figure_pack",
    },
}
INPUT_PATH_KEYS = (
    "windows_csv",
    "stability_window_csv",
    "feature_preservation_window_csv",
    "observer_trials_csv",
    "feature_posterior_trials_csv",
    "feature_axis_contrasts_csv",
    "atlas_dir",
    "matched_axis_dir",
    "hardneg_axis_dir",
    "stability_dir",
    "objective_dir",
    "alignment_dir",
    "window_dir",
    "raw_edge_audit_dir",
)


def _path_values(value: object) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, Path):
        return [value]
    if isinstance(value, (list, tuple)):
        return [Path(str(item)) for item in value if str(item)]
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]


def _validate_against_parser(module_name: str | None, args_map: dict[str, object]) -> None:
    if module_name is None:
        return
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
        config = load_config(PACKAGE_DIR / "configs" / filename)
        for section, module_name in sections.items():
            args_map = section_args(config, section)
            _validate_against_parser(module_name, args_map)
            if args.check_inputs:
                _validate_input_paths(filename, section, args_map)
            if args.check_output_freshness:
                enforce_fresh_output_paths(args_map)
            if args.print_commands:
                if module_name is None:
                    print(f"placeholder {filename}:{section}")
                else:
                    print_command(module_name, args_map)
            else:
                print(f"ok {filename}:{section}")


if __name__ == "__main__":
    main()
