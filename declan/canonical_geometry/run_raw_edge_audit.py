"""Run the canonical BackImage raw-edge roadblock audit from config."""
from __future__ import annotations

import argparse
from pathlib import Path

from ._config import add_common_wrapper_args, enforce_fresh_output_paths, load_config, run_existing_main, section_args


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "raw_edge_v1.json"
TARGET_MODULE = "declan.fixation_statistics_by_stimulus.analyze_backimage_raw_edge_roadblock"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_wrapper_args(parser, default_config=DEFAULT_CONFIG, default_section="raw_edge_audit")
    args = parser.parse_args()
    config = load_config(args.config)
    run_args = section_args(config, args.section)
    if not args.print_command:
        enforce_fresh_output_paths(run_args, allow_existing=bool(args.allow_existing_output))
    run_existing_main(TARGET_MODULE, run_args, print_only=bool(args.print_command))


if __name__ == "__main__":
    main()
