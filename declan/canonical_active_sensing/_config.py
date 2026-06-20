"""Small helpers for config-driven canonical wrappers."""
from __future__ import annotations

import argparse
import importlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return payload


def section_args(config: dict[str, Any], section: str) -> dict[str, Any]:
    if section not in config:
        available = ", ".join(sorted(config))
        raise KeyError(f"Missing config section {section!r}. Available: {available}")
    block = config[section]
    if not isinstance(block, dict):
        raise ValueError(f"Config section {section!r} must be an object")
    args = block.get("args", block)
    if not isinstance(args, dict):
        raise ValueError(f"Config section {section!r} args must be an object")
    return dict(args)


def argv_from_args(args: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            text = ",".join(str(item) for item in value)
        else:
            text = str(value)
        argv.extend([flag, text])
    return argv


def print_command(module_name: str, args: dict[str, Any]) -> None:
    rendered = " ".join(shlex.quote(part) for part in [sys.executable, "-m", module_name, *argv_from_args(args)])
    print(rendered)


def _path_has_entries(path: Path) -> bool:
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def require_fresh_output_paths(
    args: dict[str, Any],
    *,
    keys: tuple[str, ...] = ("out_dir",),
    allow_existing: bool = False,
) -> None:
    """Refuse to run canonical jobs into non-empty output paths by default."""
    if allow_existing:
        return
    blockers: list[str] = []
    for key in keys:
        value = args.get(key)
        if value is None:
            continue
        path = Path(value)
        if not path.exists():
            continue
        if path.is_dir():
            if _path_has_entries(path):
                blockers.append(f"{key}={path} (non-empty directory)")
        else:
            blockers.append(f"{key}={path} (existing file)")
    if blockers:
        joined = "\n".join(f"- {item}" for item in blockers)
        raise FileExistsError(
            "Refusing to run canonical job because output path(s) already exist.\n"
            f"{joined}\n"
            "Choose a new output path, move/archive the existing output, or pass "
            "--allow-existing-output when you intentionally want to refresh it."
        )


def enforce_fresh_output_paths(
    args: dict[str, Any],
    *,
    keys: tuple[str, ...] = ("out_dir",),
    allow_existing: bool = False,
) -> None:
    try:
        require_fresh_output_paths(args, keys=keys, allow_existing=allow_existing)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from None


def run_existing_main(module_name: str, args: dict[str, Any], *, print_only: bool = False) -> None:
    if print_only:
        print_command(module_name, args)
        return
    module = importlib.import_module(module_name)
    old_argv = sys.argv[:]
    sys.argv = [module_name, *argv_from_args(args)]
    try:
        module.main()
    finally:
        sys.argv = old_argv


def add_common_wrapper_args(parser: argparse.ArgumentParser, *, default_config: Path, default_section: str) -> None:
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--section", default=default_section)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into an existing non-empty output directory. Use only for intentional cache refreshes.",
    )
