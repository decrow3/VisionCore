#!/usr/bin/env python3
"""Generate the initial Figure 4 active-sensing figure.

This is deliberately a thin wrapper around the current
``active_sensing_movie_information`` figure generator.  It gives the Figure 4
active-sensing branch its own stable entry point and output directory while the
figure design is still identical to the existing active-sensing movie
information figure.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

from declan.active_sensing_movie_information.generate_active_sensing_movie_information_figure import (
    DEFAULT_RUN,
    make_figure as make_active_sensing_movie_information_figure,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "fig4_active_sensing" / "active_sensing_movie_information_figure"


def make_figure(run_dir: Path = DEFAULT_RUN, out_dir: Path = DEFAULT_OUT) -> dict[str, object]:
    """Create the initial Figure 4 active-sensing figure from cached outputs."""
    return make_active_sensing_movie_information_figure(run_dir=run_dir, out_dir=out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = make_figure(run_dir=args.run_dir, out_dir=args.out_dir)
    print(f"Wrote {stats['outputs']['pdf']}")
    print(f"Wrote {stats['outputs']['png']}")
    print(f"Wrote {stats['outputs']['svg']}")


if __name__ == "__main__":
    main()
