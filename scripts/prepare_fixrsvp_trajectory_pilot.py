#!/usr/bin/env python3
"""Compatibility wrapper for pilot preparation entrypoint.

This intentionally forwards to scripts/run_fixrsvp_trajectory_pilot.py so
existing references keep working while clarifying that current behavior is
preparation and Stage 0 QC, not full Stage 1-7 analysis execution.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_fixrsvp_trajectory_pilot.py")), run_name="__main__")
