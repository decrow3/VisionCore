"""
Compatibility shim: re-exports DualWindowAnalysis from scripts/mcfarland_sim.py.

The canonical functional API lives in VisionCore.covariance. This module exists
so that scripts importing `from VisionCore.dual_window import DualWindowAnalysis`
continue to work without modification.
"""
import sys
from pathlib import Path

# scripts/ is not a package, so we need it on sys.path to import mcfarland_sim.
_scripts_dir = str(Path(__file__).parent.parent / 'scripts')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from mcfarland_sim import DualWindowAnalysis  # noqa: F401
