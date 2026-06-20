"""Smoke tests for canonical geometry configs."""
from __future__ import annotations

from pathlib import Path

import pytest

from declan.canonical_geometry._config import argv_from_args, load_config, require_fresh_output_paths, section_args


def test_raw_edge_config_renders_core_flags() -> None:
    config = load_config(Path("declan/canonical_geometry/configs/raw_edge_v1.json"))
    args = section_args(config, "raw_edge_audit")
    argv = argv_from_args(args)
    assert "--windows-csv" in argv
    assert "--feature-posterior-trials-csv" in argv
    assert "--out-dir" in argv


def test_geometry_figure_pack_config_renders_core_flags() -> None:
    config = load_config(Path("declan/canonical_geometry/configs/figure_geometry_v1.json"))
    args = section_args(config, "geometry_figure_pack")
    argv = argv_from_args(args)
    assert "--matched-axis-dir" in argv
    assert "--alignment-dir" in argv
    assert "--raw-edge-audit-dir" in argv
    assert any("backimage_raw_edge_roadblock_residual_adjudication_canonical_v1" in item for item in argv)


def test_fresh_output_guard_refuses_non_empty_output(tmp_path: Path) -> None:
    out_dir = tmp_path / "existing_output"
    out_dir.mkdir()
    (out_dir / "figure_pack_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        require_fresh_output_paths({"out_dir": out_dir})
