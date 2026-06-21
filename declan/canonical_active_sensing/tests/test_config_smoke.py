"""Smoke tests for canonical active-sensing configs."""
from __future__ import annotations

from pathlib import Path
from argparse import Namespace

import pytest

from declan.canonical_active_sensing._config import argv_from_args, load_config, require_fresh_output_paths, section_args
from declan.fixation_statistics_by_stimulus import make_backimage_aggregate_fem_figure_pack as aggregate_figures


def test_aggregate_config_renders_core_flags() -> None:
    config = load_config(Path("declan/canonical_active_sensing/configs/aggregate_fem_k16_v1.json"))
    args = section_args(config, "aggregate_fem")
    argv = argv_from_args(args)
    assert "--latent-names" in argv
    assert "pyramid_local_field" in argv
    assert "--pca-k-list" in argv
    assert "16" in argv
    assert "--reuse-trace-sources-across-scales" in argv


def test_feature_adjudication_config_has_two_joint_dirs() -> None:
    config = load_config(Path("declan/canonical_active_sensing/configs/feature_adjudication_v1.json"))
    args = section_args(config, "feature_adjudication")
    assert "backimage_axis_conditioned_hard_negative_n128_rel0p25" in args["joint_feature_dirs"]
    assert "backimage_axis_conditioned_hard_negative_n128_scale_sweep" in args["joint_feature_dirs"]


def test_figure4_power_config_locks_primary_model_and_seed_replicates() -> None:
    config = load_config(Path("declan/canonical_active_sensing/configs/figure4_power_rerun_v1.json"))
    aggregate = section_args(config, "aggregate_power_primary")
    local_seed7 = section_args(config, "local_pairing_power_seed7")
    local_seed11 = section_args(config, "local_pairing_power_seed11")

    assert aggregate["latent_names"] == "pyramid_local_field"
    assert aggregate["pca_k_list"] == "16"
    assert aggregate["max_images"] == 384
    assert aggregate["trace_samples_per_condition"] == 8
    assert aggregate["twin_trace_batch_size"] == 8
    assert local_seed7["unpaired_samples_per_image"] == 64
    assert local_seed11["unpaired_samples_per_image"] == 64
    assert local_seed7["seed"] != local_seed11["seed"]


def test_figure_pack_incremental_dir_is_not_nested_under_run_dir() -> None:
    config = load_config(Path("declan/canonical_active_sensing/configs/figure_active_sensing_v1.json"))
    args = section_args(config, "aggregate_figure_pack")
    aggregate_figures._configure_from_args(Namespace(**args))
    resolved = aggregate_figures._resolve_incremental_dir(Path(args["run_dir"]))
    assert resolved == Path(args["incremental_dir"])


def test_fresh_output_guard_refuses_non_empty_output(tmp_path: Path) -> None:
    out_dir = tmp_path / "existing_output"
    out_dir.mkdir()
    (out_dir / "run_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        require_fresh_output_paths({"out_dir": out_dir})


def test_fresh_output_guard_allows_empty_output(tmp_path: Path) -> None:
    out_dir = tmp_path / "empty_output"
    out_dir.mkdir()
    require_fresh_output_paths({"out_dir": out_dir})
