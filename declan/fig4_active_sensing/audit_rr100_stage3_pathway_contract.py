#!/usr/bin/env python3
"""Audit architecture, adapter, history, and physical-unit contracts for Stage 3."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.utils import get_model_and_dataset_configs  # noqa: E402


CONFIG = ROOT / "experiments/model_configs/learned_resnet_none_convgru_gaussian.yaml"
NATIVE = ROOT / (
    "outputs/redundancy_resolved_v1_twin/"
    "rr100_zero_gaze_separable_sf_tf_native_production_v1"
)
SELECTION = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_joint_sftf_direction_tuning_checkpoint_v1/selected_unit_roles.csv"
)
TRANSFER = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_spatial_coordinate_contract_stage3_directional_probe_transfer_v1"
)
TRANSLATION = ROOT / (
    "outputs/fig4_active_sensing/"
    "rr100_spatial_coordinate_contract_stage3_directional_translation_v1"
)
OUT = ROOT / "outputs/fig4_active_sensing/rr100_stage3_pathway_contract_audit_v1"
FRAME_RATE_HZ = 120.0


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def output_size(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    return int(math.floor((size + 2 * padding - kernel) / stride) + 1)


def architecture_table(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    stem = config["convnet"]["params"]["stem_config"]["conv_params"]
    blocks = config["convnet"]["params"]["block_configs"]
    recurrent = config["recurrent"]["params"]
    stages = [
        ("retinal input", 1, 1, 0),
        ("valid spatial stem convolution", int(stem["kernel_size"][-1]), int(stem["stride"]), int(stem["padding"][-1])),
        ("valid first residual-block convolution", int(blocks[0]["conv_params"]["kernel_size"][-1]), 1, int(blocks[0]["conv_params"]["padding"][-1])),
        ("maximum pooling", int(blocks[0]["pool_params"]["kernel_size"]), int(blocks[0]["pool_params"]["stride"]), 0),
        ("valid second residual-block convolution", int(blocks[1]["conv_params"]["kernel_size"][-1]), 1, int(blocks[1]["conv_params"]["padding"][-1])),
        ("learned Gaussian readout convolution", 14, 1, 0),
    ]
    sizes = {51: 51, 151: 151}
    receptive_field = 1
    jump = 1
    rows: list[dict[str, object]] = []
    for stage_index, (name, kernel, stride, padding) in enumerate(stages):
        if stage_index > 0:
            receptive_field = receptive_field + (kernel - 1) * jump
            jump *= stride
            sizes = {input_size: output_size(value, kernel, stride, padding) for input_size, value in sizes.items()}
        rows.append(
            {
                "stage_order": stage_index,
                "stage": name,
                "kernel_size_spatial": kernel,
                "stride": stride,
                "padding": padding,
                "output_size_from_51_input": sizes[51],
                "output_size_from_151_input": sizes[151],
                "theoretical_receptive_field_span_input_px": receptive_field,
                "input_pixel_jump_between_adjacent_outputs": jump,
            }
        )
    temporal_lengths = {}
    for history in (32, 33):
        length = output_size(history, int(config["frontend"]["params"]["kernel_size"]))
        for block in blocks:
            length = output_size(length, int(block["conv_params"]["kernel_size"][0]))
        temporal_lengths[history] = length
    feedforward_span = int(rows[-1]["theoretical_receptive_field_span_input_px"])
    input_jump = int(rows[-2]["input_pixel_jump_between_adjacent_outputs"])
    recurrent_kernel = int(recurrent["kernel_size"])
    recurrent_radius_core = (recurrent_kernel - 1) // 2
    summary = {
        "feedforward_readout_receptive_field_span_input_px": feedforward_span,
        "feedforward_native_window_input_px": 51,
        "feedforward_native_window_unused_edge_px_due_pool_floor": 51 - feedforward_span,
        "map_stride_input_px": input_jump,
        "core_recurrent_steps_for_32_history_frames": temporal_lengths[32],
        "core_recurrent_steps_for_33_history_frames": temporal_lengths[33],
        "maximum_recurrently_expanded_span_for_32_history_frames_input_px": (
            feedforward_span + 2 * (temporal_lengths[32] - 1) * recurrent_radius_core * input_jump
        ),
        "maximum_recurrently_expanded_span_for_33_history_frames_input_px": (
            feedforward_span + 2 * (temporal_lengths[33] - 1) * recurrent_radius_core * input_jump
        ),
    }
    return pd.DataFrame(rows), summary


def selected_adapter_table(model: Any) -> pd.DataFrame:
    selected = pd.read_csv(SELECTION)
    mapping = pd.read_csv(NATIVE / "rr100_unit_mapping.csv")
    selected = selected.merge(
        mapping[["rr100_index", "source_unit_index", "canonical_channel"]],
        on="rr100_index",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for record in selected.itertuples(index=False):
        dataset_index = list(model.names).index(str(record.session))
        adapter = model.model.adapters[dataset_index]
        if str(adapter.transform) != "scale":
            raise ValueError(f"Expected scale adapter for {record.session}, got {adapter.transform}")
        scale = torch_functional.softplus(adapter.log_scale.detach().cpu()).numpy().astype(float)
        learned_sigma = float(torch_functional.softplus(adapter.log_sigma.detach().cpu()))
        with torch.no_grad():
            minimum_sigma = float(adapter._sigma_from_grid().detach().cpu())
        effective_sigma = float(np.sqrt(learned_sigma**2 + minimum_sigma**2))
        readout = model.model.readouts[dataset_index]
        mask = readout.compute_gaussian_mask(14, 14, torch.device("cpu"))
        rows.append(
            {
                "rr100_index": int(record.rr100_index),
                "selection_role": str(record.selection_role),
                "session": str(record.session),
                "dataset_index": dataset_index,
                "source_unit_index": int(record.source_unit_index),
                "canonical_channel": int(record.canonical_channel),
                "adapter_transform": str(adapter.transform),
                "adapter_fixed_output_grid_size_px": int(adapter.grid_size),
                "adapter_scale_x": float(scale[0]),
                "adapter_scale_y": float(scale[1]),
                "adapter_learned_blur_sigma_input_px": learned_sigma,
                "adapter_sampling_minimum_blur_sigma_input_px": minimum_sigma,
                "adapter_effective_blur_sigma_input_px": effective_sigma,
                "session_readout_mask_height_core_bins": int(mask.shape[-2]),
                "session_readout_mask_width_core_bins": int(mask.shape[-1]),
            }
        )
    return pd.DataFrame(rows)


def pathway_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pathway": "native empirical tuning production",
                "history_frames": 33,
                "history_contract": "current through t-32",
                "session_adapter": "applied: learned blur and scale resampling to fixed 51-by-51 grid",
                "core": "shared temporal frontend, valid ResNet, and ConvGRU",
                "readout": "selected session readout producing one scalar unit response",
                "response_summary": "long phase-consistent movie; temporal mean; phase schedule; matched blank subtraction",
                "stored_physical_units": "Hz after multiplying expected counts per frame by 120",
            },
            {
                "pathway": "Stage 3 sparse native-size core diagnostic",
                "history_frames": 32,
                "history_contract": "one current 32-frame localized history",
                "session_adapter": "bypassed by direct core_forward call",
                "core": "shared temporal frontend, valid ResNet, and ConvGRU",
                "readout": "assembled canonical population readout producing one scalar at 51-by-51 input size",
                "response_summary": "single instantaneous output phase; separate blank subtraction",
                "stored_physical_units": "expected counts per frame; figures and CSV columns were incorrectly labelled Hz",
            },
            {
                "pathway": "Stage 3 sparse 151-by-151 activation map diagnostic",
                "history_frames": 32,
                "history_contract": "one current 32-frame localized history",
                "session_adapter": "bypassed by direct core_forward call",
                "core": "shared temporal frontend, valid ResNet, and ConvGRU",
                "readout": "same assembled canonical readout convolved with valid padding to a 51-by-51 map",
                "response_summary": "single instantaneous map; separate blank subtraction",
                "stored_physical_units": "expected counts per frame; figures and CSV columns were incorrectly labelled Hz",
            },
        ]
    )


def correction_table() -> pd.DataFrame:
    transfer = json.loads((TRANSFER / "manifest.json").read_text(encoding="utf-8"))["validation"]
    translation = json.loads((TRANSLATION / "manifest.json").read_text(encoding="utf-8"))["validation"]
    sources = [
        ("directional probe transfer", TRANSFER / "manifest.json", transfer),
        ("directional translation", TRANSLATION / "manifest.json", translation),
    ]
    rows: list[dict[str, object]] = []
    for analysis, source, values in sources:
        for metric, value in values.items():
            is_rate = metric.endswith("_hz")
            rows.append(
                {
                    "analysis": analysis,
                    "source_manifest": str(source.resolve()),
                    "metric": metric,
                    "saved_value": float(value),
                    "saved_value_actual_units": "expected counts per frame" if is_rate else "dimensionless",
                    "corrected_value": float(value) * FRAME_RATE_HZ if is_rate else float(value),
                    "corrected_units": "Hz" if is_rate else "dimensionless",
                    "correction_factor": FRAME_RATE_HZ if is_rate else 1.0,
                    "scientific_ordering_or_correlation_changed": False,
                }
            )
    return pd.DataFrame(rows)


def plot_contract(
    adapters: pd.DataFrame,
    architecture_summary: dict[str, int],
    corrections: pd.DataFrame,
    path: Path,
) -> None:
    figure = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    native_axis = figure.add_subplot(grid[0, 0])
    native_axis.axis("off")
    native_axis.text(
        0.02,
        0.96,
        "Native empirical tuning pathway",
        fontsize=14,
        weight="bold",
        va="top",
    )
    native_axis.text(
        0.02,
        0.82,
        "33-frame history (current through t−32)\n"
        "→ learned session blur and scale adapter\n"
        "→ shared temporal/convolutional/recurrent core\n"
        "→ selected session scalar readout\n"
        "→ long-movie temporal mean and carrier-phase averaging\n"
        "→ matched blank subtraction\n"
        "→ expected counts/frame × 120 = Hz",
        fontsize=12,
        va="top",
        linespacing=1.55,
    )
    map_axis = figure.add_subplot(grid[0, 1])
    map_axis.axis("off")
    map_axis.text(0.02, 0.96, "Stage 3 activation-map diagnostic pathway", fontsize=14, weight="bold", va="top")
    map_axis.text(
        0.02,
        0.82,
        "32-frame localized history\n"
        "→ session adapter bypassed\n"
        "→ shared temporal/convolutional/recurrent core\n"
        "→ assembled readout convolved across space\n"
        "→ one instantaneous response phase\n"
        "→ separate blank subtraction\n"
        "→ expected counts/frame (previously mislabeled Hz)",
        fontsize=12,
        va="top",
        linespacing=1.55,
    )

    support_axis = figure.add_subplot(grid[1, 0])
    labels = ["feedforward\nreadout span", "32-frame maximum\nrecurrent span", "33-frame maximum\nrecurrent span"]
    values = [
        architecture_summary["feedforward_readout_receptive_field_span_input_px"],
        architecture_summary["maximum_recurrently_expanded_span_for_32_history_frames_input_px"],
        architecture_summary["maximum_recurrently_expanded_span_for_33_history_frames_input_px"],
    ]
    support_axis.bar(labels, values, color=["#0072B2", "#D55E00", "#009E73"])
    support_axis.bar_label(support_axis.containers[0], fmt="%.0f input pixels", padding=3)
    support_axis.set_ylabel("architectural support span (input pixels)")
    support_axis.set_title(
        "Model-defined support scales\nRecurrent values are maximum theoretical spans, not fitted effective weights"
    )
    support_axis.spines[["top", "right"]].set_visible(False)

    adapter_axis = figure.add_subplot(grid[1, 1])
    x = np.arange(len(adapters))
    adapter_axis.plot(x, adapters.adapter_scale_x, marker="o", label="horizontal scale")
    adapter_axis.plot(x, adapters.adapter_scale_y, marker="s", label="vertical scale")
    adapter_axis.set_xticks(x, [f"unit {unit}" for unit in adapters.rr100_index], rotation=25, ha="right")
    adapter_axis.set_ylabel("learned adapter scale")
    adapter_axis.set_title(
        "Selected units use session-specific adapters\n"
        f"Corrected rate metrics requiring ×120: {int(corrections.correction_factor.eq(FRAME_RATE_HZ).sum())}"
    )
    adapter_axis.legend(frameon=False)
    adapter_axis.grid(axis="y", alpha=0.2)
    adapter_axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "RR100 Stage 3 pathway-contract audit: native tuning and activation maps were not yet compared like for like",
        fontsize=16,
        weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise FileExistsError(f"Completed pathway audit exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    architecture, architecture_summary = architecture_table(config)
    model, _ = get_model_and_dataset_configs(mode="standard")
    model = model.to("cpu").eval()
    adapters = selected_adapter_table(model)
    pathways = pathway_table()
    corrections = correction_table()

    architecture.to_csv(OUT / "model_defined_spatial_architecture.csv", index=False)
    adapters.to_csv(OUT / "selected_unit_session_adapter_parameters.csv", index=False)
    pathways.to_csv(OUT / "native_and_activation_map_pathway_contracts.csv", index=False)
    corrections.to_csv(OUT / "stage3_rate_unit_corrections.csv", index=False)
    plot_contract(adapters, architecture_summary, corrections, OUT / "01_pathway_contract_audit")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "rr100_stage3_pathway_contract_audit",
        "status": "correctness_audit_complete_dense_tuning_transfer_not_authorized",
        "findings": {
            "native_history_frames": 33,
            "stage3_sparse_history_frames": 32,
            "native_session_adapter": "applied",
            "stage3_sparse_session_adapter": "bypassed",
            "native_tuning_estimand": "long phase-consistent temporal mean, phase scheduled, matched blank subtracted",
            "stage3_sparse_estimand": "one instantaneous localized response phase, separately blank subtracted",
            "stage3_sparse_saved_units": "expected counts per frame mislabeled as Hz",
            "physical_unit_correction_factor": FRAME_RATE_HZ,
            **architecture_summary,
        },
        "interpretation": (
            "the sparse Stage 3 comparisons remain valid tests of canvas-size and translation behavior within "
            "their direct-core pathway after relabeling units, but they are not tests of native empirical F0 "
            "tensor transfer; a new like-for-like phase-averaged design is required"
        ),
        "sources": {
            "model_config": identity(CONFIG),
            "native_manifest": identity(NATIVE / "analysis_manifest.json"),
            "native_request": identity(NATIVE / "request_identity.json"),
            "selected_roles": identity(SELECTION),
            "transfer_manifest": identity(TRANSFER / "manifest.json"),
            "translation_manifest": identity(TRANSLATION / "manifest.json"),
            "runner": identity(Path(__file__)),
        },
        "outputs": {
            name: identity(OUT / name)
            for name in (
                "model_defined_spatial_architecture.csv",
                "selected_unit_session_adapter_parameters.csv",
                "native_and_activation_map_pathway_contracts.csv",
                "stage3_rate_unit_corrections.csv",
                "01_pathway_contract_audit.png",
            )
        },
        "decision_gate": (
            "quarantine incorrect Hz labels and design a phase-averaged, history-matched, adapter-aware native-to-map comparison before GPU scoring"
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# RR100 Stage 3 pathway-contract audit\n\n"
        "This correctness checkpoint separates model-defined geometry from unresolved tuning transfer. It found "
        "that native empirical tuning and the sparse activation-map probes differed in history length, session "
        "adapter use, response estimand, and saved physical units. Sparse Stage 3 correlations and translation "
        "tests remain valid within their direct-core pathway, but their rate values are expected counts per "
        "frame rather than Hz and they do not establish transfer of the native F0 tensor.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
