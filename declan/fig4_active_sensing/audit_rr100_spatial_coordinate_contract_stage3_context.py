#!/usr/bin/env python3
"""Audit whether Stage 3 native/large mismatch is a blank-context baseline effect."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from declan.fig4_active_sensing.run_rr100_direct_fem_orientation_checkpoint import RR100_MOVIE_MEDOID_VERSION
from declan.fig4_active_sensing.run_rr100_spatial_coordinate_contract_stage3 import (
    N_HISTORY,
    forward_selected_maps,
)
from declan.redundancy_resolved_v1_population import load_canonical_twin_bundle, load_population_view


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_v1"
OUT = ROOT / "outputs/fig4_active_sensing/rr100_spatial_coordinate_contract_stage3_context_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def plot_context_audit(rows: pd.DataFrame, baselines: pd.DataFrame, path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    for axis, y_column, title in (
        (axes[0], "large_embedded_modulation_hz", "Exactly embedded probe after separate blank subtraction"),
        (axes[1], "large_extended_modulation_hz", "Analytically extended probe after separate blank subtraction"),
    ):
        axis.scatter(rows.native_modulation_hz, rows[y_column], s=30, alpha=0.75)
        low = float(min(rows.native_modulation_hz.min(), rows[y_column].min()))
        high = float(max(rows.native_modulation_hz.max(), rows[y_column].max()))
        axis.plot([low, high], [low, high], color="0.35", ls="--")
        axis.set(
            xlabel="native probe minus native blank (Hz)",
            ylabel="large-canvas probe minus large-canvas blank (Hz)",
            title=title,
        )
    axes[2].scatter(
        baselines.native_blank_rate_hz, baselines.large_blank_center_rate_hz,
        c=baselines.rr100_index, cmap="viridis", s=55,
    )
    low = float(min(baselines.native_blank_rate_hz.min(), baselines.large_blank_center_rate_hz.min()))
    high = float(max(baselines.native_blank_rate_hz.max(), baselines.large_blank_center_rate_hz.max()))
    axes[2].plot([low, high], [low, high], color="0.35", ls="--")
    axes[2].set(
        xlabel="native blank response (Hz)", ylabel="large-canvas central blank response (Hz)",
        title="Blank response depends on canvas boundary context",
    )
    axes[3].scatter(
        rows.native_modulation_hz,
        np.maximum(rows.embedded_modulation_absolute_error_hz, rows.extended_modulation_absolute_error_hz),
        s=30, alpha=0.75, color="#D55E00",
    )
    axes[3].set_yscale("symlog", linthresh=1e-9)
    axes[3].set(
        xlabel="native probe modulation (Hz)", ylabel="maximum modulation error (Hz)",
        title="Residual error after separate baseline subtraction",
    )
    figure.suptitle(
        "Stage 3 context audit: does separate blank subtraction reconcile native and large-canvas responses?",
        fontsize=14, weight="bold",
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    out = args.out_dir.resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Completed audit already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(source / "selected_units_and_preferred_probes.csv")
    equivalence = pd.read_csv(source / "native_to_large_canvas_response_equivalence.csv")
    with np.load(source / "selected_unit_spatial_contract_maps.npz", allow_pickle=False) as archive:
        large_blank = np.asarray(archive["selected_unit_blank_large_rate_maps"], dtype=float)

    view = load_population_view(version_name=RR100_MOVIE_MEDOID_VERSION)
    mapped = np.argmax(view.membership, axis=1).astype(int)
    canonical = units.canonical_channel.to_numpy(int)
    if not np.array_equal(canonical, mapped[units.rr100_index.to_numpy(int)]):
        raise ValueError("Selected canonical channels disagree with RR100 population view")
    bundle = load_canonical_twin_bundle(device=str(args.device), mode="standard")
    native_blank = forward_selected_maps(
        bundle,
        np.zeros((1, N_HISTORY, 51, 51), dtype=np.float32),
        canonical,
        batch_size=1,
    )[0, :, 0, 0].astype(float)
    large_center = large_blank[:, large_blank.shape[-2] // 2, large_blank.shape[-1] // 2]
    large_spatial_std = large_blank.reshape(len(units), -1).std(axis=1)
    baselines = pd.DataFrame(
        {
            "rr100_index": units.rr100_index.to_numpy(int),
            "selection_role": units.selection_role,
            "native_blank_rate_hz": native_blank,
            "large_blank_center_rate_hz": large_center,
            "large_blank_spatial_standard_deviation_hz": large_spatial_std,
            "blank_canvas_difference_hz": large_center - native_blank,
        }
    )
    baselines.to_csv(out / "native_and_large_blank_responses.csv", index=False)

    response_lookup = {int(unit): index for index, unit in enumerate(units.rr100_index)}
    rows = equivalence.copy()
    rows["native_blank_rate_hz"] = [native_blank[response_lookup[int(unit)]] for unit in rows.response_rr100_index]
    rows["large_blank_center_rate_hz"] = [large_center[response_lookup[int(unit)]] for unit in rows.response_rr100_index]
    rows["native_modulation_hz"] = rows.native_rate_hz - rows.native_blank_rate_hz
    rows["large_embedded_modulation_hz"] = rows.large_embedded_center_rate_hz - rows.large_blank_center_rate_hz
    rows["large_extended_modulation_hz"] = rows.large_extended_center_rate_hz - rows.large_blank_center_rate_hz
    rows["embedded_modulation_absolute_error_hz"] = np.abs(
        rows.large_embedded_modulation_hz - rows.native_modulation_hz
    )
    rows["extended_modulation_absolute_error_hz"] = np.abs(
        rows.large_extended_modulation_hz - rows.native_modulation_hz
    )
    rows.to_csv(out / "blank_subtracted_response_equivalence.csv", index=False)
    figure_path = out / "01_blank_subtracted_response_equivalence"
    plot_context_audit(rows, baselines, figure_path, int(args.dpi))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "cartographer_stage3_native_large_blank_context_audit",
        "status": "targeted_context_audit_complete_awaiting_human_review",
        "validation": {
            "maximum_large_blank_spatial_standard_deviation_hz": float(large_spatial_std.max()),
            "maximum_absolute_native_large_blank_difference_hz": float(np.abs(large_center - native_blank).max()),
            "native_vs_embedded_modulation_pearson_r": float(
                np.corrcoef(rows.native_modulation_hz, rows.large_embedded_modulation_hz)[0, 1]
            ),
            "maximum_embedded_modulation_absolute_error_hz": float(
                rows.embedded_modulation_absolute_error_hz.max()
            ),
            "mean_embedded_modulation_absolute_error_hz": float(
                rows.embedded_modulation_absolute_error_hz.mean()
            ),
            "maximum_extended_modulation_absolute_error_hz": float(
                rows.extended_modulation_absolute_error_hz.max()
            ),
        },
        "interpretation_gate": (
            "if separate blank subtraction does not reconcile modulation, native grating response scale cannot be "
            "transferred directly to large-canvas activation maps; freeze neither aperture nor Stage 4 calibration"
        ),
        "reserved_final_test_identities_opened": False,
        "source": identity(source / "manifest.json"),
        "outputs": {
            "blank_responses": identity(out / "native_and_large_blank_responses.csv"),
            "modulation_equivalence": identity(out / "blank_subtracted_response_equivalence.csv"),
            "figure": identity(figure_path.with_suffix(".png")),
        },
        "runner": identity(Path(__file__)),
    }
    (out / "manifest.json").write_text(json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Cartographer Stage 3 blank-context audit\n\n"
        "This audit measures a native 51×51 blank response and compares probe modulation after subtracting "
        "native and large-canvas blanks separately. It determines whether the Stage 3 absolute-response mismatch "
        "is only a boundary-dependent baseline or also changes stimulus modulation.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
