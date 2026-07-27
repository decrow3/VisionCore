#!/usr/bin/env python3
"""Standalone renderer for Panel D contour-relative stimulus schematic.

The main figure still owns the Panel D drawing code. This wrapper lets us
iterate on Panel D, or just its left half, without rebuilding the full SSI
figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
FIGURE_DIR = ROOT / "declan" / "fig" / "ssi_figure_v2"
if str(FIGURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_DIR))

import generate_ssi_figure_v2 as figure  # noqa: E402


OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"


def _metadata_for_json(metadata: dict[str, object]) -> dict[str, object]:
    clean: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, (int, float, str)):
            clean[key] = value
    return clean


def _save_panel_variant(
    *,
    out_dir: Path,
    stem: str,
    figsize: tuple[float, float],
    show_right_half: bool,
    xlim: tuple[float, float] | None,
    header_title: str,
    schematic_payload: dict | None,
) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.865, bottom=0.055)
    figure.draw_panel_a(
        ax,
        schematic_payload=schematic_payload,
        show_right_half=show_right_half,
        header_title=header_title,
        xlim=xlim if xlim is not None else (0.0, 1.0),
    )
    paths = {
        "png": out_dir / f"{stem}.png",
        "pdf": out_dir / f"{stem}.pdf",
        "svg": out_dir / f"{stem}.svg",
    }
    fig.savefig(paths["png"], dpi=240)
    fig.savefig(paths["pdf"], dpi=300)
    fig.savefig(paths["svg"], dpi=300)
    plt.close(fig)
    return paths


def build_panel(out_dir: Path = OUT_DIR, *, variant: str = "both") -> dict[str, Path]:
    figure.configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    schematic_payload = figure.read_schematic_payload()
    metadata = figure.contour_window_metadata(schematic_payload)
    if figure.ssi_schematic is not None and schematic_payload is not None:
        try:
            synthetic_left = figure.ssi_schematic.make_synthetic_left_side(
                schematic_payload.get("patch"),
                schematic_payload.get("contour_axis_image_deg", 10.352312),
            )
            metadata = figure.trace_fit_center_zoom_metadata(metadata, synthetic_left.get("eye"))
        except Exception:
            pass
    metadata_path = out_dir / "panel_d_contour_relative_stimulus_metadata.json"
    metadata_path.write_text(json.dumps(_metadata_for_json(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    paths: dict[str, Path] = {"metadata_json": metadata_path}
    if variant in {"full", "both"}:
        paths.update(
            {
                f"full_{key}": value
                for key, value in _save_panel_variant(
                    out_dir=out_dir,
                    stem="panel_d_contour_relative_stimulus",
                    figsize=(5.45, 2.55),
                    show_right_half=True,
                    xlim=None,
                    header_title="Contour-relative stimulus and unit responses",
                    schematic_payload=schematic_payload,
                ).items()
            }
        )
    if variant in {"left", "both"}:
        paths.update(
            {
                f"left_{key}": value
                for key, value in _save_panel_variant(
                    out_dir=out_dir,
                    stem="panel_d_contour_relative_stimulus_left",
                    figsize=(3.45, 2.55),
                    show_right_half=False,
                    xlim=(0.0, 0.62),
                    header_title="Contour window",
                    schematic_payload=schematic_payload,
                ).items()
            }
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--variant", choices=("full", "left", "both"), default="both")
    args = parser.parse_args()
    paths = build_panel(args.out_dir, variant=args.variant)
    for key in sorted(paths):
        print(paths[key])


if __name__ == "__main__":
    main()
