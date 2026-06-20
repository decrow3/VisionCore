"""Build cache-only Figure 4E subpanels from behavior/image-geometry outputs."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKIMAGE_BASE = REPO_ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_ALIGNMENT_DIR = BACKIMAGE_BASE / "backimage_edge_alignment_distribution_inspection"
DEFAULT_WINDOW_DIR = BACKIMAGE_BASE / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
DEFAULT_OUT_DIR = REPO_ROOT / "declan/figure4_active_sensing_atlas/figures/panel_E"

SUBSET_ORDER = ["All windows", "Reliable axes", "High confidence"]
SUBSET_SHORT = {
    "All windows": "all",
    "Reliable axes": "reliable",
    "High confidence": "high conf.",
}
COLORS = {
    "edge": "#242a2f",
    "drift": "#2f8f6a",
    "all": "#8e9aa6",
    "reliable": "#2f8f6a",
    "high": "#3366aa",
    "parallel": "#2f8f6a",
    "orthogonal": "#8064a2",
    "mid": "#9aa3ad",
    "unweighted": "#2f8f6a",
    "weighted": "#d07a22",
    "supported": "#2f8f6a",
    "guardrail": "#d07a22",
    "grid": "#d8dde3",
}

SOURCE_BEHAVIOR_PANELS = [
    {
        "panel_id": "E6",
        "stem": "E6_full_distribution_session_diagnostic",
        "source_file": "edge_alignment_window_and_session_distributions.png",
        "title": "Full distribution and session diagnostic",
        "role": "behavior provenance",
        "read": "Full drift-edge distribution, session scatter, and cumulative parallel preference.",
    },
    {
        "panel_id": "E7",
        "stem": "E7_confidence_signed_delta_diagnostic",
        "source_file": "edge_alignment_confidence_and_signed_delta.png",
        "title": "Confidence and signed-delta diagnostic",
        "role": "behavior provenance",
        "read": "Alignment strengthens with image-axis confidence and FEM anisotropy.",
    },
    {
        "panel_id": "E8",
        "stem": "E8_endpoint_null_diagnostic",
        "source_file": "edge_alignment_endpoint_null_diagnostic.png",
        "title": "Endpoint/null diagnostic",
        "role": "behavior provenance",
        "read": "Endpoint enrichment read against the transformed uniform-angle null.",
    },
]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_png_as_pdf(png_path: Path, pdf_path: Path) -> None:
    with Image.open(png_path) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        background.save(pdf_path, "PDF", resolution=220.0)


def copy_source_behavior_panels(alignment_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in SOURCE_BEHAVIOR_PANELS:
        source_path = alignment_dir / spec["source_file"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source behavior panel: {source_path}")
        target_png = out_dir / f"{spec['stem']}.png"
        target_pdf = out_dir / f"{spec['stem']}.pdf"
        shutil.copy2(source_path, target_png)
        _write_png_as_pdf(target_png, target_pdf)
        rows.append(
            {
                "panel_id": spec["panel_id"],
                "title": spec["title"],
                "role": spec["role"],
                "source_png": str(source_path),
                "target_png": str(target_png),
                "target_pdf": str(target_pdf),
                "read": spec["read"],
            }
        )
    return pd.DataFrame(rows)


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _axis_vec(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array([np.cos(angle), np.sin(angle)])


def _axis_delta_abs_deg(delta_deg: pd.Series) -> pd.Series:
    return ((delta_deg.astype(float) + 90.0) % 180.0 - 90.0).abs()


def _synthetic_edge_patch(edge_axis_deg: float, size: int = 128) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    normal = _axis_vec(edge_axis_deg + 90.0)
    edge = 0.50 + 0.28 * np.tanh(7.0 * (normal[0] * x + normal[1] * y))
    texture = 0.06 * np.sin(18.0 * (0.4 * x + 0.9 * y))
    texture += 0.05 * np.sin(23.0 * (0.9 * x - 0.2 * y))
    return np.clip(edge + texture, 0.0, 1.0)


def _arrow_from_center(ax: plt.Axes, angle_deg: float, color: str, label: str, length: float = 0.32) -> None:
    vec = _axis_vec(angle_deg)
    start = np.array([0.5, 0.5]) - 0.5 * length * vec
    end = np.array([0.5, 0.5]) + 0.5 * length * vec
    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=2.0,
            color=color,
            transform=ax.transAxes,
        )
    )
    label_xy = np.array([0.5, 0.5]) + 0.62 * length * vec
    ax.text(label_xy[0], label_xy[1], label, color=color, fontsize=7.5, ha="center", va="center", transform=ax.transAxes)


def _load_tables(alignment_dir: Path, window_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distribution = pd.read_csv(alignment_dir / "edge_alignment_distribution_summary.csv")
    endpoint = pd.read_csv(alignment_dir / "endpoint_zone_enrichment_summary.csv")
    orientation = pd.read_csv(window_dir / "orientation_alignment_summary.csv")
    windows = pd.read_csv(window_dir / "backimage_image_fem_windows.csv")
    return distribution, endpoint, orientation, windows


def _select_example_window(windows: pd.DataFrame) -> pd.DataFrame:
    work = windows[windows["image_feature_ok"].astype(bool)].copy()
    work["abs_edge_delta_deg"] = _axis_delta_abs_deg(work["drift_edge_delta_deg"])
    work["example_score"] = (
        work["image_orientation_coherence"].astype(float)
        * work["anisotropy"].astype(float)
        * (1.0 - work["abs_edge_delta_deg"] / 90.0)
    )
    work = work[
        (work["image_orientation_coherence"].astype(float) >= 0.45)
        & (work["anisotropy"].astype(float) >= 0.45)
        & (work["drift_edge_cos2"].astype(float) > 0.75)
    ].copy()
    if work.empty:
        raise ValueError("No high-confidence behavior example found in window table")
    example = work.sort_values(["example_score", "drift_edge_cos2"], ascending=False).head(1)
    return example[
        [
            "session",
            "trial_idx",
            "phase",
            "image_edge_axis_deg",
            "drift_orientation_deg",
            "drift_edge_delta_deg",
            "drift_edge_cos2",
            "image_orientation_coherence",
            "anisotropy",
            "abs_edge_delta_deg",
            "example_score",
        ]
    ].copy()


def plot_e1_behavior_setup(windows: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    example = _select_example_window(windows)
    row = example.iloc[0]
    edge_axis = float(row["image_edge_axis_deg"])
    drift_axis = float(row["drift_orientation_deg"])

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    ax.imshow(_synthetic_edge_patch(edge_axis), cmap="gray", vmin=0, vmax=1, extent=(0, 1, 0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    _arrow_from_center(ax, edge_axis, COLORS["edge"], "edge axis", length=0.55)
    _arrow_from_center(ax, drift_axis, COLORS["drift"], "FEM axis", length=0.40)

    ell = Ellipse(
        (0.50, 0.50),
        width=0.42,
        height=0.12,
        angle=drift_axis,
        edgecolor=COLORS["drift"],
        facecolor="none",
        linewidth=1.6,
        alpha=0.9,
        transform=ax.transAxes,
    )
    ax.add_patch(ell)
    ax.text(
        0.50,
        0.08,
        f"example: cos2(edge,FEM) = {float(row['drift_edge_cos2']):.2f}",
        ha="center",
        va="center",
        fontsize=8,
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d8dde3", "linewidth": 0.7},
    )
    ax.set_title("Measured FEM axis against local edge geometry")
    _save(fig, out_dir, "E1_behavior_setup_example")
    return example


def plot_e2_alignment_strength(distribution: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    block = distribution.set_index("subset").loc[SUBSET_ORDER].reset_index().copy()
    fig, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)
    x = np.arange(len(block))
    y = block["mean_edge_alignment_index_session"].to_numpy(dtype=float)
    lo = block["ci95_low_session_mean"].to_numpy(dtype=float)
    hi = block["ci95_high_session_mean"].to_numpy(dtype=float)
    colors = [COLORS["all"], COLORS["reliable"], COLORS["high"]]
    ax.bar(x, y, color=colors, width=0.68)
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color="#242a2f", lw=1.1, capsize=0, linestyle="none")
    ax.axhline(0.0, color="#242a2f", lw=0.8)
    ax.set_xticks(x, [SUBSET_SHORT[name] for name in block["subset"]])
    ax.set_ylabel("session mean cos2(edge,FEM)")
    ax.set_ylim(0.0, 0.45)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax.set_title("Free-viewing FEM axes align with local edges")
    _clean_axis(ax)
    for idx, row in block.iterrows():
        ax.text(idx, float(row["mean_edge_alignment_index_session"]) + 0.02, f"{row['n_windows']:.0f} windows", ha="center", fontsize=7.2)
    _save(fig, out_dir, "E2_behavior_alignment_strength")
    return block


def plot_e3_endpoint_enrichment(endpoint: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    zones = ["parallel <=15 deg", "orthogonal >=75 deg", "mid 30-60 deg"]
    zone_labels = {"parallel <=15 deg": "parallel", "orthogonal >=75 deg": "orthogonal", "mid 30-60 deg": "mid"}
    zone_colors = {
        "parallel <=15 deg": COLORS["parallel"],
        "orthogonal >=75 deg": COLORS["orthogonal"],
        "mid 30-60 deg": COLORS["mid"],
    }
    block = endpoint[endpoint["zone"].isin(zones)].copy()
    block["subset"] = pd.Categorical(block["subset"], SUBSET_ORDER, ordered=True)
    block["zone"] = pd.Categorical(block["zone"], zones, ordered=True)
    block = block.sort_values(["subset", "zone"])

    fig, ax = plt.subplots(figsize=(5.4, 3.0), constrained_layout=True)
    x = np.arange(len(SUBSET_ORDER))
    width = 0.23
    for offset, zone in zip([-width, 0.0, width], zones, strict=True):
        sub = block[block["zone"] == zone].set_index("subset").loc[SUBSET_ORDER]
        ax.bar(x + offset, sub["observed_expected_ratio"], width=width, color=zone_colors[zone], label=zone_labels[zone])
    ax.axhline(1.0, color="#242a2f", lw=0.8, linestyle="--")
    ax.set_xticks(x, [SUBSET_SHORT[name] for name in SUBSET_ORDER])
    ax.set_ylabel("observed / uniform expected")
    ax.set_ylim(0.0, 2.35)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    ax.set_title("Parallel endpoint zone is enriched")
    _clean_axis(ax)
    _save(fig, out_dir, "E3_parallel_zone_enrichment")
    return block


def plot_e4_metric_convention(distribution: pd.DataFrame, orientation: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    unweighted = distribution.set_index("subset").loc[["All windows", "Reliable axes"]]
    weighted = orientation[
        (orientation["alignment_reference"] == "edge_axis")
        & (orientation["analysis_subset"].isin(["all_windows", "reliable_axes_coh_ge_0p20_aniso_ge_0p20"]))
    ].copy()
    weighted["subset"] = weighted["analysis_subset"].map(
        {"all_windows": "All windows", "reliable_axes_coh_ge_0p20_aniso_ge_0p20": "Reliable axes"}
    )
    weighted = weighted.set_index("subset").loc[["All windows", "Reliable axes"]]
    block = pd.DataFrame(
        {
            "subset": ["All windows", "Reliable axes"],
            "unweighted_session_mean": unweighted["mean_edge_alignment_index_session"].to_numpy(dtype=float),
            "weighted_headline_mean": weighted["weighted_mean_cos2_delta"].to_numpy(dtype=float),
            "n_windows": unweighted["n_windows"].to_numpy(dtype=int),
            "n_sessions": unweighted["n_sessions"].to_numpy(dtype=int),
        }
    )

    fig, ax = plt.subplots(figsize=(4.4, 2.8), constrained_layout=True)
    x = np.arange(len(block))
    width = 0.32
    ax.bar(x - width / 2, block["unweighted_session_mean"], width=width, color=COLORS["unweighted"], label="atlas unweighted")
    ax.bar(x + width / 2, block["weighted_headline_mean"], width=width, color=COLORS["weighted"], label="weighted headline")
    ax.set_xticks(x, [SUBSET_SHORT[name] for name in block["subset"]])
    ax.set_ylabel("cos2(edge,FEM)")
    ax.set_ylim(0.0, 0.24)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Metric convention changes magnitude")
    _clean_axis(ax)
    _save(fig, out_dir, "E4_metric_convention_guardrail")
    return block


def plot_e5_scope_summary(out_dir: Path) -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {"status": "supported", "item": "Measured FEM axes align with local edges"},
            {"status": "supported", "item": "Alignment strengthens for reliable axes"},
            {"status": "supported", "item": "Parallel endpoint zone is enriched"},
            {"status": "guardrail", "item": "Weighted and unweighted summaries differ"},
            {"status": "guardrail", "item": "Current response objectives do not beat raw edge"},
        ]
    )
    fig, ax = plt.subplots(figsize=(6.2, 2.8), constrained_layout=True)
    ax.set_axis_off()
    ax.text(0.02, 0.92, "Supported", color=COLORS["supported"], fontsize=10, weight="bold", transform=ax.transAxes)
    ax.text(0.52, 0.92, "Guardrails", color=COLORS["guardrail"], fontsize=10, weight="bold", transform=ax.transAxes)

    for idx, item in enumerate(rows[rows["status"] == "supported"]["item"]):
        y = 0.74 - 0.20 * idx
        ax.add_patch(FancyBboxPatch((0.02, y - 0.06), 0.42, 0.11, boxstyle="round,pad=0.014,rounding_size=0.012", facecolor="#f5faf7", edgecolor="#cbd7d1", linewidth=0.8, transform=ax.transAxes))
        ax.text(0.04, y, item, va="center", fontsize=8.0, transform=ax.transAxes)
    for idx, item in enumerate(rows[rows["status"] == "guardrail"]["item"]):
        y = 0.74 - 0.20 * idx
        ax.add_patch(FancyBboxPatch((0.52, y - 0.06), 0.44, 0.11, boxstyle="round,pad=0.014,rounding_size=0.012", facecolor="#fff8ef", edgecolor="#decfb9", linewidth=0.8, transform=ax.transAxes))
        ax.text(0.54, y, item, va="center", fontsize=8.0, transform=ax.transAxes)
    ax.set_title("Behavior result is positive; objective adjudication remains open")
    _save(fig, out_dir, "E5_scope_summary")
    return rows


def _write_caption(out_dir: Path) -> None:
    caption = """# Panel E Subpanels

Generated cache-only from BackImage free-viewing image-geometry and
edge-alignment distribution outputs.

Subpanels:

- `E1_behavior_setup_example`: representative high-confidence window showing
  the measured FEM axis against the local edge axis.
- `E2_behavior_alignment_strength`: atlas convention for behavioral
  edge-alignment strength, using unweighted session means.
- `E3_parallel_zone_enrichment`: parallel endpoint-zone enrichment relative to
  a uniform angular expectation.
- `E6_full_distribution_session_diagnostic`: original inspection diagnostic
  showing full distribution, session scatter, and cumulative parallel
  preference.
- `E7_confidence_signed_delta_diagnostic`: original inspection diagnostic
  showing confidence dependence and signed drift-edge deltas.
- `E8_endpoint_null_diagnostic`: original inspection diagnostic showing the
  endpoint/null read that E3 compresses.
- `E4_metric_convention_guardrail`: unweighted atlas summary versus weighted
  headline-style summary.
- `E5_scope_summary`: supported behavioral claims and remaining guardrails.

Claim boundary:

```text
Measured free-viewing FEM axes align modestly but reliably with local image
geometry, especially when the local axis estimate is reliable. This supports
image-contingent FEMs, but the metric convention must be stated explicitly and
current response-objective models do not yet beat raw edge geometry.
```

Provenance note:

```text
E3 is an atlas redraw from endpoint_zone_enrichment_summary.csv. E6-E8 are
copied from the original backimage_edge_alignment_distribution_inspection
diagnostics so the endpoint result can be read with its full distribution,
confidence, and uniform-angle-null context.
```
"""
    (out_dir / "panel_E_subpanels_caption.md").write_text(caption)


def _write_index(out_dir: Path, generated: Iterable[str], copied: Iterable[str]) -> None:
    lines = ["# Panel E Generated Assets", ""]
    for stem in generated:
        lines.append(f"- `{stem}.png`")
        lines.append(f"- `{stem}.pdf`")
    lines.extend(["", "## Copied Source Diagnostics", ""])
    for stem in copied:
        lines.append(f"- `{stem}.png`")
        lines.append(f"- `{stem}.pdf`")
    lines.extend(
        [
            "",
            "- `panel_E_behavior_example_values.csv`",
            "- `panel_E_alignment_strength_values.csv`",
            "- `panel_E_endpoint_enrichment_values.csv`",
            "- `panel_E_metric_convention_values.csv`",
            "- `panel_E_scope_summary_values.csv`",
            "- `panel_E_contour_following_source_panels.csv`",
            "- `panel_E_subpanels_caption.md`",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--window-dir", type=Path, default=DEFAULT_WINDOW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    _configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    distribution, endpoint, orientation, windows = _load_tables(args.alignment_dir, args.window_dir)
    example_values = plot_e1_behavior_setup(windows, args.out_dir)
    alignment_values = plot_e2_alignment_strength(distribution, args.out_dir)
    endpoint_values = plot_e3_endpoint_enrichment(endpoint, args.out_dir)
    source_panel_values = copy_source_behavior_panels(args.alignment_dir, args.out_dir)
    convention_values = plot_e4_metric_convention(distribution, orientation, args.out_dir)
    scope_values = plot_e5_scope_summary(args.out_dir)

    example_values.to_csv(args.out_dir / "panel_E_behavior_example_values.csv", index=False)
    alignment_values.to_csv(args.out_dir / "panel_E_alignment_strength_values.csv", index=False)
    endpoint_values.to_csv(args.out_dir / "panel_E_endpoint_enrichment_values.csv", index=False)
    source_panel_values.to_csv(args.out_dir / "panel_E_contour_following_source_panels.csv", index=False)
    convention_values.to_csv(args.out_dir / "panel_E_metric_convention_values.csv", index=False)
    scope_values.to_csv(args.out_dir / "panel_E_scope_summary_values.csv", index=False)
    _write_caption(args.out_dir)
    _write_index(
        args.out_dir,
        [
            "E1_behavior_setup_example",
            "E2_behavior_alignment_strength",
            "E3_parallel_zone_enrichment",
            "E4_metric_convention_guardrail",
            "E5_scope_summary",
        ],
        [spec["stem"] for spec in SOURCE_BEHAVIOR_PANELS],
    )


if __name__ == "__main__":
    main()
