"""Build cache-only Figure 4C subpanels from trajectory-observer outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKIMAGE_BASE = REPO_ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_OBSERVER_DIR = (
    BACKIMAGE_BASE / "backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1"
)
DEFAULT_COMPACT_DIR = (
    DEFAULT_OBSERVER_DIR / "compact_mechanism_image_disjoint_fold0_n512_k2_5_10_20_rand8_log_v1"
)
DEFAULT_OUT_DIR = REPO_ROOT / "declan/figure4_active_sensing_atlas/figures/panel_C"

CANDIDATE_ORDER = ["hard_negative_structure", "matched_static_response"]
CANDIDATE_LABELS = {
    "hard_negative_structure": "hard negatives",
    "matched_static_response": "matched static",
}
PRIOR_ORDER = ["empirical", "ou"]
PRIOR_LABELS = {"empirical": "empirical prior", "ou": "OU prior"}
COLORS = {
    "known": "#242a2f",
    "zero": "#8e9aa6",
    "empirical": "#2f8f6a",
    "ou": "#3366aa",
    "matched_static_response": "#2f8f6a",
    "hard_negative_structure": "#8064a2",
    "compact": "#2f8f6a",
    "static_pc": "#8064a2",
    "random": "#9aa3ad",
    "removed": "#d07a22",
    "full": "#242a2f",
    "light": "#eef2f4",
    "muted": "#6f7a83",
}


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


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _scale_label(value: float) -> str:
    return f"{value:g}x"


def _node(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    width: float = 0.19,
    height: float = 0.13,
    facecolor: str = "#f8fafb",
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.014",
        edgecolor="#c5ccd2",
        facecolor=facecolor,
        linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=8, transform=ax.transAxes)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#5d6871",
            transform=ax.transAxes,
        )
    )


def _load_observer(observer_dir: Path) -> pd.DataFrame:
    path = observer_dir / "observer_summary.csv"
    df = pd.read_csv(path)
    required = {
        "candidate_set_mode",
        "prior_family",
        "observation_scale",
        "prior_scale",
        "trajectory_prior_mode",
        "likelihood_scale",
        "n_trials",
        "known_eye_accuracy",
        "zero_eye_accuracy",
        "joint_eye_accuracy",
        "joint_minus_zero_accuracy",
        "median_N_eff_fraction",
        "median_nearest_tau_rank",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    work = df[
        (df["trajectory_prior_mode"] == "leave_one_out")
        & (df["candidate_set_mode"].isin(CANDIDATE_ORDER))
        & (df["prior_family"].isin(PRIOR_ORDER))
    ].copy()
    for col in ["observation_scale", "prior_scale", "likelihood_scale"]:
        work[col] = pd.to_numeric(work[col])
    work = work[work["observation_scale"] == work["prior_scale"]].copy()
    return work.sort_values(["candidate_set_mode", "observation_scale", "prior_family", "likelihood_scale"])


def _primary_rows(observer: pd.DataFrame) -> pd.DataFrame:
    return observer[observer["likelihood_scale"] == 1.0].copy()


def plot_c1_observer_schematic(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.35), constrained_layout=True)
    ax.set_axis_off()

    _node(ax, (0.10, 0.62), "candidate\nimages", width=0.16)
    _node(ax, (0.10, 0.34), "latent eye\ntrajectory", width=0.16)
    _node(ax, (0.33, 0.50), "cached response\ntables", width=0.20, facecolor="#fbfcfd")
    _node(ax, (0.56, 0.50), "marginalize\nover tau", width=0.18, facecolor="#f5faf7")
    _node(ax, (0.79, 0.50), "posterior over\nimage identity", width=0.20)

    _arrow(ax, (0.18, 0.62), (0.23, 0.55))
    _arrow(ax, (0.18, 0.34), (0.23, 0.45))
    _arrow(ax, (0.43, 0.50), (0.47, 0.50))
    _arrow(ax, (0.65, 0.50), (0.69, 0.50))

    ax.text(
        0.33,
        0.22,
        "response table:\nlambda[I,tau,t,u]",
        ha="center",
        fontsize=7.2,
        transform=ax.transAxes,
    )
    ax.text(
        0.56,
        0.22,
        "sum_tau p(r | I,tau) p(tau)",
        ha="center",
        fontsize=7.2,
        transform=ax.transAxes,
    )
    ax.text(0.50, 0.92, "Panel C question: can image identity be recovered when eye position is latent?", ha="center", fontsize=9.5, transform=ax.transAxes)

    for idx, y in enumerate([0.72, 0.62, 0.52]):
        color = ["#dce7ef", "#e8f2ea", "#f2e8db"][idx]
        ax.add_patch(
            FancyBboxPatch(
                (0.025 + 0.014 * idx, y - 0.032),
                0.06,
                0.050,
                boxstyle="round,pad=0.006,rounding_size=0.006",
                edgecolor="#bac4cc",
                facecolor=color,
                linewidth=0.7,
                transform=ax.transAxes,
            )
        )
    trace_x = np.linspace(0.045, 0.155, 40)
    trace_y = 0.34 + 0.055 * np.sin(np.linspace(0, 2.2 * np.pi, 40))
    ax.plot(trace_x, trace_y, color=COLORS["empirical"], lw=1.3, transform=ax.transAxes)
    ax.plot(trace_x, trace_y - 0.045, color=COLORS["ou"], lw=1.0, alpha=0.9, transform=ax.transAxes)

    _save(fig, out_dir, "C1_observer_schematic")


def plot_c2_accuracy_ordering(primary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True, constrained_layout=True)
    scales = [0.5, 1.0]
    x = np.arange(len(scales))

    for ax, candidate_mode in zip(axes, CANDIDATE_ORDER, strict=True):
        sub = primary[primary["candidate_set_mode"] == candidate_mode]
        known = []
        zero = []
        joint_by_prior = {prior: [] for prior in PRIOR_ORDER}
        for scale in scales:
            scale_rows = sub[sub["observation_scale"] == scale]
            known.append(float(scale_rows["known_eye_accuracy"].iloc[0]))
            zero.append(float(scale_rows["zero_eye_accuracy"].iloc[0]))
            for prior in PRIOR_ORDER:
                joint_by_prior[prior].append(
                    float(scale_rows[scale_rows["prior_family"] == prior]["joint_eye_accuracy"].iloc[0])
                )
        ax.plot(x, known, marker="o", color=COLORS["known"], lw=2.0, label="known eye")
        ax.plot(x, zero, marker="o", color=COLORS["zero"], lw=2.0, label="zero eye")
        for prior in PRIOR_ORDER:
            ax.plot(x, joint_by_prior[prior], marker="o", color=COLORS[prior], lw=2.0, label=PRIOR_LABELS[prior])
        ax.set_title(CANDIDATE_LABELS[candidate_mode])
        ax.set_xticks(x, [_scale_label(v) for v in scales])
        ax.set_xlabel("motion scale")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", color="#d8dde3", lw=0.8)
        _clean_axis(ax)

    axes[0].set_ylabel("image-identification accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Trajectory marginalization rescues image identity")
    _save(fig, out_dir, "C2_accuracy_ordering")
    return primary.copy()


def plot_c3_matched_static_rescue(primary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    key = primary[
        (primary["candidate_set_mode"] == "matched_static_response") & (primary["observation_scale"] == 1.0)
    ].copy()
    emp = key[key["prior_family"] == "empirical"].iloc[0]
    ou = key[key["prior_family"] == "ou"].iloc[0]
    values = pd.DataFrame(
        [
            {"observer": "zero eye", "accuracy": emp.zero_eye_accuracy, "color_key": "zero", "recovery_fraction": np.nan},
            {
                "observer": "joint empirical",
                "accuracy": emp.joint_eye_accuracy,
                "color_key": "empirical",
                "recovery_fraction": emp.joint_minus_zero_accuracy / emp.known_minus_zero_accuracy,
            },
            {
                "observer": "joint OU",
                "accuracy": ou.joint_eye_accuracy,
                "color_key": "ou",
                "recovery_fraction": ou.joint_minus_zero_accuracy / ou.known_minus_zero_accuracy,
            },
            {"observer": "known eye", "accuracy": emp.known_eye_accuracy, "color_key": "known", "recovery_fraction": 1.0},
        ]
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    x = np.arange(len(values))
    colors = [COLORS[key] for key in values["color_key"]]
    ax.bar(x, values["accuracy"], color=colors, width=0.68)
    ax.set_xticks(x, values["observer"], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Matched-static control at 1.0x")
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    _clean_axis(ax)

    for idx, row in values.iterrows():
        ax.text(idx, float(row["accuracy"]) + 0.025, f"{row['accuracy']:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.text(
        0.03,
        0.96,
        f"known-zero gap recovered:\nempirical {values.iloc[1].recovery_fraction:.0%}; OU {values.iloc[2].recovery_fraction:.0%}",
        ha="left",
        va="top",
        fontsize=8,
        color="#303840",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d8dde3", "linewidth": 0.7},
    )
    _save(fig, out_dir, "C3_matched_static_rescue")
    return values


def plot_c4_posterior_concentration(primary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    block = primary.copy()
    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    for candidate_mode in CANDIDATE_ORDER:
        for prior in PRIOR_ORDER:
            sub = block[
                (block["candidate_set_mode"] == candidate_mode) & (block["prior_family"] == prior)
            ].sort_values("observation_scale")
            linestyle = "-" if candidate_mode == "matched_static_response" else "--"
            marker = "o" if prior == "empirical" else "s"
            ax.plot(
                sub["observation_scale"],
                sub["median_N_eff_fraction"],
                color=COLORS[prior],
                linestyle=linestyle,
                marker=marker,
                lw=1.8,
                label=f"{CANDIDATE_LABELS[candidate_mode]}, {PRIOR_LABELS[prior]}",
            )
    ax.set_xlabel("motion scale")
    ax.set_ylabel("median N_eff / K")
    ax.set_xticks([0.5, 1.0], ["0.5x", "1.0x"])
    ax.set_ylim(0.25, 0.75)
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Posterior trajectory mass concentrates at 1.0x")
    _clean_axis(ax)
    _save(fig, out_dir, "C4_posterior_concentration")
    return block[
        [
            "candidate_set_mode",
            "prior_family",
            "observation_scale",
            "likelihood_scale",
            "median_N_eff_fraction",
            "median_nearest_tau_rank",
        ]
    ].copy()


def plot_c5_gap_guardrail(observer: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for candidate_mode in CANDIDATE_ORDER:
        for scale in [0.5, 1.0]:
            sub = observer[
                (observer["candidate_set_mode"] == candidate_mode) & (observer["observation_scale"] == scale)
            ]
            rows.append(
                {
                    "candidate_set_mode": candidate_mode,
                    "observation_scale": scale,
                    "zero_accuracy": float(sub["zero_eye_accuracy"].iloc[0]),
                    "joint_min": float(sub["joint_eye_accuracy"].min()),
                    "joint_max": float(sub["joint_eye_accuracy"].max()),
                    "gap_min": float(sub["joint_minus_zero_accuracy"].min()),
                    "gap_max": float(sub["joint_minus_zero_accuracy"].max()),
                }
            )
    block = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    offsets = {"hard_negative_structure": -0.06, "matched_static_response": 0.06}
    for candidate_mode in CANDIDATE_ORDER:
        sub = block[block["candidate_set_mode"] == candidate_mode].sort_values("observation_scale")
        x = sub["observation_scale"].to_numpy(dtype=float) + offsets[candidate_mode]
        y = (sub["gap_min"].to_numpy(dtype=float) + sub["gap_max"].to_numpy(dtype=float)) / 2.0
        yerr = np.vstack([y - sub["gap_min"].to_numpy(dtype=float), sub["gap_max"].to_numpy(dtype=float) - y])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=COLORS[candidate_mode],
            marker="o",
            lw=1.8,
            capsize=0,
            label=CANDIDATE_LABELS[candidate_mode],
        )
    ax.set_xlabel("motion scale")
    ax.set_ylabel("joint minus zero accuracy")
    ax.set_xticks([0.5, 1.0], ["0.5x", "1.0x"])
    ax.set_ylim(0.0, 0.62)
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Rescue grows as zero-eye fails")
    _clean_axis(ax)
    _save(fig, out_dir, "C5_scale_gap_guardrail")
    return block


def plot_c6_compact_guardrail(compact_dir: Path, out_dir: Path) -> pd.DataFrame:
    path = compact_dir / "followup_summary/compact_mechanism_promotion_gates.csv"
    gates = pd.read_csv(path)
    required = {
        "candidate_set_mode",
        "prior_condition",
        "motion_scale",
        "likelihood_scale",
        "k_dim",
        "full_joint_accuracy",
        "zero_joint_accuracy",
        "compact_joint_accuracy",
        "random_joint_accuracy",
        "static_pc_joint_accuracy",
        "compact_removed_joint_accuracy",
    }
    missing = sorted(required.difference(gates.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    block = gates[
        (gates["candidate_set_mode"] == "matched_static_response")
        & (gates["prior_condition"] == "empirical")
        & (pd.to_numeric(gates["motion_scale"]) == 1.0)
        & (pd.to_numeric(gates["likelihood_scale"]) == 1.0)
    ].copy()
    block["k_dim"] = pd.to_numeric(block["k_dim"])
    block = block.sort_values("k_dim")

    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    x = block["k_dim"].to_numpy(dtype=float)
    ax.axhline(float(block["full_joint_accuracy"].iloc[0]), color=COLORS["full"], lw=1.5, label="full exact")
    ax.axhline(float(block["zero_joint_accuracy"].iloc[0]), color=COLORS["zero"], lw=1.2, linestyle="--", label="zero eye")
    ax.plot(x, block["compact_joint_accuracy"], color=COLORS["compact"], marker="o", lw=1.8, label="compact only")
    ax.plot(x, block["static_pc_joint_accuracy"], color=COLORS["static_pc"], marker="s", lw=1.8, label="static-PC")
    ax.plot(x, block["random_joint_accuracy"], color=COLORS["random"], marker="^", lw=1.5, label="random")
    ax.plot(
        x,
        block["compact_removed_joint_accuracy"],
        color=COLORS["removed"],
        marker="v",
        lw=1.5,
        label="compact removed",
    )
    ax.set_xlabel("basis dimension")
    ax.set_ylabel("joint accuracy")
    ax.set_xticks(x, [str(int(v)) for v in x])
    ax.set_ylim(0.25, 0.82)
    ax.grid(axis="y", color="#d8dde3", lw=0.8)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    ax.set_title("Compact projection is partial, not unique")
    _clean_axis(ax)
    _save(fig, out_dir, "C6_compact_mechanism_guardrail")
    return block


def _write_caption(out_dir: Path) -> None:
    caption = """# Panel C Subpanels

Generated cache-only from the BackImage trajectory-table observer run:

```text
outputs/fixation_statistics_by_stimulus_all_sessions_after_review/
  backimage_trajectory_table_observer_confirm_matched_static_n64_c8_k8_v1/
```

Subpanels:

- `C1_observer_schematic`: exact response-table observer schematic.
- `C2_accuracy_ordering`: known-eye, zero-eye, and latent-eye joint accuracy across
  hard-negative and matched-static candidate sets.
- `C3_matched_static_rescue`: focused matched-static 1.0x rescue panel.
- `C4_posterior_concentration`: median posterior N_eff / K across scale.
- `C5_scale_gap_guardrail`: joint-minus-zero-eye rescue range across priors and
  likelihood scales.
- `C6_compact_mechanism_guardrail`: compact projection followup for the
  matched-static empirical-prior 1.0x slice.

Claim boundary:

```text
The exact trajectory-table observer supports latent-pose marginalization as a
usable image-identification strategy. Zero-eye means the moved observation is
scored under a zero-eye-motion assumption, not that the input movie was static.
Latent-eye joint means the measured eye trace is hidden and the observer
marginalizes over candidate trajectories. Compact projection results are a
mechanistic guardrail and do not establish a unique mechanism; static-PC and
other controls remain close in some slices.
```
"""
    (out_dir / "panel_C_subpanels_caption.md").write_text(caption)


def _write_index(out_dir: Path, generated: Iterable[str]) -> None:
    lines = ["# Panel C Generated Assets", ""]
    for stem in generated:
        lines.append(f"- `{stem}.png`")
        lines.append(f"- `{stem}.pdf`")
    lines.extend(
        [
            "- `panel_C_accuracy_ordering_values.csv`",
            "- `panel_C_matched_static_rescue_values.csv`",
            "- `panel_C_posterior_concentration_values.csv`",
            "- `panel_C_scale_gap_guardrail_values.csv`",
            "- `panel_C_compact_guardrail_values.csv`",
            "- `panel_C_subpanels_caption.md`",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer-dir", type=Path, default=DEFAULT_OBSERVER_DIR)
    parser.add_argument("--compact-dir", type=Path, default=DEFAULT_COMPACT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    _configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    observer = _load_observer(args.observer_dir)
    primary = _primary_rows(observer)

    plot_c1_observer_schematic(args.out_dir)
    accuracy_values = plot_c2_accuracy_ordering(primary, args.out_dir)
    rescue_values = plot_c3_matched_static_rescue(primary, args.out_dir)
    posterior_values = plot_c4_posterior_concentration(primary, args.out_dir)
    gap_values = plot_c5_gap_guardrail(observer, args.out_dir)
    compact_values = plot_c6_compact_guardrail(args.compact_dir, args.out_dir)

    accuracy_values.to_csv(args.out_dir / "panel_C_accuracy_ordering_values.csv", index=False)
    rescue_values.to_csv(args.out_dir / "panel_C_matched_static_rescue_values.csv", index=False)
    posterior_values.to_csv(args.out_dir / "panel_C_posterior_concentration_values.csv", index=False)
    gap_values.to_csv(args.out_dir / "panel_C_scale_gap_guardrail_values.csv", index=False)
    compact_values.to_csv(args.out_dir / "panel_C_compact_guardrail_values.csv", index=False)
    _write_caption(args.out_dir)
    _write_index(
        args.out_dir,
        [
            "C1_observer_schematic",
            "C2_accuracy_ordering",
            "C3_matched_static_rescue",
            "C4_posterior_concentration",
            "C5_scale_gap_guardrail",
            "C6_compact_mechanism_guardrail",
        ],
    )


if __name__ == "__main__":
    main()
