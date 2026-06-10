"""
Compose a combined Figure 2/3 main figure:

    A: compact eye/rate matching example
    B: FEM fraction of rate modulation
    C: Fano factor before/after FEM correction
    D: compact covariance decomposition
    E: mean pairwise Fisher-z noise correlations before/after correction
    F: PSTH/FEM cumulative eigenspectra
    G: PSTH/FEM participation ratio
    H: PSTH/FEM subspace alignment

Usage:
    uv run declan/fig2/generate_figure2_3_combined.py
    uv run declan/fig2/generate_figure2_3_combined.py --refresh
"""
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from VisionCore.covariance import project_to_psd
from _panel_common import FIG_DIR
from compute_fig2_data import load_fig2_data
from generate_fig2a import (
    DT as EYE_DT,
    TRIAL_A,
    TRIAL_B,
    UNIT_ORIG,
    W1,
    W2,
    WINDOW_BINS as EYE_WINDOW_BINS,
    _load_trial_pair,
)
from generate_fig2c import plot_panel_c as plot_fem_fraction
from generate_fig2e import plot_panel_e as plot_fano
from generate_fig3c import plot_panel_c as plot_noise_corr
from generate_fig3d import plot_panel_d as plot_noise_corr_delta
from generate_fig3e import plot_panel_e as plot_eigenspectra
from generate_fig3f import plot_panel_f as plot_participation_ratio
from generate_fig3g import plot_panel_g as plot_subspace_alignment

TARGET_SESSION = "Allen_2022-02-16"
WINDOW_IDX = 0
OMIT_SUBJECTS = {"Luke"}


def _label(ax, letter):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=10)


def _normalize_axis_text(ax):
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.tick_params(labelsize=7)


def _style_matrix_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("k")
        ax.spines[side].set_linewidth(0.8)


def _despine(ax, left=False, bottom=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def _filter_subjects(data, omit=OMIT_SUBJECTS):
    """Return a shallow plotting bundle with noisy subjects omitted."""
    if not omit:
        return data

    filtered = dict(data)
    filtered["SUBJECTS"] = [s for s in data["SUBJECTS"] if s not in omit]
    filtered["SUBJECT_COLORS"] = {
        s: c for s, c in data["SUBJECT_COLORS"].items() if s not in omit
    }

    if "m_by_window" in data and "subject_per_neuron_by_window" in data:
        m_by_window = []
        labels_by_window = []
        for values, labels in zip(
            data["m_by_window"], data["subject_per_neuron_by_window"]
        ):
            labels = np.asarray(labels)
            keep = ~np.isin(labels, list(omit))
            m_by_window.append(np.asarray(values)[keep])
            labels_by_window.append(labels[keep])
        filtered["m_by_window"] = m_by_window
        filtered["subject_per_neuron_by_window"] = labels_by_window

    sub_subjects = np.asarray(data.get("sub_subjects", []))
    if sub_subjects.size:
        keep_sub = ~np.isin(sub_subjects, list(omit))
        old_to_new = {
            old_i: new_i for new_i, old_i in enumerate(np.flatnonzero(keep_sub))
        }

        for key in (
            "sub_names",
            "sub_subjects",
            "pr_fem_list",
            "pr_psth_list",
            "overlap_k1_list",
            "overlap_k_list",
            "var_p_given_f",
            "var_f_given_p",
            "spectra_psth",
            "spectra_fem",
        ):
            if key not in data:
                continue
            values = np.asarray(data[key], dtype=object)
            filtered[key] = values[keep_sub].tolist()

        null_session_idx = np.asarray(data.get("null_session_idx", []), dtype=int)
        null_subjects = np.asarray(data.get("null_subjects", []))
        if null_session_idx.size and null_subjects.size:
            keep_null = (
                ~np.isin(null_subjects, list(omit))
                & np.isin(null_session_idx, list(old_to_new))
            )
            filtered["null_session_idx"] = [
                old_to_new[int(i)] for i in null_session_idx[keep_null]
            ]
            filtered["null_subjects"] = null_subjects[keep_null].tolist()
            for key in (
                "null_var_p_given_f",
                "null_var_f_given_p",
                "null_overlap_k1",
                "null_overlap_k",
            ):
                if key in data:
                    filtered[key] = np.asarray(data[key])[keep_null].tolist()

    return filtered


def _plot_compact_eye_example(fig, subplot_spec):
    gs = GridSpecFromSubplotSpec(
        3,
        1,
        subplot_spec=subplot_spec,
        height_ratios=[1.0, 0.42, 1.0],
        hspace=0.08,
    )
    ax_eye = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[1, 0], sharex=ax_eye)
    ax_spk = fig.add_subplot(gs[2, 0], sharex=ax_eye)

    pair = _load_trial_pair()
    robs = pair["robs"]
    eyepos = pair["eyepos"]
    neuron_mask = np.asarray(pair["neuron_mask"])
    j = int(np.where(neuron_mask == UNIT_ORIG)[0][0])

    W = EYE_WINDOW_BINS
    t_ms = np.arange(W) * EYE_DT * 1000.0
    e_a = eyepos[TRIAL_A, :W, 0]
    e_b = eyepos[TRIAL_B, :W, 0]
    r_a = robs[TRIAL_A, :W, j] / EYE_DT
    r_b = robs[TRIAL_B, :W, j] / EYE_DT
    delta_e = np.abs(e_a - e_b)

    color_a, color_b = "tab:cyan", "tab:red"
    for ax in (ax_eye, ax_delta, ax_spk):
        for w in (W1, W2):
            ax.axvspan(w[0] * EYE_DT * 1000.0, w[1] * EYE_DT * 1000.0,
                       color="0.86", zorder=-1)

    ax_eye.plot(t_ms, e_a, color=color_a, lw=1.2)
    ax_eye.plot(t_ms, e_b, color=color_b, lw=1.2)
    label_y = 1.04
    for text, w in [
        ("mismatched\ntrajectories", W1),
        ("closely matched\ntrajectories", W2),
    ]:
        t_mid = 0.5 * (w[0] + w[1]) * EYE_DT * 1000.0
        ax_eye.text(t_mid, label_y, text, transform=ax_eye.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.5, clip_on=False)
    ax_eye.set_ylabel("Eye\npos.")
    ax_eye.set_yticks([])
    _despine(ax_eye, left=True, bottom=True)
    plt.setp(ax_eye.get_xticklabels(), visible=False)

    ax_delta.fill_between(t_ms, 0.0, delta_e, color="0.55", alpha=0.55, lw=0)
    ax_delta.plot(t_ms, delta_e, color="0.25", lw=0.8)
    ax_delta.set_ylabel("Δ eye")
    ax_delta.set_yticks([])
    _despine(ax_delta, left=True, bottom=True)
    plt.setp(ax_delta.get_xticklabels(), visible=False)

    ymax = float(max(r_a.max(), r_b.max(), 1.0))
    offset = 1.2 * ymax
    ax_spk.step(t_ms, r_b + offset, color=color_b, lw=1.1, where="mid")
    ax_spk.step(t_ms, r_a, color=color_a, lw=1.1, where="mid")
    ax_spk.axhline(0, color="0.72", lw=0.5, zorder=-1)
    ax_spk.axhline(offset, color="0.72", lw=0.5, zorder=-1)
    ax_spk.set_xlabel("Time from fixation onset (ms)")
    ax_spk.set_ylabel("Rate")
    ax_spk.set_yticks([])
    ax_spk.set_xlim(0, t_ms[-1])
    _despine(ax_spk, left=True)

    ax_eye.text(-0.12, 1.20, "A", transform=ax_eye.transAxes,
                fontweight="bold", fontsize=10, va="top", ha="left")
    return ax_spk


def _make_compact_cov_axes(fig, subplot_spec):
    gs = GridSpecFromSubplotSpec(
        2,
        5,
        subplot_spec=subplot_spec,
        width_ratios=[1, 0.16, 1, 0.16, 1],
        height_ratios=[0.78, 1.0],
        wspace=0.04,
        hspace=0.30,
    )
    top_mats = [fig.add_subplot(gs[0, c]) for c in (0, 2, 4)]
    bot_mats = [fig.add_subplot(gs[1, c]) for c in (0, 2, 4)]
    top_seps = [fig.add_subplot(gs[0, c]) for c in (1, 3)]
    bot_seps = [fig.add_subplot(gs[1, c]) for c in (1, 3)]
    for ax in top_seps + bot_seps:
        ax.axis("off")
    return top_mats, top_seps, bot_mats, bot_seps


def _plot_compact_cov_decomp(fig, subplot_spec, data, letter="D"):
    top_axes, top_sep_axes, bot_axes, bot_sep_axes = _make_compact_cov_axes(
        fig, subplot_spec
    )
    sr = next((s for s in data["session_results"]
               if s["session"] == TARGET_SESSION), None)
    if sr is None:
        avail = [s["session"] for s in data["session_results"]]
        raise ValueError(f"{TARGET_SESSION} not in fig2 cache. Available: {avail}")

    mats = sr["mats"][WINDOW_IDX]
    crate_raw = mats["Intercept"]
    valid = (
        np.isfinite(np.diag(crate_raw))
        & np.isfinite(np.diag(mats["PSTH"]))
    )
    ix = np.ix_(valid, valid)
    ctotal = project_to_psd(mats["Total"][ix])
    cpsth = project_to_psd(mats["PSTH"][ix])
    cfem = project_to_psd(crate_raw[ix] - mats["PSTH"][ix])
    cint = project_to_psd(mats["Total"][ix] - crate_raw[ix])
    cint_uncorr = project_to_psd(mats["Total"][ix] - mats["PSTH"][ix])

    cmax = float(np.nanmax(ctotal))
    cmap = plt.get_cmap("seismic_r")
    top_titles = [
        r"$\Sigma_{\mathrm{total}}$",
        r"$\Sigma_{\mathrm{PSTH}}$",
        r"$\Sigma_{\mathrm{int}}^{\mathrm{uncorr}}$",
    ]
    bot_titles = [
        r"$\Sigma_{\mathrm{int}}^{\mathrm{uncorr}}$",
        r"$\Sigma_{\mathrm{FEM}}$",
        r"$\Sigma_{\mathrm{int}}^{\mathrm{corr}}$",
    ]
    for ax, mat, title, frac in zip(
        top_axes,
        [ctotal, cpsth, cint_uncorr],
        top_titles,
        [0.5, 0.25, 0.5],
    ):
        vlim = frac * cmax
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  vmin=-vlim, vmax=vlim, aspect="equal")
        ax.set_title(title, fontsize=8, pad=3)
        _style_matrix_axis(ax)

    for ax, mat, title, frac in zip(
        bot_axes,
        [cint_uncorr, cfem, cint],
        bot_titles,
        [0.5, 0.5, 0.5],
    ):
        vlim = frac * cmax
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  vmin=-vlim, vmax=vlim, aspect="equal")
        ax.set_title(title, fontsize=8, pad=3)
        _style_matrix_axis(ax)

    for ax, sym in zip(top_sep_axes, ["=", "+"]):
        ax.text(0.5, 0.5, sym, ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
    for ax, sym in zip(bot_sep_axes, [r"$\approx$", "+"]):
        ax.text(0.5, 0.5, sym, ha="center", va="center",
                fontsize=12, transform=ax.transAxes)

    top_axes[0].text(-0.13, 1.16, letter, transform=top_axes[0].transAxes,
                     fontweight="bold", fontsize=10, va="top", ha="left")
    top_axes[0].set_ylabel("Classical", fontsize=8, labelpad=5)
    bot_axes[0].set_ylabel("Eye-position\nsplit", fontsize=8, labelpad=5)


def _add_condition_legend(ax, loc="upper left"):
    handles = [
        Line2D([0], [0], color="0.2", lw=1.4, ls="--",
               marker="o", markerfacecolor="white", markeredgecolor="0.2",
               markersize=4, label="uncorrected"),
        Line2D([0], [0], color="0.2", lw=1.4, ls="-",
               marker="o", markerfacecolor="0.2", markeredgecolor="0.2",
               markersize=4, label="FEM-corrected"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc=loc,
              handlelength=1.8, borderpad=0.1, labelspacing=0.25)


def _plot_noise_corr_pair(fig, subplot_spec, data):
    gs = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=subplot_spec,
        width_ratios=[1.15, 0.95],
        wspace=0.42,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    _, primary = plot_noise_corr(ax=ax_main, data=data)
    _normalize_axis_text(primary)
    _label(primary, "E")
    _add_condition_legend(primary, loc="upper left")

    ax_delta = fig.add_subplot(gs[0, 1])
    _, delta_primary = plot_noise_corr_delta(ax=ax_delta, data=data)
    _normalize_axis_text(delta_primary)
    delta_primary.set_title("Shuffle control", loc="left", fontsize=8)
    delta_primary.set_ylabel(r"$\Delta z$")
    legend = delta_primary.get_legend()
    if legend is not None:
        legend.set_title(None)
        legend.set_bbox_to_anchor((1.02, 1.02))
        legend._loc = 1
    return primary, delta_primary


def _add_pr_annotations(ax):
    ax.text(
        0.05,
        0.94,
        "FEM PR approx. 2-3",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0),
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("PSTH PR > FEM PR")
        legend.get_title().set_fontsize(6)


def compose(refresh=False):
    data = _filter_subjects(load_fig2_data(refresh=refresh))

    # Tuned for an ~8.5" manuscript-width figure. Rows follow the argument:
    # single-neuron/rate, population covariance, removed-component structure.
    rc = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    }
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(8.5, 9.2))
        gs = GridSpec(
            3,
            6,
            height_ratios=[1.05, 1.10, 1.15],
            hspace=0.32,
            wspace=0.95,
            figure=fig,
            left=0.090,
            right=0.980,
            top=0.965,
            bottom=0.080,
        )

        _plot_compact_eye_example(fig, gs[0, 0:2])

        panel_specs = [
            ("B", plot_fem_fraction, gs[0, 2:4]),
            ("C", plot_fano, gs[0, 4:6]),
            ("F", plot_eigenspectra, gs[2, 0:2]),
            ("G", plot_participation_ratio, gs[2, 2:4]),
            ("H", plot_subspace_alignment, gs[2, 4:6]),
        ]
        _plot_compact_cov_decomp(fig, gs[1, 0:3], data, letter="D")
        _plot_noise_corr_pair(fig, gs[1, 3:6], data)
        for letter, plot_fn, spec in panel_specs:
            ax = fig.add_subplot(spec)
            _, primary_ax = plot_fn(ax=ax, data=data)
            _normalize_axis_text(primary_ax)
            _label(primary_ax, letter)
            if letter == "C":
                _add_condition_legend(primary_ax, loc="upper left")
            elif letter == "G":
                _add_pr_annotations(primary_ax)

    stem = FIG_DIR / "fig2_3_combined"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight",
                pad_inches=0.08, dpi=300)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight",
                pad_inches=0.08, dpi=200)
    plt.close(fig)
    print(f"\nSaved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")


def _parse_args():
    p = argparse.ArgumentParser(description="Compose combined figure 2/3.")
    p.add_argument(
        "-r",
        "--refresh",
        action="store_true",
        help="Force recompute of derived fig2 data.",
    )
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    compose(refresh=args.refresh)
