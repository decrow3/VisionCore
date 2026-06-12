"""
Compose a combined Figure 2/3 main figure:

    A: compact eye/rate matching example
    B: FEM fraction of rate modulation
    C: Fano factor before/after FEM correction
    D: compact covariance decomposition
    E: mean pairwise Fisher-z noise correlations before/after correction
    F: pairwise noise correlations at 8 ms
    G: PSTH/FEM participation ratio
    H: PSTH/FEM subspace alignment

Usage:
    uv run declan/fig2/generate_figure2_3_combined.py
    uv run declan/fig2/generate_figure2_3_combined.py --split-subjects
    uv run declan/fig2/generate_figure2_3_combined.py --refresh
"""
import argparse
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Polygon

from VisionCore.covariance import project_to_psd
from _panel_common import FIG_DIR
from compute_fig2_data import load_fig2_data, _compute_fano_stats, _compute_nc_stats
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
from generate_fig3b import plot_panel_b as plot_pairwise_noise_corr
from generate_fig3c import plot_panel_c as plot_noise_corr
from generate_fig3f import plot_panel_f as plot_participation_ratio
from generate_fig3g import plot_panel_g as plot_subspace_alignment

TARGET_SESSION = "Allen_2022-02-16"
WINDOW_IDX = 0
OMIT_SUBJECTS = {"Luke"}
POOLED_SUBJECT = "Pooled"
POOLED_COLOR = "0.25"


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


def _add_diagonal_band_box(ax, n, band_frac=0.055):
    band = max(3.0, n * band_frac)
    extra = max(3.0, n * 0.04)
    p0 = np.array([-0.5 - extra, -0.5 - extra], dtype=float)
    p1 = np.array([n - 0.5 + extra, n - 0.5 + extra], dtype=float)
    normal = np.array([1.0, -1.0]) / np.sqrt(2.0)
    pts = [
        p0 + normal * band / 2,
        p1 + normal * band / 2,
        p1 - normal * band / 2,
        p0 - normal * band / 2,
    ]
    ax.add_patch(Polygon(
        pts,
        closed=True,
        fill=False,
        edgecolor="0.45",
        linewidth=0.9,
        alpha=0.8,
        zorder=12,
        clip_on=False,
    ))
    ax.text(
        1.02,
        0.05,
        "independent",
        transform=ax.transAxes,
        ha="left",
        va="center",
        rotation=-45,
        fontsize=6.5,
        color="0.35",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.7),
        zorder=13,
        clip_on=False,
    )


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

    if "metrics" in data:
        filtered_metrics = []
        for m_dict in data["metrics"]:
            m = copy.copy(m_dict)
            neuron_mask = ~np.isin(m_dict["subject_per_neuron"], list(omit))
            pair_mask = ~np.isin(m_dict["subject_per_pair"], list(omit))
            ds_mask = ~np.isin(m_dict["subject_by_ds"], list(omit))
            shuff_mask = ~np.isin(m_dict["shuff_rho_subject"], list(omit))

            for key in ("alpha", "uncorr", "corr", "erate",
                        "subject_per_neuron", "session_per_neuron"):
                m[key] = np.asarray(m_dict[key])[neuron_mask]
            if "shuff_var_c" in m_dict:
                m["shuff_var_c"] = np.asarray(m_dict["shuff_var_c"])[neuron_mask]

            for key in ("rho_uncorr", "rho_corr", "subject_per_pair"):
                m[key] = np.asarray(m_dict[key])[pair_mask]

            for key in ("rho_u_meanz_by_ds", "rho_c_meanz_by_ds",
                        "rho_delta_meanz_by_ds", "subject_by_ds"):
                m[key] = np.asarray(m_dict[key])[ds_mask]
            for key in ("Ctotal", "Cpsth", "Crate", "CnoiseU",
                        "CnoiseC", "Cfem"):
                if key in m_dict:
                    values = np.asarray(m_dict[key], dtype=object)
                    m[key] = values[ds_mask].tolist()

            for key in ("shuff_rho_delta_meanz", "shuff_rho_c_meanz",
                        "shuff_rho_subject"):
                if key in m_dict:
                    m[key] = np.asarray(m_dict[key])[shuff_mask]

            filtered_metrics.append(m)

        filtered["metrics"] = filtered_metrics
        filtered["fano_stats"] = _compute_fano_stats(
            filtered_metrics, filtered["WINDOWS_MS"]
        )
        filtered["nc_stats"] = _compute_nc_stats(
            filtered_metrics, filtered["WINDOWS_MS"]
        )

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


def _pool_subjects_for_plotting(data, label=POOLED_SUBJECT):
    """Relabel included subjects as one plotting group without changing values."""
    pooled = copy.copy(data)
    pooled["SUBJECTS"] = [label]
    pooled["SUBJECT_COLORS"] = {label: POOLED_COLOR}
    pooled["SUBJECT_DISPLAY_NAMES"] = {label: "pooled"}

    if "m_by_window" in data and "subject_per_neuron_by_window" in data:
        pooled["m_by_window"] = [np.asarray(v) for v in data["m_by_window"]]
        pooled["subject_per_neuron_by_window"] = [
            np.full(np.asarray(labels).shape, label, dtype=object)
            for labels in data["subject_per_neuron_by_window"]
        ]

    metrics = []
    for m_dict in data.get("metrics", []):
        m = copy.copy(m_dict)
        for key in ("subject_by_ds", "subject_per_neuron", "subject_per_pair"):
            if key in m:
                m[key] = np.full(np.asarray(m[key]).shape, label, dtype=object)
        if "shuff_rho_subject" in m:
            m["shuff_rho_subject"] = np.full(
                np.asarray(m["shuff_rho_subject"]).shape, label, dtype=object
            )
        metrics.append(m)
    if metrics:
        pooled["metrics"] = metrics

    if "fano_stats" in data:
        fano_stats = {}
        for window, stats in data["fano_stats"].items():
            s = copy.copy(stats)
            s["per_subject"] = {
                label: {
                    "slope_unc": stats["slope_unc"],
                    "slope_cor": stats["slope_cor"],
                    "slope_unc_ci": stats["slope_unc_ci"],
                    "slope_cor_ci": stats["slope_cor_ci"],
                    "slope_diff": stats["slope_diff"],
                    "slope_diff_ci": stats["slope_diff_ci"],
                    "p_slope": stats["p_slope"],
                    "n_sessions": stats["n_sessions"],
                    "n": stats["n"],
                }
            }
            if "subject_per_neuron" in s:
                s["subject_per_neuron"] = np.full(
                    np.asarray(s["subject_per_neuron"]).shape,
                    label,
                    dtype=object,
                )
            fano_stats[window] = s
        pooled["fano_stats"] = fano_stats

    for key in ("sub_subjects", "null_subjects"):
        if key in data:
            pooled[key] = [label] * len(data[key])

    return pooled


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
        7,
        subplot_spec=subplot_spec,
        width_ratios=[1.08, 0.14, 1.08, 0.34, 1.08, 0.10, 0.30],
        height_ratios=[1.0, 1.0],
        wspace=0.04,
        hspace=0.30,
    )
    top_mats = [fig.add_subplot(gs[0, c]) for c in (0, 2, 4)]
    bot_mats = [fig.add_subplot(gs[1, c]) for c in (2, 4)]
    top_seps = [fig.add_subplot(gs[0, c]) for c in (1, 3)]
    bot_seps = [fig.add_subplot(gs[1, 3])]
    note_axes = [fig.add_subplot(gs[:, 6])]
    for ax in top_seps + bot_seps + note_axes:
        ax.axis("off")
    return top_mats, top_seps, bot_mats, bot_seps, note_axes


def _shift_axes_down(axes, dy=0.018):
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - dy, pos.width, pos.height])


def _plot_compact_cov_decomp(fig, subplot_spec, data, letter="D"):
    top_axes, top_sep_axes, bot_axes, bot_sep_axes, note_axes = _make_compact_cov_axes(
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
        "Total covariance",
        "Stimulus covariance",
        "Classical residual",
    ]
    bot_titles = [
        "FEM component",
        "Corrected residual",
    ]
    for ax, mat, title, frac in zip(
        top_axes,
        [ctotal, cpsth, cint_uncorr],
        top_titles,
        [0.35, 0.18, 0.35],
    ):
        vlim = frac * cmax
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  vmin=-vlim, vmax=vlim, aspect="equal")
        ax.set_title(title, fontsize=8, pad=3)
        _style_matrix_axis(ax)

    for ax, mat, title, frac in zip(
        bot_axes,
        [cfem, cint],
        bot_titles,
        [0.35, 0.35],
    ):
        vlim = frac * cmax
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  vmin=-vlim, vmax=vlim, aspect="equal")
        ax.set_xlabel(title, fontsize=8, labelpad=4)
        _style_matrix_axis(ax)

    for ax, sym in zip(top_sep_axes, ["=", "+"]):
        ax.text(0.5, 0.5, sym, ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
    for ax, sym in zip(bot_sep_axes, ["+"]):
        ax.text(0.5, 0.5, sym, ha="center", va="center",
                fontsize=12, transform=ax.transAxes)

    _add_diagonal_band_box(bot_axes[1], cint.shape[0])

    top_residual_ax = top_axes[2]
    arrow_kw = dict(
        arrowstyle="-|>",
        mutation_scale=9,
        lw=0.9,
        color="0.45",
        alpha=0.85,
        shrinkA=9,
        shrinkB=9,
        zorder=10,
    )
    fig.add_artist(ConnectionPatch(
        xyA=(0.42, 0.01), coordsA=top_residual_ax.transAxes,
        xyB=(0.78, 1.02), coordsB=bot_axes[0].transAxes,
        **arrow_kw))
    fig.add_artist(ConnectionPatch(
        xyA=(0.58, 0.01), coordsA=top_residual_ax.transAxes,
        xyB=(0.22, 1.02), coordsB=bot_axes[1].transAxes,
        **arrow_kw))

    top_axes[0].text(-0.13, 1.16, letter, transform=top_axes[0].transAxes,
                     fontweight="bold", fontsize=10, va="top", ha="left")
    return top_axes + top_sep_axes + bot_axes + bot_sep_axes + note_axes


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


def _plot_noise_corr_panel(fig, subplot_spec, data):
    ax_main = fig.add_subplot(subplot_spec)
    _, primary = plot_noise_corr(ax=ax_main, data=data)
    _normalize_axis_text(primary)
    _label(primary, "E")
    _add_condition_legend(primary, loc="upper left")
    return primary


def _add_pr_annotations(ax):
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("PSTH PR > FEM PR")
        legend.get_title().set_fontsize(6)


def compose(refresh=False, split_subjects=False):
    data = _filter_subjects(load_fig2_data(refresh=refresh))
    if not split_subjects:
        data = _pool_subjects_for_plotting(data)

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
            height_ratios=[0.95, 1.38, 1.05],
            hspace=0.30,
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
            ("F", plot_pairwise_noise_corr, gs[2, 0:2]),
            ("G", plot_participation_ratio, gs[2, 2:4]),
            ("H", plot_subspace_alignment, gs[2, 4:6]),
        ]
        middle_axes = []
        middle_axes.extend(_plot_compact_cov_decomp(fig, gs[1, 0:4],
                                                    data, letter="D"))
        middle_axes.append(_plot_noise_corr_panel(fig, gs[1, 4:6], data))
        _shift_axes_down(middle_axes)
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
    p.add_argument(
        "--split-subjects",
        action="store_true",
        help="Plot included subjects separately instead of pooling them.",
    )
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    compose(refresh=args.refresh, split_subjects=args.split_subjects)
