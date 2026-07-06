"""Build a focused 1x-motion versus 0x-stabilized representation diagnostic.

This is a cache-only analysis for the Figure 4C question:

    Does the V1 twin population represent image features better with the
    measured motion response than with the stabilized zero-motion
    counterfactual?

The deterministic known-trace comparison is known-eye minus zero-static. The
known-eye row uses the measured trajectory in the candidate observer, so it
removes latent-eye uncertainty inside that deterministic table. It should not
be read as an independent response target. The full-joint row is kept as the
hidden-eye comparison.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "declan" / "figure4_active_sensing_atlas" / "figures" / "panel_C" / "diagnostics"
FEATURE_DIR = (
    REPO_ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_feature_posterior_compact_removed_pyramid_k8_n128_scales_0p5_1_2_v1"
)
TRIALS_CSV = FEATURE_DIR / "feature_compact_mechanism_trials.csv"

PRIMARY_LATENT = "pyramid_local_field"
PRIMARY_K = 8
COMPACT_K = 10
N_BOOT = 10_000
RNG_SEED = 20260623

VARIANT_LABELS = {
    "zero_static": "0x stabilized",
    "full_exact": "motion, eye hidden",
    "known_eye": "motion, eye known",
}
VARIANT_COLORS = {
    "zero_static": "#66717d",
    "full_exact": "#235789",
    "known_eye": "#111827",
}
CONTRAST_LABELS = {
    "known_eye_minus_zero_static": "oracle 1x gain",
    "full_exact_minus_zero_static": "hidden-eye 1x gain",
    "known_eye_minus_full_exact": "latent-eye penalty",
}
CONTRAST_COLORS = {
    "known_eye_minus_zero_static": "#111827",
    "full_exact_minus_zero_static": "#235789",
    "known_eye_minus_full_exact": "#b35c2e",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9dee5", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _scale_x(scales: pd.Series) -> np.ndarray:
    return scales.astype(float).map({0.5: 0.0, 1.0: 1.0, 2.0: 2.0}).to_numpy()


def _load_trials() -> pd.DataFrame:
    rows = pd.read_csv(TRIALS_CSV)
    selected = rows[
        (rows["candidate_set_mode"] == "hard_negative_structure")
        & (rows["latent"] == PRIMARY_LATENT)
        & (rows["requested_k"].astype(int) == PRIMARY_K)
        & (rows["k_dim"].astype(int) == COMPACT_K)
        & (rows["response_variant"].isin(VARIANT_LABELS))
    ].copy()
    if selected.empty:
        raise ValueError(f"No selected rows found in {TRIALS_CSV}")
    selected["variant_label"] = selected["response_variant"].map(VARIANT_LABELS)
    return selected


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["observation_scale", "response_variant", "variant_label"], as_index=False)
        .agg(
            mean_feature_cosine=("feature_cosine", "mean"),
            median_feature_cosine=("feature_cosine", "median"),
            mean_feature_neg_mse=("feature_neg_mse", "mean"),
            mean_candidate_true_mass=("candidate_posterior_true_mass", "mean"),
            median_candidate_N_eff_fraction=("candidate_posterior_N_eff_fraction", "median"),
            n_rows=("feature_cosine", "size"),
        )
        .sort_values(["observation_scale", "response_variant"])
    )


def _paired_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "table_index",
        "trial_id",
        "candidate_set_mode",
        "observation_scale",
        "prior_family",
        "prior_scale",
        "latent",
        "requested_k",
        "k_dim",
    ]
    pivot = rows.pivot_table(index=key_cols, columns="response_variant", values="feature_cosine", aggfunc="first")
    required = {"zero_static", "full_exact", "known_eye"}
    missing = sorted(required.difference(pivot.columns))
    if missing:
        raise ValueError(f"Missing response variants in paired matrix: {missing}")
    pivot = pivot.dropna(subset=sorted(required)).reset_index()
    return pivot


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(arr, size=(N_BOOT, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _contrasts(paired: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, object]] = []
    contrast_defs = {
        "known_eye_minus_zero_static": paired["known_eye"] - paired["zero_static"],
        "full_exact_minus_zero_static": paired["full_exact"] - paired["zero_static"],
        "known_eye_minus_full_exact": paired["known_eye"] - paired["full_exact"],
    }
    for scale, scale_rows in paired.groupby("observation_scale", sort=True):
        for contrast, all_values in contrast_defs.items():
            values = all_values.loc[scale_rows.index].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean(values, rng)
            rows.append(
                {
                    "observation_scale": float(scale),
                    "contrast": contrast,
                    "contrast_label": CONTRAST_LABELS[contrast],
                    "mean_feature_cosine_delta": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_paired_rows": int(values.size),
                    "fraction_positive": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(rows)


def _write_readme(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    scale1_summary = summary[summary["observation_scale"].astype(float) == 1.0]
    scale1_contrasts = contrasts[contrasts["observation_scale"].astype(float) == 1.0]
    value = {
        row.response_variant: float(row.mean_feature_cosine)
        for row in scale1_summary.itertuples(index=False)
    }
    delta = {
        row.contrast: float(row.mean_feature_cosine_delta)
        for row in scale1_contrasts.itertuples(index=False)
    }
    lines = [
        "# Panel C Motion-Versus-Stabilized Representation Diagnostic",
        "",
        "Question: does the V1 twin population represent the image-feature target",
        "better with measured 1x motion than with the 0x stabilized counterfactual?",
        "",
        "The deterministic known-trace comparison is `known_eye - zero_static`.",
        "`known_eye` uses the measured trajectory in the candidate observer,",
        "so it removes latent eye-position uncertainty inside this table. It",
        "is a known-trace control, not an independent response target.",
        "`full_exact` keeps the eye trace hidden and is included as the",
        "joint-decoder comparison.",
        "",
        "At the 1x scale:",
        "",
        "```text",
        f"0x stabilized feature cosine:        {value['zero_static']:.4f}",
        f"1x motion, eye hidden feature cosine: {value['full_exact']:.4f}",
        f"1x motion, eye known feature cosine:  {value['known_eye']:.4f}",
        "",
        f"known-trace 1x gain over 0x:         {delta['known_eye_minus_zero_static']:.4f}",
        f"hidden-eye 1x gain over 0x:          {delta['full_exact_minus_zero_static']:.4f}",
        f"latent-eye penalty:                  {delta['known_eye_minus_full_exact']:.4f}",
        "```",
        "",
        "Interpretation: the known-trace control supports the claim that the",
        "deterministic moving 1x table contains more recoverable local",
        "image-feature structure than the stabilized 0x counterfactual. The",
        "smaller full-joint gap shows how much of that representational",
        "advantage remains when eye position is hidden.",
        "",
        "Outputs:",
        "",
        "- `panel_C_motion_vs_stabilized_representation.png`",
        "- `panel_C_motion_vs_stabilized_representation.pdf`",
        "- `panel_C_motion_vs_stabilized_representation_summary.csv`",
        "- `panel_C_motion_vs_stabilized_representation_contrasts.csv`",
    ]
    (OUT_DIR / "panel_C_motion_vs_stabilized_representation_README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _plot(summary: pd.DataFrame, contrasts: pd.DataFrame) -> tuple[Path, Path]:
    _configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.85), constrained_layout=True)

    ax = axes[0]
    for variant in ["zero_static", "full_exact", "known_eye"]:
        block = summary[summary["response_variant"] == variant].sort_values("observation_scale")
        ax.plot(
            _scale_x(block["observation_scale"]),
            block["mean_feature_cosine"],
            marker="o",
            lw=2.0,
            color=VARIANT_COLORS[variant],
            linestyle=":" if variant == "known_eye" else "-",
            label=VARIANT_LABELS[variant],
        )
    ax.set_title("A. matched feature recovery")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.set_ylim(0.52, 0.98)
    ax.legend(frameon=False, loc="lower left")
    _clean_axis(ax)

    ax = axes[1]
    for contrast in [
        "known_eye_minus_zero_static",
        "full_exact_minus_zero_static",
        "known_eye_minus_full_exact",
    ]:
        block = contrasts[contrasts["contrast"] == contrast].sort_values("observation_scale")
        x = _scale_x(block["observation_scale"])
        y = block["mean_feature_cosine_delta"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - block["ci_low"].to_numpy(dtype=float),
                block["ci_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            lw=1.7,
            capsize=2.5,
            color=CONTRAST_COLORS[contrast],
            label=CONTRAST_LABELS[contrast],
        )
    ax.axhline(0, color="#6b7280", lw=0.9)
    ax.set_title("B. paired gains")
    ax.set_ylabel("feature cosine difference")
    ax.set_xticks([0, 1, 2], ["0.5x", "1x", "2x"])
    ax.legend(frameon=False, loc="upper left")
    _clean_axis(ax)

    ax = axes[2]
    scale1 = summary[summary["observation_scale"].astype(float) == 1.0].copy()
    order = ["zero_static", "full_exact", "known_eye"]
    scale1["order"] = scale1["response_variant"].map({v: i for i, v in enumerate(order)})
    scale1 = scale1.sort_values("order")
    x = np.arange(len(scale1))
    ax.bar(
        x,
        scale1["mean_feature_cosine"],
        color=[VARIANT_COLORS[v] for v in scale1["response_variant"]],
        width=0.62,
    )
    ax.set_title("C. 1x representation test")
    ax.set_ylabel("feature recovery (cosine)")
    ax.set_xticks(x, ["0x\nstabilized", "1x\nhidden eye", "1x\nknown eye"])
    ax.set_ylim(0.56, 0.98)
    _clean_axis(ax)

    png = OUT_DIR / "panel_C_motion_vs_stabilized_representation.png"
    pdf = OUT_DIR / "panel_C_motion_vs_stabilized_representation.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    trials = _load_trials()
    summary = _summary(trials)
    paired = _paired_matrix(trials)
    contrasts = _contrasts(paired)

    summary.to_csv(OUT_DIR / "panel_C_motion_vs_stabilized_representation_summary.csv", index=False)
    contrasts.to_csv(OUT_DIR / "panel_C_motion_vs_stabilized_representation_contrasts.csv", index=False)
    _write_readme(summary, contrasts)
    png, pdf = _plot(summary, contrasts)

    scale1 = summary[summary["observation_scale"].astype(float) == 1.0]
    values = {
        row.response_variant: float(row.mean_feature_cosine)
        for row in scale1.itertuples(index=False)
    }
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(
        "1x feature cosine: "
        f"0x stabilized={values['zero_static']:.4f}, "
        f"1x hidden-eye={values['full_exact']:.4f}, "
        f"1x known-eye={values['known_eye']:.4f}"
    )


if __name__ == "__main__":
    main()
