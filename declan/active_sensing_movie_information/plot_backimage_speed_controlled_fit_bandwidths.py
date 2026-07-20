#!/usr/bin/env python3
"""Summarize log-Gaussian SF/TF/speed tuning bandwidths by microsaccade group."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIT_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_speed_controlled_sf_tf_speed_tuning_fits_v1"
)
DEFAULT_OUT_DIR = DEFAULT_FIT_DIR / "fit_bandwidth_summary"
FAMILY_ORDER = ["cycle_valid", "subcycle_control"]
DIM_ORDER = ["sf", "tf", "speed"]
DIM_LABELS = {
    "sf": "SF",
    "tf": "TF",
    "speed": "speed",
}
FAMILY_LABELS = {
    "cycle_valid": "cycle-valid SFs",
    "subcycle_control": "sub-cycle controls",
}
GROUP_ORDER = ["high_speed_preferring", "low_speed_preferring"]
GROUP_LABELS = {
    "high_speed_preferring": "high-speed\npref.",
    "low_speed_preferring": "low-speed\npref.",
}
GROUP_COLORS = {
    "high_speed_preferring": "#1f77b4",
    "low_speed_preferring": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / math.sqrt(vals.size))


def welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    av = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    bv = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if av.size < 2 or bv.size < 2:
        return float("nan"), float("nan")
    try:
        from scipy import stats

        out = stats.ttest_ind(av, bv, equal_var=False, nan_policy="omit")
        return float(out.statistic), float(out.pvalue)
    except Exception:
        return float("nan"), float("nan")


def add_bandwidth_columns(fits: pd.DataFrame) -> pd.DataFrame:
    fits = fits.copy()
    fits["fit_sigma_octaves"] = pd.to_numeric(fits["fit_sigma_log2"], errors="coerce")
    fits["fit_fwhm_octaves"] = float(2.0 * math.sqrt(2.0 * math.log(2.0))) * fits["fit_sigma_octaves"]
    fits["is_edge_fit"] = fits["fit_status"].astype(str).isin(["lower_edge", "upper_edge"])
    return fits


def summarize(fits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    tests = []
    for subset_name, sub in [
        ("all_fit_ok", fits[fits["fit_ok"].astype(bool)]),
        ("interior_only", fits[fits["fit_status"].astype(str).eq("interior")]),
    ]:
        for keys, ss in sub.groupby(["speed_family", "dimension", "speed_pref_group", "speed_pref_label"], sort=True):
            family, dim, group, label = keys
            vals = pd.to_numeric(ss["fit_fwhm_octaves"], errors="coerce").dropna()
            rows.append(
                {
                    "subset": subset_name,
                    "speed_family": family,
                    "dimension": dim,
                    "speed_pref_group": group,
                    "speed_pref_label": label,
                    "n_units": int(vals.shape[0]),
                    "fwhm_octaves_mean": float(vals.mean()) if not vals.empty else float("nan"),
                    "fwhm_octaves_sem": sem(vals),
                    "fwhm_octaves_median": float(vals.median()) if not vals.empty else float("nan"),
                    "sigma_octaves_mean": float(pd.to_numeric(ss["fit_sigma_octaves"], errors="coerce").mean()),
                    "edge_fraction": float(ss["is_edge_fit"].mean()) if not ss.empty else float("nan"),
                    "median_fit_r2": float(pd.to_numeric(ss["fit_r2"], errors="coerce").median()),
                }
            )
        for keys, ss in sub.groupby(["speed_family", "dimension"], sort=True):
            family, dim = keys
            high = ss[ss["speed_pref_group"].eq("high_speed_preferring")]["fit_fwhm_octaves"]
            low = ss[ss["speed_pref_group"].eq("low_speed_preferring")]["fit_fwhm_octaves"]
            t_stat, p_value = welch(high, low)
            high = pd.to_numeric(high, errors="coerce").dropna()
            low = pd.to_numeric(low, errors="coerce").dropna()
            tests.append(
                {
                    "subset": subset_name,
                    "speed_family": family,
                    "dimension": dim,
                    "high_n": int(high.shape[0]),
                    "low_n": int(low.shape[0]),
                    "high_fwhm_octaves_mean": float(high.mean()) if not high.empty else float("nan"),
                    "low_fwhm_octaves_mean": float(low.mean()) if not low.empty else float("nan"),
                    "high_minus_low_fwhm_octaves": float(high.mean() - low.mean())
                    if not high.empty and not low.empty
                    else float("nan"),
                    "welch_t": t_stat,
                    "welch_p": p_value,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(tests)


def plot_bandwidths(out_dir: Path, fits: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame, *, dpi: int) -> Path:
    png = out_dir / "sf_tf_speed_fit_bandwidths_by_group.png"
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.08, top=0.86, hspace=0.52, wspace=0.28)
    fig.suptitle("Fit-derived SF, TF, and speed tuning bandwidths", y=0.965, fontsize=16)
    fig.text(
        0.5,
        0.925,
        "Bandwidth is log-Gaussian FWHM in octaves; filled points are interior fits, hollow points are edge-preference fits",
        ha="center",
        fontsize=10.5,
        color="0.35",
    )
    rng = np.random.default_rng(17)
    for row_idx, family in enumerate(FAMILY_ORDER):
        for col_idx, dim in enumerate(DIM_ORDER):
            ax = axes[row_idx, col_idx]
            sub = fits[(fits["speed_family"].eq(family)) & (fits["dimension"].eq(dim)) & (fits["fit_ok"].astype(bool))]
            if sub.empty:
                ax.axis("off")
                continue
            for x, group in enumerate(GROUP_ORDER):
                ss = sub[sub["speed_pref_group"].eq(group)].copy()
                vals = pd.to_numeric(ss["fit_fwhm_octaves"], errors="coerce")
                finite = vals.notna()
                ss = ss[finite]
                vals = vals[finite].to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                jitter = rng.uniform(-0.08, 0.08, size=vals.size)
                edge = ss["is_edge_fit"].to_numpy(dtype=bool)
                color = GROUP_COLORS[group]
                ax.scatter(
                    np.full(vals.size, x) + jitter,
                    vals,
                    s=19,
                    facecolors=np.where(edge, "none", color),
                    edgecolors=color,
                    linewidths=np.where(edge, 0.8, 0.0),
                    alpha=0.55,
                )
                summ = summary[
                    (summary["subset"].eq("all_fit_ok"))
                    & (summary["speed_family"].eq(family))
                    & (summary["dimension"].eq(dim))
                    & (summary["speed_pref_group"].eq(group))
                ]
                if not summ.empty:
                    ax.errorbar(
                        [x],
                        [float(summ["fwhm_octaves_mean"].iloc[0])],
                        yerr=[float(summ["fwhm_octaves_sem"].iloc[0])],
                        color="black",
                        marker="o",
                        markersize=5,
                        capsize=4,
                        lw=1.5,
                    )
            test_all = tests[
                (tests["subset"].eq("all_fit_ok"))
                & (tests["speed_family"].eq(family))
                & (tests["dimension"].eq(dim))
            ]
            test_int = tests[
                (tests["subset"].eq("interior_only"))
                & (tests["speed_family"].eq(family))
                & (tests["dimension"].eq(dim))
            ]
            p_text = ""
            if not test_all.empty and np.isfinite(float(test_all["welch_p"].iloc[0])):
                p_text = f"\nall p={float(test_all['welch_p'].iloc[0]):.3g}"
            if not test_int.empty and np.isfinite(float(test_int["welch_p"].iloc[0])):
                p_text += f"; interior p={float(test_int['welch_p'].iloc[0]):.3g}"
            ax.set_title(f"{FAMILY_LABELS.get(family, family)}: {DIM_LABELS.get(dim, dim)}{p_text}", fontsize=10)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
            ax.set_ylabel("FWHM (octaves)")
            ax.grid(True, axis="y", color="0.9")
    fig.savefig(png, dpi=dpi)
    fig.savefig(out_dir / "sf_tf_speed_fit_bandwidths_by_group.pdf")
    plt.close(fig)
    return png


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits = add_bandwidth_columns(pd.read_csv(Path(args.fit_dir) / "sf_tf_speed_tuning_fit_unit_summary.csv"))
    summary, tests = summarize(fits)
    fits.to_csv(out_dir / "sf_tf_speed_fit_bandwidth_unit_summary.csv", index=False)
    summary.to_csv(out_dir / "sf_tf_speed_fit_bandwidth_group_summary.csv", index=False)
    tests.to_csv(out_dir / "sf_tf_speed_fit_bandwidth_group_tests.csv", index=False)
    png = plot_bandwidths(out_dir, fits, summary, tests, dpi=int(args.dpi))
    print(f"Wrote {png}")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
