#!/usr/bin/env python3
"""Plot matched RR100 recorded-versus-twin preferred-SF distributions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    ROOT / "outputs/redundancy_resolved_v1_twin/rr100_recorded_twin_gratings_check_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--out-stem", default="rr100_recorded_twin_sf_preference_distribution")
    return parser.parse_args()


def label_sf(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "No valid\ngratings"
    if float(value) == 0.0:
        return "0\n(uniform)"
    return f"{float(value):g}"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics_path = (
        args.metrics.resolve()
        if args.metrics is not None
        else run_dir / "rr100_grating_tuning_metrics.csv"
    )
    metrics = pd.read_csv(metrics_path).sort_values("rr100_index").reset_index(drop=True)
    if len(metrics) != 100:
        raise ValueError(f"Expected 100 RR units, found {len(metrics)} in {metrics_path}")

    # Collapse harmless CSV floating-point representations such as
    # 3.999999999999999 and 4.0 back onto the actual stimulus conditions.
    real = metrics["real_peak_sf"].round(6)
    twin = metrics["twin_peak_sf"].round(6)
    valid_pair = real.notna() & twin.notna()
    valid_values = np.concatenate(
        [real[real.notna()].to_numpy(dtype=float), twin[twin.notna()].to_numpy(dtype=float)]
    )
    sf_bins = np.asarray(sorted(set(float(v) for v in valid_values)), dtype=float)

    pair_table = metrics[
        ["rr100_index", "canonical_channel", "session", "source_unit_index", "ccnorm", "peak_lag_ms"]
    ].copy()
    pair_table["recorded_preferred_sf_cpd"] = real
    pair_table["twin_preferred_sf_cpd_at_recorded_lag"] = twin
    pair_table["valid_grating_pair"] = valid_pair
    pair_table["exact_preferred_sf_match"] = valid_pair & (real == twin)
    pair_table.to_csv(run_dir / f"{args.out_stem}_paired_units.csv", index=False)

    categories: list[float | None] = [float(v) for v in sf_bins] + [None]
    count_rows: list[dict[str, object]] = []
    for source, values in (("recorded", real), ("fitted_twin", twin)):
        n_usable = int(values.notna().sum())
        for category in categories:
            if category is None:
                count = int(values.isna().sum())
            else:
                count = int((values == float(category)).sum())
            count_rows.append(
                {
                    "source": source,
                    "preferred_sf_cpd": category,
                    "preferred_sf_label": label_sf(category),
                    "count": count,
                    "percent_of_all_rr100": 100.0 * count / len(values),
                    "percent_of_usable": (
                        100.0 * count / n_usable if category is not None and n_usable else np.nan
                    ),
                    "n_all_rr100": int(len(values)),
                    "n_usable": n_usable,
                }
            )
    counts = pd.DataFrame(count_rows)
    counts.to_csv(run_dir / f"{args.out_stem}_counts.csv", index=False)

    transition = pd.crosstab(real[valid_pair], twin[valid_pair]).reindex(
        index=sf_bins, columns=sf_bins, fill_value=0
    )
    transition.index.name = "recorded_preferred_sf_cpd"
    transition.columns.name = "twin_preferred_sf_cpd_at_recorded_lag"
    transition.to_csv(run_dir / f"{args.out_stem}_transition_counts.csv")

    recorded_counts = [int((real == v).sum()) for v in sf_bins] + [int(real.isna().sum())]
    twin_counts = [int((twin == v).sum()) for v in sf_bins] + [int(twin.isna().sum())]
    x = np.arange(len(categories), dtype=float)
    width = 0.37

    fig, (ax_bar, ax_joint) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.4),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    bars_real = ax_bar.bar(
        x - width / 2,
        recorded_counts,
        width,
        color="#222222",
        label="recorded",
    )
    bars_twin = ax_bar.bar(
        x + width / 2,
        twin_counts,
        width,
        color="#d62728",
        label="fitted twin",
    )
    ax_bar.bar_label(bars_real, padding=2, fontsize=8)
    ax_bar.bar_label(bars_twin, padding=2, fontsize=8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([label_sf(v) for v in categories])
    ax_bar.set_xlabel("preferred spatial frequency (cpd)")
    ax_bar.set_ylabel("RR100 units")
    ax_bar.set_title("A  Marginal preferred-SF distributions", loc="left", fontweight="bold")
    ax_bar.legend(frameon=False)
    ax_bar.grid(axis="y", alpha=0.2)
    ax_bar.spines[["top", "right"]].set_visible(False)

    matrix = transition.to_numpy(dtype=int)
    im = ax_joint.imshow(matrix, cmap="Blues", origin="upper", aspect="equal")
    max_count = max(int(matrix.max()), 1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax_joint.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > 0.52 * max_count else "black",
                fontsize=9,
            )
    tick_labels = [label_sf(float(v)).replace("\n", " ") for v in sf_bins]
    ax_joint.set_xticks(np.arange(len(sf_bins)))
    ax_joint.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax_joint.set_yticks(np.arange(len(sf_bins)))
    ax_joint.set_yticklabels(tick_labels)
    ax_joint.set_xlabel("fitted-twin preferred SF (cpd)")
    ax_joint.set_ylabel("recorded preferred SF (cpd)")
    ax_joint.set_title("B  Matched unit transitions", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax_joint, fraction=0.046, pad=0.04, label="units")

    n_valid = int(valid_pair.sum())
    n_invalid = int((~valid_pair).sum())
    n_exact = int((valid_pair & (real == twin)).sum())
    fig.suptitle(
        "RR100 spatial-frequency preference: recorded responses vs fitted twin",
        fontsize=14,
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        (
            f"n=100 fixed RR medoids; {n_valid} have valid recorded/twin grating maps, "
            f"{n_invalid} have no valid grating samples. Exact preferred-SF bin agreement: "
            f"{n_exact}/{n_valid}. Twin evaluated at the recorded-data-selected response lag."
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.07, 1.0, 0.94])
    png_path = run_dir / f"{args.out_stem}.png"
    pdf_path = run_dir / f"{args.out_stem}.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    manifest = {
        "analysis": "rr100_recorded_twin_sf_preference_distribution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_metrics": str(metrics_path),
        "n_all_rr100": int(len(metrics)),
        "n_valid_pairs": n_valid,
        "n_no_valid_gratings": n_invalid,
        "n_exact_preferred_sf_matches": n_exact,
        "preferred_sf_definition": (
            "maximum of each unit's raw SF-by-orientation response map at the "
            "recorded-data-selected peak response lag"
        ),
        "twin_lag": "same recorded-data-selected lag",
        "sf_rounding_decimals": 6,
        "sf_bins_cpd": sf_bins.tolist(),
        "zero_cpd_note": "0 cpd is retained as the uniform/full-field stimulus condition",
        "outputs": {
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "marginal_counts": str(run_dir / f"{args.out_stem}_counts.csv"),
            "transition_counts": str(run_dir / f"{args.out_stem}_transition_counts.csv"),
            "paired_units": str(run_dir / f"{args.out_stem}_paired_units.csv"),
        },
    }
    (run_dir / f"{args.out_stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
