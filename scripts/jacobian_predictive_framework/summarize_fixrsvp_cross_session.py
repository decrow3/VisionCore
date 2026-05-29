#!/usr/bin/env python3
"""Cross-session summary figure for the full-radius fixRSVP manifold-alignment result.

Reads one output directory per session (must all be full-radius runs, i.e. no
--max-baseline-relative-radius-px filter applied) and produces:

  1. A session-level bar/strip chart:
       x-axis: session label
       y-axis: matched-minus-shuffled median A_J
       error bars: bootstrap 95 % CI
       annotation: "finite-displacement manifold regime"
       secondary axis row: median baseline-relative radius per session
       secondary axis row: Step 0 central-mass R², showing it fails

  2. A compact JSON summary written alongside the figure.

Usage example
-------------
python scripts/jacobian_predictive_framework/summarize_fixrsvp_cross_session.py \
    --session-dirs \
        outputs/jacobian_predictive_framework/allen_2022_02_16_patched_v2 \
        outputs/jacobian_predictive_framework/allen_2022_03_04_patched_v2 \
        outputs/jacobian_predictive_framework/allen_2022_02_24_patched_v2 \
        outputs/jacobian_predictive_framework/allen_2022_04_08_patched_v2 \
    --output-dir outputs/jacobian_predictive_framework/cross_session_patched_v2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from VisionCore.paths import FIGURES_DIR


def _load_session(session_dir: Path) -> dict | None:
    bs_path = session_dir / "backend_status.json"
    bu_path = session_dir / "step01_backend_units.csv"
    overview_path = session_dir / "step01_run_overview.md"
    if not bs_path.exists():
        return None

    bs = json.loads(bs_path.read_text())
    if bs.get("n_units", 0) == 0:
        return None

    # Parse session label from overview or directory name
    label = session_dir.name
    if overview_path.exists():
        for line in overview_path.read_text().splitlines():
            if line.startswith("- date:"):
                label = line.split(":", 1)[1].strip()
                break

    paired = bs.get("paired_delta_summary", {})
    align_all = paired.get("alignment_all", {})
    align_sg = paired.get("alignment_small_good", {})

    # Per-unit stats from CSV
    median_radius_px = float("nan")
    step0_central_r2 = float("nan")
    n_units = int(bs.get("n_units", 0))
    if bu_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(bu_path)
            if "centered_eye_radius_px_median" in df.columns:
                median_radius_px = float(np.nanmedian(df["centered_eye_radius_px_median"].values))
            for col in ("central_mass_r2_median", "step0_central_mass_median_r2_lin"):
                if col in df.columns:
                    step0_central_r2 = float(np.nanmedian(df[col].values))
                    break
        except Exception:
            pass

    ppd = bs.get("pixels_per_degree", float("nan"))
    median_radius_deg = median_radius_px / ppd if math.isfinite(ppd) and ppd > 0 else float("nan")

    return {
        "label": label,
        "session_dir": str(session_dir),
        "n_units": n_units,
        "matched_median_A_J": align_all.get("matched_median", float("nan")),
        "shuffled_median_A_J": align_all.get("shuffled_median", float("nan")),
        "delta_A_J": align_all.get("median_delta", float("nan")),
        "ci95_low": align_all.get("ci95_low", float("nan")),
        "ci95_high": align_all.get("ci95_high", float("nan")),
        "n_positive": align_all.get("positive_count", 0),
        "n_total": align_all.get("n", n_units),
        "small_good_delta_A_J": align_sg.get("median_delta", float("nan")),
        "small_good_n": align_sg.get("n", 0),
        "median_radius_px": median_radius_px,
        "median_radius_deg": median_radius_deg,
        "step0_central_mass_r2": step0_central_r2,
        "pixels_per_degree": ppd,
        "max_baseline_relative_radius_px": bs.get("max_baseline_relative_radius_px"),
    }


def _load_pairwise_bins(session_dir: Path, label: str) -> list[dict]:
    """Load step01_pairwise_bins.csv and return per-(session, bin) aggregate rows."""
    csv_path = session_dir / "step01_pairwise_bins.csv"
    if not csv_path.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    if df.empty:
        return []

    def _col(col: str) -> np.ndarray:
        return grp[col].values if col in grp.columns else np.array([np.nan])

    rows = []
    for (lo, hi), grp in df.groupby(["bin_lo_px", "bin_hi_px"], sort=True):
        n_units = len(grp)
        n_pairs_total = int(grp["n_pairs"].sum())
        rows.append({
            "label": label,
            "bin_lo_px": float(lo),
            "bin_hi_px": float(hi),
            "bin_mid_px": float((lo + hi) / 2),
            "n_units": n_units,
            "n_pairs_total": n_pairs_total,
            # Baseline Jacobian metrics
            "r2_lin_mean": float(np.nanmean(_col("r2_lin_median"))),
            "cosine_mean": float(np.nanmean(_col("cosine_median"))),
            "capture_V_J_delta_mean": float(np.nanmean(_col("capture_V_J_delta"))),
            "capture_V_J_matched_mean": float(np.nanmean(_col("capture_V_J_matched"))),
            "capture_V_J_shuffled_mean": float(np.nanmean(_col("capture_V_J_shuffled_median"))),
            "n_positive_delta": int(np.nansum(_col("capture_V_J_delta") > 0)),
            # Local (per-sample midpoint) Jacobian metrics
            "r2_lin_local_mean": float(np.nanmean(_col("r2_lin_local_median"))),
            "cosine_local_mean": float(np.nanmean(_col("cosine_local_median"))),
            "capture_V_J_local_delta_mean": float(np.nanmean(_col("capture_V_J_local_delta"))),
            "capture_V_J_local_matched_mean": float(np.nanmean(_col("capture_V_J_local_matched"))),
            "n_positive_local_delta": int(np.nansum(_col("capture_V_J_local_delta") > 0)),
        })
    return rows


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def write_cross_session_figure(sessions: list[dict], output_dir: Path) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    labels = [s["label"] for s in sessions]
    deltas = np.array([s["delta_A_J"] for s in sessions], dtype=np.float64)
    ci_low = np.array([s["ci95_low"] for s in sessions], dtype=np.float64)
    ci_high = np.array([s["ci95_high"] for s in sessions], dtype=np.float64)
    radii_px = np.array([s["median_radius_px"] for s in sessions], dtype=np.float64)
    radii_deg = np.array([s["median_radius_deg"] for s in sessions], dtype=np.float64)
    step0_r2 = np.array([s["step0_central_mass_r2"] for s in sessions], dtype=np.float64)
    n_units = np.array([s["n_units"] for s in sessions], dtype=np.int64)
    n_positive = np.array([s["n_positive"] for s in sessions], dtype=np.int64)
    n_total = np.array([s["n_total"] for s in sessions], dtype=np.int64)

    x = np.arange(len(sessions))

    fig, axes = plt.subplots(3, 1, figsize=(max(6, len(sessions) * 1.6), 10), sharex=True)

    # Panel 1: matched-minus-shuffled A_J delta with 95 % CI
    ax = axes[0]
    colors = ["#2a9d8f" if d > 0 else "#c44e52" for d in deltas]
    ax.bar(x, deltas, color=colors, alpha=0.75, width=0.6, zorder=2)
    for xi, d, lo, hi in zip(x, deltas, ci_low, ci_high):
        if math.isfinite(lo) and math.isfinite(hi):
            ax.plot([xi, xi], [lo, hi], color="#333333", linewidth=2, zorder=3)
    ax.axhline(0.0, color="#c44e52", linewidth=1.2, linestyle="--")
    for xi, nu, np_, nt in zip(x, n_units, n_positive, n_total):
        ax.text(xi, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.01,
                f"n={nu}\n({np_}/{nt}+)", ha="center", va="top", fontsize=7)
    ax.set_ylabel("Matched − img-shuffled $A_J$")
    ax.set_title(
        "fixRSVP cross-session: image-matched tangent-plane alignment\n"
        "(finite-displacement manifold regime — full baseline-relative radius)",
        fontsize=10,
    )
    ax.text(
        0.98, 0.97,
        "finite-displacement manifold regime\n(pointwise linearization fails)",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color="#555555",
        style="italic",
    )

    # Panel 2: median baseline-relative radius
    ax2 = axes[1]
    ax2.bar(x, radii_px, color="#5177a5", alpha=0.75, width=0.6)
    ax2_r = ax2.twinx()
    ax2_r.set_ylim(
        0,
        float(np.nanmax(radii_deg) * 1.2) if np.any(np.isfinite(radii_deg)) else 1,
    )
    ax2_r.set_ylabel("Median radius (deg)", fontsize=8)
    ax2.set_ylabel("Median baseline-relative\neye radius (model px)")
    ax2.set_title("Displacement scale (determines linearization regime)", fontsize=9)

    # Panel 3: Step 0 central-mass R²
    ax3 = axes[2]
    bar_colors = ["#2a9d8f" if r > 0.5 else "#e9c46a" if r > 0 else "#c44e52" for r in step0_r2]
    ax3.bar(x, step0_r2, color=bar_colors, alpha=0.75, width=0.6)
    ax3.axhline(0.5, color="#c44e52", linewidth=1.2, linestyle="--", label="R²=0.5 gate")
    ax3.axhline(0.0, color="#333333", linewidth=0.8, linestyle="-")
    ax3.set_ylabel("Median Step 0 $R^2_{lin}$\n(central-mass displacements)")
    ax3.set_title("Linearization validity at actual covariance displacement scale", fontsize=9)
    ax3.legend(fontsize=8, frameon=False)

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)

    fig.tight_layout(h_pad=0.5)
    rel_path = "figures/cross_session_manifold_alignment.png"
    out_path = output_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.clf()
    plt.close(fig)
    return rel_path


def write_pairwise_figure(
    pairwise_by_session: dict[str, list[dict]], output_dir: Path
) -> str:
    """6-panel pairwise distance figure comparing baseline vs local Jacobian metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    if not pairwise_by_session:
        return ""

    all_mids: set[float] = set()
    for rows in pairwise_by_session.values():
        for r in rows:
            all_mids.add(r["bin_mid_px"])
    bin_mids = sorted(all_mids)

    cmap = plt.cm.tab10
    session_labels = sorted(pairwise_by_session.keys())
    colors = {lab: cmap(i % 10) for i, lab in enumerate(session_labels)}

    has_local = any(
        np.isfinite(r.get("r2_lin_local_mean", float("nan")))
        for rows in pairwise_by_session.values()
        for r in rows
    )

    def _session_curve(rows: list[dict], key: str) -> np.ndarray:
        d = {r["bin_mid_px"]: r.get(key, float("nan")) for r in rows}
        return np.array([d.get(m, float("nan")) for m in bin_mids])

    def _grand_mean(key: str) -> np.ndarray:
        curves = [_session_curve(pairwise_by_session[lab], key) for lab in session_labels]
        return np.nanmean(np.stack(curves, axis=0), axis=0)

    nrows = 2 if has_local else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows), sharex=True, squeeze=False)
    x = np.array(bin_mids)

    for row_idx, (suffix, row_label) in enumerate([
        ("", "baseline J at fixed sample"),
        ("_local", "local midpoint J per pair"),
    ]):
        if row_idx == 1 and not has_local:
            break
        ax_r2, ax_cos, ax_vj = axes[row_idx]
        r2_key = f"r2_lin{suffix}_mean"
        cos_key = f"cosine{suffix}_mean"
        vj_key = f"capture_V_J{suffix}_delta_mean"

        # R² panel
        for lab in session_labels:
            ys = _session_curve(pairwise_by_session[lab], r2_key)
            ax_r2.plot(x, ys, "-o", color=colors[lab], alpha=0.6, linewidth=1.2,
                       markersize=4, label=lab)
        grand = _grand_mean(r2_key)
        ax_r2.plot(x, grand, "-o", color="black", linewidth=2.5, markersize=6,
                   label="grand mean", zorder=5)
        ax_r2.axhline(0.0, color="#c44e52", linewidth=1.0, linestyle="--")
        ax_r2.axhline(0.5, color="#2a9d8f", linewidth=1.0, linestyle=":", label="R²=0.5")
        ax_r2.set_ylabel(f"Mean R²_lin")
        ax_r2.set_title(f"Linearization — {row_label}", fontsize=9)
        if row_idx == 0:
            ax_r2.legend(fontsize=7, frameon=False, loc="upper right")

        # Cosine panel
        for lab in session_labels:
            ys = _session_curve(pairwise_by_session[lab], cos_key)
            ax_cos.plot(x, ys, "-o", color=colors[lab], alpha=0.6, linewidth=1.2, markersize=4)
        ax_cos.plot(x, _grand_mean(cos_key), "-o", color="black", linewidth=2.5,
                    markersize=6, zorder=5)
        ax_cos.axhline(0.0, color="#888888", linewidth=0.8, linestyle="-")
        ax_cos.set_ylabel("Mean cosine")
        ax_cos.set_title(f"Cosine — {row_label}", fontsize=9)

        # V_J delta panel
        grand_vj = _grand_mean(vj_key)
        bw = (x[1] - x[0]) * 0.6 if len(x) > 1 else 0.8
        ax_vj.bar(x, grand_vj,
                  color=["#2a9d8f" if d > 0 else "#c44e52" for d in np.nan_to_num(grand_vj)],
                  alpha=0.75, width=bw, label="grand mean")
        for lab in session_labels:
            ys = _session_curve(pairwise_by_session[lab], vj_key)
            ax_vj.plot(x, ys, "o--", color=colors[lab], alpha=0.5, linewidth=1,
                       markersize=3, label=lab)
        ax_vj.axhline(0.0, color="#c44e52", linewidth=1.0, linestyle="--")
        ax_vj.set_ylabel("Mean V_J delta (matched − shuffled)")
        ax_vj.set_title(f"V_J delta — {row_label}", fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel("Pairwise absolute eye-position distance (px)")

    if has_local:
        fig.suptitle(
            "Pairwise analysis: baseline J (row 0) vs local midpoint J (row 1)\n"
            "Distance-dependent linearization expected if local tangent story holds",
            fontsize=10,
        )
    else:
        fig.suptitle("Pairwise analysis: baseline Jacobian", fontsize=10)

    fig.tight_layout(h_pad=0.4)
    rel_path = "figures/pairwise_cross_session.png"
    out_path = output_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.clf()
    plt.close(fig)
    return rel_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-session summary of full-radius fixRSVP manifold alignment."
    )
    parser.add_argument(
        "--session-dirs",
        nargs="+",
        required=True,
        help="One output directory per session.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/jacobian_predictive_framework/cross_session_patched_v2",
        help="Where to write summary JSON and figure.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions = []
    pairwise_by_session: dict[str, list[dict]] = {}
    for d in args.session_dirs:
        result = _load_session(Path(d))
        if result is not None:
            sessions.append(result)
            pw = _load_pairwise_bins(Path(d), result["label"])
            if pw:
                pairwise_by_session[result["label"]] = pw
        else:
            print(f"  Skipping {d}: no units or missing backend_status.json")

    if not sessions:
        print("No sessions with units found. Exiting.")
        return

    print(f"\nCross-session summary ({len(sessions)} sessions with units):")
    print(f"{'Session':<22} {'N':>4} {'delta A_J':>10} {'CI95':>20} {'radius_px':>10} {'Step0_R2':>10}")
    for s in sessions:
        ci = f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]"
        print(
            f"{s['label']:<22} {s['n_units']:>4} {s['delta_A_J']:>10.4f} "
            f"{ci:>20} {s['median_radius_px']:>10.2f} {s['step0_central_mass_r2']:>10.4f}"
        )

    all_deltas = np.array([s["delta_A_J"] for s in sessions])
    n_pos = int(np.sum(all_deltas > 0))
    print(f"\nSessions with positive A_J delta: {n_pos}/{len(sessions)}")
    print(f"Grand median delta: {float(np.nanmedian(all_deltas)):.4f}")

    summary = {
        "n_sessions": len(sessions),
        "n_sessions_positive_delta": n_pos,
        "grand_median_delta_A_J": float(np.nanmedian(all_deltas)),
        "grand_median_radius_px": float(np.nanmedian([s["median_radius_px"] for s in sessions])),
        "sessions": sessions,
        "interpretation": (
            "Full-radius fixRSVP cross-session result. "
            "Pointwise linearization (J*dp) fails at the actual covariance displacement scale. "
            "Above-shuffle alignment supports a tangent-manifold interpretation only."
        ),
    }
    summary_path = output_dir / "cross_session_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary written to {summary_path}")

    fig_rel = write_cross_session_figure(sessions, output_dir)
    if fig_rel:
        print(f"Figure written to {output_dir / fig_rel}")

    if pairwise_by_session:
        print(f"\nPairwise data found for {len(pairwise_by_session)} sessions: "
              f"{list(pairwise_by_session.keys())}")
        pw_rel = write_pairwise_figure(pairwise_by_session, output_dir)
        if pw_rel:
            print(f"Pairwise figure written to {output_dir / pw_rel}")
        try:
            fig_out = FIGURES_DIR / "jacobian_predictive_framework" / "pairwise_cross_session.png"
            fig_out.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(output_dir / pw_rel, fig_out)
            print(f"Pairwise figure copied to {fig_out}")
        except Exception:
            pass
        # Add pairwise bin table to summary JSON
        summary["pairwise_by_session"] = {
            lab: rows for lab, rows in pairwise_by_session.items()
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        print("\nNo pairwise bin data found (step01_pairwise_bins.csv missing in all sessions).")

    # Also write to canonical FIGURES_DIR
    try:
        fig_out = FIGURES_DIR / "jacobian_predictive_framework" / "cross_session_manifold_alignment.png"
        fig_out.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(output_dir / fig_rel, fig_out)
        print(f"Figure copied to {fig_out}")
    except Exception:
        pass

    # Write a compact markdown summary
    md_lines = [
        "# fixRSVP Cross-Session Manifold Alignment",
        "",
        "## Interpretation frame",
        "",
        "This is the **finite-displacement manifold regime**. Pointwise linearization",
        "($J\\Delta p \\approx r(I_{\\Delta p}) - r(I)$) fails at the actual covariance",
        "displacement scale in all sessions. The result below therefore does **not** support",
        "a local-Jacobian prediction claim. It supports a weaker but still informative",
        "**tangent-manifold alignment** claim: the Jacobian column space indexes",
        "image-specific translation-manifold directions that organise FEM population covariance.",
        "",
        "## Session table",
        "",
        "| Session | N units | Δ A_J | 95% CI | Median radius (px) | Step 0 R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in sessions:
        ci = f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]"
        md_lines.append(
            f"| {s['label']} | {s['n_units']} | {s['delta_A_J']:.4f} | {ci} "
            f"| {s['median_radius_px']:.1f} | {s['step0_central_mass_r2']:.4f} |"
        )

    md_lines += [
        "",
        "## Summary",
        "",
        f"- Sessions with positive Δ A_J: {n_pos}/{len(sessions)}",
        f"- Grand median Δ A_J: {float(np.nanmedian(all_deltas)):.4f}",
        f"- Grand median baseline-relative radius: {float(np.nanmedian([s['median_radius_px'] for s in sessions])):.1f} px",
        "",
        "## Future design note",
        "",
        "The current `image_phase_radius` unit definition bins by *within-trial* centred radius,",
        "creating units that span large absolute eye-position clouds (~5–12 px) across trials.",
        "A radius filter of ≤ 3 px retains zero units. Future analyses should group by",
        "**absolute eye-position clusters** or use targeted stimulus repeats near fixed retinal",
        "positions to estimate local natural-image Jacobian covariance directly.",
    ]
    md_path = output_dir / "cross_session_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Markdown summary written to {md_path}")


if __name__ == "__main__":
    main()
