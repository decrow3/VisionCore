"""Summarize decoder-controls `.npz` outputs into compact tables.

This reads the artifacts written by:
- scripts/temporal_decoding/eoptotype_decoder_controls.py

and prints a small markdown-style report for each LogMAR.

Example
-------
/home/declan/VisionCore/.venv/bin/python scripts/temporal_decoding/summarize_decoder_controls_npz.py \
  --logmars -0.35,-0.40,-0.45,-0.50

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def _fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def _extract_at_windows(windows: np.ndarray, mean_arr: np.ndarray, std_arr: np.ndarray, key_windows: list[int]):
    out = {}
    for w in key_windows:
        idx = np.where(windows == int(w))[0]
        if len(idx) != 1:
            out[int(w)] = None
            continue
        i = int(idx[0])
        out[int(w)] = (float(mean_arr[i]), float(std_arr[i]))
    return out


def summarize_one(npz_path: Path, key_windows: list[int]) -> str:
    d = np.load(npz_path, allow_pickle=True)
    windows = d["windows"].astype(int)

    # Match the style used in declan/results_summary.md:
    # show D1 real, D1 stabilized, D2a real, and D3 real (if present).
    preferred_order = ["D1_real", "D1_stabilized", "D2a_real", "D3_real"]
    present = [
        dec
        for dec in preferred_order
        if f"{dec}_mean" in d.files and f"{dec}_std" in d.files
    ]

    lines: list[str] = []

    # Header
    lm = float(d["logmar"][0]) if "logmar" in d.files else float("nan")
    lines.append(f"LogMAR {lm:+.2f}")
    lines.append("")

    # Artifacts
    # (Figures are deterministic names written by the controls script)
    lm_str = f"{lm:.2f}"
    results_rel = f"scripts/temporal_decoding/data/results/{npz_path.name}"
    fig_d1_rel = f"scripts/temporal_decoding/figures/fig_decoder_controls_D1_lm{lm_str}.png"
    fig_d2_rel = f"scripts/temporal_decoding/figures/fig_decoder_controls_D2_lm{lm_str}.png"
    lines.append("Artifacts:")
    lines.append(f"- {results_rel}")
    lines.append(f"- {fig_d1_rel}")
    # D2 figure exists even if D2a was skipped (it overlays D1); include anyway.
    lines.append(f"- {fig_d2_rel}")
    lines.append("")

    # Table
    lines.append("| Decoder | " + " | ".join([f"W={w}" for w in key_windows]) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(key_windows)) + "|")

    # Keep a dict for gap computations
    means_by_dec = {}
    for dec in present:
        mean_arr = d[f"{dec}_mean"].astype(float)
        std_arr = d[f"{dec}_std"].astype(float)
        at = _extract_at_windows(windows, mean_arr, std_arr, key_windows)
        row = [dec.replace("_", " ")]
        for w in key_windows:
            if at[int(w)] is None:
                row.append("(n/a)")
            else:
                m, s = at[int(w)]
                row.append(_fmt_pm(m, s))
        lines.append("| " + " | ".join(row) + " |")
        means_by_dec[dec] = mean_arr

    lines.append("")

    # Auto-selection metrics (mirror eoptotype_decoder_controls.py heuristics)
    if "D1_real_mean" in d.files and "D2a_real_mean" in d.files:
        d1_real = d["D1_real_mean"].astype(float)
        d2a_real = d["D2a_real_mean"].astype(float)
        max_gain = float(np.max(d2a_real - d1_real))
        lines.append(f"Auto-selection: max(D2a_real − D1_real) = {max_gain:+.3f}")

    if "D1_stabilized_mean" in d.files and "D1_real_mean" in d.files:
        d1_stab = d["D1_stabilized_mean"].astype(float)
        best_real = d["D1_real_mean"].astype(float)
        if "D2a_real_mean" in d.files:
            best_real = np.maximum(best_real, d["D2a_real_mean"].astype(float))
        gap = float(np.max(d1_stab) - np.max(best_real))
        lines.append(f"Auto-selection: max(stabilized D1) − max(real best) = {gap:+.3f}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logmars", type=str, required=True, help="Comma-separated LogMAR values")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="scripts/temporal_decoding/data/results",
        help="Directory containing decoder_controls_lm*.npz",
    )
    parser.add_argument("--key_windows", type=str, default="1,24,60")
    args = parser.parse_args(argv)

    logmars = _parse_csv_floats(args.logmars)
    key_windows = [int(x) for x in str(args.key_windows).split(",") if str(x).strip()]

    results_dir = Path(args.results_dir)

    for lm in logmars:
        npz = results_dir / f"decoder_controls_lm{lm:.2f}.npz"
        if not npz.exists():
            print(f"[missing] {npz}")
            print()
            continue
        print(summarize_one(npz, key_windows))
        print("\n---\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
