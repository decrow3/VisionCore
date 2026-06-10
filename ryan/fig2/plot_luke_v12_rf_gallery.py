"""Plot dots-calibration RFs for Luke v12 candidate units.

This script is for auditing the units selected by ``test_rowley12.py`` gate-only
runs. It reuses the step07 RF-geometry convention: half-peak contour geometry,
PRL bias correction, eccentricity <= 1 deg, and RF diameter <= 2 deg.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ROWLEY_REPO = ROOT.parent / "DataRowleyV1V2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if ROWLEY_REPO.exists() and str(ROWLEY_REPO) not in sys.path:
    sys.path.insert(0, str(ROWLEY_REPO))

from DataRowleyV1V2.shifter.preprocess import get_contour_mask_and_properties  # noqa: E402
from VisionCore.paths import FIGURES_DIR, STATS_DIR  # noqa: E402


DEFAULT_V12_SUMMARY = (
    STATS_DIR / "test_rowley12_luke_gateonly_r005_summary_20260609.tsv"
)
DEFAULT_STEP07_GEOM = (
    ROWLEY_REPO / "outputs" / "luke_step07_rf_survey" / "luke_step07_visual_rf_geometry.csv"
)
DEFAULT_STEP07_RELIABILITY = (
    STATS_DIR / "fig2_rf_compare" / "luke_step07_visual_rf_valid_split_half_reliability.csv"
)
DEFAULT_YAML_DIR = ROOT / "experiments" / "dataset_configs" / "sessions"
DEFAULT_OUT_DIR = FIGURES_DIR / "fig2_rf_compare" / "luke_v12_r005_rf_gallery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12-summary", type=Path, default=DEFAULT_V12_SUMMARY)
    parser.add_argument("--step07-geometry", type=Path, default=DEFAULT_STEP07_GEOM)
    parser.add_argument("--step07-reliability", type=Path, default=DEFAULT_STEP07_RELIABILITY)
    parser.add_argument("--yaml-dir", type=Path, default=DEFAULT_YAML_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-eccentricity-deg", type=float, default=1.0)
    parser.add_argument("--max-rf-diameter-deg", type=float, default=2.0)
    parser.add_argument("--contour-threshold", type=float, default=0.5)
    return parser.parse_args()


def boolish(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def load_v12_units(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path, sep="\t")
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        cid_value = row.get("pool_b_cids", "")
        if pd.isna(cid_value):
            cid_value = ""
        cids = [int(x) for x in str(cid_value).split(",") if x.strip()]
        for cid in cids:
            rows.append(
                {
                    "session": str(row["session"]),
                    "eye": str(row["eye"]),
                    "cid": cid,
                    "v12_n_all": int(row["n_all"]),
                }
            )
    return pd.DataFrame(rows)


def load_dataset_metadata(dset_path: Path) -> dict:
    data = torch.load(dset_path, weights_only=False, map_location="cpu", mmap=True)
    return data.get("metadata", {})


def resolve_session_root(dataset_dir: Path) -> Path:
    for candidate in [dataset_dir, *dataset_dir.parents]:
        if (candidate / "dots_calibration").exists() or (candidate / "dpi_calibration").exists():
            return candidate
    return dataset_dir.parents[1]


def load_yaml_config(yaml_dir: Path, session: str, eye: str) -> dict:
    path = yaml_dir / f"{session}_{eye}_V1.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Session YAML not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_yaml_path"] = str(path)
    return cfg


def reconstruct_centers_deg(
    roi_deg: np.ndarray, dxy_deg: float, ppd: float, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    n_i, n_j = shape
    roi_pix = np.flipud(roi_deg * ppd)
    dxy_pix = dxy_deg * ppd
    i_edges = np.arange(roi_pix[0, 0], roi_pix[0, 1] + dxy_pix, dxy_pix)
    j_edges = np.arange(roi_pix[1, 0], roi_pix[1, 1] + dxy_pix, dxy_pix)
    i_centers_deg = ((i_edges[:-1] + i_edges[1:]) / 2) / ppd
    j_centers_deg = ((j_edges[:-1] + j_edges[1:]) / 2) / ppd
    if len(i_centers_deg) != n_i or len(j_centers_deg) != n_j:
        i_centers_deg = np.linspace(roi_deg[0, 0] + dxy_deg / 2, roi_deg[0, 1] - dxy_deg / 2, n_i)
        j_centers_deg = np.linspace(roi_deg[1, 0] + dxy_deg / 2, roi_deg[1, 1] - dxy_deg / 2, n_j)
    return i_centers_deg, j_centers_deg


def load_prl_bias(session_root: Path, eye: str, ppd: float) -> dict[str, float]:
    cal_params_path = session_root / "dpi_calibration" / f"{eye}_eye" / "calibration_params.npz"
    if not cal_params_path.exists():
        return {"prl_bias_el_deg": np.nan, "prl_bias_az_deg": np.nan}
    params = np.load(cal_params_path, allow_pickle=True)
    bias_pix = np.asarray(params.get("bias_pix", [np.nan, np.nan]), dtype=float)
    return {
        "prl_bias_el_deg": float(bias_pix[0] / ppd),
        "prl_bias_az_deg": float(bias_pix[1] / ppd),
    }


def measure_rf(
    rf_by_lag: np.ndarray,
    i_centers_deg: np.ndarray,
    j_centers_deg: np.ndarray,
    dxy_deg: float,
    prl_bias_el_deg: float,
    prl_bias_az_deg: float,
    contour_threshold: float,
) -> dict[str, object]:
    peak_lag = int(np.argmax(np.abs(rf_by_lag).max(axis=(1, 2))))
    rf_im = rf_by_lag[peak_lag]
    peak_pos = np.unravel_index(np.argmax(np.abs(rf_im)), rf_im.shape)
    peak_value = float(rf_im[peak_pos])
    polarity = 1.0 if peak_value >= 0 else -1.0
    rf_aligned = polarity * rf_im
    threshold = contour_threshold * abs(peak_value)
    contour, _, area_px, center_rc = get_contour_mask_and_properties(rf_aligned, threshold)
    if contour is None or not np.isfinite(center_rc).all() or area_px <= 0:
        return {
            "rf_status": "no_contour",
            "peak_lag": peak_lag,
            "peak_value": peak_value,
            "polarity": "ON" if polarity > 0 else "OFF",
            "rf_image": rf_aligned,
            "contour_az_deg": None,
            "contour_el_deg": None,
        }

    ctr_row, ctr_col = center_rc
    ctr_el_dots = float(np.interp(ctr_row, np.arange(len(i_centers_deg)), i_centers_deg))
    ctr_az_dots = float(np.interp(ctr_col, np.arange(len(j_centers_deg)), j_centers_deg))
    ctr_el = ctr_el_dots + prl_bias_el_deg
    ctr_az = ctr_az_dots + prl_bias_az_deg
    contour_el = np.interp(contour[:, 0], np.arange(len(i_centers_deg)), i_centers_deg) + prl_bias_el_deg
    contour_az = np.interp(contour[:, 1], np.arange(len(j_centers_deg)), j_centers_deg) + prl_bias_az_deg
    area_deg2 = float(area_px) * dxy_deg * dxy_deg
    radius = float(np.sqrt(area_deg2 / np.pi))
    return {
        "rf_status": "ok",
        "peak_lag": peak_lag,
        "peak_value": peak_value,
        "polarity": "ON" if polarity > 0 else "OFF",
        "rf_center_az_deg_dots": ctr_az_dots,
        "rf_center_el_deg_dots": ctr_el_dots,
        "rf_center_az_deg": ctr_az,
        "rf_center_el_deg": ctr_el,
        "eccentricity_deg": float(np.hypot(ctr_az, ctr_el)),
        "rf_area_deg2": area_deg2,
        "rf_equiv_radius_deg": radius,
        "rf_equiv_diameter_deg": 2.0 * radius,
        "rf_image": rf_aligned,
        "contour_az_deg": contour_az,
        "contour_el_deg": contour_el,
    }


def category_for(row: pd.Series, geom_by_key: dict, rel_by_key: dict) -> str:
    key = (row["session"], row["eye"], int(row["cid"]))
    geom_row = geom_by_key.get(key)
    rel_row = rel_by_key.get(key)
    if geom_row is None:
        return "not_step07_exported"
    if str(geom_row.get("status")) != "ok":
        return "step07_no_contour"
    if not bool(geom_row.get("rf_valid")):
        ecc_ok = bool(geom_row.get("rf_valid_eccentricity"))
        size_ok = bool(geom_row.get("rf_valid_size"))
        if not ecc_ok and not size_ok:
            return "step07_rf_fails_ecc_and_size"
        if not ecc_ok:
            return "step07_rf_fails_ecc"
        if not size_ok:
            return "step07_rf_fails_size"
        return "step07_rf_invalid"
    if rel_row is None:
        return "step07_rf_valid_no_reliability_row"
    if str(rel_row.get("status")) != "ok":
        return "step07_rf_valid_missing_after_alignment"
    if not boolish(rel_row.get("passes_psth_r2_gate")):
        return "step07_rf_valid_fails_r2"
    return "passes_current_fig2_candidate"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    v12_units = load_v12_units(args.v12_summary)
    geom = pd.read_csv(args.step07_geometry)
    rel = pd.read_csv(args.step07_reliability)
    geom_by_key = {
        (str(r["session"]), str(r["eye"]), int(r["cluster_id"])): r
        for r in geom.to_dict("records")
    }
    rel_by_key = {
        (str(r["session"]), str(r["eye"]), int(r["cid"])): r
        for r in rel.to_dict("records")
    }

    records: list[dict[str, object]] = []
    image_records: list[dict[str, object]] = []
    cal_cache: dict[tuple[str, str], dict[str, object]] = {}
    for row in v12_units.itertuples(index=False):
        session = str(row.session)
        eye = str(row.eye)
        cid = int(row.cid)
        cache_key = (session, eye)
        if cache_key not in cal_cache:
            cfg = load_yaml_config(args.yaml_dir, session, eye)
            dataset_dir = Path(cfg["directory"])
            session_root = resolve_session_root(dataset_dir)
            cal_path = session_root / "dots_calibration" / f"{eye}_eye" / "calibration_results.npz"
            dset_metadata = load_dataset_metadata(dataset_dir / "gaborium.dset")
            dataset_ppd = float(dset_metadata.get("ppd", np.nan))
            prl = load_prl_bias(session_root, eye, dataset_ppd)
            cal = np.load(cal_path, allow_pickle=True)
            stas = cal["optimized_stas"]
            cluster_ids = cal["calibration_cluster_ids"].astype(int)
            i_centers_deg, j_centers_deg = reconstruct_centers_deg(
                cal["roi_deg"],
                float(cal["dxy_deg"]),
                float(cal["ppd"]),
                tuple(stas.shape[2:4]),
            )
            cal_cache[cache_key] = {
                "cfg": cfg,
                "cal_path": cal_path,
                "cal": cal,
                "stas": stas,
                "cluster_to_index": {int(c): i for i, c in enumerate(cluster_ids)},
                "i_centers_deg": i_centers_deg,
                "j_centers_deg": j_centers_deg,
                "dxy_deg": float(cal["dxy_deg"]),
                **prl,
            }
        cc = cal_cache[cache_key]
        idx = cc["cluster_to_index"].get(cid)
        base = {
            "session": session,
            "eye": eye,
            "cid": cid,
            "category": category_for(pd.Series({"session": session, "eye": eye, "cid": cid}), geom_by_key, rel_by_key),
            "calibration_path": str(cc["cal_path"]),
            "prl_bias_az_deg": cc["prl_bias_az_deg"],
            "prl_bias_el_deg": cc["prl_bias_el_deg"],
        }
        if idx is None:
            records.append(base | {"rf_status": "missing_from_calibration"})
            continue
        measured = measure_rf(
            cc["stas"][idx],
            cc["i_centers_deg"],
            cc["j_centers_deg"],
            cc["dxy_deg"],
            cc["prl_bias_el_deg"],
            cc["prl_bias_az_deg"],
            args.contour_threshold,
        )
        serializable = {k: v for k, v in measured.items() if k not in {"rf_image", "contour_az_deg", "contour_el_deg"}}
        if measured["rf_status"] == "ok":
            serializable["rf_valid_eccentricity"] = bool(
                serializable["eccentricity_deg"] <= args.max_eccentricity_deg
            )
            serializable["rf_valid_size"] = bool(
                serializable["rf_equiv_diameter_deg"] <= args.max_rf_diameter_deg
            )
            serializable["rf_valid"] = bool(
                serializable["rf_valid_eccentricity"] and serializable["rf_valid_size"]
            )
        records.append(base | serializable)
        image_records.append(base | measured)

    measured_df = pd.DataFrame(records)
    csv_path = args.out_dir / "luke_v12_r005_rf_measurements.csv"
    measured_df.to_csv(csv_path, index=False)

    make_summary_plots(measured_df, args)
    make_gallery(image_records, args)

    print(f"Saved {csv_path}")
    print(f"Saved {args.out_dir / 'luke_v12_r005_rf_summary.pdf'}")
    print(f"Saved {args.out_dir / 'luke_v12_r005_rf_gallery.pdf'}")


def make_summary_plots(df: pd.DataFrame, args: argparse.Namespace) -> None:
    ok = df[df["rf_status"] == "ok"].copy()
    colors = {
        "passes_current_fig2_candidate": "tab:green",
        "step07_rf_fails_ecc": "tab:red",
        "step07_rf_fails_size": "tab:orange",
        "step07_rf_fails_ecc_and_size": "tab:purple",
        "step07_rf_valid_fails_r2": "tab:blue",
        "not_step07_exported": "0.45",
    }
    with PdfPages(args.out_dir / "luke_v12_r005_rf_summary.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        theta = np.linspace(0, 2 * np.pi, 256)
        ax.plot(
            args.max_eccentricity_deg * np.cos(theta),
            args.max_eccentricity_deg * np.sin(theta),
            color="black",
            lw=1,
            ls="--",
            label=f"{args.max_eccentricity_deg:g} deg eccentricity",
        )
        for cat, sub in ok.groupby("category"):
            ax.scatter(
                sub["rf_center_az_deg"],
                sub["rf_center_el_deg"],
                s=34,
                alpha=0.85,
                label=f"{cat} (n={len(sub)})",
                color=colors.get(cat, "0.7"),
                edgecolor="white",
                linewidth=0.4,
            )
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.set_aspect("equal", adjustable="box")
        lim = max(2.5, np.nanpercentile(np.abs(ok[["rf_center_az_deg", "rf_center_el_deg"]].to_numpy()), 98) * 1.1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("RF azimuth relative to PRL (deg)")
        ax.set_ylabel("RF elevation relative to PRL (deg)")
        ax.set_title("Luke v12+R2>=0.05 RF centers")
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        for cat, sub in ok.groupby("category"):
            ax.scatter(
                sub["eccentricity_deg"],
                sub["rf_equiv_diameter_deg"],
                s=34,
                alpha=0.85,
                label=f"{cat} (n={len(sub)})",
                color=colors.get(cat, "0.7"),
                edgecolor="white",
                linewidth=0.4,
            )
        ax.axvline(args.max_eccentricity_deg, color="black", lw=1, ls="--")
        ax.axhline(args.max_rf_diameter_deg, color="black", lw=1, ls="--")
        ax.set_xlabel("RF eccentricity relative to PRL (deg)")
        ax.set_ylabel("Half-peak equivalent RF diameter (deg)")
        ax.set_title("RF size vs eccentricity")
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ok["eccentricity_deg"].hist(ax=axes[0], bins=25, color="0.35")
        axes[0].axvline(args.max_eccentricity_deg, color="tab:red", lw=1.2)
        axes[0].set_xlabel("RF eccentricity (deg)")
        axes[0].set_ylabel("Unit count")
        ok["rf_equiv_diameter_deg"].hist(ax=axes[1], bins=25, color="0.35")
        axes[1].axvline(args.max_rf_diameter_deg, color="tab:red", lw=1.2)
        axes[1].set_xlabel("RF diameter (deg)")
        fig.suptitle("Luke v12+R2>=0.05 RF geometry distributions")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def make_gallery(image_records: list[dict[str, object]], args: argparse.Namespace) -> None:
    records = sorted(image_records, key=lambda r: (str(r["category"]), str(r["session"]), str(r["eye"]), int(r["cid"])))
    per_page = 12
    n_pages = math.ceil(len(records) / per_page)
    with PdfPages(args.out_dir / "luke_v12_r005_rf_gallery.pdf") as pdf:
        for page in range(n_pages):
            chunk = records[page * per_page : (page + 1) * per_page]
            fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
            axes = axes.ravel()
            for ax in axes[len(chunk) :]:
                ax.axis("off")
            for ax, rec in zip(axes, chunk):
                im = np.asarray(rec["rf_image"])
                vmax = np.nanmax(np.abs(im))
                vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
                ax.imshow(im, cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
                if rec.get("contour_az_deg") is not None and rec.get("rf_status") == "ok":
                    # Plot the contour in pixel coordinates for legibility.
                    # The summary plots carry the degree geometry.
                    pass
                ecc = rec.get("eccentricity_deg", np.nan)
                dia = rec.get("rf_equiv_diameter_deg", np.nan)
                title = (
                    f"{rec['session']} {rec['eye']} cid {rec['cid']}\n"
                    f"{rec['category']}\n"
                    f"ecc={ecc:.2f} deg dia={dia:.2f} deg"
                )
                ax.set_title(title, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"Luke v12+R2>=0.05 RF gallery page {page + 1}/{n_pages}", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
