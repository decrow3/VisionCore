from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from .utils import DEFAULT_OUT_ROOT
from .run_stg_tangent_stage1 import _cos, _subspace_overlap


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_k_list(raw: str) -> list[int]:
    values = [piece.strip() for piece in str(raw).split(",") if piece.strip()]
    return [int(v) for v in values] if values else [0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direct tangent comparison between recorded and twin Stage 1 outputs")
    p.add_argument("--subject", type=str, required=True)
    p.add_argument("--date", type=str, required=True)
    p.add_argument("--recorded-shared-mode-projection-k", type=str, default="0,1,2,3,5,10")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    return p


def _load_maps(path: Path) -> dict[int, dict[str, object]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {int(k): v for k, v in payload.get("images", {}).items()}


def main() -> None:
    args = build_parser().parse_args()
    session_root = Path(args.out_dir) / f"{args.subject}_{args.date}"
    twin_path = session_root / "source_twin" / "stg_tangent_maps.pkl"
    if not twin_path.exists():
        raise FileNotFoundError(f"Missing twin tangent maps: {twin_path}")

    twin_images = _load_maps(twin_path)
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for k in _parse_k_list(args.recorded_shared_mode_projection_k):
        recorded_path = session_root / "source_recorded" / f"projection_k{k}" / "stg_tangent_maps.pkl"
        if not recorded_path.exists():
            summary_rows.append(
                {
                    "session_id": f"{args.subject}_{args.date}",
                    "source": "recorded_vs_twin",
                    "recorded_shared_mode_projection_k": int(k),
                    "unit_dimension_match": False,
                    "direct_template_match_status": "not_run_missing_recorded_maps",
                    "recorded_n_units": 0,
                    "twin_n_units": 0,
                }
            )
            continue

        recorded_images = _load_maps(recorded_path)
        recorded_units = int(next(iter(recorded_images.values()))["n_units"]) if recorded_images else 0
        twin_units = int(next(iter(twin_images.values()))["n_units"]) if twin_images else 0
        unit_match = bool(recorded_units == twin_units and recorded_units > 0)
        if not unit_match:
            summary_rows.append(
                {
                    "session_id": f"{args.subject}_{args.date}",
                    "source": "recorded_vs_twin",
                    "recorded_shared_mode_projection_k": int(k),
                    "unit_dimension_match": False,
                    "direct_template_match_status": "not_run_dimension_mismatch",
                    "recorded_n_units": int(recorded_units),
                    "twin_n_units": int(twin_units),
                }
            )
            continue

        common_images = sorted(set(recorded_images).intersection(twin_images))
        for image_id in common_images:
            rec = recorded_images[image_id]
            twin = twin_images[image_id]
            cos_bx = _cos(rec["bx"], twin["bx"])
            cos_by = _cos(rec["by"], twin["by"])
            mean_signed = float(np.nanmean([cos_bx, cos_by]))
            overlap = _subspace_overlap(rec["basis"], twin["basis"])
            rows.append(
                {
                    "session_id": f"{args.subject}_{args.date}",
                    "source": "recorded_vs_twin",
                    "recorded_shared_mode_projection_k": int(k),
                    "image_id": int(image_id),
                    "unit_dimension_match": True,
                    "recorded_n_units": int(rec["n_units"]),
                    "twin_n_units": int(twin["n_units"]),
                    "cos_bx_rec_twin": float(cos_bx),
                    "cos_by_rec_twin": float(cos_by),
                    "mean_signed_rec_twin": float(mean_signed),
                    "subspace_overlap_rec_twin": float(overlap),
                    "direct_template_match_status": "run",
                }
            )

        per_k_rows = [r for r in rows if int(r["recorded_shared_mode_projection_k"]) == int(k)]
        if per_k_rows:
            summary_rows.append(
                {
                    "session_id": f"{args.subject}_{args.date}",
                    "source": "recorded_vs_twin",
                    "recorded_shared_mode_projection_k": int(k),
                    "unit_dimension_match": True,
                    "direct_template_match_status": "run",
                    "recorded_n_units": int(recorded_units),
                    "twin_n_units": int(twin_units),
                    "mean_cos_bx_rec_twin": float(np.nanmean([float(r["cos_bx_rec_twin"]) for r in per_k_rows])),
                    "mean_cos_by_rec_twin": float(np.nanmean([float(r["cos_by_rec_twin"]) for r in per_k_rows])),
                    "mean_signed_rec_twin": float(np.nanmean([float(r["mean_signed_rec_twin"]) for r in per_k_rows])),
                    "mean_subspace_overlap_rec_twin": float(np.nanmean([float(r["subspace_overlap_rec_twin"]) for r in per_k_rows])),
                }
            )

    _write_csv(session_root / "stg_direct_recorded_twin_tangent_match.csv", rows)
    _write_csv(session_root / "stg_direct_recorded_twin_tangent_match_summary.csv", summary_rows)
    print(str(session_root / "stg_direct_recorded_twin_tangent_match_summary.csv"))


if __name__ == "__main__":
    main()
