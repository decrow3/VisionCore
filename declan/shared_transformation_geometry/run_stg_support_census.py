from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval.fixrsvp import get_fixrsvp_data

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.shared_transformation_geometry.utils import (  # type: ignore
        DEFAULT_OUT_ROOT,
        json_dumps_pretty,
        load_sessions_from_dataset_config,
        parse_session_name,
        harmonize_fixrsvp_arrays,
    )
else:
    from .utils import DEFAULT_OUT_ROOT, json_dumps_pretty, load_sessions_from_dataset_config, parse_session_name, harmonize_fixrsvp_arrays


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _support_stats(support: np.ndarray) -> dict[str, float]:
    if support.size == 0:
        return {"min": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(support)),
        "median": float(np.median(support)),
        "p90": float(np.quantile(support, 0.9)),
        "max": float(np.max(support)),
    }


def _load_twin_model_names() -> tuple[set[str], str]:
    try:
        from scripts.utils import get_model_and_dataset_configs

        model, _ = get_model_and_dataset_configs(mode="standard")
        names = getattr(model, "names", [])
        return {str(x) for x in names}, "loaded"
    except Exception as exc:
        return set(), f"twin_model_load_failed:{type(exc).__name__}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 0 STG support census across sessions")
    p.add_argument(
        "--dataset-configs-path",
        type=Path,
        default=Path("experiments") / "dataset_configs" / "multi_basic_240_rsvp.yaml",
    )
    p.add_argument("--use-cached-data", action="store_true", default=True)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    return p


def main() -> None:
    args = build_parser().parse_args()
    sessions = load_sessions_from_dataset_config(args.dataset_configs_path)
    twin_names, twin_load_note = _load_twin_model_names()

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for session_name in sessions:
        subject, date = parse_session_name(session_name)
        try:
            data = get_fixrsvp_data(
                subject=subject,
                date=date,
                dataset_configs_path=str(args.dataset_configs_path),
                use_cached_data=bool(args.use_cached_data),
            )
            data = harmonize_fixrsvp_arrays(data)

            robs = np.asarray(data["robs"], dtype=np.float64)
            eyepos = np.asarray(data["eyepos"], dtype=np.float64)
            image_ids = np.asarray(data["image_ids"], dtype=np.int64)

            valid = np.isfinite(eyepos).all(axis=2) & (image_ids >= 0) & np.isfinite(robs).all(axis=2)
            img_ids = np.unique(image_ids[valid]).astype(np.int64)
            support = np.asarray([(image_ids[valid] == i).sum() for i in img_ids], dtype=np.int64)
            sup_stats = _support_stats(support)

            twin_ok = bool(session_name in twin_names)
            twin_note = "session_in_model_names" if twin_ok else f"session_not_in_model_names:{twin_load_note}"
            twin_cache_file = args.out_dir / "cache" / "twin_rates" / f"{session_name}.npz"
            twin_cache_status = "available" if twin_cache_file.exists() else "missing"

            row = {
                "subject": subject,
                "date": date,
                "session_id": session_name,
                "source_availability": "both" if twin_ok else "recorded",
                "validated_twin_available": bool(twin_ok),
                "twin_validation_note": twin_note,
                "n_units": int(robs.shape[2]),
                "n_unique_fixrsvp_images": int(img_ids.size),
                "image_support_min": sup_stats["min"],
                "image_support_median": sup_stats["median"],
                "image_support_p90": sup_stats["p90"],
                "image_support_max": sup_stats["max"],
                "n_images_ge_80": int((support >= 80).sum()),
                "n_images_ge_160": int((support >= 160).sum()),
                "n_images_ge_320": int((support >= 320).sum()),
                "n_images_ge_640": int((support >= 640).sum()),
                "eye_trace_availability": bool(np.isfinite(eyepos).any()),
                "twin_response_cache_status": twin_cache_status,
            }
            rows.append(row)
        except Exception as exc:
            twin_ok = bool(session_name in twin_names)
            twin_note = "session_in_model_names" if twin_ok else f"session_not_in_model_names:{twin_load_note}"
            twin_cache_file = args.out_dir / "cache" / "twin_rates" / f"{session_name}.npz"
            twin_cache_status = "available" if twin_cache_file.exists() else "missing"
            rows.append(
                {
                    "subject": subject,
                    "date": date,
                    "session_id": session_name,
                    "source_availability": "twin" if twin_ok else "none",
                    "validated_twin_available": bool(twin_ok),
                    "twin_validation_note": twin_note,
                    "n_units": 0,
                    "n_unique_fixrsvp_images": 0,
                    "image_support_min": float("nan"),
                    "image_support_median": float("nan"),
                    "image_support_p90": float("nan"),
                    "image_support_max": float("nan"),
                    "n_images_ge_80": 0,
                    "n_images_ge_160": 0,
                    "n_images_ge_320": 0,
                    "n_images_ge_640": 0,
                    "eye_trace_availability": False,
                    "twin_response_cache_status": twin_cache_status,
                }
            )
            failures.append(
                {
                    "session_id": session_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    census_csv = out_dir / "stg_support_census.csv"
    summary_json = out_dir / "stg_support_census_summary.json"

    _write_csv(census_csv, rows)
    summary_payload = {
        "dataset_configs_path": str(args.dataset_configs_path),
        "n_sessions_requested": int(len(sessions)),
        "n_sessions_completed": int(len(rows)),
        "n_failures": int(len(failures)),
        "failures": failures,
        "preferred_sessions_ge8_images_ge320": int(sum(int(r["n_images_ge_320"]) >= 8 for r in rows)),
        "minimum_sessions_ge8_images_ge160": int(sum(int(r["n_images_ge_160"]) >= 8 for r in rows)),
    }
    summary_json.write_text(json_dumps_pretty(summary_payload), encoding="utf-8")

    print(str(census_csv))
    print(str(summary_json))


if __name__ == "__main__":
    main()
