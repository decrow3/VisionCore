from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from declan.shared_transformation_geometry.utils import DEFAULT_OUT_ROOT, load_sessions_from_dataset_config, parse_session_name  # type: ignore
else:
    from .utils import DEFAULT_OUT_ROOT, load_sessions_from_dataset_config, parse_session_name


def _read_single_row_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return rows[0]


def _read_rows_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _ci(vals: np.ndarray) -> tuple[float, float]:
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 5 cross-session aggregation for controlled Stage 1 + template confirmation")
    p.add_argument("--dataset-configs-path", type=Path, default=Path("experiments") / "dataset_configs" / "multi_basic_240_rsvp.yaml")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--image-sim-control", type=str, default="pixel_correlation")
    return p


def main() -> None:
    args = build_parser().parse_args()
    sessions = load_sessions_from_dataset_config(Path(args.dataset_configs_path))

    control = str(args.image_sim_control)
    tangent_rows: list[dict[str, object]] = []
    template_rows: list[dict[str, object]] = []

    for sess in sessions:
        subject, date = parse_session_name(sess)
        sess_root = Path(args.out_dir) / sess

        sweep_path = sess_root / "stg_shared_mode_projection_sweep.csv"
        sweep_rows = [r for r in _read_rows_csv(sweep_path) if str(r.get("source", "")).strip().lower() == "recorded"]
        strict_rows = [
            r
            for r in sweep_rows
            if int(float(r.get("recorded_shared_mode_projection_k", "-1"))) == 0
        ]
        if strict_rows:
            row = strict_rows[0]
            control_is_evaluable = str(row.get("control_is_evaluable", "False")).strip().lower() in ("true", "1", "yes")
            tangent_rows.append(
                {
                    "session_id": sess,
                    "subject": subject,
                    "date": date,
                    "source": "recorded",
                    "recorded_shared_mode_projection_k": 0,
                    "control_is_evaluable": bool(control_is_evaluable),
                    "n_images": int(float(row.get("n_images", "0"))),
                    "n_pairs": int(float(row.get("n_pairs", "0"))),
                    "effect_minus_eye_shuffle": _safe_float(row, "effect_minus_eye_shuffle"),
                    "effect_minus_random_map": _safe_float(row, "effect_minus_random_map"),
                    "controlled_effect_minus_eye_shuffle": _safe_float(row, f"controlled_effect_minus_eye_shuffle_{control}"),
                    "controlled_effect_minus_random_map": _safe_float(row, f"controlled_effect_minus_random_map_{control}"),
                    "low_similarity_effect_minus_eye_shuffle": _safe_float(row, f"low_similarity_effect_minus_eye_shuffle_{control}"),
                    "low_similarity_effect_minus_random_map": _safe_float(row, f"low_similarity_effect_minus_random_map_{control}"),
                    "controlled_label": row.get("interpretation_label", "not_available"),
                    "label": row.get("interpretation_label", "not_available"),
                }
            )
        else:
            # Backward-compatible fallback for older outputs.
            row = _read_single_row_csv(sess_root / "source_recorded" / "stg_tangent_summary.csv")
            if row is not None:
                control_is_evaluable = str(row.get("control_is_evaluable", "False")).strip().lower() in ("true", "1", "yes")
                tangent_rows.append(
                    {
                        "session_id": sess,
                        "subject": subject,
                        "date": date,
                        "source": "recorded",
                        "recorded_shared_mode_projection_k": 0,
                        "control_is_evaluable": bool(control_is_evaluable),
                        "n_images": int(float(row.get("n_images", "0"))),
                        "n_pairs": int(float(row.get("n_pairs", "0"))),
                        "effect_minus_eye_shuffle": _safe_float(row, "effect_minus_eye_shuffle"),
                        "effect_minus_random_map": _safe_float(row, "effect_minus_random_map"),
                        "controlled_effect_minus_eye_shuffle": _safe_float(row, f"controlled_effect_minus_eye_shuffle_{control}"),
                        "controlled_effect_minus_random_map": _safe_float(row, f"controlled_effect_minus_random_map_{control}"),
                        "low_similarity_effect_minus_eye_shuffle": _safe_float(row, f"low_similarity_effect_minus_eye_shuffle_{control}"),
                        "low_similarity_effect_minus_random_map": _safe_float(row, f"low_similarity_effect_minus_random_map_{control}"),
                        "controlled_label": row.get("interpretation_label", "not_available"),
                        "label": row.get("interpretation_label", "not_available"),
                    }
                )

        tpath = sess_root / "stg_tangent_template_summary.csv"
        trow = _read_single_row_csv(tpath)
        if trow is not None:
            template_rows.append(
                {
                    "session_id": sess,
                    "subject": subject,
                    "date": date,
                    "template_feature_type": trow.get("template_feature_type", "unknown"),
                    "template_match_semantics": trow.get("template_match_semantics", "unknown"),
                    "n_images_template": int(float(trow.get("n_images_template", "0"))),
                    "mean_effect_signed_minus_eye_shuffle": _safe_float(trow, "mean_effect_signed_minus_eye_shuffle"),
                    "mean_effect_signed_minus_image_shuffle": _safe_float(trow, "mean_effect_signed_minus_image_shuffle"),
                    "mean_effect_signed_minus_random_map": _safe_float(trow, "mean_effect_signed_minus_random_map"),
                    "label": trow.get("interpretation_label", "not_available"),
                }
            )

    out_root = Path(args.out_dir)
    tangent_out = out_root / "stg_stage5_tangent_controlled_aggregation.csv"
    template_out = out_root / "stg_stage5_template_match_aggregation.csv"
    summary_out = out_root / "stg_stage5_summary.json"

    _write_csv(tangent_out, tangent_rows)
    _write_csv(template_out, template_rows)

    tangent_rows_evaluable = [r for r in tangent_rows if bool(r.get("control_is_evaluable", False))]
    allen_rows = [r for r in tangent_rows_evaluable if str(r.get("subject", "")).lower() == "allen"]
    logan_rows = [r for r in tangent_rows_evaluable if str(r.get("subject", "")).lower() == "logan"]

    tang_ctrl_eye = np.asarray(
        [float(r["controlled_effect_minus_eye_shuffle"]) for r in tangent_rows_evaluable if np.isfinite(float(r["controlled_effect_minus_eye_shuffle"]))],
        dtype=np.float64,
    )
    tang_ctrl_rand = np.asarray(
        [float(r["controlled_effect_minus_random_map"]) for r in tangent_rows_evaluable if np.isfinite(float(r["controlled_effect_minus_random_map"]))],
        dtype=np.float64,
    )
    tmpl_signed = np.asarray(
        [float(r["mean_effect_signed_minus_eye_shuffle"]) for r in template_rows if np.isfinite(float(r["mean_effect_signed_minus_eye_shuffle"]))],
        dtype=np.float64,
    )

    def _subject_summary(rows: list[dict[str, object]]) -> dict[str, object]:
        eye = np.asarray([float(r["effect_minus_eye_shuffle"]) for r in rows if np.isfinite(float(r["effect_minus_eye_shuffle"]))], dtype=np.float64)
        rand = np.asarray([float(r["effect_minus_random_map"]) for r in rows if np.isfinite(float(r["effect_minus_random_map"]))], dtype=np.float64)
        return {
            "n_sessions_run": int(len(rows)),
            "n_sessions_evaluable": int(sum(bool(r.get("control_is_evaluable", False)) for r in rows)),
            "mean_effect_minus_eye_shuffle": float(np.mean(eye)) if eye.size else float("nan"),
            "ci_effect_minus_eye_shuffle": list(_ci(eye)),
            "mean_effect_minus_random_map": float(np.mean(rand)) if rand.size else float("nan"),
            "ci_effect_minus_random_map": list(_ci(rand)),
            "fraction_positive_eye_shuffle": float(np.mean(eye > 0.0)) if eye.size else float("nan"),
            "fraction_positive_random_map": float(np.mean(rand > 0.0)) if rand.size else float("nan"),
        }

    summary = {
        "n_sessions_requested": int(len(sessions)),
        "n_sessions_run": int(len(tangent_rows)),
        "n_sessions_evaluable": int(len(tangent_rows_evaluable)),
        "n_sessions_not_evaluable": int(len(tangent_rows) - len(tangent_rows_evaluable)),
        "n_template_rows": int(len(template_rows)),
        "image_similarity_control": control,
        "mean_controlled_effect_minus_eye_shuffle": float(np.mean(tang_ctrl_eye)) if tang_ctrl_eye.size else float("nan"),
        "ci_controlled_effect_minus_eye_shuffle": list(_ci(tang_ctrl_eye)),
        "mean_controlled_effect_minus_random_map": float(np.mean(tang_ctrl_rand)) if tang_ctrl_rand.size else float("nan"),
        "ci_controlled_effect_minus_random_map": list(_ci(tang_ctrl_rand)),
        "mean_template_effect_signed_minus_eye_shuffle": float(np.mean(tmpl_signed)) if tmpl_signed.size else float("nan"),
        "ci_template_effect_signed_minus_eye_shuffle": list(_ci(tmpl_signed)),
        "allen": _subject_summary(allen_rows),
        "logan": _subject_summary(logan_rows),
    }
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(str(tangent_out))
    print(str(template_out))
    print(str(summary_out))


if __name__ == "__main__":
    main()
