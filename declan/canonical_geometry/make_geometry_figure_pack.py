"""Build the canonical cache-first BackImage geometry figure pack."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import shutil

import pandas as pd

from ._config import argv_from_args, enforce_fresh_output_paths, load_config, run_existing_main, section_args


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "fixation_statistics_by_stimulus_all_sessions_after_review"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "figure_geometry_v1.json"
DEFAULT_SECTION = "geometry_figure_pack"
PANEL_D_MODULE = "declan.figure4_active_sensing_atlas.scripts.plot_panel_d_subpanels"
PANEL_E_MODULE = "declan.figure4_active_sensing_atlas.scripts.plot_panel_e_subpanels"

DEFAULTS: dict[str, Any] = {
    "atlas_dir": ROOT / "declan" / "figure4_active_sensing_atlas",
    "matched_axis_dir": BASE / "backimage_axis_conditioned_matched_static_percandidate_gpu1_n64_c4_k16_v1",
    "hardneg_axis_dir": BASE / "backimage_axis_conditioned_hard_negative_shared_source_gpu1_n128_c4_k16_scales_0p5_1_2_v1",
    "stability_dir": BASE / "backimage_edge_parallel_stability_screen_yfix_n256_pop256",
    "objective_dir": BASE / "backimage_conditional_fixation_objectives_twin_axis_only_n256",
    "alignment_dir": BASE / "backimage_edge_alignment_distribution_inspection",
    "window_dir": BASE / "backimage_image_structure_reviewed_v2_screenfiltered_yfix",
    "raw_edge_audit_dir": BASE / "backimage_raw_edge_roadblock_residual_adjudication_v1",
    "out_dir": BASE / "backimage_geometry_figure_pack_canonical_v1",
    "claim_boundary": "geometry behavior evidence; not feature-readout proof",
    "require_raw_edge_audit": True,
}

PANEL_D_TABLES = [
    "panel_D_axis_conditioned_values.csv",
    "panel_D_axis_preference_values.csv",
    "panel_D_edge_stability_values.csv",
    "panel_D_objective_guardrail_values.csv",
]
PANEL_E_TABLES = [
    "panel_E_behavior_example_values.csv",
    "panel_E_alignment_strength_values.csv",
    "panel_E_endpoint_enrichment_values.csv",
    "panel_E_metric_convention_values.csv",
    "panel_E_scope_summary_values.csv",
]
RAW_EDGE_TABLES = [
    "feature_axis_contrasts_context.csv",
    "feature_posterior_axis_delta_by_window.csv",
    "incremental_r2_session_bootstrap.csv",
    "join_qc.csv",
    "join_qc.md",
    "joined_preservation_table.csv",
    "joined_raw_edge_baseline_table.csv",
    "model_block_summary.csv",
    "observer_axis_delta_by_window.csv",
    "predictor_dictionary.csv",
    "raw_edge_alignment_summary.csv",
    "raw_edge_residual_master_table.csv",
    "raw_edge_roadblock_report.md",
    "reduced_model_summary.csv",
    "reduced_model_session_bootstrap.csv",
    "run_metadata.json",
    "session_predictor_sign_counts.csv",
    "spearman_predictor_summary.csv",
    "standardized_coefficients.csv",
    "stratified_model_summary.csv",
]
RAW_EDGE_FIGURES = [
    "fig_raw_edge_alignment_by_confidence.png",
    "fig_preservation_predicts_residual_alignment.png",
    "fig_observer_axis_delta_predicts_residual_alignment.png",
    "fig_incremental_r2_by_block.png",
    "fig_session_delta_r2_signs.png",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _resolve_args(cli: argparse.Namespace) -> dict[str, Any]:
    resolved = dict(DEFAULTS)
    if cli.config is not None:
        config_path = Path(cli.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Missing geometry figure-pack config: {config_path}")
        resolved.update(section_args(load_config(config_path), str(cli.section)))
    for key in DEFAULTS:
        value = getattr(cli, key, None)
        if value is not None:
            resolved[key] = value
    for key in [
        "atlas_dir",
        "matched_axis_dir",
        "hardneg_axis_dir",
        "stability_dir",
        "objective_dir",
        "alignment_dir",
        "window_dir",
        "raw_edge_audit_dir",
        "out_dir",
    ]:
        resolved[key] = _as_path(resolved[key])
    resolved["require_raw_edge_audit"] = bool(resolved.get("require_raw_edge_audit", False))
    return resolved


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _validate_inputs(args: dict[str, Any]) -> list[dict[str, str]]:
    checks = [
        (args["matched_axis_dir"] / "observer_summary.csv", "matched axis observer summary"),
        (args["hardneg_axis_dir"] / "observer_summary.csv", "hard-negative axis observer summary"),
        (args["stability_dir"] / "stability_summary.csv", "edge-parallel stability summary"),
        (args["objective_dir"] / "paired_session_deltas_vs_raw_edge.csv", "objective delta summary"),
        (args["alignment_dir"] / "edge_alignment_distribution_summary.csv", "behavior alignment summary"),
        (args["alignment_dir"] / "endpoint_zone_enrichment_summary.csv", "endpoint enrichment summary"),
        (args["window_dir"] / "orientation_alignment_summary.csv", "orientation alignment summary"),
        (args["window_dir"] / "backimage_image_fem_windows.csv", "BackImage window table"),
    ]
    rows: list[dict[str, str]] = []
    for path, label in checks:
        _require_file(Path(path), label)
        rows.append({"role": label, "path": str(path), "status": "ok"})

    raw_edge_dir = Path(args["raw_edge_audit_dir"])
    if raw_edge_dir.exists():
        for name in ["raw_edge_alignment_summary.csv", "raw_edge_roadblock_report.md"]:
            _require_file(raw_edge_dir / name, f"raw-edge audit {name}")
        rows.append({"role": "raw-edge residual adjudication", "path": str(raw_edge_dir), "status": "ok"})
    elif args["require_raw_edge_audit"]:
        raise FileNotFoundError(f"Missing required raw-edge audit directory: {raw_edge_dir}")
    else:
        rows.append({"role": "raw-edge residual adjudication", "path": str(raw_edge_dir), "status": "missing_optional"})
    return rows


def _panel_d_args(args: dict[str, Any], panel_d_dir: Path) -> dict[str, Any]:
    return {
        "matched_axis_dir": args["matched_axis_dir"],
        "hardneg_axis_dir": args["hardneg_axis_dir"],
        "stability_dir": args["stability_dir"],
        "objective_dir": args["objective_dir"],
        "out_dir": panel_d_dir,
    }


def _panel_e_args(args: dict[str, Any], panel_e_dir: Path) -> dict[str, Any]:
    return {
        "alignment_dir": args["alignment_dir"],
        "window_dir": args["window_dir"],
        "out_dir": panel_e_dir,
    }


def _command(module_name: str, args: dict[str, Any]) -> str:
    return " ".join([sys.executable, "-m", module_name, *argv_from_args(args)])


def _copy_existing(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _collect_panel_tables(panel_dir: Path, table_names: list[str], source_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in table_names:
        src = panel_dir / name
        dst = source_dir / name
        _require_file(src, f"generated panel table {name}")
        shutil.copy2(src, dst)
        rows.append({"source": str(src), "copy": str(dst), "status": "copied"})
    return rows


def _collect_raw_edge(raw_edge_dir: Path, out_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not raw_edge_dir.exists():
        return [{"source": str(raw_edge_dir), "copy": "", "status": "missing_optional"}]
    for name in RAW_EDGE_TABLES:
        src = raw_edge_dir / name
        dst = out_dir / name
        rows.append({"source": str(src), "copy": str(dst), "status": "copied" if _copy_existing(src, dst) else "missing"})
    figure_dir = out_dir / "figures"
    for name in RAW_EDGE_FIGURES:
        src = raw_edge_dir / name
        dst = figure_dir / name
        rows.append({"source": str(src), "copy": str(dst), "status": "copied" if _copy_existing(src, dst) else "missing"})
    return rows


def _require_no_missing_raw_edge_artifacts(rows: list[dict[str, str]]) -> None:
    missing = [row["source"] for row in rows if row["status"] == "missing"]
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:12])
        suffix = "\n..." if len(missing) > 12 else ""
        raise FileNotFoundError(f"Missing required raw-edge figure-pack artifacts:\n{preview}{suffix}")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _key_numbers(out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    alignment = _read_csv_if_exists(out_dir / "panel_E" / "panel_E_alignment_strength_values.csv")
    if not alignment.empty:
        for _, row in alignment.iterrows():
            rows.append(
                {
                    "section": "behavior_edge_alignment",
                    "label": row["subset"],
                    "estimate": float(row["mean_edge_alignment_index_session"]),
                    "ci95_low": float(row["ci95_low_session_mean"]),
                    "ci95_high": float(row["ci95_high_session_mean"]),
                    "n_windows": int(row["n_windows"]),
                    "n_sessions": int(row["n_sessions"]),
                }
            )
    stability = _read_csv_if_exists(out_dir / "panel_D" / "panel_D_edge_stability_values.csv")
    if not stability.empty:
        for _, row in stability.iterrows():
            rows.append(
                {
                    "section": "edge_parallel_stability",
                    "label": row["screen"],
                    "estimate": float(row["mean_advantage_session_mean"]),
                    "ci95_low": float(row["ci95_low_session_mean"]),
                    "ci95_high": float(row["ci95_high_session_mean"]),
                    "n_windows": int(row["n_windows"]),
                    "n_sessions": int(row["n_sessions"]),
                }
            )
    raw = _read_csv_if_exists(out_dir / "raw_edge_audit" / "raw_edge_alignment_summary.csv")
    if not raw.empty:
        for _, row in raw.iterrows():
            rows.append(
                {
                    "section": "raw_edge_residual_adjudication",
                    "label": row["subset"],
                    "estimate": float(row["session_mean_drift_edge_cos2"]),
                    "ci95_low": float(row["session_bootstrap_ci_low"]),
                    "ci95_high": float(row["session_bootstrap_ci_high"]),
                    "n_windows": int(row["n_windows"]),
                    "n_sessions": int(row["n_sessions"]),
                }
            )
    return pd.DataFrame(rows)


def _write_reports(out_dir: Path, args: dict[str, Any], input_checks: list[dict[str, str]], raw_edge_rows: list[dict[str, str]]) -> None:
    source_dir = out_dir / "figure_source_tables"
    key_numbers = _key_numbers(out_dir)
    if not key_numbers.empty:
        key_numbers.to_csv(source_dir / "geometry_key_numbers_for_caption.csv", index=False)

    raw_edge_status = "present" if Path(args["raw_edge_audit_dir"]).exists() else "missing"
    report = [
        "# Canonical Geometry Figure Pack",
        "",
        f"Output folder: `{out_dir}`",
        f"Claim boundary: `{args['claim_boundary']}`",
        "",
        "## Generated Panels",
        "",
        "- `panel_D/`: axis-conditioned observer, edge-parallel stability, and objective guardrails.",
        "- `panel_E/`: free-viewing behavior-edge alignment and metric guardrails.",
        "- `raw_edge_audit/`: copied raw-edge residual adjudication tables and diagnostic figures.",
        "",
        "## Guardrails",
        "",
        "- Panel D supports image-conditioned useful axes and local edge-parallel preservation, not a universal edge-parallel policy.",
        "- Panel E supports modest but reliable behavior-edge alignment; metric convention must be stated.",
        "- Raw-edge residual adjudication is a bridge-candidate result because leave-one-session-out residual transfer remains negative.",
        f"- Raw-edge audit source status: `{raw_edge_status}`.",
        "",
        "## Key Numbers",
        "",
    ]
    if key_numbers.empty:
        report.append("_No key-number table was generated._")
    else:
        for _, row in key_numbers.iterrows():
            report.append(
                f"- `{row['section']}` / `{row['label']}`: {row['estimate']:+.4g} "
                f"[{row['ci95_low']:+.4g}, {row['ci95_high']:+.4g}], "
                f"n={int(row['n_windows'])} windows / {int(row['n_sessions'])} sessions."
            )
    report.extend(
        [
            "",
            "## Source Tables",
            "",
            "- `figure_source_tables/geometry_key_numbers_for_caption.csv`",
            "- `panel_provenance.csv`",
            "- Panel-specific `panel_D/*.csv` and `panel_E/*.csv` value tables",
            "- Raw-edge copied tables under `raw_edge_audit/`",
        ]
    )
    (out_dir / "geometry_figure_pack_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    provenance = pd.DataFrame(
        [
            {
                "panel": "D",
                "source_role": "axis-conditioned observer",
                "source_path": str(args["matched_axis_dir"]) + "; " + str(args["hardneg_axis_dir"]),
                "output_path": str(out_dir / "panel_D"),
                "claim_boundary": args["claim_boundary"],
                "status": "generated",
            },
            {
                "panel": "D",
                "source_role": "edge-parallel stability",
                "source_path": str(args["stability_dir"]),
                "output_path": str(out_dir / "panel_D"),
                "claim_boundary": args["claim_boundary"],
                "status": "generated",
            },
            {
                "panel": "D",
                "source_role": "objective guardrail",
                "source_path": str(args["objective_dir"]),
                "output_path": str(out_dir / "panel_D"),
                "claim_boundary": args["claim_boundary"],
                "status": "generated",
            },
            {
                "panel": "E",
                "source_role": "behavior edge alignment",
                "source_path": str(args["alignment_dir"]) + "; " + str(args["window_dir"]),
                "output_path": str(out_dir / "panel_E"),
                "claim_boundary": args["claim_boundary"],
                "status": "generated",
            },
            {
                "panel": "raw_edge_audit",
                "source_role": "residual roadblock adjudication",
                "source_path": str(args["raw_edge_audit_dir"]),
                "output_path": str(out_dir / "raw_edge_audit"),
                "claim_boundary": args["claim_boundary"],
                "status": raw_edge_status,
            },
        ]
    )
    provenance.to_csv(out_dir / "panel_provenance.csv", index=False)
    pd.DataFrame(input_checks).to_csv(source_dir / "input_contract_checks.csv", index=False)
    pd.DataFrame(raw_edge_rows).to_csv(source_dir / "raw_edge_copied_artifacts.csv", index=False)


def _write_index(out_dir: Path) -> None:
    lines = [
        "# Canonical Geometry Figure Pack",
        "",
        "- `panel_D/`: generated Figure 4D geometry panels and source tables.",
        "- `panel_E/`: generated Figure 4E behavior-geometry panels and source tables.",
        "- `raw_edge_audit/`: copied raw-edge residual-adjudication artifacts.",
        "- `figure_source_tables/`: consolidated checks and caption numbers.",
        "- `panel_provenance.csv`: source-to-panel ledger.",
        "- `geometry_figure_pack_report.md`: short claim-boundary report.",
        "- `figure_pack_metadata.json`: machine-readable provenance.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    args: dict[str, Any],
    *,
    print_only: bool = False,
    validate_only: bool = False,
    allow_existing_output: bool = False,
) -> Path:
    out_dir = Path(args["out_dir"])
    panel_d_dir = out_dir / "panel_D"
    panel_e_dir = out_dir / "panel_E"
    source_dir = out_dir / "figure_source_tables"
    raw_edge_out_dir = out_dir / "raw_edge_audit"

    panel_d_args = _panel_d_args(args, panel_d_dir)
    panel_e_args = _panel_e_args(args, panel_e_dir)
    if print_only:
        print(_command(PANEL_D_MODULE, panel_d_args))
        print(_command(PANEL_E_MODULE, panel_e_args))
        print(f"copy raw-edge audit artifacts from {args['raw_edge_audit_dir']} to {raw_edge_out_dir}")
        return out_dir

    input_checks = _validate_inputs(args)
    if validate_only:
        print(f"Validated canonical geometry figure-pack inputs for {out_dir}")
        return out_dir
    enforce_fresh_output_paths(args, allow_existing=bool(allow_existing_output))

    source_dir.mkdir(parents=True, exist_ok=True)
    run_existing_main(PANEL_D_MODULE, panel_d_args)
    run_existing_main(PANEL_E_MODULE, panel_e_args)
    panel_rows = _collect_panel_tables(panel_d_dir, PANEL_D_TABLES, source_dir)
    panel_rows.extend(_collect_panel_tables(panel_e_dir, PANEL_E_TABLES, source_dir))
    pd.DataFrame(panel_rows).to_csv(source_dir / "panel_value_table_copies.csv", index=False)
    raw_edge_rows = _collect_raw_edge(Path(args["raw_edge_audit_dir"]), raw_edge_out_dir)
    if bool(args.get("require_raw_edge_audit", False)):
        _require_no_missing_raw_edge_artifacts(raw_edge_rows)
    _write_reports(out_dir, args, input_checks, raw_edge_rows)
    _write_index(out_dir)
    _write_json(
        out_dir / "figure_pack_metadata.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": "declan/canonical_geometry/make_geometry_figure_pack.py",
            "output_dir": str(out_dir),
            "panel_d_module": PANEL_D_MODULE,
            "panel_e_module": PANEL_E_MODULE,
            "panel_d_command": _command(PANEL_D_MODULE, panel_d_args),
            "panel_e_command": _command(PANEL_E_MODULE, panel_e_args),
            "claim_boundary": args["claim_boundary"],
            "sources": {key: str(value) for key, value in args.items() if key.endswith("_dir") or key == "atlas_dir"},
        },
    )
    print(f"Wrote canonical geometry figure pack to {out_dir}", flush=True)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--section", default=DEFAULT_SECTION)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into an existing non-empty output directory. Use only for intentional figure-pack refreshes.",
    )
    parser.add_argument("--atlas-dir", type=Path, default=None)
    parser.add_argument("--matched-axis-dir", type=Path, default=None)
    parser.add_argument("--hardneg-axis-dir", type=Path, default=None)
    parser.add_argument("--stability-dir", type=Path, default=None)
    parser.add_argument("--objective-dir", type=Path, default=None)
    parser.add_argument("--alignment-dir", type=Path, default=None)
    parser.add_argument("--window-dir", type=Path, default=None)
    parser.add_argument("--raw-edge-audit-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--claim-boundary", default=None)
    parser.add_argument("--require-raw-edge-audit", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    args = _resolve_args(cli)
    run(
        args,
        print_only=bool(cli.print_command),
        validate_only=bool(cli.validate_only),
        allow_existing_output=bool(cli.allow_existing_output),
    )


if __name__ == "__main__":
    main()
