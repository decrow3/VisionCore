"""Summarize BackImage twin drift-geometry outputs.

This is a posthoc reporting helper for
``run_backimage_twin_drift_geometry.py``.  It does not re-score images or run
the twin.  It reads the output CSVs, writes compact result tables, and produces
a short markdown interpretation centered on the current adjudication question:
does a V1-twin objective explain observed drift axes beyond raw edge geometry?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path(
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_twin_drift_geometry_scaled_n256_twin_axis_only"
)
DEFAULT_OUT_DIR = DEFAULT_INPUT_DIR / "summary"
KEY_OBJECTIVES = (
    "raw_edge_axis",
    "optimized_PB",
    "optimized_PA",
    "optimized_Pareto_lambda_0.5",
    "adversarial_Pareto_lambda_0.5",
    "raw_gradient_axis",
    "raw_spectrum_axis",
)
REQUIRED_OBJECTIVES = (
    "raw_edge_axis",
    "optimized_PB",
    "optimized_PA",
    "optimized_Pareto_lambda_0.5",
    "adversarial_Pareto_lambda_0.5",
)
SUMMARY_COLUMNS = (
    "objective",
    "n_windows",
    "n_sessions",
    "mean_cos2_session_mean",
    "weighted_cos2_session_mean",
    "n_sessions_positive",
)
DELTA_COLUMNS = (
    "objective",
    "baseline_objective",
    "n_sessions",
    "mean_delta_cos2_session",
    "ci95_low",
    "ci95_high",
    "n_sessions_delta_positive",
)
NULL_COLUMNS = (
    "objective",
    "null_type",
    "n_nulls",
    "observed_session_mean_cos2",
    "null_mean",
    "null_ci95_low",
    "null_ci95_high",
    "p_greater_equal",
    "p_less_equal",
)
STRATA_COLUMNS = (
    "objective",
    "stratify_by",
    "stratum",
    "mean_cos2_session_mean",
)


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV is missing: {path}")
    return pd.read_csv(path)


def _read_json_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_columns(df: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _require_objectives(df: pd.DataFrame, path: Path, objectives: tuple[str, ...]) -> None:
    missing = sorted(set(objectives) - set(df["objective"]))
    if missing:
        raise ValueError(f"{path} is missing required objective rows: {missing}")


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def _select(df: pd.DataFrame, objectives: tuple[str, ...] = KEY_OBJECTIVES) -> pd.DataFrame:
    if df.empty or "objective" not in df.columns:
        return pd.DataFrame()
    return df[df["objective"].isin(objectives)].copy()


def _rows_for_null(nulls: pd.DataFrame, objective: str, null_type: str) -> pd.DataFrame:
    return nulls[(nulls["objective"] == objective) & (nulls["null_type"] == null_type)]


def _ordered_key_objectives(summary: pd.DataFrame) -> pd.DataFrame:
    key = _select(summary)
    if key.empty:
        return key
    order = {objective: i for i, objective in enumerate(KEY_OBJECTIVES)}
    key["_order"] = key["objective"].map(order).fillna(999)
    return key.sort_values("_order").drop(columns=["_order"])


def _plot_objectives(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    order = summary.sort_values("mean_cos2_session_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, 0.3 * len(order)), 4.0), dpi=150)
    colors = ["#2f6f4e" if obj == "raw_edge_axis" else "#4c78a8" for obj in order["objective"]]
    ax.bar(np.arange(order.shape[0]), order["mean_cos2_session_mean"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(order.shape[0]))
    ax.set_xticklabels(order["objective"], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("mean session cos2(real - predicted)")
    ax.set_title("BackImage drift-axis alignment by objective", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_objective_alignment_ranked.png", dpi=150)
    plt.close(fig)


def _plot_paired_deltas(deltas: pd.DataFrame, out_dir: Path) -> None:
    if deltas.empty:
        return
    key = _select(deltas, tuple(obj for obj in KEY_OBJECTIVES if obj != "raw_edge_axis"))
    if key.empty:
        key = deltas.copy()
    key = key.sort_values("mean_delta_cos2_session", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.28 * key.shape[0])), dpi=150)
    y = np.arange(key.shape[0])
    x = key["mean_delta_cos2_session"].to_numpy(dtype=float)
    lo = key["ci95_low"].to_numpy(dtype=float)
    hi = key["ci95_high"].to_numpy(dtype=float)
    ax.barh(y, x, color="#b55d60")
    ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="none", ecolor="black", linewidth=0.8, capsize=2)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(key["objective"], fontsize=8)
    ax.set_xlabel("session mean delta vs raw_edge_axis")
    ax.set_title("Paired session deltas versus raw edge", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_paired_deltas_vs_raw_edge.png", dpi=150)
    plt.close(fig)


def _plot_key_strata(strata: pd.DataFrame, out_dir: Path) -> None:
    if strata.empty:
        return
    key = strata[
        strata["objective"].isin(("raw_edge_axis", "optimized_PB", "optimized_PA", "adversarial_Pareto_lambda_0.5"))
        & strata["stratify_by"].isin(("image_orientation_coherence", "drift_anisotropy", "alignment_weight"))
    ].copy()
    if key.empty:
        return
    labels = []
    values = []
    colors = []
    palette = {
        "raw_edge_axis": "#2f6f4e",
        "optimized_PB": "#4c78a8",
        "optimized_PA": "#b55d60",
        "adversarial_Pareto_lambda_0.5": "#8063a7",
    }
    for _, row in key.sort_values(["stratify_by", "stratum", "objective"]).iterrows():
        labels.append(f"{row['stratify_by']} {int(row['stratum'])}\n{row['objective']}")
        values.append(float(row["mean_cos2_session_mean"]))
        colors.append(palette.get(str(row["objective"]), "#777777"))
    fig, ax = plt.subplots(figsize=(max(9, 0.32 * len(values)), 4.2), dpi=150)
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("mean session cos2")
    ax.set_title("Reliability-stratified alignment", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_key_stratified_alignment.png", dpi=150)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._\n"
    work = df.loc[:, [col for col in columns if col in df.columns]].copy()
    if max_rows is not None:
        work = work.head(max_rows)
    lines = ["| " + " | ".join(work.columns) + " |", "| " + " | ".join(["---"] * len(work.columns)) + " |"]
    for _, row in work.iterrows():
        vals = []
        for col in work.columns:
            val = row[col]
            is_count = col.startswith("n_")
            vals.append(str(int(val)) if is_count and pd.notna(val) else _fmt(val) if isinstance(val, (float, int, np.number)) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _best_row(df: pd.DataFrame, objective: str) -> pd.Series | None:
    if df.empty:
        return None
    rows = df[df["objective"] == objective]
    if rows.empty:
        return None
    return rows.iloc[0]


def _required_row(df: pd.DataFrame, objective: str, table_name: str) -> pd.Series:
    row = _best_row(df, objective)
    if row is None:
        raise ValueError(f"{table_name} is missing required objective row: {objective}")
    return row


def _required_null_row(nulls: pd.DataFrame, objective: str, null_type: str) -> pd.Series:
    rows = _rows_for_null(nulls, objective, null_type)
    if rows.empty:
        raise ValueError(f"alignment_null_summary is missing {objective} / {null_type}")
    return rows.iloc[0]


def _assess_result_pattern(summary: pd.DataFrame, deltas: pd.DataFrame, nulls: pd.DataFrame) -> dict[str, Any]:
    raw = _required_row(summary, "raw_edge_axis", "alignment_by_objective_summary")
    opt_pb = _required_row(summary, "optimized_PB", "alignment_by_objective_summary")
    opt_pa = _required_row(summary, "optimized_PA", "alignment_by_objective_summary")
    opt_pareto = _required_row(summary, "optimized_Pareto_lambda_0.5", "alignment_by_objective_summary")
    adv_mid = _required_row(summary, "adversarial_Pareto_lambda_0.5", "alignment_by_objective_summary")
    pb_delta = _required_row(deltas, "optimized_PB", "paired_session_deltas_vs_raw_edge")
    pa_delta = _required_row(deltas, "optimized_PA", "paired_session_deltas_vs_raw_edge")
    pareto_delta = _required_row(deltas, "optimized_Pareto_lambda_0.5", "paired_session_deltas_vs_raw_edge")

    raw_random = _required_null_row(nulls, "raw_edge_axis", "random_axis_candidate_grid")
    pb_random = _required_null_row(nulls, "optimized_PB", "random_axis_candidate_grid")
    pa_random = _required_null_row(nulls, "optimized_PA", "random_axis_candidate_grid")
    adv_random = _required_null_row(nulls, "adversarial_Pareto_lambda_0.5", "random_axis_candidate_grid")
    adv_within = _required_null_row(nulls, "adversarial_Pareto_lambda_0.5", "within_session_predicted_axis_shuffle")
    adv_across = _required_null_row(nulls, "adversarial_Pareto_lambda_0.5", "across_session_predicted_axis_shuffle")

    raw_mean = float(raw["mean_cos2_session_mean"])
    optimized_means = [
        float(opt_pb["mean_cos2_session_mean"]),
        float(opt_pa["mean_cos2_session_mean"]),
        float(opt_pareto["mean_cos2_session_mean"]),
    ]
    optimized_deltas = [pb_delta, pa_delta, pareto_delta]
    raw_beats_current_twin = all(raw_mean > val for val in optimized_means) and all(
        float(row["mean_delta_cos2_session"]) <= 0 for row in optimized_deltas
    )
    raw_is_significant = float(raw_random["p_greater_equal"]) <= 0.05 and raw_mean > 0
    pb_pa_not_random_positive = (
        float(pb_random["p_greater_equal"]) > 0.05
        and float(pa_random["p_greater_equal"]) > 0.05
    )
    adv_not_clean_shuffle = (
        float(adv_within["p_greater_equal"]) > 0.05
        and float(adv_across["p_greater_equal"]) > 0.05
    )

    return {
        "raw": raw,
        "optimized_PB": opt_pb,
        "optimized_PA": opt_pa,
        "adversarial_Pareto_lambda_0.5": adv_mid,
        "optimized_PB_delta": pb_delta,
        "raw_random": raw_random,
        "optimized_PB_random": pb_random,
        "optimized_PA_random": pa_random,
        "adversarial_random": adv_random,
        "adversarial_within_shuffle": adv_within,
        "adversarial_across_shuffle": adv_across,
        "raw_beats_current_twin": raw_beats_current_twin,
        "raw_is_significant": raw_is_significant,
        "pb_pa_not_random_positive": pb_pa_not_random_positive,
        "adversarial_not_clean_shuffle": adv_not_clean_shuffle,
        "supports_current_interpretation": raw_beats_current_twin
        and raw_is_significant
        and pb_pa_not_random_positive
        and adv_not_clean_shuffle,
    }


def _write_report(
    out_dir: Path,
    input_dir: Path,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    nulls: pd.DataFrame,
    strata: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    assessment = _assess_result_pattern(summary, deltas, nulls)
    raw = assessment["raw"]
    opt_pb = assessment["optimized_PB"]
    opt_pa = assessment["optimized_PA"]
    adv_mid = assessment["adversarial_Pareto_lambda_0.5"]
    pb_delta = assessment["optimized_PB_delta"]
    raw_null = assessment["raw_random"]

    if assessment["supports_current_interpretation"]:
        headline = (
            "Raw local edge orientation is the clearest predictor of observed BackImage drift axes. "
            "In this fixed candidate-grid run, the V1-twin PA/PB/Pareto axis selections do not explain drift axes beyond raw edge geometry."
        )
        interpretation = [
            "The scaled run resolves this particular candidate-grid test, not the full trajectory-optimization question. "
            "Observed drift is edge-aligned in this cache, while the current fixed-amplitude, fixed-shape twin-selected axes "
            "are not the winning explanation. Treat this as evidence against this specific PA/PB/Pareto axis-selector setup, "
            "not as a broad rejection of active sensing or richer trajectory objectives.",
            "",
            "Fair claim:",
            "",
            "```text",
            "Observed BackImage drift is modestly and robustly aligned with local edge geometry. "
            "In this fixed candidate-grid run, the V1-twin PA/PB/Pareto axis objectives do not outperform raw edge orientation.",
            "```",
        ]
    else:
        headline = (
            "This run does not match the expected scaled-run result pattern. "
            "Inspect the tables before carrying over the raw-edge-wins interpretation."
        )
        interpretation = [
            "At least one guard condition failed for the standard interpretation:",
            "",
            f"- raw edge above random-axis null: `{assessment['raw_is_significant']}`",
            f"- raw edge beats current optimized twin axes: `{assessment['raw_beats_current_twin']}`",
            f"- optimized PA/PB not random-axis positive: `{assessment['pb_pa_not_random_positive']}`",
            f"- adversarial midpoint not cleanly above predicted-axis shuffles: `{assessment['adversarial_not_clean_shuffle']}`",
            "",
            "Treat this report as a table/figure dump until the run is interpreted manually.",
        ]

    cfg = metadata.get("config", {})
    lines = [
        "# BackImage Twin Drift-Geometry Summary",
        "",
        f"Input directory: `{input_dir}`",
        "",
        "## Run",
        "",
        f"- Score mode: `{cfg.get('score_mode', 'unknown')}`",
        f"- Windows: `{metadata.get('n_reliable_rows', 'unknown')}`",
        f"- Candidate rows: `{metadata.get('n_candidate_rows', 'unknown')}`",
        f"- Axis grid: `{cfg.get('axes_deg', [])}`",
        f"- Patch margin: `{cfg.get('min_patch_image_margin_px', 'unknown')}` px",
        f"- Twin population: `{cfg.get('twin_population_n', 'unknown')}`",
        f"- Axis nulls: `{cfg.get('n_axis_nulls', 'unknown')}`; shuffle nulls: `{cfg.get('n_shuffle_nulls', 'unknown')}`; session bootstraps: `{cfg.get('n_session_bootstrap', 'unknown')}`",
        "",
        "## Headline",
        "",
        headline,
        "",
    ]
    raw_p = raw_null["p_greater_equal"]
    lines.append(
        f"- `raw_edge_axis`: session mean cos2 `{_fmt(raw['mean_cos2_session_mean'])}`, "
        f"weighted `{_fmt(raw['weighted_cos2_session_mean'])}`, "
        f"`{int(raw['n_sessions_positive'])}/{int(raw['n_sessions'])}` positive sessions, "
        f"random-axis p_ge `{_fmt(raw_p, 4)}`."
    )
    lines.append(
        f"- `optimized_PB`: session mean cos2 `{_fmt(opt_pb['mean_cos2_session_mean'])}`, "
        f"weighted `{_fmt(opt_pb['weighted_cos2_session_mean'])}`."
    )
    lines.append(
        f"- `optimized_PA`: session mean cos2 `{_fmt(opt_pa['mean_cos2_session_mean'])}`, "
        f"weighted `{_fmt(opt_pa['weighted_cos2_session_mean'])}`."
    )
    lines.append(
        f"- `optimized_PB - raw_edge_axis`: mean paired session delta "
        f"`{_fmt(pb_delta['mean_delta_cos2_session'])}`, CI "
        f"`[{_fmt(pb_delta['ci95_low'])}, {_fmt(pb_delta['ci95_high'])}]`, "
        f"`{int(pb_delta['n_sessions_delta_positive'])}/{int(pb_delta['n_sessions'])}` sessions positive."
    )
    lines.append(
        f"- `adversarial_Pareto_lambda_0.5` session mean cos2 "
        f"`{_fmt(adv_mid['mean_cos2_session_mean'])}`; predicted-axis shuffle p_ge values are "
        f"`{_fmt(assessment['adversarial_within_shuffle']['p_greater_equal'], 4)}` within session and "
        f"`{_fmt(assessment['adversarial_across_shuffle']['p_greater_equal'], 4)}` across session."
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        *interpretation,
        "",
        "## Key Objective Table",
        "",
        _markdown_table(
            _ordered_key_objectives(summary),
            [
                "objective",
                "n_windows",
                "n_sessions",
                "mean_cos2_session_mean",
                "weighted_cos2_session_mean",
                "n_sessions_positive",
            ],
        ),
        "",
        "## Paired Deltas Versus Raw Edge",
        "",
        _markdown_table(
            _select(deltas, tuple(obj for obj in KEY_OBJECTIVES if obj != "raw_edge_axis")),
            [
                "objective",
                "n_sessions",
                "mean_delta_cos2_session",
                "ci95_low",
                "ci95_high",
                "n_sessions_delta_positive",
            ],
        ),
        "",
        "## Null Notes",
        "",
        f"- `raw_edge_axis` random-axis p_ge: `{_fmt(assessment['raw_random']['p_greater_equal'], 4)}`.",
        f"- `optimized_PB` random-axis p_ge: `{_fmt(assessment['optimized_PB_random']['p_greater_equal'], 4)}`.",
        f"- `optimized_PA` random-axis p_ge: `{_fmt(assessment['optimized_PA_random']['p_greater_equal'], 4)}`.",
        f"- `adversarial_Pareto_lambda_0.5` random-axis p_ge: `{_fmt(assessment['adversarial_random']['p_greater_equal'], 4)}`.",
        f"- `adversarial_Pareto_lambda_0.5` predicted-axis shuffle p_ge: within-session "
        f"`{_fmt(assessment['adversarial_within_shuffle']['p_greater_equal'], 4)}`, across-session "
        f"`{_fmt(assessment['adversarial_across_shuffle']['p_greater_equal'], 4)}`.",
        "",
        "## Key Null Table",
        "",
        _markdown_table(
            nulls[
                nulls["objective"].isin(("raw_edge_axis", "optimized_PB", "optimized_PA", "adversarial_Pareto_lambda_0.5"))
            ],
            [
                "objective",
                "null_type",
                "n_nulls",
                "observed_session_mean_cos2",
                "null_mean",
                "null_ci95_low",
                "null_ci95_high",
                "p_greater_equal",
            ],
        ),
        "",
        "## Useful Figures",
        "",
        "- `fig_objective_alignment_ranked.png`",
        "- `fig_paired_deltas_vs_raw_edge.png`",
        "- `fig_key_stratified_alignment.png`",
        "",
    ])
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else input_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = input_dir / "alignment_by_objective_summary.csv"
    deltas_path = input_dir / "paired_session_deltas_vs_raw_edge.csv"
    nulls_path = input_dir / "alignment_null_summary.csv"
    strata_path = input_dir / "stratified_alignment_summary.csv"
    metadata_path = input_dir / "run_metadata.json"

    summary = _read_csv_required(summary_path)
    deltas = _read_csv_required(deltas_path)
    nulls = _read_csv_required(nulls_path)
    strata = _read_csv_required(strata_path)
    metadata = _read_json_required(metadata_path)

    _require_columns(summary, summary_path, SUMMARY_COLUMNS)
    _require_columns(deltas, deltas_path, DELTA_COLUMNS)
    _require_columns(nulls, nulls_path, NULL_COLUMNS)
    _require_columns(strata, strata_path, STRATA_COLUMNS)
    _require_objectives(summary, summary_path, REQUIRED_OBJECTIVES)
    _require_objectives(deltas, deltas_path, REQUIRED_OBJECTIVES[1:])
    _require_objectives(nulls, nulls_path, REQUIRED_OBJECTIVES)

    ranked = summary.sort_values("mean_cos2_session_mean", ascending=False) if not summary.empty else summary
    ranked.to_csv(out_dir / "objective_alignment_ranked.csv", index=False)
    _ordered_key_objectives(summary).to_csv(out_dir / "key_objective_alignment.csv", index=False)
    _select(deltas, tuple(obj for obj in KEY_OBJECTIVES if obj != "raw_edge_axis")).to_csv(out_dir / "key_paired_deltas_vs_raw_edge.csv", index=False)
    _select(nulls).to_csv(out_dir / "key_null_summary.csv", index=False)
    if not strata.empty:
        key_strata = strata[strata["objective"].isin(KEY_OBJECTIVES)].copy()
    else:
        key_strata = pd.DataFrame()
    key_strata.to_csv(out_dir / "key_stratified_alignment.csv", index=False)

    _plot_objectives(summary, out_dir)
    _plot_paired_deltas(deltas, out_dir)
    _plot_key_strata(strata, out_dir)
    _write_report(out_dir, input_dir, summary, deltas, nulls, strata, metadata)

    print(f"Wrote BackImage twin drift-geometry summary to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
