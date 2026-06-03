from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_int(value: str) -> int:
    return int(float(value))


def _to_float(value: str) -> float:
    return float(value)


def _bootstrap_delta_stats(
    within_values: np.ndarray,
    cross_values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    within_idx = rng.integers(0, within_values.size, size=(n_bootstrap, within_values.size))
    cross_idx = rng.integers(0, cross_values.size, size=(n_bootstrap, cross_values.size))
    within_means = within_values[within_idx].mean(axis=1)
    cross_means = cross_values[cross_idx].mean(axis=1)
    delta_samples = within_means - cross_means
    ci_low = float(np.quantile(delta_samples, 0.025))
    ci_high = float(np.quantile(delta_samples, 0.975))
    p_delta_le_0 = float(np.mean(delta_samples <= 0.0))
    return ci_low, ci_high, p_delta_le_0


def _interpretation_label(ci_low: float, ci_high: float, delta: float) -> str:
    if ci_low > 0.0:
        return "within_gt_cross"
    if ci_high < 0.0:
        return "within_lt_cross"
    if delta > 0.0:
        return "uncertain_within_gt_cross"
    if delta < 0.0:
        return "uncertain_within_lt_cross"
    return "uncertain_equal"


def _collect_support(split_rows: list[dict[str, str]], n_samples: int) -> set[int]:
    return {
        _to_int(r["image_id"])
        for r in split_rows
        if _to_int(r["n_samples"]) >= int(n_samples)
    }


def _collect_within(
    split_rows: list[dict[str, str]],
    *,
    n_samples: int,
    k: int,
    image_ids: set[int],
) -> np.ndarray:
    vals = [
        _to_float(r["within_overlap"])
        for r in split_rows
        if _to_int(r["n_samples"]) == int(n_samples)
        and _to_int(r["k_a3"]) == int(k)
        and _to_int(r["image_id"]) in image_ids
    ]
    return np.asarray(vals, dtype=np.float64)


def _collect_cross(
    matrix_rows: list[dict[str, str]],
    *,
    n_samples: int,
    k: int,
    image_ids: set[int],
) -> np.ndarray:
    vals = [
        _to_float(r["overlap"])
        for r in matrix_rows
        if _to_int(r["n_samples"]) == int(n_samples)
        and _to_int(r["k_a3"]) == int(k)
        and _to_int(r["image_i"]) in image_ids
        and _to_int(r["image_j"]) in image_ids
        and _to_int(r["image_i"]) != _to_int(r["image_j"])
    ]
    return np.asarray(vals, dtype=np.float64)


def _build_rows_for_source(
    *,
    source: str,
    matrix_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    direct_ids: set[int],
    common_ids: set[int],
    n_samples: int,
    k_list: list[int],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_set_name, ids in (("direct_surviving", direct_ids), ("matched_common", common_ids)):
        for k in k_list:
            within = _collect_within(split_rows, n_samples=n_samples, k=k, image_ids=ids)
            cross = _collect_cross(matrix_rows, n_samples=n_samples, k=k, image_ids=ids)
            if within.size == 0 or cross.size == 0:
                raise ValueError(
                    f"No values for source={source}, image_set={image_set_name}, n_samples={n_samples}, k={k}"
                )

            within_mean = float(np.mean(within))
            cross_mean = float(np.mean(cross))
            delta = within_mean - cross_mean
            ci_low, ci_high, p_delta_le_0 = _bootstrap_delta_stats(
                within,
                cross,
                n_bootstrap=n_bootstrap,
                seed=seed + (10000 if source == "twin" else 0) + (1000 if image_set_name == "matched_common" else 0) + k,
            )

            row = {
                "source": source,
                "image_set": image_set_name,
                "n_samples": int(n_samples),
                "k": int(k),
                # Keep n_images as a strict image-count integer; this prevents accidental metric leakage into this column.
                "n_images": int(len(ids)),
                "within_mean": within_mean,
                "cross_mean": cross_mean,
                "delta_within_minus_cross": delta,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_p_delta_le_0": p_delta_le_0,
                "cross_mean_absolute": float(abs(cross_mean)),
                "interpretation_label": _interpretation_label(ci_low, ci_high, delta),
            }
            rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "image_set",
        "n_samples",
        "k",
        "n_images",
        "within_mean",
        "cross_mean",
        "delta_within_minus_cross",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_p_delta_le_0",
        "cross_mean_absolute",
        "interpretation_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build final A3 high-support summary CSV")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs") / "twin_covariance_structure" / "a3_fixrsvp_audit" / "Allen_2022-02-16_fixrsvp_a3",
    )
    p.add_argument("--n-samples", type=int, default=320)
    p.add_argument("--k-list", type=str, default="1,2,3")
    p.add_argument("--bootstrap-repeats", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-name", type=str, default="a3_high_support_summary.csv")
    return p


def main() -> None:
    args = build_parser().parse_args()
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]

    rec_matrix = _read_csv(args.base_dir / "source_recorded" / "a3_overlap_matrix_long.csv")
    rec_split = _read_csv(args.base_dir / "source_recorded" / "a3_splithalf_repeats.csv")
    twin_matrix = _read_csv(args.base_dir / "source_twin" / "a3_overlap_matrix_long.csv")
    twin_split = _read_csv(args.base_dir / "source_twin" / "a3_splithalf_repeats.csv")

    rec_direct_ids = _collect_support(rec_split, n_samples=int(args.n_samples))
    twin_direct_ids = _collect_support(twin_split, n_samples=int(args.n_samples))
    common_ids = rec_direct_ids & twin_direct_ids
    if not common_ids:
        raise ValueError(f"No common surviving image IDs at n_samples >= {args.n_samples}")

    rows: list[dict[str, object]] = []
    rows.extend(
        _build_rows_for_source(
            source="recorded",
            matrix_rows=rec_matrix,
            split_rows=rec_split,
            direct_ids=rec_direct_ids,
            common_ids=common_ids,
            n_samples=int(args.n_samples),
            k_list=k_list,
            n_bootstrap=int(args.bootstrap_repeats),
            seed=int(args.seed),
        )
    )
    rows.extend(
        _build_rows_for_source(
            source="twin",
            matrix_rows=twin_matrix,
            split_rows=twin_split,
            direct_ids=twin_direct_ids,
            common_ids=common_ids,
            n_samples=int(args.n_samples),
            k_list=k_list,
            n_bootstrap=int(args.bootstrap_repeats),
            seed=int(args.seed),
        )
    )

    out_path = args.base_dir / args.out_name
    _write_rows(out_path, rows)
    print(str(out_path))


if __name__ == "__main__":
    main()
