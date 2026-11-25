"""Audit whether battery knee labels represent accelerated degradation.

This script is intentionally analysis-only. It does not modify manuscript text,
figures, source labels, or model outputs. It compares the existing independent
two-line labels with a continuous broken-stick model and quantifies acceleration,
model evidence, threshold sensitivity, and moving-block bootstrap stability.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = ["piecewise", "kneedle", "rate_threshold"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curves",
        default="data/processed_cycles_soh_corrected_20170630.csv",
    )
    parser.add_argument("--existing-labels", default="data/knee_labels_20170630.csv")
    parser.add_argument(
        "--alternative-labels",
        default="results/knee_label_stability/knee_labels_by_method.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/reviewer_knee_definition_audit",
    )
    parser.add_argument("--min-frac", type=float, default=0.20)
    parser.add_argument("--max-frac", type=float, default=0.90)
    parser.add_argument("--min-segment-points", type=int, default=30)
    parser.add_argument("--slope-ratio-threshold", type=float, default=1.50)
    parser.add_argument("--delta-bic-threshold", type=float, default=10.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def candidate_indices(
    n: int, min_frac: float, max_frac: float, min_segment_points: int
) -> np.ndarray:
    low = max(min_segment_points, int(np.floor(n * min_frac)))
    high = min(n - min_segment_points, int(np.ceil(n * max_frac)))
    return np.arange(low, high, dtype=int)


def continuous_scan(
    x: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
) -> pd.DataFrame:
    """Fit y = beta0 + beta1*x + beta2*max(0, x-ck) at every candidate."""
    n = len(x)
    sum_x = float(x.sum())
    sum_x2 = float(np.dot(x, x))
    sum_y = float(y.sum())
    sum_xy = float(np.dot(x, y))
    sum_y2 = float(np.dot(y, y))

    suffix_x = np.cumsum(x[::-1])[::-1]
    suffix_x2 = np.cumsum(np.square(x)[::-1])[::-1]
    suffix_y = np.cumsum(y[::-1])[::-1]
    suffix_xy = np.cumsum((x * y)[::-1])[::-1]

    starts = candidates + 1
    count_after = n - starts
    ck = x[candidates]
    sx = suffix_x[starts]
    sx2 = suffix_x2[starts]
    sy = suffix_y[starts]
    sxy = suffix_xy[starts]

    sum_h = sx - count_after * ck
    sum_xh = sx2 - ck * sx
    sum_h2 = sx2 - 2.0 * ck * sx + count_after * np.square(ck)
    sum_hy = sxy - ck * sy

    count = len(candidates)
    xtx = np.empty((count, 3, 3), dtype=float)
    xtx[:, 0, 0] = n
    xtx[:, 0, 1] = xtx[:, 1, 0] = sum_x
    xtx[:, 0, 2] = xtx[:, 2, 0] = sum_h
    xtx[:, 1, 1] = sum_x2
    xtx[:, 1, 2] = xtx[:, 2, 1] = sum_xh
    xtx[:, 2, 2] = sum_h2
    xty = np.column_stack(
        [np.full(count, sum_y), np.full(count, sum_xy), sum_hy]
    )
    beta = np.linalg.solve(xtx, xty[..., None]).squeeze(-1)
    rss = np.maximum(0.0, sum_y2 - np.einsum("ij,ij->i", beta, xty))
    pre_slope = beta[:, 1]
    slope_change = beta[:, 2]
    post_slope = pre_slope + slope_change
    slope_ratio = np.abs(post_slope) / np.maximum(np.abs(pre_slope), 1e-12)
    return pd.DataFrame(
        {
            "split_index": candidates,
            "knee_cycle": ck,
            "rss": rss,
            "intercept": beta[:, 0],
            "pre_slope": pre_slope,
            "slope_change": slope_change,
            "post_slope": post_slope,
            "abs_slope_ratio": slope_ratio,
        }
    )


def single_line_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    rss = float(np.square(y - fitted).sum())
    return beta, fitted, rss


def broken_stick_fitted(x: np.ndarray, best: pd.Series) -> np.ndarray:
    return (
        float(best.intercept)
        + float(best.pre_slope) * x
        + float(best.slope_change) * np.maximum(0.0, x - float(best.knee_cycle))
    )


def bic(rss: float, n: int, parameter_count: int) -> float:
    return float(n * np.log(max(rss, np.finfo(float).tiny) / n) + parameter_count * np.log(n))


def circular_moving_block_sample(
    residuals: np.ndarray, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    n = len(residuals)
    starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
    offsets = np.arange(block_length)
    sampled = np.concatenate([residuals[(start + offsets) % n] for start in starts])[:n]
    return sampled


def bootstrap_cell(
    x: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray,
    candidates: np.ndarray,
    replicates: int,
    seed: int,
    slope_ratio_threshold: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    block_length = max(9, int(round(np.sqrt(len(x)))))
    centered = residuals - residuals.mean()
    rows = []
    for replicate in range(replicates):
        y_star = fitted + circular_moving_block_sample(centered, block_length, rng)
        scan = continuous_scan(x, y_star, candidates)
        best = scan.loc[scan.rss.idxmin()]
        rows.append(
            {
                "replicate": replicate,
                "knee_cycle": float(best.knee_cycle),
                "pre_slope": float(best.pre_slope),
                "post_slope": float(best.post_slope),
                "slope_change": float(best.slope_change),
                "abs_slope_ratio": float(best.abs_slope_ratio),
                "accelerates": bool(best.pre_slope < 0 and best.post_slope < best.pre_slope),
                "passes_ratio": bool(best.abs_slope_ratio >= slope_ratio_threshold),
            }
        )
    result = pd.DataFrame(rows)
    result["block_length"] = block_length
    return result


def quantile(series: pd.Series, value: float) -> float:
    return float(series.quantile(value))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(args.curves)
    existing = pd.read_csv(args.existing_labels)
    alternatives = pd.read_csv(args.alternative_labels)
    alternative_wide = alternatives.pivot(
        index="cell_id", columns="method", values="knee_cycle"
    ).reset_index()

    label_rows = []
    bootstrap_rows = []
    for cell_number, (cell_id, group) in enumerate(curves.groupby("cell_id", sort=False)):
        part = group.loc[group.soh_mono.notna()].sort_values("cycle")
        x = part.cycle.to_numpy(float)
        y = part.soh_mono.to_numpy(float)
        candidates = candidate_indices(
            len(x), args.min_frac, args.max_frac, args.min_segment_points
        )
        scan = continuous_scan(x, y, candidates)
        unconstrained = scan.loc[scan.rss.idxmin()]
        eligible = scan.loc[
            (scan.pre_slope < 0)
            & (scan.post_slope < scan.pre_slope)
            & (scan.abs_slope_ratio >= args.slope_ratio_threshold)
        ]
        constrained = eligible.loc[eligible.rss.idxmin()] if len(eligible) else None
        _, _, single_rss = single_line_fit(x, y)
        delta_bic = bic(single_rss, len(x), 2) - bic(float(unconstrained.rss), len(x), 4)
        fitted = broken_stick_fitted(x, unconstrained)
        residuals = y - fitted
        cell_seed = args.seed + 1009 * cell_number
        bootstrap = bootstrap_cell(
            x,
            fitted,
            residuals,
            candidates,
            args.bootstrap_replicates,
            cell_seed,
            args.slope_ratio_threshold,
        )
        bootstrap.insert(0, "cell_id", cell_id)
        bootstrap_rows.append(bootstrap)

        current = existing.loc[existing.cell_id == cell_id].iloc[0]
        selected = constrained if constrained is not None else unconstrained
        label_rows.append(
            {
                "cell_id": cell_id,
                "batch": group.batch.iloc[0],
                "official_split": group.official_split.iloc[0],
                "cycle_life": float(group.cycle_life.iloc[0]),
                "n_points": len(x),
                "existing_independent_cycle": float(current.knee_cycle),
                "continuous_unconstrained_cycle": float(unconstrained.knee_cycle),
                "continuous_constrained_cycle": float(selected.knee_cycle),
                "continuous_minus_existing": float(selected.knee_cycle - current.knee_cycle),
                "pre_slope": float(selected.pre_slope),
                "post_slope": float(selected.post_slope),
                "slope_change": float(selected.slope_change),
                "abs_slope_ratio": float(selected.abs_slope_ratio),
                "single_line_rss": single_rss,
                "continuous_rss": float(selected.rss),
                "delta_bic": delta_bic,
                "passes_acceleration": bool(
                    selected.pre_slope < 0 and selected.post_slope < selected.pre_slope
                ),
                "passes_ratio_1_5": bool(selected.abs_slope_ratio >= 1.5),
                "passes_delta_bic_10": bool(delta_bic >= args.delta_bic_threshold),
                "passes_primary_rule": bool(
                    constrained is not None and delta_bic >= args.delta_bic_threshold
                ),
                "bootstrap_knee_ci_low": quantile(bootstrap.knee_cycle, 0.025),
                "bootstrap_knee_ci_high": quantile(bootstrap.knee_cycle, 0.975),
                "bootstrap_knee_ci_width": quantile(bootstrap.knee_cycle, 0.975)
                - quantile(bootstrap.knee_cycle, 0.025),
                "bootstrap_knee_median": quantile(bootstrap.knee_cycle, 0.5),
                "bootstrap_acceleration_support": float(bootstrap.accelerates.mean()),
                "bootstrap_ratio_support": float(bootstrap.passes_ratio.mean()),
                "bootstrap_slope_change_ci_low": quantile(bootstrap.slope_change, 0.025),
                "bootstrap_slope_change_ci_high": quantile(bootstrap.slope_change, 0.975),
                "bootstrap_block_length": int(bootstrap.block_length.iloc[0]),
            }
        )

    per_cell = pd.DataFrame(label_rows).merge(
        alternative_wide, on="cell_id", how="left", validate="one_to_one"
    )
    bootstrap_all = pd.concat(bootstrap_rows, ignore_index=True)

    thresholds = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    threshold_rows = []
    for threshold in thresholds:
        for cohort, part in [("all", per_cell), *per_cell.groupby("official_split")]:
            passes = (
                part.passes_acceleration
                & (part.abs_slope_ratio >= threshold)
                & (part.delta_bic >= args.delta_bic_threshold)
            )
            threshold_rows.append(
                {
                    "slope_ratio_threshold": threshold,
                    "cohort": cohort,
                    "n": len(part),
                    "passing_n": int(passes.sum()),
                    "passing_fraction": float(passes.mean()),
                }
            )
    threshold_sensitivity = pd.DataFrame(threshold_rows)

    agreement_rows = []
    revised = per_cell.continuous_constrained_cycle
    for method, values in [
        ("existing_independent", per_cell.existing_independent_cycle),
        ("chord_distance", per_cell.kneedle),
        ("rate_threshold", per_cell.rate_threshold),
    ]:
        difference = values - revised
        agreement_rows.append(
            {
                "comparison": f"continuous_vs_{method}",
                "n": int(difference.notna().sum()),
                "pearson": float(revised.corr(values, method="pearson")),
                "spearman": float(revised.rank().corr(values.rank(), method="pearson")),
                "mean_signed_difference_other_minus_continuous": float(difference.mean()),
                "median_absolute_difference": float(difference.abs().median()),
                "maximum_absolute_difference": float(difference.abs().max()),
                "within_10_cycles_fraction": float((difference.abs() <= 10).mean()),
                "within_50_cycles_fraction": float((difference.abs() <= 50).mean()),
                "within_100_cycles_fraction": float((difference.abs() <= 100).mean()),
            }
        )
    method_agreement = pd.DataFrame(agreement_rows)

    cohort_summary = (
        per_cell.groupby("official_split")
        .agg(
            n=("cell_id", "size"),
            passing_primary_rule_n=("passes_primary_rule", "sum"),
            median_pre_slope=("pre_slope", "median"),
            median_post_slope=("post_slope", "median"),
            median_slope_change=("slope_change", "median"),
            median_abs_slope_ratio=("abs_slope_ratio", "median"),
            minimum_abs_slope_ratio=("abs_slope_ratio", "min"),
            median_delta_bic=("delta_bic", "median"),
            minimum_delta_bic=("delta_bic", "min"),
            median_bootstrap_ci_width=("bootstrap_knee_ci_width", "median"),
            maximum_bootstrap_ci_width=("bootstrap_knee_ci_width", "max"),
            minimum_acceleration_support=("bootstrap_acceleration_support", "min"),
            minimum_ratio_support=("bootstrap_ratio_support", "min"),
        )
        .reset_index()
    )

    failure_reasons = Counter()
    for row in per_cell.itertuples():
        if not row.passes_acceleration:
            failure_reasons["no_acceleration"] += 1
        if not row.passes_ratio_1_5:
            failure_reasons["slope_ratio_below_1.5"] += 1
        if not row.passes_delta_bic_10:
            failure_reasons["delta_bic_below_10"] += 1

    summary = {
        "analysis_status": "analysis_only_no_manuscript_changes",
        "n_cells": int(len(per_cell)),
        "primary_rule": {
            "model": "continuous broken-stick",
            "search_interval": [args.min_frac, args.max_frac],
            "minimum_segment_points": args.min_segment_points,
            "requires_both_slopes_negative": True,
            "requires_post_slope_more_negative": True,
            "minimum_absolute_slope_ratio": args.slope_ratio_threshold,
            "minimum_delta_bic_vs_single_line": args.delta_bic_threshold,
        },
        "passing_primary_rule_n": int(per_cell.passes_primary_rule.sum()),
        "failure_reasons": dict(failure_reasons),
        "continuous_vs_existing": {
            "exact_matches": int((per_cell.continuous_minus_existing == 0).sum()),
            "within_one_cycle": int((per_cell.continuous_minus_existing.abs() <= 1).sum()),
            "maximum_absolute_difference": float(per_cell.continuous_minus_existing.abs().max()),
        },
        "slope_ratio": {
            "minimum": float(per_cell.abs_slope_ratio.min()),
            "median": float(per_cell.abs_slope_ratio.median()),
            "maximum": float(per_cell.abs_slope_ratio.max()),
        },
        "delta_bic": {
            "minimum": float(per_cell.delta_bic.min()),
            "median": float(per_cell.delta_bic.median()),
            "maximum": float(per_cell.delta_bic.max()),
        },
        "bootstrap": {
            "replicates_per_cell": args.bootstrap_replicates,
            "cells_with_acceleration_support_ge_0_95": int(
                (per_cell.bootstrap_acceleration_support >= 0.95).sum()
            ),
            "cells_with_ratio_support_ge_0_95": int(
                (per_cell.bootstrap_ratio_support >= 0.95).sum()
            ),
            "median_breakpoint_ci_width": float(per_cell.bootstrap_knee_ci_width.median()),
            "maximum_breakpoint_ci_width": float(per_cell.bootstrap_knee_ci_width.max()),
        },
    }

    per_cell.to_csv(output_dir / "continuous_knee_per_cell.csv", index=False)
    bootstrap_all.to_csv(output_dir / "continuous_knee_bootstrap.csv", index=False)
    threshold_sensitivity.to_csv(output_dir / "slope_ratio_threshold_sensitivity.csv", index=False)
    method_agreement.to_csv(output_dir / "label_method_agreement.csv", index=False)
    cohort_summary.to_csv(output_dir / "cohort_summary.csv", index=False)
    continuous_labels = per_cell[
        ["cell_id", "batch", "official_split", "continuous_constrained_cycle"]
    ].rename(columns={"continuous_constrained_cycle": "knee_cycle"})
    continuous_labels.insert(0, "method", "continuous_broken_stick")
    labels_for_prediction = pd.concat(
        [
            alternatives[["method", "cell_id", "batch", "official_split", "knee_cycle"]],
            continuous_labels,
        ],
        ignore_index=True,
    )
    labels_for_prediction.to_csv(output_dir / "labels_for_prediction.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    report = [
        "# Accelerated-knee definition audit",
        "",
        "This is an analysis-only audit prepared for reviewer comment R1.1. No manuscript, figure, source-label, or prediction file was modified.",
        "",
        "## Pre-specified primary rule",
        "",
        f"- Continuous broken-stick model searched over {args.min_frac:.0%}--{args.max_frac:.0%} of each full-life trajectory.",
        f"- At least {args.min_segment_points} observations on each side.",
        "- Both slopes must be negative and the post-break slope must be more negative.",
        f"- Absolute post/pre slope ratio must be at least {args.slope_ratio_threshold:.2f}.",
        f"- Delta BIC versus a single line must be at least {args.delta_bic_threshold:.1f}.",
        f"- Stability used {args.bootstrap_replicates} circular moving-block residual bootstrap replicates per cell.",
        "",
        "## Headline results",
        "",
        f"- Cells passing the primary rule: {summary['passing_primary_rule_n']}/{summary['n_cells']}.",
        f"- Continuous versus existing labels: {summary['continuous_vs_existing']['exact_matches']} exact; all {summary['continuous_vs_existing']['within_one_cycle']} within one cycle.",
        f"- Absolute slope ratio: minimum {summary['slope_ratio']['minimum']:.3f}, median {summary['slope_ratio']['median']:.3f}.",
        f"- Delta BIC: minimum {summary['delta_bic']['minimum']:.1f}, median {summary['delta_bic']['median']:.1f}.",
        f"- Bootstrap acceleration support >=0.95: {summary['bootstrap']['cells_with_acceleration_support_ge_0_95']}/{summary['n_cells']}.",
        f"- Bootstrap slope-ratio support >=0.95: {summary['bootstrap']['cells_with_ratio_support_ge_0_95']}/{summary['n_cells']}.",
        f"- Breakpoint CI width: median {summary['bootstrap']['median_breakpoint_ci_width']:.1f} cycles; maximum {summary['bootstrap']['maximum_breakpoint_ci_width']:.1f} cycles.",
        "",
        "## Interpretation boundary",
        "",
        "Passing this operational rule supports an accelerated-degradation breakpoint in the corrected SOH trajectory. It does not by itself establish a unique electrochemical mechanism.",
    ]
    (output_dir / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
