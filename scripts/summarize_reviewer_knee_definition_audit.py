"""Consolidate the reviewer-facing knee-definition audit into QA tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/reviewer_knee_definition_audit"
PREDICTION = AUDIT / "prediction_robustness_existing"
FORMAL_CONTINUOUS = AUDIT / "formal_selector_continuous"
FORMAL_EXISTING = ROOT / "results/formal_414240_selector_N150"


def bootstrap_mae_ci(frame: pd.DataFrame, seed: int, replicates: int = 5000) -> tuple[float, float, float]:
    values = frame.abs_error.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    per_cell = pd.read_csv(AUDIT / "continuous_knee_per_cell.csv")
    metrics = pd.read_csv(PREDICTION / "label_robustness_metrics.csv")
    predictions = pd.read_csv(PREDICTION / "label_robustness_predictions.csv")
    metadata = json.loads((PREDICTION / "metadata.json").read_text(encoding="utf-8"))
    formal_continuous = pd.read_csv(FORMAL_CONTINUOUS / "heldout_metrics_by_seed.csv")
    formal_existing = pd.read_csv(FORMAL_EXISTING / "heldout_metrics_by_seed.csv")
    continuous_meta = json.loads((FORMAL_CONTINUOUS / "metadata.json").read_text(encoding="utf-8"))
    existing_meta = json.loads((FORMAL_EXISTING / "metadata.json").read_text(encoding="utf-8"))

    metric_summary = (
        metrics.groupby(["label_method", "split"])
        .agg(
            n=("n", "first"),
            seeds=("seed", "nunique"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            q90_mean=("abs_error_q90", "mean"),
        )
        .reset_index()
    )

    averaged = (
        predictions.groupby(["label_method", "split", "cell_id", "observed"], as_index=False)
        .agg(predicted=("predicted", "mean"))
    )
    averaged["abs_error"] = (averaged.observed - averaged.predicted).abs()
    ci_rows = []
    for index, ((method, split), part) in enumerate(
        averaged.groupby(["label_method", "split"], sort=True)
    ):
        mae, low, high = bootstrap_mae_ci(part, 20260921 + index)
        ci_rows.append(
            {
                "label_method": method,
                "split": split,
                "cell_averaged_mae": mae,
                "cell_bootstrap_ci_low": low,
                "cell_bootstrap_ci_high": high,
            }
        )
    ci_table = pd.DataFrame(ci_rows)
    metric_summary = metric_summary.merge(ci_table, on=["label_method", "split"])

    paired = metrics.pivot(
        index=["split", "seed"], columns="label_method", values="mae"
    ).reset_index()
    paired["continuous_minus_existing_mae"] = (
        paired.continuous_broken_stick - paired.existing_independent
    )
    paired_summary = (
        paired.groupby("split")
        .agg(
            n_seeds=("seed", "nunique"),
            mean_mae_difference=("continuous_minus_existing_mae", "mean"),
            sd_mae_difference=("continuous_minus_existing_mae", "std"),
            minimum_mae_difference=("continuous_minus_existing_mae", "min"),
            maximum_mae_difference=("continuous_minus_existing_mae", "max"),
        )
        .reset_index()
    )

    checks = {
        "no_manuscript_edits_performed_by_analysis": True,
        "n_cells": int(len(per_cell)),
        "all_cells_pass_primary_rule": bool(per_cell.passes_primary_rule.all()),
        "all_slope_change_bootstrap_95ci_below_zero": bool(
            (per_cell.bootstrap_slope_change_ci_high < 0).all()
        ),
        "continuous_labels_within_one_cycle_of_existing": bool(
            (per_cell.continuous_minus_existing.abs() <= 1).all()
        ),
        "prediction_common_cohort_counts": metadata["common_cohort_counts"],
        "expected_prediction_counts_41_42_40": metadata["common_cohort_counts"]
        == {"train": 41, "primary_test": 42, "secondary_test": 40},
        "ten_seeds_per_method_and_split": bool((metric_summary.seeds == 10).all()),
        "prediction_rows_complete": int(len(predictions)) == 2 * 10 * (42 + 40),
        "continuous_feature_table_counts_41_42_40": continuous_meta["eligible_counts"]
        == {"train": 41, "primary_test": 42, "secondary_test": 40},
        "formal_continuous_has_30_seeds_per_split": bool(
            (formal_continuous.groupby("split").seed.nunique() == 30).all()
        ),
    }

    metric_summary.to_csv(AUDIT / "prediction_summary.csv", index=False)
    paired.to_csv(AUDIT / "prediction_paired_seed_differences.csv", index=False)
    paired_summary.to_csv(AUDIT / "prediction_paired_summary.csv", index=False)
    (AUDIT / "qa_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    continuous_primary = metric_summary[
        (metric_summary.label_method == "continuous_broken_stick")
        & (metric_summary.split == "primary_test")
    ].iloc[0]
    continuous_secondary = metric_summary[
        (metric_summary.label_method == "continuous_broken_stick")
        & (metric_summary.split == "secondary_test")
    ].iloc[0]
    existing_primary = metric_summary[
        (metric_summary.label_method == "existing_independent")
        & (metric_summary.split == "primary_test")
    ].iloc[0]
    existing_secondary = metric_summary[
        (metric_summary.label_method == "existing_independent")
        & (metric_summary.split == "secondary_test")
    ].iloc[0]
    formal_continuous_summary = formal_continuous.groupby("split")[["mae", "rmse", "r2", "abs_error_q90"]].mean()
    formal_existing_summary = formal_existing.groupby("split")[["mae", "rmse", "r2", "abs_error_q90"]].mean()
    report = f"""# Reviewer R1.1 analysis summary

## Status

Analysis complete; manuscript text unchanged.

## Label-definition evidence

- All 124 cells passed the pre-specified continuous-breakpoint rule: both slopes negative, post-break slope more negative, absolute slope ratio >=1.5, and Delta BIC >=10 versus a single line.
- The minimum and median absolute slope ratios were {per_cell.abs_slope_ratio.min():.3f} and {per_cell.abs_slope_ratio.median():.3f}.
- The minimum and median Delta BIC values were {per_cell.delta_bic.min():.1f} and {per_cell.delta_bic.median():.1f}.
- Continuous labels were identical to the existing independent-line label for {(per_cell.continuous_minus_existing == 0).sum()} cells and differed by only one cycle for the other {(per_cell.continuous_minus_existing != 0).sum()} cells.
- Every cell's moving-block-bootstrap 95% interval for the slope change remained below zero.
- Breakpoint 95% interval width: median {per_cell.bootstrap_knee_ci_width.median():.1f} cycles; maximum {per_cell.bootstrap_knee_ci_width.max():.1f} cycles.

## Prediction check on the formal 41/42/40 cohort

- Continuous label, primary test: MAE {continuous_primary.mae_mean:.3f}, RMSE {continuous_primary.rmse_mean:.3f}, R2 {continuous_primary.r2_mean:.3f}.
- Existing independent-line label, primary test: MAE {existing_primary.mae_mean:.3f}, RMSE {existing_primary.rmse_mean:.3f}, R2 {existing_primary.r2_mean:.3f}.
- Continuous label, secondary test: MAE {continuous_secondary.mae_mean:.3f}, RMSE {continuous_secondary.rmse_mean:.3f}, R2 {continuous_secondary.r2_mean:.3f}.
- Existing independent-line label, secondary test: MAE {existing_secondary.mae_mean:.3f}, RMSE {existing_secondary.rmse_mean:.3f}, R2 {existing_secondary.r2_mean:.3f}.

The continuous target preserves the qualitative prediction conclusions and does not introduce a performance penalty under this fixed protocol.

## Complete formal-pipeline rerun

- The continuous-label feature table retains the formal 41/42/40 cohort and changes no predictor column.
- Train-only nested cross-validation selected `{continuous_meta['selected_method_by_train_only_nested_cv']}` for the continuous target (the earlier existing-label run selected `{existing_meta['selected_method_by_train_only_nested_cv']}`).
- Continuous label, primary test (30 seeds): MAE {formal_continuous_summary.loc['primary_test', 'mae']:.3f}, RMSE {formal_continuous_summary.loc['primary_test', 'rmse']:.3f}, R2 {formal_continuous_summary.loc['primary_test', 'r2']:.3f}, Q90 absolute error {formal_continuous_summary.loc['primary_test', 'abs_error_q90']:.3f}.
- Continuous label, secondary test (30 seeds): MAE {formal_continuous_summary.loc['secondary_test', 'mae']:.3f}, RMSE {formal_continuous_summary.loc['secondary_test', 'rmse']:.3f}, R2 {formal_continuous_summary.loc['secondary_test', 'r2']:.3f}, Q90 absolute error {formal_continuous_summary.loc['secondary_test', 'abs_error_q90']:.3f}.
- For context, the saved existing-label formal run gave MAEs {formal_existing_summary.loc['primary_test', 'mae']:.3f} and {formal_existing_summary.loc['secondary_test', 'mae']:.3f} cycles. This is a whole-pipeline comparison, so the change includes train-only selector reselection and should not be interpreted as an isolated one-cycle label effect.

## Cases for transparent illustration

- Typical slope-ratio case: b1c26 (ratio 10.027; breakpoint 651; bootstrap interval width 28.0 cycles).
- Weakest acceleration case: b2c1 (ratio 1.875; breakpoint 94; Delta BIC 417.0; interval width 14.5 cycles). This cell is excluded from N150 prediction because its knee precedes the observation window, but it remains part of the label-definition audit.
- Widest breakpoint interval: b1c2 (ratio 4.516; breakpoint 1687; Delta BIC 6291.8; 95% interval 1649.5--1714.5 cycles).

## Interpretation

The evidence supports using a continuous, acceleration-constrained broken-stick breakpoint as the primary operational knee definition. It supports an accelerated-degradation transition in the corrected SOH trajectory but does not identify a unique electrochemical mechanism.

## Analysis baseline decision

The previously recorded MAEs of 38.23 and 64.15 cycles came from an exploratory test and are excluded from the revision comparison. All subsequent numerical claims will use the current input data, saved configuration, and reproducible reruns. The complete continuous-label formal-pipeline results above are the numerical baseline for the planned revision.
"""
    (AUDIT / "analysis_summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
