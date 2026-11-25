"""Export reproducible tables for the continuous-knee manuscript revision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "reviewer_knee_definition_audit"
SUPPLEMENT = ROOT / "manuscript_ionics" / "supplementary_tables"


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = observed - predicted
    absolute_error = np.abs(error)
    rss = float(np.square(error).sum())
    tss = float(np.square(observed - observed.mean()).sum())
    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - rss / tss) if tss > 0 else float("nan"),
        "q90_ae": float(np.quantile(absolute_error, 0.90)),
    }


def bootstrap_summary(per_cell: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for split, group in per_cell.groupby("split", sort=True):
        observed = group.observed.to_numpy(float)
        predicted = group.predicted_mean.to_numpy(float)
        point = regression_metrics(observed, predicted)
        indices = rng.integers(0, len(group), size=(10000, len(group)))
        draws = []
        for index in indices:
            draws.append(regression_metrics(observed[index], predicted[index]))
        draw_frame = pd.DataFrame(draws)
        row: dict[str, float | int | str] = {"split": split, "n": len(group)}
        for metric, value in point.items():
            row[metric] = value
            row[f"{metric}_ci_low"] = float(draw_frame[metric].quantile(0.025))
            row[f"{metric}_ci_high"] = float(draw_frame[metric].quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def export_s1_rows(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "cell_id", "official_split", "continuous_constrained_cycle", "pre_slope",
        "post_slope", "abs_slope_ratio", "delta_bic", "bootstrap_knee_ci_low",
        "bootstrap_knee_ci_high", "bootstrap_acceleration_support",
    ]
    lines = []
    for row in frame.sort_values(["official_split", "cell_id"])[columns].itertuples(index=False):
        lines.append(
            f"{latex_escape(row.cell_id)} & {latex_escape(row.official_split)} & "
            f"{row.continuous_constrained_cycle:.0f} & {row.pre_slope:.6f} & "
            f"{row.post_slope:.6f} & {row.abs_slope_ratio:.3f} & {row.delta_bic:.1f} & "
            f"{row.bootstrap_knee_ci_low:.1f}--{row.bootstrap_knee_ci_high:.1f} & "
            f"{row.bootstrap_acceleration_support:.2f} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT / "continuous_knee_per_cell.csv")
    sensitivity = pd.read_csv(AUDIT / "slope_ratio_threshold_sensitivity.csv")
    agreement = pd.read_csv(AUDIT / "label_method_agreement.csv")
    formal = pd.read_csv(AUDIT / "formal_selector_continuous" / "heldout_per_cell.csv")

    s1_columns = [
        "cell_id", "batch", "official_split", "cycle_life", "continuous_constrained_cycle",
        "pre_slope", "post_slope", "slope_change", "abs_slope_ratio", "delta_bic",
        "bootstrap_knee_ci_low", "bootstrap_knee_ci_high", "bootstrap_knee_ci_width",
        "bootstrap_acceleration_support", "bootstrap_ratio_support",
    ]
    s1 = audit[s1_columns].rename(columns={
        "continuous_constrained_cycle": "continuous_knee_cycle",
        "bootstrap_knee_ci_low": "breakpoint_ci95_low",
        "bootstrap_knee_ci_high": "breakpoint_ci95_high",
        "bootstrap_knee_ci_width": "breakpoint_ci95_width",
    })
    s1.to_csv(SUPPLEMENT / "Table_S1_continuous_knee_audit.csv", index=False)
    sensitivity.to_csv(SUPPLEMENT / "Table_S2_slope_ratio_threshold_sensitivity.csv", index=False)
    agreement.to_csv(SUPPLEMENT / "Table_S3_label_definition_agreement.csv", index=False)
    export_s1_rows(audit, SUPPLEMENT / "table_s1_continuous_knee_rows.tex")

    formal_summary = bootstrap_summary(formal, 20260921)
    formal_summary.to_csv(AUDIT / "formal_continuous_manuscript_summary.csv", index=False)
    formal_summary.to_csv(SUPPLEMENT / "Table_S4_formal_continuous_performance.csv", index=False)

    labels = audit[["cell_id", "batch", "official_split", "cycle_life", "continuous_constrained_cycle"]].rename(
        columns={"continuous_constrained_cycle": "knee_cycle"}
    )
    labels.insert(0, "method", "continuous_accelerated_broken_stick")
    labels.to_csv(ROOT / "data" / "knee_labels_continuous_accelerated_20170630.csv", index=False)
    s1.to_csv(ROOT / "data" / "knee_label_evidence_continuous_accelerated_20170630.csv", index=False)

    metadata = {
        "n_cells": int(len(audit)),
        "primary_rule_passes": int(audit.passes_primary_rule.sum()),
        "s1_columns": s1.columns.tolist(),
        "formal_summary_source": str(AUDIT / "formal_selector_continuous" / "heldout_per_cell.csv"),
        "bootstrap_replicates_for_manuscript_metrics": 10000,
        "revised_label_file": "data/knee_labels_continuous_accelerated_20170630.csv",
    }
    (SUPPLEMENT / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(formal_summary.round(3).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
