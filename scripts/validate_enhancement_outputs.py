from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
N150 = ROOT / "results" / "nested_contiguous_ldq_N150_20260715_nonoverlap"
N100 = ROOT / "results" / "nested_contiguous_ldq_N100_commonN150_5x2_20260715_nonoverlap"
N50 = ROOT / "results" / "nested_contiguous_ldq_N50_commonN150_5x2_20260715_nonoverlap"
ABLATION = ROOT / "results" / "nested_ablation_learning_N150_20260715_nonoverlap"
OUT = ROOT / "results" / "enhancement_validation_20260715_nonoverlap"


def check(condition: bool, message: str, rows: list[dict[str, object]]) -> None:
    rows.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main() -> None:
    rows: list[dict[str, object]] = []
    metadata = json.loads((N150 / "metadata.json").read_text(encoding="utf-8"))
    check(metadata["n_development"] == 83, "N150 development cohort contains 83 cells", rows)
    check(metadata["secondary_labels_read_for_selection"] is False, "N150 metadata records no secondary-label selection access", rows)
    folds = pd.read_csv(N150 / "development_outer_fold_metrics.csv")
    check(folds.groupby("method").size().eq(10).all(), "Every N150 selector has 10 outer-fold metrics", rows)
    check(np.isfinite(folds[["mae", "rmse", "r2"]].to_numpy()).all(), "N150 fold metrics are finite", rows)
    oof = pd.read_csv(N150 / "development_oof_predictions.csv")
    counts = oof.groupby(["method", "cell_id"]).size()
    check(counts.eq(2).all(), "Every development cell has two OOF predictions per selector", rows)
    selected = pd.read_csv(N150 / "development_selected_features.csv")
    check(selected.groupby(["method", "outer_id"]).size().isin([20, 40, 60]).all(), "Outer-fold selector feature counts match declared top-k choices", rows)

    frozen = json.loads((N150 / "frozen_config.json").read_text(encoding="utf-8"))
    check(frozen["method"] == metadata["selected_method_by_development"], "Frozen selector equals development-selected selector", rows)
    check(len(frozen["selected_local_features"]) == 40 and len(set(frozen["selected_local_features"])) == 40, "Frozen local set has 40 unique features", rows)
    check(frozen["historical_external_status"].startswith("The repository secondary batch was used"), "Frozen record discloses historical secondary use", rows)

    ext_metrics = pd.read_csv(N150 / "external_metrics_by_seed.csv")
    ext_pred = pd.read_csv(N150 / "external_predictions_by_seed.csv")
    ext_cell = pd.read_csv(N150 / "external_per_cell.csv")
    check(len(ext_metrics) == 30 and ext_metrics.seed.nunique() == 30, "External evaluation contains 30 declared seeds", rows)
    check(len(ext_cell) == 40 and ext_cell.cell_id.nunique() == 40, "External evaluation contains 40 unique cells", rows)
    check(ext_pred.groupby("seed").size().eq(40).all(), "Every seed predicts all 40 external cells", rows)
    check(np.isfinite(ext_metrics.select_dtypes(include=[np.number]).to_numpy()).all(), "External metrics are finite", rows)
    interval = pd.read_csv(N150 / "external_prediction_intervals.csv")
    summary = pd.read_csv(N150 / "external_interval_summary.csv")
    empirical = interval.covered.astype(str).str.lower().isin(["true", "1"]).mean()
    check(abs(empirical - summary.loc[summary.stratum == "all", "empirical_coverage"].iloc[0]) < 1e-12, "Interval coverage summary matches cell-level intervals", rows)

    for name, path in [("N50", N50), ("N100", N100)]:
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        budget_folds = pd.read_csv(path / "development_outer_fold_metrics.csv")
        check(meta["n_development"] == 83, f"{name} uses the common 83-cell cohort", rows)
        check(meta["secondary_labels_read_for_selection"] is False, f"{name} records no secondary-label selection access", rows)
        check(budget_folds.groupby("method").size().eq(10).all(), f"{name} has 10 outer folds per selector", rows)

    ablation_meta = json.loads((ABLATION / "metadata.json").read_text(encoding="utf-8"))
    ablation = pd.read_csv(ABLATION / "fold_metrics.csv")
    check(ablation_meta["secondary_cells_used"] == 0, "Ablations use no secondary cells", rows)
    check(ablation.loc[ablation.analysis == "feature_group", "variant"].nunique() == 7, "Seven feature groups were evaluated", rows)
    check(set(ablation.loc[ablation.analysis == "learning_curve", "fraction"].unique()) == {0.25, 0.5, 0.75, 1.0}, "Learning curve contains 25/50/75/100 percent budgets", rows)
    check({"model_family", "region_control", "region_permutation"}.issubset(set(ablation.analysis.unique())), "Model-family and region controls are present", rows)

    stems = [
        "fig1_protocol_nested_workflow",
        "fig2_strict_nested_selector_comparison",
        "fig3_observation_budget_and_data_efficiency",
        "fig4_local_region_contribution_and_controls",
        "fig5_frozen_secondary_batch_performance",
        "fig6_feature_group_and_model_family_ablation",
        "fig7_uncertainty_tail_and_complexity",
    ]
    figures = [ROOT / "manuscript_ionics" / "figures" / f"{stem}.svg" for stem in stems]
    check(all(path.exists() for path in figures), "Seven enhanced editable SVG figures exist", rows)
    check(all((f.with_suffix(".pdf").exists() and f.with_suffix(".png").exists()) for f in figures), "Every enhanced figure has PDF and PNG exports", rows)
    sources = list((ROOT / "manuscript_ionics" / "figure_source_data").glob("fig[1-7]_*.csv"))
    check(len(sources) >= 14, "Enhanced figures have traceable source-data CSVs", rows)

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "pass",
        "checks": rows,
        "n_checks": len(rows),
        "key_values": {
            "development_selected_method": frozen["method"],
            "external_mae_seed_mean": float(ext_metrics.mae.mean()),
            "external_mae_seed_std": float(ext_metrics.mae.std()),
            "external_interval_coverage": float(empirical),
        },
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Enhancement validation report", "", f"Status: **PASS** ({len(rows)} checks).", ""]
    lines.extend([f"- [{'x' if row['passed'] else ' '}] {row['check']}" for row in rows])
    (OUT / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["key_values"], indent=2))


if __name__ == "__main__":
    main()
