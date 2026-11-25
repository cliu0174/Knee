"""Create an analysis-only feature table using continuous knee labels.

The predictor columns are copied byte-for-value from the frozen N150 feature
table. Only target-derived columns are recalculated. The source dataset is not
overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--curves", default="data/processed_cycles_soh_corrected_20170630.csv")
    parser.add_argument("--audit", default="results/reviewer_knee_definition_audit/continuous_knee_per_cell.csv")
    parser.add_argument("--output", default="results/reviewer_knee_definition_audit/early_features_N150_continuous_knee.csv")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    original = features.copy(deep=True)
    curves = pd.read_csv(args.curves)
    labels = pd.read_csv(args.audit)[[
        "cell_id", "continuous_constrained_cycle", "passes_primary_rule"
    ]]
    if not labels.passes_primary_rule.all():
        raise RuntimeError("At least one continuous label fails the primary evidence rule")
    if labels.cell_id.duplicated().any() or features.cell_id.duplicated().any():
        raise RuntimeError("Cell IDs must be unique")

    table = features.merge(labels, on="cell_id", how="left", validate="one_to_one")
    if table.continuous_constrained_cycle.isna().any():
        raise RuntimeError("Missing continuous labels after merge")
    table["knee_cycle"] = table.pop("continuous_constrained_cycle").astype(float)
    table["relative_knee"] = table.knee_cycle / table.cycle_life
    table["knee_after_window"] = table.knee_cycle > 150
    table["remaining_cycles_from_window"] = table.knee_cycle - 150

    curve_lookup = {
        cell_id: part.sort_values("cycle")
        for cell_id, part in curves.groupby("cell_id", sort=False)
    }
    knee_soh = []
    for row in table.itertuples():
        part = curve_lookup[row.cell_id]
        knee_soh.append(float(np.interp(row.knee_cycle, part.cycle, part.soh_mono)))
    table["knee_soh"] = knee_soh
    table = table.drop(columns=["passes_primary_rule"])
    table = table[original.columns]

    target_columns = {
        "knee_cycle", "knee_soh", "relative_knee", "knee_after_window",
        "remaining_cycles_from_window"
    }
    predictor_columns = [c for c in original.columns if c not in target_columns]
    unchanged = all(
        original[c].equals(table[c]) for c in predictor_columns
    )
    if not unchanged:
        raise RuntimeError("A non-target column changed while constructing the table")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    metadata = {
        "source_features": args.features,
        "source_curves": args.curves,
        "source_audit": args.audit,
        "rows": len(table),
        "columns": len(table.columns),
        "updated_columns": sorted(target_columns),
        "label_confidence_policy": "preserved from source; bootstrap uncertainty remains in the audit table",
        "all_non_target_columns_unchanged": unchanged,
        "eligible_counts": table.loc[table.knee_after_window].official_split.value_counts().to_dict(),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
