from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META_COLUMNS = {
    "cell_id",
    "batch",
    "official_order",
    "official_split",
    "cycle_life",
    "knee_cycle",
    "knee_soh",
    "relative_knee",
    "label_confidence",
    "knee_after_window",
    "remaining_cycles_from_window",
}


def is_base_or_local(col: str, prefix: str) -> bool:
    if (
        col == f"{prefix}_n_observed"
        or col == f"{prefix}_window_complete"
        or col.startswith(f"{prefix}_qd_")
        or col.startswith(f"{prefix}_soh_raw_")
        or col.startswith(f"{prefix}_raw_regen_")
        or col.startswith(f"{prefix}_raw_max_single_cycle")
        or col.startswith(f"{prefix}_delta_q_")
        or col.startswith(f"{prefix}_ldq_")
    ):
        return True
    return False


def abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return 0.0
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return 0.0
    return float(abs(np.corrcoef(xv, yv)[0, 1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--output", default="data/early_features_N150_local_deltaq_top60_20170630.csv")
    parser.add_argument("--ranking-output", default="data/early_features_N150_local_deltaq_feature_ranking_20170630.csv")
    parser.add_argument("--prefix", default="N150")
    parser.add_argument("--top-k", type=int, default=60)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    train = df["official_split"] == "train"
    y = df.loc[train, "knee_cycle"].to_numpy(dtype=float)

    local_cols = [
        col
        for col in df.columns
        if col.startswith(f"{args.prefix}_ldq_")
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique(dropna=True) > 1
    ]
    ranking = []
    for col in local_cols:
        ranking.append(
            {
                "feature": col,
                "abs_train_corr": abs_corr(df.loc[train, col].to_numpy(dtype=float), y),
                "missing_count": int(df[col].isna().sum()),
                "n_unique": int(df[col].nunique(dropna=True)),
            }
        )
    ranking_df = pd.DataFrame(ranking).sort_values(
        ["abs_train_corr", "missing_count", "feature"], ascending=[False, True, True]
    )
    selected_local = ranking_df.head(args.top_k)["feature"].tolist()

    base_cols = [
        col
        for col in df.columns
        if col in META_COLUMNS
        or (is_base_or_local(col, args.prefix) and not col.startswith(f"{args.prefix}_ldq_"))
    ]
    selected_cols = base_cols + selected_local
    slim = df[selected_cols].copy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(args.output, index=False)
    ranking_df.to_csv(args.ranking_output, index=False)
    print(f"Input columns: {df.shape[1]}")
    print(f"Selected local features: {len(selected_local)}")
    print(f"Output columns: {slim.shape[1]} -> {args.output}")
    print(ranking_df.head(20).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
