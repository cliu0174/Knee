from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_feature_ablation import feature_bucket  # noqa: E402
from train_knee_baselines import (  # noqa: E402
    META_COLUMNS,
    build_models,
    detect_window_prefix,
    regression_metrics,
)


OPTIONAL_MODULES = ["deltaq", "charge", "ir", "temp"]


def all_feature_groups() -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for size in range(len(OPTIONAL_MODULES) + 1):
        for combo in itertools.combinations(OPTIONAL_MODULES, size):
            buckets = ["raw", *combo]
            name = "raw_only" if not combo else "raw_" + "_".join(combo)
            groups.append((name, buckets))
    return groups


def select_features(df: pd.DataFrame, prefix: str, buckets: list[str]) -> list[str]:
    bucket_set = set(buckets)
    features: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col == f"{prefix}_window_complete":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        if feature_bucket(col, prefix) in bucket_set:
            features.append(col)
    return features


def load_window_table(path: Path, min_label_confidence: float) -> tuple[pd.DataFrame, str, int]:
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["knee_after_window"].astype(bool)].copy()
    if min_label_confidence > 0:
        df = df[df["label_confidence"] >= min_label_confidence].copy()
    df = df.reset_index(drop=True)
    prefix = detect_window_prefix(df)
    return df, prefix, before


def run_search_for_window(
    feature_path: Path,
    random_state: int,
    min_label_confidence: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    df, prefix, before = load_window_table(feature_path, min_label_confidence)
    y = df["knee_cycle"].to_numpy(dtype=float)
    train_mask = df["official_split"] == "train"
    primary_mask = df["official_split"] == "primary_test"
    secondary_mask = df["official_split"] == "secondary_test"

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for group_name, buckets in all_feature_groups():
        features = select_features(df, prefix, buckets)
        if not features:
            raise ValueError(f"No features selected for {prefix} {group_name}")
        if df[features].isna().any().any():
            missing = df[features].isna().sum()
            raise ValueError(f"Feature matrix contains missing values:\n{missing[missing > 0]}")

        x = df[features]
        metadata_rows.append(
            {
                "window": prefix,
                "input_file": str(feature_path),
                "input_rows": before,
                "used_rows": len(df),
                "excluded_rows": before - len(df),
                "feature_group": group_name,
                "buckets": buckets,
                "feature_count": len(features),
                "features": features,
            }
        )

        for model_name, model in build_models(random_state).items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x.loc[train_mask], y[train_mask.to_numpy()])
            pred = model.predict(x)

            for split_name, mask in [
                ("train", train_mask),
                ("primary_test", primary_mask),
                ("secondary_test", secondary_mask),
            ]:
                idx = mask.to_numpy()
                row: dict[str, object] = {
                    "window": prefix,
                    "feature_group": group_name,
                    "model": model_name,
                    "split": split_name,
                    "n": int(idx.sum()),
                    "feature_count": len(features),
                }
                row.update(regression_metrics("knee_cycle", y[idx], pred[idx]))
                metric_rows.append(row)

            for row_index, row in df.iterrows():
                prediction_rows.append(
                    {
                        "window": prefix,
                        "feature_group": group_name,
                        "model": model_name,
                        "cell_id": row["cell_id"],
                        "split": row["official_split"],
                        "cycle_life": float(row["cycle_life"]),
                        "observed_knee_cycle": float(row["knee_cycle"]),
                        "predicted_knee_cycle": float(pred[row_index]),
                        "absolute_error": float(abs(float(row["knee_cycle"]) - pred[row_index])),
                        "label_confidence": float(row["label_confidence"]),
                    }
                )

    return metric_rows, prediction_rows, metadata_rows


def build_robust_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    eval_df = metrics[metrics["split"].isin(["primary_test", "secondary_test"])].copy()
    pivot = eval_df.pivot_table(
        index=["window", "feature_group", "model", "feature_count"],
        columns="split",
        values=["knee_cycle_mae", "knee_cycle_r2"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{split}" for metric, split in pivot.columns]
    ranking = pivot.reset_index()
    ranking["mean_test_mae"] = ranking[["knee_cycle_mae_primary_test", "knee_cycle_mae_secondary_test"]].mean(axis=1)
    ranking["max_test_mae"] = ranking[["knee_cycle_mae_primary_test", "knee_cycle_mae_secondary_test"]].max(axis=1)
    ranking["min_test_r2"] = ranking[["knee_cycle_r2_primary_test", "knee_cycle_r2_secondary_test"]].min(axis=1)
    ranking["stable_score"] = ranking["mean_test_mae"] + 0.25 * ranking["max_test_mae"]
    return ranking.sort_values(["stable_score", "max_test_mae", "mean_test_mae"]).reset_index(drop=True)


def plot_top_models(ranking: pd.DataFrame, output_dir: Path, top_n: int = 16) -> None:
    top = ranking.head(top_n).copy()
    labels = top["window"] + " | " + top["model"] + " | " + top["feature_group"]
    x = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(14, 6.8))
    ax.bar(x - 0.18, top["knee_cycle_mae_primary_test"], width=0.36, label="primary", color="#4C78A8")
    ax.bar(x + 0.18, top["knee_cycle_mae_secondary_test"], width=0.36, label="secondary", color="#F58518")
    ax.set_ylabel("MAE (cycles)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "top_combination_models_mae.png", dpi=220)
    plt.close(fig)


def plot_module_heatmap(metrics: pd.DataFrame, output_dir: Path) -> None:
    eval_df = metrics[
        metrics["split"].isin(["primary_test", "secondary_test"])
        & metrics["model"].isin(["random_forest", "xgboost"])
    ].copy()
    eval_df["panel"] = eval_df["window"] + " " + eval_df["model"] + " " + eval_df["split"]
    groups = [name for name, _ in all_feature_groups()]
    panels = eval_df["panel"].drop_duplicates().tolist()
    matrix = []
    for panel in panels:
        part = eval_df[eval_df["panel"] == panel]
        values = []
        for group in groups:
            hit = part[part["feature_group"] == group]
            values.append(float(hit["knee_cycle_mae"].iloc[0]) if not hit.empty else np.nan)
        matrix.append(values)

    fig, ax = plt.subplots(figsize=(15, 0.48 * len(panels) + 4.0))
    image = ax.imshow(np.array(matrix), aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(panels)))
    ax.set_yticklabels(panels)
    ax.set_title("MAE across feature-module combinations")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("MAE (cycles)")
    fig.tight_layout()
    fig.savefig(output_dir / "combination_search_heatmap.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        nargs="+",
        default=[
            "data/early_features_N100_extended_20170630.csv",
            "data/early_features_N150_extended_20170630.csv",
        ],
    )
    parser.add_argument("--output-dir", default="results/feature_combination_search_N100_N150")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-label-confidence", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    all_metadata: list[dict[str, object]] = []

    for feature_file in args.features:
        metrics, predictions, metadata = run_search_for_window(
            Path(feature_file),
            args.random_state,
            args.min_label_confidence,
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        all_metadata.extend(metadata)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(all_predictions)
    ranking = build_robust_ranking(metrics_df)
    split_best = (
        metrics_df[metrics_df["split"].isin(["primary_test", "secondary_test"])]
        .sort_values(["window", "split", "knee_cycle_mae"])
        .groupby(["window", "split"], as_index=False)
        .first()
    )

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    ranking.to_csv(output_dir / "robust_ranking.csv", index=False)
    split_best.to_csv(output_dir / "best_by_window_split.csv", index=False)
    with open(output_dir / "feature_group_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(all_metadata, fp, indent=2, ensure_ascii=False)

    plot_top_models(ranking, output_dir)
    plot_module_heatmap(metrics_df, output_dir)

    print(f"Saved outputs to {output_dir}")
    print("Top robust combinations:")
    print(
        ranking[
            [
                "window",
                "feature_group",
                "model",
                "feature_count",
                "knee_cycle_mae_primary_test",
                "knee_cycle_mae_secondary_test",
                "mean_test_mae",
                "max_test_mae",
                "min_test_r2",
            ]
        ]
        .head(12)
        .round(3)
        .to_string(index=False)
    )
    print("\nBest by window/split:")
    print(split_best[["window", "split", "feature_group", "model", "knee_cycle_mae", "knee_cycle_r2"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
