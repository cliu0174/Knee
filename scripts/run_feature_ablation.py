from __future__ import annotations

import argparse
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

from train_knee_baselines import (  # noqa: E402
    META_COLUMNS,
    build_models,
    detect_window_prefix,
    regression_metrics,
)


ABLATION_GROUPS = [
    ("raw_only", ["raw"]),
    ("raw_deltaq", ["raw", "deltaq"]),
    ("raw_deltaq_charge", ["raw", "deltaq", "charge"]),
    ("raw_deltaq_charge_ir", ["raw", "deltaq", "charge", "ir"]),
    ("raw_deltaq_charge_ir_temp", ["raw", "deltaq", "charge", "ir", "temp"]),
]


def feature_bucket(column: str, prefix: str) -> str | None:
    if (
        column == f"{prefix}_n_observed"
        or column.startswith(f"{prefix}_qd_")
        or column.startswith(f"{prefix}_soh_raw_")
        or column.startswith(f"{prefix}_raw_regen_")
        or column.startswith(f"{prefix}_raw_max_single_cycle")
    ):
        return "raw"
    if column.startswith(f"{prefix}_delta_q_"):
        return "deltaq"
    if column.startswith(f"{prefix}_chargetime_"):
        return "charge"
    if column.startswith(f"{prefix}_ir_"):
        return "ir"
    if (
        column.startswith(f"{prefix}_tavg_")
        or column.startswith(f"{prefix}_tmin_")
        or column.startswith(f"{prefix}_tmax_")
    ):
        return "temp"
    return None


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
        bucket = feature_bucket(col, prefix)
        if bucket in bucket_set:
            features.append(col)
    return features


def load_window_table(path: Path, min_label_confidence: float) -> tuple[pd.DataFrame, str, int, int]:
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["knee_after_window"].astype(bool)].copy()
    if min_label_confidence > 0:
        df = df[df["label_confidence"] >= min_label_confidence].copy()
    df = df.reset_index(drop=True)
    prefix = detect_window_prefix(df)
    window = int(prefix[1:])
    return df, prefix, window, before


def run_ablation_for_window(
    feature_path: Path,
    output_dir: Path,
    random_state: int,
    min_label_confidence: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    df, prefix, window, before = load_window_table(feature_path, min_label_confidence)
    y = df["knee_cycle"].to_numpy(dtype=float)
    train_mask = df["official_split"] == "train"
    primary_mask = df["official_split"] == "primary_test"
    secondary_mask = df["official_split"] == "secondary_test"

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for group_name, buckets in ABLATION_GROUPS:
        features = select_features(df, prefix, buckets)
        if not features:
            raise ValueError(f"No features selected for {prefix} {group_name}")
        if df[features].isna().any().any():
            missing = df[features].isna().sum()
            raise ValueError(f"Feature matrix contains missing values:\n{missing[missing > 0]}")

        x = df[features]
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

    return metric_rows, prediction_rows, metadata_rows


def plot_ablation(metrics: pd.DataFrame, output_dir: Path) -> None:
    plot_df = metrics[
        metrics["split"].isin(["primary_test", "secondary_test"])
        & metrics["model"].isin(["random_forest", "xgboost"])
    ].copy()
    order = [name for name, _ in ABLATION_GROUPS]
    plot_df["feature_group"] = pd.Categorical(plot_df["feature_group"], categories=order, ordered=True)

    windows = plot_df["window"].drop_duplicates().tolist()
    splits = ["primary_test", "secondary_test"]
    fig, axes = plt.subplots(len(windows), len(splits), figsize=(13.5, 4.8 * len(windows)), squeeze=False)
    colors = {"random_forest": "#4C78A8", "xgboost": "#F58518"}
    x = np.arange(len(order))
    width = 0.36

    for r, window in enumerate(windows):
        for c, split in enumerate(splits):
            ax = axes[r, c]
            subset = plot_df[(plot_df["window"] == window) & (plot_df["split"] == split)]
            for offset, model_name in [(-width / 2, "random_forest"), (width / 2, "xgboost")]:
                part = subset[subset["model"] == model_name].sort_values("feature_group")
                values = [
                    float(part[part["feature_group"] == group]["knee_cycle_mae"].iloc[0])
                    if not part[part["feature_group"] == group].empty
                    else np.nan
                    for group in order
                ]
                ax.bar(x + offset, values, width=width, label=model_name, color=colors[model_name])
            ax.set_title(f"{window} {split}")
            ax.set_ylabel("MAE (cycles)")
            ax.set_xticks(x)
            ax.set_xticklabels(
                ["raw", "+DeltaQ", "+charge", "+IR", "+temp"],
                rotation=25,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "feature_ablation_mae.png", dpi=220)
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
    parser.add_argument("--output-dir", default="results/feature_ablation_N100_N150")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-label-confidence", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    all_metadata: list[dict[str, object]] = []
    for feature_file in args.features:
        metrics, predictions, metadata = run_ablation_for_window(
            Path(feature_file),
            output_dir,
            args.random_state,
            args.min_label_confidence,
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        all_metadata.extend(metadata)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(all_predictions)
    metadata_path = output_dir / "feature_group_metadata.json"

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(all_metadata, fp, indent=2, ensure_ascii=False)

    best = (
        metrics_df[metrics_df["split"].isin(["primary_test", "secondary_test"])]
        .sort_values(["window", "split", "knee_cycle_mae"])
        .groupby(["window", "split"], as_index=False)
        .first()
    )
    best.to_csv(output_dir / "best_by_window_split.csv", index=False)
    plot_ablation(metrics_df, output_dir)

    print(f"Saved metrics: {output_dir / 'metrics.csv'}")
    print(f"Saved predictions: {output_dir / 'predictions.csv'}")
    print(f"Saved metadata: {metadata_path}")
    print(best[["window", "split", "feature_group", "model", "knee_cycle_mae", "knee_cycle_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
