from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_feature_ablation import feature_bucket  # noqa: E402
from train_knee_baselines import META_COLUMNS, detect_window_prefix  # noqa: E402


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape_percent": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
    }


def select_features(df: pd.DataFrame, prefix: str) -> list[str]:
    features: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col == f"{prefix}_window_complete":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        if feature_bucket(col, prefix) in {"raw", "deltaq"} or col.startswith(f"{prefix}_ldq_"):
            features.append(col)
    return features


def make_model(params: dict[str, object], seed: int) -> ExtraTreesRegressor:
    max_depth = params.get("max_depth")
    if max_depth is not None:
        max_depth = int(float(max_depth))
    return ExtraTreesRegressor(
        n_estimators=int(float(params["n_estimators"])),
        max_features=params["max_features"],
        min_samples_leaf=int(float(params["min_samples_leaf"])),
        min_samples_split=int(float(params["min_samples_split"])),
        max_depth=max_depth,
        bootstrap=bool(params["bootstrap"]),
        random_state=seed,
        n_jobs=-1,
    )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_params = json.loads(Path(args.hpo_metadata).read_text(encoding="utf-8"))["selected_params"]
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for topk in args.topk:
        feature_path = Path(args.feature_pattern.format(topk=topk))
        df = pd.read_csv(feature_path)
        df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
        prefix = detect_window_prefix(df)
        features = select_features(df, prefix)
        x = df[features]
        y = df["knee_cycle"].to_numpy(dtype=float)
        dev_mask = df["official_split"].isin(["train", "primary_test"]).to_numpy()
        external_mask = (df["official_split"] == "secondary_test").to_numpy()

        for feature in features:
            feature_rows.append({"topk": topk, "feature": feature})

        for seed in range(args.random_state, args.random_state + args.seed_repeats):
            model = make_model(selected_params, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x.loc[dev_mask], y[dev_mask])
            pred = np.asarray(model.predict(x), dtype=float)
            for split_name, mask in [("development_train_primary", dev_mask), ("external_secondary", external_mask)]:
                rows.append(
                    {
                        "topk": topk,
                        "seed": seed,
                        "split": split_name,
                        "n": int(mask.sum()),
                        **regression_metrics(y[mask], pred[mask]),
                    }
                )
            for row_index, row in df.iterrows():
                prediction_rows.append(
                    {
                        "topk": topk,
                        "seed": seed,
                        "cell_id": row["cell_id"],
                        "batch": row["batch"],
                        "official_split": row["official_split"],
                        "observed_knee_cycle": float(row["knee_cycle"]),
                        "predicted_knee_cycle": float(pred[row_index]),
                        "abs_error": float(abs(float(row["knee_cycle"]) - float(pred[row_index]))),
                    }
                )

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["topk", "split"])[["mae", "rmse", "mape_percent", "r2"]]
        .agg(["mean", "std", "min", "median", "max"])
        .reset_index()
    )
    external = metrics[metrics["split"] == "external_secondary"].copy()
    external_summary = (
        external.groupby("topk")[["mae", "rmse", "mape_percent", "r2"]]
        .agg(["mean", "std", "min", "median", "max"])
        .reset_index()
    )

    metrics.to_csv(output_dir / "ldq_et_topk_metrics_by_seed.csv", index=False)
    summary.to_csv(output_dir / "ldq_et_topk_summary.csv", index=False)
    external_summary.to_csv(output_dir / "ldq_et_topk_external_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "ldq_et_topk_predictions.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(output_dir / "ldq_et_topk_features.csv", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "feature_pattern": args.feature_pattern,
                "topk": args.topk,
                "hpo_metadata": args.hpo_metadata,
                "selected_params": selected_params,
                "seed_repeats": args.seed_repeats,
                "note": "Top-k ablation uses the same LDQ-ET HPO parameters; only the number of selected Local DeltaQ features changes.",
            },
            fp,
            indent=2,
        )

    print("Saved outputs to", output_dir)
    print("External top-k summary:")
    print(external_summary.round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-pattern", default="data/early_features_N150_local_deltaq_top{topk}_20170630.csv")
    parser.add_argument("--hpo-metadata", default="results/extratrees_ldq_hpo_N150_quick30/metadata.json")
    parser.add_argument("--output-dir", default="results/ldq_et_topk_ablation_N150")
    parser.add_argument("--topk", nargs="+", type=int, default=[20, 40, 60, 80])
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--seed-repeats", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
