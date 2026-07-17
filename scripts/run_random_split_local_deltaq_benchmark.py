from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    raise RuntimeError("xgboost is required for this script") from exc

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


def select_features(df: pd.DataFrame, prefix: str, include_local_deltaq: bool) -> list[str]:
    features: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col == f"{prefix}_window_complete":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        if feature_bucket(col, prefix) in {"raw", "deltaq"}:
            features.append(col)
            continue
        if include_local_deltaq and col.startswith(f"{prefix}_ldq_"):
            features.append(col)
    return features


def make_model(params: dict[str, object], random_state: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(float(params["n_estimators"])),
        max_depth=int(float(params["max_depth"])),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        min_child_weight=float(params["min_child_weight"]),
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=4,
    )


def fit_predict(params: dict[str, object], x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, random_state: int) -> np.ndarray:
    model = make_model(params, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def stratify_key(df: pd.DataFrame) -> np.ndarray:
    life_bin = pd.qcut(df["knee_cycle"], q=4, labels=False, duplicates="drop")
    return df["batch"].astype(str).to_numpy() + "_q" + life_bin.astype(str).to_numpy()


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.feature_path)
    df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    prefix = detect_window_prefix(df)
    features = select_features(df, prefix, args.include_local_deltaq)
    x = df[features]
    y = df["knee_cycle"].to_numpy(dtype=float)

    params_df = pd.read_csv(args.params_path)
    params_row = params_df[params_df["candidate_id"] == args.candidate_id]
    if params_row.empty:
        raise ValueError(f"candidate not found: {args.candidate_id}")
    params = {k: params_row.iloc[0][k] for k in params_df.columns if k != "candidate_id"}

    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    all_idx = np.arange(len(df))
    stratify = stratify_key(df) if args.stratify else None

    for repeat in range(args.repeats):
        seed = args.random_state + repeat
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=args.test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify if args.stratify else None,
        )
        pred = fit_predict(params, x.iloc[train_idx], y[train_idx], x.iloc[test_idx], seed)
        metric = regression_metrics(y[test_idx], pred)
        rows.append(
            {
                "repeat": repeat,
                "seed": seed,
                "candidate_id": args.candidate_id,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                **metric,
            }
        )
        for split_name, indices in [("train", train_idx), ("test", test_idx)]:
            batch_counts = df.iloc[indices]["batch"].value_counts().to_dict()
            split_rows.append(
                {
                    "repeat": repeat,
                    "split": split_name,
                    "n": int(len(indices)),
                    **{f"n_{batch}": int(batch_counts.get(batch, 0)) for batch in sorted(df["batch"].unique())},
                }
            )
        for row_pos, pred_value in zip(test_idx, pred):
            prediction_rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "cell_id": df.loc[row_pos, "cell_id"],
                    "batch": df.loc[row_pos, "batch"],
                    "official_split": df.loc[row_pos, "official_split"],
                    "observed_knee_cycle": float(y[row_pos]),
                    "predicted_knee_cycle": float(pred_value),
                    "abs_error": float(abs(y[row_pos] - pred_value)),
                }
            )

    metrics = pd.DataFrame(rows)
    summary = metrics[["mae", "rmse", "mape_percent", "r2"]].agg(["mean", "std", "min", "median", "max"]).reset_index()
    metrics.to_csv(output_dir / "random_split_metrics.csv", index=False)
    summary.to_csv(output_dir / "random_split_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "random_split_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "random_split_batch_counts.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(output_dir / "features.csv", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "feature_path": args.feature_path,
                "params_path": args.params_path,
                "candidate_id": args.candidate_id,
                "test_size": args.test_size,
                "repeats": args.repeats,
                "stratify": args.stratify,
                "note": "Random split mixes batches and is a same-distribution benchmark, not an external batch validation protocol.",
            },
            fp,
            indent=2,
        )

    print("Saved outputs to", output_dir)
    print("Random split summary:")
    print(summary.round(4).to_string(index=False))
    print("\nBest repeats:")
    print(metrics.sort_values("r2", ascending=False).head(10).round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", default="data/early_features_N150_local_deltaq_top40_20170630.csv")
    parser.add_argument("--params-path", default="results/hybrid_hpo_local_deltaq_top40_N150/xgb_params.csv")
    parser.add_argument("--candidate-id", default="xgb_hpo_0160")
    parser.add_argument("--output-dir", default="results/random_split_local_deltaq_N150")
    parser.add_argument("--random-state", type=int, default=20260709)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-local-deltaq", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
