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
from sklearn.model_selection import RepeatedKFold

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


def make_xgb(params: dict[str, object], random_state: int) -> XGBRegressor:
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


def fit_predict(
    params: dict[str, object],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_pred: pd.DataFrame,
    random_state: int,
) -> np.ndarray:
    model = make_xgb(params, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return np.asarray(model.predict(x_pred), dtype=float)


def repeated_cv_score(
    params: dict[str, object],
    x_dev: pd.DataFrame,
    y_dev: np.ndarray,
    random_state: int,
    n_splits: int,
    n_repeats: int,
) -> dict[str, float]:
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    maes: list[float] = []
    rmses: list[float] = []
    for fold_id, (tr, va) in enumerate(splitter.split(x_dev)):
        pred = fit_predict(params, x_dev.iloc[tr], y_dev[tr], x_dev.iloc[va], random_state + fold_id)
        maes.append(float(mean_absolute_error(y_dev[va], pred)))
        rmses.append(float(mean_squared_error(y_dev[va], pred) ** 0.5))
    return {
        "dev_cv_mae_mean": float(np.mean(maes)),
        "dev_cv_mae_std": float(np.std(maes)),
        "dev_cv_mae_max": float(np.max(maes)),
        "dev_cv_rmse_mean": float(np.mean(rmses)),
        "dev_selection_score": float(np.mean(maes) + 0.10 * np.std(maes) + 0.10 * np.max(maes)),
    }


def add_metrics(rows: list[dict[str, object]], method: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rows.append({"method": method, "split": split, "n": int(y_true.size), **regression_metrics(y_true, y_pred)})


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.feature_path)
    df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    prefix = detect_window_prefix(df)
    features = select_features(df, prefix, args.include_local_deltaq)
    x = df[features]
    y = df["knee_cycle"].to_numpy(dtype=float)
    dev_mask = df["official_split"].isin(["train", "primary_test"]).to_numpy()
    external_mask = (df["official_split"] == "secondary_test").to_numpy()
    x_dev = x.loc[dev_mask].reset_index(drop=True)
    y_dev = y[dev_mask]

    params = pd.read_csv(args.params_path)
    if args.max_candidates:
        params = params.head(args.max_candidates).copy()

    selection_rows: list[dict[str, object]] = []
    for _, row in params.iterrows():
        param_dict = {k: row[k] for k in row.index if k != "candidate_id"}
        score = repeated_cv_score(param_dict, x_dev, y_dev, args.random_state, args.n_splits, args.n_repeats)
        selection_rows.append({"candidate_id": row["candidate_id"], **score, **param_dict})
    selection = pd.DataFrame(selection_rows).sort_values(["dev_selection_score", "dev_cv_mae_mean"]).reset_index(drop=True)
    selected = selection.iloc[0].to_dict()
    selected_params = {k: selected[k] for k in params.columns if k != "candidate_id"}
    selected_id = str(selected["candidate_id"])

    all_params = pd.read_csv(args.full_params_path)
    params_by_id = {
        str(row["candidate_id"]): {k: row[k] for k in all_params.columns if k != "candidate_id"}
        for _, row in all_params.iterrows()
    }
    params_by_id[selected_id] = selected_params
    comparison_ids = [selected_id]
    for extra_id in args.extra_candidates.split(","):
        extra_id = extra_id.strip()
        if extra_id and extra_id not in comparison_ids:
            comparison_ids.append(extra_id)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for candidate_id in comparison_ids:
        if candidate_id not in params_by_id:
            continue
        method = "dev_cv_selected_local_deltaq_xgb" if candidate_id == selected_id else f"sensitivity_{candidate_id}_dev_trained"
        pred = fit_predict(params_by_id[candidate_id], x.loc[dev_mask], y[dev_mask], x, args.random_state)
        add_metrics(metric_rows, method, "development_train_primary", y[dev_mask], pred[dev_mask])
        add_metrics(metric_rows, method, "external_secondary", y[external_mask], pred[external_mask])
        for row_index, row in df.iterrows():
            prediction_rows.append(
                {
                    "method": method,
                    "candidate_id": candidate_id,
                    "cell_id": row["cell_id"],
                    "split": row["official_split"],
                    "observed_knee_cycle": float(row["knee_cycle"]),
                    "predicted_knee_cycle": float(pred[row_index]),
                    "abs_error": float(abs(float(row["knee_cycle"]) - float(pred[row_index]))),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    selection.to_csv(output_dir / "development_cv_param_selection.csv", index=False)
    metrics.to_csv(output_dir / "development_external_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "development_external_predictions.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(output_dir / "features.csv", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "feature_path": args.feature_path,
                "params_path": args.params_path,
                "selected_candidate": selected_id,
                "development_split": "official train + primary_test",
                "external_split": "official secondary_test",
                "n_splits": args.n_splits,
                "n_repeats": args.n_repeats,
                "note": "Primary test is treated as development validation in this protocol; only secondary is external validation.",
            },
            fp,
            indent=2,
        )

    print("Saved outputs to", output_dir)
    print("Development CV top:")
    print(selection.head(12).round(3).to_string(index=False))
    print("\nDevelopment/external metrics:")
    print(metrics.round(3).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", default="data/early_features_N150_local_deltaq_top40_20170630.csv")
    parser.add_argument("--params-path", default="results/hybrid_hpo_local_deltaq_top40_N150/xgb_params_train_top60.csv")
    parser.add_argument("--full-params-path", default="results/hybrid_hpo_local_deltaq_top40_N150/xgb_params.csv")
    parser.add_argument("--output-dir", default="results/development_external_finalization_N150")
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--extra-candidates", default="xgb_hpo_0212,xgb_hpo_0052,xgb_hpo_0038")
    parser.add_argument("--include-local-deltaq", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
