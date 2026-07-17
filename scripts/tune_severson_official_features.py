from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from reproduce_severson_baseline import LEAKAGE_COLUMNS, evaluate_predictions


def load_xy(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if "official_split" not in df.columns:
        raise ValueError("Expected an official_split column")
    y = df["lifetime"].astype(float)
    feature_cols = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    x = df[feature_cols].astype(float)
    return x, y, df["official_split"]


def candidate_models(random_state: int) -> list[tuple[str, object, dict[str, object]]]:
    models: list[tuple[str, object, dict[str, object]]] = []
    for min_leaf in [1, 2, 4]:
        for max_features in ["sqrt", 0.8, None]:
            params = {"min_samples_leaf": min_leaf, "max_features": max_features}
            models.append(
                (
                    "random_forest_tuned",
                    RandomForestRegressor(
                        n_estimators=800,
                        min_samples_leaf=min_leaf,
                        max_features=max_features,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                    params,
                )
            )

    for depth in [1, 2, 3]:
        for learning_rate in [0.02, 0.05]:
            for reg_lambda in [1.0, 5.0, 10.0]:
                params = {
                    "max_depth": depth,
                    "learning_rate": learning_rate,
                    "reg_lambda": reg_lambda,
                }
                models.append(
                    (
                        "xgboost_tuned",
                        XGBRegressor(
                            n_estimators=350,
                            max_depth=depth,
                            learning_rate=learning_rate,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            reg_lambda=reg_lambda,
                            reg_alpha=0.05,
                            objective="reg:squarederror",
                            random_state=random_state,
                            n_jobs=4,
                        ),
                        params,
                    )
                )

    for c_value in [1.0, 3.0, 10.0, 30.0, 100.0]:
        for epsilon in [0.02, 0.03, 0.05]:
            params = {"C": c_value, "epsilon": epsilon}
            models.append(
                (
                    "svr_rbf_tuned",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            ("model", SVR(kernel="rbf", C=c_value, gamma="scale", epsilon=epsilon)),
                        ]
                    ),
                    params,
                )
            )
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/severson_features/severson_official_features.csv")
    parser.add_argument("--output-dir", default="results/severson_official_features_tuning")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, split = load_xy(Path(args.features))
    y_arr = y.to_numpy()
    y_log = np.log10(y_arr)
    train_idx = np.where(split.to_numpy() == "train")[0]
    test_sets = {
        "official_primary_test": np.where(split.to_numpy() == "primary_test")[0],
        "official_secondary_test": np.where(split.to_numpy() == "secondary_test")[0],
    }
    clip_bounds = (float(y_log[train_idx].min() - 0.1), float(y_log[train_idx].max() + 0.1))

    rows: list[dict[str, object]] = []
    cv = KFold(n_splits=4, shuffle=True, random_state=args.random_state)
    for model_family, model, params in candidate_models(args.random_state):
        cv_mapes: list[float] = []
        train_x = x.iloc[train_idx]
        train_y_log = y_log[train_idx]
        train_y_cycles = y_arr[train_idx]
        for cv_train_rel, cv_test_rel in cv.split(train_x):
            cv_model = clone(model)
            cv_model.fit(train_x.iloc[cv_train_rel], train_y_log[cv_train_rel])
            fold_clip_bounds = (
                float(train_y_log[cv_train_rel].min() - 0.1),
                float(train_y_log[cv_train_rel].max() + 0.1),
            )
            fold_metric = evaluate_predictions(
                "severson_official_features",
                model_family,
                "train_cv4",
                train_y_cycles[cv_test_rel],
                cv_model.predict(train_x.iloc[cv_test_rel]),
                fold_clip_bounds,
            )
            cv_mapes.append(fold_metric.mape_percent)

        model.fit(x.iloc[train_idx], y_log[train_idx])
        family_rows: list[dict[str, object]] = []
        for split_name, test_idx in test_sets.items():
            pred_log = model.predict(x.iloc[test_idx])
            metric = evaluate_predictions(
                "severson_official_features",
                model_family,
                split_name,
                y_arr[test_idx],
                pred_log,
                clip_bounds,
            )
            row = asdict(metric)
            row.update(params)
            row["train_cv4_mape_mean"] = float(np.mean(cv_mapes))
            row["train_cv4_mape_std"] = float(np.std(cv_mapes, ddof=1))
            family_rows.append(row)
        mean_mape = float(np.mean([row["mape_percent"] for row in family_rows]))
        for row in family_rows:
            row["mean_primary_secondary_mape"] = mean_mape
            rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "tuning_metrics.csv", index=False)
    summary = (
        metrics.groupby(["model", "train_cv4_mape_mean", "train_cv4_mape_std"], as_index=False)
        .first()
        .sort_values("train_cv4_mape_mean")
    )
    summary.to_csv(output_dir / "tuning_summary.csv", index=False)
    print(
        metrics.sort_values(["train_cv4_mape_mean", "model", "split"])
        .head(20)
        .to_string(index=False)
    )
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
