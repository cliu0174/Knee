"""Controlled, matched XGBoost baseline for the Table-5 regressor comparison.

The run uses the already frozen 70-input LDQ representation, 41-cell training
cohort, and the project 42/40 held-out cohorts.  Model selection is confined to
the training cohort through nested CV; held-out labels are opened only after the
final XGBoost configuration is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from xgboost import XGBRegressor


EXPECTED_COUNTS = {"train": 41, "primary_test": 42, "secondary_test": 40}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.abs(y_true - y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "abs_error_q90": float(np.quantile(error, 0.90)),
    }


def parameter_grid() -> list[dict[str, float]]:
    """A compact prespecified grid appropriate for 41 development cells."""
    return [
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.90, "colsample_bytree": 0.80, "min_child_weight": 1.0, "reg_lambda": 4.0},
        {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.90, "colsample_bytree": 0.80, "min_child_weight": 1.0, "reg_lambda": 8.0},
        {"n_estimators": 250, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 1.0, "reg_lambda": 4.0},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.80, "min_child_weight": 2.0, "reg_lambda": 8.0},
        {"n_estimators": 350, "max_depth": 2, "learning_rate": 0.02, "subsample": 0.90, "colsample_bytree": 0.90, "min_child_weight": 1.0, "reg_lambda": 8.0},
        {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80, "min_child_weight": 2.0, "reg_lambda": 12.0},
    ]


def make_model(params: dict[str, float], seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=float(params["min_child_weight"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


def numeric_matrix(frame: pd.DataFrame, features: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median(axis=0, numeric_only=True)
    return x.fillna(medians), medians


def inner_select(x: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[dict[str, float], pd.DataFrame]:
    splitter = RepeatedKFold(n_splits=3, n_repeats=3, random_state=seed)
    rows: list[dict[str, object]] = []
    for candidate_id, params in enumerate(parameter_grid()):
        scores: list[float] = []
        for fold, (tr, va) in enumerate(splitter.split(x)):
            model = make_model(params, seed + fold)
            model.fit(x.iloc[tr], y[tr])
            scores.append(float(mean_absolute_error(y[va], model.predict(x.iloc[va]))))
        rows.append({
            "candidate_id": candidate_id,
            "inner_cv_mae_mean": float(np.mean(scores)),
            "inner_cv_mae_std": float(np.std(scores, ddof=1)),
            "params": json.dumps(params),
        })
    ranking = pd.DataFrame(rows).sort_values(["inner_cv_mae_mean", "inner_cv_mae_std"]).reset_index(drop=True)
    return parameter_grid()[int(ranking.iloc[0]["candidate_id"])], ranking


def nested_train_cv(train: pd.DataFrame, features: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_all = train["knee_cycle"].to_numpy(float)
    outer = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)
    fold_rows: list[dict[str, object]] = []
    inner_rows: list[pd.DataFrame] = []
    for fold, (tr, va) in enumerate(outer.split(train)):
        tr_frame = train.iloc[tr].reset_index(drop=True)
        va_frame = train.iloc[va].reset_index(drop=True)
        x_tr, medians = numeric_matrix(tr_frame, features)
        x_va, _ = numeric_matrix(va_frame, features, medians)
        params, ranking = inner_select(x_tr, y_all[tr], seed + 10_000 * fold)
        ranking.insert(0, "outer_fold", fold)
        inner_rows.append(ranking)
        model = make_model(params, seed + 1_000_000 + fold)
        model.fit(x_tr, y_all[tr])
        pred = np.asarray(model.predict(x_va), float)
        fold_rows.append({"outer_fold": fold, "n_train": len(tr), "n_valid": len(va), **regression_metrics(y_all[va], pred), "selected_params": json.dumps(params)})
    return pd.DataFrame(fold_rows), pd.concat(inner_rows, ignore_index=True)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(Path(args.frozen_config).read_text(encoding="utf-8"))
    features = list(frozen["base_features"]) + list(frozen["selected_local_features"])
    if len(features) != 70 or len(set(features)) != 70:
        raise RuntimeError(f"Expected 70 unique frozen features; got {len(features)} total and {len(set(features))} unique.")
    table = pd.read_csv(args.feature_path)
    table = table[table["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    counts = table["official_split"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Eligible split counts are {counts}; expected {EXPECTED_COUNTS}.")
    missing = [feature for feature in features if feature not in table.columns]
    if missing:
        raise RuntimeError(f"Frozen features missing from input: {missing[:5]}")

    train = table[table["official_split"] == "train"].copy().reset_index(drop=True)
    y_train = train["knee_cycle"].to_numpy(float)
    cv_folds, nested_inner = nested_train_cv(train, features, args.random_state)
    x_train, medians = numeric_matrix(train, features)
    final_params, final_ranking = inner_select(x_train, y_train, args.random_state + 9_999_999)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split in ["primary_test", "secondary_test"]:
        test = table[table["official_split"] == split].copy().reset_index(drop=True)
        x_test, _ = numeric_matrix(test, features, medians)
        observed = test["knee_cycle"].to_numpy(float)
        for model_seed in range(args.seed_start, args.seed_start + args.seed_repeats):
            model = make_model(final_params, model_seed)
            model.fit(x_train, y_train)
            predicted = np.asarray(model.predict(x_test), float)
            metric_rows.append({"split": split, "seed": model_seed, "n": len(test), **regression_metrics(observed, predicted)})
            prediction_rows.extend({
                "split": split, "seed": model_seed, "cell_id": row.cell_id, "batch": row.batch,
                "observed": float(y), "predicted": float(pred), "abs_error": float(abs(y - pred)),
            } for row, y, pred in zip(test.itertuples(index=False), observed, predicted, strict=True))

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    per_cell = predictions.groupby(["split", "cell_id", "batch", "observed"], as_index=False).agg(
        predicted_mean=("predicted", "mean"), abs_error_mean=("abs_error", "mean"),
    )
    heldout = metrics.groupby("split")[["mae", "rmse", "r2", "abs_error_q90"]].agg(["mean", "std"]).reset_index()
    heldout.columns = ["_".join(c).rstrip("_") for c in heldout.columns]
    row = {
        "regressor": "XGBoost",
        "train_cv_mae": float(cv_folds["mae"].mean()),
        "primary_mae": float(metrics.loc[metrics["split"] == "primary_test", "mae"].mean()),
        "primary_r2": float(metrics.loc[metrics["split"] == "primary_test", "r2"].mean()),
        "primary_q90": float(metrics.loc[metrics["split"] == "primary_test", "abs_error_q90"].mean()),
        "secondary_mae": float(metrics.loc[metrics["split"] == "secondary_test", "mae"].mean()),
        "secondary_r2": float(metrics.loc[metrics["split"] == "secondary_test", "r2"].mean()),
        "secondary_q90": float(metrics.loc[metrics["split"] == "secondary_test", "abs_error_q90"].mean()),
    }

    cv_folds.to_csv(output / "train_nested_cv_folds.csv", index=False)
    nested_inner.to_csv(output / "nested_inner_cv_rankings.csv", index=False)
    final_ranking.to_csv(output / "freeze_inner_cv_ranking.csv", index=False)
    metrics.to_csv(output / "heldout_metrics_by_seed.csv", index=False)
    predictions.to_csv(output / "heldout_predictions_by_seed.csv", index=False)
    per_cell.to_csv(output / "heldout_per_cell.csv", index=False)
    heldout.to_csv(output / "heldout_summary.csv", index=False)
    pd.DataFrame([row]).to_csv(output / "table5_xgboost_row.csv", index=False)
    (output / "frozen_config.json").write_text(json.dumps({
        "experiment": "matched_xgboost_full70",
        "representation": "the same frozen 70-input LDQ representation used for the formal final model",
        "feature_count": 70,
        "feature_source": str(args.frozen_config),
        "training_protocol": "41-cell nested development; 5x2 outer CV with 3x3 inner CV for Train-CV metric",
        "test_protocol": "42-cell primary and 40-cell secondary evaluations after final configuration freeze",
        "seed_list": list(range(args.seed_start, args.seed_start + args.seed_repeats)),
        "final_params": final_params,
        "heldout_labels_used_for_selection_or_tuning": False,
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame([row]).round(3).to_string(index=False))
    print("\nSelected final XGBoost configuration:")
    print(json.dumps(final_params, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", default="data/early_features_N150_local_deltaq_local9_rerun_20260730.csv")
    parser.add_argument("--frozen-config", default="results/formal_414240_selector_N150_local9_table3_verified_20260730_complete/frozen_config.json")
    parser.add_argument("--output-dir", default="results/formal_414240_matched_xgboost_full70_20260731")
    parser.add_argument("--random-state", type=int, default=20260731)
    parser.add_argument("--seed-start", type=int, default=22660721)
    parser.add_argument("--seed-repeats", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
