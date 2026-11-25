"""Training-only Bai-inspired XGBoost--Random Forest knee-prediction baseline.

This is a transparent, literature-inspired reimplementation rather than a
claim of exact reproduction.  It retains the published method's early-cycle
candidate-feature and XGBoost--RF ensemble ideas, while preserving this
project's 150-cycle information boundary, knee labels, and 41/42/40 split.
The unreported NRBO optimizer and the paper's random 70/30 split are not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor


META = [
    "cell_id", "batch", "official_split", "knee_cycle", "knee_after_window",
]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = np.abs(y_true - y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "abs_error_q90": float(np.quantile(errors, 0.90)),
    }


def value_at(series: pd.DataFrame, cycle: int, field: str) -> float:
    values = series.loc[series["cycle"] == cycle, field].dropna().to_numpy(float)
    return float(values[0]) if values.size else float("nan")


def construct_candidates(extended: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    """Build Bai-style F1--F11 candidates under the existing N=150 boundary."""
    prefix = "N150"
    rows: list[dict[str, object]] = []
    for _, base in extended.iterrows():
        cell_id = str(base["cell_id"])
        part = cycles[(cycles["cell_id"] == cell_id) & (cycles["cycle"] <= 150)].copy()
        part = part[part["is_positive_capacity"].astype(bool)]
        qd = part[(part["cycle"] >= 2) & (part["cycle"] <= 150)][["cycle", "qd"]].dropna()
        if len(qd) >= 2:
            slope, intercept = np.polyfit(qd["cycle"].to_numpy(float), qd["qd"].to_numpy(float), 1)
        else:
            slope, intercept = float("nan"), float("nan")

        # The source paper integrates the difference between the two Q(V)
        # curves.  The project stores the same 1,000-point DeltaQ curve; its
        # mean times the fixed 1.5-V span is that Riemann integral.
        f1 = float(base[f"{prefix}_delta_q_mean"]) * 1.5
        # The original paper's DeltaQ/DeltaV peak is not tabulated directly.
        # The available peak DeltaQ magnitude is the documented proxy used in
        # this adapted implementation.  Selection is training-only.
        f2 = float(base[f"{prefix}_delta_q_max"])
        c10_tmax = value_at(part, 10, "tmax")
        c150_tmax = value_at(part, 150, "tmax")
        c10_tmin = value_at(part, 10, "tmin")
        c150_tmin = value_at(part, 150, "tmin")
        first5 = part[(part["cycle"] >= 1) & (part["cycle"] <= 5)]["chargetime"].dropna()
        rows.append({
            **{key: base[key] for key in META},
            "F1_deltaq_area_150_minus_10": f1,
            "F2_deltaq_peak_proxy": f2,
            "F3_capacity_slope_2_150": float(slope),
            "F4_capacity_intercept_2_150": float(intercept),
            "F5_capacity_cycle_2": value_at(part, 2, "qd"),
            "F6_ir_min_1_150": float(base[f"{prefix}_ir_min"]),
            "F7_ir_max_1_150": float(base[f"{prefix}_ir_max"]),
            "F8_tmax_change_150_minus_10": c150_tmax - c10_tmax,
            "F9_tmin_cycle_10": c10_tmin,
            "F10_tmin_cycle_150": c150_tmin,
            "F11_chargetime_mean_first5": float(first5.mean()) if len(first5) else float("nan"),
        })
    return pd.DataFrame(rows)


def median_fill(train: pd.DataFrame, target: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    medians = train[columns].median(numeric_only=True)
    return target[columns].replace([np.inf, -np.inf], np.nan).fillna(medians)


def select_features(train: pd.DataFrame, columns: list[str], seed: int) -> pd.DataFrame:
    x = median_fill(train, train, columns)
    y = train["knee_cycle"].to_numpy(float)
    rf = RandomForestRegressor(
        n_estimators=700, min_samples_leaf=2, max_features=0.7,
        random_state=seed, n_jobs=-1,
    ).fit(x, y)
    rows = []
    for col, importance in zip(columns, rf.feature_importances_, strict=True):
        rho = spearmanr(x[col].to_numpy(float), y).statistic
        rows.append({"feature": col, "abs_spearman": abs(float(rho)), "rf_importance": float(importance)})
    ranking = pd.DataFrame(rows)
    ranking["spearman_rank"] = ranking["abs_spearman"].rank(ascending=False, method="min")
    ranking["rf_rank"] = ranking["rf_importance"].rank(ascending=False, method="min")
    ranking["combined_rank"] = ranking["spearman_rank"] + ranking["rf_rank"]
    return ranking.sort_values(["combined_rank", "abs_spearman", "rf_importance"], ascending=[True, False, False]).reset_index(drop=True)


def make_xgb(params: dict[str, float], seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror", n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]), learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=float(params["min_child_weight"]), reg_lambda=float(params["reg_lambda"]),
        random_state=seed, n_jobs=-1, verbosity=0,
    )


def make_rf(params: dict[str, float], seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]), min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=float(params["max_features"]), max_depth=None if params["max_depth"] == 0 else int(params["max_depth"]),
        random_state=seed, n_jobs=-1,
    )


def candidate_grids() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    xgb = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": sub,
         "colsample_bytree": col, "min_child_weight": 1.0, "reg_lambda": lam}
        for n, d, lr, sub, col, lam in [
            (200, 2, 0.03, 0.85, 0.80, 2.0), (300, 2, 0.03, 0.90, 0.85, 4.0),
            (250, 3, 0.03, 0.85, 0.80, 4.0), (200, 2, 0.05, 0.90, 0.90, 4.0),
            (300, 3, 0.02, 0.90, 0.80, 8.0), (150, 2, 0.05, 0.80, 0.80, 8.0),
        ]
    ]
    rf = [
        {"n_estimators": n, "min_samples_leaf": leaf, "max_features": frac, "max_depth": depth}
        for n, leaf, frac, depth in [
            (600, 1, 0.70, 0), (700, 2, 0.70, 0), (800, 2, 1.00, 0),
            (700, 3, 0.70, 0), (700, 2, 0.70, 8), (900, 1, 0.50, 0),
        ]
    ]
    return xgb, rf


def cv_mae(factory, params: dict[str, float], x: pd.DataFrame, y: np.ndarray, seed: int) -> float:
    splitter = RepeatedKFold(n_splits=3, n_repeats=3, random_state=seed)
    values: list[float] = []
    for fold, (tr, va) in enumerate(splitter.split(x)):
        model = factory(params, seed + fold)
        model.fit(x.iloc[tr], y[tr])
        values.append(mean_absolute_error(y[va], model.predict(x.iloc[va])))
    return float(np.mean(values))


def tune(x: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    xgb_grid, rf_grid = candidate_grids()
    rows: list[dict[str, object]] = []
    for family, factory, grid in [("xgboost", make_xgb, xgb_grid), ("random_forest", make_rf, rf_grid)]:
        for index, params in enumerate(grid):
            rows.append({"family": family, "candidate": index, "cv_mae": cv_mae(factory, params, x, y, seed), "params": json.dumps(params)})
    table = pd.DataFrame(rows).sort_values(["family", "cv_mae"]).reset_index(drop=True)
    selected = {}
    for family, grid in [("xgboost", xgb_grid), ("random_forest", rf_grid)]:
        index = int(table[table["family"] == family].iloc[0]["candidate"])
        selected[family] = grid[index]
    return selected["xgboost"], selected["random_forest"], table


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    extended = pd.read_csv(args.extended_features)
    cycles = pd.read_csv(args.cycles)
    table = construct_candidates(extended, cycles)
    table = table[table["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    expected = {"train": 41, "primary_test": 42, "secondary_test": 40}
    counts = table["official_split"].value_counts().to_dict()
    if counts != expected:
        raise RuntimeError(f"Eligible split counts mismatch: {counts}; expected {expected}")

    features = [col for col in table.columns if col.startswith("F")]
    train = table[table["official_split"] == "train"].copy().reset_index(drop=True)
    ranking = select_features(train, features, args.random_state)
    selected = ranking.head(7)["feature"].tolist()
    ranking["selected"] = ranking["feature"].isin(selected)
    x_train = median_fill(train, train, selected)
    y_train = train["knee_cycle"].to_numpy(float)
    xgb_params, rf_params, tuning = tune(x_train, y_train, args.random_state)

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_repeats):
        xgb = make_xgb(xgb_params, seed)
        rf = make_rf(rf_params, seed)
        xgb.fit(x_train, y_train)
        rf.fit(x_train, y_train)
        for split in ["primary_test", "secondary_test"]:
            test = table[table["official_split"] == split].copy().reset_index(drop=True)
            x_test = median_fill(train, test, selected)
            observed = test["knee_cycle"].to_numpy(float)
            predicted = 0.5 * np.asarray(xgb.predict(x_test), float) + 0.5 * np.asarray(rf.predict(x_test), float)
            all_metrics.append({"split": split, "seed": seed, "n": len(test), **metrics(observed, predicted)})
            all_predictions.extend({
                "split": split, "seed": seed, "cell_id": row.cell_id, "batch": row.batch,
                "observed": float(obs), "predicted": float(pred), "abs_error": float(abs(obs - pred)),
            } for row, obs, pred in zip(test.itertuples(index=False), observed, predicted, strict=True))

    metric_table = pd.DataFrame(all_metrics)
    pred_table = pd.DataFrame(all_predictions)
    per_cell = pred_table.groupby(["split", "cell_id", "batch", "observed"], as_index=False).agg(
        predicted_mean=("predicted", "mean"), abs_error_mean=("abs_error", "mean"),
    )
    summary = metric_table.groupby("split")[["mae", "rmse", "r2", "abs_error_q90"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    formal = pd.read_csv(Path(args.formal_dir) / "heldout_metrics_by_seed.csv")
    formal_summary = formal.groupby("split")[["mae", "rmse", "r2", "abs_error_q90"]].mean().reset_index()
    formal_summary.insert(0, "method", "LDQ-ET ExtraTrees (formal 70-input representation)")
    baseline_summary = metric_table.groupby("split")[["mae", "rmse", "r2", "abs_error_q90"]].mean().reset_index()
    baseline_summary.insert(0, "method", "Bai-inspired XGB-RF (adapted 7 early features)")

    table.to_csv(out / "bai_inspired_candidate_features.csv", index=False)
    ranking.to_csv(out / "training_only_feature_selection.csv", index=False)
    tuning.to_csv(out / "inner_cv_tuning.csv", index=False)
    metric_table.to_csv(out / "heldout_metrics_by_seed.csv", index=False)
    pred_table.to_csv(out / "heldout_predictions_by_seed.csv", index=False)
    per_cell.to_csv(out / "heldout_per_cell.csv", index=False)
    summary.to_csv(out / "heldout_summary.csv", index=False)
    pd.concat([formal_summary, baseline_summary], ignore_index=True).to_csv(out / "comparison_with_formal_ldqet.csv", index=False)
    (out / "frozen_config.json").write_text(json.dumps({
        "method": "Bai-inspired XGB-RF ensemble (adapted reimplementation)",
        "observation_window": "cycles 1--150; selected curve contrasts use cycles 10 and 150",
        "selected_features": selected,
        "feature_selection": "Spearman association plus RF importance, computed using 41 training cells only",
        "xgboost_params": xgb_params,
        "rf_params": rf_params,
        "ensemble": "equal-weight mean of XGBoost and RandomForest predictions",
        "not_reproduced": ["Newton--Raphson-based optimizer", "source paper random 70/30 split", "source paper knee labels"],
        "heldout_labels_used_for_selection_or_tuning": False,
    }, indent=2), encoding="utf-8")
    print("Selected features:", ", ".join(selected))
    print("\nBai-inspired held-out mean across 30 seeds")
    print(metric_table.groupby("split")[["mae", "rmse", "r2", "abs_error_q90"]].mean().round(3).to_string())
    print("\nComparison with formal LDQ-ET")
    print(pd.concat([formal_summary, baseline_summary], ignore_index=True).round(3).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended-features", default="data/early_features_N150_extended_20170630.csv")
    parser.add_argument("--cycles", default="data/processed_cycles_20170630.csv")
    parser.add_argument("--formal-dir", default="results/formal_414240_selector_N150_local9_table3_verified_20260730_complete")
    parser.add_argument("--output-dir", default="results/bai_inspired_xgbrf_N150_training_only_20260731")
    parser.add_argument("--random-state", type=int, default=20260721)
    parser.add_argument("--seed-start", type=int, default=123)
    parser.add_argument("--seed-repeats", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
