from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

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
    cols: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col == f"{prefix}_window_complete":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        if feature_bucket(col, prefix) in {"raw", "deltaq"} or col.startswith(f"{prefix}_ldq_"):
            cols.append(col)
    return cols


def parameter_grid() -> list[dict[str, object]]:
    grid = {
        "n_estimators": [500, 800, 1200],
        "max_features": ["sqrt", 0.35, 0.5, 0.75, 1.0],
        "min_samples_leaf": [1, 2, 3, 4],
        "min_samples_split": [2, 4, 8],
        "max_depth": [None, 6, 10, 16],
        "bootstrap": [False, True],
    }
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def make_model(params: dict[str, object], seed: int) -> ExtraTreesRegressor:
    max_depth = params["max_depth"]
    if pd.notna(max_depth) and max_depth is not None:
        max_depth = int(float(max_depth))
    else:
        max_depth = None
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


def repeated_cv_score(
    params: dict[str, object],
    x_dev: pd.DataFrame,
    y_dev: np.ndarray,
    seed: int,
    n_splits: int,
    n_repeats: int,
) -> dict[str, float]:
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    maes: list[float] = []
    rmses: list[float] = []
    for fold_id, (tr, va) in enumerate(splitter.split(x_dev)):
        model = make_model(params, seed + fold_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_dev.iloc[tr], y_dev[tr])
        pred = np.asarray(model.predict(x_dev.iloc[va]), dtype=float)
        maes.append(float(mean_absolute_error(y_dev[va], pred)))
        rmses.append(float(mean_squared_error(y_dev[va], pred) ** 0.5))
    return {
        "dev_cv_mae_mean": float(np.mean(maes)),
        "dev_cv_mae_std": float(np.std(maes)),
        "dev_cv_mae_max": float(np.max(maes)),
        "dev_cv_rmse_mean": float(np.mean(rmses)),
        "dev_selection_score": float(np.mean(maes) + 0.10 * np.std(maes) + 0.10 * np.max(maes)),
    }


def evaluate_final(
    params: dict[str, object],
    df: pd.DataFrame,
    x: pd.DataFrame,
    y: np.ndarray,
    dev_mask: np.ndarray,
    external_mask: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = make_model(params, seed)
    model.fit(x.loc[dev_mask], y[dev_mask])
    pred = np.asarray(model.predict(x), dtype=float)
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split_name, mask in [("development_train_primary", dev_mask), ("external_secondary", external_mask)]:
        metrics_rows.append(
            {
                "seed": seed,
                "split": split_name,
                "n": int(mask.sum()),
                **regression_metrics(y[mask], pred[mask]),
            }
        )
    for row_index, row in df.iterrows():
        prediction_rows.append(
            {
                "seed": seed,
                "cell_id": row["cell_id"],
                "batch": row["batch"],
                "official_split": row["official_split"],
                "observed_knee_cycle": float(row["knee_cycle"]),
                "predicted_knee_cycle": float(pred[row_index]),
                "abs_error": float(abs(float(row["knee_cycle"]) - float(pred[row_index]))),
            }
        )
    return metrics_rows, prediction_rows


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.feature_path)
    df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    prefix = detect_window_prefix(df)
    features = select_features(df, prefix)
    x = df[features]
    y = df["knee_cycle"].to_numpy(dtype=float)
    dev_mask = df["official_split"].isin(["train", "primary_test"]).to_numpy()
    external_mask = (df["official_split"] == "secondary_test").to_numpy()
    x_dev = x.loc[dev_mask].reset_index(drop=True)
    y_dev = y[dev_mask]

    rng = np.random.default_rng(args.random_state)
    params_all = parameter_grid()
    if args.max_candidates and args.max_candidates < len(params_all):
        idx = rng.choice(len(params_all), size=args.max_candidates, replace=False)
        params_all = [params_all[i] for i in idx]

    # Include the current baseline ExtraTrees configuration as an anchor.
    anchor = {
        "n_estimators": 700,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "max_depth": None,
        "bootstrap": False,
    }
    if anchor not in params_all:
        params_all.append(anchor)

    selection_rows: list[dict[str, object]] = []
    for i, params in enumerate(params_all):
        candidate_id = f"et_hpo_{i:04d}"
        score = repeated_cv_score(params, x_dev, y_dev, args.random_state, args.n_splits, args.n_repeats)
        selection_rows.append({"candidate_id": candidate_id, **score, **params})

    selection = pd.DataFrame(selection_rows).sort_values(["dev_selection_score", "dev_cv_mae_mean"]).reset_index(drop=True)
    selected = selection.iloc[0].to_dict()
    selected_params = {k: selected[k] for k in ["n_estimators", "max_features", "min_samples_leaf", "min_samples_split", "max_depth", "bootstrap"]}

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    for seed in range(args.random_state, args.random_state + args.seed_repeats):
        metrics_rows, prediction_rows = evaluate_final(selected_params, df, x, y, dev_mask, external_mask, seed)
        for row in metrics_rows:
            row["candidate_id"] = selected["candidate_id"]
            row.update(selected_params)
        all_metrics.extend(metrics_rows)
        all_predictions.extend(prediction_rows)

    metrics = pd.DataFrame(all_metrics)
    summary = metrics.groupby("split")[["mae", "rmse", "mape_percent", "r2"]].agg(["mean", "std", "min", "median", "max"])

    selection.to_csv(output_dir / "extratrees_hpo_selection.csv", index=False)
    metrics.to_csv(output_dir / "extratrees_hpo_final_metrics_by_seed.csv", index=False)
    summary.to_csv(output_dir / "extratrees_hpo_final_summary.csv")
    pd.DataFrame(all_predictions).to_csv(output_dir / "extratrees_hpo_final_predictions.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(output_dir / "features.csv", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "feature_path": args.feature_path,
                "selected_candidate": selected["candidate_id"],
                "selected_params": selected_params,
                "n_splits": args.n_splits,
                "n_repeats": args.n_repeats,
                "seed_repeats": args.seed_repeats,
                "max_candidates": args.max_candidates,
                "note": "Hyperparameters are selected only on the development cohort. Secondary is external validation.",
            },
            fp,
            indent=2,
        )

    print("Saved outputs to", output_dir)
    print("Selected candidate:")
    print(selection.head(10).round(4).to_string(index=False))
    print("\nFinal seed summary:")
    print(summary.round(4).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", default="data/early_features_N150_local_deltaq_top40_20170630.csv")
    parser.add_argument("--output-dir", default="results/extratrees_ldq_hpo_N150")
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=160)
    parser.add_argument("--seed-repeats", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

