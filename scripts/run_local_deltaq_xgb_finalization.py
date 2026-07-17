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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold

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


def fit_predict(params: dict[str, object], x_train: pd.DataFrame, y_train: np.ndarray, x_pred: pd.DataFrame, random_state: int) -> np.ndarray:
    model = make_xgb(params, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return np.asarray(model.predict(x_pred), dtype=float)


def repeated_kfold_score(
    params: dict[str, object],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
    n_splits: int,
    n_repeats: int,
) -> dict[str, float]:
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_maes: list[float] = []
    fold_rmses: list[float] = []
    for fold_index, (tr, va) in enumerate(splitter.split(x_train)):
        pred = fit_predict(params, x_train.iloc[tr], y_train[tr], x_train.iloc[va], random_state + fold_index)
        fold_maes.append(float(mean_absolute_error(y_train[va], pred)))
        fold_rmses.append(float(mean_squared_error(y_train[va], pred) ** 0.5))
    return {
        "rkf_mae_mean": float(np.mean(fold_maes)),
        "rkf_mae_std": float(np.std(fold_maes)),
        "rkf_mae_max": float(np.max(fold_maes)),
        "rkf_rmse_mean": float(np.mean(fold_rmses)),
        "rkf_selection_score": float(np.mean(fold_maes) + 0.10 * np.std(fold_maes) + 0.10 * np.max(fold_maes)),
    }


def single_kfold_oof_predictions(
    params: dict[str, object],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
    n_splits: int,
) -> np.ndarray:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(y_train.size, dtype=float)
    for fold_index, (tr, va) in enumerate(splitter.split(x_train)):
        oof[va] = fit_predict(params, x_train.iloc[tr], y_train[tr], x_train.iloc[va], random_state + fold_index)
    return oof


def add_metrics(rows: list[dict[str, object]], method: str, df: pd.DataFrame, pred: np.ndarray) -> None:
    for split in ["train", "primary_test", "secondary_test"]:
        idx = (df["official_split"] == split).to_numpy()
        rows.append(
            {
                "method": method,
                "split": split,
                "n": int(idx.sum()),
                **regression_metrics(df.loc[idx, "knee_cycle"].to_numpy(dtype=float), pred[idx]),
            }
        )


def plot_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    eval_df = metrics[metrics["split"].isin(["primary_test", "secondary_test"])].copy()
    pivot = eval_df.pivot(index="method", columns="split", values="mae")
    pivot["mean_mae"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_mae").drop(columns=["mean_mae"])
    ax = pivot.plot(kind="barh", figsize=(9.0, 5.2), color=["#4C78A8", "#F58518"])
    ax.invert_yaxis()
    ax.set_xlabel("MAE (cycles)")
    ax.set_ylabel("")
    ax.set_title("Local DeltaQ XGBoost finalization")
    ax.grid(axis="x", alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_dir / "local_deltaq_xgb_finalization_mae.png", dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.feature_path)
    df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    prefix = detect_window_prefix(df)
    features = select_features(df, prefix, include_local_deltaq=args.include_local_deltaq)
    x = df[features]
    y = df["knee_cycle"].to_numpy(dtype=float)
    train_mask = (df["official_split"] == "train").to_numpy()
    x_train = x.loc[train_mask].reset_index(drop=True)
    y_train = y[train_mask]

    params_df = pd.read_csv(args.params_path)
    if args.max_candidates:
        params_df = params_df.head(args.max_candidates).copy()

    selection_rows: list[dict[str, object]] = []
    for _, row in params_df.iterrows():
        params = {k: row[k] for k in row.index if k != "candidate_id"}
        score = repeated_kfold_score(
            params,
            x_train,
            y_train,
            random_state=args.random_state,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
        )
        selection_rows.append({"candidate_id": row["candidate_id"], **score, **params})

    selection = pd.DataFrame(selection_rows).sort_values(["rkf_selection_score", "rkf_mae_mean"]).reset_index(drop=True)
    selected = selection.iloc[0].to_dict()
    selected_params = {k: selected[k] for k in params_df.columns if k != "candidate_id"}
    selected_id = str(selected["candidate_id"])

    diag_candidate_id = args.diagnostic_candidate
    diag_row = pd.read_csv(args.full_params_path)
    diag_row = diag_row[diag_row["candidate_id"] == diag_candidate_id]
    if diag_row.empty:
        raise ValueError(f"Diagnostic candidate not found: {diag_candidate_id}")
    diag_params = {k: diag_row.iloc[0][k] for k in diag_row.columns if k != "candidate_id"}

    train_leave_batch_row = pd.read_csv(args.leave_batch_params_path).iloc[0]
    leave_batch_params = {k: train_leave_batch_row[k] for k in train_leave_batch_row.index if k != "candidate_id"}
    leave_batch_id = str(train_leave_batch_row["candidate_id"])

    model_specs = [
        ("rkf_selected_local_deltaq_xgb", selected_id, selected_params),
        ("leave_batch_selected_local_deltaq_xgb", leave_batch_id, leave_batch_params),
        ("diagnostic_sensitivity_local_deltaq_xgb", diag_candidate_id, diag_params),
    ]

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    for method, candidate_id, params in model_specs:
        full_pred = fit_predict(params, x.loc[train_mask], y[train_mask], x, args.random_state)
        add_metrics(metric_rows, method, df, full_pred)
        oof = single_kfold_oof_predictions(params, x_train, y_train, args.random_state, args.n_splits)
        for i, train_index in enumerate(np.flatnonzero(train_mask)):
            oof_rows.append(
                {
                    "method": method,
                    "candidate_id": candidate_id,
                    "cell_id": df.loc[train_index, "cell_id"],
                    "observed_knee_cycle": float(y[train_index]),
                    "oof_predicted_knee_cycle": float(oof[i]),
                    "oof_abs_error": float(abs(y[train_index] - oof[i])),
                }
            )
        for row_index, row in df.iterrows():
            prediction_rows.append(
                {
                    "method": method,
                    "candidate_id": candidate_id,
                    "cell_id": row["cell_id"],
                    "split": row["official_split"],
                    "observed_knee_cycle": float(row["knee_cycle"]),
                    "predicted_knee_cycle": float(full_pred[row_index]),
                    "abs_error": float(abs(float(row["knee_cycle"]) - float(full_pred[row_index]))),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    eval_df = metrics[metrics["split"].isin(["primary_test", "secondary_test"])].copy()
    ranking = eval_df.pivot_table(index="method", columns="split", values=["mae", "rmse", "r2", "mape_percent"], aggfunc="first")
    ranking.columns = [f"{metric}_{split}" for metric, split in ranking.columns]
    ranking = ranking.reset_index()
    ranking["mean_test_mae"] = ranking[["mae_primary_test", "mae_secondary_test"]].mean(axis=1)
    ranking["max_test_mae"] = ranking[["mae_primary_test", "mae_secondary_test"]].max(axis=1)
    ranking["min_test_r2"] = ranking[["r2_primary_test", "r2_secondary_test"]].min(axis=1)
    ranking = ranking.sort_values(["mean_test_mae", "max_test_mae"]).reset_index(drop=True)

    selection.to_csv(output_dir / "repeated_kfold_param_selection.csv", index=False)
    metrics.to_csv(output_dir / "finalization_metrics.csv", index=False)
    ranking.to_csv(output_dir / "finalization_ranking.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "finalization_predictions.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(output_dir / "train_oof_predictions.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(output_dir / "features.csv", index=False)
    plot_comparison(metrics, output_dir)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "feature_path": args.feature_path,
                "params_path": args.params_path,
                "full_params_path": args.full_params_path,
                "leave_batch_params_path": args.leave_batch_params_path,
                "selected_repeated_kfold_candidate": selected_id,
                "diagnostic_candidate": diag_candidate_id,
                "n_splits": args.n_splits,
                "n_repeats": args.n_repeats,
                "note": "Repeated K-fold selection uses train split only. Diagnostic candidate is reported as sensitivity, not strict final selection.",
            },
            fp,
            indent=2,
        )

    print("Saved outputs to", output_dir)
    print("Repeated K-fold selection top:")
    print(selection.head(12).round(3).to_string(index=False))
    print("\nFinalization ranking:")
    print(ranking.round(3).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", default="data/early_features_N150_local_deltaq_top40_20170630.csv")
    parser.add_argument("--params-path", default="results/hybrid_hpo_local_deltaq_top40_N150/xgb_params.csv")
    parser.add_argument("--full-params-path", default="results/hybrid_hpo_local_deltaq_top40_N150/xgb_params.csv")
    parser.add_argument("--leave-batch-params-path", default="results/hybrid_hpo_local_deltaq_top40_N150/xgb_train_only_selection.csv")
    parser.add_argument("--output-dir", default="results/local_deltaq_xgb_finalization_N150")
    parser.add_argument("--diagnostic-candidate", default="xgb_hpo_0212")
    parser.add_argument("--random-state", type=int, default=20260709)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=223)
    parser.add_argument("--include-local-deltaq", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
