from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover
    root_mean_squared_error = None


META_COLUMNS = {
    "cell_id",
    "batch",
    "official_order",
    "official_split",
    "cycle_life",
    "knee_cycle",
    "knee_soh",
    "relative_knee",
    "label_confidence",
    "knee_after_window",
    "remaining_cycles_from_window",
}

DROP_FEATURE_COLUMNS = set()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(mean_squared_error(y_true, y_pred, squared=False))


def regression_metrics(prefix: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": rmse(y_true, y_pred),
        f"{prefix}_mape_percent": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
    }


def target_to_knee_cycles(target: str, predicted: np.ndarray, df: pd.DataFrame, window: int) -> np.ndarray:
    if target == "relative_knee":
        return predicted * df["cycle_life"].to_numpy(dtype=float)
    if target == "remaining_cycles_from_window":
        return predicted + float(window)
    return predicted


def detect_window_prefix(df: pd.DataFrame) -> str:
    matches = sorted({c.split("_")[0] for c in df.columns if c.startswith("N") and "_" in c and c.split("_")[0][1:].isdigit()})
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one N-window prefix, found {matches}")
    return matches[0]


def feature_columns(df: pd.DataFrame, feature_set: str, prefix: str) -> list[str]:
    drop_cols = set(DROP_FEATURE_COLUMNS) | {f"{prefix}_window_complete"}
    cols = [c for c in df.columns if c not in META_COLUMNS and c not in drop_cols]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if feature_set == "raw_only":
        numeric_cols = [
            c
            for c in numeric_cols
            if c == f"{prefix}_n_observed"
            or c.startswith(f"{prefix}_qd_")
            or c.startswith(f"{prefix}_soh_raw_")
            or c.startswith(f"{prefix}_raw_regen_")
            or c.startswith(f"{prefix}_raw_max_single_cycle")
        ]
    constant = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]
    return [c for c in numeric_cols if c not in constant]


def build_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "ridge_cv": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 41))),
            ]
        ),
        "svr_rbf": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=10.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5.0,
            reg_alpha=0.1,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=4,
        )
    return models


def plot_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    models = predictions["model"].drop_duplicates().tolist()
    ncols = 2
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.7 * nrows), squeeze=False)
    colors = {"train": "#4C78A8", "primary_test": "#E45756", "secondary_test": "#F58518"}
    for ax, model_name in zip(axes.ravel(), models):
        part = predictions[predictions["model"] == model_name]
        for split, color in colors.items():
            subset = part[part["split"] == split]
            ax.scatter(subset["observed_knee_cycle"], subset["predicted_knee_cycle"], s=42, alpha=0.82, label=split, color=color)
        low = float(min(part["observed_knee_cycle"].min(), part["predicted_knee_cycle"].min()))
        high = float(max(part["observed_knee_cycle"].max(), part["predicted_knee_cycle"].max()))
        ax.plot([low, high], [low, high], color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(model_name)
        ax.set_xlabel("Observed knee cycle")
        ax.set_ylabel("Predicted knee cycle")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(models) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "knee_baseline_predictions.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/early_features_N100_20170630.csv")
    parser.add_argument("--output-dir", default="results/knee_baselines_N100")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target", choices=["knee_cycle", "relative_knee", "remaining_cycles_from_window"], default="knee_cycle")
    parser.add_argument("--feature-set", choices=["all", "raw_only"], default="raw_only")
    parser.add_argument("--min-label-confidence", type=float, default=0.0)
    parser.add_argument("--window", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    before = len(df)
    df = df[df["knee_after_window"].astype(bool)].copy()
    if args.min_label_confidence > 0:
        df = df[df["label_confidence"] >= args.min_label_confidence].copy()
    df = df.reset_index(drop=True)

    prefix = detect_window_prefix(df)
    window = args.window if args.window is not None else int(prefix[1:])
    features = feature_columns(df, args.feature_set, prefix)
    if df[features].isna().any().any():
        missing = df[features].isna().sum()
        raise ValueError(f"Feature matrix contains missing values:\n{missing[missing > 0]}")

    x = df[features]
    y = df[args.target].to_numpy(dtype=float)
    train_mask = df["official_split"] == "train"
    primary_mask = df["official_split"] == "primary_test"
    secondary_mask = df["official_split"] == "secondary_test"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    feature_importance_rows: list[dict[str, object]] = []

    for model_name, model in build_models(args.random_state).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x.loc[train_mask], y[train_mask.to_numpy()])
        pred = model.predict(x)
        pred_knee = target_to_knee_cycles(args.target, pred, df, window)
        for split_name, mask in [
            ("train", train_mask),
            ("primary_test", primary_mask),
            ("secondary_test", secondary_mask),
        ]:
            idx = mask.to_numpy()
            row = {
                "model": model_name,
                "split": split_name,
                "n": int(idx.sum()),
            }
            row.update(regression_metrics("target", y[idx], pred[idx]))
            row.update(regression_metrics("knee_cycle", df["knee_cycle"].to_numpy(dtype=float)[idx], pred_knee[idx]))
            metrics_rows.append(row)

        for row_index, row in df.iterrows():
            prediction_rows.append(
                {
                    "model": model_name,
                    "cell_id": row["cell_id"],
                    "split": row["official_split"],
                    "cycle_life": float(row["cycle_life"]),
                    "observed_knee_cycle": float(row["knee_cycle"]),
                    "observed_target": float(row[args.target]),
                    "predicted_target": float(pred[row_index]),
                    "predicted_knee_cycle": float(pred_knee[row_index]),
                    "absolute_error": float(abs(float(row["knee_cycle"]) - pred_knee[row_index])),
                    "target_absolute_error": float(abs(float(row[args.target]) - pred[row_index])),
                    "label_confidence": float(row["label_confidence"]),
                }
            )

        estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
        if hasattr(estimator, "coef_"):
            for feature, value in zip(features, np.ravel(estimator.coef_)):
                feature_importance_rows.append({"model": model_name, "feature": feature, "importance": float(value)})
        elif hasattr(estimator, "feature_importances_"):
            for feature, value in zip(features, estimator.feature_importances_):
                feature_importance_rows.append({"model": model_name, "feature": feature, "importance": float(value)})

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    importances = pd.DataFrame(feature_importance_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    importances.to_csv(output_dir / "feature_importance.csv", index=False)
    with open(output_dir / "model_run_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "input_rows": before,
                "used_rows": len(df),
                "excluded_rows": before - len(df),
                "feature_set": args.feature_set,
                "window": window,
                "window_prefix": prefix,
                "feature_count": len(features),
                "features": features,
                "dropped_feature_columns": sorted(DROP_FEATURE_COLUMNS),
                "target": args.target,
                "excluded_reason": "knee_after_window=False and optional low label confidence",
            },
            fp,
            indent=2,
        )
    plot_predictions(predictions, output_dir)

    print(f"Saved metrics to {output_dir / 'metrics.csv'}")
    print(metrics.to_string(index=False))
    print(f"Saved predictions to {output_dir / 'predictions.csv'}")
    print(f"Used {len(df)} of {before} rows and {len(features)} feature columns")


if __name__ == "__main__":
    main()








