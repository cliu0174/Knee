from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - for older scikit-learn
    root_mean_squared_error = None


LEAKAGE_COLUMNS = {
    "Unnamed: 0",
    "lifetime",
    "cycle_life",
    "lifePercent",
    "cell_id",
    "batch",
    "official_split",
    "official_order",
    "cycle_a",
    "cycle_b",
}


@dataclass
class Metrics:
    dataset: str
    model: str
    split: str
    target_space: str
    mae_cycles: float
    rmse_cycles: float
    mape_percent: float
    r2_cycles: float
    rmse_log10: float
    clipped_predictions: int


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "lifetime" not in df.columns:
        raise ValueError(f"{path} does not contain required target column 'lifetime'")

    y = df["lifetime"].astype(float)
    feature_cols = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    x = df[feature_cols].astype(float)
    if x.isna().any().any() or y.isna().any():
        raise ValueError(f"{path} contains NaN values after feature selection")
    return x, y


def select_feature_window(x: pd.DataFrame, window_index: int | None) -> pd.DataFrame:
    if window_index is None:
        return x
    suffix = f"-{window_index}"
    selected = [c for c in x.columns if c.endswith(suffix)]
    if not selected:
        raise ValueError(f"No feature columns end with {suffix!r}")
    return x[selected].copy()


def build_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "ridge_cv": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 41))),
            ]
        ),
        "elastic_net_cv": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.05, 0.1, 0.5, 0.9],
                        alphas=np.logspace(-4, 1, 30),
                        cv=5,
                        max_iter=50000,
                        tol=1e-5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "svr_rbf": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.03)),
            ]
        ),
        "gpr_rbf": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0, (1e-3, 1e3))
                        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
                        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0)),
                        normalize_y=True,
                        n_restarts_optimizer=2,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=1,
            max_features=None,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5.0,
            reg_alpha=0.05,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=4,
        )
    return models


def evaluate_predictions(
    dataset_name: str,
    model_name: str,
    split_name: str,
    y_true_cycles: np.ndarray,
    y_pred_log: np.ndarray,
    clip_bounds: tuple[float, float],
) -> Metrics:
    y_pred_log_clipped = np.clip(y_pred_log, clip_bounds[0], clip_bounds[1])
    clipped_predictions = int(np.sum(y_pred_log != y_pred_log_clipped))
    y_pred_cycles = np.power(10.0, y_pred_log_clipped)
    y_true_log = np.log10(y_true_cycles)
    rmse_cycles = (
        root_mean_squared_error(y_true_cycles, y_pred_cycles)
        if root_mean_squared_error is not None
        else mean_squared_error(y_true_cycles, y_pred_cycles, squared=False)
    )
    rmse_log10 = (
        root_mean_squared_error(y_true_log, y_pred_log_clipped)
        if root_mean_squared_error is not None
        else mean_squared_error(y_true_log, y_pred_log_clipped, squared=False)
    )
    return Metrics(
        dataset=dataset_name,
        model=model_name,
        split=split_name,
        target_space="log10_lifetime",
        mae_cycles=float(mean_absolute_error(y_true_cycles, y_pred_cycles)),
        rmse_cycles=float(rmse_cycles),
        mape_percent=float(np.mean(np.abs((y_true_cycles - y_pred_cycles) / y_true_cycles)) * 100.0),
        r2_cycles=float(r2_score(y_true_cycles, y_pred_cycles)),
        rmse_log10=float(rmse_log10),
        clipped_predictions=clipped_predictions,
    )


def fit_predict_holdout(
    x: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    random_state: int,
    output_dir: Path,
) -> tuple[list[Metrics], dict[str, np.ndarray]]:
    y_log = np.log10(y.to_numpy())
    x_train, x_test, y_train_log, y_test_log, y_train_cycles, y_test_cycles = train_test_split(
        x,
        y_log,
        y.to_numpy(),
        test_size=0.2,
        random_state=random_state,
    )

    metrics: list[Metrics] = []
    predictions: dict[str, np.ndarray] = {}
    clip_bounds = (float(y_train_log.min() - 0.1), float(y_train_log.max() + 0.1))
    for model_name, model in build_models(random_state).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train_log)
        pred_log = model.predict(x_test)
        pred_log = np.clip(pred_log, clip_bounds[0], clip_bounds[1])
        metrics.append(
            evaluate_predictions(
                dataset_name,
                model_name,
                "holdout_80_20",
                y_test_cycles,
                pred_log,
                clip_bounds,
            )
        )
        predictions[model_name] = np.column_stack([y_test_cycles, np.power(10.0, pred_log)])

    plot_prediction_scatter(dataset_name, predictions, output_dir)
    return metrics, predictions


def fit_predict_official_split(
    x: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    random_state: int,
    output_dir: Path,
) -> tuple[list[Metrics], dict[str, np.ndarray]]:
    """Approximate the Severson paper split.

    The official loader reports numBat1=41, numBat2=43, numBat3=40, then uses
    alternating indices in batch1+batch2 for train/primary test and all batch3
    cells for secondary test. The processed CSV used here has 123 rows because
    one short-life outlier is already removed, so the secondary set has 39 rows.
    """
    if len(y) < 85:
        raise ValueError("Official split requires the combined processed Severson rows")
    train_idx = np.arange(1, 83, 2)
    primary_test_idx = np.hstack((np.arange(0, 84, 2), 83))
    secondary_test_idx = np.arange(84, len(y))

    y_arr = y.to_numpy()
    y_log = np.log10(y_arr)
    x_train = x.iloc[train_idx]
    y_train_log = y_log[train_idx]
    clip_bounds = (float(y_train_log.min() - 0.1), float(y_train_log.max() + 0.1))

    metrics: list[Metrics] = []
    predictions: dict[str, np.ndarray] = {}
    split_indices = {
        "official_primary_test": primary_test_idx,
        "official_secondary_test": secondary_test_idx,
    }

    for model_name, model in build_models(random_state).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train_log)
        for split_name, test_idx in split_indices.items():
            pred_log = np.clip(model.predict(x.iloc[test_idx]), clip_bounds[0], clip_bounds[1])
            metrics.append(
                evaluate_predictions(
                    dataset_name,
                    model_name,
                    split_name,
                    y_arr[test_idx],
                    pred_log,
                    clip_bounds,
                )
            )
            predictions[f"{model_name}_{split_name.replace('official_', '')}"] = np.column_stack(
                [y_arr[test_idx], np.power(10.0, pred_log)]
            )

    plot_prediction_scatter(dataset_name + "_official", predictions, output_dir)
    split_summary = {
        "train_idx": train_idx.tolist(),
        "primary_test_idx": primary_test_idx.tolist(),
        "secondary_test_idx": secondary_test_idx.tolist(),
        "n_train": int(len(train_idx)),
        "n_primary_test": int(len(primary_test_idx)),
        "n_secondary_test": int(len(secondary_test_idx)),
    }
    with open(output_dir / f"{dataset_name}_official_split_indices.json", "w", encoding="utf-8") as fp:
        json.dump(split_summary, fp, indent=2)
    return metrics, predictions


def fit_predict_labeled_official_split(
    x: pd.DataFrame,
    y: pd.Series,
    split_labels: pd.Series,
    dataset_name: str,
    random_state: int,
    output_dir: Path,
) -> tuple[list[Metrics], dict[str, np.ndarray]]:
    y_arr = y.to_numpy()
    y_log = np.log10(y_arr)
    train_idx = np.where(split_labels.to_numpy() == "train")[0]
    primary_test_idx = np.where(split_labels.to_numpy() == "primary_test")[0]
    secondary_test_idx = np.where(split_labels.to_numpy() == "secondary_test")[0]
    if len(train_idx) == 0 or len(primary_test_idx) == 0:
        raise ValueError("Labeled official split requires train and primary_test rows")

    x_train = x.iloc[train_idx]
    y_train_log = y_log[train_idx]
    clip_bounds = (float(y_train_log.min() - 0.1), float(y_train_log.max() + 0.1))
    split_indices = {
        "official_primary_test": primary_test_idx,
        "official_secondary_test": secondary_test_idx,
    }

    metrics: list[Metrics] = []
    predictions: dict[str, np.ndarray] = {}
    for model_name, model in build_models(random_state).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train_log)
        for split_name, test_idx in split_indices.items():
            if len(test_idx) == 0:
                continue
            pred_log = np.clip(model.predict(x.iloc[test_idx]), clip_bounds[0], clip_bounds[1])
            metrics.append(
                evaluate_predictions(
                    dataset_name,
                    model_name,
                    split_name,
                    y_arr[test_idx],
                    pred_log,
                    clip_bounds,
                )
            )
            predictions[f"{model_name}_{split_name.replace('official_', '')}"] = np.column_stack(
                [y_arr[test_idx], np.power(10.0, pred_log)]
            )

    plot_prediction_scatter(dataset_name + "_official", predictions, output_dir)
    split_summary = {
        "source": "official_split column",
        "n_train": int(len(train_idx)),
        "n_primary_test": int(len(primary_test_idx)),
        "n_secondary_test": int(len(secondary_test_idx)),
    }
    with open(output_dir / f"{dataset_name}_official_split_indices.json", "w", encoding="utf-8") as fp:
        json.dump(split_summary, fp, indent=2)
    return metrics, predictions


def fit_predict_cv(
    x: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    random_state: int,
) -> list[Metrics]:
    y_arr = y.to_numpy()
    y_log = np.log10(y_arr)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    all_metrics: list[Metrics] = []

    for model_name in build_models(random_state):
        pred_log = np.zeros_like(y_log)
        clipped_total = 0
        for train_idx, test_idx in kf.split(x):
            model = build_models(random_state)[model_name]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x.iloc[train_idx], y_log[train_idx])
            fold_pred = model.predict(x.iloc[test_idx])
            clip_bounds = (float(y_log[train_idx].min() - 0.1), float(y_log[train_idx].max() + 0.1))
            fold_pred_clipped = np.clip(fold_pred, clip_bounds[0], clip_bounds[1])
            clipped_total += int(np.sum(fold_pred != fold_pred_clipped))
            pred_log[test_idx] = fold_pred_clipped
        metric = evaluate_predictions(
            dataset_name,
            model_name,
            "cv5_out_of_fold",
            y_arr,
            pred_log,
            (float(y_log.min() - 0.1), float(y_log.max() + 0.1)),
        )
        metric.clipped_predictions = clipped_total
        all_metrics.append(metric)
    return all_metrics


def plot_lifetime_distribution(dataset_name: str, y: pd.Series, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(y, bins=20, color="#4C78A8", edgecolor="white")
    ax.set_title(f"{dataset_name}: lifetime distribution")
    ax.set_xlabel("Cycle life")
    ax.set_ylabel("Cell count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset_name}_lifetime_distribution.png", dpi=220)
    plt.close(fig)


def plot_prediction_scatter(
    dataset_name: str,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.2), squeeze=False)
    all_values = np.concatenate([arr.ravel() for arr in predictions.values()])
    lim_min = max(0, float(np.nanmin(all_values)) * 0.9)
    lim_max = float(np.nanmax(all_values)) * 1.05
    for ax, (model_name, arr) in zip(axes.ravel(), predictions.items()):
        y_true, y_pred = arr[:, 0], arr[:, 1]
        ax.scatter(y_true, y_pred, s=36, alpha=0.8, color="#F58518", edgecolor="white", linewidth=0.5)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#333333", linestyle="--", linewidth=1)
        ax.set_title(model_name)
        ax.set_xlabel("Observed cycle life")
        ax.set_ylabel("Predicted cycle life")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.grid(alpha=0.25)
    fig.suptitle(f"{dataset_name}: holdout predictions", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset_name}_holdout_prediction_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    holdout = metrics_df[metrics_df["split"] == "holdout_80_20"].copy()
    if holdout.empty:
        return
    labels = holdout["dataset"] + "\n" + holdout["model"]
    fig, ax = plt.subplots(figsize=(max(8, len(holdout) * 0.9), 4.8))
    ax.bar(np.arange(len(holdout)), holdout["mape_percent"], color="#54A24B")
    ax.set_xticks(np.arange(len(holdout)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Holdout model comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_model_comparison_mape.png", dpi=220)
    plt.close(fig)


def plot_feature_importance(
    x: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    random_state: int,
    output_dir: Path,
) -> None:
    y_log = np.log10(y.to_numpy())
    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x, y_log)
    order = np.argsort(model.feature_importances_)[::-1][:20]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(np.arange(len(order)), model.feature_importances_[order][::-1], color="#72B7B2")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([x.columns[i] for i in order][::-1])
    ax.set_xlabel("Random forest feature importance")
    ax.set_title(f"{dataset_name}: top 20 features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset_name}_feature_importance.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/severson_processed")
    parser.add_argument(
        "--csv-files",
        nargs="*",
        default=None,
        help="CSV file names or paths to run. Default: original.csv regularized.csv inside --data-dir.",
    )
    parser.add_argument("--output-dir", default="results/severson_baseline")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--full-cv",
        action="store_true",
        help="Run 5-fold out-of-fold CV. Default is holdout only for a fast first reproduction.",
    )
    parser.add_argument(
        "--split",
        choices=["holdout", "official"],
        default="holdout",
        help="Use a random 80/20 holdout or the official Severson-style split.",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Use only one extracted early-cycle window: 0=30-100, 1=30-150, 2=30-200, 3=30-250. Default uses all windows.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[Metrics] = []
    dataset_summaries = {}
    csv_paths = args.csv_files or ["original.csv", "regularized.csv"]
    for csv_arg in csv_paths:
        csv_path = Path(csv_arg)
        if not csv_path.is_absolute():
            csv_path = data_dir / csv_path
        dataset_name = csv_path.stem
        raw_df = pd.read_csv(csv_path)
        split_labels = raw_df["official_split"].copy() if "official_split" in raw_df.columns else None
        x, y = load_dataset(csv_path)
        x = select_feature_window(x, args.window_index)
        if args.window_index is not None:
            dataset_name = f"{dataset_name}_window{args.window_index}"
        dataset_summaries[dataset_name] = {
            "path": str(csv_path),
            "n_samples": int(len(y)),
            "n_features": int(x.shape[1]),
            "window_index": args.window_index,
            "target_min": float(y.min()),
            "target_median": float(y.median()),
            "target_max": float(y.max()),
            "excluded_columns": sorted(LEAKAGE_COLUMNS),
        }
        plot_lifetime_distribution(dataset_name, y, output_dir)
        if args.split == "official":
            if split_labels is not None:
                split_metrics, _ = fit_predict_labeled_official_split(
                    x,
                    y,
                    split_labels,
                    dataset_name,
                    args.random_state,
                    output_dir,
                )
            else:
                split_metrics, _ = fit_predict_official_split(x, y, dataset_name, args.random_state, output_dir)
        else:
            split_metrics, _ = fit_predict_holdout(x, y, dataset_name, args.random_state, output_dir)
        all_metrics.extend(split_metrics)
        if args.full_cv:
            cv_metrics = fit_predict_cv(x, y, dataset_name, args.random_state)
            all_metrics.extend(cv_metrics)
        plot_feature_importance(x, y, dataset_name, args.random_state, output_dir)

    metrics_df = pd.DataFrame([asdict(m) for m in all_metrics])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    plot_model_comparison(metrics_df, output_dir)
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as fp:
        json.dump(dataset_summaries, fp, indent=2)
    print(metrics_df.sort_values(["dataset", "split", "mape_percent"]).to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
