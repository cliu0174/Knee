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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover
    root_mean_squared_error = None


FULL_MODEL_EXCLUDED_CELLS = {"b1c18", "b1c0", "b1c41", "b2c24"}

FEATURE_SETS = {
    "variance": ["dq_log_var"],
    "discharge_candidates": [
        "dq_min",
        "dq_mean",
        "dq_var",
        "dq_log_var",
        "dq_skewness",
        "dq_kurtosis",
        "dq_value_at_2v",
        "qd_slope_2_100",
        "qd_intercept_2_100",
        "qd_slope_91_100",
        "qd_intercept_91_100",
        "qd_cycle_2",
        "qd_max_minus_cycle_2",
        "qd_cycle_100",
    ],
    "full_candidates": [
        "dq_min",
        "dq_mean",
        "dq_var",
        "dq_log_var",
        "dq_skewness",
        "dq_kurtosis",
        "dq_value_at_2v",
        "qd_slope_2_100",
        "qd_intercept_2_100",
        "qd_slope_91_100",
        "qd_intercept_91_100",
        "qd_cycle_2",
        "qd_max_minus_cycle_2",
        "qd_cycle_100",
        "charge_time_avg_first_5",
        "temp_max_2_100",
        "temp_min_2_100",
        "temp_integral_2_100",
        "ir_cycle_2",
        "ir_min_2_100",
        "ir_diff_100_2",
    ],
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(mean_squared_error(y_true, y_pred, squared=False))


def metrics_for(y_true_cycles: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_pred_cycles = np.power(10.0, y_pred_log)
    return {
        "mae_cycles": float(mean_absolute_error(y_true_cycles, y_pred_cycles)),
        "rmse_cycles": rmse(y_true_cycles, y_pred_cycles),
        "mape_percent": float(np.mean(np.abs((y_true_cycles - y_pred_cycles) / y_true_cycles)) * 100.0),
        "rmse_log10": rmse(np.log10(y_true_cycles), y_pred_log),
    }


def build_model(model_name: str, random_state: int) -> Pipeline:
    if model_name == "linear":
        return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    if model_name == "elastic_net_cv":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.5, 0.8, 0.95, 1.0],
                        alphas=np.logspace(-4, 0, 40),
                        cv=4,
                        max_iter=50000,
                        tol=1e-5,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def coefficient_rows(model: Pipeline, feature_cols: list[str], model_name: str, feature_set: str) -> list[dict[str, object]]:
    estimator = model.named_steps["model"]
    coefs = getattr(estimator, "coef_", np.full(len(feature_cols), np.nan))
    rows = []
    for feature, coef in zip(feature_cols, coefs):
        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "feature": feature,
                "coefficient_on_z_scored_feature": float(coef),
                "selected_nonzero": bool(abs(coef) > 1e-10),
            }
        )
    rows.append(
        {
            "feature_set": feature_set,
            "model": model_name,
            "feature": "intercept",
            "coefficient_on_z_scored_feature": float(getattr(estimator, "intercept_", np.nan)),
            "selected_nonzero": True,
        }
    )
    return rows


def plot_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    model_keys = predictions[["feature_set", "model"]].drop_duplicates().to_records(index=False)
    ncols = 3
    nrows = int(np.ceil(len(model_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False)
    for ax, (feature_set, model) in zip(axes.ravel(), model_keys):
        part = predictions[(predictions["feature_set"] == feature_set) & (predictions["model"] == model)]
        for split, marker, color in [
            ("train", "o", "#4C78A8"),
            ("primary_test", "s", "#E45756"),
            ("secondary_test", "^", "#F58518"),
        ]:
            subset = part[part["split"] == split]
            ax.scatter(subset["observed_lifetime"], subset["predicted_lifetime"], label=split, marker=marker, color=color, s=36)
        low = float(min(part["observed_lifetime"].min(), part["predicted_lifetime"].min()))
        high = float(max(part["observed_lifetime"].max(), part["predicted_lifetime"].max()))
        ax.plot([low, high], [low, high], color="#333333", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{feature_set}\n{model}")
        ax.set_xlabel("Observed cycle life")
        ax.set_ylabel("Predicted cycle life")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(model_keys) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "table1_linear_model_predictions.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="data/severson_features_20170630/severson_supplementary_table1_features_20170630.csv",
    )
    parser.add_argument("--output-dir", default="results/severson_20170630_table1_linear_models")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_mask = df["official_split"] == "train"
    primary_mask = df["official_split"] == "primary_test"
    secondary_mask = df["official_split"] == "secondary_test"
    primary_no_outlier_mask = primary_mask & (df["cell_id"] != "b2c1")
    clean4_mask = ~df["cell_id"].isin(FULL_MODEL_EXCLUDED_CELLS)

    metrics_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    model_metadata: dict[str, object] = {}

    for feature_set, feature_cols in FEATURE_SETS.items():
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for {feature_set}: {missing_cols}")

        model_names = ["linear"] if feature_set == "variance" else ["elastic_net_cv"]
        x = df[feature_cols].astype(float)
        y_cycles = df["lifetime"].astype(float).to_numpy()
        y_log = np.log10(y_cycles)

        for model_name in model_names:
            model = build_model(model_name, args.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x.loc[train_mask], y_log[train_mask.to_numpy()])

            estimator = model.named_steps["model"]
            key = f"{feature_set}:{model_name}"
            model_metadata[key] = {
                "features": feature_cols,
                "alpha": float(getattr(estimator, "alpha_", np.nan)),
                "l1_ratio": float(getattr(estimator, "l1_ratio_", np.nan)),
                "nonzero_feature_count": int(
                    np.sum(np.abs(getattr(estimator, "coef_", np.asarray([]))) > 1e-10)
                ),
            }
            coef_rows.extend(coefficient_rows(model, feature_cols, model_name, feature_set))

            pred_log_all = model.predict(x)
            pred_cycles_all = np.power(10.0, pred_log_all)
            split_masks = [
                ("train", train_mask),
                ("primary_test", primary_mask),
                ("primary_test_no_b2c1", primary_no_outlier_mask),
                ("secondary_test", secondary_mask),
            ]
            if feature_set == "full_candidates":
                split_masks.extend(
                    [
                        ("train_clean4", train_mask & clean4_mask),
                        ("primary_test_clean4", primary_mask & clean4_mask),
                        ("secondary_test_clean4", secondary_mask & clean4_mask),
                    ]
                )
            for split_name, mask in split_masks:
                idx = mask.to_numpy()
                row = {
                    "feature_set": feature_set,
                    "model": model_name,
                    "split": split_name,
                    "n": int(idx.sum()),
                }
                row.update(metrics_for(y_cycles[idx], pred_log_all[idx]))
                metrics_rows.append(row)

            for _, row in df.iterrows():
                prediction_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "cell_id": row["cell_id"],
                        "split": row["official_split"],
                        "observed_lifetime": float(row["lifetime"]),
                        "predicted_lifetime": float(pred_cycles_all[row.name]),
                        "predicted_log10_lifetime": float(pred_log_all[row.name]),
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    coefs = pd.DataFrame(coef_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    coefs.to_csv(output_dir / "coefficients.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(model_metadata, fp, indent=2)
    plot_predictions(predictions, output_dir)

    print(f"Saved metrics to {output_dir / 'metrics.csv'}")
    print(metrics[["feature_set", "model", "split", "n", "rmse_cycles", "mape_percent"]].to_string(index=False))
    print(f"Saved coefficients to {output_dir / 'coefficients.csv'}")
    print(f"Saved predictions to {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()




