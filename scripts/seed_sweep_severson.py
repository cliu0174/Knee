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
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reproduce_severson_baseline import evaluate_predictions, load_dataset, select_feature_window


def build_lightweight_models(random_state: int) -> dict[str, object]:
    return {
        "ridge_cv": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 41))),
            ]
        ),
        "elastic_net_cv_light": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.05, 0.1, 0.5],
                        alphas=np.logspace(-4, 0, 18),
                        cv=3,
                        max_iter=30000,
                        tol=1e-5,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def run_one_seed(
    x: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    seed: int,
) -> list[dict[str, object]]:
    y_log = np.log10(y.to_numpy())
    x_train, x_test, y_train_log, _y_test_log, y_train_cycles, y_test_cycles = train_test_split(
        x,
        y_log,
        y.to_numpy(),
        test_size=0.2,
        random_state=seed,
    )
    _ = y_train_cycles
    clip_bounds = (float(y_train_log.min() - 0.1), float(y_train_log.max() + 0.1))
    rows: list[dict[str, object]] = []
    for model_name, model in build_lightweight_models(seed).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train_log)
        pred_log = model.predict(x_test)
        metric = evaluate_predictions(
            dataset_name,
            model_name,
            f"holdout_seed_{seed}",
            y_test_cycles,
            pred_log,
            clip_bounds,
        )
        row = vars(metric).copy()
        row["seed"] = seed
        rows.append(row)
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["dataset", "model"], as_index=False)
    return grouped.agg(
        n_runs=("seed", "count"),
        mape_mean=("mape_percent", "mean"),
        mape_std=("mape_percent", "std"),
        mape_min=("mape_percent", "min"),
        mape_median=("mape_percent", "median"),
        mape_max=("mape_percent", "max"),
        mae_mean=("mae_cycles", "mean"),
        rmse_mean=("rmse_cycles", "mean"),
        r2_mean=("r2_cycles", "mean"),
        clipped_total=("clipped_predictions", "sum"),
    )


def plot_seed_sweep(results: pd.DataFrame, output_dir: Path) -> None:
    labels: list[str] = []
    values: list[np.ndarray] = []
    for (dataset, model), group in results.groupby(["dataset", "model"]):
        labels.append(f"{dataset}\n{model}")
        values.append(group["mape_percent"].to_numpy())

    fig, ax = plt.subplots(figsize=(max(8, len(values) * 1.9), 5.2))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("100-cycle holdout stability across random seeds")
    ax.grid(axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(output_dir / "seed_sweep_mape_boxplot.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/severson_processed")
    parser.add_argument("--output-dir", default="results/severson_seed_sweep")
    parser.add_argument("--window-index", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    all_rows: list[dict[str, object]] = []
    for csv_path in [data_dir / "original.csv", data_dir / "regularized.csv"]:
        dataset_name = f"{csv_path.stem}_window{args.window_index}"
        x, y = load_dataset(csv_path)
        x = select_feature_window(x, args.window_index)
        for seed in seeds:
            all_rows.extend(run_one_seed(x, y, dataset_name, seed))

    results = pd.DataFrame(all_rows)
    summary = summarize(results)
    results.to_csv(output_dir / "seed_sweep_metrics.csv", index=False)
    summary.to_csv(output_dir / "seed_sweep_summary.csv", index=False)
    plot_seed_sweep(results, output_dir)
    with open(output_dir / "seed_sweep_config.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "data_dir": str(data_dir),
                "window_index": args.window_index,
                "seeds": seeds,
                "models": sorted(results["model"].unique().tolist()),
            },
            fp,
            indent=2,
        )
    print(summary.sort_values(["dataset", "mape_mean"]).to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
