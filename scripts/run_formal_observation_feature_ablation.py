from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_nested_contiguous_ldq as core
from run_feature_ablation import feature_bucket


SPLITS = ["primary_test", "secondary_test"]


def load_common_ids(reference: Path) -> set[str]:
    frame = pd.read_csv(reference, usecols=["cell_id", "knee_after_window"])
    return set(frame.loc[frame["knee_after_window"].astype(bool), "cell_id"].astype(str))


def load_window(path: Path, common_ids: set[str]) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    frame = pd.read_csv(path)
    frame = frame[frame["cell_id"].astype(str).isin(common_ids)].copy().reset_index(drop=True)
    counts = frame["official_split"].value_counts().to_dict()
    expected = {"train": 41, "primary_test": 42, "secondary_test": 40}
    if counts != expected:
        raise RuntimeError(f"Common-cohort counts for {path} are {counts}, expected {expected}")
    prefix = core.detect_window_prefix(frame)
    base, local = core.base_and_local_features(frame, prefix)
    return frame, prefix, base, local


def fit_features(
    train: pd.DataFrame,
    base: list[str],
    local: list[str],
    frozen: dict[str, Any],
    seed: int,
) -> tuple[list[str], list[str]]:
    selector = core.fit_selector(
        train,
        train["knee_cycle"].to_numpy(float),
        local,
        frozen["method"],
        frozen["selector_config"],
        seed,
    )
    return base + selector.selected, selector.selected


def observation_budget(
    feature_paths: list[Path],
    common_ids: set[str],
    frozen: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for path in feature_paths:
        frame, prefix, base, local = load_window(path, common_ids)
        train = frame[frame["official_split"] == "train"].copy().reset_index(drop=True)
        y = train["knee_cycle"].to_numpy(float)
        folds = core.split_indices(train["batch"], y, args.cv_splits, args.cv_repeats, args.random_state)
        for fold_id, (tr, va) in enumerate(folds):
            features, selected_local = fit_features(
                train.iloc[tr], base, local, frozen, args.random_state + 1009 * (fold_id + 1)
            )
            x_train, medians = core.safe_numeric_matrix(train.iloc[tr], features)
            x_valid, _ = core.safe_numeric_matrix(train.iloc[va], features, medians)
            model = core.make_model(frozen["model_config"], args.random_state + fold_id)
            model.fit(x_train, y[tr])
            pred = np.asarray(model.predict(x_valid), float)
            cv_rows.append({
                "window": prefix,
                "outer_id": fold_id,
                "n_train": len(tr),
                "n_valid": len(va),
                "feature_count": len(features),
                "local_feature_count": len(selected_local),
                **core.regression_metrics(y[va], pred),
            })

        features, selected_local = fit_features(train, base, local, frozen, args.random_state + 515151)
        x_train, medians = core.safe_numeric_matrix(train, features)
        y_train = train["knee_cycle"].to_numpy(float)
        for split in SPLITS:
            test = frame[frame["official_split"] == split].copy().reset_index(drop=True)
            x_test, _ = core.safe_numeric_matrix(test, features, medians)
            y_test = test["knee_cycle"].to_numpy(float)
            for seed in range(args.seed_start, args.seed_start + args.seed_repeats):
                model = core.make_model(frozen["model_config"], seed)
                model.fit(x_train, y_train)
                pred = np.asarray(model.predict(x_test), float)
                metric_rows.append({
                    "window": prefix,
                    "split": split,
                    "seed": seed,
                    "n": len(test),
                    "feature_count": len(features),
                    **core.regression_metrics(y_test, pred),
                })
                for idx, estimate in enumerate(pred):
                    prediction_rows.append({
                        "window": prefix,
                        "split": split,
                        "seed": seed,
                        "cell_id": test.iloc[idx]["cell_id"],
                        "observed": float(y_test[idx]),
                        "predicted": float(estimate),
                        "abs_error": abs(float(y_test[idx]) - float(estimate)),
                    })
        print(f"observation budget {prefix} complete", flush=True)
    return pd.DataFrame(cv_rows), pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def group_columns(frame: pd.DataFrame, prefix: str, selected_local: list[str]) -> dict[str, list[str]]:
    raw = [
        col for col in frame.columns
        if pd.api.types.is_numeric_dtype(frame[col]) and feature_bucket(col, prefix) == "raw"
        and frame[col].nunique(dropna=True) > 1
    ]
    global_deltaq = [
        col for col in frame.columns
        if pd.api.types.is_numeric_dtype(frame[col]) and feature_bucket(col, prefix) == "deltaq"
        and frame[col].nunique(dropna=True) > 1
    ]
    return {
        "raw": sorted(raw),
        "raw_global_deltaq": sorted(raw + global_deltaq),
        "raw_local_deltaq": sorted(raw + selected_local),
        "full_global_local": sorted(raw + global_deltaq + selected_local),
    }


def feature_ablation(
    feature_path: Path,
    common_ids: set[str],
    frozen: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    frame, prefix, base, local = load_window(feature_path, common_ids)
    train = frame[frame["official_split"] == "train"].copy().reset_index(drop=True)
    _, selected_local = fit_features(train, base, local, frozen, args.random_state + 515151)
    groups = group_columns(frame, prefix, selected_local)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for group, features in groups.items():
        x_train, medians = core.safe_numeric_matrix(train, features)
        y_train = train["knee_cycle"].to_numpy(float)
        for split in SPLITS:
            test = frame[frame["official_split"] == split].copy().reset_index(drop=True)
            x_test, _ = core.safe_numeric_matrix(test, features, medians)
            y_test = test["knee_cycle"].to_numpy(float)
            for seed in range(args.seed_start, args.seed_start + args.seed_repeats):
                model = core.make_model(frozen["model_config"], seed)
                model.fit(x_train, y_train)
                pred = np.asarray(model.predict(x_test), float)
                metric_rows.append({
                    "feature_group": group,
                    "split": split,
                    "seed": seed,
                    "n": len(test),
                    "feature_count": len(features),
                    **core.regression_metrics(y_test, pred),
                })
                for idx, estimate in enumerate(pred):
                    prediction_rows.append({
                        "feature_group": group,
                        "split": split,
                        "seed": seed,
                        "cell_id": test.iloc[idx]["cell_id"],
                        "observed": float(y_test[idx]),
                        "predicted": float(estimate),
                        "abs_error": abs(float(y_test[idx]) - float(estimate)),
                    })
        print(f"feature ablation {group} complete", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal observation-budget and feature-group ablations")
    parser.add_argument("--feature-paths", nargs="+", default=[
        "data/early_features_N50_local_deltaq_20170630.csv",
        "data/early_features_N100_local_deltaq_20170630.csv",
        "data/early_features_N150_local_deltaq_20170630.csv",
    ])
    parser.add_argument("--reference", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--frozen-config", default="results/formal_414240_selector_N150/frozen_config.json")
    parser.add_argument("--output-dir", default="results/formal_414240_observation_ablation")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=123)
    parser.add_argument("--seed-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=20260721)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = Path(args.frozen_config)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("phase") != "frozen_on_train_before_any_held_out_evaluation":
        raise RuntimeError("Formal frozen configuration is missing or invalid")
    common_ids = load_common_ids(Path(args.reference))
    paths = [Path(item) for item in args.feature_paths]
    cv, budget_metrics, budget_predictions = observation_budget(paths, common_ids, frozen, args)
    ablation_metrics, ablation_predictions, groups = feature_ablation(
        Path(args.reference), common_ids, frozen, args
    )
    cv.to_csv(output / "observation_budget_train_cv.csv", index=False)
    budget_metrics.to_csv(output / "observation_budget_heldout_metrics.csv", index=False)
    budget_predictions.to_csv(output / "observation_budget_heldout_predictions.csv", index=False)
    ablation_metrics.to_csv(output / "feature_ablation_heldout_metrics.csv", index=False)
    ablation_predictions.to_csv(output / "feature_ablation_heldout_predictions.csv", index=False)
    (output / "feature_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    metadata = {
        "created_utc": core.utc_now(),
        "common_cohort_reference": args.reference,
        "common_cohort_reference_sha256": core.sha256_file(Path(args.reference)),
        "frozen_config": args.frozen_config,
        "frozen_config_sha256": core.sha256_file(frozen_path),
        "eligible_counts": {"train": 41, "primary_test": 42, "secondary_test": 40},
        "seed_repeats": args.seed_repeats,
        "test_results_used_for_configuration": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nObservation budget held-out means")
    print(budget_metrics.groupby(["window", "split"])[["mae", "rmse", "r2"]].mean().round(3).to_string())
    print("\nFeature ablation held-out means")
    print(ablation_metrics.groupby(["feature_group", "split"])[["mae", "rmse", "r2"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
