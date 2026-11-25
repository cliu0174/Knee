"""Second-round, leakage-safe experiments for the Local DeltaQ knee predictor.

This script implements the experiments recorded in
``manuscript_ionics/second_round_revision_log_20260724.md`` without touching
the manuscript, existing formal results, or figure source data:

E1  Fair input-representation comparison.
E2  Training-median reference predictor.
E3  Regression-family comparison with the same fold-wise Local DeltaQ selector.
E4  Training-only sensitivity to the number of selected Local DeltaQ features.
E5  Prespecified Local DeltaQ window-design sensitivity.
E6  Empirical residual-interval calibration diagnostic for the E1 full model.

The script deliberately separates development from held-out evaluation.  All
feature selection, imputation, and hyperparameter selection use only the 41
official training cells.  The two held-out cohorts are read only after a
configuration has been frozen.  E4 is intentionally training-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import run_nested_contiguous_ldq as core
from run_feature_ablation import feature_bucket


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {"train": 41, "primary_test": 42, "secondary_test": 40}
TEST_SPLITS = ["primary_test", "secondary_test"]
SELECTOR_METHOD = "bootstrap_stable"

REPRESENTATIONS = ["raw_soh", "raw_soh_global", "raw_soh_local", "full_ldq_et"]
REPRESENTATION_LABELS = {
    "raw_soh": "Raw/SOH",
    "raw_soh_global": "Raw/SOH + Global DeltaQ",
    "raw_soh_local": "Raw/SOH + selected Local DeltaQ-derived",
    "full_ldq_et": "Raw/SOH + Global DeltaQ + selected Local DeltaQ-derived",
}

ET_GRID = [
    {
        "family": "extratrees",
        "config_id": "et_compact",
        "n_estimators": 500,
        "max_features": 0.5,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "max_depth": 6,
    },
    {
        "family": "extratrees",
        "config_id": "et_anchor",
        "n_estimators": 1200,
        "max_features": 0.5,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_depth": 6,
    },
]

MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "ridge": [
        {"family": "ridge", "config_id": f"ridge_alpha_{alpha:g}", "alpha": alpha}
        for alpha in (0.1, 1.0, 10.0, 100.0)
    ],
    "svr_rbf": [
        {
            "family": "svr_rbf",
            "config_id": f"svr_C_{c:g}_eps_{epsilon:g}",
            "C": c,
            "epsilon": epsilon,
            "gamma": "scale",
        }
        for c in (1.0, 10.0, 100.0)
        for epsilon in (5.0, 15.0)
    ],
    "random_forest": [
        {
            "family": "random_forest",
            "config_id": f"rf_leaf_{leaf}_mf_{str(max_features).replace('.', 'p')}",
            "n_estimators": 800,
            "min_samples_leaf": leaf,
            "min_samples_split": 2,
            "max_features": max_features,
            "max_depth": 6,
        }
        for leaf in (1, 2)
        for max_features in ("sqrt", 0.5)
    ],
    "extratrees": ET_GRID,
}


@dataclass(frozen=True)
class Schema:
    prefix: str
    raw: list[str]
    global_deltaq: list[str]
    local: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def software_versions() -> dict[str, str]:
    return {**core.software_versions(), "platform": platform.platform()}


def load_table(path: Path) -> tuple[pd.DataFrame, Schema]:
    table = pd.read_csv(path)
    table = table[table["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    counts = table["official_split"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Expected eligible split counts {EXPECTED_COUNTS}, found {counts} in {path}")
    prefix = core.detect_window_prefix(table)

    raw: list[str] = []
    global_deltaq: list[str] = []
    local: list[str] = []
    for column in table.columns:
        if not pd.api.types.is_numeric_dtype(table[column]) or table[column].nunique(dropna=True) <= 1:
            continue
        bucket = feature_bucket(column, prefix)
        if bucket == "raw":
            raw.append(column)
        elif bucket == "deltaq":
            global_deltaq.append(column)
        elif column.startswith(f"{prefix}_ldq_"):
            local.append(column)
    schema = Schema(prefix=prefix, raw=sorted(raw), global_deltaq=sorted(global_deltaq), local=sorted(local))
    if not (schema.raw and schema.global_deltaq and schema.local):
        raise RuntimeError(
            "The required Raw/SOH, Global DeltaQ, or Local DeltaQ blocks are missing: "
            f"raw={len(schema.raw)}, global={len(schema.global_deltaq)}, local={len(schema.local)}"
        )
    return table, schema


def needs_local_selection(representation: str) -> bool:
    return representation in {"raw_soh_local", "full_ldq_et"}


def feature_list(representation: str, schema: Schema, selected_local: list[str]) -> list[str]:
    if representation == "raw_soh":
        return schema.raw
    if representation == "raw_soh_global":
        return schema.raw + schema.global_deltaq
    if representation == "raw_soh_local":
        return schema.raw + selected_local
    if representation == "full_ldq_et":
        return schema.raw + schema.global_deltaq + selected_local
    raise ValueError(f"Unknown representation: {representation}")


def make_estimator(config: dict[str, Any], seed: int) -> object:
    family = config["family"]
    if family == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=float(config["alpha"])) )])
    if family == "svr_rbf":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(C=float(config["C"]), epsilon=float(config["epsilon"]), gamma=config["gamma"])),
        ])
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(config["n_estimators"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            min_samples_split=int(config["min_samples_split"]),
            max_features=config["max_features"],
            max_depth=config["max_depth"],
            random_state=int(seed),
            n_jobs=-1,
        )
    if family == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=int(config["n_estimators"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            min_samples_split=int(config["min_samples_split"]),
            max_features=config["max_features"],
            max_depth=config["max_depth"],
            bootstrap=False,
            random_state=int(seed),
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model family: {family}")


def selector_config(top_k: int, bootstrap_repeats: int) -> dict[str, Any]:
    config = dict(core.selector_grid(SELECTOR_METHOD, bootstrap_repeats, quick=True)[0])
    config["top_k"] = int(top_k)
    return config


def selected_local_features(
    train: pd.DataFrame,
    schema: Schema,
    top_k: int,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    selector = core.fit_selector(
        train,
        train["knee_cycle"].to_numpy(float),
        schema.local,
        SELECTOR_METHOD,
        selector_config(top_k, bootstrap_repeats),
        seed,
    )
    return selector.selected, selector.metadata


def fit_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    schema: Schema,
    representation: str,
    config: dict[str, Any],
    top_k: int,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    selected: list[str] = []
    selector_metadata: dict[str, Any] = {}
    if needs_local_selection(representation):
        selected, selector_metadata = selected_local_features(train, schema, top_k, bootstrap_repeats, seed + 719)
    features = feature_list(representation, schema, selected)
    x_train, medians = core.safe_numeric_matrix(train, features)
    x_valid, _ = core.safe_numeric_matrix(valid, features, medians)
    model = make_estimator(config, seed)
    model.fit(x_train, train["knee_cycle"].to_numpy(float))
    prediction = np.asarray(model.predict(x_valid), float)
    return prediction, features, selector_metadata


def choose_model_config(
    train: pd.DataFrame,
    schema: Schema,
    representation: str,
    candidate_configs: list[dict[str, Any]],
    top_k: int,
    bootstrap_repeats: int,
    inner_splits: int,
    inner_repeats: int,
    seed: int,
    run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a model configuration using only inner folds of ``train``."""
    y = train["knee_cycle"].to_numpy(float)
    splits = core.split_indices(train["batch"], y, inner_splits, inner_repeats, seed)
    rows: list[dict[str, Any]] = []
    for config_id, config in enumerate(candidate_configs):
        fold_mae: list[float] = []
        for fold_id, (tr, va) in enumerate(splits):
            pred, features, selector_metadata = fit_predict(
                train.iloc[tr],
                train.iloc[va],
                schema,
                representation,
                config,
                top_k,
                bootstrap_repeats,
                seed + config_id * 100_003 + fold_id * 1_009,
            )
            fold_mae.append(float(mean_absolute_error(y[va], pred)))
            rows.append({
                "run_id": run_id,
                "candidate_config": json_compact(config),
                "family": config["family"],
                "config_id": config["config_id"],
                "inner_fold": fold_id,
                "n_train": len(tr),
                "n_valid": len(va),
                "feature_count": len(features),
                "local_feature_count": sum(feature in schema.local for feature in features),
                "mae": fold_mae[-1],
                **{f"selector_{key}": value for key, value in selector_metadata.items()},
            })
        # The deterministic ranking below uses mean MAE, then dispersion and config id.
    summary = (
        pd.DataFrame(rows)
        .groupby(["candidate_config", "family", "config_id"], as_index=False)
        .agg(inner_mae_mean=("mae", "mean"), inner_mae_std=("mae", "std"), inner_mae_max=("mae", "max"))
        .sort_values(["inner_mae_mean", "inner_mae_std", "inner_mae_max", "config_id"])
        .reset_index(drop=True)
    )
    best_json = str(summary.iloc[0]["candidate_config"])
    best = next(config for config in candidate_configs if json_compact(config) == best_json)
    rows.extend({
        "run_id": run_id,
        "candidate_config": row.candidate_config,
        "family": row.family,
        "config_id": row.config_id,
        "inner_fold": "summary",
        "n_train": len(train),
        "n_valid": 0,
        "feature_count": np.nan,
        "local_feature_count": np.nan,
        "mae": row.inner_mae_mean,
        "inner_mae_std": row.inner_mae_std,
        "inner_mae_max": row.inner_mae_max,
        "selected": bool(row.candidate_config == best_json),
    } for row in summary.itertuples(index=False))
    return best, rows


def fold_rows_and_oof(
    train: pd.DataFrame,
    schema: Schema,
    representation: str,
    candidate_configs: list[dict[str, Any]],
    top_k: int,
    bootstrap_repeats: int,
    outer_splits: int,
    outer_repeats: int,
    inner_splits: int,
    inner_repeats: int,
    seed: int,
    experiment: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = train["knee_cycle"].to_numpy(float)
    splits = core.split_indices(train["batch"], y, outer_splits, outer_repeats, seed)
    inner_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for outer_id, (tr, va) in enumerate(splits):
        run_id = f"{experiment}_outer_{outer_id:02d}"
        config, rows = choose_model_config(
            train.iloc[tr], schema, representation, candidate_configs, top_k, bootstrap_repeats,
            inner_splits, inner_repeats, seed + 10_000 * (outer_id + 1), run_id,
        )
        inner_rows.extend(rows)
        pred, features, selector_metadata = fit_predict(
            train.iloc[tr], train.iloc[va], schema, representation, config,
            top_k, bootstrap_repeats, seed + 50_000 + outer_id,
        )
        metrics = core.regression_metrics(y[va], pred)
        fold_rows.append({
            "experiment": experiment,
            "representation": representation,
            "outer_id": outer_id,
            "outer_repeat": outer_id // outer_splits,
            "outer_fold": outer_id % outer_splits,
            "family": config["family"],
            "config_id": config["config_id"],
            "config": json_compact(config),
            "n_train": len(tr),
            "n_valid": len(va),
            "feature_count": len(features),
            "local_feature_count": sum(feature in schema.local for feature in features),
            **selector_metadata,
            **metrics,
        })
        selected = [feature for feature in features if feature in schema.local]
        for rank, feature in enumerate(selected, 1):
            selection_rows.append({"experiment": experiment, "representation": representation, "outer_id": outer_id, "rank": rank, "feature": feature})
        for index, estimate in zip(va, pred):
            row = train.iloc[index]
            prediction_rows.append({
                "experiment": experiment,
                "representation": representation,
                "outer_id": outer_id,
                "cell_id": row["cell_id"],
                "observed": float(row["knee_cycle"]),
                "predicted": float(estimate),
                "abs_error": abs(float(row["knee_cycle"]) - float(estimate)),
            })
        print(f"{experiment}: outer fold {outer_id + 1}/{len(splits)} complete", flush=True)
    return pd.DataFrame(inner_rows), pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows)


def freeze_and_evaluate(
    table: pd.DataFrame,
    schema: Schema,
    representation: str,
    candidate_configs: list[dict[str, Any]],
    top_k: int,
    bootstrap_repeats: int,
    inner_splits: int,
    inner_repeats: int,
    seed: int,
    seed_repeats: int,
    experiment: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = table[table["official_split"] == "train"].copy().reset_index(drop=True)
    config, inner_rows = choose_model_config(
        train, schema, representation, candidate_configs, top_k, bootstrap_repeats,
        inner_splits, inner_repeats, seed + 700_001, f"{experiment}_freeze",
    )
    selected: list[str] = []
    selector_metadata: dict[str, Any] = {}
    if needs_local_selection(representation):
        selected, selector_metadata = selected_local_features(train, schema, top_k, bootstrap_repeats, seed + 700_701)
    features = feature_list(representation, schema, selected)
    x_train, medians = core.safe_numeric_matrix(train, features)
    y_train = train["knee_cycle"].to_numpy(float)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for split in TEST_SPLITS:
        test = table[table["official_split"] == split].copy().reset_index(drop=True)
        x_test, _ = core.safe_numeric_matrix(test, features, medians)
        y_test = test["knee_cycle"].to_numpy(float)
        for model_seed in range(seed, seed + seed_repeats):
            model = make_estimator(config, model_seed)
            model.fit(x_train, y_train)
            pred = np.asarray(model.predict(x_test), float)
            metric_rows.append({
                "experiment": experiment,
                "representation": representation,
                "split": split,
                "seed": model_seed,
                "family": config["family"],
                "config_id": config["config_id"],
                "feature_count": len(features),
                "local_feature_count": len(selected),
                **core.regression_metrics(y_test, pred),
            })
            for idx, estimate in enumerate(pred):
                row = test.iloc[idx]
                prediction_rows.append({
                    "experiment": experiment,
                    "representation": representation,
                    "split": split,
                    "seed": model_seed,
                    "cell_id": row["cell_id"],
                    "observed": float(row["knee_cycle"]),
                    "predicted": float(estimate),
                    "abs_error": abs(float(row["knee_cycle"]) - float(estimate)),
                })
    frozen = {
        "experiment": experiment,
        "representation": representation,
        "phase": "frozen_on_41_training_cells_before_heldout_evaluation",
        "model_config": config,
        "selector_method": "bootstrap_stable" if needs_local_selection(representation) else None,
        "selector_config": selector_config(top_k, bootstrap_repeats) if needs_local_selection(representation) else None,
        "selector_metadata": selector_metadata,
        "feature_count": len(features),
        "base_feature_count": len(features) - len(selected),
        "selected_local_features": selected,
        "heldout_labels_used_for_selection": False,
        "seed_list": list(range(seed, seed + seed_repeats)),
    }
    return frozen, pd.DataFrame(inner_rows), pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def summarize_predictions(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        predictions.groupby(group_columns + ["cell_id", "observed"], as_index=False)
        .agg(predicted_mean=("predicted", "mean"), predicted_std=("predicted", "std"), abs_error=("abs_error", "mean"))
    )


def jaccard_by_group(selected: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if selected.empty:
        return pd.DataFrame(records)
    for keys, group in selected.groupby(group_columns):
        sets = [set(part["feature"]) for _, part in group.groupby("outer_id")]
        values: list[float] = []
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                union = sets[left] | sets[right]
                values.append(len(sets[left] & sets[right]) / len(union) if union else 1.0)
        item = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        item.update({"pair_count": len(values), "jaccard_mean": float(np.mean(values)), "jaccard_std": float(np.std(values))})
        records.append(item)
    return pd.DataFrame(records)


def run_e1(table: pd.DataFrame, schema: Schema, output: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E1: independently develop each input representation on the 41-cell cohort."""
    train = table[table["official_split"] == "train"].copy().reset_index(drop=True)
    all_inner: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    all_oof: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    frozen_rows: dict[str, Any] = {}
    all_freeze_inner: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    for offset, representation in enumerate(REPRESENTATIONS):
        inner, folds, oof, selected = fold_rows_and_oof(
            train, schema, representation, ET_GRID, args.top_k, args.bootstrap_repeats,
            args.outer_splits, args.outer_repeats, args.inner_splits, args.inner_repeats,
            args.random_state + 100_000 * offset, "E1_fair_representation",
        )
        all_inner.append(inner); all_folds.append(folds); all_oof.append(oof); all_selected.append(selected)
        frozen, freeze_inner, metrics, predictions = freeze_and_evaluate(
            table, schema, representation, ET_GRID, args.top_k, args.bootstrap_repeats,
            args.inner_splits, args.inner_repeats, args.random_state + 100_000 * offset,
            args.seed_repeats, "E1_fair_representation",
        )
        frozen_rows[representation] = frozen
        all_freeze_inner.append(freeze_inner); all_metrics.append(metrics); all_predictions.append(predictions)
    inner = pd.concat(all_inner, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    oof = pd.concat(all_oof, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    freeze_inner = pd.concat(all_freeze_inner, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    per_cell = summarize_predictions(predictions, ["experiment", "representation", "split"])
    summary = folds.groupby("representation", as_index=False).agg(
        train_cv_mae_mean=("mae", "mean"), train_cv_mae_std=("mae", "std"), train_cv_rmse_mean=("rmse", "mean"),
        train_cv_feature_count_mean=("feature_count", "mean"),
    )
    held = metrics.groupby(["representation", "split"], as_index=False).agg(
        heldout_mae_mean=("mae", "mean"), heldout_mae_std=("mae", "std"), heldout_rmse_mean=("rmse", "mean"),
        heldout_r2_mean=("r2", "mean"), heldout_q90_mean=("abs_error_q90", "mean"), feature_count=("feature_count", "first"),
    )
    held_wide = held.pivot(index="representation", columns="split")
    held_wide.columns = [f"{metric}_{split}" for metric, split in held_wide.columns]
    summary = summary.merge(held_wide.reset_index(), on="representation", how="left")
    output.mkdir(parents=True, exist_ok=True)
    inner.to_csv(output / "e1_inner_selection.csv", index=False)
    folds.to_csv(output / "e1_train_nested_cv_folds.csv", index=False)
    oof.to_csv(output / "e1_train_oof_predictions.csv", index=False)
    selected.to_csv(output / "e1_selected_local_features.csv", index=False)
    jaccard_by_group(selected, ["representation"]).to_csv(output / "e1_selection_jaccard.csv", index=False)
    freeze_inner.to_csv(output / "e1_freeze_inner_selection.csv", index=False)
    metrics.to_csv(output / "e1_heldout_metrics_by_seed.csv", index=False)
    predictions.to_csv(output / "e1_heldout_predictions_by_seed.csv", index=False)
    per_cell.to_csv(output / "e1_heldout_per_cell.csv", index=False)
    summary.to_csv(output / "e1_summary.csv", index=False)
    (output / "e1_frozen_configs.json").write_text(json.dumps(frozen_rows, indent=2), encoding="utf-8")
    return oof, per_cell


def run_e2(table: pd.DataFrame, output: Path, args: argparse.Namespace) -> None:
    """E2: a feature-free reference based on the 41-cell training median."""
    train = table[table["official_split"] == "train"].copy().reset_index(drop=True)
    median = float(train["knee_cycle"].median())
    y = train["knee_cycle"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(core.split_indices(train["batch"], y, args.outer_splits, args.outer_repeats, args.random_state + 901)):
        estimate = float(np.median(y[tr]))
        pred = np.full(len(va), estimate)
        rows.append({"split": "train_oof", "outer_id": fold_id, "n": len(va), "training_median": estimate, **core.regression_metrics(y[va], pred)})
        for index in va:
            oof_rows.append({"split": "train_oof", "outer_id": fold_id, "cell_id": train.iloc[index]["cell_id"], "observed": float(y[index]), "predicted": estimate, "abs_error": abs(float(y[index]) - estimate)})
    prediction_rows: list[dict[str, Any]] = []
    for split in TEST_SPLITS:
        test = table[table["official_split"] == split].copy().reset_index(drop=True)
        pred = np.full(len(test), median)
        rows.append({"split": split, "outer_id": "frozen", "n": len(test), "training_median": median, **core.regression_metrics(test["knee_cycle"].to_numpy(float), pred)})
        for _, row in test.iterrows():
            prediction_rows.append({"split": split, "cell_id": row["cell_id"], "observed": float(row["knee_cycle"]), "predicted": median, "abs_error": abs(float(row["knee_cycle"]) - median)})
    pd.DataFrame(rows).to_csv(output / "e2_training_median_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(output / "e2_training_median_oof_predictions.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "e2_training_median_heldout_predictions.csv", index=False)
    (output / "e2_training_median_metadata.json").write_text(json.dumps({"training_median": median, "n_train": len(train)}, indent=2), encoding="utf-8")


def run_e3(table: pd.DataFrame, schema: Schema, output: Path, args: argparse.Namespace) -> None:
    """E3: compare four regressor families using the same Local DeltaQ pipeline."""
    train = table[table["official_split"] == "train"].copy().reset_index(drop=True)
    inner_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    oof_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    freeze_inner_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    frozen_configs: dict[str, Any] = {}
    for offset, family in enumerate(["ridge", "svr_rbf", "random_forest", "extratrees"]):
        experiment = f"E3_{family}"
        candidate_configs = MODEL_GRIDS[family][:1] if args.quick else MODEL_GRIDS[family]
        inner, folds, oof, selected = fold_rows_and_oof(
            train, schema, "full_ldq_et", candidate_configs, args.top_k, args.bootstrap_repeats,
            args.outer_splits, args.outer_repeats, args.inner_splits, args.inner_repeats,
            args.random_state + 2_000_000 + offset * 100_000, experiment,
        )
        frozen, freeze_inner, metrics, predictions = freeze_and_evaluate(
            table, schema, "full_ldq_et", candidate_configs, args.top_k, args.bootstrap_repeats,
            args.inner_splits, args.inner_repeats, args.random_state + 2_000_000 + offset * 100_000,
            args.seed_repeats, experiment,
        )
        inner_parts.append(inner); fold_parts.append(folds); oof_parts.append(oof); selected_parts.append(selected)
        freeze_inner_parts.append(freeze_inner); metric_parts.append(metrics); prediction_parts.append(predictions)
        frozen_configs[family] = frozen
    inner = pd.concat(inner_parts, ignore_index=True)
    folds = pd.concat(fold_parts, ignore_index=True)
    oof = pd.concat(oof_parts, ignore_index=True)
    selected = pd.concat(selected_parts, ignore_index=True)
    metrics = pd.concat(metric_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_cell = summarize_predictions(predictions, ["experiment", "representation", "split"])
    summary = folds.groupby("family", as_index=False).agg(train_cv_mae_mean=("mae", "mean"), train_cv_mae_std=("mae", "std"), train_cv_rmse_mean=("rmse", "mean"))
    held = metrics.groupby(["family", "split"], as_index=False).agg(heldout_mae_mean=("mae", "mean"), heldout_mae_std=("mae", "std"), heldout_rmse_mean=("rmse", "mean"), heldout_r2_mean=("r2", "mean"), heldout_q90_mean=("abs_error_q90", "mean"))
    held_wide = held.pivot(index="family", columns="split")
    held_wide.columns = [f"{metric}_{split}" for metric, split in held_wide.columns]
    summary = summary.merge(held_wide.reset_index(), on="family", how="left")
    inner.to_csv(output / "e3_inner_selection.csv", index=False)
    folds.to_csv(output / "e3_train_nested_cv_folds.csv", index=False)
    oof.to_csv(output / "e3_train_oof_predictions.csv", index=False)
    selected.to_csv(output / "e3_selected_local_features.csv", index=False)
    jaccard_by_group(selected, ["experiment"]).to_csv(output / "e3_selection_jaccard.csv", index=False)
    pd.concat(freeze_inner_parts, ignore_index=True).to_csv(output / "e3_freeze_inner_selection.csv", index=False)
    metrics.to_csv(output / "e3_heldout_metrics_by_seed.csv", index=False)
    predictions.to_csv(output / "e3_heldout_predictions_by_seed.csv", index=False)
    per_cell.to_csv(output / "e3_heldout_per_cell.csv", index=False)
    summary.to_csv(output / "e3_summary.csv", index=False)
    (output / "e3_frozen_configs.json").write_text(json.dumps(frozen_configs, indent=2), encoding="utf-8")


def run_e4(table: pd.DataFrame, schema: Schema, output: Path, args: argparse.Namespace) -> None:
    """E4: training-only K sensitivity; held-out cohorts are never read here."""
    train = table[table["official_split"] == "train"].copy().reset_index(drop=True)
    y = train["knee_cycle"].to_numpy(float)
    folds = core.split_indices(train["batch"], y, args.outer_splits, args.outer_repeats, args.random_state + 3_000_000)
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    # The existing formal anchor is fixed here; K is the only varied factor.
    anchor = ET_GRID[1]
    for top_k in args.k_values:
        for fold_id, (tr, va) in enumerate(folds):
            pred, features, _ = fit_predict(train.iloc[tr], train.iloc[va], schema, "full_ldq_et", anchor, top_k, args.bootstrap_repeats, args.random_state + top_k * 1000 + fold_id)
            rows.append({"top_k": top_k, "outer_id": fold_id, "n_train": len(tr), "n_valid": len(va), "feature_count": len(features), **core.regression_metrics(y[va], pred)})
            selected, _ = selected_local_features(train.iloc[tr], schema, top_k, args.bootstrap_repeats, args.random_state + top_k * 2000 + fold_id)
            for rank, feature in enumerate(selected, 1):
                selected_rows.append({"top_k": top_k, "outer_id": fold_id, "rank": rank, "feature": feature})
        print(f"E4: K={top_k} complete", flush=True)
    frame = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_rows)
    summary = frame.groupby("top_k", as_index=False).agg(train_cv_mae_mean=("mae", "mean"), train_cv_mae_std=("mae", "std"), train_cv_mae_max=("mae", "max"), train_cv_rmse_mean=("rmse", "mean"), train_cv_q90_mean=("abs_error_q90", "mean"))
    frame.to_csv(output / "e4_k_sensitivity_train_cv_folds.csv", index=False)
    selected.to_csv(output / "e4_k_sensitivity_selected_features.csv", index=False)
    jaccard_by_group(selected, ["top_k"]).to_csv(output / "e4_k_sensitivity_jaccard.csv", index=False)
    summary.to_csv(output / "e4_k_sensitivity_summary.csv", index=False)


def make_window_design_tables(output: Path, args: argparse.Namespace) -> dict[str, Path]:
    """Derive prespecified local-window feature tables without changing shared data."""
    designs = {
        "non_overlap_W50_H50": (50, 50),
        "current_W50_H25": (50, 25),
        "wide_W100_H50": (100, 50),
    }
    derived = output / "e5_window_design_feature_tables"
    derived.mkdir(parents=True, exist_ok=True)
    extractor = ROOT / "scripts" / "extract_local_deltaq_features.py"
    base = ROOT / "data" / "early_features_N150_extended_20170630.csv"
    pkl_dir = ROOT / "data" / "severson_raw" / "pkl_20170630"
    paths: dict[str, Path] = {}
    for name, (width, stride) in designs.items():
        target = derived / f"{name}.csv"
        paths[name] = target
        if target.exists() and not args.regenerate_window_features:
            continue
        command = [
            sys.executable, str(extractor), "--base-features", str(base), "--pkl-dir", str(pkl_dir),
            "--output", str(target), "--qc-output", str(derived / f"{name}_qc.csv"),
            "--window", "150", "--cycle-a", "9", "--cycle-b", "149",
            "--segment-length", str(width), "--segment-stride", str(stride),
        ]
        subprocess.run(command, check=True, cwd=ROOT)
    return paths


def run_e5(output: Path, args: argparse.Namespace) -> None:
    """E5: fixed, prespecified window designs; no design is chosen from tests."""
    tables = make_window_design_tables(output, args)
    all_cv: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    configs: dict[str, Any] = {}
    for offset, (design, path) in enumerate(tables.items()):
        table, schema = load_table(path)
        train = table[table["official_split"] == "train"].copy().reset_index(drop=True)
        # No model or window is selected from test performance.  The formal anchor
        # and bootstrap-stable top-40 selector are fixed for this diagnostic.
        anchor = ET_GRID[1]
        y = train["knee_cycle"].to_numpy(float)
        splits = core.split_indices(train["batch"], y, args.outer_splits, args.outer_repeats, args.random_state + 4_000_000 + offset)
        cv_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        for fold_id, (tr, va) in enumerate(splits):
            pred, features, _ = fit_predict(train.iloc[tr], train.iloc[va], schema, "full_ldq_et", anchor, args.top_k, args.bootstrap_repeats, args.random_state + 4_100_000 + offset * 10_000 + fold_id)
            cv_rows.append({"design": design, "outer_id": fold_id, "feature_count": len(features), "local_candidate_count": len(schema.local), **core.regression_metrics(y[va], pred)})
            selected, _ = selected_local_features(train.iloc[tr], schema, args.top_k, args.bootstrap_repeats, args.random_state + 4_200_000 + offset * 10_000 + fold_id)
            for rank, feature in enumerate(selected, 1):
                selected_rows.append({"design": design, "outer_id": fold_id, "rank": rank, "feature": feature})
        selected, selector_meta = selected_local_features(train, schema, args.top_k, args.bootstrap_repeats, args.random_state + 4_300_000 + offset)
        features = feature_list("full_ldq_et", schema, selected)
        x_train, medians = core.safe_numeric_matrix(train, features)
        seed_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        for split in TEST_SPLITS:
            test = table[table["official_split"] == split].copy().reset_index(drop=True)
            x_test, _ = core.safe_numeric_matrix(test, features, medians)
            y_test = test["knee_cycle"].to_numpy(float)
            for model_seed in range(args.random_state + 4_400_000 + offset * 10_000, args.random_state + 4_400_000 + offset * 10_000 + args.seed_repeats):
                model = make_estimator(anchor, model_seed)
                model.fit(x_train, y)
                pred = np.asarray(model.predict(x_test), float)
                seed_rows.append({"design": design, "split": split, "seed": model_seed, "feature_count": len(features), "local_candidate_count": len(schema.local), **core.regression_metrics(y_test, pred)})
                for idx, estimate in enumerate(pred):
                    prediction_rows.append({"design": design, "split": split, "seed": model_seed, "cell_id": test.iloc[idx]["cell_id"], "observed": float(y_test[idx]), "predicted": float(estimate), "abs_error": abs(float(y_test[idx]) - float(estimate))})
        all_cv.append(pd.DataFrame(cv_rows)); all_metrics.append(pd.DataFrame(seed_rows)); all_predictions.append(pd.DataFrame(prediction_rows)); all_selected.append(pd.DataFrame(selected_rows))
        configs[design] = {"feature_path": str(path), "feature_sha256": sha256_file(path), "selector_config": selector_config(args.top_k, args.bootstrap_repeats), "selector_metadata": selector_meta, "model_config": anchor, "selected_local_features": selected, "heldout_labels_used_for_selection": False}
        print(f"E5: {design} complete", flush=True)
    cv = pd.concat(all_cv, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    cv.to_csv(output / "e5_window_design_train_cv_folds.csv", index=False)
    metrics.to_csv(output / "e5_window_design_heldout_metrics_by_seed.csv", index=False)
    predictions.to_csv(output / "e5_window_design_heldout_predictions_by_seed.csv", index=False)
    summarize_predictions(predictions, ["design", "split"]).to_csv(output / "e5_window_design_heldout_per_cell.csv", index=False)
    selected.to_csv(output / "e5_window_design_selected_features.csv", index=False)
    jaccard_by_group(selected, ["design"]).to_csv(output / "e5_window_design_jaccard.csv", index=False)
    cv.groupby("design", as_index=False).agg(train_cv_mae_mean=("mae", "mean"), train_cv_mae_std=("mae", "std"), train_cv_rmse_mean=("rmse", "mean")).to_csv(output / "e5_window_design_summary.csv", index=False)
    (output / "e5_window_design_frozen_configs.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")


def run_e6(e1_oof: pd.DataFrame, e1_per_cell: pd.DataFrame, output: Path) -> None:
    """E6: empirical residual interval diagnostic for E1's frozen full representation."""
    oof = e1_oof[e1_oof["representation"] == "full_ldq_et"].copy()
    calibration = oof.groupby(["cell_id", "observed"], as_index=False).agg(predicted=("predicted", "mean"))
    residuals = np.abs(calibration["observed"].to_numpy(float) - calibration["predicted"].to_numpy(float))
    alpha = 0.10
    quantile_level = min(1.0, np.ceil((len(residuals) + 1) * (1.0 - alpha)) / len(residuals))
    half_width = float(np.quantile(residuals, quantile_level, method="higher"))
    intervals = e1_per_cell[e1_per_cell["representation"] == "full_ldq_et"].copy()
    intervals["interval_lower"] = intervals["predicted_mean"] - half_width
    intervals["interval_upper"] = intervals["predicted_mean"] + half_width
    intervals["interval_width"] = 2.0 * half_width
    intervals["covered"] = (intervals["observed"] >= intervals["interval_lower"]) & (intervals["observed"] <= intervals["interval_upper"])
    rows: list[dict[str, Any]] = []
    for split, group in intervals.groupby("split"):
        covered = int(group["covered"].sum())
        rows.append({
            "split": split,
            "n": len(group),
            "covered": covered,
            "empirical_coverage": covered / len(group),
            "nominal_coverage": 1.0 - alpha,
            "mean_interval_width": float(group["interval_width"].mean()),
            "calibration_cells": len(calibration),
            "calibration_quantile_level": quantile_level,
            "calibration_half_width": half_width,
            "interpretation": "cross-fitted residual interval; empirical diagnostic, not a batch-shift coverage guarantee",
        })
    calibration.assign(abs_residual=residuals).to_csv(output / "e6_oof_calibration_residuals.csv", index=False)
    intervals.to_csv(output / "e6_prediction_intervals.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "e6_interval_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prespecified second-round LDQ experiments.")
    parser.add_argument("--feature-path", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--output-dir", default="results/second_round_experiments_20260724")
    parser.add_argument("--experiments", nargs="+", choices=["e1", "e2", "e3", "e4", "e5", "e6", "all"], default=["all"])
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--inner-repeats", type=int, default=1)
    parser.add_argument("--bootstrap-repeats", type=int, default=40)
    parser.add_argument(
        "--selector-method",
        choices=["pearson", "bootstrap_stable", "region_balanced", "contiguous", "mrmr"],
        default="bootstrap_stable",
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--k-values", type=int, nargs="+", default=[20, 40, 60])
    parser.add_argument("--seed-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=20260724)
    parser.add_argument("--regenerate-window-features", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Smoke test with fewer folds, selector resamples, and seeds.")
    args = parser.parse_args()
    if args.quick:
        args.outer_splits = 2; args.outer_repeats = 1; args.inner_splits = 2; args.inner_repeats = 1
        args.bootstrap_repeats = min(args.bootstrap_repeats, 4); args.seed_repeats = 1
    return args


def main() -> None:
    global SELECTOR_METHOD
    args = parse_args()
    SELECTOR_METHOD = args.selector_method
    experiments = {"e1", "e2", "e3", "e4", "e5", "e6"} if "all" in args.experiments else set(args.experiments)
    if "e6" in experiments:
        experiments.add("e1")
    output = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_path = (ROOT / args.feature_path).resolve() if not Path(args.feature_path).is_absolute() else Path(args.feature_path)
    table, schema = load_table(feature_path)
    metadata = {
        "created_utc": utc_now(),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "eligible_counts": table["official_split"].value_counts().to_dict(),
        "raw_feature_count": len(schema.raw),
        "global_deltaq_feature_count": len(schema.global_deltaq),
        "local_deltaq_candidate_count": len(schema.local),
        "experiments_requested": sorted(experiments),
        "protocol": "All fitted preprocessing, selector fitting, and model selection use only the 41 official training cells. Held-out results never choose a representation, model, K, or window design.",
        "software_versions": software_versions(),
        "arguments": vars(args),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    e1_oof = e1_per_cell = None
    if "e1" in experiments:
        e1_oof, e1_per_cell = run_e1(table, schema, output, args)
    if "e2" in experiments:
        run_e2(table, output, args)
    if "e3" in experiments:
        run_e3(table, schema, output, args)
    if "e4" in experiments:
        run_e4(table, schema, output, args)
    if "e5" in experiments:
        run_e5(output, args)
    if "e6" in experiments:
        if e1_oof is None or e1_per_cell is None:
            e1_oof = pd.read_csv(output / "e1_train_oof_predictions.csv")
            e1_per_cell = pd.read_csv(output / "e1_heldout_per_cell.csv")
        run_e6(e1_oof, e1_per_cell, output)
    print(f"Second-round outputs written to: {output}")


if __name__ == "__main__":
    main()
