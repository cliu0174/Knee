from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_nested_contiguous_ldq as core


EXPECTED_COUNTS = {"train": 41, "primary_test": 42, "secondary_test": 40}

SELECTION_MODEL_GRID = [
    {**core.MODEL_GRID[0], "n_estimators": 200, "model_id": "et_compact_screen"},
    {**core.MODEL_GRID[1], "n_estimators": 500, "model_id": "et_anchor_screen"},
]


def selection_model_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    return SELECTION_MODEL_GRID[:1] if args.quick else SELECTION_MODEL_GRID


def final_model_config(screen_config: dict[str, Any]) -> dict[str, Any]:
    """Restore the prespecified high-precision tree count after capacity screening."""
    config = dict(screen_config)
    if config["model_id"] == "et_anchor_screen":
        config.update({"model_id": "et_formal_anchor", "n_estimators": 1200})
    else:
        config.update({"model_id": "et_compact", "n_estimators": 500})
    return config


def formal_selector_grid(method: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Use one prespecified structural setting per method unless explicitly expanded.

    With 41 development cells, selecting many adjacent top-k/band settings would
    spend most of the inner-CV budget on weakly identifiable hyperparameters.
    The formal comparison therefore fixes top-k at 40 (and one representative
    structural setting) while retaining the two prespecified model capacities.
    """
    return core.selector_grid(
        method,
        args.bootstrap_repeats,
        quick=(args.quick or not args.expanded_selector_grid),
    )


def load_formal_table(path: Path) -> tuple[pd.DataFrame, str, list[str], list[str], pd.DataFrame]:
    raw = pd.read_csv(path)
    excluded = raw.loc[~raw["knee_after_window"].astype(bool), [
        "cell_id", "official_split", "knee_cycle", "cycle_life"
    ]].copy()
    table = raw[raw["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    counts = table["official_split"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Expected eligible split counts {EXPECTED_COUNTS}, found {counts}")
    prefix = core.detect_window_prefix(table)
    base, local = core.base_and_local_features(table, prefix)
    if not base or not local:
        raise RuntimeError(f"Missing formal features: base={len(base)}, local={len(local)}")
    return table, prefix, base, local, excluded


def selector_development(
    train: pd.DataFrame,
    base: list[str],
    local: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = train["knee_cycle"].to_numpy(float)
    splits = core.split_indices(train["batch"], y, args.outer_splits, args.outer_repeats, args.random_state)
    model_grid = selection_model_grid(args)
    inner_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for outer_id, (tr, va) in enumerate(splits):
        outer_train = train.iloc[tr]
        outer_valid = train.iloc[va]
        for method in args.methods:
            selector_config, model_config, candidates = core.choose_config(
                outer_train,
                base,
                local,
                method,
                formal_selector_grid(method, args),
                model_grid,
                args.inner_splits,
                args.inner_repeats,
                args.random_state + outer_id * 100003,
                outer_id,
            )
            inner_rows.extend(candidates)
            selector = core.fit_selector(
                outer_train,
                y[tr],
                local,
                method,
                selector_config,
                args.random_state + outer_id * 100003 + 7919,
            )
            features = base + selector.selected
            x_train, medians = core.safe_numeric_matrix(outer_train, features)
            x_valid, _ = core.safe_numeric_matrix(outer_valid, features, medians)
            model = core.make_model(model_config, args.random_state + outer_id)
            started = time.perf_counter()
            model.fit(x_train, y[tr])
            fit_seconds = time.perf_counter() - started
            pred = np.asarray(model.predict(x_valid), float)
            fold_rows.append({
                "outer_id": outer_id,
                "outer_repeat": outer_id // args.outer_splits,
                "outer_fold": outer_id % args.outer_splits,
                "method": method,
                "n_train": len(tr),
                "n_valid": len(va),
                "selector_config": core.json_compact(selector_config),
                "model_id": model_config["model_id"],
                "feature_count": len(features),
                "local_feature_count": len(selector.selected),
                "fit_seconds": fit_seconds,
                "model_bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
                **selector.metadata,
                **core.regression_metrics(y[va], pred),
            })
            for idx, estimate in zip(va, pred):
                row = train.iloc[idx]
                pred_rows.append({
                    "outer_id": outer_id,
                    "outer_repeat": outer_id // args.outer_splits,
                    "outer_fold": outer_id % args.outer_splits,
                    "method": method,
                    "cell_id": row["cell_id"],
                    "batch": row["batch"],
                    "observed": float(row["knee_cycle"]),
                    "predicted": float(estimate),
                    "abs_error": abs(float(row["knee_cycle"]) - float(estimate)),
                })
            for rank, feature in enumerate(selector.selected, 1):
                selected_rows.append({
                    "outer_id": outer_id,
                    "method": method,
                    "rank": rank,
                    "feature": feature,
                    **(core.parse_region(feature) or {}),
                })
        print(f"formal outer split {outer_id + 1}/{len(splits)} complete", flush=True)

    inner = pd.DataFrame(inner_rows)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(pred_rows)
    selected = pd.DataFrame(selected_rows)
    summary = folds.groupby("method", as_index=False).agg(
        outer_mae_mean=("mae", "mean"),
        outer_mae_std=("mae", "std"),
        outer_mae_median=("mae", "median"),
        outer_mae_max=("mae", "max"),
        outer_rmse_mean=("rmse", "mean"),
        outer_r2_mean=("r2", "mean"),
        outer_q90_mean=("abs_error_q90", "mean"),
        feature_count_mean=("feature_count", "mean"),
    )
    summary["development_selection_score"] = (
        summary["outer_mae_mean"]
        + 0.10 * summary["outer_mae_std"]
        + 0.10 * summary["outer_mae_max"]
    )
    summary = summary.sort_values(["development_selection_score", "outer_mae_mean"]).reset_index(drop=True)
    return inner, folds, predictions, selected, summary


def freeze_and_evaluate(
    table: pd.DataFrame,
    base: list[str],
    local: list[str],
    method: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = table[table["official_split"] == "train"].copy().reset_index(drop=True)
    model_grid = selection_model_grid(args)
    selector_config, model_config, freeze_rows = core.choose_config(
        train,
        base,
        local,
        method,
        formal_selector_grid(method, args),
        model_grid,
        args.freeze_splits,
        args.freeze_repeats,
        args.random_state + 424242,
        "full_train",
    )
    selector = core.fit_selector(
        train,
        train["knee_cycle"].to_numpy(float),
        local,
        method,
        selector_config,
        args.random_state + 515151,
    )
    screen_model_config = dict(model_config)
    model_config = final_model_config(screen_model_config)
    features = base + selector.selected
    x_train, medians = core.safe_numeric_matrix(train, features)
    y_train = train["knee_cycle"].to_numpy(float)
    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for split in ["primary_test", "secondary_test"]:
        test = table[table["official_split"] == split].copy().reset_index(drop=True)
        x_test, _ = core.safe_numeric_matrix(test, features, medians)
        y_test = test["knee_cycle"].to_numpy(float)
        for seed in range(args.seed_start, args.seed_start + args.seed_repeats):
            model = core.make_model(model_config, seed)
            started = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - started
            pred = np.asarray(model.predict(x_test), float)
            metric_rows.append({
                "split": split,
                "seed": seed,
                "n": len(test),
                "fit_seconds": fit_seconds,
                "model_bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
                **core.regression_metrics(y_test, pred),
            })
            for idx, estimate in enumerate(pred):
                row = test.iloc[idx]
                pred_rows.append({
                    "split": split,
                    "seed": seed,
                    "cell_id": row["cell_id"],
                    "batch": row["batch"],
                    "observed": float(row["knee_cycle"]),
                    "predicted": float(estimate),
                    "abs_error": abs(float(row["knee_cycle"]) - float(estimate)),
                    "cycle_life": float(row["cycle_life"]),
                    "label_confidence": float(row["label_confidence"]),
                })
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(pred_rows)
    per_cell = predictions.groupby(
        ["split", "cell_id", "batch", "observed", "cycle_life", "label_confidence"], as_index=False
    ).agg(
        predicted_mean=("predicted", "mean"),
        predicted_std=("predicted", "std"),
        abs_error=("abs_error", "mean"),
    )
    frozen = {
        "phase": "frozen_on_train_before_any_held_out_evaluation",
        "method": method,
        "selector_config": selector_config,
        "selector_metadata": selector.metadata,
        "model_config": model_config,
        "capacity_screen_model_config": screen_model_config,
        "base_features": base,
        "selected_local_features": selector.selected,
        "n_train": len(train),
        "seed_list": list(range(args.seed_start, args.seed_start + args.seed_repeats)),
        "test_labels_used_for_selection": False,
    }
    return frozen, pd.DataFrame(freeze_rows), metrics, predictions, per_cell


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal leakage-safe 41/42/40 selector experiment")
    parser.add_argument("--feature-path", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--output-dir", default="results/formal_414240_selector_N150")
    parser.add_argument("--methods", nargs="+", default=[
        "pearson", "bootstrap_stable", "region_balanced", "contiguous", "mrmr"
    ])
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--inner-repeats", type=int, default=1)
    parser.add_argument("--freeze-splits", type=int, default=5)
    parser.add_argument("--freeze-repeats", type=int, default=2)
    parser.add_argument("--bootstrap-repeats", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=20260721)
    parser.add_argument("--seed-start", type=int, default=123)
    parser.add_argument("--seed-repeats", type=int, default=30)
    parser.add_argument(
        "--expanded-selector-grid",
        action="store_true",
        help="Explore adjacent top-k/band settings; not recommended for the 41-cell formal analysis.",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_path = Path(args.feature_path)
    table, prefix, base, local, excluded = load_formal_table(feature_path)
    train = table[table["official_split"] == "train"].copy().reset_index(drop=True)
    inner, folds, oof, selected, summary = selector_development(train, base, local, args)
    selected_method = str(summary.iloc[0]["method"])
    frozen, freeze_rows, metrics, predictions, per_cell = freeze_and_evaluate(
        table, base, local, selected_method, args
    )

    inner.to_csv(output / "development_inner_selection.csv", index=False)
    folds.to_csv(output / "development_outer_fold_metrics.csv", index=False)
    oof.to_csv(output / "development_oof_predictions.csv", index=False)
    selected.to_csv(output / "development_selected_features.csv", index=False)
    summary.to_csv(output / "development_summary.csv", index=False)
    core.jaccard_summary(selected).to_csv(output / "development_selection_jaccard.csv", index=False)
    core.paired_method_tests(oof, 5000, args.random_state + 31).to_csv(
        output / "development_paired_comparisons.csv", index=False
    )
    freeze_rows.to_csv(output / "freeze_inner_selection.csv", index=False)
    metrics.to_csv(output / "heldout_metrics_by_seed.csv", index=False)
    predictions.to_csv(output / "heldout_predictions_by_seed.csv", index=False)
    per_cell.to_csv(output / "heldout_per_cell.csv", index=False)
    excluded.to_csv(output / "eligibility_exclusions.csv", index=False)
    (output / "frozen_config.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    metadata = {
        "created_utc": core.utc_now(),
        "feature_path": str(feature_path),
        "feature_sha256": core.sha256_file(feature_path),
        "window_prefix": prefix,
        "eligible_counts": table["official_split"].value_counts().to_dict(),
        "n_base_features": len(base),
        "n_local_candidates": len(local),
        "outer_splits": args.outer_splits,
        "outer_repeats": args.outer_repeats,
        "inner_splits": args.inner_splits,
        "inner_repeats": args.inner_repeats,
        "selector_grid_policy": "expanded" if args.expanded_selector_grid else "one prespecified structural setting per method",
        "capacity_screen_grid": SELECTION_MODEL_GRID,
        "final_tree_policy": "500 trees for compact or 1200 trees for anchor after train-only capacity screening",
        "development_selection_score": "mean_outer_MAE + 0.10*std_outer_MAE + 0.10*worst_outer_MAE",
        "selected_method_by_train_only_nested_cv": selected_method,
        "heldout_opened_after_freeze": True,
        "software_versions": core.software_versions(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.round(3).to_string(index=False))
    print("\nHeld-out mean across seeds")
    print(metrics.groupby("split")[["mae", "rmse", "r2", "abs_error_q90"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
