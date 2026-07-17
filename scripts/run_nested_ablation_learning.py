from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_feature_ablation import feature_bucket  # noqa: E402
from run_nested_contiguous_ldq import (  # noqa: E402
    MODEL_GRID,
    fit_selector,
    make_model,
    parse_region,
    safe_numeric_matrix,
    split_indices,
)
from train_knee_baselines import META_COLUMNS, build_models, detect_window_prefix  # noqa: E402


GROUPS = {
    "raw_only": (True, False, False),
    "global_deltaq_only": (False, True, False),
    "local_deltaq_only": (False, False, True),
    "raw_global_deltaq": (True, True, False),
    "raw_local_deltaq": (True, False, True),
    "global_local_deltaq": (False, True, True),
    "raw_global_local_deltaq": (True, True, True),
}


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mape_percent": float(np.mean(np.abs((y - pred) / y)) * 100.0),
        "r2": float(r2_score(y, pred)),
    }


def load_data(path: Path) -> tuple[pd.DataFrame, str, list[str], list[str], list[str]]:
    frame = pd.read_csv(path)
    frame = frame[frame["knee_after_window"].astype(bool)].reset_index(drop=True)
    frame = frame[frame["official_split"].isin(["train", "primary_test"])].reset_index(drop=True)
    if len(frame) != 83:
        raise RuntimeError(f"Expected 83 development cells, found {len(frame)}")
    prefix = detect_window_prefix(frame)
    raw: list[str] = []
    global_deltaq: list[str] = []
    local: list[str] = []
    for column in frame.columns:
        if column in META_COLUMNS or column == f"{prefix}_window_complete":
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]) or frame[column].nunique(dropna=True) <= 1:
            continue
        if column.startswith(f"{prefix}_ldq_"):
            local.append(column)
        elif feature_bucket(column, prefix) == "raw":
            raw.append(column)
        elif feature_bucket(column, prefix) == "deltaq":
            global_deltaq.append(column)
    return frame, prefix, sorted(raw), sorted(global_deltaq), sorted(local)


def resolve_model(model_id: str) -> dict[str, Any]:
    for config in MODEL_GRID:
        if config["model_id"] == model_id:
            return config
    raise ValueError(model_id)


def select_local(
    train: pd.DataFrame,
    y_train: np.ndarray,
    local_cols: list[str],
    method: str,
    selector_config: dict[str, Any],
    seed: int,
) -> list[str]:
    return fit_selector(train, y_train, local_cols, method, selector_config, seed).selected


def run_feature_groups(
    frame: pd.DataFrame,
    raw: list[str],
    global_deltaq: list[str],
    local: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    method: str,
    selector_config: dict[str, Any],
    model_config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    y = frame["knee_cycle"].to_numpy(float)
    for fold, (tr, va) in enumerate(splits):
        train = frame.iloc[tr]
        valid = frame.iloc[va]
        chosen = select_local(train, y[tr], local, method, selector_config, seed + 1009 * fold)
        for group, (use_raw, use_global, use_local) in GROUPS.items():
            features = (raw if use_raw else []) + (global_deltaq if use_global else []) + (chosen if use_local else [])
            x_train, medians = safe_numeric_matrix(train, features)
            x_valid, _ = safe_numeric_matrix(valid, features, medians)
            model = make_model(model_config, seed + 10007 * fold)
            model.fit(x_train, y[tr])
            pred = np.asarray(model.predict(x_valid), dtype=float)
            metric_rows.append({"analysis": "feature_group", "variant": group, "fold": fold, "n_train": len(tr), "n_features": len(features), **metrics(y[va], pred)})
            for row, index in enumerate(va):
                prediction_rows.append({"analysis": "feature_group", "variant": group, "fold": fold, "cell_id": frame.iloc[index]["cell_id"], "observed": y[index], "predicted": pred[row], "abs_error": abs(y[index] - pred[row])})
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def run_model_families(
    frame: pd.DataFrame,
    raw: list[str],
    global_deltaq: list[str],
    local: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    method: str,
    selector_config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    y = frame["knee_cycle"].to_numpy(float)
    for fold, (tr, va) in enumerate(splits):
        train, valid = frame.iloc[tr], frame.iloc[va]
        chosen = select_local(train, y[tr], local, method, selector_config, seed + 1009 * fold)
        features = raw + global_deltaq + chosen
        x_train, medians = safe_numeric_matrix(train, features)
        x_valid, _ = safe_numeric_matrix(valid, features, medians)
        models = build_models(seed + fold)
        # Keep one representative from each broad family in the formal comparison.
        keep = [name for name in ["ridge_cv", "svr_rbf", "random_forest", "xgboost"] if name in models]
        keep.append("extra_trees_compact")
        for name in keep:
            model = make_model(resolve_model("et_compact"), seed + fold) if name == "extra_trees_compact" else models[name]
            model.fit(x_train, y[tr])
            pred = np.asarray(model.predict(x_valid), dtype=float)
            metric_rows.append({"analysis": "model_family", "variant": name, "fold": fold, "n_train": len(tr), "n_features": len(features), **metrics(y[va], pred)})
            for row, index in enumerate(va):
                prediction_rows.append({"analysis": "model_family", "variant": name, "fold": fold, "cell_id": frame.iloc[index]["cell_id"], "observed": y[index], "predicted": pred[row], "abs_error": abs(y[index] - pred[row])})
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def region_pool(local: list[str], lower: float, upper: float, include: bool = True) -> list[str]:
    chosen: list[str] = []
    for feature in local:
        region = parse_region(feature)
        if region is None:
            inside = False
        else:
            midpoint = (float(region["start"]) + float(region["stop"])) / 2.0
            inside = lower <= midpoint <= upper
        if inside == include:
            chosen.append(feature)
    return chosen


def choose_top_from_pool(train: pd.DataFrame, y: np.ndarray, pool: list[str], top_k: int, seed: int) -> list[str]:
    config = {"top_k": min(top_k, len(pool))}
    return fit_selector(train, y, pool, "pearson", config, seed).selected


def run_region_controls(
    frame: pd.DataFrame,
    raw: list[str],
    global_deltaq: list[str],
    local: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    y = frame["knee_cycle"].to_numpy(float)
    core_pool = region_pool(local, 0.525, 0.650, True)
    noncore_pool = region_pool(local, 0.525, 0.650, False)
    starts = np.array(sorted({float(parse_region(c)["start"]) for c in local if parse_region(c) is not None}), dtype=float)
    for fold, (tr, va) in enumerate(splits):
        train, valid = frame.iloc[tr], frame.iloc[va]
        rng = np.random.default_rng(seed + 7919 * fold)
        full = choose_top_from_pool(train, y[tr], local, 40, seed + fold)
        variants: dict[str, list[str]] = {
            "all_regions_top40": full,
            "core_0525_0650_top40": choose_top_from_pool(train, y[tr], core_pool, 40, seed + fold),
            "exclude_core_top40": choose_top_from_pool(train, y[tr], noncore_pool, 40, seed + fold),
            "selected_minus_core": [c for c in full if c in noncore_pool],
        }
        for repeat in range(3):
            band_lower = float(rng.choice(starts[:-5]))
            band_upper = band_lower + 0.125
            pool = region_pool(local, band_lower, band_upper, True)
            variants[f"random_band_r{repeat}_{band_lower:.3f}_{band_upper:.3f}"] = choose_top_from_pool(train, y[tr], pool, 40, seed + fold + repeat)
            variants[f"random_features_r{repeat}"] = sorted(rng.choice(local, size=40, replace=False).tolist())

        base = raw + global_deltaq
        for variant, selected in variants.items():
            features = base + selected
            x_train, medians = safe_numeric_matrix(train, features)
            x_valid, _ = safe_numeric_matrix(valid, features, medians)
            model = make_model(model_config, seed + 10007 * fold)
            model.fit(x_train, y[tr])
            pred = np.asarray(model.predict(x_valid), dtype=float)
            metric_rows.append({"analysis": "region_control", "variant": variant, "fold": fold, "n_train": len(tr), "n_local": len(selected), **metrics(y[va], pred)})
            for row, index in enumerate(va):
                prediction_rows.append({"analysis": "region_control", "variant": variant, "fold": fold, "cell_id": frame.iloc[index]["cell_id"], "observed": y[index], "predicted": pred[row], "abs_error": abs(y[index] - pred[row])})

        # Region permutation is a diagnostic on the fitted full model, not a retrained competitor.
        features = base + full
        x_train, medians = safe_numeric_matrix(train, features)
        x_valid, _ = safe_numeric_matrix(valid, features, medians)
        model = make_model(model_config, seed + 10007 * fold)
        model.fit(x_train, y[tr])
        core_indices = [i for i, feature in enumerate(features) if feature in core_pool]
        for repeat in range(5):
            permuted = x_valid.copy()
            order = rng.permutation(len(va))
            if core_indices:
                permuted[:, core_indices] = permuted[order][:, core_indices]
            pred = np.asarray(model.predict(permuted), dtype=float)
            metric_rows.append({"analysis": "region_permutation", "variant": f"core_permuted_r{repeat}", "fold": fold, "n_train": len(tr), "n_local": len(core_indices), **metrics(y[va], pred)})
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def stratified_subsample(indices: np.ndarray, y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    if fraction >= 1.0:
        return indices.copy()
    ranked = pd.qcut(pd.Series(y[indices]).rank(method="first"), q=3, labels=False, duplicates="drop").to_numpy()
    chosen: list[int] = []
    for label in np.unique(ranked):
        group = indices[ranked == label]
        count = max(2, int(round(len(group) * fraction)))
        chosen.extend(rng.choice(group, size=min(count, len(group)), replace=False).tolist())
    return np.array(sorted(chosen), dtype=int)


def run_learning_curve(
    frame: pd.DataFrame,
    raw: list[str],
    global_deltaq: list[str],
    local: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    method: str,
    selector_config: dict[str, Any],
    model_config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    y = frame["knee_cycle"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    for fold, (tr, va) in enumerate(splits):
        for fraction in (0.25, 0.50, 0.75, 1.00):
            repeats = 3 if fraction < 1.0 else 1
            for repeat in range(repeats):
                rng = np.random.default_rng(seed + 100003 * fold + 101 * repeat + int(fraction * 100))
                sub = stratified_subsample(tr, y, fraction, rng)
                train, valid = frame.iloc[sub], frame.iloc[va]
                chosen = select_local(train, y[sub], local, method, selector_config, seed + fold + repeat)
                features = raw + global_deltaq + chosen
                x_train, medians = safe_numeric_matrix(train, features)
                x_valid, _ = safe_numeric_matrix(valid, features, medians)
                model = make_model(model_config, seed + 10007 * fold + repeat)
                model.fit(x_train, y[sub])
                pred = np.asarray(model.predict(x_valid), dtype=float)
                rows.append({"analysis": "learning_curve", "fraction": fraction, "repeat": repeat, "fold": fold, "n_train": len(sub), "n_features": len(features), **metrics(y[va], pred)})
    return pd.DataFrame(rows)


def summarize(metrics_table: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["analysis", "variant", "fraction"] if c in metrics_table.columns]
    return metrics_table.groupby(keys, dropna=False).agg(
        folds=("fold", "count"),
        n_train_mean=("n_train", "mean"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        mape_mean=("mape_percent", "mean"),
        r2_mean=("r2", "mean"),
    ).reset_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Development-only nested feature, region, and data-efficiency ablations")
    parser.add_argument("--feature-path", type=Path, default=Path("data/early_features_N150_local_deltaq_20170630.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/nested_ablation_learning_N150_20260713"))
    parser.add_argument("--method", choices=["pearson", "bootstrap_stable", "region_balanced", "contiguous", "mrmr"], default="pearson")
    parser.add_argument("--selector-config", default='{"top_k":40}')
    parser.add_argument("--frozen-config", type=Path, default=None, help="Optional frozen_config.json; overrides method, selector config, and model id")
    parser.add_argument("--model-id", choices=[c["model_id"] for c in MODEL_GRID], default="et_compact")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, prefix, raw, global_deltaq, local = load_data(args.feature_path)
    if args.frozen_config is not None:
        frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
        args.method = frozen["method"]
        selector_config = frozen["selector_config"]
        model_config = frozen["model_config"]
    else:
        selector_config = json.loads(args.selector_config)
        model_config = resolve_model(args.model_id)
    y = frame["knee_cycle"].to_numpy(float)
    splits = split_indices(frame["official_split"], y, args.outer_splits, args.outer_repeats, args.seed)
    group_metrics, group_predictions = run_feature_groups(frame, raw, global_deltaq, local, splits, args.method, selector_config, model_config, args.seed)
    model_metrics, model_predictions = run_model_families(frame, raw, global_deltaq, local, splits, args.method, selector_config, args.seed)
    region_metrics, region_predictions = run_region_controls(frame, raw, global_deltaq, local, splits, model_config, args.seed)
    learning_metrics = run_learning_curve(frame, raw, global_deltaq, local, splits, args.method, selector_config, model_config, args.seed)
    all_metrics = pd.concat([group_metrics, model_metrics, region_metrics, learning_metrics], ignore_index=True, sort=False)
    all_predictions = pd.concat([group_predictions, model_predictions, region_predictions], ignore_index=True, sort=False)
    all_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    all_predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    summarize(all_metrics).to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "feature_path": str(args.feature_path),
        "prefix": prefix,
        "development_cells": len(frame),
        "secondary_cells_used": 0,
        "raw_features": len(raw),
        "global_deltaq_features": len(global_deltaq),
        "local_candidates": len(local),
        "selector_method": args.method,
        "selector_config": selector_config,
        "model_config": model_config,
        "outer_splits": args.outer_splits,
        "outer_repeats": args.outer_repeats,
        "region_core_definition": "region midpoint in [0.525, 0.650]",
        "interpretation_note": "Region permutation is a predictive reliance diagnostic, not a causal intervention.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summarize(all_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
