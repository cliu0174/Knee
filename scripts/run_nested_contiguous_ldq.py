from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_feature_ablation import feature_bucket  # noqa: E402
from train_knee_baselines import META_COLUMNS, detect_window_prefix  # noqa: E402


REGION_RE = re.compile(
    r"_ldq_seg(?P<seg>\d+)_(?P<start>\d{4})_(?P<stop>\d{4})_(?P<descriptor>.+)$"
)

METHOD_COMPLEXITY_PENALTY = {
    "pearson": 0.0,
    "bootstrap_stable": 0.25,
    "region_balanced": 0.50,
    "contiguous": 0.75,
    "mrmr": 0.50,
}

MODEL_GRID = [
    {
        "model_id": "et_compact",
        "n_estimators": 500,
        "max_features": 0.5,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "max_depth": 6,
        "bootstrap": False,
        "complexity_penalty": 0.0,
    },
    {
        "model_id": "et_formal_anchor",
        "n_estimators": 1200,
        "max_features": 0.5,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_depth": 6,
        "bootstrap": False,
        "complexity_penalty": 0.25,
    },
]


@dataclass
class SelectorResult:
    selected: list[str]
    score_table: pd.DataFrame
    metadata: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(y_true - y_pred)
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape_percent": float(np.mean(abs_error / y_true) * 100.0),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
        "abs_error_q90": float(np.quantile(abs_error, 0.90)),
        "abs_error_max": float(np.max(abs_error)),
    }


def safe_numeric_matrix(df: pd.DataFrame, columns: list[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = df[columns].to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    if medians is None:
        medians = np.nanmedian(values, axis=0)
        medians[~np.isfinite(medians)] = 0.0
    missing = np.where(np.isnan(values))
    if missing[0].size:
        values[missing] = medians[missing[1]]
    return values, medians


def abs_corr_vector(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    target = np.asarray(y, dtype=float)
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    y_centered = target - np.mean(target)
    numerator = np.sum(x_centered * y_centered[:, None], axis=0)
    denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.abs(numerator / denominator)
    corr[~np.isfinite(corr)] = 0.0
    return corr


def deterministic_order(scores: np.ndarray, names: list[str]) -> list[int]:
    return sorted(range(len(names)), key=lambda idx: (-float(scores[idx]), names[idx]))


def bootstrap_scores(
    values: np.ndarray,
    y: np.ndarray,
    names: list[str],
    top_k: int,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = values.shape
    counts = np.zeros(p, dtype=float)
    corr_sum = np.zeros(p, dtype=float)
    k = min(max(1, int(top_k)), p)
    for _ in range(int(repeats)):
        idx = rng.integers(0, n, size=n)
        score = abs_corr_vector(values[idx], y[idx])
        corr_sum += score
        chosen = deterministic_order(score, names)[:k]
        counts[chosen] += 1.0
    frequency = counts / max(int(repeats), 1)
    mean_corr = corr_sum / max(int(repeats), 1)
    corr_scale = np.max(mean_corr)
    corr_norm = mean_corr / corr_scale if corr_scale > 0 else mean_corr
    combined = frequency + 0.25 * corr_norm
    return frequency, mean_corr, combined


def parse_region(feature: str) -> dict[str, Any] | None:
    match = REGION_RE.search(feature)
    if not match:
        return None
    item = match.groupdict()
    return {
        "seg": int(item["seg"]),
        "start": int(item["start"]) / 1000.0,
        "stop": int(item["stop"]) / 1000.0,
        "descriptor": item["descriptor"],
    }


def selector_grid(method: str, bootstrap_repeats: int, quick: bool) -> list[dict[str, Any]]:
    if method == "pearson":
        return [{"top_k": k} for k in ([40] if quick else [20, 40, 60])]
    if method == "bootstrap_stable":
        return [{"top_k": 40, "bootstrap_repeats": bootstrap_repeats}]
    if method == "region_balanced":
        caps = [2] if quick else [2, 3]
        return [
            {"top_k": 40, "bootstrap_repeats": bootstrap_repeats, "max_per_region": cap}
            for cap in caps
        ]
    if method == "contiguous":
        bands = [7] if quick else [5, 7, 9]
        fractions = [0.8] if quick else [0.6, 0.8]
        return [
            {
                "top_k": 40,
                "bootstrap_repeats": bootstrap_repeats,
                "band_regions": band,
                "core_fraction": fraction,
            }
            for band in bands
            for fraction in fractions
        ]
    if method == "mrmr":
        penalties = [0.15] if quick else [0.10, 0.25]
        return [{"top_k": 40, "redundancy_penalty": penalty} for penalty in penalties]
    raise ValueError(f"Unknown selector method: {method}")


def fit_selector(
    train: pd.DataFrame,
    y: np.ndarray,
    local_cols: list[str],
    method: str,
    config: dict[str, Any],
    seed: int,
) -> SelectorResult:
    values, _ = safe_numeric_matrix(train, local_cols)
    top_k = min(int(config["top_k"]), len(local_cols))
    relevance = abs_corr_vector(values, y)
    frequencies = np.full(len(local_cols), np.nan)
    mean_corr = relevance.copy()
    combined = relevance.copy()

    if method in {"bootstrap_stable", "region_balanced", "contiguous"}:
        frequencies, mean_corr, combined = bootstrap_scores(
            values,
            y,
            local_cols,
            top_k=top_k,
            repeats=int(config["bootstrap_repeats"]),
            seed=seed,
        )

    metadata: dict[str, Any] = {}
    if method == "pearson":
        selected_indices = deterministic_order(relevance, local_cols)[:top_k]
    elif method == "bootstrap_stable":
        selected_indices = deterministic_order(combined, local_cols)[:top_k]
    elif method == "region_balanced":
        selected_indices = []
        per_region: dict[int, int] = {}
        cap = int(config["max_per_region"])
        for idx in deterministic_order(combined, local_cols):
            region = parse_region(local_cols[idx])
            seg = int(region["seg"]) if region else -1
            if per_region.get(seg, 0) >= cap:
                continue
            selected_indices.append(idx)
            per_region[seg] = per_region.get(seg, 0) + 1
            if len(selected_indices) == top_k:
                break
        if len(selected_indices) < top_k:
            for idx in deterministic_order(combined, local_cols):
                if idx not in selected_indices:
                    selected_indices.append(idx)
                if len(selected_indices) == top_k:
                    break
        metadata["selected_region_count"] = len(per_region)
    elif method == "contiguous":
        region_to_indices: dict[int, list[int]] = {}
        region_bounds: dict[int, tuple[float, float]] = {}
        for idx, feature in enumerate(local_cols):
            region = parse_region(feature)
            if region is None:
                continue
            seg = int(region["seg"])
            region_to_indices.setdefault(seg, []).append(idx)
            region_bounds[seg] = (float(region["start"]), float(region["stop"]))
        segments = sorted(region_to_indices)
        if not segments:
            raise RuntimeError("No ordered Local DeltaQ segment features were found")
        raw_region = np.array([max(combined[region_to_indices[seg]]) for seg in segments], dtype=float)
        smooth_region = np.empty_like(raw_region)
        for pos in range(len(segments)):
            left = raw_region[pos - 1] if pos > 0 else raw_region[pos]
            right = raw_region[pos + 1] if pos + 1 < len(segments) else raw_region[pos]
            smooth_region[pos] = 0.25 * left + 0.50 * raw_region[pos] + 0.25 * right
        width = min(int(config["band_regions"]), len(segments))
        band_scores = [float(np.sum(smooth_region[start : start + width])) for start in range(len(segments) - width + 1)]
        band_start_pos = int(np.argmax(band_scores))
        band_segments = segments[band_start_pos : band_start_pos + width]
        band_set = set(band_segments)
        inside = [idx for seg in band_segments for idx in region_to_indices[seg]]
        outside = [
            idx
            for idx, feature in enumerate(local_cols)
            if (parse_region(feature) is None or int(parse_region(feature)["seg"]) not in band_set)
        ]
        core_count = min(len(inside), int(round(top_k * float(config["core_fraction"]))))
        selected_indices = sorted(inside, key=lambda idx: (-float(combined[idx]), local_cols[idx]))[:core_count]
        complement = sorted(outside, key=lambda idx: (-float(combined[idx]), local_cols[idx]))
        selected_indices.extend(complement[: max(0, top_k - len(selected_indices))])
        if len(selected_indices) < top_k:
            remaining_inside = [
                idx
                for idx in sorted(inside, key=lambda idx: (-float(combined[idx]), local_cols[idx]))
                if idx not in selected_indices
            ]
            selected_indices.extend(remaining_inside[: top_k - len(selected_indices)])
        start_coord = min(region_bounds[seg][0] for seg in band_segments)
        stop_coord = max(region_bounds[seg][1] for seg in band_segments)
        metadata.update(
            {
                "band_segments": band_segments,
                "band_start": start_coord,
                "band_stop": stop_coord,
                "band_score": band_scores[band_start_pos],
                "core_selected": core_count,
                "complement_selected": top_k - core_count,
            }
        )
    elif method == "mrmr":
        standardized = values - np.mean(values, axis=0, keepdims=True)
        scale = np.std(standardized, axis=0, keepdims=True)
        scale[scale == 0] = 1.0
        standardized = standardized / scale
        corr = np.abs(np.corrcoef(standardized, rowvar=False))
        corr[~np.isfinite(corr)] = 0.0
        np.fill_diagonal(corr, 0.0)
        selected_indices = [deterministic_order(relevance, local_cols)[0]]
        penalty = float(config["redundancy_penalty"])
        while len(selected_indices) < top_k:
            redundancy = np.mean(corr[:, selected_indices], axis=1)
            score = relevance - penalty * redundancy
            score[selected_indices] = -np.inf
            next_idx = deterministic_order(score, local_cols)[0]
            selected_indices.append(next_idx)
    else:  # pragma: no cover
        raise ValueError(method)

    selected = [local_cols[idx] for idx in selected_indices[:top_k]]
    rows = []
    for idx, feature in enumerate(local_cols):
        region = parse_region(feature) or {}
        rows.append(
            {
                "feature": feature,
                "selected": feature in selected,
                "relevance_abs_corr": float(relevance[idx]),
                "bootstrap_frequency": float(frequencies[idx]) if np.isfinite(frequencies[idx]) else np.nan,
                "bootstrap_mean_abs_corr": float(mean_corr[idx]),
                "selection_score": float(combined[idx]),
                "seg": region.get("seg", np.nan),
                "start": region.get("start", np.nan),
                "stop": region.get("stop", np.nan),
                "descriptor": region.get("descriptor", ""),
            }
        )
    score_table = pd.DataFrame(rows).sort_values(
        ["selected", "selection_score", "feature"], ascending=[False, False, True]
    )
    return SelectorResult(selected=selected, score_table=score_table, metadata=metadata)


def base_and_local_features(df: pd.DataFrame, prefix: str) -> tuple[list[str], list[str]]:
    base: list[str] = []
    local: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col == f"{prefix}_window_complete":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique(dropna=True) <= 1:
            continue
        if col.startswith(f"{prefix}_ldq_"):
            local.append(col)
        elif feature_bucket(col, prefix) in {"raw", "deltaq"}:
            base.append(col)
    return sorted(base), sorted(local)


def make_model(config: dict[str, Any], seed: int) -> ExtraTreesRegressor:
    depth = config.get("max_depth")
    return ExtraTreesRegressor(
        n_estimators=int(config["n_estimators"]),
        max_features=config["max_features"],
        min_samples_leaf=int(config["min_samples_leaf"]),
        min_samples_split=int(config["min_samples_split"]),
        max_depth=None if depth is None else int(depth),
        bootstrap=bool(config["bootstrap"]),
        random_state=int(seed),
        n_jobs=-1,
    )


def make_strata(groups: pd.Series, y: np.ndarray, n_bins: int = 3) -> np.ndarray:
    rank = pd.Series(y).rank(method="first")
    bins = pd.qcut(rank, q=min(n_bins, len(rank)), labels=False, duplicates="drop").astype(str)
    return (groups.reset_index(drop=True).astype(str) + "_q" + bins).to_numpy()


def split_indices(
    groups: pd.Series,
    y: np.ndarray,
    n_splits: int,
    repeats: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    strata = make_strata(groups, y)
    counts = pd.Series(strata).value_counts()
    if not counts.empty and int(counts.min()) >= n_splits:
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), strata))
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=repeats, random_state=seed)
    return list(splitter.split(np.zeros(len(y))))


def inner_score(
    train: pd.DataFrame,
    base_cols: list[str],
    local_cols: list[str],
    method: str,
    selector_config: dict[str, Any],
    model_config: dict[str, Any],
    n_splits: int,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    y = train["knee_cycle"].to_numpy(dtype=float)
    splits = split_indices(train["official_split"], y, n_splits, repeats, seed)
    maes: list[float] = []
    rmses: list[float] = []
    for fold_id, (tr, va) in enumerate(splits):
        inner_train = train.iloc[tr]
        inner_valid = train.iloc[va]
        selector = fit_selector(
            inner_train,
            y[tr],
            local_cols,
            method,
            selector_config,
            seed=seed + 1009 * (fold_id + 1),
        )
        features = base_cols + selector.selected
        x_train, medians = safe_numeric_matrix(inner_train, features)
        x_valid, _ = safe_numeric_matrix(inner_valid, features, medians)
        model = make_model(model_config, seed + fold_id)
        model.fit(x_train, y[tr])
        pred = np.asarray(model.predict(x_valid), dtype=float)
        maes.append(float(mean_absolute_error(y[va], pred)))
        rmses.append(float(mean_squared_error(y[va], pred) ** 0.5))
    feature_penalty = 0.01 * float(selector_config["top_k"])
    complexity = (
        METHOD_COMPLEXITY_PENALTY[method]
        + float(model_config["complexity_penalty"])
        + feature_penalty
    )
    return {
        "inner_mae_mean": float(np.mean(maes)),
        "inner_mae_std": float(np.std(maes)),
        "inner_mae_max": float(np.max(maes)),
        "inner_rmse_mean": float(np.mean(rmses)),
        "complexity_penalty": complexity,
        "inner_selection_score": float(
            np.mean(maes) + 0.10 * np.std(maes) + 0.10 * np.max(maes) + complexity
        ),
    }


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def choose_config(
    train: pd.DataFrame,
    base_cols: list[str],
    local_cols: list[str],
    method: str,
    selector_configs: list[dict[str, Any]],
    model_grid: list[dict[str, Any]],
    inner_splits: int,
    inner_repeats: int,
    seed: int,
    outer_id: int | str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for selector_id, selector_config in enumerate(selector_configs):
        for model_config in model_grid:
            score = inner_score(
                train,
                base_cols,
                local_cols,
                method,
                selector_config,
                model_config,
                inner_splits,
                inner_repeats,
                seed + selector_id * 10007,
            )
            rows.append(
                {
                    "outer_id": outer_id,
                    "method": method,
                    "selector_config": json_compact(selector_config),
                    "model_id": model_config["model_id"],
                    **score,
                }
            )
    ranked = sorted(rows, key=lambda row: (row["inner_selection_score"], row["inner_mae_mean"], row["selector_config"], row["model_id"]))
    best = ranked[0]
    selector_config = json.loads(best["selector_config"])
    model_config = next(item for item in model_grid if item["model_id"] == best["model_id"])
    return selector_config, model_config, rows


def jaccard_summary(selected_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in selected_features.groupby("method"):
        sets = [set(part["feature"]) for _, part in group.groupby("outer_id")]
        values: list[float] = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                values.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
        rows.append(
            {
                "method": method,
                "pair_count": len(values),
                "jaccard_mean": float(np.mean(values)) if values else np.nan,
                "jaccard_std": float(np.std(values)) if values else np.nan,
                "jaccard_median": float(np.median(values)) if values else np.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_method_tests(oof: pd.DataFrame, bootstrap_repeats: int, seed: int) -> pd.DataFrame:
    collapsed = (
        oof.groupby(["method", "cell_id", "observed"], as_index=False)
        .agg(predicted=("predicted", "mean"))
    )
    collapsed["abs_error"] = np.abs(collapsed["observed"] - collapsed["predicted"])
    methods = sorted(collapsed["method"].unique())
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for i, first in enumerate(methods):
        a = collapsed[collapsed["method"] == first].set_index("cell_id")
        for second in methods[i + 1 :]:
            b = collapsed[collapsed["method"] == second].set_index("cell_id")
            shared = a.index.intersection(b.index)
            diff = a.loc[shared, "abs_error"].to_numpy() - b.loc[shared, "abs_error"].to_numpy()
            boot = np.empty(bootstrap_repeats, dtype=float)
            for rep in range(bootstrap_repeats):
                idx = rng.integers(0, len(diff), size=len(diff))
                boot[rep] = float(np.mean(diff[idx]))
            signs = rng.choice([-1.0, 1.0], size=(bootstrap_repeats, len(diff)))
            permuted = np.mean(signs * diff[None, :], axis=1)
            observed = float(np.mean(diff))
            p_value = float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (bootstrap_repeats + 1))
            rows.append(
                {
                    "method_a": first,
                    "method_b": second,
                    "mean_abs_error_difference_a_minus_b": observed,
                    "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                    "sign_flip_p_value": p_value,
                    "n_cells": len(diff),
                }
            )
    return pd.DataFrame(rows)


def common_cohort_ids(cohort_path: Path | None) -> set[str] | None:
    if cohort_path is None:
        return None
    cohort = pd.read_csv(cohort_path, usecols=["cell_id", "knee_after_window"])
    return set(cohort.loc[cohort["knee_after_window"].astype(bool), "cell_id"].astype(str))


def load_development(feature_path: Path, cohort_path: Path | None = None) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    df = pd.read_csv(feature_path)
    df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    cohort_ids = common_cohort_ids(cohort_path)
    if cohort_ids is not None:
        df = df[df["cell_id"].astype(str).isin(cohort_ids)].copy().reset_index(drop=True)
    prefix = detect_window_prefix(df)
    base_cols, local_cols = base_and_local_features(df, prefix)
    development = df[df["official_split"].isin(["train", "primary_test"])].copy().reset_index(drop=True)
    if len(development) != 83:
        raise RuntimeError(f"Expected 83 development cells, found {len(development)}")
    if not local_cols:
        raise RuntimeError("No Local DeltaQ candidates were found")
    return development, prefix, base_cols, local_cols


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def run_develop(args: argparse.Namespace) -> None:
    feature_path = Path(args.feature_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cohort_path = Path(args.cohort_path) if args.cohort_path else None
    dev, prefix, base_cols, local_cols = load_development(feature_path, cohort_path)
    y = dev["knee_cycle"].to_numpy(dtype=float)
    outer_splits = split_indices(dev["official_split"], y, args.outer_splits, args.outer_repeats, args.random_state)
    model_grid = MODEL_GRID[:1] if args.quick else MODEL_GRID

    inner_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    score_rows: list[pd.DataFrame] = []

    for outer_id, (tr, va) in enumerate(outer_splits):
        outer_train = dev.iloc[tr]
        outer_valid = dev.iloc[va]
        for method in args.methods:
            selector_config, model_config, candidate_rows = choose_config(
                outer_train,
                base_cols,
                local_cols,
                method,
                selector_grid(method, args.bootstrap_repeats, args.quick),
                model_grid,
                args.inner_splits,
                args.inner_repeats,
                args.random_state + outer_id * 100003,
                outer_id,
            )
            inner_rows.extend(candidate_rows)
            selector = fit_selector(
                outer_train,
                y[tr],
                local_cols,
                method,
                selector_config,
                seed=args.random_state + outer_id * 100003 + 7919,
            )
            features = base_cols + selector.selected
            x_train, medians = safe_numeric_matrix(outer_train, features)
            x_valid, _ = safe_numeric_matrix(outer_valid, features, medians)
            model = make_model(model_config, args.random_state + outer_id)
            fit_start = time.perf_counter()
            model.fit(x_train, y[tr])
            fit_seconds = time.perf_counter() - fit_start
            infer_start = time.perf_counter()
            pred = np.asarray(model.predict(x_valid), dtype=float)
            infer_seconds = time.perf_counter() - infer_start
            metrics = regression_metrics(y[va], pred)
            fold_rows.append(
                {
                    "outer_id": outer_id,
                    "outer_repeat": outer_id // args.outer_splits,
                    "outer_fold": outer_id % args.outer_splits,
                    "method": method,
                    "n_train": len(tr),
                    "n_valid": len(va),
                    "selector_config": json_compact(selector_config),
                    "model_id": model_config["model_id"],
                    "feature_count": len(features),
                    "local_feature_count": len(selector.selected),
                    "fit_seconds": fit_seconds,
                    "inference_seconds_total": infer_seconds,
                    "model_bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
                    **selector.metadata,
                    **metrics,
                }
            )
            for row_idx, estimate in zip(va, pred):
                row = dev.iloc[row_idx]
                prediction_rows.append(
                    {
                        "outer_id": outer_id,
                        "outer_repeat": outer_id // args.outer_splits,
                        "outer_fold": outer_id % args.outer_splits,
                        "method": method,
                        "cell_id": row["cell_id"],
                        "batch": row["batch"],
                        "official_split": row["official_split"],
                        "observed": float(row["knee_cycle"]),
                        "predicted": float(estimate),
                        "abs_error": float(abs(float(row["knee_cycle"]) - float(estimate))),
                    }
                )
            for rank, feature in enumerate(selector.selected, 1):
                selected_rows.append(
                    {
                        "outer_id": outer_id,
                        "method": method,
                        "rank": rank,
                        "feature": feature,
                        **(parse_region(feature) or {}),
                    }
                )
            table = selector.score_table.copy()
            table.insert(0, "outer_id", outer_id)
            table.insert(1, "method", method)
            score_rows.append(table)
        print(f"Completed outer split {outer_id + 1}/{len(outer_splits)}", flush=True)

    inner = pd.DataFrame(inner_rows)
    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(prediction_rows)
    selected = pd.DataFrame(selected_rows)
    scores = pd.concat(score_rows, ignore_index=True)
    summary = (
        folds.groupby("method", as_index=False)
        .agg(
            outer_mae_mean=("mae", "mean"),
            outer_mae_std=("mae", "std"),
            outer_mae_min=("mae", "min"),
            outer_mae_median=("mae", "median"),
            outer_mae_max=("mae", "max"),
            outer_rmse_mean=("rmse", "mean"),
            outer_r2_mean=("r2", "mean"),
            outer_abs_error_q90_mean=("abs_error_q90", "mean"),
            outer_abs_error_max=("abs_error_max", "max"),
            feature_count_mean=("feature_count", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
            inference_seconds_mean=("inference_seconds_total", "mean"),
            model_bytes_mean=("model_bytes", "mean"),
        )
    )
    summary["method_complexity_penalty"] = summary["method"].map(METHOD_COMPLEXITY_PENALTY)
    summary["development_selection_score"] = (
        summary["outer_mae_mean"]
        + 0.10 * summary["outer_mae_std"]
        + 0.10 * summary["outer_mae_max"]
        + summary["method_complexity_penalty"]
    )
    summary = summary.sort_values(["development_selection_score", "outer_mae_mean"]).reset_index(drop=True)
    selected_method = str(summary.iloc[0]["method"])

    inner.to_csv(output / "development_inner_selection.csv", index=False)
    folds.to_csv(output / "development_outer_fold_metrics.csv", index=False)
    oof.to_csv(output / "development_oof_predictions.csv", index=False)
    selected.to_csv(output / "development_selected_features.csv", index=False)
    scores.to_csv(output / "development_feature_scores.csv", index=False)
    summary.to_csv(output / "development_summary.csv", index=False)
    jaccard_summary(selected).to_csv(output / "development_selection_jaccard.csv", index=False)
    paired_method_tests(oof, args.stat_repeats, args.random_state + 31).to_csv(
        output / "development_paired_comparisons.csv", index=False
    )
    pd.DataFrame({"feature": base_cols}).to_csv(output / "base_features.csv", index=False)
    metadata = {
        "phase": "development_only",
        "created_utc": utc_now(),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "common_cohort_path": str(cohort_path) if cohort_path else None,
        "common_cohort_sha256": sha256_file(cohort_path) if cohort_path else None,
        "window_prefix": prefix,
        "n_development": len(dev),
        "n_base_features": len(base_cols),
        "n_local_candidates": len(local_cols),
        "methods": args.methods,
        "outer_splits": args.outer_splits,
        "outer_repeats": args.outer_repeats,
        "inner_splits": args.inner_splits,
        "inner_repeats": args.inner_repeats,
        "bootstrap_repeats": args.bootstrap_repeats,
        "random_state": args.random_state,
        "predeclared_method_complexity_penalty": METHOD_COMPLEXITY_PENALTY,
        "predeclared_model_grid": model_grid,
        "predeclared_score": "mean_outer_MAE + 0.10*std_outer_MAE + 0.10*worst_outer_MAE + method_complexity_penalty",
        "selected_method_by_development": selected_method,
        "secondary_labels_read_for_selection": False,
        "software_versions": software_versions(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(f"Development-selected method: {selected_method}")


def run_freeze(args: argparse.Namespace) -> None:
    feature_path = Path(args.feature_path)
    output = Path(args.output_dir)
    metadata_path = output / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Run --phase develop before --phase freeze")
    development_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if development_metadata.get("phase") != "development_only":
        raise RuntimeError("Development metadata is missing or was modified")
    cohort_path = Path(args.cohort_path) if args.cohort_path else None
    dev, prefix, base_cols, local_cols = load_development(feature_path, cohort_path)
    method = args.freeze_method or development_metadata["selected_method_by_development"]
    model_grid = MODEL_GRID[:1] if args.quick else MODEL_GRID
    selector_config, model_config, rows = choose_config(
        dev,
        base_cols,
        local_cols,
        method,
        selector_grid(method, args.bootstrap_repeats, args.quick),
        model_grid,
        args.freeze_splits,
        args.freeze_repeats,
        args.random_state + 424242,
        "full_development",
    )
    selector = fit_selector(
        dev,
        dev["knee_cycle"].to_numpy(dtype=float),
        local_cols,
        method,
        selector_config,
        args.random_state + 515151,
    )
    frozen = {
        "phase": "frozen_before_external_evaluation",
        "frozen_utc": utc_now(),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "common_cohort_path": str(cohort_path) if cohort_path else None,
        "common_cohort_sha256": sha256_file(cohort_path) if cohort_path else None,
        "window_prefix": prefix,
        "method": method,
        "selector_config": selector_config,
        "selector_metadata": selector.metadata,
        "model_config": model_config,
        "base_features": base_cols,
        "selected_local_features": selector.selected,
        "n_development": len(dev),
        "external_seed_list": list(range(args.seed_start, args.seed_start + args.seed_repeats)),
        "evaluation_metrics": ["mae", "rmse", "mape_percent", "r2", "abs_error_q90", "abs_error_max"],
        "secondary_labels_read_for_freeze": False,
        "secondary_evaluation_status": "No secondary observation or label determined or affected the formal selector, model family, hyperparameters, or fitted parameters; the present evaluation is protocol-frozen.",
        "software_versions": software_versions(),
    }
    pd.DataFrame(rows).to_csv(output / "freeze_inner_selection.csv", index=False)
    pd.DataFrame({"feature": selector.selected}).to_csv(output / "frozen_local_features.csv", index=False)
    selector.score_table.to_csv(output / "freeze_feature_scores.csv", index=False)
    (output / "frozen_config.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps({k: frozen[k] for k in ["frozen_utc", "method", "selector_config", "selector_metadata", "model_config"]}, indent=2))


def bootstrap_metric_ci(
    y: np.ndarray,
    pred: np.ndarray,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []
    for _ in range(repeats):
        idx = rng.integers(0, len(y), size=len(y))
        records.append(regression_metrics(y[idx], pred[idx]))
    frame = pd.DataFrame(records)
    rows = []
    point = regression_metrics(y, pred)
    for metric in frame.columns:
        rows.append(
            {
                "metric": metric,
                "point_estimate": point[metric],
                "ci_low": float(frame[metric].quantile(0.025)),
                "ci_high": float(frame[metric].quantile(0.975)),
                "bootstrap_repeats": repeats,
            }
        )
    return pd.DataFrame(rows)


def compare_external_baseline(
    per_cell: pd.DataFrame,
    baseline_path: Path,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    if not baseline_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline["official_split"] == "secondary_test"].copy()
    baseline = baseline.groupby(["cell_id", "observed_knee_cycle"], as_index=False).agg(
        baseline_prediction=("predicted_knee_cycle", "mean")
    )
    merged = per_cell.merge(baseline, on="cell_id", how="inner")
    old_error = np.abs(merged["observed"] - merged["baseline_prediction"])
    new_error = merged["abs_error"]
    diff = np.asarray(new_error - old_error, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=float)
    for rep in range(repeats):
        idx = rng.integers(0, len(diff), size=len(diff))
        boot[rep] = float(np.mean(diff[idx]))
    signs = rng.choice([-1.0, 1.0], size=(repeats, len(diff)))
    permuted = np.mean(signs * diff[None, :], axis=1)
    observed = float(np.mean(diff))
    return pd.DataFrame(
        [
            {
                "comparison": "frozen_new_minus_original_LDQ_ET_top40_abs_error",
                "mean_paired_difference": observed,
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "sign_flip_p_value": float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (repeats + 1)),
                "n_cells": len(diff),
            }
        ]
    )


def crossfit_residual_interval(
    output: Path,
    method: str,
    per_cell: pd.DataFrame,
    alpha: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an empirical constant-width interval from development OOF residuals.

    Repeated-fold predictions are averaged by cell before calibration. Because the
    calibration residuals are cross-fitted and the external batch may shift, this
    is reported as an empirical interval diagnostic rather than an exact split-
    conformal guarantee.
    """
    oof_path = output / "development_oof_predictions.csv"
    if not oof_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    oof = pd.read_csv(oof_path)
    oof = oof[oof["method"] == method].copy()
    calibration = oof.groupby(["cell_id", "observed"], as_index=False).agg(predicted=("predicted", "mean"))
    residuals = np.abs(calibration["observed"].to_numpy(float) - calibration["predicted"].to_numpy(float))
    n = len(residuals)
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    half_width = float(np.quantile(residuals, level, method="higher"))
    intervals = per_cell.copy()
    intervals["interval_lower"] = intervals["predicted_mean"] - half_width
    intervals["interval_upper"] = intervals["predicted_mean"] + half_width
    intervals["interval_width"] = 2.0 * half_width
    intervals["covered"] = (
        (intervals["observed"] >= intervals["interval_lower"])
        & (intervals["observed"] <= intervals["interval_upper"])
    )
    ranked = intervals["observed"].rank(method="first")
    intervals["observed_knee_stratum"] = pd.qcut(ranked, q=3, labels=["early", "mid", "late"])
    rows: list[dict[str, Any]] = [
        {
            "stratum": "all",
            "n": len(intervals),
            "nominal_coverage": 1.0 - alpha,
            "empirical_coverage": float(intervals["covered"].mean()),
            "mean_width": float(intervals["interval_width"].mean()),
            "median_width": float(intervals["interval_width"].median()),
            "calibration_cells": n,
            "calibration_quantile_level": level,
            "calibration_half_width": half_width,
            "status": "cross-fitted residual interval; empirical diagnostic under possible batch shift",
        }
    ]
    for stratum, group in intervals.groupby("observed_knee_stratum", observed=False):
        rows.append(
            {
                "stratum": str(stratum),
                "n": len(group),
                "nominal_coverage": 1.0 - alpha,
                "empirical_coverage": float(group["covered"].mean()),
                "mean_width": float(group["interval_width"].mean()),
                "median_width": float(group["interval_width"].median()),
                "calibration_cells": n,
                "calibration_quantile_level": level,
                "calibration_half_width": half_width,
                "status": "post-hoc subgroup coverage; not a conditional coverage guarantee",
            }
        )
    return intervals, pd.DataFrame(rows)


def run_external(args: argparse.Namespace) -> None:
    feature_path = Path(args.feature_path)
    output = Path(args.output_dir)
    frozen_path = output / "frozen_config.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Run --phase freeze before --phase external")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("phase") != "frozen_before_external_evaluation":
        raise RuntimeError("The frozen configuration is invalid")
    current_hash = sha256_file(feature_path)
    if current_hash != frozen["feature_sha256"]:
        raise RuntimeError("Feature file hash changed after freezing")
    df = pd.read_csv(feature_path)
    df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    cohort_path = Path(frozen["common_cohort_path"]) if frozen.get("common_cohort_path") else None
    if cohort_path is not None:
        if sha256_file(cohort_path) != frozen.get("common_cohort_sha256"):
            raise RuntimeError("Common-cohort file hash changed after freezing")
        cohort_ids = common_cohort_ids(cohort_path)
        df = df[df["cell_id"].astype(str).isin(cohort_ids)].copy().reset_index(drop=True)
    dev = df[df["official_split"].isin(["train", "primary_test"])].copy().reset_index(drop=True)
    ext = df[df["official_split"] == "secondary_test"].copy().reset_index(drop=True)
    features = list(frozen["base_features"]) + list(frozen["selected_local_features"])
    x_dev, medians = safe_numeric_matrix(dev, features)
    x_ext, _ = safe_numeric_matrix(ext, features, medians)
    y_dev = dev["knee_cycle"].to_numpy(dtype=float)
    y_ext = ext["knee_cycle"].to_numpy(dtype=float)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in frozen["external_seed_list"]:
        model = make_model(frozen["model_config"], int(seed))
        fit_start = time.perf_counter()
        model.fit(x_dev, y_dev)
        fit_seconds = time.perf_counter() - fit_start
        infer_start = time.perf_counter()
        pred = np.asarray(model.predict(x_ext), dtype=float)
        inference_seconds = time.perf_counter() - infer_start
        metric_rows.append(
            {
                "seed": seed,
                "n": len(ext),
                "fit_seconds": fit_seconds,
                "inference_seconds_total": inference_seconds,
                "inference_ms_per_cell": inference_seconds * 1000.0 / len(ext),
                "model_bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
                **regression_metrics(y_ext, pred),
            }
        )
        for row_idx, estimate in enumerate(pred):
            row = ext.iloc[row_idx]
            prediction_rows.append(
                {
                    "seed": seed,
                    "cell_id": row["cell_id"],
                    "batch": row["batch"],
                    "official_split": row["official_split"],
                    "observed": float(row["knee_cycle"]),
                    "predicted": float(estimate),
                    "abs_error": float(abs(float(row["knee_cycle"]) - float(estimate))),
                    "cycle_life": float(row["cycle_life"]),
                    "label_confidence": float(row["label_confidence"]),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = metrics[[
        "mae", "rmse", "mape_percent", "r2", "abs_error_q90", "abs_error_max",
        "fit_seconds", "inference_ms_per_cell", "model_bytes",
    ]].agg(["mean", "std", "min", "median", "max"]).T.reset_index().rename(columns={"index": "metric"})
    per_cell = (
        predictions.groupby(
            ["cell_id", "batch", "official_split", "observed", "cycle_life", "label_confidence"],
            as_index=False,
        )
        .agg(
            predicted_mean=("predicted", "mean"),
            predicted_std=("predicted", "std"),
            abs_error=("abs_error", "mean"),
        )
    )
    mean_pred = per_cell["predicted_mean"].to_numpy(dtype=float)
    metrics.to_csv(output / "external_metrics_by_seed.csv", index=False)
    predictions.to_csv(output / "external_predictions_by_seed.csv", index=False)
    summary.to_csv(output / "external_summary.csv", index=False)
    per_cell.to_csv(output / "external_per_cell.csv", index=False)
    intervals, interval_summary = crossfit_residual_interval(output, frozen["method"], per_cell)
    intervals.to_csv(output / "external_prediction_intervals.csv", index=False)
    interval_summary.to_csv(output / "external_interval_summary.csv", index=False)
    tail_summary = pd.DataFrame(
        [
            {
                "n": len(per_cell),
                "abs_error_median": float(per_cell["abs_error"].median()),
                "abs_error_q80": float(per_cell["abs_error"].quantile(0.80)),
                "abs_error_q90": float(per_cell["abs_error"].quantile(0.90)),
                "abs_error_q95": float(per_cell["abs_error"].quantile(0.95)),
                "abs_error_max": float(per_cell["abs_error"].max()),
                "cells_over_100_cycles": int((per_cell["abs_error"] > 100).sum()),
                "cells_over_200_cycles": int((per_cell["abs_error"] > 200).sum()),
            }
        ]
    )
    tail_summary.to_csv(output / "external_tail_summary.csv", index=False)
    bootstrap_metric_ci(
        per_cell["observed"].to_numpy(dtype=float),
        mean_pred,
        args.stat_repeats,
        args.random_state + 707,
    ).to_csv(
        output / "external_bootstrap_ci.csv", index=False
    )
    comparison = compare_external_baseline(
        per_cell,
        Path(args.baseline_predictions),
        args.stat_repeats,
        args.random_state + 909,
    )
    comparison.to_csv(output / "external_paired_comparison_to_original.csv", index=False)
    external_metadata = {
        "phase": "external_evaluation_complete",
        "evaluated_utc": utc_now(),
        "frozen_config_sha256": sha256_file(frozen_path),
        "n_external": len(ext),
        "seed_list": frozen["external_seed_list"],
        "cell_bootstrap_repeats": args.stat_repeats,
        "status": "protocol-frozen secondary-batch evaluation; no secondary observation or label entered or determined formal model development",
        "software_versions": software_versions(),
    }
    (output / "external_metadata.json").write_text(json.dumps(external_metadata, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    if not comparison.empty:
        print(comparison.round(4).to_string(index=False))


def finalize_external(args: argparse.Namespace) -> None:
    """Finalize metadata/comparisons from already saved external predictions.

    This recovery path does not reopen the feature table or refit a model, so a
    post-processing failure cannot cause an avoidable second evaluation pass.
    """
    output = Path(args.output_dir)
    frozen_path = output / "frozen_config.json"
    per_cell_path = output / "external_per_cell.csv"
    metrics_path = output / "external_metrics_by_seed.csv"
    if not frozen_path.exists() or not per_cell_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Frozen config and saved external predictions are required")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    per_cell = pd.read_csv(per_cell_path)
    metrics = pd.read_csv(metrics_path)
    bootstrap_metric_ci(
        per_cell["observed"].to_numpy(dtype=float),
        per_cell["predicted_mean"].to_numpy(dtype=float),
        args.stat_repeats,
        args.random_state + 707,
    ).to_csv(output / "external_bootstrap_ci.csv", index=False)
    comparison = compare_external_baseline(
        per_cell,
        Path(args.baseline_predictions),
        args.stat_repeats,
        args.random_state + 909,
    )
    comparison.to_csv(output / "external_paired_comparison_to_original.csv", index=False)
    external_metadata = {
        "phase": "external_evaluation_complete",
        "evaluated_utc": utc_now(),
        "frozen_config_sha256": sha256_file(frozen_path),
        "n_external": len(per_cell),
        "seed_list": metrics["seed"].astype(int).tolist(),
        "cell_bootstrap_repeats": args.stat_repeats,
        "status": "protocol-frozen secondary-batch evaluation; no secondary observation or label entered or determined formal model development",
        "recovery_note": "Finalized from saved predictions after a post-processing error; feature data were not reopened and models were not refit.",
        "software_versions": software_versions(),
    }
    (output / "external_metadata.json").write_text(json.dumps(external_metadata, indent=2), encoding="utf-8")
    print(comparison.round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested development-only selection and frozen evaluation for ordered Local DeltaQ regions."
    )
    parser.add_argument("--phase", choices=["develop", "freeze", "external", "finalize"], required=True)
    parser.add_argument("--feature-path", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--cohort-path", default=None, help="Optional feature table defining a common knee-after-window cell cohort")
    parser.add_argument("--output-dir", default="results/nested_contiguous_ldq_N150_20260713")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["pearson", "bootstrap_stable", "region_balanced", "contiguous", "mrmr"],
        default=["pearson", "bootstrap_stable", "region_balanced", "contiguous", "mrmr"],
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--inner-repeats", type=int, default=1)
    parser.add_argument("--freeze-splits", type=int, default=5)
    parser.add_argument("--freeze-repeats", type=int, default=2)
    parser.add_argument("--bootstrap-repeats", type=int, default=40)
    parser.add_argument("--stat-repeats", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=20260713)
    parser.add_argument("--freeze-method", choices=["pearson", "bootstrap_stable", "region_balanced", "contiguous", "mrmr"])
    parser.add_argument("--seed-start", type=int, default=123)
    parser.add_argument("--seed-repeats", type=int, default=30)
    parser.add_argument(
        "--baseline-predictions",
        default="results/extratrees_ldq_hpo_N150_quick30/extratrees_hpo_final_predictions.csv",
    )
    parser.add_argument("--quick", action="store_true", help="Use one selector setting and compact ET for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "develop":
        run_develop(args)
    elif args.phase == "freeze":
        run_freeze(args)
    elif args.phase == "external":
        run_external(args)
    else:
        finalize_external(args)


if __name__ == "__main__":
    main()
