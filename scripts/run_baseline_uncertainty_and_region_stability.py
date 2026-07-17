from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    raise RuntimeError("xgboost is required") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_feature_ablation import feature_bucket  # noqa: E402
from train_knee_baselines import META_COLUMNS, detect_window_prefix  # noqa: E402


REGION_RE = re.compile(r"_seg(?P<seg>\d+)_(?P<start>\d{4})_(?P<stop>\d{4})_(?P<descriptor>[^_]+)$")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mape_percent": float(np.mean(np.abs((y - pred) / y)) * 100.0),
        "r2": float(r2_score(y, pred)),
    }


def abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return 0.0
    return float(abs(np.corrcoef(x[mask], y[mask])[0, 1]))


def select_features(df: pd.DataFrame, prefix: str, local: bool) -> list[str]:
    selected: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col == f"{prefix}_window_complete":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique(dropna=True) <= 1:
            continue
        bucket = feature_bucket(col, prefix)
        if bucket in {"raw", "deltaq"} or (local and col.startswith(f"{prefix}_ldq_")):
            selected.append(col)
    return selected


def make_xgb(row: pd.Series, seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(float(row["n_estimators"])),
        max_depth=int(float(row["max_depth"])),
        learning_rate=float(row["learning_rate"]),
        subsample=float(row["subsample"]),
        colsample_bytree=float(row["colsample_bytree"]),
        reg_lambda=float(row["reg_lambda"]),
        reg_alpha=float(row["reg_alpha"]),
        min_child_weight=float(row["min_child_weight"]),
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=4,
    )


def run_baselines(args: argparse.Namespace, out: Path) -> None:
    df = pd.read_csv(args.top40_features)
    df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    prefix = detect_window_prefix(df)
    dev = df["official_split"].isin(["train", "primary_test"]).to_numpy()
    ext = (df["official_split"] == "secondary_test").to_numpy()
    y = df["knee_cycle"].to_numpy(float)
    params = pd.read_csv(args.xgb_params)
    specs = [
        ("Raw/DeltaQ + XGBoost", False, args.raw_candidate),
        ("Local DeltaQ + XGBoost", True, args.local_candidate),
    ]
    rows: list[dict[str, object]] = []
    constant = np.repeat(y[dev].mean(), ext.sum())
    rows.append({"model": "Development-mean constant", "seed": -1, **metrics(y[ext], constant)})
    for name, local, candidate in specs:
        cols = select_features(df, prefix, local)
        param_row = params.loc[params["candidate_id"] == candidate].iloc[0]
        for seed in range(args.seed_start, args.seed_start + args.seed_repeats):
            model = make_xgb(param_row, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(df.loc[dev, cols], y[dev])
            pred = np.asarray(model.predict(df.loc[ext, cols]), float)
            rows.append({"model": name, "seed": seed, **metrics(y[ext], pred)})
    by_seed = pd.DataFrame(rows)
    summary = (
        by_seed.groupby("model")[["mae", "rmse", "mape_percent", "r2"]]
        .agg(["mean", "std", "min", "median", "max"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    by_seed.to_csv(out / "baseline_metrics_by_seed.csv", index=False)
    summary.to_csv(out / "baseline_metrics_summary.csv", index=False)


def run_stability(args: argparse.Namespace, out: Path) -> None:
    df = pd.read_csv(args.full_features)
    df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    prefix = detect_window_prefix(df)
    train = df[df["official_split"] == "train"].copy().reset_index(drop=True)
    local_cols = sorted(
        c for c in df.columns
        if c.startswith(f"{prefix}_ldq_") and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1
    )
    y = train["knee_cycle"].to_numpy(float)
    full_scores = {c: abs_corr(train[c].to_numpy(float), y) for c in local_cols}
    reference = set(sorted(local_cols, key=lambda c: (-full_scores[c], c))[: args.top_k])
    counts = {c: 0 for c in local_cols}
    ranks = {c: [] for c in local_cols}
    jaccards: list[dict[str, float]] = []
    rng = np.random.default_rng(args.bootstrap_seed)
    n = len(train)
    for b in range(args.bootstrap_repeats):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        scored = sorted(local_cols, key=lambda c: (-abs_corr(train[c].to_numpy(float)[idx], yb), c))
        selected = set(scored[: args.top_k])
        for rank, col in enumerate(scored, 1):
            ranks[col].append(rank)
        for col in selected:
            counts[col] += 1
        union = reference | selected
        jaccards.append({"bootstrap": b, "jaccard_with_full_train_top40": len(reference & selected) / len(union)})

    rows: list[dict[str, object]] = []
    for col in local_cols:
        match = REGION_RE.search(col)
        region = match.groupdict() if match else {}
        rows.append(
            {
                "feature": col,
                "selected_in_full_train_top40": col in reference,
                "full_train_abs_corr": full_scores[col],
                "bootstrap_selection_frequency": counts[col] / args.bootstrap_repeats,
                "bootstrap_rank_median": float(np.median(ranks[col])),
                "bootstrap_rank_iqr": float(np.percentile(ranks[col], 75) - np.percentile(ranks[col], 25)),
                "seg": int(region["seg"]) if region else np.nan,
                "start": int(region["start"]) / 1000 if region else np.nan,
                "stop": int(region["stop"]) / 1000 if region else np.nan,
                "descriptor": region.get("descriptor", ""),
            }
        )
    feature_stability = pd.DataFrame(rows).sort_values(
        ["bootstrap_selection_frequency", "full_train_abs_corr"], ascending=False
    )
    region_stability = (
        feature_stability.groupby(["seg", "start", "stop"], as_index=False)
        .agg(
            max_selection_frequency=("bootstrap_selection_frequency", "max"),
            mean_selection_frequency=("bootstrap_selection_frequency", "mean"),
            descriptors_selected=("selected_in_full_train_top40", "sum"),
        )
        .sort_values(["max_selection_frequency", "mean_selection_frequency"], ascending=False)
    )
    jac = pd.DataFrame(jaccards)
    feature_stability.to_csv(out / "selected_feature_bootstrap_stability.csv", index=False)
    region_stability.to_csv(out / "selected_region_bootstrap_stability.csv", index=False)
    jac.to_csv(out / "selected_set_jaccard_by_bootstrap.csv", index=False)
    summary = {
        "selection_scope": "official training cells only",
        "n_training_cells": n,
        "candidate_local_features": len(local_cols),
        "top_k": args.top_k,
        "bootstrap_repeats": args.bootstrap_repeats,
        "jaccard_mean": float(jac.iloc[:, 1].mean()),
        "jaccard_std": float(jac.iloc[:, 1].std()),
        "jaccard_median": float(jac.iloc[:, 1].median()),
        "full_top40_frequency_mean": float(feature_stability.loc[feature_stability["selected_in_full_train_top40"], "bootstrap_selection_frequency"].mean()),
        "full_top40_frequency_median": float(feature_stability.loc[feature_stability["selected_in_full_train_top40"], "bootstrap_selection_frequency"].median()),
    }
    (out / "stability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top = region_stability.head(20).sort_values("start")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    centers = (top["start"] + top["stop"]) / 2
    ax.bar(centers, top["max_selection_frequency"], width=0.018, color="#2f6f8f", alpha=0.85)
    ax.set_xlabel("Normalized discharge-curve coordinate")
    ax.set_ylabel("Bootstrap selection frequency")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stability of the 20 most recurrent Local DeltaQ regions")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out / "selected_region_stability.pdf", bbox_inches="tight")
    fig.savefig(out / "selected_region_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top40-features", default="data/early_features_N150_local_deltaq_top40_20170630.csv")
    parser.add_argument("--full-features", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--xgb-params", default="results/hybrid_hpo_local_deltaq_top40_N150/xgb_params_train_top60.csv")
    parser.add_argument("--raw-candidate", default="xgb_hpo_0071")
    parser.add_argument("--local-candidate", default="xgb_hpo_0160")
    parser.add_argument("--output-dir", default="results/baseline_uncertainty_region_stability_N150")
    parser.add_argument("--seed-start", type=int, default=123)
    parser.add_argument("--seed-repeats", type=int, default=30)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_baselines(args, out)
    run_stability(args, out)
    print(f"Saved outputs to {out}")


if __name__ == "__main__":
    main()
