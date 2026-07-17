from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_feature_ablation import feature_bucket  # noqa: E402
from train_knee_baselines import META_COLUMNS, detect_window_prefix  # noqa: E402


SEG_RE = re.compile(r"_seg(?P<seg>\d+)_")


def reg_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mape_percent": float(np.mean(np.abs((y - pred) / y)) * 100.0),
        "r2": float(r2_score(y, pred)),
    }


def make_model(params: dict[str, object], seed: int) -> ExtraTreesRegressor:
    depth = params.get("max_depth")
    return ExtraTreesRegressor(
        n_estimators=int(float(params["n_estimators"])),
        max_features=params["max_features"],
        min_samples_leaf=int(float(params["min_samples_leaf"])),
        min_samples_split=int(float(params["min_samples_split"])),
        max_depth=None if depth is None else int(float(depth)),
        bootstrap=bool(params["bootstrap"]),
        random_state=seed,
        n_jobs=-1,
    )


def base_features(df: pd.DataFrame, prefix: str) -> list[str]:
    return [
        c for c in df.columns
        if c not in META_COLUMNS
        and c != f"{prefix}_window_complete"
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 1
        and feature_bucket(c, prefix) in {"raw", "deltaq"}
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-features", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--original-top40", default="data/early_features_N150_local_deltaq_top40_20170630.csv")
    parser.add_argument("--stability", default="results/baseline_uncertainty_region_stability_N150/selected_feature_bootstrap_stability.csv")
    parser.add_argument("--hpo-metadata", default="results/extratrees_ldq_hpo_N150_quick30/metadata.json")
    parser.add_argument("--output-dir", default="results/stability_selected_ldq_N150")
    parser.add_argument("--seed-start", type=int, default=123)
    parser.add_argument("--seed-repeats", type=int, default=30)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.full_features)
    df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    original = pd.read_csv(args.original_top40)
    prefix = detect_window_prefix(df)
    stability = pd.read_csv(args.stability).sort_values(
        ["bootstrap_selection_frequency", "full_train_abs_corr"], ascending=False
    )
    stable40 = stability.head(40)["feature"].tolist()
    original40 = [c for c in original.columns if c.startswith(f"{prefix}_ldq_")]

    # Limit descriptor duplication within a segment, then fill remaining slots by stability.
    balanced: list[str] = []
    per_seg: dict[int, int] = {}
    for feature in stability["feature"]:
        match = SEG_RE.search(feature)
        seg = int(match.group("seg")) if match else -1
        if per_seg.get(seg, 0) < 2:
            balanced.append(feature)
            per_seg[seg] = per_seg.get(seg, 0) + 1
        if len(balanced) == 40:
            break

    base = base_features(df, prefix)
    candidates = {
        "original_corr_top40": original40,
        "bootstrap_stable_top40": stable40,
        "region_balanced_stable_top40": balanced,
    }
    params = json.loads(Path(args.hpo_metadata).read_text(encoding="utf-8"))["selected_params"]
    y = df["knee_cycle"].to_numpy(float)
    dev = df["official_split"].isin(["train", "primary_test"]).to_numpy()
    ext = (df["official_split"] == "secondary_test").to_numpy()
    y_dev = y[dev]
    cv_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []

    splitter = RepeatedKFold(n_splits=5, n_repeats=3, random_state=20260712)
    for name, ldq in candidates.items():
        features = base + ldq
        x_dev = df.loc[dev, features].reset_index(drop=True)
        fold_mae: list[float] = []
        for fold, (tr, va) in enumerate(splitter.split(x_dev)):
            model = make_model(params, 1000 + fold)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_dev.iloc[tr], y_dev[tr])
            pred = model.predict(x_dev.iloc[va])
            fold_mae.append(float(mean_absolute_error(y_dev[va], pred)))
        cv_rows.append(
            {
                "candidate": name,
                "dev_cv_mae_mean": float(np.mean(fold_mae)),
                "dev_cv_mae_std": float(np.std(fold_mae)),
                "dev_selection_score": float(np.mean(fold_mae) + 0.1 * np.std(fold_mae)),
            }
        )
        for seed in range(args.seed_start, args.seed_start + args.seed_repeats):
            model = make_model(params, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(df.loc[dev, features], y[dev])
            pred = np.asarray(model.predict(df.loc[ext, features]), float)
            final_rows.append({"candidate": name, "seed": seed, **reg_metrics(y[ext], pred)})
            for cell, observed, estimate in zip(df.loc[ext, "cell_id"], y[ext], pred):
                pred_rows.append(
                    {"candidate": name, "seed": seed, "cell_id": cell, "observed": observed, "predicted": estimate}
                )
        pd.DataFrame({"feature": ldq}).to_csv(out / f"features_{name}.csv", index=False)

    cv = pd.DataFrame(cv_rows).sort_values("dev_selection_score")
    final = pd.DataFrame(final_rows)
    summary = final.groupby("candidate")[["mae", "rmse", "mape_percent", "r2"]].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    cv.to_csv(out / "development_cv_ranking.csv", index=False)
    final.to_csv(out / "external_metrics_by_seed.csv", index=False)
    summary.to_csv(out / "external_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "external_predictions.csv", index=False)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "selection_rule": "Candidates ranked only by repeated CV on official train + primary_test; secondary labels excluded from selection.",
                "selected_by_development_cv": str(cv.iloc[0]["candidate"]),
                "candidate_feature_source": "official-training-only bootstrap stability",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(cv.round(4).to_string(index=False))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
