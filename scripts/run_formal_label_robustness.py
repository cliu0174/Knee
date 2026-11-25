from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_nested_contiguous_ldq as core


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only frozen-protocol knee-label robustness analysis")
    parser.add_argument("--features", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--labels", default="results/knee_label_stability/knee_labels_by_method.csv")
    parser.add_argument("--frozen-config", default="results/formal_414240_selector_N150/frozen_config.json")
    parser.add_argument("--output-dir", default="results/formal_414240_label_robustness")
    parser.add_argument("--methods", nargs="+", default=["piecewise", "kneedle", "rate_threshold"])
    parser.add_argument("--seed-start", type=int, default=123)
    parser.add_argument("--seed-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=20260721)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_path = Path(args.features)
    label_path = Path(args.labels)
    frozen_path = Path(args.frozen_config)
    features = pd.read_csv(feature_path)
    labels = pd.read_csv(label_path)
    frozen: dict[str, Any] = json.loads(frozen_path.read_text(encoding="utf-8"))
    labels = labels[labels["method"].isin(args.methods)].copy()

    wide = labels.pivot(index="cell_id", columns="method", values="knee_cycle")
    eligible = set(wide.dropna().index.astype(str))
    for method in args.methods:
        eligible &= set(wide.index[wide[method] > 150].astype(str))
    eligible &= set(features.loc[features["knee_after_window"].astype(bool), "cell_id"].astype(str))
    frame = features[features["cell_id"].astype(str).isin(eligible)].copy().reset_index(drop=True)
    counts = frame["official_split"].value_counts().to_dict()
    if min(counts.get(name, 0) for name in ["train", "primary_test", "secondary_test"]) < 35:
        raise RuntimeError(f"Alternative-label common cohort unexpectedly small: {counts}")

    prefix = core.detect_window_prefix(frame)
    base, local = core.base_and_local_features(frame, prefix)
    label_lookup = labels.set_index(["method", "cell_id"])["knee_cycle"]
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for label_method in args.methods:
        current = frame.copy()
        current["analysis_target"] = [
            float(label_lookup.loc[(label_method, str(cell_id))]) for cell_id in current["cell_id"]
        ]
        train = current[current["official_split"] == "train"].copy().reset_index(drop=True)
        selector = core.fit_selector(
            train,
            train["analysis_target"].to_numpy(float),
            local,
            frozen["method"],
            frozen["selector_config"],
            args.random_state + 515151,
        )
        used_features = base + selector.selected
        x_train, medians = core.safe_numeric_matrix(train, used_features)
        y_train = train["analysis_target"].to_numpy(float)
        for split in ["primary_test", "secondary_test"]:
            test = current[current["official_split"] == split].copy().reset_index(drop=True)
            x_test, _ = core.safe_numeric_matrix(test, used_features, medians)
            y_test = test["analysis_target"].to_numpy(float)
            for seed in range(args.seed_start, args.seed_start + args.seed_repeats):
                model = core.make_model(frozen["model_config"], seed)
                model.fit(x_train, y_train)
                pred = np.asarray(model.predict(x_test), float)
                metric_rows.append({
                    "label_method": label_method,
                    "split": split,
                    "seed": seed,
                    "n": len(test),
                    "feature_count": len(used_features),
                    **core.regression_metrics(y_test, pred),
                })
                for idx, estimate in enumerate(pred):
                    prediction_rows.append({
                        "label_method": label_method,
                        "split": split,
                        "seed": seed,
                        "cell_id": test.iloc[idx]["cell_id"],
                        "observed": float(y_test[idx]),
                        "predicted": float(estimate),
                        "abs_error": abs(float(y_test[idx]) - float(estimate)),
                    })
        print(f"label robustness {label_method} complete", flush=True)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    label_common = labels[labels["cell_id"].astype(str).isin(eligible)].copy()
    label_common.to_csv(output / "common_cohort_labels.csv", index=False)
    metrics.to_csv(output / "label_robustness_metrics.csv", index=False)
    predictions.to_csv(output / "label_robustness_predictions.csv", index=False)
    metadata = {
        "created_utc": core.utc_now(),
        "feature_sha256": core.sha256_file(feature_path),
        "label_sha256": core.sha256_file(label_path),
        "frozen_config_sha256": core.sha256_file(frozen_path),
        "label_methods": args.methods,
        "common_cohort_counts": counts,
        "common_cohort_rule": "feature knee_after_window and every alternative knee_cycle > 150",
        "selector_and_model_family_reselected_on_test": False,
        "selector_refit_with_each_training-label_definition": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.groupby(["label_method", "split"])[["mae", "rmse", "r2"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
