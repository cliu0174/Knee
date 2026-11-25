from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_nested_contiguous_ldq as core


def local_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted([
        col for col in frame.columns
        if col.startswith(f"{prefix}_ldq_")
        and pd.api.types.is_numeric_dtype(frame[col])
        and frame[col].nunique(dropna=True) > 1
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal Local DeltaQ cycle-pair scan with fixed N150 descriptors")
    parser.add_argument("--reference", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--pair-features", nargs="+", default=[
        "data/early_features_N50_local_deltaq_20170630.csv",
        "data/early_features_N100_local_deltaq_20170630.csv",
        "data/early_features_N150_local_deltaq_20170630.csv",
    ])
    parser.add_argument("--frozen-config", default="results/formal_414240_selector_N150/frozen_config.json")
    parser.add_argument("--output-dir", default="results/formal_414240_cycle_pair_scan")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=123)
    parser.add_argument("--seed-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=20260721)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path(args.reference)
    frozen_path = Path(args.frozen_config)
    frozen: dict[str, Any] = json.loads(frozen_path.read_text(encoding="utf-8"))
    reference = pd.read_csv(reference_path)
    reference = reference[reference["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    base = list(frozen["base_features"])
    keep = ["cell_id", "batch", "official_split", "knee_cycle"] + base
    fixed = reference[keep].copy()

    cv_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    input_meta: list[dict[str, Any]] = []
    for pair_path_text in args.pair_features:
        pair_path = Path(pair_path_text)
        source = pd.read_csv(pair_path)
        prefix = core.detect_window_prefix(source)
        local = local_columns(source, prefix)
        frame = fixed.merge(source[["cell_id"] + local], on="cell_id", how="inner", validate="one_to_one")
        counts = frame["official_split"].value_counts().to_dict()
        expected = {"train": 41, "primary_test": 42, "secondary_test": 40}
        if counts != expected:
            raise RuntimeError(f"Cycle-pair common cohort mismatch for {prefix}: {counts}")
        pair_label = {"N50": "10-50", "N100": "10-100", "N150": "10-150"}[prefix]
        train = frame[frame.official_split == "train"].copy().reset_index(drop=True)
        y = train["knee_cycle"].to_numpy(float)
        folds = core.split_indices(train["batch"], y, args.cv_splits, args.cv_repeats, args.random_state)
        for fold_id, (tr, va) in enumerate(folds):
            selector = core.fit_selector(
                train.iloc[tr], y[tr], local, frozen["method"], frozen["selector_config"],
                args.random_state + 1009 * (fold_id + 1),
            )
            used = base + selector.selected
            x_train, medians = core.safe_numeric_matrix(train.iloc[tr], used)
            x_valid, _ = core.safe_numeric_matrix(train.iloc[va], used, medians)
            model = core.make_model(frozen["model_config"], args.random_state + fold_id)
            model.fit(x_train, y[tr])
            pred = np.asarray(model.predict(x_valid), float)
            cv_rows.append({
                "cycle_pair": pair_label, "outer_id": fold_id, "n_train": len(tr), "n_valid": len(va),
                **core.regression_metrics(y[va], pred),
            })

        selector = core.fit_selector(
            train, y, local, frozen["method"], frozen["selector_config"], args.random_state + 515151
        )
        used = base + selector.selected
        for rank, feature in enumerate(selector.selected, 1):
            selected_rows.append({"cycle_pair": pair_label, "rank": rank, "feature": feature,
                                  **(core.parse_region(feature) or {})})
        x_train, medians = core.safe_numeric_matrix(train, used)
        for split in ["primary_test", "secondary_test"]:
            test = frame[frame.official_split == split].copy().reset_index(drop=True)
            x_test, _ = core.safe_numeric_matrix(test, used, medians)
            y_test = test["knee_cycle"].to_numpy(float)
            for seed in range(args.seed_start, args.seed_start + args.seed_repeats):
                model = core.make_model(frozen["model_config"], seed)
                model.fit(x_train, y)
                pred = np.asarray(model.predict(x_test), float)
                metric_rows.append({"cycle_pair": pair_label, "split": split, "seed": seed, "n": len(test),
                                    **core.regression_metrics(y_test, pred)})
                for idx, estimate in enumerate(pred):
                    pred_rows.append({"cycle_pair": pair_label, "split": split, "seed": seed,
                                      "cell_id": test.iloc[idx]["cell_id"], "observed": float(y_test[idx]),
                                      "predicted": float(estimate),
                                      "abs_error": abs(float(y_test[idx]) - float(estimate))})
        input_meta.append({"cycle_pair": pair_label, "source": str(pair_path),
                           "source_sha256": core.sha256_file(pair_path), "local_candidates": len(local)})
        print(f"cycle pair {pair_label} complete", flush=True)

    cv = pd.DataFrame(cv_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(pred_rows)
    selected = pd.DataFrame(selected_rows)
    cv.to_csv(output / "cycle_pair_train_cv.csv", index=False)
    metrics.to_csv(output / "cycle_pair_heldout_metrics.csv", index=False)
    predictions.to_csv(output / "cycle_pair_heldout_predictions.csv", index=False)
    selected.to_csv(output / "cycle_pair_selected_features.csv", index=False)
    metadata = {
        "created_utc": core.utc_now(), "fixed_descriptor_source": str(reference_path),
        "fixed_descriptor_source_sha256": core.sha256_file(reference_path),
        "frozen_config_sha256": core.sha256_file(frozen_path), "pair_inputs": input_meta,
        "design": "N150 raw/global descriptors held fixed; only Local DeltaQ cycle pair changes",
        "test_results_used_for_pair_or_model_selection": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.groupby(["cycle_pair", "split"])[["mae", "rmse", "r2"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
