from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detect_knee_points import detect_piecewise_knee  # noqa: E402
from run_feature_ablation import feature_bucket  # noqa: E402
from train_knee_baselines import build_models, detect_window_prefix, regression_metrics  # noqa: E402


META_COLUMNS = {
    "cell_id",
    "batch",
    "official_order",
    "official_split",
    "cycle_life",
    "knee_cycle",
    "knee_soh",
    "relative_knee",
    "label_confidence",
    "knee_after_window",
    "remaining_cycles_from_window",
}


def normalized(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def detect_kneedle_knee(
    cycles: np.ndarray,
    soh: np.ndarray,
    min_frac: float,
    max_frac: float,
    min_segment_points: int,
) -> dict[str, object]:
    order = np.argsort(cycles)
    x = cycles[order].astype(float)
    y = soh[order].astype(float)
    n = x.size
    if n < 2 * min_segment_points + 1:
        return {"status": "too_short", "knee_cycle": np.nan, "knee_index": np.nan, "knee_soh": np.nan, "label_confidence": np.nan}

    low = max(min_segment_points, int(np.floor(n * min_frac)))
    high = min(n - min_segment_points, int(np.ceil(n * max_frac)))
    x_norm = normalized(x)
    y_norm = normalized(y)

    x0, y0 = x_norm[0], y_norm[0]
    x1, y1 = x_norm[-1], y_norm[-1]
    denom = np.hypot(y1 - y0, x1 - x0)
    if denom == 0:
        return {"status": "flat_curve", "knee_cycle": np.nan, "knee_index": np.nan, "knee_soh": np.nan, "label_confidence": np.nan}
    distance = np.abs((y1 - y0) * x_norm - (x1 - x0) * y_norm + x1 * y0 - y1 * x0) / denom
    candidate = np.arange(low, high)
    split = int(candidate[np.argmax(distance[candidate])])
    return {
        "status": "ok",
        "knee_cycle": float(x[split]),
        "knee_index": split,
        "knee_soh": float(y[split]),
        "label_confidence": float(distance[split]),
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values.astype(float), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_rate_threshold_knee(
    cycles: np.ndarray,
    soh: np.ndarray,
    min_frac: float,
    max_frac: float,
    min_segment_points: int,
    rate_window_frac: float = 0.035,
    threshold_fraction: float = 0.40,
) -> dict[str, object]:
    order = np.argsort(cycles)
    x = cycles[order].astype(float)
    y = soh[order].astype(float)
    n = x.size
    if n < 2 * min_segment_points + 1:
        return {"status": "too_short", "knee_cycle": np.nan, "knee_index": np.nan, "knee_soh": np.nan, "label_confidence": np.nan}

    low = max(min_segment_points, int(np.floor(n * min_frac)))
    high = min(n - min_segment_points, int(np.ceil(n * max_frac)))
    raw_rate = -np.gradient(y, x)
    window = max(9, int(round(n * rate_window_frac)))
    if window % 2 == 0:
        window += 1
    rate = rolling_mean(raw_rate, window)
    candidate = np.arange(low, high)
    early = rate[min_segment_points:low] if low > min_segment_points else rate[:low]
    if early.size == 0:
        early = rate[: max(1, low)]
    baseline = float(np.nanmedian(early))
    high_rate = float(np.nanpercentile(rate[candidate], 90))
    threshold = baseline + threshold_fraction * max(0.0, high_rate - baseline)
    hits = candidate[rate[candidate] >= threshold]
    split = int(hits[0]) if hits.size else int(candidate[np.nanargmax(rate[candidate])])
    confidence = (float(rate[split]) - baseline) / max(abs(baseline), 1e-9)
    return {
        "status": "ok",
        "knee_cycle": float(x[split]),
        "knee_index": split,
        "knee_soh": float(y[split]),
        "label_confidence": float(confidence),
        "rate_baseline": baseline,
        "rate_threshold": threshold,
        "rate_at_knee": float(rate[split]),
    }


def label_row(method: str, cell_id: str, group: pd.DataFrame, result: dict[str, object]) -> dict[str, object]:
    cycle_life = float(group["cycle_life"].iloc[0])
    knee_cycle = float(result["knee_cycle"]) if np.isfinite(result["knee_cycle"]) else np.nan
    return {
        "method": method,
        "cell_id": cell_id,
        "batch": group["batch"].iloc[0],
        "official_order": int(group["official_order"].iloc[0]),
        "official_split": group["official_split"].iloc[0],
        "cycle_life": cycle_life,
        "knee_cycle": knee_cycle,
        "knee_soh": float(result["knee_soh"]) if np.isfinite(result["knee_soh"]) else np.nan,
        "relative_knee": float(knee_cycle / cycle_life) if np.isfinite(knee_cycle) and cycle_life > 0 else np.nan,
        "label_confidence": float(result["label_confidence"]) if np.isfinite(result["label_confidence"]) else np.nan,
        "status": result["status"],
    }


def detect_all_labels(cycles: pd.DataFrame, min_frac: float, max_frac: float, min_segment_points: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_id, group in cycles.groupby("cell_id", sort=False):
        valid = group["soh_mono"].notna()
        x = group.loc[valid, "cycle"].to_numpy(dtype=float)
        y = group.loc[valid, "soh_mono"].to_numpy(dtype=float)
        methods = {
            "piecewise": detect_piecewise_knee(x, y, min_frac, max_frac, min_segment_points),
            "kneedle": detect_kneedle_knee(x, y, min_frac, max_frac, min_segment_points),
            "rate_threshold": detect_rate_threshold_knee(x, y, min_frac, max_frac, min_segment_points),
        }
        for method, result in methods.items():
            rows.append(label_row(method, cell_id, group, result))
    return pd.DataFrame(rows)


def label_comparison(labels: pd.DataFrame, reference_method: str = "piecewise") -> pd.DataFrame:
    pivot = labels.pivot(index="cell_id", columns="method", values="knee_cycle")
    rows: list[dict[str, object]] = []
    for method in pivot.columns:
        if method == reference_method:
            continue
        valid = pivot[[reference_method, method]].dropna()
        diff = valid[method] - valid[reference_method]
        rows.append(
            {
                "reference": reference_method,
                "method": method,
                "n": int(len(valid)),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "median_abs_diff": float(np.median(np.abs(diff))),
                "mean_signed_diff": float(np.mean(diff)),
                "pearson": float(valid[reference_method].corr(valid[method], method="pearson")),
                "spearman": float(valid[reference_method].corr(valid[method], method="spearman")),
                "within_50_cycles": float((np.abs(diff) <= 50).mean()),
                "within_100_cycles": float((np.abs(diff) <= 100).mean()),
            }
        )
    return pd.DataFrame(rows)


def select_raw_deltaq_features(df: pd.DataFrame, prefix: str) -> list[str]:
    features: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col == f"{prefix}_window_complete":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        if feature_bucket(col, prefix) in {"raw", "deltaq"}:
            features.append(col)
    return features


def train_candidate(
    candidate_name: str,
    feature_path: Path,
    model_name: str,
    labels: pd.DataFrame,
    method: str,
    random_state: int,
) -> list[dict[str, object]]:
    features_df = pd.read_csv(feature_path)
    prefix = detect_window_prefix(features_df)
    window = int(prefix[1:])
    label_part = labels[labels["method"] == method][["cell_id", "knee_cycle", "knee_soh", "relative_knee", "label_confidence", "status"]]
    df = features_df.drop(columns=["knee_cycle", "knee_soh", "relative_knee", "label_confidence"], errors="ignore").merge(
        label_part,
        on="cell_id",
        how="inner",
    )
    df["knee_after_window"] = df["knee_cycle"] > window
    before = len(df)
    df = df[(df["status"] == "ok") & df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)

    selected_features = select_raw_deltaq_features(df, prefix)
    x = df[selected_features]
    y = df["knee_cycle"].to_numpy(dtype=float)
    train_mask = df["official_split"] == "train"
    primary_mask = df["official_split"] == "primary_test"
    secondary_mask = df["official_split"] == "secondary_test"
    models = build_models(random_state)
    model = models[model_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x.loc[train_mask], y[train_mask.to_numpy()])
    pred = model.predict(x)

    rows: list[dict[str, object]] = []
    for split_name, mask in [("train", train_mask), ("primary_test", primary_mask), ("secondary_test", secondary_mask)]:
        idx = mask.to_numpy()
        row: dict[str, object] = {
            "candidate": candidate_name,
            "label_method": method,
            "window": prefix,
            "model": model_name,
            "feature_group": "raw_deltaq",
            "feature_count": len(selected_features),
            "input_rows": before,
            "used_rows": len(df),
            "split": split_name,
            "n": int(idx.sum()),
        }
        row.update(regression_metrics("knee_cycle", y[idx], pred[idx]))
        rows.append(row)
    return rows


def run_candidate_robustness(labels: pd.DataFrame, random_state: int) -> pd.DataFrame:
    candidates = [
        {
            "candidate_name": "performance_candidate",
            "feature_path": Path("data/early_features_N150_extended_20170630.csv"),
            "model_name": "xgboost",
        },
        {
            "candidate_name": "early_warning_candidate",
            "feature_path": Path("data/early_features_N100_extended_20170630.csv"),
            "model_name": "random_forest",
        },
    ]
    rows: list[dict[str, object]] = []
    for method in ["piecewise", "kneedle", "rate_threshold"]:
        for candidate in candidates:
            rows.extend(
                train_candidate(
                    candidate["candidate_name"],
                    candidate["feature_path"],
                    candidate["model_name"],
                    labels,
                    method,
                    random_state,
                )
            )
    return pd.DataFrame(rows)


def plot_label_scatter(labels: pd.DataFrame, output_dir: Path) -> None:
    pivot = labels.pivot(index="cell_id", columns="method", values="knee_cycle")
    methods = [m for m in ["kneedle", "rate_threshold"] if m in pivot.columns]
    fig, axes = plt.subplots(1, len(methods), figsize=(6.1 * len(methods), 5.2), squeeze=False)
    for ax, method in zip(axes.ravel(), methods):
        valid = pivot[["piecewise", method]].dropna()
        ax.scatter(valid["piecewise"], valid[method], s=38, alpha=0.82, color="#4C78A8")
        low = float(valid.min().min())
        high = float(valid.max().max())
        ax.plot([low, high], [low, high], color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(f"{method} vs piecewise")
        ax.set_xlabel("Piecewise knee cycle")
        ax.set_ylabel(f"{method} knee cycle")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "label_method_scatter.png", dpi=220)
    plt.close(fig)


def plot_model_robustness(metrics: pd.DataFrame, output_dir: Path) -> None:
    plot_df = metrics[metrics["split"].isin(["primary_test", "secondary_test"])].copy()
    candidates = plot_df["candidate"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(candidates), figsize=(6.8 * len(candidates), 5.2), squeeze=False)
    methods = ["piecewise", "kneedle", "rate_threshold"]
    colors = {"primary_test": "#4C78A8", "secondary_test": "#F58518"}
    x = np.arange(len(methods))
    width = 0.36
    for ax, candidate in zip(axes.ravel(), candidates):
        part = plot_df[plot_df["candidate"] == candidate]
        for offset, split in [(-width / 2, "primary_test"), (width / 2, "secondary_test")]:
            values = []
            for method in methods:
                hit = part[(part["label_method"] == method) & (part["split"] == split)]
                values.append(float(hit["knee_cycle_mae"].iloc[0]) if not hit.empty else np.nan)
            ax.bar(x + offset, values, width=width, label=split, color=colors[split])
        ax.set_title(candidate)
        ax.set_ylabel("MAE (cycles)")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "candidate_label_robustness_mae.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", default="data/processed_cycles_soh_corrected_20170630.csv")
    parser.add_argument("--output-dir", default="results/knee_label_stability")
    parser.add_argument("--min-frac", type=float, default=0.2)
    parser.add_argument("--max-frac", type=float, default=0.9)
    parser.add_argument("--min-segment-points", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cycles = pd.read_csv(args.cycles)
    labels = detect_all_labels(cycles, args.min_frac, args.max_frac, args.min_segment_points)
    comparison = label_comparison(labels)
    robustness = run_candidate_robustness(labels, args.random_state)

    labels.to_csv(output_dir / "knee_labels_by_method.csv", index=False)
    comparison.to_csv(output_dir / "label_method_comparison.csv", index=False)
    robustness.to_csv(output_dir / "candidate_model_label_robustness.csv", index=False)
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "cycles": args.cycles,
                "methods": ["piecewise", "kneedle", "rate_threshold"],
                "min_frac": args.min_frac,
                "max_frac": args.max_frac,
                "min_segment_points": args.min_segment_points,
                "candidate_models": [
                    "N150 raw_deltaq XGBoost",
                    "N100 raw_deltaq RandomForest",
                ],
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )
    plot_label_scatter(labels, output_dir)
    plot_model_robustness(robustness, output_dir)

    print(f"Saved label stability outputs to {output_dir}")
    print("\nLabel method comparison:")
    print(comparison.round(4).to_string(index=False))
    print("\nCandidate model robustness:")
    print(
        robustness[robustness["split"].isin(["primary_test", "secondary_test"])][
            [
                "candidate",
                "label_method",
                "window",
                "model",
                "split",
                "n",
                "knee_cycle_mae",
                "knee_cycle_r2",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
