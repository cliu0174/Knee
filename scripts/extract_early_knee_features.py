from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def linear_slope(cycles: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(cycles) & np.isfinite(values)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(cycles[mask], values[mask], 1)
    return float(slope), float(intercept)


def regeneration_metrics(values: np.ndarray, threshold: float) -> tuple[int, float, float]:
    clean = finite(values)
    if clean.size < 2:
        return 0, 0.0, 0.0
    diffs = np.diff(clean)
    increases = diffs[diffs > threshold]
    if increases.size == 0:
        return 0, 0.0, 0.0
    return int(increases.size), float(np.sum(increases)), float(np.max(increases))


def window_features(group: pd.DataFrame, window: int, regen_threshold: float) -> dict[str, float | int | bool]:
    prefix = f"N{window}"
    part = group[(group["cycle"] <= window) & group["is_positive_capacity"].astype(bool)].copy()
    cycles = part["cycle"].to_numpy(dtype=float)
    qd = part["qd"].to_numpy(dtype=float)
    soh_raw = part["soh_raw"].to_numpy(dtype=float)
    soh_smooth = part["soh_smooth"].to_numpy(dtype=float)
    soh_mono = part["soh_mono"].to_numpy(dtype=float)

    qd_clean = finite(qd)
    raw_clean = finite(soh_raw)
    smooth_clean = finite(soh_smooth)
    mono_clean = finite(soh_mono)
    raw_slope, raw_intercept = linear_slope(cycles, soh_raw)
    smooth_slope, smooth_intercept = linear_slope(cycles, soh_smooth)
    mono_slope, mono_intercept = linear_slope(cycles, soh_mono)
    raw_regen_count, raw_regen_amp, raw_regen_max = regeneration_metrics(soh_raw, regen_threshold)
    smooth_regen_count, smooth_regen_amp, smooth_regen_max = regeneration_metrics(soh_smooth, regen_threshold)

    def first_or_nan(arr: np.ndarray) -> float:
        return float(arr[0]) if arr.size else float("nan")

    def last_or_nan(arr: np.ndarray) -> float:
        return float(arr[-1]) if arr.size else float("nan")

    features: dict[str, float | int | bool] = {
        f"{prefix}_n_observed": int(len(part)),
        f"{prefix}_window_complete": bool(len(part) >= window - 1),
        f"{prefix}_qd_mean": float(np.nanmean(qd_clean)) if qd_clean.size else float("nan"),
        f"{prefix}_qd_std": float(np.nanstd(qd_clean, ddof=1)) if qd_clean.size > 1 else float("nan"),
        f"{prefix}_qd_min": float(np.nanmin(qd_clean)) if qd_clean.size else float("nan"),
        f"{prefix}_qd_max": float(np.nanmax(qd_clean)) if qd_clean.size else float("nan"),
        f"{prefix}_qd_first": first_or_nan(qd_clean),
        f"{prefix}_qd_last": last_or_nan(qd_clean),
        f"{prefix}_qd_range": float(np.nanmax(qd_clean) - np.nanmin(qd_clean)) if qd_clean.size else float("nan"),
        f"{prefix}_qd_change": float(last_or_nan(qd_clean) - first_or_nan(qd_clean)) if qd_clean.size else float("nan"),
        f"{prefix}_qd_ratio_last_first": (
            float(last_or_nan(qd_clean) / first_or_nan(qd_clean))
            if qd_clean.size and first_or_nan(qd_clean) != 0
            else float("nan")
        ),
        f"{prefix}_soh_raw_mean": float(np.nanmean(raw_clean)) if raw_clean.size else float("nan"),
        f"{prefix}_soh_raw_std": float(np.nanstd(raw_clean, ddof=1)) if raw_clean.size > 1 else float("nan"),
        f"{prefix}_soh_raw_min": float(np.nanmin(raw_clean)) if raw_clean.size else float("nan"),
        f"{prefix}_soh_raw_max": float(np.nanmax(raw_clean)) if raw_clean.size else float("nan"),
        f"{prefix}_soh_raw_first": first_or_nan(raw_clean),
        f"{prefix}_soh_raw_last": last_or_nan(raw_clean),
        f"{prefix}_soh_raw_change": float(last_or_nan(raw_clean) - first_or_nan(raw_clean)) if raw_clean.size else float("nan"),
        f"{prefix}_soh_raw_slope": raw_slope,
        f"{prefix}_soh_raw_intercept": raw_intercept,
        f"{prefix}_soh_smooth_last": last_or_nan(smooth_clean),
        f"{prefix}_soh_smooth_change": (
            float(last_or_nan(smooth_clean) - first_or_nan(smooth_clean)) if smooth_clean.size else float("nan")
        ),
        f"{prefix}_soh_smooth_slope": smooth_slope,
        f"{prefix}_soh_smooth_intercept": smooth_intercept,
        f"{prefix}_soh_mono_last": last_or_nan(mono_clean),
        f"{prefix}_soh_mono_change": float(last_or_nan(mono_clean) - first_or_nan(mono_clean)) if mono_clean.size else float("nan"),
        f"{prefix}_soh_mono_slope": mono_slope,
        f"{prefix}_soh_mono_intercept": mono_intercept,
        f"{prefix}_raw_regen_count": raw_regen_count,
        f"{prefix}_raw_regen_amplitude": raw_regen_amp,
        f"{prefix}_raw_max_single_cycle_increase": raw_regen_max,
        f"{prefix}_smooth_regen_count": smooth_regen_count,
        f"{prefix}_smooth_regen_amplitude": smooth_regen_amp,
        f"{prefix}_smooth_max_single_cycle_increase": smooth_regen_max,
        f"{prefix}_raw_minus_mono_mean_abs": (
            float(np.nanmean(np.abs(soh_raw - soh_mono))) if raw_clean.size and mono_clean.size else float("nan")
        ),
        f"{prefix}_smooth_minus_mono_mean_abs": (
            float(np.nanmean(np.abs(soh_smooth - soh_mono))) if smooth_clean.size and mono_clean.size else float("nan")
        ),
    }
    return features


def plot_feature_qc(features: pd.DataFrame, output_dir: Path, window: int) -> None:
    prefix = f"N{window}"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].hist(features["knee_cycle"], bins=24, color="#4C78A8", alpha=0.85)
    axes[0].set_title("Target: knee_cycle")
    axes[0].set_xlabel("Cycle")
    axes[1].hist(features[f"{prefix}_soh_raw_slope"], bins=24, color="#F58518", alpha=0.85)
    axes[1].set_title(f"{prefix} raw SOH slope")
    axes[1].set_xlabel("Slope")
    axes[2].scatter(features[f"{prefix}_soh_raw_slope"], features["knee_cycle"], s=30, alpha=0.8, color="#54A24B")
    axes[2].set_title("Early SOH slope vs knee_cycle")
    axes[2].set_xlabel(f"{prefix} raw SOH slope")
    axes[2].set_ylabel("knee_cycle")
    axes[3].scatter(features["cycle_life"], features["knee_cycle"], s=30, alpha=0.8, color="#E45756")
    axes[3].set_title("cycle_life vs knee_cycle")
    axes[3].set_xlabel("cycle_life")
    axes[3].set_ylabel("knee_cycle")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"early_features_N{window}_qc.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", default="data/processed_cycles_soh_corrected_20170630.csv")
    parser.add_argument("--labels", default="data/knee_labels_20170630.csv")
    parser.add_argument("--output", default="data/early_features_N100_20170630.csv")
    parser.add_argument("--qc-output", default="data/early_features_N100_qc_20170630.csv")
    parser.add_argument("--output-dir", default="results/early_features_N100")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--regen-threshold", type=float, default=1e-4)
    args = parser.parse_args()

    cycles = pd.read_csv(args.cycles)
    labels = pd.read_csv(args.labels)
    rows: list[dict[str, object]] = []
    for cell_id, group in cycles.groupby("cell_id", sort=False):
        label = labels[labels["cell_id"] == cell_id].iloc[0].to_dict()
        row: dict[str, object] = {
            "cell_id": cell_id,
            "batch": group["batch"].iloc[0],
            "official_order": int(group["official_order"].iloc[0]),
            "official_split": group["official_split"].iloc[0],
            "cycle_life": float(group["cycle_life"].iloc[0]),
            "knee_cycle": float(label["knee_cycle"]),
            "knee_soh": float(label["knee_soh"]),
            "relative_knee": float(label["relative_knee"]),
            "label_confidence": float(label["label_confidence"]),
            "knee_after_window": bool(float(label["knee_cycle"]) > args.window),
            "remaining_cycles_from_window": float(label["knee_cycle"]) - args.window,
        }
        row.update(window_features(group, args.window, args.regen_threshold))
        rows.append(row)

    features = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)

    feature_cols = [
        c
        for c in features.columns
        if c
        not in {
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
    ]
    qc_rows = []
    for col in feature_cols:
        qc_rows.append(
            {
                "feature": col,
                "missing_count": int(features[col].isna().sum()),
                "missing_fraction": float(features[col].isna().mean()),
                "n_unique": int(features[col].nunique(dropna=True)),
            }
        )
    qc = pd.DataFrame(qc_rows).sort_values(["missing_count", "feature"], ascending=[False, True])
    qc_output = Path(args.qc_output)
    qc.to_csv(qc_output, index=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_feature_qc(features, output_dir, args.window)

    print(f"Saved early features: {len(features)} rows, {features.shape[1]} columns -> {output}")
    print(f"Saved feature QC: {len(qc)} rows -> {qc_output}")
    print(features[["knee_cycle", "knee_after_window", f"N{args.window}_n_observed", f"N{args.window}_window_complete"]].describe(include="all").to_string())
    print(features["knee_after_window"].value_counts().to_string())


if __name__ == "__main__":
    main()
