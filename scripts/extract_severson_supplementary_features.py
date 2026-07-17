from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from extract_severson_official_features import (
    get_cycle,
    load_combined_batches,
    official_split,
    safe_summary,
    safe_summary_mean,
    scalar,
)


def summary_values(summary: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(summary[key], dtype=float).reshape(-1)



def sample_skewness(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = clean.size
    if n < 3:
        return float("nan")
    centered = clean - np.mean(clean)
    std = np.std(clean, ddof=1)
    if std == 0:
        return float("nan")
    return float((n / ((n - 1) * (n - 2))) * np.sum((centered / std) ** 3))


def sample_kurtosis(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = clean.size
    if n < 4:
        return float("nan")
    centered = clean - np.mean(clean)
    std = np.std(clean, ddof=1)
    if std == 0:
        return float("nan")
    z4_sum = np.sum((centered / std) ** 4)
    return float(
        (n * (n + 1) * z4_sum / ((n - 1) * (n - 2) * (n - 3)))
        - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    )

def safe_window(summary: dict[str, Any], key: str, start_cycle: int, stop_cycle: int) -> np.ndarray:
    values = summary_values(summary, key)
    if values.size == 0:
        return np.asarray([], dtype=float)
    start = min(max(start_cycle - 1, 0), values.size)
    stop = min(max(stop_cycle, start), values.size)
    return values[start:stop]


def linear_fit_feature(summary: dict[str, Any], key: str, start_cycle: int, stop_cycle: int) -> tuple[float, float]:
    y = safe_window(summary, key, start_cycle, stop_cycle)
    if y.size < 2:
        return float("nan"), float("nan")
    x = np.arange(start_cycle, start_cycle + y.size, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def temperature_integral(cell: dict[str, Any], start_cycle: int, stop_cycle: int) -> float:
    summary = cell["summary"]
    tavg = safe_window(summary, "Tavg", start_cycle, stop_cycle)
    chargetime = safe_window(summary, "chargetime", start_cycle, stop_cycle)
    n = min(tavg.size, chargetime.size)
    if n == 0:
        return float("nan")
    # Use summary-level cycle averages to avoid raw time-trace glitches from
    # thermocouple/contact noise dominating the full model feature.
    return float(np.nansum(tavg[:n] * chargetime[:n]))


def extract_supplementary_features(
    cell_id: str,
    cell: dict[str, Any],
    row_index: int,
    counts: dict[str, int],
    cycle_a: int,
    cycle_b: int,
) -> dict[str, Any]:
    cycle_low = get_cycle(cell, cycle_a)
    cycle_high = get_cycle(cell, cycle_b)
    qd_low = np.asarray(cycle_low["Qdlin"], dtype=float).reshape(-1)
    qd_high = np.asarray(cycle_high["Qdlin"], dtype=float).reshape(-1)
    n = min(qd_low.size, qd_high.size)
    delta_q = qd_high[:n] - qd_low[:n]
    delta_q_clean = delta_q[np.isfinite(delta_q)]

    summary = cell["summary"]
    slope_2_100, intercept_2_100 = linear_fit_feature(summary, "QD", 2, cycle_b)
    slope_91_100, intercept_91_100 = linear_fit_feature(summary, "QD", 91, cycle_b)
    qd_cycle_2 = safe_summary(summary, "QD", 1)
    qd_cycle_100 = safe_summary(summary, "QD", cycle_b - 1)
    qd_2_100 = safe_window(summary, "QD", 2, cycle_b)
    max_qd_minus_cycle_2 = float(np.nanmax(qd_2_100) - qd_cycle_2) if qd_2_100.size else float("nan")

    ir_2_100 = safe_window(summary, "IR", 2, cycle_b)
    ir_cycle_2 = safe_summary(summary, "IR", 1)
    ir_cycle_100 = safe_summary(summary, "IR", cycle_b - 1)

    tmax_2_100 = safe_window(summary, "Tmax", 2, cycle_b)
    tmin_2_100 = safe_window(summary, "Tmin", 2, cycle_b)

    delta_var = float(np.var(delta_q_clean)) if delta_q_clean.size else float("nan")
    row: dict[str, Any] = {
        "cell_id": cell_id,
        "batch": cell_id[:2],
        "official_order": row_index,
        "official_split": official_split(row_index, counts),
        "lifetime": scalar(cell["cycle_life"]),
        "cycle_life": scalar(cell["cycle_life"]),
        "cycle_a": cycle_a,
        "cycle_b": cycle_b,
        "delta_q_points": int(n),
        "dq_min": float(np.min(delta_q_clean)) if delta_q_clean.size else float("nan"),
        "dq_mean": float(np.mean(delta_q_clean)) if delta_q_clean.size else float("nan"),
        "dq_var": delta_var,
        "dq_log_var": float(np.log10(delta_var)) if delta_var > 0 else float("nan"),
        "dq_skewness": sample_skewness(delta_q_clean),
        "dq_kurtosis": sample_kurtosis(delta_q_clean),
        "dq_value_at_2v": float(delta_q_clean[-1]) if delta_q_clean.size else float("nan"),
        "qd_slope_2_100": slope_2_100,
        "qd_intercept_2_100": intercept_2_100,
        "qd_slope_91_100": slope_91_100,
        "qd_intercept_91_100": intercept_91_100,
        "qd_cycle_2": qd_cycle_2,
        "qd_max_minus_cycle_2": max_qd_minus_cycle_2,
        "qd_cycle_100": qd_cycle_100,
        "charge_time_avg_first_5": safe_summary_mean(summary, "chargetime", 1, 6),
        "temp_max_2_100": float(np.nanmax(tmax_2_100)) if tmax_2_100.size else float("nan"),
        "temp_min_2_100": float(np.nanmin(tmin_2_100)) if tmin_2_100.size else float("nan"),
        "temp_integral_2_100": temperature_integral(cell, 2, cycle_b),
        "ir_cycle_2": ir_cycle_2,
        "ir_min_2_100": float(np.nanmin(ir_2_100)) if ir_2_100.size else float("nan"),
        "ir_diff_100_2": ir_cycle_100 - ir_cycle_2,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", default="data/severson_raw/pkl_20170630")
    parser.add_argument(
        "--output",
        default="data/severson_features_20170630/severson_supplementary_table1_features_20170630.csv",
    )
    parser.add_argument("--cycle-a", type=int, default=10)
    parser.add_argument("--cycle-b", type=int, default=100)
    args = parser.parse_args()

    bat_dict, counts = load_combined_batches(Path(args.pkl_dir))
    rows = [
        extract_supplementary_features(cell_id, cell, idx, counts, args.cycle_a, args.cycle_b)
        for idx, (cell_id, cell) in enumerate(bat_dict.items())
    ]
    features = pd.DataFrame(rows)
    missing_target = features["lifetime"].isna()
    if missing_target.any():
        dropped = features.loc[missing_target, "cell_id"].tolist()
        print(f"Dropping {len(dropped)} rows with missing lifetime: {', '.join(dropped)}")
        features = features.loc[~missing_target].reset_index(drop=True)

    non_feature_cols = {"cell_id", "batch", "official_split"}
    feature_cols = [c for c in features.columns if c not in non_feature_cols]
    if features[feature_cols].isna().any().any():
        missing = features[feature_cols].isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        print("Warning: extracted features contain NaNs:")
        print(missing.to_string())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(f"Saved {len(features)} rows and {features.shape[1]} columns to {output}")
    print(f"Official counts: {counts}")
    print(features["official_split"].value_counts().to_string())


if __name__ == "__main__":
    main()



