from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from scipy.stats import kurtosis, skew
except ModuleNotFoundError:
    def skew(values, bias=False):
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean)]
        n = clean.size
        if n < 3:
            return float("nan")
        centered = clean - np.mean(clean)
        std = np.std(clean, ddof=1 if not bias else 0)
        if std == 0:
            return float("nan")
        if bias:
            return float(np.mean((centered / std) ** 3))
        return float((n / ((n - 1) * (n - 2))) * np.sum((centered / std) ** 3))

    def kurtosis(values, bias=False):
        clean = np.asarray(values, dtype=float)
        clean = clean[np.isfinite(clean)]
        n = clean.size
        if n < 4:
            return float("nan")
        centered = clean - np.mean(clean)
        std = np.std(clean, ddof=1 if not bias else 0)
        if std == 0:
            return float("nan")
        if bias:
            return float(np.mean((centered / std) ** 4) - 3.0)
        z4_sum = np.sum((centered / std) ** 4)
        return float(
            (n * (n + 1) * z4_sum / ((n - 1) * (n - 2) * (n - 3)))
            - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        )


BATCH1_REMOVE = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
BATCH2_CARRYOVER = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
BATCH1_CARRYOVER_TARGETS = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
BATCH2_CARRYOVER_LENGTHS = [662, 981, 1060, 208, 482]
BATCH3_REMOVE = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]


def load_pickle(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def scalar(value: Any) -> float:
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def remove_keys(batch: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        batch.pop(key, None)


def merge_batch2_carryover(batch1: dict[str, Any], batch2: dict[str, Any]) -> None:
    for batch1_key, batch2_key, add_len in zip(
        BATCH1_CARRYOVER_TARGETS,
        BATCH2_CARRYOVER,
        BATCH2_CARRYOVER_LENGTHS,
    ):
        if batch1_key not in batch1 or batch2_key not in batch2:
            continue
        batch1[batch1_key]["cycle_life"] = scalar(batch1[batch1_key]["cycle_life"]) + add_len
        for summary_key, summary_values in batch1[batch1_key]["summary"].items():
            extra = np.asarray(batch2[batch2_key]["summary"][summary_key]).reshape(-1)
            base = np.asarray(summary_values).reshape(-1)
            if summary_key == "cycle":
                extra = extra + len(base)
            batch1[batch1_key]["summary"][summary_key] = np.hstack((base, extra))

        last_cycle = len(batch1[batch1_key]["cycles"])
        for offset, cycle_key in enumerate(batch2[batch2_key]["cycles"].keys()):
            batch1[batch1_key]["cycles"][str(last_cycle + offset)] = batch2[batch2_key]["cycles"][cycle_key]

    remove_keys(batch2, BATCH2_CARRYOVER)


def load_combined_batches(pkl_dir: Path) -> tuple[dict[str, Any], dict[str, int]]:
    batch1 = load_pickle(pkl_dir / "batch1.pkl")
    remove_keys(batch1, BATCH1_REMOVE)

    batch2 = load_pickle(pkl_dir / "batch2.pkl")
    merge_batch2_carryover(batch1, batch2)

    batch3 = load_pickle(pkl_dir / "batch3.pkl")
    remove_keys(batch3, BATCH3_REMOVE)

    counts = {
        "numBat1": len(batch1),
        "numBat2": len(batch2),
        "numBat3": len(batch3),
    }
    return {**batch1, **batch2, **batch3}, counts


def official_split(index: int, counts: dict[str, int]) -> str:
    num_bat1 = counts["numBat1"]
    num_bat2 = counts["numBat2"]
    num_bat3 = counts["numBat3"]
    num_bat = num_bat1 + num_bat2 + num_bat3
    primary_test = set(np.hstack((np.arange(0, (num_bat1 + num_bat2), 2), 83)).astype(int))
    train = set(np.arange(1, (num_bat1 + num_bat2 - 1), 2).astype(int))
    secondary_test = set(np.arange(num_bat - num_bat3, num_bat).astype(int))
    if index in train:
        return "train"
    if index in primary_test:
        return "primary_test"
    if index in secondary_test:
        return "secondary_test"
    return "unused"


def get_cycle(cell: dict[str, Any], cycle_number: int) -> dict[str, Any]:
    cycles = cell["cycles"]
    candidates = [str(cycle_number), str(cycle_number - 1)]
    for key in candidates:
        if key in cycles:
            return cycles[key]
    raise KeyError(f"Could not find cycle {cycle_number} or {cycle_number - 1}")


def safe_summary(summary: dict[str, Any], key: str, index: int) -> float:
    values = np.asarray(summary[key], dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    idx = min(max(index, 0), values.size - 1)
    return float(values[idx])


def safe_summary_mean(summary: dict[str, Any], key: str, start: int, stop: int) -> float:
    values = np.asarray(summary[key], dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    start = min(max(start, 0), values.size)
    stop = min(max(stop, start + 1), values.size)
    return float(np.nanmean(values[start:stop]))


def finite_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_var": float("nan"),
            f"{prefix}_log_var": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_skew": float("nan"),
            f"{prefix}_kurtosis": float("nan"),
        }
    variance = float(np.var(clean))
    return {
        f"{prefix}_mean": float(np.mean(clean)),
        f"{prefix}_var": variance,
        f"{prefix}_log_var": float(np.log10(variance)) if variance > 0 else float("nan"),
        f"{prefix}_min": float(np.min(clean)),
        f"{prefix}_max": float(np.max(clean)),
        f"{prefix}_skew": float(skew(clean, bias=False)) if clean.size > 2 else float("nan"),
        f"{prefix}_kurtosis": float(kurtosis(clean, bias=False)) if clean.size > 3 else float("nan"),
    }


def extract_cell_features(
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

    summary = cell["summary"]
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
    }
    row.update(finite_stats("delta_q", delta_q))
    row.update(
        {
            "qd_cycle_2": safe_summary(summary, "QD", 1),
            "qd_cycle_100": safe_summary(summary, "QD", cycle_b - 1),
            "qd_diff_100_2": safe_summary(summary, "QD", cycle_b - 1) - safe_summary(summary, "QD", 1),
            "charge_time_mean_2_6": safe_summary_mean(summary, "chargetime", 1, 6),
            "ir_cycle_2": safe_summary(summary, "IR", 1),
            "ir_cycle_100": safe_summary(summary, "IR", cycle_b - 1),
            "ir_diff_100_2": safe_summary(summary, "IR", cycle_b - 1) - safe_summary(summary, "IR", 1),
            "tavg_mean_1_100": safe_summary_mean(summary, "Tavg", 0, cycle_b),
            "tmax_mean_1_100": safe_summary_mean(summary, "Tmax", 0, cycle_b),
            "tmin_mean_1_100": safe_summary_mean(summary, "Tmin", 0, cycle_b),
            "qd_slope_2_100": (
                safe_summary(summary, "QD", cycle_b - 1) - safe_summary(summary, "QD", 1)
            )
            / max(cycle_b - 2, 1),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", default="data/severson_raw/pkl")
    parser.add_argument("--output", default="data/severson_features/severson_official_features.csv")
    parser.add_argument("--cycle-a", type=int, default=10)
    parser.add_argument("--cycle-b", type=int, default=100)
    parser.add_argument(
        "--keep-missing-target",
        action="store_true",
        help="Keep rows whose cycle_life/lifetime is NaN. Default drops them for supervised training.",
    )
    args = parser.parse_args()

    pkl_dir = Path(args.pkl_dir)
    output = Path(args.output)
    bat_dict, counts = load_combined_batches(pkl_dir)
    rows = [
        extract_cell_features(cell_id, cell, idx, counts, args.cycle_a, args.cycle_b)
        for idx, (cell_id, cell) in enumerate(bat_dict.items())
    ]
    features = pd.DataFrame(rows)
    if not args.keep_missing_target:
        missing_target = features["lifetime"].isna()
        if missing_target.any():
            dropped = features.loc[missing_target, "cell_id"].tolist()
            print(f"Dropping {len(dropped)} rows with missing lifetime: {', '.join(dropped)}")
            features = features.loc[~missing_target].reset_index(drop=True)

    feature_cols = [c for c in features.columns if c not in {"cell_id", "batch", "official_split"}]
    if features[feature_cols].isna().any().any():
        missing = features[feature_cols].isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        print("Warning: extracted features contain NaNs:")
        print(missing.to_string())

    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(f"Saved {len(features)} rows and {features.shape[1]} columns to {output}")
    print(f"Official counts: {counts}")
    print(features["official_split"].value_counts().to_string())


if __name__ == "__main__":
    main()

