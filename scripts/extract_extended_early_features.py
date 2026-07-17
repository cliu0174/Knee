from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOG_VARIANCE_EPSILON = 1e-12

# Pickles produced with NumPy 2.x reference ``numpy._core`` while the
# project runtime currently uses NumPy 1.x, where the modules live under
# ``numpy.core``.  Register read-only aliases before unpickling the archived
# Severson dictionaries.  This mirrors the compatibility guard used by the
# Local DeltaQ extractor.
if "numpy._core.numeric" not in sys.modules:
    import numpy.core as _np_core
    import numpy.core.multiarray as _np_multiarray
    import numpy.core.numeric as _np_numeric

    sys.modules["numpy._core"] = _np_core
    sys.modules["numpy._core.numeric"] = _np_numeric
    sys.modules["numpy._core.multiarray"] = _np_multiarray

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_severson_official_features import get_cycle, load_combined_batches


META_COLUMNS = [
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
]


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def slope_intercept(cycles: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(cycles) & np.isfinite(values)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(cycles[mask], values[mask], 1)
    return float(slope), float(intercept)


def basic_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    clean = finite(values)
    if clean.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_first": float("nan"),
            f"{prefix}_last": float("nan"),
            f"{prefix}_change": float("nan"),
            f"{prefix}_range": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.nanmean(clean)),
        f"{prefix}_std": float(np.nanstd(clean, ddof=1)) if clean.size > 1 else 0.0,
        f"{prefix}_min": float(np.nanmin(clean)),
        f"{prefix}_max": float(np.nanmax(clean)),
        f"{prefix}_first": float(clean[0]),
        f"{prefix}_last": float(clean[-1]),
        f"{prefix}_change": float(clean[-1] - clean[0]),
        f"{prefix}_range": float(np.nanmax(clean) - np.nanmin(clean)),
    }


def series_features(prefix: str, part: pd.DataFrame, column: str) -> dict[str, float]:
    values = part[column].to_numpy(dtype=float)
    cycles = part["cycle"].to_numpy(dtype=float)
    features = basic_stats(prefix, values)
    slope, intercept = slope_intercept(cycles, values)
    features[f"{prefix}_slope"] = slope
    features[f"{prefix}_intercept"] = intercept
    return features


def sample_skewness(values: np.ndarray) -> float:
    clean = finite(values)
    n = clean.size
    if n < 3:
        return float("nan")
    centered = clean - np.mean(clean)
    std = np.std(clean, ddof=1)
    if std == 0:
        return float("nan")
    return float((n / ((n - 1) * (n - 2))) * np.sum((centered / std) ** 3))


def sample_kurtosis(values: np.ndarray) -> float:
    clean = finite(values)
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


def delta_q_features(
    cell_id: str,
    cell: dict[str, Any],
    prefix: str,
    cycle_a: int,
    cycle_b: int,
) -> dict[str, float | int]:
    try:
        low = get_cycle(cell, cycle_a)
        high = get_cycle(cell, cycle_b)
    except KeyError:
        return {
            f"{prefix}_delta_q_points": 0,
            f"{prefix}_delta_q_mean": float("nan"),
            f"{prefix}_delta_q_std": float("nan"),
            f"{prefix}_delta_q_var": float("nan"),
            f"{prefix}_delta_q_log_var": float("nan"),
            f"{prefix}_delta_q_min": float("nan"),
            f"{prefix}_delta_q_max": float("nan"),
            f"{prefix}_delta_q_skewness": float("nan"),
            f"{prefix}_delta_q_kurtosis": float("nan"),
        }
    qd_low = np.asarray(low["Qdlin"], dtype=float).reshape(-1)
    qd_high = np.asarray(high["Qdlin"], dtype=float).reshape(-1)
    n = min(qd_low.size, qd_high.size)
    delta = qd_high[:n] - qd_low[:n]
    clean = finite(delta)
    variance = float(np.var(clean)) if clean.size else float("nan")
    return {
        f"{prefix}_delta_q_points": int(clean.size),
        f"{prefix}_delta_q_mean": float(np.mean(clean)) if clean.size else float("nan"),
        f"{prefix}_delta_q_std": float(np.std(clean, ddof=1)) if clean.size > 1 else float("nan"),
        f"{prefix}_delta_q_var": variance,
        f"{prefix}_delta_q_log_var": (
            float(np.log10(variance + LOG_VARIANCE_EPSILON))
            if np.isfinite(variance)
            else float("nan")
        ),
        f"{prefix}_delta_q_min": float(np.min(clean)) if clean.size else float("nan"),
        f"{prefix}_delta_q_max": float(np.max(clean)) if clean.size else float("nan"),
        f"{prefix}_delta_q_skewness": sample_skewness(clean),
        f"{prefix}_delta_q_kurtosis": sample_kurtosis(clean),
    }


def raw_feature_columns(base: pd.DataFrame, prefix: str) -> list[str]:
    cols: list[str] = []
    for col in base.columns:
        if col in META_COLUMNS:
            continue
        if (
            col == f"{prefix}_n_observed"
            or col == f"{prefix}_window_complete"
            or col.startswith(f"{prefix}_qd_")
            or col.startswith(f"{prefix}_soh_raw_")
            or col.startswith(f"{prefix}_raw_regen_")
            or col.startswith(f"{prefix}_raw_max_single_cycle")
        ):
            cols.append(col)
    return cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", required=True)
    parser.add_argument("--cycles", default="data/processed_cycles_soh_corrected_20170630.csv")
    parser.add_argument("--pkl-dir", default="data/severson_raw/pkl_20170630")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qc-output", required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--delta-cycle-a", type=int, default=9)
    parser.add_argument(
        "--delta-cycle-b",
        type=int,
        default=None,
        help="Default is window - 1 to match the c9_c99 indexing convention used in the Severson reproduction.",
    )
    args = parser.parse_args()

    prefix = f"N{args.window}"
    delta_cycle_b = args.delta_cycle_b if args.delta_cycle_b is not None else args.window - 1

    base = pd.read_csv(args.base_features)
    cycles = pd.read_csv(args.cycles)
    bat_dict, _ = load_combined_batches(Path(args.pkl_dir))

    keep_cols = META_COLUMNS + raw_feature_columns(base, prefix)
    rows: list[dict[str, object]] = []
    for _, base_row in base.iterrows():
        cell_id = base_row["cell_id"]
        row = {col: base_row[col] for col in keep_cols if col in base.columns}
        part = cycles[
            (cycles["cell_id"] == cell_id)
            & (cycles["cycle"] <= args.window)
            & cycles["is_positive_capacity"].astype(bool)
        ].copy()
        for col in ["chargetime", "ir", "tavg", "tmin", "tmax"]:
            row.update(series_features(f"{prefix}_{col}", part, col))
        row.update(
            delta_q_features(
                cell_id,
                bat_dict[cell_id],
                prefix,
                args.delta_cycle_a,
                delta_cycle_b,
            )
        )
        row[f"{prefix}_delta_cycle_a"] = args.delta_cycle_a
        row[f"{prefix}_delta_cycle_b"] = delta_cycle_b
        rows.append(row)

    features = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)

    feature_cols = [c for c in features.columns if c not in META_COLUMNS]
    qc = pd.DataFrame(
        [
            {
                "feature": col,
                "missing_count": int(features[col].isna().sum()),
                "missing_fraction": float(features[col].isna().mean()),
                "n_unique": int(features[col].nunique(dropna=True)),
            }
            for col in feature_cols
        ]
    ).sort_values(["missing_count", "feature"], ascending=[False, True])
    qc_output = Path(args.qc_output)
    qc_output.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(qc_output, index=False)

    print(f"Saved extended early features: {len(features)} rows, {features.shape[1]} columns -> {output}")
    print(f"Saved QC: {len(qc)} rows -> {qc_output}")
    print(f"Delta Q cycles: {args.delta_cycle_a} to {delta_cycle_b}")
    print(qc[qc["missing_count"] > 0].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
