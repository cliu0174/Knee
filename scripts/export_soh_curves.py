from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_severson_official_features import load_combined_batches, official_split, scalar


def as_1d(summary: dict[str, Any], key: str) -> np.ndarray:
    if key not in summary:
        return np.asarray([], dtype=float)
    return np.asarray(summary[key], dtype=float).reshape(-1)


def value_at(values: np.ndarray, index: int) -> float:
    if index >= values.size:
        return float("nan")
    return float(values[index])


def soh_base_from_qd(qd: np.ndarray, first_n: int) -> float:
    positive = qd[np.isfinite(qd) & (qd > 0)]
    if positive.size == 0:
        return float("nan")
    return float(np.nanmean(positive[:first_n]))


def monotonic_metrics(soh: np.ndarray, threshold: float) -> tuple[int, float, float]:
    clean = soh[np.isfinite(soh)]
    if clean.size < 2:
        return 0, 0.0, 0.0
    diffs = np.diff(clean)
    increases = diffs[diffs > threshold]
    return int(increases.size), float(np.sum(increases)), float(np.max(increases)) if increases.size else 0.0


def build_rows(
    cell_id: str,
    cell: dict[str, Any],
    row_index: int,
    counts: dict[str, int],
    base_first_n: int,
    regen_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = cell["summary"]
    cycles = as_1d(summary, "cycle")
    qd = as_1d(summary, "QD")
    qc = as_1d(summary, "QC")
    ir = as_1d(summary, "IR")
    tavg = as_1d(summary, "Tavg")
    tmin = as_1d(summary, "Tmin")
    tmax = as_1d(summary, "Tmax")
    chargetime = as_1d(summary, "chargetime")
    n = max(arr.size for arr in [cycles, qd, qc, ir, tavg, tmin, tmax, chargetime])

    base_qd = soh_base_from_qd(qd, base_first_n)
    lifetime = scalar(cell["cycle_life"])
    split = official_split(row_index, counts)

    rows: list[dict[str, Any]] = []
    soh_values: list[float] = []
    for idx in range(n):
        cycle = value_at(cycles, idx)
        qd_value = value_at(qd, idx)
        soh = (
            qd_value / base_qd
            if np.isfinite(qd_value) and qd_value > 0 and np.isfinite(base_qd) and base_qd > 0
            else float("nan")
        )
        soh_values.append(soh)
        rows.append(
            {
                "cell_id": cell_id,
                "batch": cell_id[:2],
                "official_order": row_index,
                "official_split": split,
                "cycle": int(cycle) if np.isfinite(cycle) else idx + 1,
                "cycle_array_index": idx,
                "cycle_life": lifetime,
                "qd": qd_value,
                "qc": value_at(qc, idx),
                "soh_raw": soh,
                "soh_base_qd_first_positive_mean": base_qd,
                "ir": value_at(ir, idx),
                "tavg": value_at(tavg, idx),
                "tmin": value_at(tmin, idx),
                "tmax": value_at(tmax, idx),
                "chargetime": value_at(chargetime, idx),
                "is_positive_capacity": bool(np.isfinite(qd_value) and qd_value > 0),
            }
        )

    soh_arr = np.asarray(soh_values, dtype=float)
    positive_qd = qd[np.isfinite(qd) & (qd > 0)]
    mono_count, regen_amp, regen_max = monotonic_metrics(soh_arr, regen_threshold)
    qc_row = {
        "cell_id": cell_id,
        "batch": cell_id[:2],
        "official_order": row_index,
        "official_split": split,
        "cycle_life": lifetime,
        "n_summary_rows": int(n),
        "n_positive_capacity": int(positive_qd.size),
        "soh_base_qd_first_positive_mean": base_qd,
        "first_positive_qd": float(positive_qd[0]) if positive_qd.size else float("nan"),
        "final_positive_qd": float(positive_qd[-1]) if positive_qd.size else float("nan"),
        "min_positive_qd": float(np.nanmin(positive_qd)) if positive_qd.size else float("nan"),
        "max_positive_qd": float(np.nanmax(positive_qd)) if positive_qd.size else float("nan"),
        "final_soh": float(positive_qd[-1] / base_qd) if positive_qd.size and base_qd > 0 else float("nan"),
        "nonpositive_qd_count": int(np.sum(np.isfinite(qd) & (qd <= 0))),
        "missing_qd_count": int(n - qd.size + np.sum(~np.isfinite(qd))),
        "monotonic_violation_count": mono_count,
        "regeneration_amplitude": regen_amp,
        "max_single_cycle_soh_increase": regen_max,
        "has_capacity_regeneration": bool(regen_amp > 0),
    }
    return rows, qc_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", default="data/severson_raw/pkl_20170630")
    parser.add_argument("--cycles-output", default="data/processed_cycles_20170630.csv")
    parser.add_argument("--qc-output", default="data/cell_qc_20170630.csv")
    parser.add_argument("--base-first-n", type=int, default=5)
    parser.add_argument("--regen-threshold", type=float, default=1e-4)
    args = parser.parse_args()

    bat_dict, counts = load_combined_batches(Path(args.pkl_dir))
    all_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for row_index, (cell_id, cell) in enumerate(bat_dict.items()):
        rows, qc_row = build_rows(cell_id, cell, row_index, counts, args.base_first_n, args.regen_threshold)
        all_rows.extend(rows)
        qc_rows.append(qc_row)

    cycles = pd.DataFrame(all_rows)
    qc = pd.DataFrame(qc_rows)

    cycles_output = Path(args.cycles_output)
    qc_output = Path(args.qc_output)
    cycles_output.parent.mkdir(parents=True, exist_ok=True)
    qc_output.parent.mkdir(parents=True, exist_ok=True)
    cycles.to_csv(cycles_output, index=False)
    qc.to_csv(qc_output, index=False)

    print(f"Saved processed cycles: {len(cycles)} rows, {cycles.shape[1]} columns -> {cycles_output}")
    print(f"Saved cell QC: {len(qc)} rows, {qc.shape[1]} columns -> {qc_output}")
    print(f"Official counts: {counts}")
    print(qc["official_split"].value_counts().to_string())


if __name__ == "__main__":
    main()

