from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return float(np.sum(residual**2)), float(slope), float(intercept)


def detect(
    cycles: np.ndarray,
    soh: np.ndarray,
    *,
    overlap_boundary: bool,
    min_frac: float,
    max_frac: float,
    min_segment_points: int,
) -> dict[str, float]:
    order = np.argsort(cycles)
    x = cycles[order].astype(float)
    y = soh[order].astype(float)
    n = x.size
    low = max(min_segment_points, int(np.floor(n * min_frac)))
    high = min(n - min_segment_points, int(np.ceil(n * max_frac)))

    best: dict[str, float] | None = None
    for split in range(low, high):
        x1, y1 = x[: split + 1], y[: split + 1]
        second_start = split if overlap_boundary else split + 1
        x2, y2 = x[second_start:], y[second_start:]
        sse1, slope1, intercept1 = fit_line(x1, y1)
        sse2, slope2, intercept2 = fit_line(x2, y2)
        total = sse1 + sse2
        if best is None or total < best["piecewise_sse"]:
            best = {
                "knee_cycle": float(x[split]),
                "knee_index": float(split),
                "piecewise_sse": total,
                "first_slope": slope1,
                "first_intercept": intercept1,
                "second_slope": slope2,
                "second_intercept": intercept2,
            }

    if best is None:
        raise RuntimeError("No admissible split")

    single_sse, _, _ = fit_line(x, y)
    residual_count = n + 1 if overlap_boundary else n
    best["single_sse"] = single_sse
    best["confidence_sse"] = (
        (single_sse - best["piecewise_sse"]) / single_sse if single_sse > 0 else np.nan
    )
    best["confidence_mse"] = (
        1.0 - (best["piecewise_sse"] / residual_count) / (single_sse / n)
        if single_sse > 0
        else np.nan
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare overlapping- and non-overlapping-boundary knee labels."
    )
    parser.add_argument("--cycles", default="data/processed_cycles_soh_corrected_20170630.csv")
    parser.add_argument("--existing-labels", default="data/knee_labels_20170630.csv")
    parser.add_argument("--output-dir", default="results/knee_split_convention_audit")
    parser.add_argument("--min-frac", type=float, default=0.2)
    parser.add_argument("--max-frac", type=float, default=0.9)
    parser.add_argument("--min-segment-points", type=int, default=30)
    args = parser.parse_args()

    cycles = pd.read_csv(args.cycles)
    existing = pd.read_csv(args.existing_labels)[["cell_id", "knee_cycle"]].rename(
        columns={"knee_cycle": "existing_knee_cycle"}
    )

    rows: list[dict[str, float | str]] = []
    for cell_id, group in cycles.groupby("cell_id", sort=False):
        valid = group["soh_mono"].notna()
        x = group.loc[valid, "cycle"].to_numpy(dtype=float)
        y = group.loc[valid, "soh_mono"].to_numpy(dtype=float)
        overlapping = detect(
            x,
            y,
            overlap_boundary=True,
            min_frac=args.min_frac,
            max_frac=args.max_frac,
            min_segment_points=args.min_segment_points,
        )
        nonoverlapping = detect(
            x,
            y,
            overlap_boundary=False,
            min_frac=args.min_frac,
            max_frac=args.max_frac,
            min_segment_points=args.min_segment_points,
        )
        rows.append(
            {
                "cell_id": cell_id,
                "n_life": int(valid.sum()),
                "overlap_knee_cycle": overlapping["knee_cycle"],
                "nonoverlap_knee_cycle": nonoverlapping["knee_cycle"],
                "knee_delta_nonoverlap_minus_overlap": (
                    nonoverlapping["knee_cycle"] - overlapping["knee_cycle"]
                ),
                "overlap_piecewise_sse": overlapping["piecewise_sse"],
                "nonoverlap_piecewise_sse": nonoverlapping["piecewise_sse"],
                "single_line_sse": overlapping["single_sse"],
                "overlap_confidence_sse": overlapping["confidence_sse"],
                "overlap_confidence_mse": overlapping["confidence_mse"],
                "nonoverlap_confidence_sse": nonoverlapping["confidence_sse"],
                "nonoverlap_confidence_mse": nonoverlapping["confidence_mse"],
            }
        )

    audit = pd.DataFrame(rows).merge(existing, on="cell_id", how="left", validate="one_to_one")
    audit["existing_minus_overlap"] = audit["existing_knee_cycle"] - audit["overlap_knee_cycle"]
    audit["existing_minus_nonoverlap"] = (
        audit["existing_knee_cycle"] - audit["nonoverlap_knee_cycle"]
    )
    delta = audit["knee_delta_nonoverlap_minus_overlap"].abs()
    summary = {
        "n_cells": int(len(audit)),
        "existing_overlap_exact_matches": int((audit["existing_minus_overlap"] == 0).sum()),
        "existing_nonoverlap_exact_matches": int(
            (audit["existing_minus_nonoverlap"] == 0).sum()
        ),
        "changed_knees_nonoverlap": int((delta > 0).sum()),
        "unchanged_knees_nonoverlap": int((delta == 0).sum()),
        "mean_absolute_knee_change": float(delta.mean()),
        "median_absolute_knee_change": float(delta.median()),
        "maximum_absolute_knee_change": float(delta.max()),
        "within_1_cycle": int((delta <= 1).sum()),
        "within_5_cycles": int((delta <= 5).sum()),
        "within_10_cycles": int((delta <= 10).sum()),
        "eligibility_y_gt_150_changed": int(
            (
                (audit["overlap_knee_cycle"] > 150)
                != (audit["nonoverlap_knee_cycle"] > 150)
            ).sum()
        ),
        "overlap_confidence_sse_mean": float(audit["overlap_confidence_sse"].mean()),
        "overlap_confidence_mse_mean": float(audit["overlap_confidence_mse"].mean()),
        "nonoverlap_confidence_sse_mean": float(
            audit["nonoverlap_confidence_sse"].mean()
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "cell_level_comparison.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved audit to {output_dir}")


if __name__ == "__main__":
    main()
