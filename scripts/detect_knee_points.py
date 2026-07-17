from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fit_line_sse(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    if x.size < 2:
        return float("inf"), np.full_like(y, np.nan, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    return float(np.sum((y - pred) ** 2)), pred


def detect_piecewise_knee(
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
        return {
            "knee_cycle": np.nan,
            "knee_index": np.nan,
            "knee_soh": np.nan,
            "piecewise_sse": np.nan,
            "single_line_sse": np.nan,
            "label_confidence": np.nan,
            "first_slope": np.nan,
            "second_slope": np.nan,
            "status": "too_short",
        }

    low = max(min_segment_points, int(np.floor(n * min_frac)))
    high = min(n - min_segment_points, int(np.ceil(n * max_frac)))
    candidates = range(low, high)
    best: dict[str, object] | None = None
    for split in candidates:
        x1, y1 = x[: split + 1], y[: split + 1]
        x2, y2 = x[split + 1 :], y[split + 1 :]
        sse1, pred1 = fit_line_sse(x1, y1)
        sse2, pred2 = fit_line_sse(x2, y2)
        total = sse1 + sse2
        if best is None or total < float(best["piecewise_sse"]):
            slope1, intercept1 = np.polyfit(x1, y1, 1)
            slope2, intercept2 = np.polyfit(x2, y2, 1)
            best = {
                "knee_cycle": float(x[split]),
                "knee_index": int(split),
                "knee_soh": float(y[split]),
                "piecewise_sse": float(total),
                "first_slope": float(slope1),
                "first_intercept": float(intercept1),
                "second_slope": float(slope2),
                "second_intercept": float(intercept2),
                "first_pred": pred1,
                "second_pred": pred2,
                "x_sorted": x,
                "y_sorted": y,
            }

    single_sse, _ = fit_line_sse(x, y)
    confidence = (single_sse - float(best["piecewise_sse"])) / single_sse if single_sse > 0 else np.nan
    best.update(
        {
            "single_line_sse": float(single_sse),
            "label_confidence": float(confidence),
            "status": "ok",
        }
    )
    return best


def build_label_row(cell_id: str, group: pd.DataFrame, result: dict[str, object]) -> dict[str, object]:
    cycle_life = float(group["cycle_life"].iloc[0])
    knee_cycle = float(result["knee_cycle"]) if np.isfinite(result["knee_cycle"]) else np.nan
    return {
        "cell_id": cell_id,
        "batch": group["batch"].iloc[0],
        "official_order": int(group["official_order"].iloc[0]),
        "official_split": group["official_split"].iloc[0],
        "cycle_life": cycle_life,
        "n_points": int(group["soh_mono"].notna().sum()),
        "knee_cycle": knee_cycle,
        "knee_soh": result["knee_soh"],
        "relative_knee": float(knee_cycle / cycle_life) if np.isfinite(knee_cycle) and cycle_life > 0 else np.nan,
        "remaining_cycles_from_100": float(knee_cycle - 100) if np.isfinite(knee_cycle) else np.nan,
        "piecewise_sse": result["piecewise_sse"],
        "single_line_sse": result["single_line_sse"],
        "label_confidence": result["label_confidence"],
        "first_slope": result["first_slope"],
        "second_slope": result["second_slope"],
        "slope_change": (
            float(result["second_slope"] - result["first_slope"])
            if np.isfinite(result["second_slope"]) and np.isfinite(result["first_slope"])
            else np.nan
        ),
        "status": result["status"],
    }


def plot_cell(group: pd.DataFrame, result: dict[str, object], output_path: Path) -> None:
    valid = group["soh_mono"].notna()
    part = group.loc[valid]
    x = part["cycle"].to_numpy(dtype=float)
    raw = part["soh_raw"].to_numpy(dtype=float)
    smooth = part["soh_smooth"].to_numpy(dtype=float)
    mono = part["soh_mono"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(x, raw, color="#BBBBBB", linewidth=0.7, alpha=0.8, label="raw")
    ax.plot(x, smooth, color="#4C78A8", linewidth=1.0, alpha=0.85, label="smooth")
    ax.plot(x, mono, color="#E45756", linewidth=1.25, label="mono")
    if result["status"] == "ok":
        x_sorted = result["x_sorted"]
        split = int(result["knee_index"])
        pred1 = result["first_pred"]
        pred2 = result["second_pred"]
        ax.plot(x_sorted[: split + 1], pred1, color="#2A2A2A", linewidth=1.15, linestyle="-")
        ax.plot(x_sorted[split + 1 :], pred2, color="#2A2A2A", linewidth=1.15, linestyle="-")
        ax.axvline(float(result["knee_cycle"]), color="#111111", linestyle="--", linewidth=1.0)
        ax.scatter([float(result["knee_cycle"])], [float(result["knee_soh"])], color="#111111", s=34, zorder=5)
        title_extra = f"knee={int(result['knee_cycle'])}, conf={float(result['label_confidence']):.3f}"
    else:
        title_extra = f"status={result['status']}"
    cell_id = group["cell_id"].iloc[0]
    life = int(group["cycle_life"].iloc[0])
    ax.axhline(0.8, color="#333333", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.set_title(f"{cell_id} | life={life} | {title_extra}")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_selected_grid(cycles: pd.DataFrame, labels: pd.DataFrame, selected: list[str], output_dir: Path) -> None:
    ncols = 4
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.0 * nrows), squeeze=False)
    for ax, cell_id in zip(axes.ravel(), selected):
        part = cycles[(cycles["cell_id"] == cell_id) & cycles["soh_mono"].notna()]
        label = labels[labels["cell_id"] == cell_id].iloc[0]
        ax.plot(part["cycle"], part["soh_raw"], color="#BBBBBB", linewidth=0.6, alpha=0.75)
        ax.plot(part["cycle"], part["soh_mono"], color="#E45756", linewidth=1.1)
        if label["status"] == "ok":
            ax.axvline(label["knee_cycle"], color="#111111", linestyle="--", linewidth=0.9)
            ax.scatter([label["knee_cycle"]], [label["knee_soh"]], color="#111111", s=22)
        ax.axhline(0.8, color="#333333", linestyle=":", linewidth=0.75, alpha=0.6)
        ax.set_title(
            f"{cell_id} | life={int(label['cycle_life'])} | knee={int(label['knee_cycle'])}",
            fontsize=9,
        )
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_20_knee_detection_grid.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", default="data/processed_cycles_soh_corrected_20170630.csv")
    parser.add_argument("--selected-cells", default="results/soh_raw_curves/selected_20_cells.csv")
    parser.add_argument("--labels-output", default="data/knee_labels_20170630.csv")
    parser.add_argument("--output-dir", default="results/knee_detection")
    parser.add_argument("--min-frac", type=float, default=0.2)
    parser.add_argument("--max-frac", type=float, default=0.9)
    parser.add_argument("--min-segment-points", type=int, default=30)
    args = parser.parse_args()

    cycles = pd.read_csv(args.cycles)
    output_dir = Path(args.output_dir)
    individual_dir = output_dir / "selected_20_individual"
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir.mkdir(parents=True, exist_ok=True)

    labels: list[dict[str, object]] = []
    results_by_cell: dict[str, dict[str, object]] = {}
    for cell_id, group in cycles.groupby("cell_id", sort=False):
        valid = group["soh_mono"].notna()
        x = group.loc[valid, "cycle"].to_numpy(dtype=float)
        y = group.loc[valid, "soh_mono"].to_numpy(dtype=float)
        result = detect_piecewise_knee(x, y, args.min_frac, args.max_frac, args.min_segment_points)
        labels.append(build_label_row(cell_id, group, result))
        results_by_cell[cell_id] = result

    labels_df = pd.DataFrame(labels)
    labels_output = Path(args.labels_output)
    labels_output.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(labels_output, index=False)

    selected = pd.read_csv(args.selected_cells)["cell_id"].tolist()
    for cell_id in selected:
        group = cycles[cycles["cell_id"] == cell_id]
        plot_cell(group, results_by_cell[cell_id], individual_dir / f"{cell_id}_knee_detection.png")
    plot_selected_grid(cycles, labels_df, selected, output_dir)

    print(f"Saved knee labels: {len(labels_df)} rows, {labels_df.shape[1]} columns -> {labels_output}")
    print(f"Saved knee plots to {output_dir}")
    print(labels_df[["knee_cycle", "relative_knee", "knee_soh", "label_confidence"]].describe().to_string())
    print(labels_df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
