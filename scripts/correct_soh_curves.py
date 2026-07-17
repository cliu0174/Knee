from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - optional dependency
    savgol_filter = None

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover - optional dependency
    IsotonicRegression = None


def odd_window(n: int, requested: int, polyorder: int) -> int:
    max_window = min(requested, n if n % 2 == 1 else n - 1)
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if max_window < min_window:
        return 0
    return max_window


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def smooth_soh(values: np.ndarray, window: int, polyorder: int) -> tuple[np.ndarray, str, int]:
    clean = np.asarray(values, dtype=float)
    actual_window = odd_window(clean.size, window, polyorder)
    if actual_window == 0:
        return clean.copy(), "none_short_series", 0
    if savgol_filter is not None:
        return savgol_filter(clean, actual_window, polyorder, mode="interp"), "savitzky_golay", actual_window
    return moving_average(clean, actual_window), "moving_average_fallback", actual_window


def pava_decreasing(values: np.ndarray) -> np.ndarray:
    y = -np.asarray(values, dtype=float)
    levels: list[float] = []
    weights: list[int] = []
    for value in y:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_weight = weights[-2] + weights[-1]
            merged = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total_weight
            levels[-2] = merged
            weights[-2] = total_weight
            levels.pop()
            weights.pop()
    fitted = np.concatenate([np.full(weight, level) for level, weight in zip(levels, weights)])
    return -fitted


def monotonic_decreasing(cycles: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, str]:
    if IsotonicRegression is not None:
        model = IsotonicRegression(increasing=False, out_of_bounds="clip")
        return model.fit_transform(cycles.astype(float), values.astype(float)), "isotonic_regression"
    return pava_decreasing(values), "pava_fallback"


def violation_metrics(values: np.ndarray, threshold: float) -> tuple[int, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return 0, 0.0, 0.0
    increases = np.diff(clean)
    increases = increases[increases > threshold]
    if increases.size == 0:
        return 0, 0.0, 0.0
    return int(increases.size), float(np.sum(increases)), float(np.max(increases))


def trend_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def correct_group(
    group: pd.DataFrame,
    window: int,
    polyorder: int,
    regen_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    corrected = group.copy()
    corrected["soh_smooth"] = np.nan
    corrected["soh_mono"] = np.nan

    valid = corrected["soh_raw"].notna() & corrected["is_positive_capacity"].astype(bool)
    valid_part = corrected.loc[valid].copy()
    raw = valid_part["soh_raw"].to_numpy(dtype=float)
    cycles = valid_part["cycle"].to_numpy(dtype=float)

    smooth, smooth_method, actual_window = smooth_soh(raw, window, polyorder)
    mono, mono_method = monotonic_decreasing(cycles, smooth)

    corrected.loc[valid, "soh_smooth"] = smooth
    corrected.loc[valid, "soh_mono"] = mono

    raw_count, raw_amp, raw_max = violation_metrics(raw, regen_threshold)
    smooth_count, smooth_amp, smooth_max = violation_metrics(smooth, regen_threshold)
    mono_count, mono_amp, mono_max = violation_metrics(mono, regen_threshold)
    qc = {
        "cell_id": group["cell_id"].iloc[0],
        "batch": group["batch"].iloc[0],
        "official_order": int(group["official_order"].iloc[0]),
        "official_split": group["official_split"].iloc[0],
        "cycle_life": float(group["cycle_life"].iloc[0]),
        "n_positive_capacity": int(valid.sum()),
        "smooth_method": smooth_method,
        "smooth_window": int(actual_window),
        "mono_method": mono_method,
        "raw_violation_count": raw_count,
        "smooth_violation_count": smooth_count,
        "mono_violation_count": mono_count,
        "raw_regeneration_amplitude": raw_amp,
        "smooth_regeneration_amplitude": smooth_amp,
        "mono_regeneration_amplitude": mono_amp,
        "raw_max_single_cycle_increase": raw_max,
        "smooth_max_single_cycle_increase": smooth_max,
        "mono_max_single_cycle_increase": mono_max,
        "mean_abs_smooth_minus_raw": float(np.nanmean(np.abs(smooth - raw))),
        "mean_abs_mono_minus_raw": float(np.nanmean(np.abs(mono - raw))),
        "mean_abs_mono_minus_smooth": float(np.nanmean(np.abs(mono - smooth))),
        "trend_fidelity_smooth_vs_raw": trend_fidelity(smooth, raw),
        "trend_fidelity_mono_vs_smooth": trend_fidelity(mono, smooth),
        "final_raw_soh": float(raw[-1]) if raw.size else float("nan"),
        "final_smooth_soh": float(smooth[-1]) if smooth.size else float("nan"),
        "final_mono_soh": float(mono[-1]) if mono.size else float("nan"),
    }
    return corrected, qc


def plot_selected(corrected: pd.DataFrame, selected: list[str], output_dir: Path) -> None:
    ncols = 4
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax, cell_id in zip(axes.ravel(), selected):
        part = corrected[corrected["cell_id"] == cell_id]
        valid = part["soh_raw"].notna()
        ax.plot(part.loc[valid, "cycle"], part.loc[valid, "soh_raw"], color="#8E8E8E", linewidth=0.8, alpha=0.8, label="raw")
        ax.plot(part.loc[valid, "cycle"], part.loc[valid, "soh_smooth"], color="#4C78A8", linewidth=1.1, label="smooth")
        ax.plot(part.loc[valid, "cycle"], part.loc[valid, "soh_mono"], color="#E45756", linewidth=1.2, label="mono")
        ax.axhline(0.8, color="#333333", linestyle="--", linewidth=0.8, alpha=0.55)
        life = int(part["cycle_life"].iloc[0])
        split = part["official_split"].iloc[0]
        ax.set_title(f"{cell_id} | life={life} | {split}", fontsize=9)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "selected_20_soh_correction_grid.png", dpi=220)
    plt.close(fig)


def plot_individuals(corrected: pd.DataFrame, selected: list[str], output_dir: Path) -> None:
    individual_dir = output_dir / "selected_20_individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for cell_id in selected:
        part = corrected[corrected["cell_id"] == cell_id]
        valid = part["soh_raw"].notna()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(part.loc[valid, "cycle"], part.loc[valid, "soh_raw"], color="#8E8E8E", linewidth=0.8, alpha=0.8, label="raw")
        ax.plot(part.loc[valid, "cycle"], part.loc[valid, "soh_smooth"], color="#4C78A8", linewidth=1.2, label="smooth")
        ax.plot(part.loc[valid, "cycle"], part.loc[valid, "soh_mono"], color="#E45756", linewidth=1.3, label="mono")
        ax.axhline(0.8, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(f"{cell_id}: Raw / Smoothed / Monotonic SOH")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(individual_dir / f"{cell_id}_soh_correction.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", default="data/processed_cycles_20170630.csv")
    parser.add_argument("--selected-cells", default="results/soh_raw_curves/selected_20_cells.csv")
    parser.add_argument("--corrected-output", default="data/processed_cycles_soh_corrected_20170630.csv")
    parser.add_argument("--qc-output", default="data/soh_correction_qc_20170630.csv")
    parser.add_argument("--output-dir", default="results/soh_correction_curves")
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--regen-threshold", type=float, default=1e-4)
    args = parser.parse_args()

    cycles = pd.read_csv(args.cycles)
    corrected_parts: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    for _, group in cycles.groupby("cell_id", sort=False):
        corrected, qc = correct_group(group, args.window, args.polyorder, args.regen_threshold)
        corrected_parts.append(corrected)
        qc_rows.append(qc)

    corrected_df = pd.concat(corrected_parts, ignore_index=True)
    qc_df = pd.DataFrame(qc_rows)

    corrected_output = Path(args.corrected_output)
    qc_output = Path(args.qc_output)
    corrected_output.parent.mkdir(parents=True, exist_ok=True)
    qc_output.parent.mkdir(parents=True, exist_ok=True)
    corrected_df.to_csv(corrected_output, index=False)
    qc_df.to_csv(qc_output, index=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.selected_cells)["cell_id"].tolist()
    plot_selected(corrected_df, selected, output_dir)
    plot_individuals(corrected_df, selected, output_dir)

    print(f"Saved corrected cycles: {len(corrected_df)} rows, {corrected_df.shape[1]} columns -> {corrected_output}")
    print(f"Saved correction QC: {len(qc_df)} rows, {qc_df.shape[1]} columns -> {qc_output}")
    print(f"Saved correction plots to {output_dir}")
    print(qc_df[["raw_violation_count", "smooth_violation_count", "mono_violation_count"]].describe().to_string())


if __name__ == "__main__":
    main()
