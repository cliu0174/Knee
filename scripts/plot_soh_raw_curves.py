from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def choose_representative_cells(qc: pd.DataFrame, n_cells: int) -> list[str]:
    ranked = qc.sort_values("cycle_life").reset_index(drop=True)
    if len(ranked) <= n_cells:
        return ranked["cell_id"].tolist()
    positions = np.linspace(0, len(ranked) - 1, n_cells).round().astype(int)
    return ranked.loc[positions, "cell_id"].drop_duplicates().tolist()


def plot_overview(cycles: pd.DataFrame, qc: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = {"train": "#4C78A8", "primary_test": "#E45756", "secondary_test": "#F58518"}
    for cell_id, part in cycles.groupby("cell_id", sort=False):
        split = qc.loc[qc["cell_id"] == cell_id, "official_split"].iloc[0]
        valid = part["is_positive_capacity"].astype(bool) & part["soh_raw"].notna()
        ax.plot(
            part.loc[valid, "cycle"],
            part.loc[valid, "soh_raw"],
            color=colors.get(split, "#777777"),
            alpha=0.22,
            linewidth=0.8,
        )
    ax.axhline(0.8, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Raw SOH = QD / mean(first 5 positive QD)")
    ax.set_title("All Severson Cells: Raw SOH Curves")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "all_cells_raw_soh_overview.png", dpi=220)
    plt.close(fig)


def plot_selected_grid(cycles: pd.DataFrame, qc: pd.DataFrame, selected: list[str], output_dir: Path) -> None:
    ncols = 4
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.9 * nrows), squeeze=False)
    for ax, cell_id in zip(axes.ravel(), selected):
        part = cycles[cycles["cell_id"] == cell_id]
        meta = qc[qc["cell_id"] == cell_id].iloc[0]
        valid = part["is_positive_capacity"].astype(bool) & part["soh_raw"].notna()
        ax.plot(part.loc[valid, "cycle"], part.loc[valid, "soh_raw"], color="#4C78A8", linewidth=1.2)
        ax.axhline(0.8, color="#333333", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_title(f"{cell_id} | life={int(meta['cycle_life'])} | {meta['official_split']}", fontsize=9)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_20_raw_soh_grid.png", dpi=220)
    plt.close(fig)


def plot_individuals(cycles: pd.DataFrame, qc: pd.DataFrame, selected: list[str], output_dir: Path) -> None:
    individual_dir = output_dir / "selected_20_individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for cell_id in selected:
        part = cycles[cycles["cell_id"] == cell_id]
        meta = qc[qc["cell_id"] == cell_id].iloc[0]
        valid = part["is_positive_capacity"].astype(bool) & part["soh_raw"].notna()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(part.loc[valid, "cycle"], part.loc[valid, "soh_raw"], color="#4C78A8", linewidth=1.3)
        ax.axhline(0.8, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Raw SOH")
        ax.set_title(
            f"{cell_id} | life={int(meta['cycle_life'])} | regen_count={int(meta['monotonic_violation_count'])}"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(individual_dir / f"{cell_id}_raw_soh.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", default="data/processed_cycles_20170630.csv")
    parser.add_argument("--qc", default="data/cell_qc_20170630.csv")
    parser.add_argument("--output-dir", default="results/soh_raw_curves")
    parser.add_argument("--n-cells", type=int, default=20)
    args = parser.parse_args()

    cycles = pd.read_csv(args.cycles)
    qc = pd.read_csv(args.qc)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = choose_representative_cells(qc, args.n_cells)
    pd.DataFrame({"cell_id": selected}).to_csv(output_dir / "selected_20_cells.csv", index=False)
    plot_overview(cycles, qc, output_dir)
    plot_selected_grid(cycles, qc, selected, output_dir)
    plot_individuals(cycles, qc, selected, output_dir)

    print(f"Selected cells: {', '.join(selected)}")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
