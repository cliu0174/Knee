from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "results/formal_414240_selector_N150_no_local_var"
ABLATION = ROOT / "results/formal_414240_observation_ablation_no_local_var"
LABEL = ROOT / "results/formal_414240_label_robustness_no_local_var"
CYCLE = ROOT / "results/formal_414240_cycle_pair_scan"
FIGURES = ROOT / "manuscript_ionics/figures"
SOURCE = ROOT / "manuscript_ionics/figure_source_data/formal_414240"

BLUE = "#0072B2"
GOLD = "#D55E00"
TEAL = "#0072B2"
VIOLET = "#7B61A8"
CHARCOAL = "#4E555B"
GRAY = "#A7AFB8"
PALE = "#DDE4EC"
GRID = "#D9DEE5"
PREVIEW_NOTICE: str | None = None


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 7.0,
        "axes.titlesize": 8.4,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.3,
        "legend.frameon": False,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
    })


def make_grid() -> tuple[plt.Figure, np.ndarray]:
    """Create identical panel geometry for every four-panel results figure."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45))
    fig.subplots_adjust(left=0.095, right=0.92, bottom=0.105, top=0.95, wspace=0.34, hspace=0.36)
    return fig, axes


def add_colorbar(fig: plt.Figure, im, label: str, *, label_size: float = 7.0, tick_size: float = 6.4) -> None:
    """Use the reserved outer margin so the heatmap panel keeps the same aspect ratio."""
    cax = fig.add_axes([0.942, 0.105, 0.012, 0.335])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=label_size, fontweight="bold")
    cbar.ax.tick_params(labelsize=tick_size, width=0.7, length=2.4)


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.45, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)


def emphasize_selector_axes(ax: plt.Axes) -> None:
    """Apply the Chapter-2-sized axis treatment to the selector comparison."""
    ax.xaxis.label.set_fontsize(9.0)
    ax.yaxis.label.set_fontsize(9.0)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.tick_params(axis="both", labelsize=7.4, pad=2.2)


def emphasize_selector_method_labels(ax: plt.Axes) -> None:
    """Increase the categorical method labels without crowding the panels."""
    for tick in ax.get_yticklabels():
        tick.set_fontsize(8.1)
        tick.set_rotation(18)
        tick.set_horizontalalignment("right")


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontweight="bold", fontsize=9,
            va="top", ha="left")


def save(fig: plt.Figure, stem: str, output_dir: Path | None = None) -> None:
    destination = FIGURES if output_dir is None else output_dir
    destination.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"facecolor": "white", "edgecolor": "none"}
    if PREVIEW_NOTICE:
        fig.text(0.5, 0.992, PREVIEW_NOTICE, ha="center", va="top", fontsize=6.4,
                 color="#8A4B08", fontweight="bold")
    fig.savefig(destination / f"{stem}.svg", **save_kwargs)
    fig.savefig(destination / f"{stem}.pdf", **save_kwargs)
    fig.savefig(destination / f"{stem}.png", dpi=600, **save_kwargs)
    plt.close(fig)


def bootstrap_metric(frame: pd.DataFrame, value: str = "abs_error", repeats: int = 4000) -> tuple[float, float, float]:
    values = frame[value].to_numpy(float)
    rng = np.random.default_rng(20260721)
    draws = np.empty(repeats)
    for idx in range(repeats):
        draws[idx] = np.mean(values[rng.integers(0, len(values), len(values))])
    return float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z**2 / (4.0 * total**2)) / denominator
    return float(center - half), float(center + half)


def bootstrap_performance_summary(frame: pd.DataFrame, repeats: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(20260721)
    rows: list[dict[str, float | str | int]] = []
    for split in sorted(frame["split"].unique()):
        part = frame[frame["split"] == split]
        observed = part["observed"].to_numpy(float)
        predicted = part["predicted_mean"].to_numpy(float)
        n = len(part)

        def calculate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
            errors = y - p
            absolute = np.abs(errors)
            denominator = np.sum((y - y.mean()) ** 2)
            return {
                "mae": float(absolute.mean()),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "r2": float(1.0 - np.sum(errors**2) / denominator),
                "q90": float(np.quantile(absolute, 0.9)),
            }

        point = calculate(observed, predicted)
        draws = {metric: np.empty(repeats) for metric in point}
        for draw_id in range(repeats):
            indices = rng.integers(0, n, n)
            sampled = calculate(observed[indices], predicted[indices])
            for metric, value in sampled.items():
                draws[metric][draw_id] = value
        for metric, value in point.items():
            rows.append({
                "split": split,
                "n": n,
                "metric": metric,
                "estimate": value,
                "ci_low": float(np.quantile(draws[metric], 0.025)),
                "ci_high": float(np.quantile(draws[metric], 0.975)),
            })
    return pd.DataFrame(rows)


def selector_figure() -> None:
    folds = pd.read_csv(SELECTOR / "development_outer_fold_metrics.csv")
    summary = pd.read_csv(SELECTOR / "development_summary.csv")
    jaccard = pd.read_csv(SELECTOR / "development_selection_jaccard.csv")
    selected = pd.read_csv(SELECTOR / "development_selected_features.csv")
    preferred_order = globals().get("PREFERRED_SELECTOR_ORDER")
    order = list(preferred_order) if preferred_order is not None else summary.sort_values("development_selection_score")["method"].tolist()
    labels = {
        "bootstrap_stable": "Bootstrap-stable",
        "contiguous": "Contiguous-band",
        "region_balanced": "Region-balanced",
        "pearson": "Pearson",
        "mrmr": "mRMR-style",
    }
    colors = [TEAL if item == order[0] else GRAY for item in order]
    rng = np.random.default_rng(13)

    fig, axes = make_grid()
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.105, top=0.95, wspace=0.36, hspace=0.36)
    ax = axes[0, 0]
    data = [folds.loc[folds.method == method, "mae"].to_numpy() for method in order]
    box = ax.boxplot(data, positions=np.arange(len(order)), widths=0.55, patch_artist=True,
                     showfliers=False, medianprops={"color": "white", "linewidth": 1.2})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color); patch.set_edgecolor(color); patch.set_alpha(0.86)
    for idx, values in enumerate(data):
        jitter = rng.uniform(-0.15, 0.15, len(values))
        ax.scatter(idx + jitter, values, s=11, facecolor="white", edgecolor=CHARCOAL, linewidth=0.45, zorder=3)
    ax.set_xticks(range(len(order)), [labels[item] for item in order], rotation=24, ha="right")
    ax.set_ylabel("Outer-fold MAE (cycles)")
    ax.set_ylim(20, 140)
    ax.set_yticks([30, 60, 90, 120])
    emphasize_selector_axes(ax)
    clean_axis(ax, "y"); panel(ax, "a")

    ax = axes[0, 1]
    method_markers = {
        "bootstrap_stable": "o",
        "contiguous": "^",
        "region_balanced": "s",
        "pearson": "D",
        "mrmr": "P",
    }
    for _, row in summary.iterrows():
        method = row["method"]
        ax.scatter(row["outer_mae_mean"], row["outer_mae_max"], s=45,
            marker=method_markers[method], color=TEAL if method == order[0] else GRAY,
                    edgecolor="white", linewidth=0.7, zorder=3)
    legend_handles = [
        Line2D([0], [0], marker=method_markers[item], color="none",
            markerfacecolor=TEAL if item == order[0] else GRAY, markeredgecolor="white",
               markeredgewidth=0.6, markersize=6.5, label=labels[item])
        for item in order
    ]
    ax.legend(handles=legend_handles, loc="upper left", ncol=1, fontsize=7.6,
              handletextpad=0.38, columnspacing=0.8, borderaxespad=0.25)
    ax.set_xlabel("Mean outer-fold MAE (cycles)")
    ax.set_ylabel("Worst outer-fold MAE (cycles)")
    scatter_limits = globals().get("SELECTOR_SCATTER_LIMITS")
    if scatter_limits is None:
        ax.set_xlim(51.8, 65.5)
        ax.set_ylim(107.5, 140.0)
    else:
        ax.set_xlim(*scatter_limits["xlim"])
        ax.set_ylim(*scatter_limits["ylim"])
    emphasize_selector_axes(ax)
    clean_axis(ax); panel(ax, "b")

    ax = axes[1, 0]
    jj = jaccard.set_index("method").loc[order]
    y = np.arange(len(order))
    ax.errorbar(jj["jaccard_mean"], y, xerr=jj["jaccard_std"], fmt="none", ecolor=CHARCOAL,
                elinewidth=0.9, capsize=2, zorder=2)
    ax.scatter(jj["jaccard_mean"], y, s=38, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(y, [labels[item] for item in order])
    ax.invert_yaxis(); ax.set_xlim(0, 1.02)
    ax.set_xlabel("Pairwise feature-set Jaccard")
    emphasize_selector_axes(ax)
    emphasize_selector_method_labels(ax)
    clean_axis(ax, "x"); panel(ax, "c")

    ax = axes[1, 1]
    heat = (selected.dropna(subset=["seg"]).groupby(["method", "seg"]).size()
            .unstack(fill_value=0).reindex(order).reindex(columns=range(39), fill_value=0))
    heat = heat / 10.0
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=max(1, heat.to_numpy().max()))
    ax.set_yticks(range(len(order)), [labels[item] for item in order])
    emphasize_selector_method_labels(ax)
    ticks = np.linspace(0, heat.shape[1] - 1, 5).astype(int)
    ax.set_xticks(ticks, [f"{int(heat.columns[i]) + 1:02d}" for i in ticks])
    ax.set_xlabel("Ordered discharge segment (1--39)")
    emphasize_selector_axes(ax)
    add_colorbar(fig, im, "Descriptors per fold", label_size=9.0, tick_size=7.4)
    panel(ax, "d")

    folds.to_csv(SOURCE / "fig2_selector_outer_folds.csv", index=False)
    summary.to_csv(SOURCE / "fig2_selector_summary.csv", index=False)
    jaccard.to_csv(SOURCE / "fig2_selector_jaccard.csv", index=False)
    heat.reset_index().to_csv(SOURCE / "fig2_selector_region_frequency.csv", index=False)
    save(fig, "fig2_formal_selector_stability")


def average_predictions(path: Path, group_cols: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.groupby(group_cols + ["cell_id", "observed"], as_index=False).agg(
        predicted=("predicted", "mean"), abs_error=("abs_error", "mean")
    )


def observation_figure(*, shared_legend: bool = False, panel_legends: bool = True) -> None:
    if shared_legend and panel_legends:
        raise ValueError("Choose either a shared legend or panel-specific legends.")
    cv = pd.read_csv(ABLATION / "observation_budget_train_cv.csv")
    held = average_predictions(
        ABLATION / "observation_budget_heldout_predictions.csv", ["window", "split"]
    )
    order = ["N50", "N100", "N150"]
    x = np.arange(3)
    fig, axes = make_grid()

    ax = axes[0, 0]
    stats = cv.groupby("window")["mae"].agg(["mean", "std"]).reindex(order)
    ax.errorbar(x, stats["mean"], yerr=stats["std"], color=CHARCOAL, marker="o", markersize=5,
                linewidth=1.4, capsize=3)
    for idx, window in enumerate(order):
        vals = cv.loc[cv.window == window, "mae"].to_numpy()
        ax.scatter(np.full(len(vals), idx) + np.linspace(-0.08, 0.08, len(vals)), vals,
                   s=9, facecolor="white", edgecolor=CHARCOAL, linewidth=0.4, zorder=3)
    ax.set_xticks(x, ["50", "100", "150"]); ax.set_xlabel("Observed cycles")
    ax.set_ylabel("Train-only CV MAE (cycles)")
    emphasize_selector_axes(ax)
    clean_axis(ax, "y"); panel(ax, "a")

    ci_rows = []
    ax = axes[0, 1]
    for split, color, marker, offset in [("primary_test", BLUE, "o", -0.05), ("secondary_test", GOLD, "s", 0.05)]:
        means, low, high = [], [], []
        for window in order:
            point, lo, hi = bootstrap_metric(held[(held.window == window) & (held.split == split)])
            means.append(point); low.append(point - lo); high.append(hi - point)
            ci_rows.append({"window": window, "split": split, "mae": point, "ci_low": lo, "ci_high": hi})
        ax.errorbar(x + offset, means, yerr=np.vstack([low, high]), color=color, marker=marker,
                    linewidth=1.4, markersize=5, capsize=3, label="Primary" if split.startswith("primary") else "Secondary")
    ax.set_xticks(x, ["50", "100", "150"]); ax.set_xlabel("Observed cycles")
    ax.set_ylabel("Test-cohort MAE (cycles)")
    emphasize_selector_axes(ax)
    if panel_legends:
        # Reserve a true empty corner above the longest confidence interval.
        ax.set_ylim(top=max(ax.dataLim.y1 * 1.25, ax.dataLim.y1 + 28))
        ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.995), borderaxespad=0.12, fontsize=7.2)
    elif not shared_legend:
        ax.legend(loc="upper right", fontsize=7.6)
    clean_axis(ax, "y"); panel(ax, "b")

    ax = axes[1, 0]
    pivot = pd.DataFrame(ci_rows).pivot(index="window", columns="split", values="mae").reindex(order)
    for split, color, marker in [("primary_test", BLUE, "o"), ("secondary_test", GOLD, "s")]:
        delta = pivot[split] - pivot.loc["N50", split]
        ax.plot(
            x,
            delta,
            color=color,
            marker=marker,
            linewidth=1.4,
            label="Primary" if split.startswith("primary") else "Secondary",
        )
    ax.axhline(0, color=CHARCOAL, linewidth=0.7)
    ax.set_xticks(x, ["50", "100", "150"]); ax.set_xlabel("Observed cycles")
    ax.set_ylabel(r"$\Delta$MAE vs 50 cycles")
    emphasize_selector_axes(ax)
    if panel_legends:
        ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.005), borderaxespad=0.12, fontsize=7.2)
    clean_axis(ax, "y"); panel(ax, "c")

    ax = axes[1, 1]
    widths = 0.34
    for pos, (split, color) in enumerate([("primary_test", BLUE), ("secondary_test", GOLD)]):
        q90 = [held[(held.window == w) & (held.split == split)].abs_error.quantile(0.9) for w in order]
        ax.bar(x + (pos - 0.5) * widths, q90, width=widths, color=color, alpha=0.88,
               label="Primary" if pos == 0 else "Secondary")
    ax.set_xticks(x, ["50", "100", "150"]); ax.set_xlabel("Observed cycles")
    ax.set_ylabel("90th-percentile error (cycles)")
    emphasize_selector_axes(ax)
    if panel_legends:
        # Keep a visible gap above the tallest bar for the vertical legend.
        top = max(250, int(np.ceil((ax.dataLim.y1 + 50) / 25) * 25))
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, top + 1, 25))
        ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.995), borderaxespad=0.12, fontsize=7.2)
    clean_axis(ax, "y"); panel(ax, "d")

    if shared_legend:
        handles, labels = axes[0, 1].get_legend_handles_labels()
        fig.subplots_adjust(top=0.88)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.51, 0.985),
            ncol=2,
            fontsize=7.6,
            columnspacing=1.7,
            handlelength=2.2,
        )

    cv.to_csv(SOURCE / "fig4_observation_train_cv.csv", index=False)
    held.to_csv(SOURCE / "fig4_observation_per_cell.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(SOURCE / "fig4_observation_bootstrap_ci.csv", index=False)
    save(fig, "fig4_observation_budget")


def cycle_pair_figure() -> None:
    cv = pd.read_csv(CYCLE / "cycle_pair_train_cv.csv")
    held = average_predictions(CYCLE / "cycle_pair_heldout_predictions.csv", ["cycle_pair", "split"])
    selected = pd.read_csv(CYCLE / "cycle_pair_selected_features.csv")
    order = ["10-50", "10-100", "10-150"]
    x = np.arange(3)
    fig, axes = make_grid()

    ax = axes[0, 0]
    stats = cv.groupby("cycle_pair")["mae"].agg(["mean", "std"]).reindex(order)
    ax.errorbar(x, stats["mean"], yerr=stats["std"], color=CHARCOAL, marker="o", capsize=3, linewidth=1.4)
    for idx, pair in enumerate(order):
        vals = cv.loc[cv.cycle_pair == pair, "mae"].to_numpy()
        ax.scatter(np.full(len(vals), idx) + np.linspace(-.08, .08, len(vals)), vals,
                   s=9, facecolor="white", edgecolor=CHARCOAL, linewidth=.4, zorder=3)
    ax.set_xticks(x, order); ax.set_xlabel(r"Local $\Delta Q$ cycle pair")
    ax.set_ylabel("Train-only CV MAE (cycles)"); ax.set_title("Train-only performance")
    clean_axis(ax, "y"); panel(ax, "a")

    ci_rows = []
    ax = axes[0, 1]
    for split, color, marker, offset in [("primary_test", BLUE, "o", -.05), ("secondary_test", GOLD, "s", .05)]:
        means, lows, highs = [], [], []
        for pair in order:
            point, lo, hi = bootstrap_metric(held[(held.cycle_pair == pair) & (held.split == split)])
            means.append(point); lows.append(point - lo); highs.append(hi - point)
            ci_rows.append({"cycle_pair": pair, "split": split, "mae": point, "ci_low": lo, "ci_high": hi})
        ax.errorbar(x + offset, means, yerr=np.vstack([lows, highs]), color=color, marker=marker,
                    capsize=3, linewidth=1.4, label="Primary" if split.startswith("primary") else "Secondary")
    ax.set_xticks(x, order); ax.set_xlabel(r"Local $\Delta Q$ cycle pair")
    ax.set_ylabel("Test-cohort MAE (cycles)"); ax.set_title("Held-out performance")
    ax.legend(loc="upper right"); clean_axis(ax, "y"); panel(ax, "b")

    ax = axes[1, 0]
    width = .34
    for idx, (split, color) in enumerate([("primary_test", BLUE), ("secondary_test", GOLD)]):
        q90 = [held[(held.cycle_pair == p) & (held.split == split)].abs_error.quantile(.9) for p in order]
        ax.bar(x + (idx - .5) * width, q90, width=width, color=color,
               label="Primary" if idx == 0 else "Secondary")
    ax.set_xticks(x, order); ax.set_xlabel(r"Local $\Delta Q$ cycle pair")
    ax.set_ylabel("90th-percentile error (cycles)"); ax.set_title("Tail error")
    clean_axis(ax, "y"); panel(ax, "c")

    ax = axes[1, 1]
    heat = (selected.dropna(subset=["seg"]).groupby(["cycle_pair", "seg"]).size()
            .unstack(fill_value=0).reindex(order).reindex(columns=range(39), fill_value=0))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=max(1, heat.to_numpy().max()))
    ax.set_yticks(range(3), order)
    ticks = np.linspace(0, heat.shape[1] - 1, 5).astype(int)
    ax.set_xticks(ticks, [f"{int(heat.columns[i]) + 1:02d}" for i in ticks])
    ax.set_xlabel("Ordered discharge segment (1--39)"); ax.set_ylabel("Cycle pair")
    ax.set_title("Selected local-region descriptor counts")
    add_colorbar(fig, im, "Descriptor count")
    panel(ax, "d")

    cv.to_csv(SOURCE / "fig5_cycle_pair_train_cv.csv", index=False)
    held.to_csv(SOURCE / "fig5_cycle_pair_per_cell.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(SOURCE / "fig5_cycle_pair_bootstrap_ci.csv", index=False)
    heat.reset_index().to_csv(SOURCE / "fig5_cycle_pair_region_counts.csv", index=False)
    save(fig, "fig5_cycle_pair_scan")


def heldout_figure(
    data_dir: Path = SELECTOR,
    *,
    stem: str = "fig3_heldout_performance",
    source_dir: Path = SOURCE,
    output_dir: Path | None = None,
) -> None:
    per_cell = pd.read_csv(data_dir / "heldout_per_cell.csv")
    # Define the point-prediction error from the frozen 30-seed mean prediction.
    # The stored abs_error may instead average seed-level absolute errors, which
    # differs when predictions from different seeds cross the observed target.
    per_cell["abs_error"] = (per_cell["observed"] - per_cell["predicted_mean"]).abs()
    metrics = pd.read_csv(data_dir / "heldout_metrics_by_seed.csv")
    fig, axes = make_grid()
    split_style = {"primary_test": (BLUE, "o", "Primary"), "secondary_test": (GOLD, "s", "Secondary")}

    ax = axes[0, 0]
    x_hi = per_cell.observed.max() + 30
    y_hi = per_cell.predicted_mean.max() + 30
    ax.plot([0, max(x_hi, y_hi)], [0, max(x_hi, y_hi)], color=CHARCOAL, linewidth=0.9, linestyle="--")
    for split, (color, marker, label) in split_style.items():
        part = per_cell[per_cell.split == split]
        ax.scatter(part.observed, part.predicted_mean, s=24, marker=marker, color=color,
                   alpha=0.82, edgecolor="white", linewidth=0.45, label=label)
    ax.set_xlim(0, x_hi); ax.set_ylim(0, y_hi); ax.set_aspect("auto")
    ax.set_xlabel("Observed knee cycle"); ax.set_ylabel("Predicted knee cycle")
    emphasize_selector_axes(ax); ax.legend(loc="lower right", fontsize=7.6)
    clean_axis(ax); panel(ax, "a")

    ax = axes[0, 1]
    for idx, (split, (color, _, label)) in enumerate(split_style.items()):
        values = per_cell.loc[per_cell.split == split, "abs_error"].to_numpy()
        vp = ax.violinplot(values, positions=[idx], widths=0.65, showextrema=False)
        vp["bodies"][0].set_facecolor(color); vp["bodies"][0].set_alpha(0.35)
        ax.boxplot(values, positions=[idx], widths=0.25, showfliers=False,
                   medianprops={"color": color, "linewidth": 1.3},
                   boxprops={"color": CHARCOAL}, whiskerprops={"color": CHARCOAL}, capprops={"color": CHARCOAL})
    ax.set_xticks([0, 1], ["Primary\n(n=42)", "Secondary\n(n=40)"])
    ax.set_ylabel("Absolute error (cycles)")
    emphasize_selector_axes(ax)
    clean_axis(ax, "y"); panel(ax, "b")

    ax = axes[1, 0]
    top = per_cell.nlargest(12, "abs_error").sort_values("abs_error")
    y = np.arange(len(top))
    c = [split_style[s][0] for s in top.split]
    ax.hlines(y, 0, top.abs_error, color=PALE, linewidth=2)
    ax.scatter(top.abs_error, y, color=c, s=28, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, top.cell_id)
    ax.set_xlim(left=0); ax.set_xlabel("Absolute error (cycles)")
    emphasize_selector_axes(ax)
    clean_axis(ax, "x"); panel(ax, "c")

    ax = axes[1, 1]
    for split, (color, _, label) in split_style.items():
        values = metrics.loc[metrics.split == split, "mae"].to_numpy()
        ax.plot(metrics.loc[metrics.split == split, "seed"], values, color=color, alpha=0.55, linewidth=0.8)
        ax.axhline(values.mean(), color=color, linewidth=1.6, label=f"{label}: {values.mean():.1f}")
    ax.set_xlabel("Prespecified random seed"); ax.set_ylabel("MAE (cycles)")
    ax.set_ylim(*globals().get("HELDOUT_SEED_YLIM", (40, 100)))
    emphasize_selector_axes(ax); ax.legend(loc="center right", fontsize=7.6)
    clean_axis(ax, "y"); panel(ax, "d")

    source_dir.mkdir(parents=True, exist_ok=True)
    source_stem = "fig3_heldout" if stem == "fig3_heldout_performance" else stem
    per_cell.to_csv(source_dir / f"{source_stem}_per_cell.csv", index=False)
    metrics.to_csv(source_dir / f"{source_stem}_seed_metrics.csv", index=False)
    bootstrap_performance_summary(per_cell).to_csv(source_dir / f"{source_stem}_metric_ci.csv", index=False)
    save(fig, stem, output_dir=output_dir)


def ablation_figure() -> None:
    per_cell = average_predictions(
        ABLATION / "feature_ablation_heldout_predictions.csv", ["feature_group", "split"]
    )
    metrics = pd.read_csv(ABLATION / "feature_ablation_heldout_metrics.csv")
    groups = ["raw", "raw_global_deltaq", "raw_local_deltaq", "full_global_local"]
    labels = ["Raw/SOH", "Raw/SOH +\nGlobal $\\Delta Q$", "Raw/SOH + selected\n$\\Delta Q$-derived", "Full\nrepresentation"]
    fig, axes = make_grid()
    ci_rows = []

    for ax, split, letter in [(axes[0, 0], "primary_test", "a"), (axes[0, 1], "secondary_test", "b")]:
        points, lows, highs = [], [], []
        for group in groups:
            point, lo, hi = bootstrap_metric(per_cell[(per_cell.feature_group == group) & (per_cell.split == split)])
            points.append(point); lows.append(point - lo); highs.append(hi - point)
            ci_rows.append({"feature_group": group, "split": split, "mae": point, "ci_low": lo, "ci_high": hi})
        colors = [GRAY, BLUE, TEAL, CHARCOAL]
        ax.errorbar(np.arange(4), points, yerr=np.vstack([lows, highs]), fmt="none", ecolor=CHARCOAL,
                    capsize=3, linewidth=0.9)
        ax.scatter(np.arange(4), points, s=45, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_xticks(range(4), labels)
        ax.set_ylabel("MAE (cycles)"); ax.set_title("Primary test" if split.startswith("primary") else "Secondary test")
        clean_axis(ax); panel(ax, letter)

    ax = axes[1, 0]
    full = per_cell[per_cell.feature_group == "full_global_local"][["split", "cell_id", "abs_error"]].rename(columns={"abs_error": "full_error"})
    delta_rows = []
    for group in groups[:-1]:
        merged = per_cell[per_cell.feature_group == group].merge(full, on=["split", "cell_id"])
        merged["delta_error"] = merged["abs_error"] - merged["full_error"]
        for split in ["primary_test", "secondary_test"]:
            vals = merged.loc[merged.split == split, "delta_error"].to_numpy()
            rng = np.random.default_rng(20260721)
            boot = [np.mean(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(4000)]
            delta_rows.append({"feature_group": group, "split": split, "delta_mae": vals.mean(),
                               "ci_low": np.quantile(boot, .025), "ci_high": np.quantile(boot, .975)})
    delta = pd.DataFrame(delta_rows)
    y = np.arange(3)
    for split, color, offset, marker in [("primary_test", BLUE, -0.1, "o"), ("secondary_test", GOLD, 0.1, "s")]:
        part = delta[delta.split == split].set_index("feature_group").loc[groups[:-1]]
        ax.errorbar(part.delta_mae, y + offset,
                    xerr=np.vstack([part.delta_mae - part.ci_low, part.ci_high - part.delta_mae]),
                    fmt=marker, color=color, capsize=2.5, markersize=4, label="Primary" if split.startswith("primary") else "Secondary")
    ax.axvline(0, color=CHARCOAL, linewidth=0.7)
    ax.set_yticks(y, labels[:-1]); ax.invert_yaxis()
    ax.set_xlabel(r"Paired mean difference in absolute error vs full (cycles)")
    ax.set_title("Paired absolute-error difference"); ax.legend(loc="lower right")
    clean_axis(ax, "x"); panel(ax, "c")

    ax = axes[1, 1]
    counts = metrics.groupby("feature_group")["feature_count"].first().reindex(groups)
    bars = ax.bar(range(4), counts, color=[GRAY, BLUE, TEAL, CHARCOAL], width=0.65)
    ax.bar_label(bars, labels=[str(int(v)) for v in counts], padding=2, fontsize=7)
    ax.set_xticks(range(4), labels, fontsize=6.3)
    ax.set_ylabel("Input descriptors"); ax.set_title("Input dimensionality")
    clean_axis(ax, "y"); panel(ax, "d")

    per_cell.to_csv(SOURCE / "fig6_ablation_per_cell.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(SOURCE / "fig6_ablation_bootstrap_ci.csv", index=False)
    delta.to_csv(SOURCE / "fig6_ablation_paired_delta.csv", index=False)
    save(fig, "fig6_formal_representation_ablation")


def reliability_label_figure() -> None:
    per_cell = pd.read_csv(SELECTOR / "heldout_per_cell.csv")
    # Use the same cell-level point predictions as the performance figure.
    per_cell["abs_error"] = (per_cell["observed"] - per_cell["predicted_mean"]).abs()
    oof = pd.read_csv(SELECTOR / "development_oof_predictions.csv")
    frozen = json.loads((SELECTOR / "frozen_config.json").read_text(encoding="utf-8"))
    oof = oof[oof.method == frozen["method"]].groupby(["cell_id", "observed"], as_index=False).agg(predicted=("predicted", "mean"))
    residuals = np.abs(oof.observed - oof.predicted).to_numpy()
    level = min(1.0, np.ceil((len(residuals) + 1) * 0.9) / len(residuals))
    half_width = float(np.quantile(residuals, level, method="higher"))
    intervals = per_cell.copy()
    intervals["lower"] = intervals.predicted_mean - half_width
    intervals["upper"] = intervals.predicted_mean + half_width
    intervals["covered"] = (intervals.observed >= intervals.lower) & (intervals.observed <= intervals.upper)
    label_pred = average_predictions(LABEL / "label_robustness_predictions.csv", ["label_method", "split"])

    # The two panels answer distinct reliability questions: calibration of the
    # prediction interval and sensitivity to the offline knee-label rule.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.20, top=0.93, wspace=0.42)
    ax = axes[0]
    coverage_rows = []
    for split in ["primary_test", "secondary_test"]:
        values = intervals.loc[intervals.split == split, "covered"]
        successes, total = int(values.sum()), int(values.size)
        lo, hi = wilson_interval(successes, total)
        coverage_rows.append({
            "split": split,
            "covered": successes,
            "n": total,
            "coverage": successes / total,
            "ci_low": lo,
            "ci_high": hi,
        })
    coverage_frame = pd.DataFrame(coverage_rows).set_index("split")
    coverage = coverage_frame["coverage"].reindex(["primary_test", "secondary_test"])
    low = coverage - coverage_frame["ci_low"].reindex(coverage.index)
    high = coverage_frame["ci_high"].reindex(coverage.index) - coverage
    bars = ax.bar([0, 1], coverage, color=[BLUE, GOLD], width=0.58)
    ax.errorbar([0, 1], coverage, yerr=np.vstack([low, high]), fmt="none",
                ecolor=CHARCOAL, capsize=3, linewidth=0.9)
    ax.axhline(0.9, color=CHARCOAL, linestyle="--", linewidth=0.9)
    for xpos, value, upper in zip([0, 1], coverage, coverage_frame["ci_high"].reindex(coverage.index)):
        label_y = min(max(float(upper) + 0.035, float(value) + 0.08), 1.07)
        ax.text(xpos, label_y, f"{value:.2f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax.set_ylim(0, 1.10); ax.set_xticks([0, 1], ["Primary", "Secondary"])
    ax.set_ylabel("Empirical coverage")
    emphasize_selector_axes(ax)
    clean_axis(ax, "y"); panel(ax, "a")

    ax = axes[1]
    methods = ["piecewise", "kneedle", "rate_threshold"]
    names = ["Piecewise", "Chord distance", "Rate threshold"]
    y = np.arange(3)
    label_ci_rows = []
    for split, color, offset, marker in [("primary_test", BLUE, -0.1, "o"), ("secondary_test", GOLD, 0.1, "s")]:
        means, lows, highs = [], [], []
        for method in methods:
            point, lo, hi = bootstrap_metric(label_pred[(label_pred.label_method == method) & (label_pred.split == split)])
            means.append(point); lows.append(point - lo); highs.append(hi - point)
            label_ci_rows.append({"label_method": method, "split": split, "mae": point, "ci_low": lo, "ci_high": hi})
        ax.errorbar(means, y + offset, xerr=np.vstack([lows, highs]), fmt=marker, color=color,
                    capsize=2.5, markersize=4, label="Primary" if split.startswith("primary") else "Secondary")
    ax.set_yticks(y, names); ax.invert_yaxis(); ax.set_xlabel("MAE (cycles)")
    emphasize_selector_axes(ax); ax.legend(loc="upper right", ncol=1, fontsize=7.6)
    clean_axis(ax, "x"); panel(ax, "b")

    intervals.to_csv(SOURCE / "fig7_prediction_intervals.csv", index=False)
    oof.to_csv(SOURCE / "fig7_oof_calibration_predictions.csv", index=False)
    coverage_frame.reset_index().to_csv(SOURCE / "fig7_coverage_ci.csv", index=False)
    label_pred.to_csv(SOURCE / "fig7_label_per_cell.csv", index=False)
    pd.DataFrame(label_ci_rows).to_csv(SOURCE / "fig7_label_bootstrap_ci.csv", index=False)
    save(fig, "fig7_formal_reliability_and_label_robustness")


def main() -> None:
    setup_style()
    SOURCE.mkdir(parents=True, exist_ok=True)
    selector_figure()
    observation_figure()
    cycle_pair_figure()
    heldout_figure()
    ablation_figure()
    reliability_label_figure()
    print(f"Formal figures written to {FIGURES}")


if __name__ == "__main__":
    main()
