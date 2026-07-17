from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Mandatory editable-text and publication export settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "manuscript_ionics" / "figures"
SOURCE_DIR = ROOT / "manuscript_ionics" / "figure_source_data"
QA_DIR = ROOT / "manuscript_ionics" / "figure_qa"
N150 = ROOT / "results" / "nested_contiguous_ldq_N150_20260715_nonoverlap"
N100 = ROOT / "results" / "nested_contiguous_ldq_N100_commonN150_5x2_20260715_nonoverlap"
N50 = ROOT / "results" / "nested_contiguous_ldq_N50_commonN150_5x2_20260715_nonoverlap"
ABLATION = ROOT / "results" / "nested_ablation_learning_N150_20260715_nonoverlap"
REPRO = ROOT / "results" / "reproduction_ldq_et_N150_20260713"

BLUE = "#0F4D92"
BLUE2 = "#3775BA"
RED = "#B64342"
TEAL = "#42949E"
VIOLET = "#8055A5"
GOLD = "#D99B22"
GREY = "#767676"
LIGHT = "#D9E2EC"
BLACK = "#272727"
METHOD_COLORS = {
    "region_balanced": BLUE,
    "bootstrap_stable": TEAL,
    "mrmr": VIOLET,
    "contiguous": GOLD,
    "pearson": GREY,
}


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def export(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    for suffix, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 600},
    }.items():
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def save_source(frame: pd.DataFrame, name: str) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SOURCE_DIR / name, index=False)


def clean_method(value: str) -> str:
    return {
        "region_balanced": "Region-balanced",
        "bootstrap_stable": "Bootstrap-stable",
        "mrmr": "mRMR",
        "contiguous": "Contiguous band",
        "pearson": "Pearson top-k",
    }.get(value, value)


def bootstrap_difference(diff: np.ndarray, seed: int = 20260713, repeats: int = 5000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), size=(repeats, len(diff)))
    values = np.mean(diff[indices], axis=1)
    return float(np.mean(diff)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def figure1_protocol() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_axis_off()
    boxes = [
        (0.03, 0.36, 0.19, 0.38, "Corrected dataset", "124 LFP cells\nSOH correction\npiecewise knee labels", LIGHT),
        (0.28, 0.36, 0.19, 0.38, "Common cohort", "N50 / N100 / N150\n83 development cells\n40 secondary cells", "#DDF3DE"),
        (0.53, 0.36, 0.19, 0.38, "Strict development", "5-fold CV × 2\nselection inside folds\nselector + model choice", "#E8DFF2"),
        (0.78, 0.36, 0.19, 0.38, "Immutable freeze", "region-balanced top40\n14 regions; ET anchor\n30 fixed seeds", "#F6E7C8"),
    ]
    for x, y, w, h, title, body, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015", facecolor=color, edgecolor=BLACK, linewidth=0.9)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="center", fontweight="bold", fontsize=9)
        ax.text(x + w / 2, y + 0.15, body, ha="center", va="center", linespacing=1.45, fontsize=8)
    for x in [0.22, 0.47, 0.72]:
        ax.add_patch(FancyArrowPatch((x + 0.01, 0.55), (x + 0.05, 0.55), arrowstyle="-|>", mutation_scale=12, color=GREY, linewidth=1.2))
    ax.text(0.5, 0.90, "Development labels drive every selection decision", ha="center", fontsize=10, fontweight="bold", color=BLUE)
    ax.add_patch(FancyArrowPatch((0.875, 0.34), (0.875, 0.20), arrowstyle="-|>", mutation_scale=12, color=RED, linewidth=1.2))
    ax.text(0.875, 0.10, "Protocol-frozen batch evaluation\nno formal model-development input", ha="center", va="center", fontsize=8.5, color=RED, fontweight="bold")
    ax.text(0.03, 0.10, "Historical anchor reproduced exactly: MAE 77.23 ± 0.89 cycles", ha="left", va="center", fontsize=8.5, color=BLACK)
    ax.text(0.03, 0.02, "No cell is split across folds; repeated seeds quantify algorithmic variation, not independent external cohorts.", ha="left", va="bottom", fontsize=7.5, color=GREY)
    source = pd.DataFrame(
        [
            {"stage": "source", "cells": 124, "selection_labels": False, "note": "corrected Severson release"},
            {"stage": "development", "cells": 83, "selection_labels": True, "note": "official train + primary"},
            {"stage": "secondary", "cells": 40, "selection_labels": False, "note": "protocol-frozen batch evaluation"},
        ]
    )
    save_source(source, "fig1_protocol_source.csv")
    export(fig, "fig1_protocol_nested_workflow")


def figure2_selectors() -> None:
    folds = pd.read_csv(N150 / "development_outer_fold_metrics.csv")
    summary = pd.read_csv(N150 / "development_summary.csv")
    jac = pd.read_csv(N150 / "development_selection_jaccard.csv")
    paired = pd.read_csv(N150 / "development_paired_comparisons.csv")
    order = summary.sort_values("development_selection_score")["method"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = axes[0, 0]
    data = [folds.loc[folds.method == method, "mae"].to_numpy() for method in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.65, showfliers=False)
    for patch, method in zip(bp["boxes"], order):
        patch.set_facecolor(METHOD_COLORS[method]); patch.set_alpha(0.78)
    ax.set_xticklabels([clean_method(x) for x in order], rotation=24, ha="right")
    ax.set_ylabel("Outer-fold MAE (cycles)")
    ax.set_title("Strict nested performance")
    panel(ax, "a")

    ax = axes[0, 1]
    ranked = summary.set_index("method").loc[order].reset_index()
    y = np.arange(len(order))
    ax.barh(y, ranked.development_selection_score, color=[METHOD_COLORS[m] for m in order], alpha=0.82)
    ax.set_yticks(y, [clean_method(x) for x in order]); ax.invert_yaxis()
    ax.set_xlabel("Predeclared selection score")
    ax.axvline(ranked.development_selection_score.min(), color=BLACK, linestyle="--", linewidth=0.8)
    ax.set_title("Mean + variability + worst fold + complexity")
    panel(ax, "b")

    ax = axes[1, 0]
    js = jac.set_index("method").loc[order].reset_index()
    ax.bar(np.arange(len(order)), js.jaccard_mean, yerr=js.jaccard_std, capsize=3, color=[METHOD_COLORS[m] for m in order], alpha=0.82)
    ax.set_xticks(np.arange(len(order)), [clean_method(x) for x in order], rotation=24, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Feature-set Jaccard")
    ax.set_title("Selection stability across 10 outer folds")
    panel(ax, "c")

    ax = axes[1, 1]
    rows = []
    for method in order:
        if method == "region_balanced":
            continue
        match = paired[((paired.method_a == method) & (paired.method_b == "region_balanced")) | ((paired.method_b == method) & (paired.method_a == "region_balanced"))].iloc[0]
        sign = 1.0 if match.method_a == method else -1.0
        rows.append({"method": method, "estimate": sign * match.mean_abs_error_difference_a_minus_b, "low": sign * (match.bootstrap_ci_low if sign > 0 else match.bootstrap_ci_high), "high": sign * (match.bootstrap_ci_high if sign > 0 else match.bootstrap_ci_low)})
    forest = pd.DataFrame(rows)
    yy = np.arange(len(forest))[::-1]
    for yi, row in zip(yy, forest.itertuples()):
        ax.plot([row.low, row.high], [yi, yi], color=METHOD_COLORS[row.method], linewidth=1.5)
        ax.plot(row.estimate, yi, "o", color=METHOD_COLORS[row.method], ms=5)
    ax.axvline(0, color=GREY, linestyle="--", linewidth=0.9)
    ax.set_yticks(yy, [clean_method(x) for x in forest.method])
    ax.set_xlabel("Paired MAE difference vs region-balanced (cycles)")
    ax.set_title("Cell-bootstrap 95% CI")
    panel(ax, "d")

    source = folds.merge(summary[["method", "development_selection_score"]], on="method").merge(jac[["method", "jaccard_mean", "jaccard_std"]], on="method")
    save_source(source, "fig2_selector_fold_source.csv")
    save_source(forest, "fig2_selector_paired_source.csv")
    export(fig, "fig2_strict_nested_selector_comparison")


def selected_budget(path: Path, budget: int) -> dict[str, float | str]:
    table = pd.read_csv(path / "development_summary.csv").sort_values("development_selection_score")
    row = table.iloc[0]
    return {"budget": budget, **row.to_dict()}


def figure3_budget_learning() -> None:
    budgets = pd.DataFrame([selected_budget(N50, 50), selected_budget(N100, 100), selected_budget(N150, 150)])
    learning = pd.read_csv(ABLATION / "fold_metrics.csv")
    learning = learning[learning.analysis == "learning_curve"].copy()
    curve = learning.groupby("fraction", as_index=False).agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), rmse_mean=("rmse", "mean"), n_train=("n_train", "mean"), runs=("fold", "count"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ax = axes[0, 0]
    ax.errorbar(budgets.budget, budgets.outer_mae_mean, yerr=budgets.outer_mae_std, marker="o", ms=6, color=BLUE, capsize=4, linewidth=1.8)
    for row in budgets.itertuples():
        ax.annotate(clean_method(row.method), (row.budget, row.outer_mae_mean), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks([50, 100, 150]); ax.set_xlabel("Observed cycles"); ax.set_ylabel("Nested MAE (cycles)")
    ax.set_title("Common-cohort observation budget"); panel(ax, "a")

    ax = axes[0, 1]
    ax.plot(budgets.budget, budgets.outer_rmse_mean, marker="s", color=RED, linewidth=1.8)
    ax.set_xticks([50, 100, 150]); ax.set_xlabel("Observed cycles"); ax.set_ylabel("Nested RMSE (cycles)")
    ax.set_title("Tail-sensitive error"); panel(ax, "b")

    ax = axes[1, 0]
    ax.errorbar(curve.fraction * 100, curve.mae_mean, yerr=curve.mae_std, marker="o", color=TEAL, capsize=4, linewidth=1.8)
    ax.set_xticks([25, 50, 75, 100]); ax.set_xlabel("Outer-training data used (%)"); ax.set_ylabel("MAE (cycles)")
    ax.set_title("Repeated subsampling learning curve"); panel(ax, "c")

    ax = axes[1, 1]
    ax.bar(curve.fraction * 100, curve.n_train, width=13, color=LIGHT, edgecolor=BLUE)
    for x, y in zip(curve.fraction * 100, curve.n_train):
        ax.text(x, y + 1.2, f"{y:.1f}", ha="center", fontsize=7)
    ax.set_xticks([25, 50, 75, 100]); ax.set_xlabel("Outer-training data used (%)"); ax.set_ylabel("Mean training cells")
    ax.set_title("Effective sample size"); panel(ax, "d")
    save_source(budgets, "fig3_budget_source.csv"); save_source(curve, "fig3_learning_curve_source.csv")
    export(fig, "fig3_observation_budget_and_data_efficiency")


def parse_segment(feature: str) -> tuple[int | None, float | None, float | None]:
    match = re.search(r"_ldq_seg(\d+)_([0-9]{4})_([0-9]{4})_", feature)
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)) / 1000.0, int(match.group(3)) / 1000.0


def figure4_regions() -> None:
    frozen = pd.read_csv(N150 / "frozen_local_features.csv")
    frozen[["segment", "start", "stop"]] = frozen.feature.apply(lambda x: pd.Series(parse_segment(x)))
    region_counts = frozen.dropna(subset=["segment"]).groupby(["segment", "start", "stop"], as_index=False).size().rename(columns={"size": "selected_features"})
    region_counts["midpoint"] = (region_counts.start + region_counts.stop) / 2
    selected = pd.read_csv(N150 / "development_selected_features.csv")
    selected = selected[selected.method == "region_balanced"].copy()
    selected["segment"] = selected.feature.apply(lambda x: parse_segment(x)[0])
    heat = selected.dropna(subset=["segment"]).groupby(["outer_id", "segment"]).size().unstack(fill_value=0)
    controls = pd.read_csv(ABLATION / "fold_metrics.csv")
    region = controls[controls.analysis == "region_control"].copy()
    region["family"] = np.select(
        [region.variant.str.startswith("random_band"), region.variant.str.startswith("random_features")],
        ["Random equal-width bands", "Random 40 features"],
        default=region.variant.map({"all_regions_top40": "All regions top40", "core_0525_0650_top40": "Core only", "exclude_core_top40": "Exclude core", "selected_minus_core": "Selected minus core"}),
    )
    region = region[region.family.notna()].copy()
    permutation = controls[controls.analysis == "region_permutation"].copy()

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(region_counts.midpoint, region_counts.selected_features, width=0.021, color=BLUE2, edgecolor=BLUE)
    ax.axvspan(0.525, 0.650, color=GOLD, alpha=0.18, label="historical core")
    ax.set_xlabel("Normalized discharge coordinate"); ax.set_ylabel("Frozen selected descriptors")
    ax.set_title("Frozen 14-region representation"); ax.legend(loc="upper left", fontsize=7); panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=3)
    ax.set_xlabel("Ordered region index"); ax.set_ylabel("Outer fold")
    ax.set_title("Per-fold selected descriptors")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="count")
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    families = ["All regions top40", "Core only", "Exclude core", "Selected minus core", "Random equal-width bands", "Random 40 features"]
    values = [region.loc[region.family == f, "mae"].to_numpy() for f in families]
    bp = ax.boxplot(values, patch_artist=True, widths=0.65, showfliers=False)
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(BLUE if idx < 4 else LIGHT); patch.set_alpha(0.8)
    ax.set_xticklabels(families, rotation=24, ha="right"); ax.set_ylabel("Outer-fold MAE (cycles)")
    ax.set_title("Predictive signal spans multiple regions"); panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    baseline = region.loc[region.family == "All regions top40", "mae"].to_numpy()
    perm = permutation.mae.to_numpy()
    ax.boxplot([baseline, perm], tick_labels=["Unpermuted", "Core permuted"], patch_artist=True, boxprops={"facecolor": LIGHT}, medianprops={"color": RED})
    ax.set_ylabel("MAE (cycles)"); ax.set_title("Core-region reliance diagnostic"); panel(ax, "d")

    save_source(region_counts, "fig4_frozen_regions_source.csv")
    heat.reset_index().to_csv(SOURCE_DIR / "fig4_region_fold_heatmap_source.csv", index=False)
    save_source(region, "fig4_region_controls_source.csv"); save_source(permutation, "fig4_region_permutation_source.csv")
    export(fig, "fig4_local_region_contribution_and_controls")


def figure5_external() -> None:
    ext = pd.read_csv(N150 / "external_prediction_intervals.csv")
    seeds = pd.read_csv(N150 / "external_metrics_by_seed.csv")
    old = pd.read_csv(REPRO / "extratrees_hpo_final_predictions.csv")
    old = old[old.official_split == "secondary_test"].groupby(["cell_id", "observed_knee_cycle"], as_index=False).agg(old_prediction=("predicted_knee_cycle", "mean"))
    merged = ext.merge(old, on="cell_id")
    merged["old_abs_error"] = np.abs(merged.observed - merged.old_prediction)
    merged["new_abs_error"] = np.abs(merged.observed - merged.predicted_mean)
    ext["residual"] = ext.predicted_mean - ext.observed

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    ax = axes[0, 0]
    error = np.vstack([ext.predicted_mean - ext.interval_lower, ext.interval_upper - ext.predicted_mean])
    covered_colors = np.where(ext.covered.astype(str).str.lower().isin(["true", "1"]), BLUE2, RED)
    ax.errorbar(ext.observed, ext.predicted_mean, yerr=error, fmt="none", ecolor=LIGHT, elinewidth=0.8, alpha=0.8)
    ax.scatter(ext.observed, ext.predicted_mean, c=covered_colors, s=20, edgecolor="white", linewidth=0.3)
    limits = [min(ext.observed.min(), ext.interval_lower.min()) - 30, max(ext.observed.max(), ext.interval_upper.max()) + 30]
    ax.plot(limits, limits, "--", color=GREY, linewidth=1); ax.set_xlim(limits); ax.set_ylim(limits)
    ax.set_xlabel("Observed knee cycle"); ax.set_ylabel("Frozen prediction")
    ax.set_title("Secondary batch with empirical 90% intervals"); panel(ax, "a")

    ax = axes[0, 1]
    ax.axhline(0, color=GREY, linestyle="--", linewidth=0.9)
    ax.scatter(ext.observed, ext.residual, c=ext.abs_error, cmap="magma", s=24)
    ax.set_xlabel("Observed knee cycle"); ax.set_ylabel("Prediction − observation (cycles)")
    ax.set_title("Residual structure under batch shift"); panel(ax, "b")

    ax = axes[1, 0]
    for row in merged.itertuples():
        ax.plot([0, 1], [row.old_abs_error, row.new_abs_error], color=LIGHT, linewidth=0.7)
    ax.scatter(np.zeros(len(merged)), merged.old_abs_error, color=GREY, s=15, label="Historical anchor")
    ax.scatter(np.ones(len(merged)), merged.new_abs_error, color=BLUE, s=15, label="Frozen nested")
    ax.set_xticks([0, 1], ["Historical\ntop40", "Frozen\nregion-balanced"]); ax.set_ylabel("Cell absolute error (cycles)")
    ax.set_title("Paired difference −0.61 cycles; p=0.778"); panel(ax, "c")

    ax = axes[1, 1]
    ax.hist(seeds.mae, bins=10, color=BLUE2, edgecolor="white", alpha=0.9)
    ax.axvline(seeds.mae.mean(), color=RED, linewidth=1.3, label=f"mean {seeds.mae.mean():.2f}")
    ax.set_xlabel("External MAE across seeds (cycles)"); ax.set_ylabel("Seed count")
    ax.set_title("Algorithmic variation (30 seeds)"); ax.legend(fontsize=7); panel(ax, "d")
    save_source(ext, "fig5_external_predictions_source.csv"); save_source(merged, "fig5_external_paired_source.csv"); save_source(seeds, "fig5_external_seed_source.csv")
    export(fig, "fig5_frozen_secondary_batch_performance")


def figure6_ablation_models() -> None:
    fold = pd.read_csv(ABLATION / "fold_metrics.csv")
    pred = pd.read_csv(ABLATION / "oof_predictions.csv")
    groups = fold[fold.analysis == "feature_group"].copy()
    models = fold[fold.analysis == "model_family"].copy()
    group_summary = groups.groupby("variant", as_index=False).agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), feature_count=("n_features", "mean"))
    model_summary = models.groupby("variant", as_index=False).agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"))
    full = pred[(pred.analysis == "feature_group") & (pred.variant == "raw_global_local_deltaq")].groupby("cell_id", as_index=False).agg(full_error=("abs_error", "mean"))
    rows = []
    for variant, part in pred[pred.analysis == "feature_group"].groupby("variant"):
        other = part.groupby("cell_id", as_index=False).agg(error=("abs_error", "mean")).merge(full, on="cell_id")
        estimate, low, high = bootstrap_difference((other.error - other.full_error).to_numpy(), seed=20260713 + len(rows))
        rows.append({"variant": variant, "mae_difference_vs_full": estimate, "ci_low": low, "ci_high": high})
    differences = pd.DataFrame(rows)

    label_map = {
        "raw_only": "Raw only", "global_deltaq_only": "Global ΔQ only", "local_deltaq_only": "Local ΔQ only",
        "raw_global_deltaq": "Raw + global", "raw_local_deltaq": "Raw + local", "global_local_deltaq": "Global + local",
        "raw_global_local_deltaq": "Raw + global + local",
    }
    order = ["raw_only", "global_deltaq_only", "local_deltaq_only", "raw_global_deltaq", "raw_local_deltaq", "global_local_deltaq", "raw_global_local_deltaq"]
    group_summary = group_summary.set_index("variant").loc[order].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax = axes[0, 0]
    y = np.arange(len(order))
    ax.barh(y, group_summary.mae_mean, xerr=group_summary.mae_std, color=[LIGHT] * 3 + [BLUE2] * 4, capsize=3)
    ax.set_yticks(y, [label_map[x] for x in order]); ax.invert_yaxis(); ax.set_xlabel("Nested MAE (cycles)")
    ax.set_title("Seven feature groups"); panel(ax, "a")

    ax = axes[0, 1]
    model_order = model_summary.sort_values("mae_mean").variant.tolist()
    ms = model_summary.set_index("variant").loc[model_order].reset_index()
    ax.bar(np.arange(len(ms)), ms.mae_mean, yerr=ms.mae_std, color=[BLUE if x == "extra_trees_compact" else GREY for x in ms.variant], capsize=3)
    ax.set_yscale("log"); ax.set_xticks(np.arange(len(ms)), ms.variant.str.replace("_", " "), rotation=25, ha="right")
    ax.set_ylabel("Nested MAE (cycles; log scale)"); ax.set_title("Representative model families"); panel(ax, "b")

    ax = axes[1, 0]
    diff = differences[differences.variant != "raw_global_local_deltaq"].set_index("variant").loc[order[:-1]].reset_index()
    yy = np.arange(len(diff))[::-1]
    for yi, row in zip(yy, diff.itertuples()):
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=TEAL, linewidth=1.4)
        ax.plot(row.mae_difference_vs_full, yi, "o", color=TEAL, ms=4)
    ax.axvline(0, color=GREY, linestyle="--", linewidth=0.9)
    ax.set_yticks(yy, [label_map[x] for x in diff.variant]); ax.set_xlabel("Paired MAE difference vs full (cycles)")
    ax.set_title("Cell-bootstrap 95% CI"); panel(ax, "c")

    ax = axes[1, 1]
    ax.bar(np.arange(len(order)), group_summary.feature_count, color=[LIGHT] * 3 + [BLUE2] * 4)
    ax.set_xticks(np.arange(len(order)), [label_map[x] for x in order], rotation=28, ha="right")
    ax.set_ylabel("Mean features per outer fold"); ax.set_title("Representation size"); panel(ax, "d")
    save_source(group_summary, "fig6_feature_group_source.csv"); save_source(model_summary, "fig6_model_family_source.csv"); save_source(differences, "fig6_paired_ablation_source.csv")
    export(fig, "fig6_feature_group_and_model_family_ablation")


def figure7_uncertainty_tail_complexity() -> None:
    interval = pd.read_csv(N150 / "external_interval_summary.csv")
    ext = pd.read_csv(N150 / "external_per_cell.csv").sort_values("abs_error", ascending=False)
    seeds = pd.read_csv(N150 / "external_metrics_by_seed.csv")
    tail = pd.read_csv(N150 / "external_tail_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    ax = axes[0, 0]
    coverage = interval.set_index("stratum").loc[["all", "early", "mid", "late"]].reset_index()
    colors = [BLUE, TEAL, GOLD, RED]
    ax.bar(np.arange(len(coverage)), coverage.empirical_coverage, color=colors)
    ax.axhline(0.9, color=BLACK, linestyle="--", linewidth=1, label="nominal 90%")
    ax.set_xticks(np.arange(len(coverage)), coverage.stratum.str.title()); ax.set_ylim(0, 1.03); ax.set_ylabel("Empirical coverage")
    ax.set_title("Development-calibrated residual interval"); ax.legend(fontsize=7); panel(ax, "a")

    ax = axes[0, 1]
    errors = np.sort(ext.abs_error.to_numpy())
    ecdf = np.arange(1, len(errors) + 1) / len(errors)
    ax.step(errors, ecdf, where="post", color=BLUE, linewidth=1.8)
    ax.axvline(tail.abs_error_q90.iloc[0], color=GOLD, linestyle="--", label="q90")
    ax.axvline(tail.abs_error_q95.iloc[0], color=RED, linestyle="--", label="q95")
    ax.set_xlabel("Absolute error (cycles)"); ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title("External error tail"); ax.legend(fontsize=7); panel(ax, "b")

    ax = axes[1, 0]
    worst = ext.head(10).sort_values("abs_error")
    ax.barh(np.arange(len(worst)), worst.abs_error, color=[RED if x > 200 else GOLD if x > 100 else BLUE2 for x in worst.abs_error])
    ax.set_yticks(np.arange(len(worst)), worst.cell_id); ax.set_xlabel("Absolute error (cycles)")
    ax.set_title("Ten most difficult secondary cells"); panel(ax, "c")

    ax = axes[1, 1]
    ax.set_axis_off()
    cards = [
        (0.04, 0.66, "Fit", f"{seeds.fit_seconds.mean():.2f} s", "30-seed mean"),
        (0.04, 0.35, "Inference", f"{seeds.inference_ms_per_cell.mean():.2f} ms/cell", "40-cell batch"),
        (0.04, 0.04, "Model", f"{seeds.model_bytes.mean()/1e6:.2f} MB", "serialized estimator"),
    ]
    for x, y, title, value, note in cards:
        patch = FancyBboxPatch((x, y), 0.90, 0.23, boxstyle="round,pad=0.012", facecolor=LIGHT, edgecolor=BLUE, linewidth=0.9, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x + 0.05, y + 0.15, title, transform=ax.transAxes, fontweight="bold", color=BLUE)
        ax.text(x + 0.35, y + 0.15, value, transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.text(x + 0.35, y + 0.06, note, transform=ax.transAxes, color=GREY, fontsize=7)
    ax.set_title("Lightweight point model and calibration priorities")
    panel(ax, "d")
    save_source(interval, "fig7_interval_coverage_source.csv"); save_source(ext.head(10), "fig7_worst_cells_source.csv"); save_source(tail, "fig7_tail_source.csv")
    save_source(pd.DataFrame([{"fit_seconds_mean": seeds.fit_seconds.mean(), "inference_ms_per_cell_mean": seeds.inference_ms_per_cell.mean(), "model_mb_mean": seeds.model_bytes.mean()/1e6, "seeds": len(seeds)}]), "fig7_complexity_source.csv")
    export(fig, "fig7_uncertainty_tail_and_complexity")


def qa_report() -> None:
    from PIL import Image

    rows = []
    stems = [
        "fig1_protocol_nested_workflow",
        "fig2_strict_nested_selector_comparison",
        "fig3_observation_budget_and_data_efficiency",
        "fig4_local_region_contribution_and_controls",
        "fig5_frozen_secondary_batch_performance",
        "fig6_feature_group_and_model_family_ablation",
        "fig7_uncertainty_tail_and_complexity",
    ]
    for stem in stems:
        png = FIG_DIR / f"{stem}.png"
        image = Image.open(png)
        svg = FIG_DIR / f"{stem}.svg"
        pdf = FIG_DIR / f"{stem}.pdf"
        svg_text = svg.read_text(encoding="utf-8") if svg.exists() else ""
        rows.append(
            {
                "figure": stem,
                "png_width_px": image.width,
                "png_height_px": image.height,
                "png_dpi_x": image.info.get("dpi", (np.nan, np.nan))[0],
                "svg_exists": svg.exists(),
                "pdf_exists": pdf.exists(),
                "svg_text_nodes": svg_text.count("<text"),
                "source_files": len(list(SOURCE_DIR.glob(f"fig{stem[3]}_*.csv"))),
            }
        )
    table = pd.DataFrame(rows)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(QA_DIR / "figure_export_qa.csv", index=False)
    passed = bool(len(table) == 7 and table.svg_exists.all() and table.pdf_exists.all() and (table.svg_text_nodes > 0).all() and (table.png_width_px >= 3000).all())
    text = [
        "# Figure QA report",
        "",
        f"Overall export check: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "- Backend: Python/matplotlib only.",
        "- Primary editable export: SVG with text nodes preserved.",
        "- Secondary exports: PDF and 600-dpi PNG.",
        "- Quantitative source data: CSV files in `figure_source_data/`.",
        "- Image-integrity note: plots are regenerated from numeric artifacts; no photographic panels or local image adjustments are present.",
        "- Statistical note: folds, seeds, cell bootstrap intervals, and the protocol-frozen secondary-batch status are stated in captions/manuscript text.",
        "",
        table.to_markdown(index=False),
    ]
    (QA_DIR / "figure_qa_report.md").write_text("\n".join(text), encoding="utf-8")
    if not passed:
        raise RuntimeError("Figure export QA failed; inspect figure_export_qa.csv")


def main() -> None:
    required = [
        N50 / "development_summary.csv", N100 / "development_summary.csv", N150 / "external_per_cell.csv",
        ABLATION / "fold_metrics.csv", REPRO / "extratrees_hpo_final_predictions.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required result files are missing:\n" + "\n".join(missing))
    FIG_DIR.mkdir(parents=True, exist_ok=True); SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    figure1_protocol(); figure2_selectors(); figure3_budget_learning(); figure4_regions()
    figure5_external(); figure6_ablation_models(); figure7_uncertainty_tail_complexity(); qa_report()
    print(f"Created 7 figure bundles in {FIG_DIR}")


if __name__ == "__main__":
    main()
