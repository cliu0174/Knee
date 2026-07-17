from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript_ionics" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.axis("off")
    steps = [
        "Early cycles\n(1-150)",
        "Knee labels\nand SOH curves",
        "Raw/DeltaQ\nfeatures",
        "Local DeltaQ\ntop40 descriptors",
        "ExtraTrees\nreadout",
        "External-batch\nknee prediction",
    ]
    x = np.linspace(0.08, 0.92, len(steps))
    colors = ["#d8ead3", "#e7e1f6", "#ddeaf7", "#f8e3ca", "#d9eef0", "#ead7d7"]
    for i, (xi, label) in enumerate(zip(x, steps)):
        ax.text(
            xi,
            0.55,
            label,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=colors[i], ec="#555555", lw=0.8),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x[i + 1] - 0.065, 0.55),
                xytext=(xi + 0.065, 0.55),
                arrowprops=dict(arrowstyle="->", lw=1.1, color="#444444"),
            )
    ax.text(
        0.5,
        0.13,
        "Development cohort: official train + primary test; final validation: secondary test only",
        ha="center",
        va="center",
        color="#333333",
    )
    save(fig, "fig1_pipeline")


def fig_prediction_scatter() -> None:
    pred = pd.read_csv(ROOT / "results/extratrees_ldq_hpo_N150_quick30/extratrees_hpo_final_predictions.csv")
    ext = pred[pred["official_split"] == "secondary_test"].copy()
    mean_pred = (
        ext.groupby(["cell_id", "batch", "official_split", "observed_knee_cycle"], as_index=False)
        .agg(predicted_knee_cycle=("predicted_knee_cycle", "mean"), pred_std=("predicted_knee_cycle", "std"))
    )
    y = mean_pred["observed_knee_cycle"].to_numpy()
    yp = mean_pred["predicted_knee_cycle"].to_numpy()
    lim_min = min(y.min(), yp.min()) - 80
    lim_max = max(y.max(), yp.max()) + 80

    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    ax.errorbar(
        y,
        yp,
        yerr=mean_pred["pred_std"].fillna(0),
        fmt="o",
        ms=4.2,
        color="#276c8e",
        ecolor="#9bbbd0",
        elinewidth=0.8,
        capsize=1.5,
        alpha=0.9,
    )
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#555555", lw=1)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Observed knee cycle")
    ax.set_ylabel("Predicted knee cycle")
    ax.set_title("External secondary batch")
    ax.text(
        0.05,
        0.95,
        "MAE = 77.23 cycles\nMAPE = 9.04%\nR2 = 0.794",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.9),
    )
    save(fig, "fig2_external_prediction")


def fig_model_comparison() -> None:
    ranking = pd.read_csv(ROOT / "results/horizontal_model_comparison_N150_fast/horizontal_external_ranking.csv")
    keep = [
        "ExtraTrees",
        "XGBoost",
        "RandomForest",
        "Adapted CNN-XGBoost",
        "Adapted TCN-ECA",
        "Ridge",
        "SVR-RBF",
    ]
    df = ranking[ranking["method"].isin(keep)].copy()
    df["label"] = df["method"] + "\n(" + df["input_type"].str.replace("_", " ") + ")"
    df = df.sort_values("mae", ascending=True).head(10)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    colors = ["#2f6f8f" if "local_deltaq" in x else "#b87c4c" for x in df["input_type"]]
    ax.barh(np.arange(len(df)), df["mae"], color=colors, alpha=0.9)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("External MAE (cycles)")
    ax.set_title("Model comparison on the external secondary batch")
    for i, v in enumerate(df["mae"]):
        ax.text(v + 3, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, max(df["mae"]) * 1.18)
    save(fig, "fig3_model_comparison")


def fig_topk_ablation() -> None:
    metrics = pd.read_csv(ROOT / "results/ldq_et_topk_ablation_N150/ldq_et_topk_metrics_by_seed.csv")
    ext = metrics[metrics["split"] == "external_secondary"].copy()
    agg = ext.groupby("topk").agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
    )
    x = agg.index.to_numpy()

    fig, ax1 = plt.subplots(figsize=(4.4, 3.2))
    ax1.errorbar(x, agg["mae_mean"], yerr=agg["mae_std"], marker="o", color="#2f6f8f", lw=1.6, capsize=3)
    ax1.set_xlabel("Number of selected Local DeltaQ descriptors")
    ax1.set_ylabel("MAE (cycles)", color="#2f6f8f")
    ax1.tick_params(axis="y", labelcolor="#2f6f8f")
    ax1.set_xticks(x)
    ax1.axvline(40, color="#777777", ls="--", lw=0.9)
    ax1.text(40, agg["mae_mean"].max() + 0.4, "main", ha="center", va="bottom", fontsize=8)

    ax2 = ax1.twinx()
    ax2.errorbar(x, agg["r2_mean"], yerr=agg["r2_std"], marker="s", color="#b45d4a", lw=1.4, capsize=3)
    ax2.set_ylabel("R2", color="#b45d4a")
    ax2.tick_params(axis="y", labelcolor="#b45d4a")
    ax2.spines["top"].set_visible(False)
    ax2.set_title("Top-k Local DeltaQ descriptor ablation")
    save(fig, "fig4_topk_ablation")


def fig_capacity_knee_examples() -> None:
    cycles = pd.read_csv(ROOT / "data/processed_cycles_soh_corrected_20170630.csv")
    labels = pd.read_csv(ROOT / "data/early_features_N150_local_deltaq_top40_20170630.csv")[[
        "cell_id", "batch", "official_split", "knee_cycle", "cycle_life"
    ]]
    examples = ["b1c1", "b2c20", "b3c4"]
    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "pdf.fonttype": 42,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
        }
    ):
        fig, ax = plt.subplots(figsize=(8.2, 3.9))
        colors = ["#2f6f8f", "#b45d4a", "#5e8c61"]
        max_cycle = 0
        for cell, color in zip(examples, colors):
            cell_curve = cycles[(cycles["cell_id"] == cell) & cycles["is_positive_capacity"].astype(bool)].copy()
            cell_curve = cell_curve.sort_values("cycle")
            meta = labels[labels["cell_id"] == cell].iloc[0]
            x = cell_curve["cycle"].to_numpy()
            y = cell_curve["soh_mono"].fillna(cell_curve["soh_raw"]).to_numpy()
            max_cycle = max(max_cycle, int(np.nanmax(x)))
            split_name = str(meta["official_split"]).replace("_", " ")
            ax.plot(x, y, lw=2.1, color=color, label=f"{cell} ({split_name})", zorder=2)
            ax.axvline(meta["knee_cycle"], color=color, ls="--", lw=1.6, alpha=0.85, zorder=1.5)
        ax.axvspan(
            0,
            150,
            facecolor=(0.85, 0.78, 0.91, 0.24),
            edgecolor="#7651a8",
            linewidth=2.0,
            linestyle="--",
            label="early observation window",
            zorder=0,
        )
        ax.set_xlim(0, 2500)
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.set_ylim(0.75, 1.035)
        ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
        ax.set_xlabel("Cycle", fontweight="bold")
        ax.set_ylabel("SOH", fontweight="bold")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.set_axisbelow(True)
        ax.grid(
            True,
            which="major",
            color="#aaaaaa",
            linestyle="--",
            linewidth=0.8,
            alpha=0.42,
        )
        ax.tick_params(axis="both", which="major", width=1.2, length=5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        legend = ax.legend(
            loc="upper right",
            ncol=1,
            frameon=True,
            fancybox=False,
            framealpha=0.94,
            facecolor="white",
            edgecolor="#aaaaaa",
            fontsize=12,
            handlelength=2.0,
            handletextpad=0.6,
            borderpad=0.5,
        )
        legend.get_frame().set_linewidth(1.0)
        save(fig, "fig5_knee_examples")


def fig_protocol_comparison() -> None:
    ext = pd.read_csv(ROOT / "results/extratrees_ldq_hpo_N150_quick30/extratrees_hpo_final_metrics_by_seed.csv")
    ext = ext[ext["split"] == "external_secondary"].copy()
    unstrat = pd.read_csv(ROOT / "results/random_split_local_deltaq_N150_unstratified/random_split_metrics.csv")
    strat = pd.read_csv(ROOT / "results/random_split_local_deltaq_N150_batch_stratified/random_split_metrics.csv")

    data = [ext["r2"].to_numpy(), unstrat["r2"].to_numpy(), strat["r2"].to_numpy()]
    labels = ["External\nsecondary", "Random\nsplit", "Batch-stratified\nrandom split"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], ["#2f6f8f", "#b87c4c", "#8a7bb8"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.2)
    ax.set_ylabel("R2")
    ax.set_title("Influence of validation protocol")
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    save(fig, "fig6_protocol_comparison")
def main() -> None:
    style()
    fig_pipeline()
    fig_prediction_scatter()
    fig_model_comparison()
    fig_topk_ablation()
    fig_capacity_knee_examples()
    fig_protocol_comparison()
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()


