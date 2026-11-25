"""Create revised Figures 2 and 7 from the reproducible continuous-label audit."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "reviewer_knee_definition_audit"
MANUSCRIPT = ROOT / "manuscript_ionics"
OVERLEAF = ROOT / "overleaf_package_current"
OUT = MANUSCRIPT / "figures" / "continuous_knee_revision"
SOURCE = MANUSCRIPT / "figure_source_data" / "continuous_knee_revision"

INK = "#1f2937"
BLUE = "#2f6f9f"
ORANGE = "#d97706"
RED = "#b4534b"
PURPLE = "#735fa4"
GREY = "#6b7280"
GRID = "#d9dee5"
GREEN = "#30745f"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.1,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(length=3, width=0.7, color=INK, labelcolor=INK)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=10,
            va="top", color=INK)


def broken_stick(x: np.ndarray, y: np.ndarray, knee: float) -> tuple[np.ndarray, float, float]:
    design = np.column_stack([np.ones_like(x), x, np.maximum(0.0, x - knee)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return design @ beta, float(beta[1]), float(beta[1] + beta[2])


def save_figure(fig: plt.Figure, stem: str, manuscript_name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        fig.savefig(OUT / f"{stem}{suffix}", facecolor="white", bbox_inches="tight", **kwargs)
    target = MANUSCRIPT / manuscript_name
    fig.savefig(target, facecolor="white", dpi=600, bbox_inches="tight")
    shutil.copy2(target, OVERLEAF / manuscript_name)
    plt.close(fig)


def figure2() -> None:
    curves = pd.read_csv(ROOT / "data" / "processed_cycles_soh_corrected_20170630.csv")
    audit = pd.read_csv(AUDIT / "continuous_knee_per_cell.csv").set_index("cell_id")
    cases = [
        ("b1c26", "Typical transition"),
        ("b2c1", "Weakest accepted acceleration"),
        ("b1c2", "Widest breakpoint interval"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.35), constrained_layout=True)
    axes = axes.ravel()
    source_rows = []
    for letter, (ax, (cell, title)) in enumerate(zip(axes[:3], cases)):
        part = curves.loc[(curves.cell_id == cell) & curves.soh_mono.notna()].sort_values("cycle")
        x = part.cycle.to_numpy(float)
        y = part.soh_mono.to_numpy(float)
        row = audit.loc[cell]
        knee = float(row.continuous_constrained_cycle)
        fit, pre, post = broken_stick(x, y, knee)
        lo, hi = float(row.bootstrap_knee_ci_low), float(row.bootstrap_knee_ci_high)
        ax.plot(x, y, color="#b8c0ca", lw=0.75, label="Corrected SOH", zorder=1)
        ax.plot(x, fit, color=BLUE, lw=1.65, label="Continuous broken-stick fit", zorder=2)
        ax.axvspan(lo, hi, color=ORANGE, alpha=0.16, label="95% bootstrap interval", zorder=0)
        ax.axvline(knee, color=RED, lw=1.15, ls=(0, (3, 2)), label="Selected breakpoint")
        ax.scatter([knee], [np.interp(knee, x, fit)], s=24, color=RED, zorder=3)
        ratio = abs(post) / abs(pre)
        ax.text(0.03, 0.05,
                f"k = {knee:.0f} cycles\n|post/pre| = {ratio:.2f}\nΔBIC = {row.delta_bic:.0f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#ccd3db", "alpha": 0.94})
        ax.set_title(title, loc="left", pad=5)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Corrected SOH")
        ax.set_ylim(min(y.min(), fit.min()) - 0.015, 1.02)
        style(ax)
        panel_label(ax, chr(ord("a") + letter))
        for xv, yv in zip(x, y):
            source_rows.append({"cell_id": cell, "cycle": xv, "soh_mono": yv, "fit": np.interp(xv, x, fit),
                                "knee_cycle": knee, "ci_low": lo, "ci_high": hi})
    axes[0].legend(loc="upper right", frameon=False, fontsize=6.5)

    ax = axes[3]
    ratios = audit.abs_slope_ratio.to_numpy(float)
    delta = audit.delta_bic.to_numpy(float)
    ax.scatter(ratios, delta, s=18, color=BLUE, alpha=0.78, edgecolor="white", linewidth=0.35, zorder=3)
    ax.axvline(1.5, color=ORANGE, lw=1.2, ls=(0, (3, 2)), label="Minimum slope ratio")
    ax.axhline(10, color=ORANGE, lw=1.2, ls=(0, (3, 2)), label="Minimum ΔBIC")
    for cell, marker, label in [("b1c26", "o", "typical"), ("b2c1", "s", "weakest"), ("b1c2", "^", "widest CI")]:
        r = audit.loc[cell]
        ax.scatter(r.abs_slope_ratio, r.delta_bic, s=45, marker=marker, color=RED, edgecolor="white", linewidth=0.6,
                   label=label, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("Absolute post/pre slope ratio")
    ax.set_ylabel("ΔBIC versus a single line")
    ax.set_title("Evidence across all 124 cells", loc="left", pad=5)
    ax.text(0.04, 0.94, "124/124 pass both criteria\n(minimum ratio = 1.88; minimum ΔBIC = 417)",
            transform=ax.transAxes, va="top", fontsize=7.0, color=GREEN, weight="bold",
            bbox={"boxstyle": "round,pad=0.24", "fc": "white", "ec": "none", "alpha": 0.90})
    ax.legend(loc="lower right", frameon=False, fontsize=6.2)
    style(ax)
    panel_label(ax, "d")
    pd.DataFrame(source_rows).to_csv(SOURCE / "figure2_curve_source.csv", index=False)
    audit.reset_index().to_csv(SOURCE / "figure2_evidence_source.csv", index=False)
    save_figure(fig, "figure2_continuous_knee_evidence", "Figure2.png")


def mae_bootstrap(predictions: pd.DataFrame, replicates: int = 3000) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260921)
    averaged = predictions.groupby(["label_method", "split", "cell_id"], as_index=False)["abs_error"].mean()
    for (method, split), part in averaged.groupby(["label_method", "split"], sort=False):
        values = part.abs_error.to_numpy(float)
        sample = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
        rows.append({"label_method": method, "split": split, "mae": values.mean(),
                     "ci_low": np.quantile(sample, 0.025), "ci_high": np.quantile(sample, 0.975), "n": len(values)})
    return pd.DataFrame(rows)


def figure7() -> None:
    interval = pd.read_csv(AUDIT / "second_round_continuous" / "e6_interval_summary.csv")
    predictions = pd.read_csv(AUDIT / "label_robustness_all_definitions" / "label_robustness_predictions.csv")
    source = mae_bootstrap(predictions)
    method_names = {
        "continuous_broken_stick": "Continuous\nbroken-stick",
        "existing_independent": "Independent\ntwo-line",
        "piecewise": "Alternative\npiecewise",
        "kneedle": "Chord\ndistance",
        "rate_threshold": "Rate\nthreshold",
    }
    order = list(method_names)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True, gridspec_kw={"width_ratios": [0.85, 1.45]})
    ax = axes[0]
    z = 1.96
    names = ["Primary", "Secondary"]
    coverage = interval.empirical_coverage.to_numpy(float)
    ns = interval.n.to_numpy(float)
    wilson_center = (coverage + z*z/(2*ns)) / (1 + z*z/ns)
    wilson_half = z*np.sqrt(coverage*(1-coverage)/ns + z*z/(4*ns*ns)) / (1 + z*z/ns)
    colors = [BLUE, PURPLE]
    bars = ax.bar(np.arange(2), coverage, color=colors, width=0.58, zorder=3)
    ax.errorbar(np.arange(2), coverage, yerr=wilson_half, fmt="none", ecolor=INK, capsize=3, lw=0.85, zorder=4)
    ax.axhline(0.90, color=ORANGE, lw=1.2, ls=(0, (3, 2)), label="Nominal 90%")
    ax.set_xticks(np.arange(2), names)
    ax.set_ylim(0.58, 1.08)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Prediction-interval calibration", loc="left", pad=5)
    ax.legend(loc="lower left", frameon=False, fontsize=6.8)
    style(ax)
    panel_label(ax, "a")

    ax = axes[1]
    x = np.arange(len(order))
    width = 0.34
    for offset, split, color, label in [(-width/2, "primary_test", BLUE, "Primary"), (width/2, "secondary_test", PURPLE, "Secondary")]:
        part = source.set_index(["label_method", "split"]).loc[[(m, split) for m in order]].reset_index()
        err = np.vstack([part.mae - part.ci_low, part.ci_high - part.mae])
        ax.bar(x+offset, part.mae, width, color=color, label=label, zorder=3)
        ax.errorbar(x+offset, part.mae, yerr=err, fmt="none", ecolor=INK, capsize=2.5, lw=0.75, zorder=4)
    ax.set_xticks(x, [method_names[m] for m in order])
    ax.set_ylabel("MAE (cycles)")
    ax.set_title("Sensitivity to operational label definition", loc="left", pad=5)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    style(ax)
    panel_label(ax, "b")
    interval.to_csv(SOURCE / "figure7_interval_source.csv", index=False)
    source.to_csv(SOURCE / "figure7_label_robustness_bootstrap_source.csv", index=False)
    save_figure(fig, "figure7_reliability_and_sensitivity", "Figure7.png")


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    figure2()
    figure7()


if __name__ == "__main__":
    main()
