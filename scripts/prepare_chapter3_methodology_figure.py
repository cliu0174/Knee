from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "manuscript_ionics" / "figures"
STEM = "fig1_methodology_framework"

BLUE = "#3F78B5"
BLUE_LIGHT = "#E7F0F8"
TEAL = "#4B9298"
TEAL_LIGHT = "#E5F1F1"
VIOLET = "#7B61A8"
VIOLET_LIGHT = "#ECE7F4"
GOLD = "#C98A2E"
GOLD_LIGHT = "#F7E9DF"
GREEN = "#6A9A63"
GREEN_LIGHT = "#E9F1E4"
RED = "#B54A4A"
RED_LIGHT = "#F7E5E3"
INK = "#30363B"
MUTED = "#73777B"
PANEL_EDGE = "#CBD2D8"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 7.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def box(ax, x, y, w, h, text, *, face="white", edge=INK, size=6.6,
        weight="normal", color=INK, linewidth=0.9, radius=0.018, zorder=3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, fontweight=weight, color=color, linespacing=1.05,
            zorder=zorder + 1)
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=INK, linewidth=1.0, dashed=False,
          connectionstyle="arc3,rad=0"):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
        linewidth=linewidth, color=color,
        linestyle=(0, (3, 2)) if dashed else "solid",
        connectionstyle=connectionstyle, shrinkA=2, shrinkB=2, zorder=4,
    )
    ax.add_patch(patch)
    return patch


def panel_frame(ax, letter: str, title: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.03), 0.98, 0.90,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor="white", edgecolor=PANEL_EDGE, linewidth=0.8, zorder=0,
    ))
    ax.text(0.015, 0.985, letter, ha="left", va="top", fontsize=9,
            fontweight="bold", color=INK)
    ax.text(0.055, 0.977, title, ha="left", va="top", fontsize=8.4,
            fontweight="bold", color=INK)


def draw_panel_a(ax) -> None:
    panel_frame(ax, "a", "Early observations and offline target")
    box(ax, 0.045, 0.27, 0.25, 0.47,
        "EARLY RAW / SOH\ncycles 1–150\n\n22 descriptors",
        face=BLUE_LIGHT, edge=BLUE, weight="bold", color=BLUE)
    box(ax, 0.375, 0.27, 0.25, 0.47,
        "$Q(V)$ DIFFERENCE\ncycle 10 vs 150\n\n8 global + 365 local",
        face=TEAL_LIGHT, edge=TEAL, weight="bold", color=TEAL)
    box(ax, 0.705, 0.27, 0.25, 0.47,
        "FULL-LIFE SOH\noffline construction\n\n$y_{\\mathrm{knee}}$ label only",
        face=RED_LIGHT, edge=RED, weight="bold", color=RED)
    ax.text(0.17, 0.15, "predictor inputs", ha="center", va="center",
            fontsize=6.5, color=MUTED)
    ax.text(0.50, 0.15, "predictor inputs", ha="center", va="center",
            fontsize=6.5, color=MUTED)
    ax.text(0.83, 0.15, "never used as an early predictor", ha="center", va="center",
            fontsize=6.5, color=RED)


def draw_panel_b(ax) -> None:
    panel_frame(ax, "b", "Training-only selection and model freeze")

    # Fold-local feature selection stream.
    box(ax, 0.035, 0.55, 0.13, 0.22, "365 local\n$\\Delta Q$ candidates",
        face=TEAL_LIGHT, edge=TEAL, size=6.2, weight="bold")
    box(ax, 0.205, 0.55, 0.12, 0.22, "Median\nimputation",
        face=BLUE_LIGHT, edge=BLUE, size=6.2, weight="bold")
    box(ax, 0.365, 0.55, 0.15, 0.22, "Bootstrap rank\n+ region cap",
        face=VIOLET_LIGHT, edge=VIOLET, size=6.2, weight="bold")
    box(ax, 0.555, 0.55, 0.12, 0.22, "40 local\n$\\Delta Q$",
        face=TEAL_LIGHT, edge=TEAL, size=6.2, weight="bold")
    for x1, x2 in [(0.165, 0.205), (0.325, 0.365), (0.515, 0.555)]:
        arrow(ax, x1, 0.66, x2, 0.66)

    # Fixed feature stream and joint model input.
    box(ax, 0.035, 0.18, 0.13, 0.19, "22 raw / SOH",
        face=BLUE_LIGHT, edge=BLUE, size=6.2, weight="bold")
    box(ax, 0.205, 0.18, 0.12, 0.19, "8 global\n$\\Delta Q$",
        face=GOLD_LIGHT, edge=GOLD, size=6.2, weight="bold")
    box(ax, 0.365, 0.18, 0.15, 0.19, "30 fixed",
        face="white", edge=BLUE, size=6.3, weight="bold", color=BLUE)
    arrow(ax, 0.165, 0.275, 0.205, 0.275)
    ax.text(0.185, 0.305, "+", ha="center", va="center", fontsize=8,
            fontweight="bold", color=MUTED)
    arrow(ax, 0.325, 0.275, 0.365, 0.275)

    box(ax, 0.715, 0.40, 0.09, 0.27, "70-D\ninput",
        face="white", edge=BLUE, size=7.0, weight="bold", color=BLUE)
    arrow(ax, 0.677, 0.66, 0.715, 0.57, color=TEAL)
    arrow(ax, 0.515, 0.275, 0.715, 0.49, color=BLUE,
          connectionstyle="arc3,rad=-0.08")
    box(ax, 0.835, 0.40, 0.13, 0.27, "ExtraTrees\ncandidates\n\nnested CV",
        face=GREEN_LIGHT, edge=GREEN, size=6.3, weight="bold", color=GREEN)
    arrow(ax, 0.805, 0.535, 0.835, 0.535, color=GREEN)
    ax.text(0.90, 0.24, "freeze selector, model settings,\nfeature hash and seed list",
            ha="center", va="center", fontsize=6.2, color=MUTED)
    ax.text(0.50, 0.87, "outer 5-fold ×2  |  inner 3-fold  |  training cohort only",
            ha="center", va="center", fontsize=6.5, fontweight="bold", color=VIOLET)


def draw_panel_c(ax) -> None:
    panel_frame(ax, "c", "Protocol-frozen fitting and held-out evaluation")
    box(ax, 0.035, 0.39, 0.17, 0.32, "FROZEN\nCONFIGURATION\n\n40 local + 30 fixed",
        face=GREEN_LIGHT, edge=GREEN, size=6.2, weight="bold", color=GREEN)
    box(ax, 0.245, 0.49, 0.11, 0.20, "41 training\ncells",
        face=BLUE_LIGHT, edge=BLUE, size=6.2, weight="bold")
    box(ax, 0.245, 0.22, 0.11, 0.20, "30 preset\nseeds",
        face=GOLD_LIGHT, edge=GOLD, size=6.2, weight="bold")
    box(ax, 0.405, 0.37, 0.12, 0.30, "FIT ×30\nExtraTrees",
        face="white", edge=GREEN, size=6.7, weight="bold", color=GREEN)
    arrow(ax, 0.205, 0.55, 0.405, 0.55, color=GREEN)
    arrow(ax, 0.356, 0.59, 0.405, 0.57, color=BLUE)
    arrow(ax, 0.356, 0.32, 0.405, 0.47, color=GOLD)

    ax.plot([0.565, 0.565], [0.13, 0.79], color=RED, linewidth=1.0,
            linestyle=(0, (3, 2)), zorder=2)
    ax.text(0.565, 0.84, "protocol boundary", ha="center", va="center",
            fontsize=6.2, fontweight="bold", color=RED)
    arrow(ax, 0.527, 0.52, 0.605, 0.64, color=GREEN)
    arrow(ax, 0.527, 0.52, 0.605, 0.31, color=GREEN)

    box(ax, 0.605, 0.55, 0.14, 0.22, "PRIMARY TEST\n42 cells | same batch",
        face=BLUE_LIGHT, edge=BLUE, size=6.2, weight="bold", color=BLUE)
    box(ax, 0.605, 0.18, 0.14, 0.22, "SECONDARY TEST\n40 cells | cross batch",
        face=GOLD_LIGHT, edge=GOLD, size=6.2, weight="bold", color=GOLD)
    box(ax, 0.805, 0.33, 0.16, 0.30, "HELD-OUT\nPREDICTIONS\n\n$\\hat{y}_{\\mathrm{knee}}$",
        face="white", edge=GREEN, size=6.5, weight="bold", color=GREEN)
    arrow(ax, 0.745, 0.66, 0.805, 0.55, color=BLUE)
    arrow(ax, 0.745, 0.29, 0.805, 0.41, color=GOLD)
    ax.text(0.68, 0.10, "no test-cohort feedback: impute | score | select | tune | fit",
            ha="center", va="center", fontsize=6.2, fontweight="bold", color=RED)


def build() -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.1))
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.025, top=0.99, hspace=0.10)
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])
    return fig


def main() -> None:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig = build()
    save_kwargs = {"facecolor": "white", "edgecolor": "none"}
    fig.savefig(FIGURES / f"{STEM}.svg", **save_kwargs)
    fig.savefig(FIGURES / f"{STEM}.pdf", **save_kwargs)
    fig.savefig(FIGURES / f"{STEM}.png", dpi=600, **save_kwargs)
    plt.close(fig)
    print(f"Wrote {STEM}.svg/.pdf/.png to {FIGURES}")


if __name__ == "__main__":
    main()
