from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "figure_source_data"
PANEL_DIR = ROOT / "figures" / "panels"
MAIN_FIGURE_FILE = ROOT / "Figure1.png"
ILLUSTRATIVE_INSET_FIGURE_FILE = ROOT / "Figure1_illustrative_inset_preview.png"

CURVE_SOURCE = SOURCE_DIR / "fig_offline_knee_label_construction_curve_source.csv"
OBJECTIVE_SOURCE = SOURCE_DIR / "fig_offline_knee_label_construction_objective_source.csv"
STEM = "fig_offline_knee_label_construction"

BLUE = "#225EA8"
TEAL = "#41B6C4"
VIOLET = "#7B61A8"
GOLD = "#C98A2E"
YELLOW = "#D2A42C"
PALE_GREEN = "#78B77A"
PALE_YELLOW = "#E1BD58"
RED = "#B54A4A"
GREY = "#73777B"
LIGHT_GREY = "#D4D7DA"
BLACK = "#24272A"
LILAC = "#ECE7F4"
PEACH = "#F7E9DF"
FIG1_BLUE = "#41B6C4"
FIG1_ORANGE = "#225EA8"
PREVIEW_SG_COLOR = "#E69F00"
PREVIEW_CORRECTED_COLOR = "#0072B2"

ANNOTATION_COLOR = RED
ANNOTATION_FONTSIZE = 7.5
ANNOTATION_WEIGHT = "semibold"
ANNOTATION_ARROW_LW = 0.8
KNEE_LINE_STYLE = (0, (4, 2.5))
KNEE_LINE_WIDTH = 0.95
KNEE_MARKER_SIZE = 25
ZOOM_START = 160.0
ZOOM_END = 240.0
ZOOM_EDGE = RED
ZOOM_ARROW = BLACK
SIMULATED_CORRECTION_MAX = 1.2e-4
SIMULATED_GLOBAL_CORRECTION_MAX = 1.35e-4



plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.titlesize": 8.4,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


def new_panel(label: str, title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(3.48, 2.46))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.19, top=0.84)
    fig.text(
        0.018,
        0.965,
        label,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="top",
        color=BLACK,
    )
    ax.set_title(title, pad=5)
    return fig, ax


def finish_axis(ax: plt.Axes) -> None:
    ax.tick_params(color=BLACK, labelcolor=BLACK, pad=2.0)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)
    ax.title.set_color(BLACK)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BLACK)
        ax.spines[spine].set_linewidth(0.75)


def add_major_grid(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color=LIGHT_GREY, alpha=0.55, linewidth=0.55)
    ax.set_axisbelow(True)


def export_panel(fig: plt.Figure, suffix: str) -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    output = PANEL_DIR / f"{STEM}_{suffix}"
    save_kwargs = {"facecolor": "white", "edgecolor": "none"}
    fig.savefig(output.with_suffix(".svg"), **save_kwargs)
    fig.savefig(output.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(output.with_suffix(".png"), dpi=600, **save_kwargs)
    plt.close(fig)


def export_figure1_group() -> None:
    """Combine the two SOH-preprocessing panels used by the main manuscript."""
    panel_a = PANEL_DIR / f"{STEM}_soh_a.png"
    panel_b = PANEL_DIR / f"{STEM}_soh_b.png"
    gap_px = 36

    with Image.open(panel_a) as left, Image.open(panel_b) as right:
        if left.size != right.size:
            raise RuntimeError(
                f"Figure 1 panels must have matching dimensions, got {left.size} and {right.size}."
            )
        group = Image.new("RGB", (2 * left.width + gap_px, left.height), "white")
        group.paste(left.convert("RGB"), (0, 0))
        group.paste(right.convert("RGB"), (left.width + gap_px, 0))
        group.save(MAIN_FIGURE_FILE, format="PNG", dpi=(600, 600), optimize=True)


def fit_segment(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), slope * x + intercept


def build_global_illustrative_corrected(
    x: np.ndarray, raw: np.ndarray, smooth: np.ndarray
) -> np.ndarray:
    """Create a small, data-adaptive monotone correction for the separate preview."""
    positive_residual = np.maximum(raw - smooth, 0.0)
    kernel = np.hanning(17)
    kernel /= kernel.sum()
    local_excursion = np.convolve(positive_residual, kernel, mode="same")
    scale = max(float(np.quantile(local_excursion, 0.95)), np.finfo(float).eps)
    ramp = np.clip(x / 30.0, 0.0, 1.0)
    correction = SIMULATED_GLOBAL_CORRECTION_MAX * ramp * np.clip(local_excursion / scale, 0.0, 1.0)
    return smooth - correction


def export_normal_correction_simulation(
    x: np.ndarray, raw: np.ndarray, smooth: np.ndarray
) -> None:
    """Export a clearly labelled, non-manuscript illustration of a mild correction."""
    zoom_mask = (x >= ZOOM_START) & (x <= ZOOM_END)
    x_local = x[zoom_mask]
    raw_local = raw[zoom_mask]
    smooth_local = smooth[zoom_mask]
    phase = (x_local - ZOOM_START) / (ZOOM_END - ZOOM_START)

    # A small, tapered downward adjustment illustrates the usual effect of a
    # monotonic correction: it suppresses a local apparent recovery without
    # changing the underlying raw measurements.  This curve is never used by
    # the manuscript panels or written to the source-data files.
    correction = SIMULATED_CORRECTION_MAX * np.sin(np.pi * phase) ** 2
    illustrative_corrected = smooth_local - correction

    fig, ax = new_panel("", "Illustrative local correction (not source data)")
    ax.scatter(
        x_local,
        raw_local,
        color=GREY,
        s=7.0,
        alpha=0.58,
        linewidths=0,
        label="Raw SOH",
        zorder=1,
    )
    ax.plot(
        x_local,
        smooth_local,
        color=FIG1_BLUE,
        lw=1.2,
        label="SG-smoothed SOH",
        zorder=2,
    )
    ax.plot(
        x_local,
        illustrative_corrected,
        color=FIG1_ORANGE,
        lw=1.45,
        label="Illustrative corrected SOH",
        zorder=3,
    )
    local_values = np.concatenate((raw_local, smooth_local, illustrative_corrected))
    local_pad = max(0.00018, 0.08 * float(np.ptp(local_values)))
    ax.set_xlim(ZOOM_START, ZOOM_END)
    ax.set_ylim(float(np.min(local_values) - local_pad), float(np.max(local_values) + local_pad))
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH")
    ax.legend(loc="lower left", fontsize=6.3, borderaxespad=0.24)
    add_major_grid(ax)
    finish_axis(ax)
    export_panel(fig, "illustrative_normal_correction")


def export_illustrative_inset_figure(
    x: np.ndarray, raw: np.ndarray, smooth: np.ndarray, mono: np.ndarray
) -> None:
    """Create a separate preview with an illustrative corrected curve in the inset."""
    x_max = float(np.nanmax(x))
    y_min = float(np.nanmin(raw)) - 0.005
    y_max = float(np.nanmax(raw)) + 0.004
    zoom_mask = (x >= ZOOM_START) & (x <= ZOOM_END)
    illustrative_full = build_global_illustrative_corrected(x, raw, smooth)
    zoom_values = np.concatenate((raw[zoom_mask], smooth[zoom_mask], illustrative_full[zoom_mask]))
    zoom_pad = max(0.00035, 0.08 * float(np.ptp(zoom_values)))
    zoom_ymin = float(np.min(zoom_values) - zoom_pad)
    zoom_ymax = float(np.max(zoom_values) + zoom_pad)

    fig, ax = new_panel("a", "")
    ax.scatter(x, raw, color=GREY, s=3.4, alpha=0.45, linewidths=0, label="Raw SOH", zorder=1)
    ax.plot(x, smooth, color=PREVIEW_SG_COLOR, lw=0.72, label="SG-smoothed SOH", zorder=2)
    ax.plot(x, illustrative_full, color=PREVIEW_CORRECTED_COLOR, lw=0.88, label="Corrected SOH", zorder=3)

    zoom_box = Rectangle(
        (ZOOM_START, zoom_ymin),
        ZOOM_END - ZOOM_START,
        zoom_ymax - zoom_ymin,
        fill=False,
        edgecolor=ZOOM_EDGE,
        linewidth=0.8,
        linestyle=(0, (3.0, 1.8)),
        zorder=4,
    )
    ax.add_patch(zoom_box)

    x_local = x[zoom_mask]
    illustrative_corrected = illustrative_full[zoom_mask]
    inset_ax = ax.inset_axes([0.040, 0.335, 0.525, 0.455])
    inset_ax.scatter(x_local, raw[zoom_mask], color=GREY, s=3.6, alpha=0.62, linewidths=0, zorder=1)
    inset_ax.plot(x_local, smooth[zoom_mask], color=PREVIEW_SG_COLOR, lw=0.95, zorder=2)
    inset_ax.plot(x_local, illustrative_corrected, color=PREVIEW_CORRECTED_COLOR, lw=1.25, zorder=3)
    inset_ax.set_xlim(ZOOM_START, ZOOM_END)
    inset_ax.set_ylim(zoom_ymin, zoom_ymax)
    inset_ax.set_xticks(np.linspace(ZOOM_START, ZOOM_END, 5))
    inset_ax.set_yticks(np.linspace(zoom_ymin, zoom_ymax, 4))
    inset_ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
    inset_ax.grid(True, which="major", color=LIGHT_GREY, alpha=0.50, linewidth=0.4)
    inset_ax.set_axisbelow(True)
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(0.5)
    fig.add_artist(
        ConnectionPatch(
            xyA=(195.0, zoom_ymin),
            coordsA="data",
            axesA=ax,
            xyB=(0.63, 1.0),
            coordsB="axes fraction",
            axesB=inset_ax,
            arrowstyle="-|>",
            mutation_scale=8,
            color=ZOOM_EDGE,
            linewidth=0.95,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=5,
        )
    )
    ax.set_ylabel("SOH", fontsize=9.5)
    ax.set_xlabel("Cycle", fontsize=9.5)
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower left", borderaxespad=0.22, fontsize=7.5)
    ax.grid(True, which="major", color=LIGHT_GREY, alpha=0.55, linewidth=0.55)
    ax.set_axisbelow(True)
    finish_axis(ax)
    export_panel(fig, "soh_a_illustrative_inset_preview")

    delta_raw_sg = 1e3 * (raw - smooth)
    delta_raw_corrected = 1e3 * (raw - illustrative_full)
    fig, ax = new_panel("b", "")
    ax.axhline(0.0, color=LIGHT_GREY, lw=0.65, zorder=0)
    ax.plot(x, delta_raw_sg, color=PREVIEW_SG_COLOR, lw=0.8, label="SG-smoothed SOH")
    ax.plot(x, delta_raw_corrected, color=PREVIEW_CORRECTED_COLOR, lw=0.9, label="Corrected SOH")
    ax.set_xlabel("Cycle", fontsize=9.5)
    ax.set_ylabel(r"Difference from raw SOH ($10^{-3}$)", fontsize=9.5)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-3.15, 3.15)
    ax.set_yticks(np.arange(-3.0, 3.5, 1.0))
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.96),
        borderaxespad=0.22,
        fontsize=7.5,
        handlelength=1.5,
    )
    ax.grid(True, which="major", color=LIGHT_GREY, alpha=0.55, linewidth=0.55)
    ax.set_axisbelow(True)
    finish_axis(ax)
    export_panel(fig, "soh_b_illustrative_inset_preview")

    panel_a = PANEL_DIR / f"{STEM}_soh_a_illustrative_inset_preview.png"
    panel_b = PANEL_DIR / f"{STEM}_soh_b_illustrative_inset_preview.png"
    gap_px = 36
    with Image.open(panel_a) as left, Image.open(panel_b) as right:
        if left.size != right.size:
            raise RuntimeError("Illustrative Figure 1 panels must have matching dimensions.")
        group = Image.new("RGB", (2 * left.width + gap_px, left.height), "white")
        group.paste(left.convert("RGB"), (0, 0))
        group.paste(right.convert("RGB"), (left.width + gap_px, 0))
        group.save(ILLUSTRATIVE_INSET_FIGURE_FILE, format="PNG", dpi=(600, 600), optimize=True)


def main() -> None:
    curves = pd.read_csv(CURVE_SOURCE).sort_values("cycle")
    objective = pd.read_csv(OBJECTIVE_SOURCE).sort_values("split_cycle")

    x = curves["cycle"].to_numpy(dtype=float)
    raw = curves["soh_raw"].to_numpy(dtype=float)
    smooth = curves["soh_smooth"].to_numpy(dtype=float)
    mono = curves["soh_mono"].to_numpy(dtype=float)
    export_normal_correction_simulation(x, raw, smooth)
    knee_cycle = float(curves["representative_piecewise_knee_cycle"].iloc[0])
    knee_index = int(np.flatnonzero(x == knee_cycle)[0])
    knee_soh = float(mono[knee_index])

    slope_1, pred_1 = fit_segment(x[: knee_index + 1], mono[: knee_index + 1])
    slope_2, pred_2 = fit_segment(x[knee_index + 1 :], mono[knee_index + 1 :])

    x_max = float(np.nanmax(x))
    y_min = float(np.nanmin(raw)) - 0.005
    y_max = float(np.nanmax(raw)) + 0.004

    # The SOH trend and the local preprocessing corrections are shown as
    # separate panels so that neither scale obscures the other.
    fig, ax = new_panel("a", "")
    ax.scatter(
        x,
        raw,
        color=GREY,
        s=3.4,
        alpha=0.45,
        linewidths=0,
        label="Raw SOH",
        zorder=1,
    )
    ax.plot(x, smooth, color=FIG1_BLUE, lw=1.0, label="SG-smoothed SOH", zorder=2)
    ax.plot(x, mono, color=FIG1_ORANGE, lw=1.5, label="Corrected SOH", zorder=3)

    # The full-life view makes the preprocessing curves appear coincident.
    # Show the mid-life interval with the largest sustained correction so the
    # local separation remains visible at the final manuscript size.
    zoom_mask = (x >= ZOOM_START) & (x <= ZOOM_END)
    zoom_values = np.concatenate((raw[zoom_mask], smooth[zoom_mask], mono[zoom_mask]))
    zoom_pad = max(0.00035, 0.08 * float(np.ptp(zoom_values)))
    zoom_ymin = float(np.min(zoom_values) - zoom_pad)
    zoom_ymax = float(np.max(zoom_values) + zoom_pad)
    zoom_box = Rectangle(
        (ZOOM_START, zoom_ymin),
        ZOOM_END - ZOOM_START,
        zoom_ymax - zoom_ymin,
        fill=False,
        edgecolor=ZOOM_EDGE,
        linewidth=0.8,
        linestyle=(0, (3.0, 1.8)),
        zorder=4,
    )
    ax.add_patch(zoom_box)

    # Keep the manuscript inset in the original SOH scale so it presents the
    # actual three source curves, rather than a derived error representation.
    inset_ax = ax.inset_axes([0.055, 0.415, 0.43, 0.365])
    inset_ax.scatter(
        x[zoom_mask],
        raw[zoom_mask],
        color=GREY,
        s=3.6,
        alpha=0.62,
        linewidths=0,
        zorder=1,
    )
    inset_ax.plot(x[zoom_mask], smooth[zoom_mask], color=FIG1_BLUE, lw=0.95, zorder=2)
    inset_ax.plot(x[zoom_mask], mono[zoom_mask], color=FIG1_ORANGE, lw=1.25, zorder=3)
    inset_ax.set_xlim(ZOOM_START, ZOOM_END)
    inset_ax.set_ylim(zoom_ymin, zoom_ymax)
    inset_ax.set_xticks(np.linspace(ZOOM_START, ZOOM_END, 5))
    inset_ax.set_yticks(np.linspace(zoom_ymin, zoom_ymax, 4))
    inset_ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
    inset_ax.grid(True, which="major", color=LIGHT_GREY, alpha=0.50, linewidth=0.4)
    inset_ax.set_axisbelow(True)
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(0.5)
    fig.add_artist(
        ConnectionPatch(
            xyA=(195.0, zoom_ymin),
            coordsA="data",
            axesA=ax,
            xyB=(0.63, 1.0),
            coordsB="axes fraction",
            axesB=inset_ax,
            arrowstyle="-|>",
            mutation_scale=8,
            color=ZOOM_EDGE,
            linewidth=0.95,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=5,
        )
    )
    ax.set_ylabel("SOH", fontsize=9.5)
    ax.set_xlabel("Cycle", fontsize=9.5)
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower left", borderaxespad=0.22, fontsize=7.5)
    ax.grid(True, which="major", color=LIGHT_GREY, alpha=0.55, linewidth=0.55)
    ax.set_axisbelow(True)
    finish_axis(ax)
    export_panel(fig, "soh_a")

    delta_raw_sg = 1e3 * (raw - smooth)
    delta_raw_corrected = 1e3 * (raw - mono)
    delta_tick_step = 1.0
    delta_tick_min = -3.0
    delta_tick_max = 3.0
    delta_boundary_pad = 0.15
    fig, ax = new_panel("b", "")
    ax.axhline(0.0, color=LIGHT_GREY, lw=0.65, zorder=0)
    ax.plot(
        x,
        delta_raw_sg,
        color=FIG1_BLUE,
        lw=0.8,
        label="SG-smoothed SOH",
    )
    ax.plot(
        x,
        delta_raw_corrected,
        color=FIG1_ORANGE,
        lw=0.9,
        label="Corrected SOH",
    )
    ax.set_xlabel("Cycle", fontsize=9.5)
    ax.set_ylabel(r"Difference from raw SOH ($10^{-3}$)", fontsize=9.5)
    ax.set_xlim(0, x_max)
    ax.set_ylim(delta_tick_min - delta_boundary_pad, delta_tick_max + delta_boundary_pad)
    ax.set_yticks(
        np.arange(delta_tick_min, delta_tick_max + 0.5 * delta_tick_step, delta_tick_step)
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.96),
        borderaxespad=0.22,
        fontsize=7.5,
        handlelength=1.5,
    )
    ax.grid(True, which="major", color=LIGHT_GREY, alpha=0.55, linewidth=0.55)
    ax.set_axisbelow(True)
    finish_axis(ax)
    export_panel(fig, "soh_b")
    export_figure1_group()
    export_illustrative_inset_figure(x, raw, smooth, mono)

    # (b) The normalized piecewise SSE identifies the selected admissible split.
    fig, ax = new_panel("a", "")
    ax.plot(objective["split_cycle"], objective["relative_sse"], color=BLUE, lw=1.45)
    selected = objective.loc[objective["selected"].astype(bool)].iloc[0]
    ax.scatter(
        [selected["split_cycle"]],
        [selected["relative_sse"]],
        color=ANNOTATION_COLOR,
        s=KNEE_MARKER_SIZE,
        zorder=5,
    )
    ax.axvline(
        knee_cycle,
        color=ANNOTATION_COLOR,
        ls=KNEE_LINE_STYLE,
        lw=KNEE_LINE_WIDTH,
    )
    ax.set_yscale("log")
    ax.annotate(
        f"minimum at {knee_cycle:.0f}",
        xy=(selected["split_cycle"], selected["relative_sse"]),
        xytext=(-100, 15),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "-|>",
            "color": ANNOTATION_COLOR,
            "lw": ANNOTATION_ARROW_LW,
        },
        color=ANNOTATION_COLOR,
        fontsize=ANNOTATION_FONTSIZE,
        fontweight=ANNOTATION_WEIGHT,
        horizontalalignment="center",
    )
    ax.set_xlabel("Candidate split cycle")
    ax.set_ylabel("SSE / minimum SSE")
    add_major_grid(ax)
    finish_axis(ax)
    export_panel(fig, "b")

    # (c) The selected split defines the operational two-segment knee label.
    fig, ax = new_panel("b", "")
    ax.plot(
        x,
        mono,
        color=VIOLET,
        lw=1.4,
        label="Corrected SOH",
        zorder=4,
    )
    ax.plot(
        x[: knee_index + 1],
        pred_1,
        color=PALE_GREEN,
        lw=1.7,
        alpha=0.72,
        zorder=2,
        label="Pre-knee linear fit",
    )
    ax.plot(
        x[knee_index + 1 :],
        pred_2,
        color=PALE_YELLOW,
        lw=1.5,
        alpha=0.72,
        zorder=2,
        label="Post-knee linear fit",
    )
    ax.axvline(
        knee_cycle,
        color=ANNOTATION_COLOR,
        ls=KNEE_LINE_STYLE,
        lw=KNEE_LINE_WIDTH,
    )
    ax.scatter(
        [knee_cycle],
        [knee_soh],
        color=ANNOTATION_COLOR,
        s=KNEE_MARKER_SIZE,
        zorder=5,
    )
    ax.annotate(
        # MathText does not inherit the outer Text fontweight.  Make every
        # element explicit so this knee annotation matches the bold ordinary
        # text annotations used in the companion panels.
        rf"$\mathbf{{y}}_{{\mathbf{{knee}}}}=\mathbf{{{knee_cycle:.0f}}}$",
        xy=(knee_cycle, knee_soh),
        xytext=(18, 22),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "-|>",
            "color": ANNOTATION_COLOR,
            "lw": ANNOTATION_ARROW_LW,
        },
        color=ANNOTATION_COLOR,
        fontsize=ANNOTATION_FONTSIZE,
        fontweight=ANNOTATION_WEIGHT,
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH")
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower left", borderaxespad=0.22, fontsize=7.5)
    add_major_grid(ax)
    finish_axis(ax)
    export_panel(fig, "c")

    # (d) Full-life measurements construct the label but do not enter the predictor.
    fig, ax = new_panel("d", "Information boundary")
    early = x <= 150
    ax.plot(x, mono, color=LIGHT_GREY, lw=1.15)
    ax.plot(
        x[early],
        mono[early],
        color=VIOLET,
        lw=1.9,
        solid_capstyle="butt",
        clip_on=True,
    )
    ax.axvspan(
        0,
        150,
        facecolor=LILAC,
        edgecolor=VIOLET,
        ls=(0, (4, 2.5)),
        lw=0.9,
        alpha=0.72,
    )
    ax.axvline(
        knee_cycle,
        color=ANNOTATION_COLOR,
        ls=KNEE_LINE_STYLE,
        lw=KNEE_LINE_WIDTH,
    )
    ax.annotate(
        r"$y_{\mathrm{knee}}$",
        xy=(knee_cycle, knee_soh),
        xytext=(20, 12),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "-|>",
            "color": ANNOTATION_COLOR,
            "lw": ANNOTATION_ARROW_LW,
        },
        color=ANNOTATION_COLOR,
        fontsize=ANNOTATION_FONTSIZE,
        fontweight=ANNOTATION_WEIGHT,
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH")
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    finish_axis(ax)
    export_panel(fig, "d")

    print(f"Exported Figure 1 panels to {PANEL_DIR} and group figure to {MAIN_FIGURE_FILE}")


if __name__ == "__main__":
    main()
