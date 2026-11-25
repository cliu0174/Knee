from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "figure_source_data"
PANEL_DIR = ROOT / "figures" / "panels"

CURVE_SOURCE = SOURCE_DIR / "fig_offline_knee_label_construction_curve_source.csv"
STEM = "fig_knee_label_definition_audit"

BLUE = "#3F78B5"
VIOLET = "#7B61A8"
GREY = "#73777B"
LIGHT_GREY = "#D4D7DA"
BLACK = "#24272A"
RED = "#B54A4A"

MIN_FRAC = 0.2
MAX_FRAC = 0.9
MIN_SEGMENT_POINTS = 30

ANNOTATION_COLOR = RED
ANNOTATION_FONTSIZE = 7.5
ANNOTATION_WEIGHT = "semibold"
ANNOTATION_ARROW_LW = 0.8
KNEE_LINE_STYLE = (0, (4, 2.5))
KNEE_LINE_WIDTH = 0.95
KNEE_MARKER_SIZE = 25


# Match the typography and export contract used for Figure 1.
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


def new_panel(
    label: str,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
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


def normalized(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    return (values - low) / (high - low)


def kneedle_geometry(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | int | float]:
    x_norm = normalized(x)
    y_norm = normalized(y)
    n = len(x)
    low = max(MIN_SEGMENT_POINTS, int(np.floor(n * MIN_FRAC)))
    high = min(n - MIN_SEGMENT_POINTS, int(np.ceil(n * MAX_FRAC)))

    a = np.array([x_norm[0], y_norm[0]], dtype=float)
    b = np.array([x_norm[-1], y_norm[-1]], dtype=float)
    v = b - a
    denom = float(np.hypot(v[0], v[1]))
    distance = np.abs(
        v[1] * x_norm - v[0] * y_norm + b[0] * a[1] - b[1] * a[0]
    ) / denom
    candidates = np.arange(low, high)
    split = int(candidates[np.argmax(distance[candidates])])
    point = np.array([x_norm[split], y_norm[split]], dtype=float)
    projection = a + (np.dot(point - a, v) / np.dot(v, v)) * v
    return {
        "x_norm": x_norm,
        "y_norm": y_norm,
        "a": a,
        "b": b,
        "distance": distance,
        "split": split,
        "point": point,
        "projection": projection,
        "low": low,
        "high": high,
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values.astype(float), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def rate_threshold_details(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    n = len(x)
    low = max(MIN_SEGMENT_POINTS, int(np.floor(n * MIN_FRAC)))
    high = min(n - MIN_SEGMENT_POINTS, int(np.ceil(n * MAX_FRAC)))
    raw_rate = -np.gradient(y, x)
    window = max(9, int(round(0.035 * n)))
    if window % 2 == 0:
        window += 1
    rate = rolling_mean(raw_rate, window)
    candidates = np.arange(low, high)
    early = rate[MIN_SEGMENT_POINTS:low] if low > MIN_SEGMENT_POINTS else rate[:low]
    if early.size == 0:
        early = rate[: max(1, low)]
    baseline = float(np.nanmedian(early))
    high_rate = float(np.nanpercentile(rate[candidates], 90))
    threshold = baseline + 0.40 * max(0.0, high_rate - baseline)
    hits = candidates[rate[candidates] >= threshold]
    split = int(hits[0]) if hits.size else int(candidates[np.nanargmax(rate[candidates])])
    return {
        "rate": rate,
        "window": window,
        "threshold": threshold,
        "split": split,
        "low": low,
        "high": high,
    }


def main() -> None:
    curves = pd.read_csv(CURVE_SOURCE).sort_values("cycle")
    x = curves["cycle"].to_numpy(dtype=float)
    corrected = curves["soh_mono"].to_numpy(dtype=float)

    geometry = kneedle_geometry(x, corrected)
    kneedle_split = int(geometry["split"])
    distance = np.asarray(geometry["distance"], dtype=float)

    # (c) Candidate-cycle objective for the maximum-distance construction.
    fig, ax = new_panel("c", "")
    ax.plot(x, distance, color=BLUE, lw=1.45)
    ax.axvline(
        x[kneedle_split],
        color=ANNOTATION_COLOR,
        ls=KNEE_LINE_STYLE,
        lw=KNEE_LINE_WIDTH,
    )
    ax.scatter(
        [x[kneedle_split]],
        [distance[kneedle_split]],
        color=ANNOTATION_COLOR,
        s=KNEE_MARKER_SIZE,
        zorder=5,
    )
    ax.annotate(
        f"maximum at {x[kneedle_split]:.0f}",
        xy=(x[kneedle_split], distance[kneedle_split]),
        xytext=(-100, -28),
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
    ax.set_xlabel("Candidate cycle")
    ax.set_ylabel("Chord distance")
    add_major_grid(ax)
    finish_axis(ax)
    export_panel(fig, "c")

    # (d) Geometry of the deterministic maximum-distance construction.
    p = np.asarray(geometry["point"])
    q = np.asarray(geometry["projection"])
    fig, ax = new_panel("d", "")
    ax.plot(geometry["x_norm"], geometry["y_norm"], color=BLUE, lw=1.45)
    ax.plot(
        [geometry["a"][0], geometry["b"][0]],
        [geometry["a"][1], geometry["b"][1]],
        color=GREY,
        ls=(0, (4, 2.5)),
        lw=0.95,
    )
    ax.plot([p[0], q[0]], [p[1], q[1]], color=VIOLET, ls=(0, (1.5, 2.0)), lw=1.25)
    ax.axvline(
        p[0],
        color=ANNOTATION_COLOR,
        ls=KNEE_LINE_STYLE,
        lw=KNEE_LINE_WIDTH,
    )
    ax.scatter(
        [p[0]],
        [p[1]],
        s=KNEE_MARKER_SIZE,
        color=ANNOTATION_COLOR,
        zorder=5,
    )
    ax.annotate(
        # MathText does not inherit the outer Text fontweight.  Explicit
        # bold math keeps the knee label consistent with panels a, c, and e.
        rf"$\mathbf{{y}}_{{\mathbf{{knee}}}}=\mathbf{{{x[kneedle_split]:.0f}}}$",
        xy=(p[0], p[1]),
        xytext=(0.76, 0.94),
        textcoords="data",
        arrowprops={
            "arrowstyle": "-|>",
            "color": ANNOTATION_COLOR,
            "lw": ANNOTATION_ARROW_LW,
        },
        color=ANNOTATION_COLOR,
        fontsize=ANNOTATION_FONTSIZE,
        fontweight=ANNOTATION_WEIGHT,
        ha="center",
        va="bottom",
    )
    ax.set_xlabel("Normalized cycle")
    ax.set_ylabel("Normalized SOH")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    add_major_grid(ax)
    finish_axis(ax)
    export_panel(fig, "d")

    # (e) The first admissible threshold crossing defines the rate-based label.
    rate_details = rate_threshold_details(x, corrected)
    rate = 1e3 * np.asarray(rate_details["rate"], dtype=float)
    threshold = 1e3 * float(rate_details["threshold"])
    rate_split = int(rate_details["split"])
    low = int(rate_details["low"])
    high = int(rate_details["high"])

    fig, ax = new_panel("e", "")
    ax.plot(x, rate, color=BLUE, lw=1.25, label="Smoothed degradation rate")
    ax.axhline(
        threshold,
        color=VIOLET,
        ls=(0, (4, 2.5)),
        lw=1.0,
        label="Rate threshold",
    )
    ax.axvspan(x[low], x[high - 1], color=GREY, alpha=0.08, zorder=0)
    ax.axvline(
        x[rate_split],
        color=ANNOTATION_COLOR,
        ls=KNEE_LINE_STYLE,
        lw=KNEE_LINE_WIDTH,
    )
    ax.scatter(
        [x[rate_split]],
        [rate[rate_split]],
        color=ANNOTATION_COLOR,
        s=KNEE_MARKER_SIZE,
        zorder=5,
    )
    ax.annotate(
        f"first crossing at {x[rate_split]:.0f}",
        xy=(x[rate_split], rate[rate_split]),
        xytext=(-82, 28),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "-|>",
            "color": ANNOTATION_COLOR,
            "lw": ANNOTATION_ARROW_LW,
        },
        color=ANNOTATION_COLOR,
        fontsize=ANNOTATION_FONTSIZE,
        fontweight=ANNOTATION_WEIGHT,
        ha="center",
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel(r"Degradation rate ($10^{-3}$ cycle$^{-1}$)")
    ax.legend(loc="upper left", borderaxespad=0.25)
    add_major_grid(ax)
    finish_axis(ax)
    export_panel(fig, "e")

    # (f) Map the threshold crossing back to the corrected SOH trajectory.
    fig, ax = new_panel("f", "")
    ax.plot(x, corrected, color=BLUE, lw=1.35, label="Corrected SOH")
    ax.axvline(
        x[rate_split],
        color=ANNOTATION_COLOR,
        ls=KNEE_LINE_STYLE,
        lw=KNEE_LINE_WIDTH,
    )
    ax.scatter(
        [x[rate_split]],
        [corrected[rate_split]],
        color=ANNOTATION_COLOR,
        s=KNEE_MARKER_SIZE,
        zorder=5,
    )
    ax.annotate(
        rf"$\mathbf{{y}}_{{\mathbf{{knee}}}}=\mathbf{{{x[rate_split]:.0f}}}$",
        xy=(x[rate_split], corrected[rate_split]),
        xytext=(-72, -34),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "-|>",
            "color": ANNOTATION_COLOR,
            "lw": ANNOTATION_ARROW_LW,
        },
        color=ANNOTATION_COLOR,
        fontsize=ANNOTATION_FONTSIZE,
        fontweight=ANNOTATION_WEIGHT,
        ha="right",
        va="top",
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH")
    ax.set_xlim(0, float(np.nanmax(x)))
    ax.legend(loc="lower left", borderaxespad=0.25)
    add_major_grid(ax)
    finish_axis(ax)
    export_panel(fig, "f")


if __name__ == "__main__":
    main()
