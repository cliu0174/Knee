from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox


# Editable-text and publication export settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.2
plt.rcParams["axes.titlesize"] = 8.6
plt.rcParams["axes.titleweight"] = "semibold"
plt.rcParams["axes.labelsize"] = 7.8
plt.rcParams["xtick.labelsize"] = 7.0
plt.rcParams["ytick.labelsize"] = 7.0
plt.rcParams["legend.fontsize"] = 6.5
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.handlelength"] = 1.8
plt.rcParams["legend.handletextpad"] = 0.55
plt.rcParams["legend.labelspacing"] = 0.35
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["xtick.major.size"] = 2.6
plt.rcParams["ytick.major.size"] = 2.6
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "manuscript_ionics" / "figures"
PANEL_DIR = FIG_DIR / "panels"
SOURCE_DIR = ROOT / "manuscript_ionics" / "figure_source_data"
QA_DIR = ROOT / "manuscript_ionics" / "figure_qa"

CYCLES_PATH = ROOT / "data" / "processed_cycles_soh_corrected_20170630.csv"
FORMAL_LABELS_PATH = ROOT / "data" / "knee_labels_20170630.csv"

# Restrained, colorblind-aware palette shared by both figures.
BLUE = "#355F8C"
BLUE2 = "#3F78B5"
TEAL = "#4B9298"
VIOLET = "#7B61A8"
GOLD = "#C98A2E"
RED = "#B54A4A"
GREY = "#73777B"
LIGHT_GREY = "#D4D7DA"
BLACK = "#24272A"
LILAC = "#ECE7F4"
PEACH = "#F7E9DF"

DOUBLE_COLUMN_WIDTH = 183.0 / 25.4

REPRESENTATIVE_CELL = "b1c44"
MIN_FRAC = 0.2
MAX_FRAC = 0.9
MIN_SEGMENT_POINTS = 30


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.145,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=BLACK,
    )


def finish_axis(ax: plt.Axes) -> None:
    ax.tick_params(color=BLACK, labelcolor=BLACK, pad=2.2)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)
    ax.title.set_color(BLACK)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BLACK)
        ax.spines[spine].set_linewidth(0.75)


def export(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.align_labels()
    fig.canvas.draw()
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.04, "facecolor": "white"}
    fig.savefig(FIG_DIR / f"{stem}.svg", **save_kwargs)
    fig.savefig(FIG_DIR / f"{stem}.pdf", **save_kwargs)
    fig.savefig(
        FIG_DIR / f"{stem}.png",
        dpi=600,
        **save_kwargs,
    )
    plt.close(fig)


def export_panel_set(
    fig: plt.Figure,
    axes: list[plt.Axes],
    stem: str,
    *,
    square_canvas: bool = False,
) -> None:
    """Export equal-canvas standalone panels from the approved composite layout."""
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_boxes = [
        ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        for ax in axes
    ]
    panel_width = max(box.width for box in tight_boxes) + 0.08
    panel_height = max(box.height for box in tight_boxes) + 0.08
    if square_canvas:
        panel_width = panel_height = max(panel_width, panel_height)

    for label, box in zip("abcdefghijklmnopqrstuvwxyz", tight_boxes):
        center_x = 0.5 * (box.x0 + box.x1)
        center_y = 0.5 * (box.y0 + box.y1)
        crop = Bbox.from_bounds(
            center_x - 0.5 * panel_width,
            center_y - 0.5 * panel_height,
            panel_width,
            panel_height,
        )
        panel_stem = PANEL_DIR / f"{stem}_{label}"
        save_kwargs = {"bbox_inches": crop, "pad_inches": 0, "facecolor": "white"}
        fig.savefig(panel_stem.with_suffix(".svg"), **save_kwargs)
        fig.savefig(panel_stem.with_suffix(".pdf"), **save_kwargs)
        fig.savefig(panel_stem.with_suffix(".png"), dpi=600, **save_kwargs)


def save_source(frame: pd.DataFrame, name: str) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SOURCE_DIR / name, index=False)


def normalized(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def candidate_bounds(n: int) -> tuple[int, int]:
    low = max(MIN_SEGMENT_POINTS, int(np.floor(n * MIN_FRAC)))
    high = min(n - MIN_SEGMENT_POINTS, int(np.ceil(n * MAX_FRAC)))
    return low, high


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    sse = float(np.sum((y - prediction) ** 2))
    return float(slope), float(intercept), prediction, sse


def piecewise_details(cycles: np.ndarray, soh: np.ndarray) -> dict[str, object]:
    order = np.argsort(cycles)
    x = cycles[order].astype(float)
    y = soh[order].astype(float)
    low, high = candidate_bounds(len(x))
    objective_rows: list[dict[str, float]] = []
    best: dict[str, object] | None = None

    # Formal convention: the boundary observation belongs to the first segment;
    # the second segment starts at split + 1 (non-overlapping boundary).
    for split in range(low, high):
        x1, y1 = x[: split + 1], y[: split + 1]
        x2, y2 = x[split + 1 :], y[split + 1 :]
        slope1, intercept1, pred1, sse1 = fit_line(x1, y1)
        slope2, intercept2, pred2, sse2 = fit_line(x2, y2)
        total = sse1 + sse2
        objective_rows.append({"split_cycle": float(x[split]), "piecewise_sse": total})
        if best is None or total < float(best["piecewise_sse"]):
            best = {
                "split": split,
                "knee_cycle": float(x[split]),
                "knee_soh": float(y[split]),
                "piecewise_sse": total,
                "slope1": slope1,
                "intercept1": intercept1,
                "slope2": slope2,
                "intercept2": intercept2,
                "pred1": pred1,
                "pred2": pred2,
                "x": x,
                "y": y,
            }

    if best is None:
        raise RuntimeError("No admissible piecewise split")
    objective = pd.DataFrame(objective_rows)
    objective["relative_sse"] = objective["piecewise_sse"] / objective["piecewise_sse"].min()
    best["objective"] = objective
    best["candidate_low_cycle"] = float(x[low])
    best["candidate_high_cycle"] = float(x[high - 1])
    return best


def kneedle_details(cycles: np.ndarray, soh: np.ndarray) -> dict[str, object]:
    order = np.argsort(cycles)
    x = cycles[order].astype(float)
    y = soh[order].astype(float)
    x_norm = normalized(x)
    y_norm = normalized(y)
    low, high = candidate_bounds(len(x))

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
        "knee_cycle": float(x[split]),
        "knee_soh": float(y[split]),
        "split": split,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "chord_x": np.array([a[0], b[0]]),
        "chord_y": np.array([a[1], b[1]]),
        "point": point,
        "projection": projection,
        "distance": float(distance[split]),
    }


def build_label_audit(cycles: pd.DataFrame, formal_labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_id, group in cycles.groupby("cell_id", sort=False):
        part = group[group["soh_mono"].notna()].sort_values("cycle")
        kneedle = kneedle_details(
            part["cycle"].to_numpy(dtype=float),
            part["soh_mono"].to_numpy(dtype=float),
        )
        formal = formal_labels[formal_labels["cell_id"] == cell_id]
        if len(formal) != 1:
            raise RuntimeError(f"Expected one formal label for {cell_id}, found {len(formal)}")
        row = formal.iloc[0]
        piecewise_cycle = float(row["knee_cycle"])
        kneedle_cycle = float(kneedle["knee_cycle"])
        rows.append(
            {
                "cell_id": cell_id,
                "official_split": row["official_split"],
                "batch": row["batch"],
                "cycle_life": float(row["cycle_life"]),
                "piecewise_knee_cycle": piecewise_cycle,
                "kneedle_knee_cycle": kneedle_cycle,
                "signed_difference_kneedle_minus_piecewise": kneedle_cycle - piecewise_cycle,
                "absolute_difference": abs(kneedle_cycle - piecewise_cycle),
            }
        )
    audit = pd.DataFrame(rows)
    if len(audit) != 124:
        raise RuntimeError(f"Expected 124 cells for the label audit, found {len(audit)}")
    return audit


def audit_statistics(audit: pd.DataFrame) -> dict[str, float]:
    x = audit["piecewise_knee_cycle"]
    y = audit["kneedle_knee_cycle"]
    diff = audit["signed_difference_kneedle_minus_piecewise"]
    return {
        "n": int(len(audit)),
        "pearson": float(x.corr(y)),
        "spearman": float(x.rank(method="average").corr(y.rank(method="average"))),
        "mean_absolute_difference": float(np.mean(np.abs(diff))),
        "median_absolute_difference": float(np.median(np.abs(diff))),
        "mean_signed_difference": float(np.mean(diff)),
        "median_signed_difference": float(np.median(diff)),
        "minimum_signed_difference": float(np.min(diff)),
        "maximum_signed_difference": float(np.max(diff)),
        "within_50_cycles": float(np.mean(np.abs(diff) <= 50)),
        "within_100_cycles": float(np.mean(np.abs(diff) <= 100)),
    }


def representative_data(cycles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    part = cycles[
        (cycles["cell_id"] == REPRESENTATIVE_CELL) & cycles["soh_mono"].notna()
    ].sort_values("cycle")
    if part.empty:
        raise RuntimeError(f"Representative cell {REPRESENTATIVE_CELL} was not found")
    x = part["cycle"].to_numpy(dtype=float)
    y = part["soh_mono"].to_numpy(dtype=float)
    return part, piecewise_details(x, y), kneedle_details(x, y)


def figure_label_definition_audit(
    cycles: pd.DataFrame,
    formal_labels: pd.DataFrame,
    audit: pd.DataFrame,
    stats: dict[str, float],
) -> None:
    part, piecewise, kneedle = representative_data(cycles)
    formal_cycle = float(
        formal_labels.loc[formal_labels["cell_id"] == REPRESENTATIVE_CELL, "knee_cycle"].iloc[0]
    )
    if formal_cycle != float(piecewise["knee_cycle"]):
        raise RuntimeError(
            f"Formal label ({formal_cycle}) and recomputed non-overlap label "
            f"({piecewise['knee_cycle']}) disagree for {REPRESENTATIVE_CELL}"
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 5.25),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.92, 1.08]},
    )
    fig.set_constrained_layout_pads(w_pad=0.035, h_pad=0.035, wspace=0.07, hspace=0.08)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # a: formal piecewise construction on the representative cell.
    x = piecewise["x"]
    y = piecewise["y"]
    split = int(piecewise["split"])
    ax_a.plot(x, y, color=BLUE2, lw=1.45, label="Monotone SOH")
    ax_a.plot(x[: split + 1], piecewise["pred1"], color=TEAL, lw=1.45, label="Segment 1")
    ax_a.plot(x[split + 1 :], piecewise["pred2"], color=GOLD, lw=1.45, label="Segment 2")
    ax_a.axvline(formal_cycle, color=RED, ls=(0, (4, 2.5)), lw=1.0)
    ax_a.scatter([formal_cycle], [piecewise["knee_soh"]], s=25, color=RED, zorder=5)
    ax_a.annotate(
        f"piecewise = {formal_cycle:.0f}",
        xy=(formal_cycle, piecewise["knee_soh"]),
        xytext=(-55, -28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 0.7},
        color=RED,
        fontsize=6.9,
    )
    ax_a.set_title("Piecewise-linear label", pad=5)
    ax_a.set_xlabel("Cycle")
    ax_a.set_ylabel("SOH")
    ax_a.legend(loc="lower left", ncol=1, borderaxespad=0.25)
    panel(ax_a, "a")

    # b: Kneedle construction in the normalized coordinate system actually used.
    ax_b.plot(kneedle["x_norm"], kneedle["y_norm"], color=BLUE2, lw=1.45)
    ax_b.plot(kneedle["chord_x"], kneedle["chord_y"], color=GREY, ls=(0, (4, 2.5)), lw=0.95)
    p = kneedle["point"]
    q = kneedle["projection"]
    ax_b.plot([p[0], q[0]], [p[1], q[1]], color=VIOLET, ls=(0, (1.5, 2.0)), lw=1.25)
    ax_b.scatter([p[0]], [p[1]], s=25, color=VIOLET, zorder=5)
    ax_b.annotate(
        f"Kneedle = {kneedle['knee_cycle']:.0f}",
        xy=(p[0], p[1]),
        xytext=(-58, -25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-|>", "color": VIOLET, "lw": 0.7},
        color=VIOLET,
        fontsize=6.9,
    )
    ax_b.text(
        0.74,
        0.55,
        "maximum chord\ndistance",
        transform=ax_b.transAxes,
        color=VIOLET,
        fontsize=6.3,
        ha="center",
        va="center",
        linespacing=0.95,
    )
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_title("Kneedle label", pad=5)
    ax_b.set_xlabel("Normalized cycle")
    ax_b.set_ylabel("Normalized SOH")
    panel(ax_b, "b")

    # c: cohort-wide agreement with official cohort provenance.
    split_style = {
        "train": (BLUE, "o", "Official train"),
        "primary_test": (TEAL, "^", "Primary test"),
        "secondary_test": (GOLD, "s", "Secondary test"),
    }
    for split_name, (color, marker, label) in split_style.items():
        sub = audit[audit["official_split"] == split_name]
        ax_c.scatter(
            sub["piecewise_knee_cycle"],
            sub["kneedle_knee_cycle"],
            s=19,
            color=color,
            marker=marker,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.3,
            label=label,
        )
    low = float(audit[["piecewise_knee_cycle", "kneedle_knee_cycle"]].min().min())
    high = float(audit[["piecewise_knee_cycle", "kneedle_knee_cycle"]].max().max())
    margin = 0.04 * (high - low)
    ax_c.plot(
        [low - margin, high + margin],
        [low - margin, high + margin],
        color=GREY,
        ls=(0, (4, 2.5)),
        lw=0.85,
    )
    ax_c.set_xlim(low - margin, high + margin)
    ax_c.set_ylim(low - margin, high + margin)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.text(
        0.04,
        0.94,
        f"n = {int(stats['n'])}\nSpearman $\\rho$ = {stats['spearman']:.3f}",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=6.9,
    )
    ax_c.set_title("Cohort-wide agreement", pad=5)
    ax_c.set_xlabel("Piecewise knee cycle")
    ax_c.set_ylabel("Kneedle knee cycle")
    ax_c.legend(loc="lower right", borderaxespad=0.25)
    panel(ax_c, "c")

    # d: signed definition-dependent offset.
    diff = audit["signed_difference_kneedle_minus_piecewise"].to_numpy(dtype=float)
    bins = np.arange(np.floor(diff.min() / 10) * 10, np.ceil(diff.max() / 10) * 10 + 10, 10)
    ax_d.hist(diff, bins=bins, color=LILAC, edgecolor=VIOLET, linewidth=0.7)
    ax_d.axvline(0, color=GREY, ls=(0, (4, 2.5)), lw=0.85)
    ax_d.axvline(stats["mean_signed_difference"], color=RED, lw=1.15)
    ax_d.annotate(
        r"mean $\Delta y$",
        xy=(stats["mean_signed_difference"], 0.985),
        xycoords=("data", "axes fraction"),
        xytext=(3, 0),
        textcoords="offset points",
        color=RED,
        fontsize=6.2,
        ha="left",
        va="top",
    )
    ax_d.text(
        0.04,
        0.94,
        f"Mean $|\\Delta y|$ = {stats['mean_absolute_difference']:.2f} cycles\n"
        f"Median $\\Delta y$ = {stats['median_signed_difference']:.0f} cycles\n"
        f"Range = [{stats['minimum_signed_difference']:.0f}, {stats['maximum_signed_difference']:.0f}]",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=6.9,
    )
    ax_d.set_title("Definition-dependent offset", pad=5)
    ax_d.set_xlabel(r"$\Delta y$ = Kneedle - piecewise (cycles)")
    ax_d.set_ylabel("Cells")
    panel(ax_d, "d")

    for ax in axes.ravel():
        finish_axis(ax)
    export_panel_set(fig, list(axes.ravel()), "fig_knee_label_definition_audit")
    export(fig, "fig_knee_label_definition_audit")


def figure_offline_label_construction(cycles: pd.DataFrame, formal_labels: pd.DataFrame) -> None:
    part, piecewise, _ = representative_data(cycles)
    formal = formal_labels[formal_labels["cell_id"] == REPRESENTATIVE_CELL].iloc[0]
    if float(formal["knee_cycle"]) != float(piecewise["knee_cycle"]):
        raise RuntimeError("Representative formal label does not match the recomputed objective")

    source = part[
        ["cell_id", "official_split", "cycle", "soh_raw", "soh_smooth", "soh_mono"]
    ].copy()
    source["representative_piecewise_knee_cycle"] = float(piecewise["knee_cycle"])
    source["early_observation"] = source["cycle"] <= 150
    save_source(source, "fig_offline_knee_label_construction_curve_source.csv")
    objective = piecewise["objective"].copy()
    objective["cell_id"] = REPRESENTATIVE_CELL
    objective["selected"] = objective["split_cycle"] == float(piecewise["knee_cycle"])
    save_source(objective, "fig_offline_knee_label_construction_objective_source.csv")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH, 4.76),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.035, wspace=0.055, hspace=0.08)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    x = part["cycle"].to_numpy(dtype=float)
    raw = part["soh_raw"].to_numpy(dtype=float)
    smooth = part["soh_smooth"].to_numpy(dtype=float)
    mono = part["soh_mono"].to_numpy(dtype=float)
    x_max = float(np.nanmax(x))
    y_min = float(np.nanmin(raw)) - 0.005
    y_max = float(np.nanmax(raw)) + 0.004

    # a: raw SOH.
    ax_a.plot(x, raw, color=GREY, lw=0.95)
    ax_a.scatter(x[::8], raw[::8], color=GREY, s=3.2, alpha=0.62, linewidths=0)
    ax_a.set_title("Raw SOH", pad=5)
    ax_a.set_xlabel("Cycle")
    ax_a.set_ylabel("SOH")
    ax_a.set_xlim(0, x_max)
    ax_a.set_ylim(y_min, y_max)
    panel(ax_a, "a")

    # b: Savitzky-Golay smoothing.
    ax_b.plot(x, raw, color=LIGHT_GREY, lw=0.85, label="Raw")
    ax_b.plot(x, smooth, color=BLUE2, lw=1.4, label="SG smoothed")
    mean_delta = float(np.nanmean(np.abs(smooth - raw)))
    ax_b.text(
        0.04,
        0.25,
        f"window = 21, order = 2\nmean $|\\Delta|$ = {mean_delta:.1e}",
        transform=ax_b.transAxes,
        fontsize=6.5,
    )
    ax_b.set_title("Savitzky-Golay smoothing", pad=5)
    ax_b.set_xlabel("Cycle")
    ax_b.set_xlim(0, x_max)
    ax_b.set_ylim(y_min, y_max)
    ax_b.legend(loc="lower left", borderaxespad=0.25)
    panel(ax_b, "b")

    # c: monotone correction.
    ax_c.plot(x, smooth, color=LIGHT_GREY, lw=0.85, label="SG smoothed")
    ax_c.plot(x, mono, color=TEAL, lw=1.4, label="Monotone SOH")
    mean_delta_mono = float(np.nanmean(np.abs(mono - smooth)))
    ax_c.text(
        0.04,
        0.25,
        f"non-increasing constraint\nmean $|\\Delta|$ = {mean_delta_mono:.1e}",
        transform=ax_c.transAxes,
        fontsize=6.5,
    )
    ax_c.set_title("Isotonic correction", pad=5)
    ax_c.set_xlabel("Cycle")
    ax_c.set_xlim(0, x_max)
    ax_c.set_ylim(y_min, y_max)
    ax_c.legend(loc="lower left", borderaxespad=0.25)
    panel(ax_c, "c")

    # d: objective over admissible split locations.
    obj = piecewise["objective"]
    ax_d.plot(obj["split_cycle"], obj["relative_sse"], color=VIOLET, lw=1.35)
    selected = obj[obj["split_cycle"] == float(piecewise["knee_cycle"])].iloc[0]
    ax_d.scatter([selected["split_cycle"]], [selected["relative_sse"]], color=RED, s=24, zorder=5)
    ax_d.axvline(float(piecewise["knee_cycle"]), color=RED, ls=(0, (4, 2.5)), lw=0.95)
    ax_d.set_yscale("log")
    ax_d.annotate(
        f"minimum at {piecewise['knee_cycle']:.0f}",
        xy=(selected["split_cycle"], selected["relative_sse"]),
        xytext=(-52, 23),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 0.7},
        color=RED,
        fontsize=6.7,
    )
    ax_d.set_title("Admissible split objective", pad=5)
    ax_d.set_xlabel("Candidate split cycle")
    ax_d.set_ylabel("SSE / minimum SSE")
    panel(ax_d, "d")

    # e: final piecewise fit and label.
    split = int(piecewise["split"])
    ax_e.plot(x, mono, color=BLUE2, lw=1.25, label="Monotone SOH")
    ax_e.plot(x[: split + 1], piecewise["pred1"], color=TEAL, lw=1.45)
    ax_e.plot(x[split + 1 :], piecewise["pred2"], color=GOLD, lw=1.45)
    ax_e.axvline(float(piecewise["knee_cycle"]), color=RED, ls=(0, (4, 2.5)), lw=1.0)
    ax_e.scatter([piecewise["knee_cycle"]], [piecewise["knee_soh"]], color=RED, s=24, zorder=5)
    ax_e.text(
        0.04,
        0.11,
        f"$\\beta_1$ = {piecewise['slope1']:.2e} cycle$^{{-1}}$\n"
        f"$\\beta_2$ = {piecewise['slope2']:.2e} cycle$^{{-1}}$",
        transform=ax_e.transAxes,
        fontsize=6.5,
    )
    ax_e.annotate(
        f"$y_{{\\mathrm{{knee}}}}$ = {piecewise['knee_cycle']:.0f}",
        xy=(piecewise["knee_cycle"], piecewise["knee_soh"]),
        xytext=(-20, 26),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 0.7},
        color=RED,
        fontsize=6.9,
        fontweight="normal",
    )
    ax_e.set_title("Final offline label", pad=5)
    ax_e.set_xlabel("Cycle")
    ax_e.set_ylabel("SOH")
    ax_e.set_xlim(0, x_max)
    ax_e.set_ylim(y_min, y_max)
    panel(ax_e, "e")

    # f: strict separation between the early predictor input and the full-life label path.
    early = x <= 150
    ax_f.plot(x, mono, color=LIGHT_GREY, lw=1.05)
    ax_f.plot(x[early], mono[early], color=VIOLET, lw=1.8)
    ax_f.axvspan(
        0,
        150,
        facecolor=LILAC,
        edgecolor=VIOLET,
        ls=(0, (4, 2.5)),
        lw=0.85,
        alpha=0.7,
    )
    ax_f.axvline(float(piecewise["knee_cycle"]), color=RED, ls=(0, (4, 2.5)), lw=1.0)
    ax_f.text(
        0.05,
        0.12,
        "Predictor input\ncycles 1-150",
        transform=ax_f.transAxes,
        color=VIOLET,
        fontsize=6.8,
        fontweight="semibold",
        bbox={"boxstyle": "round,pad=0.24", "fc": "#F4F1F8", "ec": VIOLET, "lw": 0.7},
    )
    ax_f.text(
        0.56,
        0.71,
        "Full-life SOH\nlabel only",
        transform=ax_f.transAxes,
        color=RED,
        fontsize=6.8,
        fontweight="semibold",
        bbox={"boxstyle": "round,pad=0.24", "fc": PEACH, "ec": RED, "lw": 0.7},
    )
    ax_f.annotate(
        r"$y_{\mathrm{knee}}$",
        xy=(piecewise["knee_cycle"], piecewise["knee_soh"]),
        xytext=(-15, 26),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 0.7},
        color=RED,
        fontsize=6.9,
        fontweight="normal",
    )
    ax_f.set_title("Information boundary", pad=5)
    ax_f.set_xlabel("Cycle")
    ax_f.set_xlim(0, x_max)
    ax_f.set_ylim(y_min, y_max)
    panel(ax_f, "f")

    for ax in axes.ravel():
        finish_axis(ax)
    export_panel_set(
        fig,
        list(axes.ravel()),
        "fig_offline_knee_label_construction",
        square_canvas=True,
    )
    export(fig, "fig_offline_knee_label_construction")


def write_qa_summary(stats: dict[str, float]) -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": "Python/matplotlib",
        "representative_cell": REPRESENTATIVE_CELL,
        "piecewise_convention": "non-overlapping boundary; second segment starts at split + 1",
        "label_audit_population": "all 124 cells before y > 150 eligibility filtering",
        "label_audit_statistics": stats,
        "exports": [
            "fig_knee_label_definition_audit.svg/pdf/png",
            "fig_offline_knee_label_construction.svg/pdf/png",
        ],
        "image_integrity": "All panels are regenerated from numeric CSV artifacts; no raster image manipulation.",
    }
    (QA_DIR / "knee_label_argument_figures_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    cycles = pd.read_csv(CYCLES_PATH)
    formal_labels = pd.read_csv(FORMAL_LABELS_PATH)
    audit = build_label_audit(cycles, formal_labels)
    stats = audit_statistics(audit)

    save_source(audit, "fig_knee_label_definition_audit_source.csv")
    save_source(pd.DataFrame([stats]), "fig_knee_label_definition_audit_summary.csv")
    figure_label_definition_audit(cycles, formal_labels, audit, stats)
    figure_offline_label_construction(cycles, formal_labels)
    write_qa_summary(stats)

    print("Generated knee-label argument figures.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
