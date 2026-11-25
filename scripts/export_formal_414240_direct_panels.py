"""Export manuscript Figures 4--7 and every panel directly from matplotlib artists.

This module deliberately does not read, crop, or resample any raster figure.
It invokes the formal figure-construction functions, then asks matplotlib to
render the completed figure and each panel axis independently at 600 dpi.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "scripts" / "prepare_formal_414240_figures.py"

FIGURE_NAMES = {
    "fig2_formal_selector_stability": ("Figure4", 4),
    "fig3_heldout_performance": ("Figure5", 4),
    "fig4_observation_budget": ("Figure6", 4),
    "fig7_formal_reliability_and_label_robustness": ("Figure7", 2),
}


def load_formal_module():
    spec = importlib.util.spec_from_file_location("formal_figures", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_axes(fig: plt.Figure, stem: str, destination: Path) -> None:
    """Render a complete figure plus panel axes directly from matplotlib."""
    figure_name, expected_panel_count = FIGURE_NAMES[stem]
    destination.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"facecolor": "white", "edgecolor": "none", "dpi": 600}
    fig.savefig(destination / f"{figure_name}.png", **save_kwargs)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [axis for axis in fig.axes if not hasattr(axis, "_colorbar")]
    colorbar_axes = [axis for axis in fig.axes if hasattr(axis, "_colorbar")]
    if len(axes) != expected_panel_count:
        raise RuntimeError(
            f"{stem}: expected {expected_panel_count} manuscript panels, found {len(axes)} non-colourbar axes"
        )
    for index, axis in enumerate(axes):
        # Axes.get_tightbbox includes the panel letter, labels, legend, and all
        # artists. This is a direct vector/raster render from matplotlib, not a
        # crop of the full PNG output.
        panel_bbox = axis.get_tightbbox(renderer)
        # The selector heatmap's colourbar is not a standalone manuscript
        # panel: it is the legend for Figure 4d.  Union the two matplotlib
        # artists before rendering so this remains a native export rather
        # than a raster append/crop operation.
        if stem == "fig2_formal_selector_stability" and index == 3:
            if len(colorbar_axes) != 1:
                raise RuntimeError(f"{stem}: expected one colourbar for panel d")
            panel_bbox = Bbox.union([panel_bbox, colorbar_axes[0].get_tightbbox(renderer)])
        bbox = panel_bbox.expanded(1.04, 1.08)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        suffix = chr(ord("a") + index)
        fig.savefig(destination / f"{figure_name}{suffix}.png", bbox_inches=bbox_inches, **save_kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    formal = load_formal_module()
    original_save = formal.save

    def direct_save(fig: plt.Figure, stem: str, output_dir: Path | None = None) -> None:
        if stem in FIGURE_NAMES:
            export_axes(fig, stem, args.output_dir)
        else:
            original_save(fig, stem, output_dir=output_dir)

    formal.save = direct_save
    formal.setup_style()
    # Keep regenerated source-data side products out of the manuscript's
    # canonical source-data directory.  The plotting functions are unchanged;
    # this only redirects their reproducibility CSVs to a disposable preview
    # location while the figures are being rendered.
    staging_source = ROOT / "manuscript_ionics" / "_latex_preview" / "direct_panel_source_staging"
    formal.SOURCE = staging_source
    staging_source.mkdir(parents=True, exist_ok=True)
    formal.selector_figure()
    formal.heldout_figure(source_dir=staging_source)
    formal.observation_figure()
    formal.reliability_label_figure()
    print(f"Directly rendered Figures 4--7 and all panels to {args.output_dir}")


if __name__ == "__main__":
    main()
