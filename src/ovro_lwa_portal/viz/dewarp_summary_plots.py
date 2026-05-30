"""Bokeh heatmaps for dewarp ``median_shift_arcmin`` vs LST and frequency."""

from __future__ import annotations

import pandas as pd
import panel as pn
from bokeh.plotting import figure

from ovro_lwa_portal.viz.flux_check_plots import (
    FLUX_RATIO_GRID_TOTAL_WIDTH,
    FLUX_RATIO_PLOT_HEIGHT,
    FLUX_RATIO_PLOT_WIDTH,
    FLUX_RATIO_TILE_WIDTH,
    build_lst_freq_heatmap_figure,
    lst_hour_label_map,
)
from ovro_lwa_portal.viz.pipeline_qa import dewarp_shift_grid, load_dewarp_summary_dataframe

DEWARP_SHIFT_PLOT_WIDTH = FLUX_RATIO_PLOT_WIDTH
DEWARP_SHIFT_PLOT_HEIGHT = FLUX_RATIO_PLOT_HEIGHT


def build_dewarp_shift_figure(
    grid: pd.DataFrame,
    *,
    lst_labels: dict[int, str] | None = None,
    width: int = DEWARP_SHIFT_PLOT_WIDTH,
    height: int = DEWARP_SHIFT_PLOT_HEIGHT,
) -> figure:
    """Build a Bokeh heatmap of dewarp median shift vs LST and frequency."""
    return build_lst_freq_heatmap_figure(
        "Dewarp median shift",
        grid,
        value_label="median shift",
        hover_value_label="median_shift_arcmin",
        lst_labels=lst_labels,
        width=width,
        height=height,
    )


def build_dewarp_shift_panel(
    dewarp_df: pd.DataFrame,
    *,
    width: int = DEWARP_SHIFT_PLOT_WIDTH,
    height: int = DEWARP_SHIFT_PLOT_HEIGHT,
) -> pn.Column:
    """Return a single dewarp-summary heatmap below the thermal-noise PNG grid."""
    grid = dewarp_shift_grid(dewarp_df)
    if grid.empty:
        return pn.Column(
            pn.pane.Markdown("*No dewarp_summary.csv data for this day.*"),
            sizing_mode="stretch_width",
        )

    labels = lst_hour_label_map(dewarp_df)
    figure = build_dewarp_shift_figure(grid, lst_labels=labels, width=width, height=height)
    return pn.Column(
        pn.pane.Bokeh(
            figure,
            sizing_mode="scale_width",
            width=FLUX_RATIO_GRID_TOTAL_WIDTH,
            height=height,
        ),
        sizing_mode="stretch_width",
        max_width=FLUX_RATIO_GRID_TOTAL_WIDTH,
    )


def load_dewarp_shift_panel(
    select_day: str,
    coverage: pd.DataFrame,
    *,
    width: int = DEWARP_SHIFT_PLOT_WIDTH,
    height: int = DEWARP_SHIFT_PLOT_HEIGHT,
) -> tuple[pd.DataFrame, pn.Column]:
    """Load dewarp summary CSVs for one day and build the heatmap panel."""
    dewarp_df = load_dewarp_summary_dataframe(select_day, coverage)
    return dewarp_df, build_dewarp_shift_panel(dewarp_df, width=width, height=height)
