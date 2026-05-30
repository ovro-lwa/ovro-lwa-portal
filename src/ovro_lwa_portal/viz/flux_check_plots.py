"""Bokeh heatmaps for hybrid flux-check ratios (imfit / model) by source."""

from __future__ import annotations

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, FixedTicker, HoverTool, LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from ovro_lwa_portal.viz.pipeline_qa import flux_ratio_grids, load_flux_check_hybrid_dataframe

FLUX_RATIO_GRID_COLS = 2
FLUX_RATIO_GRID_TOTAL_WIDTH = 1048
FLUX_RATIO_PLOT_WIDTH = 440
FLUX_RATIO_PLOT_HEIGHT = 360
FLUX_RATIO_TILE_WIDTH = FLUX_RATIO_GRID_TOTAL_WIDTH // FLUX_RATIO_GRID_COLS


def _cell_center(idx: int) -> float:
    return idx + 0.5


def _axis_ticks(n: int, labels: dict[int, str], *, max_ticks: int = 24) -> tuple[list[float], dict[float, str]]:
    if n <= 0:
        return [], {}
    step = 1 if n <= max_ticks else int(np.ceil(n / max_ticks))
    indices = range(0, n, step)
    ticks = [_cell_center(i) for i in indices]
    tick_labels = {_cell_center(i): labels.get(i, str(i)) for i in indices}
    return ticks, tick_labels


def _color_mapper_for_values(values: np.ndarray) -> tuple[LinearColorMapper, float, float]:
    """Build a linear color mapper using the 2nd–98th percentile of finite values."""
    data = values.astype(np.float64, copy=True)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return LinearColorMapper(palette=Viridis256, low=0.0, high=1.0), 0.0, 1.0
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    mapper = LinearColorMapper(
        palette=Viridis256,
        low=float(lo),
        high=float(hi),
        nan_color=(128, 128, 128, 0.4),
    )
    return mapper, float(lo), float(hi)


def _hover_source(
    value_map: np.ndarray,
    *,
    freq_mhz: list[float],
    lst_hour_nums: list[int],
    lst_labels: dict[int, str],
    value_field: str = "metric_value",
) -> ColumnDataSource:
    lst_idx, freq_idx = np.meshgrid(
        np.arange(len(lst_hour_nums), dtype=int),
        np.arange(len(freq_mhz), dtype=int),
        indexing="ij",
    )
    values = value_map[lst_idx, freq_idx]
    return ColumnDataSource(
        data={
            "x": lst_idx.ravel() + 0.5,
            "y": freq_idx.ravel() + 0.5,
            "frequency_mhz": np.asarray(freq_mhz)[freq_idx.ravel()],
            "lst_hour": [lst_labels.get(int(n), f"{int(n)}h") for n in np.asarray(lst_hour_nums)[lst_idx.ravel()]],
            "lst_hour_num": np.asarray(lst_hour_nums)[lst_idx.ravel()],
            value_field: values.ravel(),
        }
    )


def build_lst_freq_heatmap_figure(
    title: str,
    grid: pd.DataFrame,
    *,
    value_label: str,
    hover_value_label: str,
    lst_labels: dict[int, str] | None = None,
    extra_tooltips: list[tuple[str, str]] | None = None,
    width: int = FLUX_RATIO_PLOT_WIDTH,
    height: int = FLUX_RATIO_PLOT_HEIGHT,
) -> figure:
    """Build a Bokeh LST × frequency heatmap from a pivoted metric grid."""
    if grid.empty:
        plot = figure(
            width=width,
            height=height,
            title=f"{title}: no data",
            x_range=(0, 1),
            y_range=(0, 1),
            tools="pan,wheel_zoom,reset",
        )
        return plot

    freq_mhz = [float(value) for value in grid.columns]
    lst_hour_nums = [int(value) for value in grid.index]
    value_map = grid.to_numpy(dtype=float)
    n_freqs = len(freq_mhz)
    n_lsts = len(lst_hour_nums)

    labels = lst_labels or {num: f"{num:02d}h" for num in lst_hour_nums}
    color_mapper, _clim_lo, _clim_hi = _color_mapper_for_values(value_map)

    plot = figure(
        width=width,
        height=height,
        title=title,
        x_range=(0, n_lsts),
        y_range=(0, n_freqs),
        tools="pan,wheel_zoom,reset",
        active_drag="pan",
    )
    plot.image(
        image=[value_map.T],
        x=0,
        y=0,
        dw=n_lsts,
        dh=n_freqs,
        color_mapper=color_mapper,
    )
    hover_renderer = plot.rect(
        x="x",
        y="y",
        width=1,
        height=1,
        source=_hover_source(
            value_map,
            freq_mhz=freq_mhz,
            lst_hour_nums=lst_hour_nums,
            lst_labels=labels,
        ),
        fill_alpha=0,
        line_alpha=0,
        hover_fill_alpha=0,
        hover_line_alpha=0,
    )
    tooltips: list[tuple[str, str]] = [
        ("Frequency (MHz)", "@frequency_mhz{0.0}"),
        ("LST hour", "@lst_hour"),
        (hover_value_label, "@metric_value{0.3g}"),
    ]
    if extra_tooltips:
        tooltips = extra_tooltips + tooltips
    plot.add_tools(HoverTool(renderers=[hover_renderer], tooltips=tooltips))
    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(desired_num_ticks=5),
        label_standoff=8,
        border_line_color=None,
        title=value_label,
    )
    plot.add_layout(color_bar, "right")

    x_ticks, x_labels = _axis_ticks(n_lsts, labels)
    y_ticks, y_labels = _axis_ticks(
        n_freqs,
        {index: f"{freq:g}" for index, freq in enumerate(freq_mhz)},
    )
    plot.xaxis.ticker = FixedTicker(ticks=x_ticks)
    plot.yaxis.ticker = FixedTicker(ticks=y_ticks)
    plot.xaxis.major_label_overrides = x_labels
    plot.yaxis.major_label_overrides = y_labels
    plot.xaxis.axis_label = "LST hour"
    plot.yaxis.axis_label = "Frequency (MHz)"
    return plot


def build_flux_ratio_figure(
    source: str,
    grid: pd.DataFrame,
    *,
    lst_labels: dict[int, str] | None = None,
    width: int = FLUX_RATIO_PLOT_WIDTH,
    height: int = FLUX_RATIO_PLOT_HEIGHT,
) -> figure:
    """Build a Bokeh heatmap of flux ratio (imfit/model) vs LST and frequency."""
    return build_lst_freq_heatmap_figure(
        f"{source}: imfit / model flux ratio",
        grid,
        value_label="imfit/model",
        hover_value_label="imfit/model",
        lst_labels=lst_labels,
        extra_tooltips=[("Source", source)],
        width=width,
        height=height,
    )


def lst_hour_label_map(flux_df: pd.DataFrame) -> dict[int, str]:
    """Map ``lst_hour_num`` to directory-style labels such as ``08h``."""
    if flux_df.empty:
        return {}
    mapping = (
        flux_df.drop_duplicates("lst_hour_num")
        .set_index("lst_hour_num")["lst_hour"]
        .astype(str)
        .to_dict()
    )
    return {int(key): value for key, value in mapping.items()}


def build_flux_ratio_figures(
    flux_df: pd.DataFrame,
    *,
    width: int = FLUX_RATIO_PLOT_WIDTH,
    height: int = FLUX_RATIO_PLOT_HEIGHT,
) -> dict[str, figure]:
    """Return one Bokeh heatmap figure per calibrator source."""
    labels = lst_hour_label_map(flux_df)
    return {
        source: build_flux_ratio_figure(
            source,
            grid,
            lst_labels=labels,
            width=width,
            height=height,
        )
        for source, grid in flux_ratio_grids(flux_df).items()
    }


def load_flux_ratio_figures(
    select_day: str,
    coverage: pd.DataFrame,
    *,
    width: int = FLUX_RATIO_PLOT_WIDTH,
    height: int = FLUX_RATIO_PLOT_HEIGHT,
) -> tuple[pd.DataFrame, dict[str, figure]]:
    """Load flux-check CSVs for one day and build source heatmaps."""
    flux_df = load_flux_check_hybrid_dataframe(select_day, coverage)
    figures = build_flux_ratio_figures(flux_df, width=width, height=height)
    return flux_df, figures


def build_flux_ratio_panel_grid(
    figures: dict[str, figure],
    *,
    n_cols: int = FLUX_RATIO_GRID_COLS,
    plot_width: int = FLUX_RATIO_PLOT_WIDTH,
    plot_height: int = FLUX_RATIO_PLOT_HEIGHT,
) -> pn.Column:
    """Arrange flux-ratio Bokeh heatmaps in a responsive grid."""
    if not figures:
        return pn.Column(
            pn.pane.Markdown("*No flux_check_hybrid.csv data for this day.*"),
            sizing_mode="stretch_width",
        )

    tiles: list[pn.Column] = []
    for source in sorted(figures):
        tiles.append(
            pn.Column(
                pn.pane.Bokeh(
                    figures[source],
                    sizing_mode="scale_width",
                    width=FLUX_RATIO_TILE_WIDTH,
                    height=plot_height,
                ),
                width=FLUX_RATIO_TILE_WIDTH,
                sizing_mode="fixed",
                margin=(6, 6),
            )
        )

    grid_rows: list[pn.Row] = []
    for start in range(0, len(tiles), n_cols):
        grid_rows.append(
            pn.Row(*tiles[start : start + n_cols], sizing_mode="stretch_width")
        )
    return pn.Column(
        *grid_rows,
        sizing_mode="stretch_width",
        max_width=FLUX_RATIO_GRID_TOTAL_WIDTH,
    )
