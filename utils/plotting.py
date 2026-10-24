"""
utils/plotting.py

Centralizes ALL Plotly visualization logic for the AI Race Engineer
dashboard.

This module produces `plotly.graph_objects.Figure` objects and nothing
else. It contains NO telemetry calculations, NO business/rule-engine
logic, and NO Streamlit calls -- the Streamlit layer's job is simply:

    fig = plot_speed(distance, driver_speed, reference_speed, "VER", "NOR")
    st.plotly_chart(fig, use_container_width=True)

Imported by: app/, analysis/ (if it ever wants a figure for debugging),
tests/.
Must NOT import: fastf1, streamlit, google-generativeai (Gemini).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from utils.helpers import setup_logger

_logger = setup_logger(__name__)

ArrayLike = Sequence[float] | np.ndarray

# --------------------------------------------------------------------------
# Theme constants -- the single source of truth for the dashboard's
# "professional dark racing" visual identity. Referenced only by
# apply_plot_theme() and the trace-color constants below, so a future
# re-theme touches this block and nothing else.
# --------------------------------------------------------------------------

_PLOT_BG_COLOR: str = "#0E1117"
_PAPER_BG_COLOR: str = "#0E1117"
_GRID_COLOR: str = "rgba(255, 255, 255, 0.08)"
_FONT_COLOR: str = "#E6E6E6"
_SPIKE_COLOR: str = "rgba(255, 255, 255, 0.35)"
_FONT_FAMILY: str = "Inter, Segoe UI, Arial, sans-serif"

#: Trace color for the primary (analyzed) driver.
DRIVER_COLOR: str = "#00D9FF"
#: Trace color for the reference/comparison driver.
REFERENCE_COLOR: str = "#FF6B35"
#: Color for the delta-time trace.
DELTA_TIME_COLOR: str = "#FFD23F"
#: Color for the zero reference line on the delta-time chart.
ZERO_LINE_COLOR: str = "#FF3366"


def apply_plot_theme(
    title: str,
    x_axis_title: str,
    y_axis_title: str,
    y_axis_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build the shared Plotly layout configuration for the dark racing dashboard theme.

    This is the single source of truth for chart-wide styling: background
    colors, grid styling, hover behavior, legend placement, and axis
    formatting. Every `plot_*` function in this module applies this same
    layout via `fig.update_layout(**apply_plot_theme(...))`, so visual
    consistency across charts is structural, not a matter of each
    function remembering to match the others.

    Args:
        title: Chart title text.
        x_axis_title: Label for the x-axis.
        y_axis_title: Label for the y-axis.
        y_axis_range: Optional (min, max) to fix the y-axis range (e.g.
            0-100 for percentage-based channels like throttle/brake).

    Returns:
        A dict of keyword arguments suitable for `go.Figure.update_layout(**layout)`.
    """
    yaxis_config: dict[str, Any] = dict(
        title=y_axis_title,
        showgrid=True,
        gridcolor=_GRID_COLOR,
        zeroline=False,
    )
    if y_axis_range is not None:
        yaxis_config["range"] = list(y_axis_range)

    return dict(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=18, color=_FONT_COLOR)),
        template="plotly_dark",
        plot_bgcolor=_PLOT_BG_COLOR,
        paper_bgcolor=_PAPER_BG_COLOR,
        font=dict(family=_FONT_FAMILY, color=_FONT_COLOR, size=12),
        xaxis=dict(
            title=x_axis_title,
            showgrid=True,
            gridcolor=_GRID_COLOR,
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor=_SPIKE_COLOR,
        ),
        yaxis=yaxis_config,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        margin=dict(l=60, r=30, t=60, b=50),
        autosize=True,
    )


def _validate_series(distance: ArrayLike, *series: ArrayLike, names: Sequence[str]) -> None:
    """Validate that distance and every data series are non-empty and equal in length.

    Private helper shared by every `plot_*` function so length/emptiness
    validation is written exactly once instead of duplicated per chart.

    Args:
        distance: The shared x-axis (distance) array.
        *series: One or more y-axis data series to validate against `distance`.
        names: Human-readable names for each array in `series`, in the
            same order, used to build an informative error message.

    Raises:
        ValueError: If `distance` is empty, if any series is empty, or
            if any series' length does not match `distance`'s length.
    """
    distance_length = len(distance)

    if distance_length == 0:
        raise ValueError("'distance' must not be empty.")

    if len(names) != len(series):
        raise ValueError(
            f"'names' must match the number of series provided "
            f"({len(names)} vs {len(series)})."
        )

    for name, values in zip(names, series):
        if len(values) != distance_length:
            raise ValueError(
                f"'{name}' length ({len(values)}) does not match "
                f"'distance' length ({distance_length})."
            )


def _build_line_trace(
    x: ArrayLike,
    y: ArrayLike,
    name: str,
    color: str,
    *,
    dash: str | None = None,
    line_shape: str = "linear",
    hover_suffix: str = "",
) -> go.Scatter:
    """Build a single styled line trace shared by all telemetry-comparison charts.

    Centralizing trace construction here means every chart's lines share
    consistent width, hover formatting, and color conventions, and a
    future style tweak (e.g. line width) is a one-place change.

    Args:
        x: X-axis values (distance).
        y: Y-axis values (the telemetry channel).
        name: Trace name shown in the legend and hover label.
        color: Line color (hex or rgba string).
        dash: Optional Plotly dash style (e.g. "dot") to visually
            distinguish a reference-driver trace from the primary driver.
        line_shape: Plotly line shape; use "hv" for step-like binary
            signals such as brake on/off.
        hover_suffix: Optional unit suffix appended to the hover value
            (e.g. " km/h", " %").

    Returns:
        A configured `go.Scatter` trace.
    """
    return go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="lines",
        line=dict(color=color, width=2, dash=dash, shape=line_shape),
        hovertemplate=f"%{{y:.1f}}{hover_suffix}<extra>{name}</extra>",
    )


def plot_speed(
    distance: ArrayLike,
    driver_speed: ArrayLike,
    reference_speed: ArrayLike,
    driver_name: str,
    reference_name: str,
) -> go.Figure:
    """Build a Speed vs Distance comparison chart for a driver and a reference driver.

    Args:
        distance: Shared distance grid (meters).
        driver_speed: Driver's speed (km/h) at each point in `distance`.
        reference_speed: Reference driver's speed (km/h) at each point in `distance`.
        driver_name: Display name/code for the primary driver (legend label).
        reference_name: Display name/code for the reference driver (legend label).

    Returns:
        A themed Plotly Figure with two speed traces sharing a distance axis.

    Raises:
        ValueError: If input arrays are empty or mismatched in length.
    """
    _validate_series(
        distance,
        driver_speed,
        reference_speed,
        names=["driver_speed", "reference_speed"],
    )
    _logger.debug("Building speed chart for %s vs %s.", driver_name, reference_name)

    fig = go.Figure()
    fig.add_trace(
        _build_line_trace(distance, driver_speed, driver_name, DRIVER_COLOR, hover_suffix=" km/h")
    )
    fig.add_trace(
        _build_line_trace(
            distance,
            reference_speed,
            reference_name,
            REFERENCE_COLOR,
            dash="dot",
            hover_suffix=" km/h",
        )
    )

    fig.update_layout(
        **apply_plot_theme(
            title="Speed vs Distance",
            x_axis_title="Distance (m)",
            y_axis_title="Speed (km/h)",
        )
    )
    return fig


def plot_throttle(
    distance: ArrayLike,
    driver_throttle: ArrayLike,
    reference_throttle: ArrayLike,
    driver_name: str = "Driver",
    reference_name: str = "Reference",
) -> go.Figure:
    """Build a Throttle Application vs Distance comparison chart.

    Args:
        distance: Shared distance grid (meters).
        driver_throttle: Driver's throttle position (%) at each point in `distance`.
        reference_throttle: Reference driver's throttle position (%) at each point in `distance`.
        driver_name: Display name/code for the primary driver (legend label).
        reference_name: Display name/code for the reference driver (legend label).

    Returns:
        A themed Plotly Figure with two throttle traces sharing a distance axis.

    Raises:
        ValueError: If input arrays are empty or mismatched in length.
    """
    _validate_series(
        distance,
        driver_throttle,
        reference_throttle,
        names=["driver_throttle", "reference_throttle"],
    )
    _logger.debug("Building throttle chart for %s vs %s.", driver_name, reference_name)

    fig = go.Figure()
    fig.add_trace(
        _build_line_trace(distance, driver_throttle, driver_name, DRIVER_COLOR, hover_suffix=" %")
    )
    fig.add_trace(
        _build_line_trace(
            distance,
            reference_throttle,
            reference_name,
            REFERENCE_COLOR,
            dash="dot",
            hover_suffix=" %",
        )
    )

    fig.update_layout(
        **apply_plot_theme(
            title="Throttle Application vs Distance",
            x_axis_title="Distance (m)",
            y_axis_title="Throttle (%)",
            y_axis_range=(0, 105),
        )
    )
    return fig


def plot_brake(
    distance: ArrayLike,
    driver_brake: ArrayLike,
    reference_brake: ArrayLike,
    driver_name: str = "Driver",
    reference_name: str = "Reference",
) -> go.Figure:
    """Build a Brake Application vs Distance comparison chart.

    Uses a step ("hv") line shape rather than linear interpolation, since
    brake signals are typically near-binary (on/off) and a step shape
    represents that behavior more accurately than a smoothed line.

    Args:
        distance: Shared distance grid (meters).
        driver_brake: Driver's brake position (%) at each point in `distance`.
        reference_brake: Reference driver's brake position (%) at each point in `distance`.
        driver_name: Display name/code for the primary driver (legend label).
        reference_name: Display name/code for the reference driver (legend label).

    Returns:
        A themed Plotly Figure with two brake traces sharing a distance axis.

    Raises:
        ValueError: If input arrays are empty or mismatched in length.
    """
    _validate_series(
        distance,
        driver_brake,
        reference_brake,
        names=["driver_brake", "reference_brake"],
    )
    _logger.debug("Building brake chart for %s vs %s.", driver_name, reference_name)

    fig = go.Figure()
    fig.add_trace(
        _build_line_trace(
            distance,
            driver_brake,
            driver_name,
            DRIVER_COLOR,
            line_shape="hv",
            hover_suffix=" %",
        )
    )
    fig.add_trace(
        _build_line_trace(
            distance,
            reference_brake,
            reference_name,
            REFERENCE_COLOR,
            dash="dot",
            line_shape="hv",
            hover_suffix=" %",
        )
    )

    fig.update_layout(
        **apply_plot_theme(
            title="Brake Application vs Distance",
            x_axis_title="Distance (m)",
            y_axis_title="Brake (%)",
            y_axis_range=(0, 105),
        )
    )
    return fig


def plot_delta_time(distance: ArrayLike, delta_time: ArrayLike) -> go.Figure:
    """Build a cumulative Time Delta vs Distance chart, with a zero reference line.

    Args:
        distance: Shared distance grid (meters).
        delta_time: Cumulative time delta (seconds) at each point in
            `distance`. Positive values indicate the driver is behind the
            reference; negative values indicate the driver is ahead.

    Returns:
        A themed Plotly Figure showing the delta-time trace and a
        horizontal zero reference line.

    Raises:
        ValueError: If input arrays are empty or mismatched in length.
    """
    _validate_series(distance, delta_time, names=["delta_time"])
    _logger.debug("Building delta-time chart.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=delta_time,
            name="Time Delta",
            mode="lines",
            line=dict(color=DELTA_TIME_COLOR, width=2),
            hovertemplate="%{y:+.3f} s<extra>Time Delta</extra>",
        )
    )

    fig.add_hline(
        y=0,
        line=dict(color=ZERO_LINE_COLOR, width=1, dash="dash"),
        annotation_text="Even",
        annotation_position="bottom right",
        annotation_font=dict(color=_FONT_COLOR, size=11),
    )

    fig.update_layout(
        **apply_plot_theme(
            title="Time Delta vs Distance",
            x_axis_title="Distance (m)",
            y_axis_title="Delta Time (s)",
        )
    )
    return fig
