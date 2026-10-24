"""
analysis/corners.py

Responsible ONLY for extracting deterministic, per-corner telemetry
metrics from the aligned DataFrame produced by
analysis.alignment.load_and_align_laps().

This module must NOT:
    - classify driving mistakes (that is analysis/rules.py)
    - compute an overall driver score (that is analysis/engine.py)
    - call Gemini (that is llm/feedback.py)
    - contain Streamlit code (that is app/main.py)
    - contain machine learning

Corner boundaries are supplied as input data, not hardcoded, so this
module works identically for any circuit and any number of corners.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

from models.schemas import CornerMetric
from utils.helpers import setup_logger

_logger = setup_logger(__name__)

#: Telemetry columns that must be present in the aligned DataFrame.
_REQUIRED_TELEMETRY_COLUMNS: set[str] = {
    "Distance",
    "DriverSpeed",
    "ReferenceSpeed",
    "DriverBrake",
    "ReferenceBrake",
    "DriverThrottle",
    "ReferenceThrottle",
    "DeltaTime",
}

#: Default brake-signal threshold (%) above which braking is considered to have begun.
_DEFAULT_BRAKE_THRESHOLD_PCT: float = 10.0

#: Default throttle-signal threshold (%) above which throttle application is considered begun.
_DEFAULT_THROTTLE_THRESHOLD_PCT: float = 80.0


class CornerBoundary(TypedDict):
    """
    Shape of a single corner boundary definition supplied by the caller.

    Example:
        {"corner": 1, "start": 120, "end": 220}
    """

    corner: int
    start: float
    end: float


class CornerMetricsError(Exception):
    """Base exception for all errors raised by analysis/corners.py."""


class InvalidTelemetryError(CornerMetricsError):
    """Raised when the aligned telemetry DataFrame is missing data or malformed."""


class InvalidCornerBoundaryError(CornerMetricsError):
    """Raised when a corner boundary definition is malformed or out of range."""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate_aligned_telemetry(aligned_telemetry: pd.DataFrame) -> None:
    """Validate that the aligned telemetry DataFrame has the expected shape.

    Args:
        aligned_telemetry: The DataFrame returned by
            analysis.alignment.load_and_align_laps().

    Raises:
        InvalidTelemetryError: If the DataFrame is empty or missing any
            required column.
    """
    if aligned_telemetry.empty:
        raise InvalidTelemetryError("Aligned telemetry DataFrame is empty.")

    missing_columns = _REQUIRED_TELEMETRY_COLUMNS - set(aligned_telemetry.columns)
    if missing_columns:
        raise InvalidTelemetryError(
            f"Aligned telemetry is missing required column(s): {sorted(missing_columns)}."
        )


def _validate_corner_boundary(boundary: CornerBoundary, max_distance: float) -> None:
    """Validate a single corner boundary against the available telemetry range.

    Args:
        boundary: The corner boundary to validate.
        max_distance: The maximum distance value present in the aligned
            telemetry (i.e. the common distance grid's upper bound).

    Raises:
        InvalidCornerBoundaryError: If the boundary's start/end are out of
            order, negative, or extend beyond the available telemetry range.
    """
    start = float(boundary["start"])
    end = float(boundary["end"])

    if start < 0:
        raise InvalidCornerBoundaryError(
            f"Corner {boundary['corner']}: start ({start}) must be non-negative."
        )
    if end <= start:
        raise InvalidCornerBoundaryError(
            f"Corner {boundary['corner']}: end ({end}) must be greater than start ({start})."
        )
    if end > max_distance:
        raise InvalidCornerBoundaryError(
            f"Corner {boundary['corner']}: end ({end}) exceeds available telemetry "
            f"range (max distance = {max_distance})."
        )


# --------------------------------------------------------------------------
# Reusable per-channel analysis helpers
# --------------------------------------------------------------------------


def _first_threshold_crossing(
    distance: np.ndarray,
    channel: np.ndarray,
    threshold: float,
    window_start: float,
    window_end: float,
) -> float | None:
    """Find the first distance within a window where a channel exceeds a threshold.

    Shared by braking-point detection (Brake > 10%) and throttle-application
    detection (Throttle > 80%), so the "find first crossing" logic exists
    exactly once rather than being duplicated per channel.

    Args:
        distance: Full-lap distance array (meters).
        channel: Full-lap channel values aligned with `distance` (e.g. Brake or Throttle, %).
        threshold: Value the channel must exceed.
        window_start: Inclusive lower bound of the search window (meters).
        window_end: Inclusive upper bound of the search window (meters).

    Returns:
        The distance (meters) of the first sample within the window where
        `channel > threshold`, or None if no such sample exists.
    """
    window_mask = (distance >= window_start) & (distance <= window_end)
    windowed_distance = distance[window_mask]
    windowed_channel = channel[window_mask]

    crossing_indices = np.where(windowed_channel > threshold)[0]
    if crossing_indices.size == 0:
        return None

    return float(windowed_distance[crossing_indices[0]])


def _find_apex(
    distance: np.ndarray,
    speed: np.ndarray,
    window_start: float,
    window_end: float,
) -> tuple[float, float]:
    """Find the apex (minimum-speed point) within a corner window.

    Args:
        distance: Full-lap distance array (meters).
        speed: Full-lap speed array aligned with `distance` (km/h).
        window_start: Inclusive lower bound of the corner window (meters).
        window_end: Inclusive upper bound of the corner window (meters).

    Returns:
        A tuple of (apex_distance_m, apex_speed_kph).

    Raises:
        InvalidTelemetryError: If no telemetry samples fall within the window.
    """
    window_mask = (distance >= window_start) & (distance <= window_end)
    windowed_distance = distance[window_mask]
    windowed_speed = speed[window_mask]

    if windowed_distance.size == 0:
        raise InvalidTelemetryError(
            f"No telemetry samples found between {window_start}m and {window_end}m."
        )

    apex_index = int(np.argmin(windowed_speed))
    return float(windowed_distance[apex_index]), float(windowed_speed[apex_index])


def _interpolate_at_distance(
    distance: np.ndarray,
    values: np.ndarray,
    target_distance: float,
) -> float:
    """Interpolate a channel's value at an arbitrary target distance.

    Used for exit speed (value at the corner's end boundary) and for
    reading DeltaTime precisely at corner start/end boundaries, which may
    not fall exactly on a sampled grid point.

    Args:
        distance: Full-lap distance array (meters), strictly increasing.
        values: Channel values aligned with `distance`.
        target_distance: The distance at which to interpolate.

    Returns:
        The interpolated value at `target_distance`.
    """
    return float(np.interp(target_distance, distance, values))


# --------------------------------------------------------------------------
# Single-corner extraction
# --------------------------------------------------------------------------


def _extract_single_corner_metric(
    aligned_telemetry: pd.DataFrame,
    boundary: CornerBoundary,
    brake_threshold_pct: float,
    throttle_threshold_pct: float,
) -> CornerMetric:
    """Compute all metrics for a single corner and return a CornerMetric.

    Args:
        aligned_telemetry: The full aligned telemetry DataFrame.
        boundary: The corner boundary to analyze.
        brake_threshold_pct: Brake % threshold used to detect the braking point.
        throttle_threshold_pct: Throttle % threshold used to detect throttle application.

    Returns:
        A validated CornerMetric for this corner.
    """
    corner_number = boundary["corner"]
    start = float(boundary["start"])
    end = float(boundary["end"])

    distance = aligned_telemetry["Distance"].to_numpy(dtype=float)
    driver_speed = aligned_telemetry["DriverSpeed"].to_numpy(dtype=float)
    reference_speed = aligned_telemetry["ReferenceSpeed"].to_numpy(dtype=float)
    driver_brake = aligned_telemetry["DriverBrake"].to_numpy(dtype=float)
    reference_brake = aligned_telemetry["ReferenceBrake"].to_numpy(dtype=float)
    driver_throttle = aligned_telemetry["DriverThrottle"].to_numpy(dtype=float)
    reference_throttle = aligned_telemetry["ReferenceThrottle"].to_numpy(dtype=float)
    delta_time = aligned_telemetry["DeltaTime"].to_numpy(dtype=float)

    # --- Braking point ---
    driver_braking_point = _first_threshold_crossing(
        distance, driver_brake, brake_threshold_pct, start, end
    )
    reference_braking_point = _first_threshold_crossing(
        distance, reference_brake, brake_threshold_pct, start, end
    )

    if driver_braking_point is None:
        _logger.debug(
            "Corner %s: driver brake never exceeded %.1f%%; treating corner start as "
            "braking point (likely a flat-out/no-braking corner).",
            corner_number,
            brake_threshold_pct,
        )
        driver_braking_point = start
    if reference_braking_point is None:
        _logger.debug(
            "Corner %s: reference brake never exceeded %.1f%%; treating corner start as "
            "braking point.",
            corner_number,
            brake_threshold_pct,
        )
        reference_braking_point = start

    braking_delta = driver_braking_point - reference_braking_point

    # --- Apex speed ---
    driver_apex_distance, driver_apex_speed = _find_apex(distance, driver_speed, start, end)
    reference_apex_distance, reference_apex_speed = _find_apex(
        distance, reference_speed, start, end
    )
    apex_speed_delta = driver_apex_speed - reference_apex_speed

    # --- Exit speed ---
    driver_exit_speed = _interpolate_at_distance(distance, driver_speed, end)
    reference_exit_speed = _interpolate_at_distance(distance, reference_speed, end)
    exit_speed_delta = driver_exit_speed - reference_exit_speed

    # --- Throttle application (searched from each driver's own apex to corner end) ---
    driver_throttle_point = _first_threshold_crossing(
        distance, driver_throttle, throttle_threshold_pct, driver_apex_distance, end
    )
    reference_throttle_point = _first_threshold_crossing(
        distance, reference_throttle, throttle_threshold_pct, reference_apex_distance, end
    )

    if driver_throttle_point is None:
        _logger.debug(
            "Corner %s: driver throttle never exceeded %.1f%% before corner exit; "
            "using corner end as throttle application point.",
            corner_number,
            throttle_threshold_pct,
        )
        driver_throttle_point = end
    if reference_throttle_point is None:
        _logger.debug(
            "Corner %s: reference throttle never exceeded %.1f%% before corner exit; "
            "using corner end as throttle application point.",
            corner_number,
            throttle_threshold_pct,
        )
        reference_throttle_point = end

    throttle_delay = driver_throttle_point - reference_throttle_point

    # --- Corner time loss (marginal delta across this corner's window) ---
    delta_time_at_start = _interpolate_at_distance(distance, delta_time, start)
    delta_time_at_end = _interpolate_at_distance(distance, delta_time, end)
    time_loss = delta_time_at_end - delta_time_at_start

    return CornerMetric(
        corner_number=corner_number,
        start_distance_m=start,
        end_distance_m=end,
        braking_point_m=driver_braking_point,
        braking_delta_m=braking_delta,
        apex_speed_kph=driver_apex_speed,
        apex_speed_delta_kph=apex_speed_delta,
        exit_speed_kph=driver_exit_speed,
        exit_speed_delta_kph=exit_speed_delta,
        throttle_application_m=driver_throttle_point,
        throttle_delay_m=throttle_delay,
        time_loss_s=time_loss,
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def extract_corner_metrics(
    aligned_telemetry: pd.DataFrame,
    corner_boundaries: list[CornerBoundary],
    brake_threshold_pct: float = _DEFAULT_BRAKE_THRESHOLD_PCT,
    throttle_threshold_pct: float = _DEFAULT_THROTTLE_THRESHOLD_PCT,
) -> list[CornerMetric]:
    """Extract deterministic per-corner metrics for every supplied corner boundary.

    Args:
        aligned_telemetry: The aligned DataFrame returned by
            analysis.alignment.load_and_align_laps(), containing Distance,
            DriverSpeed, ReferenceSpeed, DriverBrake, ReferenceBrake,
            DriverThrottle, ReferenceThrottle, and DeltaTime columns.
        corner_boundaries: A list of corner boundary definitions, e.g.
            [{"corner": 1, "start": 120, "end": 220}, ...]. Not
            circuit-specific -- any number of corners, in any order, is
            supported.
        brake_threshold_pct: Brake % above which braking is considered to
            have begun (default: 10.0).
        throttle_threshold_pct: Throttle % above which throttle
            application is considered begun (default: 80.0).

    Returns:
        A list of CornerMetric objects, one per corner boundary, sorted
        by corner_number.

    Raises:
        InvalidTelemetryError: If `aligned_telemetry` is empty or missing
            required columns.
        InvalidCornerBoundaryError: If `corner_boundaries` is empty, or
            any boundary is malformed or exceeds the available telemetry range.
    """
    _validate_aligned_telemetry(aligned_telemetry)

    if not corner_boundaries:
        raise InvalidCornerBoundaryError("corner_boundaries must not be empty.")

    max_distance = float(aligned_telemetry["Distance"].max())

    corner_metrics: list[CornerMetric] = []
    for boundary in corner_boundaries:
        _validate_corner_boundary(boundary, max_distance)
        metric = _extract_single_corner_metric(
            aligned_telemetry, boundary, brake_threshold_pct, throttle_threshold_pct
        )
        corner_metrics.append(metric)
        _logger.debug(
            "Corner %s: braking_delta=%.2fm, apex_speed_delta=%.2fkm/h, "
            "throttle_delay=%.2fm, time_loss=%.3fs.",
            metric.corner_number,
            metric.braking_delta_m,
            metric.apex_speed_delta_kph,
            metric.throttle_delay_m,
            metric.time_loss_s,
        )

    corner_metrics.sort(key=lambda metric: metric.corner_number)

    _logger.info("Extracted metrics for %d corner(s).", len(corner_metrics))
    return corner_metrics
