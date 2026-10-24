"""
tests/fixtures/synthetic_lap.py

Deterministic synthetic telemetry generation for testing analysis/alignment.py
and analysis/corners.py without FastF1, network access, or a real F1 session.

All telemetry here is generated from a smooth, closed-form "one corner"
speed profile (a raised-cosine dip between two straights) -- there is no
randomness anywhere in this module, so calling the same function with the
same arguments always produces bit-for-bit identical output across test
runs.

This module has no dependency on any other project module: it does not
import fastf1, analysis/, models/, or utils/. Metrics that mirror
production logic (e.g. the cumulative delta-time integral) are
deliberately reimplemented locally rather than imported, so a bug in the
production implementation cannot silently make a test's "expected" data
wrong in the same way as the code under test.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

#: Default lap length used when the caller does not specify one.
DEFAULT_DISTANCE_END_M: float = 1000.0

#: Default distance grid spacing.
DEFAULT_STEP_M: float = 1.0

#: Default straight-line speed either side of the synthetic corner.
DEFAULT_STRAIGHT_SPEED_KPH: float = 300.0

#: Default minimum (apex) speed at the center of the synthetic corner.
DEFAULT_APEX_SPEED_KPH: float = 120.0

#: Default half-width of the synthetic corner's speed dip, in meters.
DEFAULT_CORNER_HALF_WIDTH_M: float = 150.0

#: Speed floor used only to avoid division-by-zero when integrating time;
#: never affects the Speed column itself.
_MIN_SPEED_FLOOR_KPH: float = 1.0

#: Conversion factor: kilometers-per-hour to meters-per-second.
_KMH_TO_MS_FACTOR: float = 1.0 / 3.6


def _cosine_dip(distance_m: np.ndarray, center_m: float, half_width_m: float) -> np.ndarray:
    """Build a smooth, deterministic "dip" window centered on a corner.

    Returns 1.0 at `center_m` (the apex), tapering smoothly to 0.0 at
    `center_m +/- half_width_m` via a raised cosine, and staying at 0.0
    outside that range. Used as the shared shape driving the synthetic
    Speed, Throttle, and Brake channels.

    Args:
        distance_m: Distance grid to evaluate the dip over.
        center_m: Distance at which the dip is at its maximum (1.0).
        half_width_m: Distance from the center at which the dip reaches 0.0.

    Returns:
        An array the same shape as `distance_m`, with values in [0.0, 1.0].

    Raises:
        ValueError: If `half_width_m` is not positive.
    """
    if half_width_m <= 0:
        raise ValueError(f"half_width_m must be positive, got {half_width_m}.")

    normalized_offset = (distance_m - center_m) / half_width_m
    return np.where(
        np.abs(normalized_offset) <= 1.0,
        0.5 * (1.0 + np.cos(np.pi * normalized_offset)),
        0.0,
    )


def _compute_time_channel(distance_m: np.ndarray, speed_kph: np.ndarray) -> np.ndarray:
    """Compute a monotonically increasing synthetic Time channel from Distance and Speed.

    Integrates 1/v (seconds per meter) cumulatively over distance, the
    same physical relationship real telemetry Time channels satisfy.

    Args:
        distance_m: Distance grid (meters), strictly increasing.
        speed_kph: Speed at each point in `distance_m` (km/h).

    Returns:
        Elapsed time (seconds) at each point in `distance_m`, starting at 0.0.
    """
    if distance_m.size == 0:
        return np.array([], dtype=float)

    speed_kph_safe = np.maximum(speed_kph, _MIN_SPEED_FLOOR_KPH)
    speed_ms = speed_kph_safe * _KMH_TO_MS_FACTOR
    inverse_speed = 1.0 / speed_ms

    dx = np.diff(distance_m)
    segment_durations = 0.5 * (inverse_speed[:-1] + inverse_speed[1:]) * dx
    return np.concatenate(([0.0], np.cumsum(segment_durations)))


def _compute_cumulative_delta_time(
    distance_m: np.ndarray,
    driver_speed_kph: np.ndarray,
    reference_speed_kph: np.ndarray,
) -> np.ndarray:
    """Compute a synthetic cumulative delta-time trace between two speed profiles.

    Mirrors the physical relationship used by analysis/alignment.py
    (Δt = ∫ (1/v_driver - 1/v_reference) ds), reimplemented locally so
    this fixture has no import dependency on production code.

    Args:
        distance_m: Shared distance grid (meters).
        driver_speed_kph: Driver speed at each point in `distance_m` (km/h).
        reference_speed_kph: Reference speed at each point in `distance_m` (km/h).

    Returns:
        Cumulative delta time (seconds) at each point in `distance_m`.
    """
    if distance_m.size == 0:
        return np.array([], dtype=float)

    driver_speed_ms = np.maximum(driver_speed_kph, _MIN_SPEED_FLOOR_KPH) * _KMH_TO_MS_FACTOR
    reference_speed_ms = np.maximum(reference_speed_kph, _MIN_SPEED_FLOOR_KPH) * _KMH_TO_MS_FACTOR

    integrand = (1.0 / driver_speed_ms) - (1.0 / reference_speed_ms)
    dx = np.diff(distance_m)
    segment_areas = 0.5 * (integrand[:-1] + integrand[1:]) * dx
    return np.concatenate(([0.0], np.cumsum(segment_areas)))


def create_synthetic_telemetry(
    distance_end: float = DEFAULT_DISTANCE_END_M,
    step_m: float = DEFAULT_STEP_M,
    straight_speed_kph: float = DEFAULT_STRAIGHT_SPEED_KPH,
    apex_speed_kph: float = DEFAULT_APEX_SPEED_KPH,
    corner_center_m: float | None = None,
    corner_half_width_m: float = DEFAULT_CORNER_HALF_WIDTH_M,
) -> pd.DataFrame:
    """Generate one driver's deterministic synthetic lap telemetry.

    Models a single straight-corner-straight lap: Speed dips smoothly
    from `straight_speed_kph` to `apex_speed_kph` around `corner_center_m`,
    Brake is non-zero only on the approach to the corner (before the
    apex), and Throttle is inversely related to the speed dip (low near
    the apex, full on the straights). Time is derived from Distance and
    Speed via numerical integration, so it is internally consistent with
    the other channels rather than independently fabricated.

    Args:
        distance_end: Total lap distance to generate, in meters.
        step_m: Distance grid spacing, in meters.
        straight_speed_kph: Speed on the straights, away from the corner.
        apex_speed_kph: Minimum speed at the corner's center. May be set
            equal to `straight_speed_kph` to simulate a flat-out corner
            with no braking event, which is a valid and useful test case.
        corner_center_m: Distance at which the corner's apex is located.
            Defaults to the midpoint of the lap if not provided.
        corner_half_width_m: Distance from the corner center at which the
            speed profile returns to `straight_speed_kph`.

    Returns:
        A DataFrame with columns: Distance, Speed, Throttle, Brake, Time.

    Raises:
        ValueError: If `distance_end` or `step_m` is not positive.
    """
    if distance_end <= 0:
        raise ValueError(f"distance_end must be positive, got {distance_end}.")
    if step_m <= 0:
        raise ValueError(f"step_m must be positive, got {step_m}.")

    if corner_center_m is None:
        corner_center_m = distance_end / 2.0

    distance = np.arange(0.0, distance_end, step_m)

    dip = _cosine_dip(distance, corner_center_m, corner_half_width_m)
    speed = straight_speed_kph - dip * (straight_speed_kph - apex_speed_kph)
    throttle = 100.0 * (1.0 - dip)
    brake = np.where(distance <= corner_center_m, dip * 100.0, 0.0)
    time = _compute_time_channel(distance, speed)

    _logger.debug(
        "Generated synthetic telemetry: %d samples, 0-%.1fm, corner at %.1fm.",
        distance.size,
        distance_end,
        corner_center_m,
    )

    return pd.DataFrame(
        {
            "Distance": distance,
            "Speed": speed,
            "Throttle": throttle,
            "Brake": brake,
            "Time": time,
        }
    )


def create_synthetic_driver_and_reference_telemetry(
    distance_end: float = DEFAULT_DISTANCE_END_M,
    step_m: float = DEFAULT_STEP_M,
    straight_speed_kph: float = DEFAULT_STRAIGHT_SPEED_KPH,
    driver_apex_speed_kph: float = DEFAULT_APEX_SPEED_KPH,
    reference_apex_speed_kph: float = DEFAULT_APEX_SPEED_KPH + 20.0,
    corner_center_m: float | None = None,
    corner_half_width_m: float = DEFAULT_CORNER_HALF_WIDTH_M,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate two single-driver telemetry DataFrames sharing one corner location.

    Both drivers share the same distance grid, straight-line speed, and
    corner position, but the reference driver carries more apex speed
    through the corner by default -- giving alignment tests a
    deterministic, non-trivial performance difference to detect.

    Args:
        distance_end: Total lap distance to generate, in meters.
        step_m: Distance grid spacing, in meters.
        straight_speed_kph: Shared straight-line speed for both drivers.
        driver_apex_speed_kph: Primary driver's apex speed.
        reference_apex_speed_kph: Reference driver's apex speed.
        corner_center_m: Shared corner apex location. Defaults to the lap midpoint.
        corner_half_width_m: Shared corner width.

    Returns:
        A (driver_telemetry, reference_telemetry) tuple, each with
        columns: Distance, Speed, Throttle, Brake, Time.
    """
    driver_telemetry = create_synthetic_telemetry(
        distance_end=distance_end,
        step_m=step_m,
        straight_speed_kph=straight_speed_kph,
        apex_speed_kph=driver_apex_speed_kph,
        corner_center_m=corner_center_m,
        corner_half_width_m=corner_half_width_m,
    )
    reference_telemetry = create_synthetic_telemetry(
        distance_end=distance_end,
        step_m=step_m,
        straight_speed_kph=straight_speed_kph,
        apex_speed_kph=reference_apex_speed_kph,
        corner_center_m=corner_center_m,
        corner_half_width_m=corner_half_width_m,
    )
    return driver_telemetry, reference_telemetry


def create_synthetic_aligned_telemetry(
    distance_end: float = DEFAULT_DISTANCE_END_M,
    step_m: float = DEFAULT_STEP_M,
    straight_speed_kph: float = DEFAULT_STRAIGHT_SPEED_KPH,
    driver_apex_speed_kph: float = DEFAULT_APEX_SPEED_KPH,
    reference_apex_speed_kph: float = DEFAULT_APEX_SPEED_KPH + 20.0,
    corner_center_m: float | None = None,
    corner_half_width_m: float = DEFAULT_CORNER_HALF_WIDTH_M,
) -> pd.DataFrame:
    """Generate a synthetic ALIGNED telemetry DataFrame, matching load_and_align_laps()'s output shape.

    Produces the exact column shape analysis.corners.extract_corner_metrics()
    expects (Distance, DriverSpeed, ReferenceSpeed, DriverBrake,
    ReferenceBrake, DriverThrottle, ReferenceThrottle, DeltaTime), so
    corner-metric tests can exercise that function directly without
    running the full alignment pipeline.

    Args:
        distance_end: Total lap distance to generate, in meters.
        step_m: Distance grid spacing, in meters.
        straight_speed_kph: Shared straight-line speed for both drivers.
        driver_apex_speed_kph: Primary driver's apex speed.
        reference_apex_speed_kph: Reference driver's apex speed.
        corner_center_m: Shared corner apex location. Defaults to the lap midpoint.
        corner_half_width_m: Shared corner width.

    Returns:
        A DataFrame with columns: Distance, DriverSpeed, ReferenceSpeed,
        DriverBrake, ReferenceBrake, DriverThrottle, ReferenceThrottle, DeltaTime.
    """
    driver_telemetry, reference_telemetry = create_synthetic_driver_and_reference_telemetry(
        distance_end=distance_end,
        step_m=step_m,
        straight_speed_kph=straight_speed_kph,
        driver_apex_speed_kph=driver_apex_speed_kph,
        reference_apex_speed_kph=reference_apex_speed_kph,
        corner_center_m=corner_center_m,
        corner_half_width_m=corner_half_width_m,
    )

    distance = driver_telemetry["Distance"].to_numpy()
    delta_time = _compute_cumulative_delta_time(
        distance,
        driver_telemetry["Speed"].to_numpy(),
        reference_telemetry["Speed"].to_numpy(),
    )

    return pd.DataFrame(
        {
            "Distance": distance,
            "DriverSpeed": driver_telemetry["Speed"].to_numpy(),
            "ReferenceSpeed": reference_telemetry["Speed"].to_numpy(),
            "DriverBrake": driver_telemetry["Brake"].to_numpy(),
            "ReferenceBrake": reference_telemetry["Brake"].to_numpy(),
            "DriverThrottle": driver_telemetry["Throttle"].to_numpy(),
            "ReferenceThrottle": reference_telemetry["Throttle"].to_numpy(),
            "DeltaTime": delta_time,
        }
    )


def create_synthetic_corner_boundary(
    corner_number: int = 1,
    corner_center_m: float = DEFAULT_DISTANCE_END_M / 2.0,
    corner_half_width_m: float = DEFAULT_CORNER_HALF_WIDTH_M,
) -> dict[str, float | int]:
    """Build a corner boundary dict spanning a synthetic corner's dip region.

    Matches the shape analysis.corners.CornerBoundary expects
    ({"corner": int, "start": float, "end": float}), so it can be passed
    directly to extract_corner_metrics() alongside telemetry generated
    with the same corner_center_m / corner_half_width_m.

    Args:
        corner_number: The corner number to assign.
        corner_center_m: The corner's apex location; must match the value
            used to generate the corresponding telemetry.
        corner_half_width_m: The corner's half-width; must match the
            value used to generate the corresponding telemetry.

    Returns:
        A dict with keys "corner", "start", and "end".
    """
    return {
        "corner": corner_number,
        "start": corner_center_m - corner_half_width_m,
        "end": corner_center_m + corner_half_width_m,
    }

