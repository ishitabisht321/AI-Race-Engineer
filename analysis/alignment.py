"""
analysis/alignment.py

Responsible ONLY for: loading FastF1 telemetry, selecting the fastest
valid lap for a driver and a reference driver, spatially aligning their
telemetry onto a shared distance grid, and computing cumulative delta
time between them.

This module must NOT:
    - calculate corner-level metrics (that is analysis/corners.py)
    - classify driving mistakes (that is analysis/rules.py)
    - call Gemini (that is llm/feedback.py)
    - contain Streamlit code (that is app/main.py)
    - generate visualizations (that is utils/plotting.py)

The only external dependencies this module imports beyond the standard
library are: fastf1, numpy, pandas, and utils.helpers (generic,
domain-free utilities).
"""

from __future__ import annotations

import os

import fastf1
import numpy as np
import pandas as pd

from utils.helpers import kmh_to_ms, safe_divide, setup_logger, validate_driver_code

_logger = setup_logger(__name__)

#: Telemetry channels that must be present (and interpolated) for both drivers.
_INTERPOLATED_CHANNELS: tuple[str, ...] = ("Speed", "Throttle", "Brake")

#: Telemetry columns required to exist before analysis can proceed.
_REQUIRED_TELEMETRY_COLUMNS: set[str] = {"Distance", "Speed", "Throttle", "Brake"}


class AlignmentError(Exception):
    """Base exception for all errors raised by analysis/alignment.py."""


class SessionLoadError(AlignmentError):
    """Raised when a FastF1 session fails to load."""


class DriverNotFoundError(AlignmentError):
    """Raised when a requested driver code is not present in the session."""


class LapNotFoundError(AlignmentError):
    """Raised when no valid lap can be selected for a driver."""


class TelemetryChannelMissingError(AlignmentError):
    """Raised when required telemetry channels are missing, empty, or malformed."""


# --------------------------------------------------------------------------
# Session / lap loading
# --------------------------------------------------------------------------


def enable_fastf1_cache(cache_dir: str = ".fastf1_cache") -> None:
    """Enable FastF1's on-disk cache, creating the directory if needed.

    Args:
        cache_dir: Path to the directory FastF1 should use for caching
            downloaded session data.

    Raises:
        SessionLoadError: If the cache directory cannot be created or
            FastF1 fails to enable caching at the given path.
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        _logger.debug("FastF1 cache enabled at '%s'.", cache_dir)
    except Exception as exc:
        raise SessionLoadError(f"Failed to enable FastF1 cache at '{cache_dir}': {exc}") from exc


def _load_session(year: int, grand_prix: str, session_type: str) -> fastf1.core.Session:
    """Load a FastF1 session, including lap and telemetry data.

    Args:
        year: Season year (e.g. 2024).
        grand_prix: Grand Prix name or round identifier (e.g. "Monza").
        session_type: Session type code (e.g. "FP1", "Q", "R").

    Returns:
        A loaded `fastf1.core.Session` instance.

    Raises:
        SessionLoadError: If the session cannot be found or fails to load.
    """
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(telemetry=True, laps=True, weather=False, messages=False)
    except Exception as exc:
        raise SessionLoadError(
            f"Failed to load session (year={year}, grand_prix='{grand_prix}', "
            f"session_type='{session_type}'): {exc}"
        ) from exc

    _logger.info("Loaded session: %d %s %s.", year, grand_prix, session_type)
    return session


def _validate_drivers_present(session: fastf1.core.Session, *driver_codes: str) -> None:
    """Validate that every given driver code participated in the session.

    Args:
        session: A loaded FastF1 session.
        *driver_codes: One or more driver codes to check for.

    Raises:
        DriverNotFoundError: If any driver code is not present in the
            session's lap data.
    """
    available_codes = set(session.laps["Driver"].unique())
    missing_codes = [code for code in driver_codes if code not in available_codes]

    if missing_codes:
        raise DriverNotFoundError(
            f"Driver code(s) {missing_codes} not found in this session. "
            f"Available drivers: {sorted(available_codes)}."
        )


def _select_fastest_lap(session: fastf1.core.Session, driver_code: str) -> fastf1.core.Lap:
    """Select a driver's fastest valid ("accurate") lap from a session.

    Prefers laps FastF1 has flagged as accurate (i.e. not affected by
    yellow flags, pit stops, or timing anomalies); falls back to all of
    the driver's laps if none are flagged accurate, logging a warning.

    Args:
        session: A loaded FastF1 session.
        driver_code: FIA driver code to select a lap for.

    Returns:
        The selected fastest lap as a `fastf1.core.Lap`.

    Raises:
        LapNotFoundError: If the driver has no recorded laps, or no valid
            fastest lap can be determined.
    """
    driver_laps = session.laps.pick_drivers(driver_code)
    if driver_laps.empty:
        raise LapNotFoundError(f"No laps found for driver '{driver_code}' in this session.")

    valid_laps = driver_laps.pick_accurate()
    if valid_laps.empty:
        _logger.warning(
            "No laps flagged as accurate for driver '%s'; falling back to all recorded laps.",
            driver_code,
        )
        valid_laps = driver_laps

    fastest_lap = valid_laps.pick_fastest()
    if fastest_lap is None or pd.isna(fastest_lap.get("LapTime")):
        raise LapNotFoundError(
            f"Could not determine a valid fastest lap for driver '{driver_code}'."
        )

    _logger.info("Selected fastest lap for '%s': %s.", driver_code, fastest_lap.get("LapTime"))
    return fastest_lap


def _extract_telemetry(lap: fastf1.core.Lap) -> pd.DataFrame:
    """Extract car telemetry (with a Distance channel) from a lap.

    Args:
        lap: A `fastf1.core.Lap` to extract telemetry from.

    Returns:
        A DataFrame containing at least Distance, Speed, Throttle, and
        Brake columns.

    Raises:
        TelemetryChannelMissingError: If telemetry cannot be extracted,
            is empty, is missing a required channel, or has a malformed
            (null-containing) Distance channel.
    """
    try:
        telemetry = lap.get_car_data().add_distance()
    except Exception as exc:
        raise TelemetryChannelMissingError(
            f"Failed to extract car telemetry for lap: {exc}"
        ) from exc

    if telemetry.empty:
        raise TelemetryChannelMissingError("Extracted telemetry is empty.")

    missing_columns = _REQUIRED_TELEMETRY_COLUMNS - set(telemetry.columns)
    if missing_columns:
        raise TelemetryChannelMissingError(
            f"Telemetry is missing required channel(s): {sorted(missing_columns)}."
        )

    if telemetry["Distance"].isna().any():
        raise TelemetryChannelMissingError("Distance channel contains missing values.")

    return telemetry


# --------------------------------------------------------------------------
# Spatial alignment
# --------------------------------------------------------------------------


def build_distance_grid(
    driver_telemetry: pd.DataFrame,
    reference_telemetry: pd.DataFrame,
    step_m: float = 1.0,
) -> np.ndarray:
    """Build a common 1-meter-spaced distance grid bounded by the shorter lap.

    The grid runs from 0 to the minimum of the two drivers' max recorded
    distance, so both drivers have real (non-extrapolated) telemetry
    covering the entire grid.

    Args:
        driver_telemetry: Extracted telemetry for the primary driver.
        reference_telemetry: Extracted telemetry for the reference driver.
        step_m: Grid spacing in meters (default: 1.0).

    Returns:
        A 1D NumPy array of distance values from 0 to the common max
        distance, spaced by `step_m`.

    Raises:
        ValueError: If `step_m` is not positive, or if the resulting
            common max distance is non-positive.
    """
    if step_m <= 0:
        raise ValueError(f"step_m must be positive, got {step_m}.")

    driver_max_distance = float(driver_telemetry["Distance"].max())
    reference_max_distance = float(reference_telemetry["Distance"].max())
    common_max_distance = min(driver_max_distance, reference_max_distance)

    if common_max_distance <= 0:
        raise ValueError(
            "Computed common max distance is non-positive; cannot build a distance "
            f"grid (driver_max={driver_max_distance}, reference_max={reference_max_distance})."
        )

    grid = np.arange(0.0, common_max_distance, step_m)
    _logger.debug(
        "Built distance grid: 0 to %.1fm in %.1fm steps (%d points).",
        common_max_distance,
        step_m,
        grid.size,
    )
    return grid


def _prepare_for_interpolation(
    distance: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate a distance array (keeping first occurrence) for safe interpolation.

    `numpy.interp` expects a monotonically increasing x array; raw
    telemetry can contain repeated distance values from consecutive
    samples at near-identical track position. This collapses duplicates
    before interpolation.

    Args:
        distance: Raw source distance values.
        values: Corresponding channel values (same length as `distance`).

    Returns:
        A tuple of (unique_distance, corresponding_values), both sorted
        ascending by distance.
    """
    unique_distance, unique_indices = np.unique(distance, return_index=True)
    return unique_distance, values[unique_indices]


def interpolate_channel(
    source_distance: np.ndarray,
    source_values: np.ndarray,
    target_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate a single telemetry channel onto a target distance grid.

    Args:
        source_distance: Strictly increasing distance values for the
            source samples.
        source_values: Channel values corresponding to `source_distance`.
        target_grid: Distance values to interpolate onto.

    Returns:
        Interpolated channel values, one per point in `target_grid`.
    """
    return np.interp(target_grid, source_distance, source_values)


def _interpolate_channels(
    telemetry: pd.DataFrame, distance_grid: np.ndarray
) -> dict[str, np.ndarray]:
    """Interpolate Speed, Throttle, and Brake for one driver onto a distance grid.

    Centralizes the per-channel interpolation loop so it is written once
    and reused for both the primary and reference driver, rather than
    duplicated at each call site.

    Args:
        telemetry: Extracted telemetry containing Distance and the
            channels named in `_INTERPOLATED_CHANNELS`.
        distance_grid: Target distance grid to interpolate onto.

    Returns:
        A dict mapping channel name ("Speed", "Throttle", "Brake") to its
        interpolated NumPy array.
    """
    source_distance = telemetry["Distance"].to_numpy(dtype=float)
    interpolated: dict[str, np.ndarray] = {}

    for channel in _INTERPOLATED_CHANNELS:
        unique_distance, unique_values = _prepare_for_interpolation(
            source_distance, telemetry[channel].to_numpy(dtype=float)
        )
        interpolated[channel] = interpolate_channel(unique_distance, unique_values, distance_grid)

    return interpolated


# --------------------------------------------------------------------------
# Delta time
# --------------------------------------------------------------------------

#: Vectorized form of utils.helpers.safe_divide, applied elementwise across
#: whole telemetry arrays instead of looping in Python.
_vectorized_safe_divide = np.vectorize(safe_divide, otypes=[float])


def _cumulative_trapezoidal(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute the cumulative trapezoidal integral of y with respect to x.

    Args:
        y: Integrand values.
        x: Points at which `y` is sampled (must be the same length as `y`).

    Returns:
        An array the same length as `x`, where each element is the
        running integral from `x[0]` up to that point (starting at 0.0).
    """
    dx = np.diff(x)
    segment_areas = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate(([0.0], np.cumsum(segment_areas)))


def compute_cumulative_delta_time(
    distance_grid: np.ndarray,
    driver_speed_kph: np.ndarray,
    reference_speed_kph: np.ndarray,
) -> np.ndarray:
    """Compute cumulative delta time between a driver and a reference driver.

    Implements Δt(s) = ∫ (1/v_driver − 1/v_reference) ds, where velocity
    is converted to meters-per-second so the result is in seconds. Zero
    or near-zero speed samples are guarded against division-by-zero via
    `utils.helpers.safe_divide`.

    Args:
        distance_grid: Common distance grid (meters) both speed arrays
            are aligned to.
        driver_speed_kph: Driver's interpolated speed (km/h) on the grid.
        reference_speed_kph: Reference driver's interpolated speed (km/h)
            on the grid.

    Returns:
        Cumulative delta time (seconds) at each point in `distance_grid`.
        Positive values indicate the driver is behind the reference at
        that point; negative values indicate the driver is ahead.
    """
    driver_speed_ms = kmh_to_ms(driver_speed_kph)
    reference_speed_ms = kmh_to_ms(reference_speed_kph)

    inverse_driver = _vectorized_safe_divide(1.0, driver_speed_ms, 0.0)
    inverse_reference = _vectorized_safe_divide(1.0, reference_speed_ms, 0.0)

    integrand = inverse_driver - inverse_reference
    delta_time = _cumulative_trapezoidal(integrand, distance_grid)

    _logger.debug(
        "Computed cumulative delta time over %d points; final delta = %.3fs.",
        distance_grid.size,
        delta_time[-1] if delta_time.size else 0.0,
    )
    return delta_time


# --------------------------------------------------------------------------
# Public orchestration function
# --------------------------------------------------------------------------


def load_and_align_laps(
    year: int,
    grand_prix: str,
    session_type: str,
    driver: str,
    reference_driver: str,
    distance_step_m: float = 1.0,
    cache_dir: str = ".fastf1_cache",
) -> pd.DataFrame:
    """Load, spatially align, and compare a driver's fastest lap against a reference driver's.

    Orchestrates the full alignment pipeline: enabling the FastF1 cache,
    loading the session, validating both drivers are present, selecting
    each driver's fastest valid lap, extracting telemetry, building a
    common distance grid, interpolating Speed/Throttle/Brake for both
    drivers onto that grid, and computing cumulative delta time.

    Args:
        year: Season year (e.g. 2024).
        grand_prix: Grand Prix name or round identifier (e.g. "Monza").
        session_type: Session type code (e.g. "FP1", "Q", "R").
        driver: FIA driver code for the primary driver being analyzed.
        reference_driver: FIA driver code for the comparison/reference driver.
        distance_step_m: Spacing of the common distance grid, in meters
            (default: 1.0).
        cache_dir: Directory FastF1 should use for its on-disk cache
            (default: ".fastf1_cache").

    Returns:
        A DataFrame with columns: Distance, DriverSpeed, ReferenceSpeed,
        DriverBrake, ReferenceBrake, DriverThrottle, ReferenceThrottle,
        DeltaTime -- one row per point on the common distance grid.

    Raises:
        HelperValidationError: If `driver` or `reference_driver` is not a
            valid FIA driver code.
        ValueError: If `driver` and `reference_driver` resolve to the
            same code.
        SessionLoadError: If the FastF1 cache or session fails to load.
        DriverNotFoundError: If either driver did not participate in the
            session.
        LapNotFoundError: If a valid fastest lap cannot be selected for
            either driver.
        TelemetryChannelMissingError: If required telemetry channels are
            missing or malformed for either driver.
    """
    driver_code = validate_driver_code(driver)
    reference_code = validate_driver_code(reference_driver)

    if driver_code == reference_code:
        raise ValueError(
            f"driver and reference_driver must differ, both resolved to '{driver_code}'."
        )

    _logger.info(
        "Loading and aligning laps: %s vs %s (%d %s %s).",
        driver_code,
        reference_code,
        year,
        grand_prix,
        session_type,
    )

    enable_fastf1_cache(cache_dir)
    session = _load_session(year, grand_prix, session_type)
    _validate_drivers_present(session, driver_code, reference_code)

    driver_lap = _select_fastest_lap(session, driver_code)
    reference_lap = _select_fastest_lap(session, reference_code)

    driver_telemetry = _extract_telemetry(driver_lap)
    reference_telemetry = _extract_telemetry(reference_lap)

    distance_grid = build_distance_grid(
        driver_telemetry, reference_telemetry, step_m=distance_step_m
    )

    driver_channels = _interpolate_channels(driver_telemetry, distance_grid)
    reference_channels = _interpolate_channels(reference_telemetry, distance_grid)

    delta_time = compute_cumulative_delta_time(
        distance_grid, driver_channels["Speed"], reference_channels["Speed"]
    )

    aligned = pd.DataFrame(
        {
            "Distance": distance_grid,
            "DriverSpeed": driver_channels["Speed"],
            "ReferenceSpeed": reference_channels["Speed"],
            "DriverBrake": driver_channels["Brake"],
            "ReferenceBrake": reference_channels["Brake"],
            "DriverThrottle": driver_channels["Throttle"],
            "ReferenceThrottle": reference_channels["Throttle"],
            "DeltaTime": delta_time,
        }
    )

    final_delta = float(aligned["DeltaTime"].iloc[-1]) if not aligned.empty else 0.0
    _logger.info(
        "Alignment complete: %d common distance points, final delta = %.3fs.",
        len(aligned),
        final_delta,
    )
    return aligned
