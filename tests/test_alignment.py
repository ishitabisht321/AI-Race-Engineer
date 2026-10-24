"""
tests/test_alignment.py

Unit and integration tests for analysis/alignment.py.

Verifies:
    1. TestDistanceGrid: Grid spacing, shorter lap bounding, step_m <= 0 and non-positive max distance.
    2. TestInterpolation: Distance deduplication and non-monotonic sorting in _prepare_for_interpolation,
       and channel linear interpolation.
    3. TestDeltaTime:
       - Analytic case: 200 vs 100 kph over 1000m yields -18.0s.
       - Sign conventions (faster driver -> negative delta; slower driver -> positive delta).
       - Zero delta for identical speeds and safe division for 0 speed.
    4. TestFastF1LoadingAndValidation:
       - Cache creation and SessionLoadError mapping.
       - DriverNotFoundError when driver absent.
       - Accurate laps fallback and LapNotFoundError on NA LapTime.
       - TelemetryChannelMissingError on missing channels or NaN in Distance.
    5. TestLoadAndAlignLapsOrchestration:
       - Output contract: exact 8 columns, DeltaTime[0] == 0, bounded by shorter lap.
       - Same driver / invalid driver validations.
       - Brake-scale test (xfail strict=True pending production alignment.py brake fix).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd
import pytest

from analysis.alignment import (
    DriverNotFoundError,
    LapNotFoundError,
    SessionLoadError,
    TelemetryChannelMissingError,
    _cumulative_trapezoidal,
    _extract_telemetry,
    _load_session,
    _prepare_for_interpolation,
    _select_fastest_lap,
    _validate_drivers_present,
    build_distance_grid,
    compute_cumulative_delta_time,
    enable_fastf1_cache,
    interpolate_channel,
    load_and_align_laps,
)
from tests.fixtures.synthetic_corner_lap import build_raw_car_data
from utils.helpers import HelperValidationError


# --------------------------------------------------------------------------
# Test Doubles (Mocking FastF1 - Zero Network)
# --------------------------------------------------------------------------


class _FakeTelemetry:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def add_distance(self, drop_existing: bool = True) -> pd.DataFrame:
        return self._df.copy()


class _FakeLap(dict):
    def __init__(self, telemetry_df: pd.DataFrame, lap_time: Any = pd.Timedelta(80, "s")) -> None:
        super().__init__()
        self["LapTime"] = lap_time
        self._telemetry_df = telemetry_df

    def get_car_data(self, **kwargs) -> _FakeTelemetry:
        return _FakeTelemetry(self._telemetry_df)


class _FakeLaps:
    def __init__(
        self,
        driver_laps_map: dict[str, list[_FakeLap]],
        accurate_flags: dict[str, bool] | None = None,
    ) -> None:
        self._driver_laps_map = driver_laps_map
        self._accurate_flags = accurate_flags or {}
        drivers = []
        for d, laps in driver_laps_map.items():
            drivers.extend([d] * len(laps))
        self._drivers = pd.Series(drivers, name="Driver")

    def __getitem__(self, key: str) -> pd.Series:
        if key == "Driver":
            return self._drivers
        raise KeyError(key)

    @property
    def empty(self) -> bool:
        return len(self._drivers) == 0

    def pick_drivers(self, driver_code: str) -> _FakeLaps:
        if driver_code not in self._driver_laps_map:
            return _FakeLaps({})
        return _FakeLaps({driver_code: self._driver_laps_map[driver_code]}, self._accurate_flags)

    def pick_accurate(self) -> _FakeLaps:
        filtered = {}
        for d, laps in self._driver_laps_map.items():
            if self._accurate_flags.get(d, True):
                filtered[d] = laps
        return _FakeLaps(filtered, self._accurate_flags)

    def pick_fastest(self, only_by_time: bool = False) -> _FakeLap | None:
        for laps in self._driver_laps_map.values():
            if laps:
                return laps[0]
        return None


class _FakeSession:
    def __init__(self, laps: _FakeLaps) -> None:
        self.laps = laps

    def load(
        self,
        telemetry: bool = True,
        laps: bool = True,
        weather: bool = False,
        messages: bool = False,
    ) -> None:
        pass


def _create_mock_session(
    driver_distance: float = 1000.0,
    reference_distance: float = 1000.0,
    driver_code: str = "VER",
    reference_code: str = "NOR",
    driver_lap_time: Any = pd.Timedelta(80, "s"),
    driver_accurate: bool = True,
    driver_has_nan_distance: bool = False,
    driver_missing_channel: str | None = None,
) -> _FakeSession:
    """Helper to build a configured mock FastF1 session."""
    d_raw = build_raw_car_data(distance_end=driver_distance)
    d_raw["Brake"] = d_raw["Brake"].astype(bool)

    r_raw = build_raw_car_data(distance_end=reference_distance)
    r_raw["Brake"] = r_raw["Brake"].astype(bool)

    if driver_has_nan_distance:
        d_raw.loc[5, "Distance"] = np.nan

    if driver_missing_channel:
        d_raw = d_raw.drop(columns=[driver_missing_channel])

    driver_lap = _FakeLap(d_raw, lap_time=driver_lap_time)
    ref_lap = _FakeLap(r_raw, lap_time=pd.Timedelta(81, "s"))

    laps_map = {
        driver_code: [driver_lap],
        reference_code: [ref_lap],
    }
    accurate_flags = {driver_code: driver_accurate, reference_code: True}
    return _FakeSession(_FakeLaps(laps_map, accurate_flags))


# --------------------------------------------------------------------------
# 1. Distance Grid Tests
# --------------------------------------------------------------------------


class TestDistanceGrid:
    def test_build_distance_grid_bounded_by_shorter_lap(self) -> None:
        d_df = pd.DataFrame({"Distance": [0.0, 500.0, 900.0]})
        r_df = pd.DataFrame({"Distance": [0.0, 500.0, 1000.0]})
        grid = build_distance_grid(d_df, r_df, step_m=1.0)
        assert grid[0] == 0.0
        assert grid[-1] == 899.0
        assert np.all(np.diff(grid) == 1.0)

    def test_build_distance_grid_negative_or_zero_step_raises_value_error(self) -> None:
        d_df = pd.DataFrame({"Distance": [0.0, 500.0]})
        r_df = pd.DataFrame({"Distance": [0.0, 500.0]})
        with pytest.raises(ValueError, match="step_m must be positive"):
            build_distance_grid(d_df, r_df, step_m=0.0)
        with pytest.raises(ValueError, match="step_m must be positive"):
            build_distance_grid(d_df, r_df, step_m=-1.0)

    def test_build_distance_grid_non_positive_max_distance_raises_value_error(self) -> None:
        d_df = pd.DataFrame({"Distance": [0.0]})
        r_df = pd.DataFrame({"Distance": [0.0]})
        with pytest.raises(ValueError, match="non-positive"):
            build_distance_grid(d_df, r_df, step_m=1.0)


# --------------------------------------------------------------------------
# 2. Interpolation Tests
# --------------------------------------------------------------------------


class TestInterpolation:
    def test_prepare_for_interpolation_deduplicates_and_sorts_non_monotonic(self) -> None:
        # Distance contains duplicate 10.0 and non-monotonic order
        distance = np.array([0.0, 20.0, 10.0, 10.0, 30.0])
        values = np.array([0.0, 200.0, 100.0, 999.0, 300.0])
        unique_dist, unique_vals = _prepare_for_interpolation(distance, values)

        # np.unique sorts ascending and returns first occurrence (100.0, not 999.0)
        np.testing.assert_array_equal(unique_dist, np.array([0.0, 10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(unique_vals, np.array([0.0, 100.0, 200.0, 300.0]))

    def test_interpolate_channel_linear(self) -> None:
        src_dist = np.array([0.0, 100.0, 200.0])
        src_val = np.array([0.0, 50.0, 100.0])
        grid = np.array([0.0, 50.0, 150.0, 200.0])
        result = interpolate_channel(src_dist, src_val, grid)
        np.testing.assert_allclose(result, np.array([0.0, 25.0, 75.0, 100.0]))


# --------------------------------------------------------------------------
# 3. Delta Time Tests
# --------------------------------------------------------------------------


class TestDeltaTime:
    def test_cumulative_trapezoidal_exact(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([2.0, 2.0, 2.0, 2.0])  # Integral of 2 dx is 2x
        result = _cumulative_trapezoidal(y, x)
        np.testing.assert_allclose(result, np.array([0.0, 2.0, 4.0, 6.0]))

    def test_compute_cumulative_delta_time_analytic_200_vs_100_kph(self) -> None:
        """Analytic test: 200 vs 100 kph over 1000m.

        Time driver = 1000m / (200/3.6 m/s) = 18.0 s
        Time ref = 1000m / (100/3.6 m/s) = 36.0 s
        Delta time = 18.0 - 36.0 = -18.0 s.
        """
        grid = np.arange(0.0, 1001.0, 1.0)
        d_speed = np.full(grid.size, 200.0)
        r_speed = np.full(grid.size, 100.0)

        dt = compute_cumulative_delta_time(grid, d_speed, r_speed)
        assert dt[0] == 0.0
        assert dt[-1] == pytest.approx(-18.0, rel=1e-3)

    def test_compute_cumulative_delta_time_driver_faster_yields_negative(self) -> None:
        grid = np.arange(0.0, 500.0, 1.0)
        d_speed = np.full(grid.size, 180.0)
        r_speed = np.full(grid.size, 150.0)
        dt = compute_cumulative_delta_time(grid, d_speed, r_speed)
        assert dt[-1] < 0.0

    def test_compute_cumulative_delta_time_driver_slower_yields_positive(self) -> None:
        grid = np.arange(0.0, 500.0, 1.0)
        d_speed = np.full(grid.size, 120.0)
        r_speed = np.full(grid.size, 160.0)
        dt = compute_cumulative_delta_time(grid, d_speed, r_speed)
        assert dt[-1] > 0.0

    def test_compute_cumulative_delta_time_identical_speeds_yields_zero(self) -> None:
        grid = np.arange(0.0, 500.0, 1.0)
        speed = np.full(grid.size, 200.0)
        dt = compute_cumulative_delta_time(grid, speed, speed)
        np.testing.assert_allclose(dt, 0.0)

    def test_compute_cumulative_delta_time_zero_speed_does_not_raise(self) -> None:
        grid = np.array([0.0, 10.0, 20.0])
        d_speed = np.array([0.0, 50.0, 100.0])
        r_speed = np.array([0.0, 50.0, 100.0])
        dt = compute_cumulative_delta_time(grid, d_speed, r_speed)
        assert not np.any(np.isnan(dt))


# --------------------------------------------------------------------------
# 4. FastF1 Loading & Validation Tests
# --------------------------------------------------------------------------


class TestFastF1LoadingAndValidation:
    def test_enable_fastf1_cache_creates_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache_dir = tmp_path / "mock_cache"
        monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda path: None)
        enable_fastf1_cache(str(cache_dir))
        assert cache_dir.exists()

    def test_enable_fastf1_cache_failure_raises_session_load_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _failing_enable(path: str) -> None:
            raise PermissionError("Access denied")

        monkeypatch.setattr(fastf1.Cache, "enable_cache", _failing_enable)
        with pytest.raises(SessionLoadError, match="Failed to enable FastF1 cache"):
            enable_fastf1_cache("some_invalid_dir")

    def test_load_session_failure_raises_session_load_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _failing_get_session(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("API timeout")

        monkeypatch.setattr(fastf1, "get_session", _failing_get_session)
        with pytest.raises(SessionLoadError, match="Failed to load session"):
            _load_session(2024, "Monza", "Q")

    def test_validate_drivers_present_raises_driver_not_found(self) -> None:
        session = _create_mock_session(driver_code="VER", reference_code="NOR")
        with pytest.raises(DriverNotFoundError, match="Driver code.*not found"):
            _validate_drivers_present(session, "VER", "HAM")

    def test_select_fastest_lap_fallback_when_no_accurate_laps(self) -> None:
        session = _create_mock_session(driver_code="VER", driver_accurate=False)
        lap = _select_fastest_lap(session, "VER")
        assert lap is not None
        assert lap.get("LapTime") == pd.Timedelta(80, "s")

    def test_select_fastest_lap_na_lap_time_raises_lap_not_found_error(self) -> None:
        session = _create_mock_session(driver_code="VER", driver_lap_time=pd.NaT)
        with pytest.raises(LapNotFoundError, match="Could not determine a valid fastest lap"):
            _select_fastest_lap(session, "VER")

    def test_extract_telemetry_missing_columns_raises_channel_missing_error(self) -> None:
        session = _create_mock_session(driver_missing_channel="Speed")
        lap = _select_fastest_lap(session, "VER")
        with pytest.raises(TelemetryChannelMissingError, match="missing required channel"):
            _extract_telemetry(lap)

    def test_extract_telemetry_nan_in_distance_raises_channel_missing_error(self) -> None:
        session = _create_mock_session(driver_has_nan_distance=True)
        lap = _select_fastest_lap(session, "VER")
        with pytest.raises(TelemetryChannelMissingError, match="contains missing values"):
            _extract_telemetry(lap)


# --------------------------------------------------------------------------
# 5. Pipeline Orchestration Tests
# --------------------------------------------------------------------------


class TestLoadAndAlignLapsOrchestration:
    def test_load_and_align_laps_same_driver_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="driver and reference_driver must differ"):
            load_and_align_laps(2024, "Monza", "Q", "VER", "VER")

    def test_load_and_align_laps_invalid_driver_code_raises_helper_validation_error(self) -> None:
        with pytest.raises(HelperValidationError):
            load_and_align_laps(2024, "Monza", "Q", "INVALID", "NOR")

    def test_load_and_align_laps_successful_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_session = _create_mock_session(driver_distance=900.0, reference_distance=1000.0)
        monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda path: None)
        monkeypatch.setattr(fastf1, "get_session", lambda *a, **kw: mock_session)

        aligned = load_and_align_laps(
            2024, "Monza", "Q", "VER", "NOR", distance_step_m=1.0, cache_dir=str(tmp_path)
        )

        # Output contract: exact 8 columns
        expected_columns = [
            "Distance",
            "DriverSpeed",
            "ReferenceSpeed",
            "DriverBrake",
            "ReferenceBrake",
            "DriverThrottle",
            "ReferenceThrottle",
            "DeltaTime",
        ]
        assert list(aligned.columns) == expected_columns

        # Grid bounded by shorter lap (driver has 900 samples, 0.0 to 898.0m)
        assert len(aligned) == 899
        assert aligned["Distance"].iloc[-1] == 898.0
        # Delta time starts at 0.0
        assert aligned["DeltaTime"].iloc[0] == 0.0

    @pytest.mark.xfail(
        strict=True,
        reason="Known production bug: alignment.py does not scale boolean FastF1 Brake (0/1) to percentage (0-100%)",
    )
    def test_load_and_align_laps_scales_brake_to_percentage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FastF1 returns Brake as 0/1 (boolean). Alignment must scale it to [0, 100] % for corners.py."""
        mock_session = _create_mock_session(driver_distance=1000.0, reference_distance=1000.0)
        monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda path: None)
        monkeypatch.setattr(fastf1, "get_session", lambda *a, **kw: mock_session)

        aligned = load_and_align_laps(
            2024, "Monza", "Q", "VER", "NOR", distance_step_m=1.0, cache_dir=str(tmp_path)
        )

        # Values must be within percentage range [0.0, 100.0]
        assert aligned["DriverBrake"].min() >= 0.0
        assert aligned["DriverBrake"].max() <= 100.0
        assert aligned["ReferenceBrake"].min() >= 0.0
        assert aligned["ReferenceBrake"].max() <= 100.0

        # Maximum brake applied during corner braking must be approx 100%, not 1.0
        assert aligned["DriverBrake"].max() == pytest.approx(100.0, abs=1.0)
        assert aligned["ReferenceBrake"].max() == pytest.approx(100.0, abs=1.0)
