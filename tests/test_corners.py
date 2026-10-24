"""
tests/test_corners.py

Unit and integration tests for analysis/corners.py.

Verifies:
    1. TestValidation: Aligned telemetry validation (empty, missing columns) and
       corner boundary validation (negative start, end <= start, end > max_distance, empty list).
    2. TestPrivateHelpers: Small, focused checks for _first_threshold_crossing,
       _find_apex, and _interpolate_at_distance.
    3. TestExtractCornerMetricsDeltas:
       - All 4 sign variations (braking earlier/later, throttle later/earlier).
       - Pre-apex throttle blip (>80%) ignored due to apex-to-exit search window.
       - Independent numerical integration of time loss compared with pytest.approx.
    4. TestFallbackBehaviors:
       - Flat-out corner: braking defaults to start distance.
       - No-throttle corner: throttle application defaults to end distance.
       - Custom brake and throttle thresholds.
    5. TestExtractCornerMetricsMultiCorner:
       - Boundaries passed in reverse order are returned sorted by corner_number.
       - Overlapping corner windows supported without error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.corners import (
    CornerBoundary,
    InvalidCornerBoundaryError,
    InvalidTelemetryError,
    _find_apex,
    _first_threshold_crossing,
    _interpolate_at_distance,
    _validate_aligned_telemetry,
    _validate_corner_boundary,
    extract_corner_metrics,
)
from models.schemas import CornerMetric
from tests.fixtures.synthetic_corner_lap import (
    build_corner_boundary,
    build_custom_aligned_telemetry,
    build_flat_out_aligned_telemetry,
    build_no_throttle_aligned_telemetry,
)


class TestValidation:
    """Test telemetry DataFrame and corner boundary validation rules."""

    def test_validate_aligned_telemetry_empty_raises(self) -> None:
        empty_df = pd.DataFrame()
        with pytest.raises(InvalidTelemetryError, match="Aligned telemetry DataFrame is empty"):
            _validate_aligned_telemetry(empty_df)

    @pytest.mark.parametrize(
        "missing_column",
        [
            "Distance",
            "DriverSpeed",
            "ReferenceSpeed",
            "DriverBrake",
            "ReferenceBrake",
            "DriverThrottle",
            "ReferenceThrottle",
            "DeltaTime",
        ],
    )
    def test_validate_aligned_telemetry_missing_columns_raises(self, missing_column: str) -> None:
        df = build_custom_aligned_telemetry().drop(columns=[missing_column])
        with pytest.raises(InvalidTelemetryError, match="missing required"):
            _validate_aligned_telemetry(df)

    def test_validate_corner_boundary_negative_start_raises(self) -> None:
        boundary: CornerBoundary = {"corner": 1, "start": -10.0, "end": 200.0}
        with pytest.raises(InvalidCornerBoundaryError, match="non-negative"):
            _validate_corner_boundary(boundary, max_distance=1000.0)

    def test_validate_corner_boundary_end_equal_to_start_raises(self) -> None:
        boundary: CornerBoundary = {"corner": 1, "start": 150.0, "end": 150.0}
        with pytest.raises(InvalidCornerBoundaryError, match="greater than start"):
            _validate_corner_boundary(boundary, max_distance=1000.0)

    def test_validate_corner_boundary_end_before_start_raises(self) -> None:
        boundary: CornerBoundary = {"corner": 1, "start": 200.0, "end": 150.0}
        with pytest.raises(InvalidCornerBoundaryError, match="greater than start"):
            _validate_corner_boundary(boundary, max_distance=1000.0)

    def test_validate_corner_boundary_end_exceeds_max_distance_raises(self) -> None:
        # 1000m lap with step 1.0 has max_distance = 999.0
        boundary: CornerBoundary = {"corner": 1, "start": 100.0, "end": 1000.0}
        with pytest.raises(InvalidCornerBoundaryError, match="exceeds available"):
            _validate_corner_boundary(boundary, max_distance=999.0)

    def test_validate_corner_boundary_end_at_max_distance_valid(self) -> None:
        boundary: CornerBoundary = {"corner": 1, "start": 100.0, "end": 999.0}
        # Should not raise
        _validate_corner_boundary(boundary, max_distance=999.0)

    def test_extract_corner_metrics_empty_boundaries_raises(self) -> None:
        df = build_custom_aligned_telemetry()
        with pytest.raises(InvalidCornerBoundaryError, match="must not be empty"):
            extract_corner_metrics(df, [])


class TestPrivateHelpers:
    """Test focused behavior of low-level calculation helpers."""

    def test_first_threshold_crossing_finds_crossing(self) -> None:
        distance = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        channel = np.array([0.0, 5.0, 15.0, 25.0, 30.0])
        crossing = _first_threshold_crossing(distance, channel, threshold=10.0, window_start=100.0, window_end=140.0)
        assert crossing == 120.0

    def test_first_threshold_crossing_returns_none_when_never_exceeded(self) -> None:
        distance = np.array([100.0, 110.0, 120.0])
        channel = np.array([0.0, 5.0, 8.0])
        crossing = _first_threshold_crossing(distance, channel, threshold=10.0, window_start=100.0, window_end=120.0)
        assert crossing is None

    def test_first_threshold_crossing_respects_window_boundaries(self) -> None:
        distance = np.array([50.0, 100.0, 150.0, 200.0])
        channel = np.array([20.0, 5.0, 20.0, 5.0])
        # Crossing at 50m is before window_start (80m); first in window is at 150m
        crossing = _first_threshold_crossing(distance, channel, threshold=10.0, window_start=80.0, window_end=180.0)
        assert crossing == 150.0

    def test_find_apex_finds_minimum_speed(self) -> None:
        distance = np.array([100.0, 200.0, 300.0, 400.0])
        speed = np.array([250.0, 120.0, 95.0, 180.0])
        apex_dist, apex_speed = _find_apex(distance, speed, window_start=150.0, window_end=350.0)
        assert apex_dist == 300.0
        assert apex_speed == 95.0

    def test_find_apex_raises_when_window_empty(self) -> None:
        distance = np.array([10.0, 20.0, 30.0])
        speed = np.array([200.0, 180.0, 190.0])
        with pytest.raises(InvalidTelemetryError, match="No telemetry samples found"):
            _find_apex(distance, speed, window_start=50.0, window_end=80.0)

    def test_interpolate_at_distance_exact_and_interpolated(self) -> None:
        distance = np.array([100.0, 200.0, 300.0])
        values = np.array([10.0, 20.0, 30.0])
        assert _interpolate_at_distance(distance, values, 200.0) == pytest.approx(20.0)
        assert _interpolate_at_distance(distance, values, 150.0) == pytest.approx(15.0)


class TestExtractCornerMetricsDeltas:
    """Test metric delta calculations and sign conventions."""

    def test_braking_earlier_negative_delta_and_throttle_later_positive_delta(self) -> None:
        """Driver brakes 20m earlier (delta < 0) and gets on throttle 12m later (delay > 0)."""
        df = build_custom_aligned_telemetry(
            driver_brake_start_m=200.0,
            reference_brake_start_m=220.0,
            driver_throttle_start_m=330.0,
            reference_throttle_start_m=315.0,
        )
        boundary = build_corner_boundary(1, start=150.0, end=450.0)
        metrics = extract_corner_metrics(df, [boundary])
        m = metrics[0]

        assert m.braking_point_m == 200.0
        assert m.braking_delta_m == pytest.approx(-20.0)
        assert m.braking_delta_m < 0.0

        assert m.throttle_application_m == 325.0
        assert m.throttle_delay_m == pytest.approx(12.0)
        assert m.throttle_delay_m > 0.0

    def test_braking_later_positive_delta_and_throttle_earlier_negative_delta(self) -> None:
        """Driver brakes 20m later (delta > 0) and gets on throttle 12m earlier (delay < 0)."""
        df = build_custom_aligned_telemetry(
            driver_brake_start_m=220.0,
            reference_brake_start_m=200.0,
            driver_throttle_start_m=315.0,
            reference_throttle_start_m=330.0,
        )
        boundary = build_corner_boundary(1, start=150.0, end=450.0)
        metrics = extract_corner_metrics(df, [boundary])
        m = metrics[0]

        assert m.braking_point_m == 220.0
        assert m.braking_delta_m == pytest.approx(20.0)
        assert m.braking_delta_m > 0.0

        assert m.throttle_application_m == 313.0
        assert m.throttle_delay_m == pytest.approx(-12.0)
        assert m.throttle_delay_m < 0.0

    def test_pre_apex_throttle_blip_is_ignored(self) -> None:
        """A throttle blip > 80% before the apex must be ignored because throttle search starts at apex."""
        df = build_custom_aligned_telemetry(
            driver_apex_m=300.0,
            driver_throttle_start_m=330.0,
            reference_throttle_start_m=315.0,
        )
        # Inject an aggressive throttle blip at 250m (before apex at 300m)
        df.loc[df["Distance"] == 250.0, "DriverThrottle"] = 95.0

        boundary = build_corner_boundary(1, start=150.0, end=450.0)
        metrics = extract_corner_metrics(df, [boundary])
        m = metrics[0]

        # Throttle application point must still be detected after apex at 325m, ignoring the 250m blip
        assert m.throttle_application_m == 325.0
        assert m.throttle_delay_m == pytest.approx(12.0)

    def test_apex_and_exit_speed_deltas(self) -> None:
        df = build_custom_aligned_telemetry(
            driver_apex_speed_kph=110.0,
            reference_apex_speed_kph=125.0,
            driver_exit_speed_kph=190.0,
            reference_exit_speed_kph=205.0,
        )
        boundary = build_corner_boundary(1, start=150.0, end=450.0)
        metrics = extract_corner_metrics(df, [boundary])
        m = metrics[0]

        assert m.apex_speed_kph == pytest.approx(110.0)
        assert m.apex_speed_delta_kph == pytest.approx(-15.0)
        assert m.exit_speed_kph == pytest.approx(190.0)
        assert m.exit_speed_delta_kph == pytest.approx(-15.0)

    def test_time_loss_matches_independent_quadrature(self) -> None:
        """Independently compute time loss via numerical trapezoidal integration in the test."""
        start_m = 150.0
        end_m = 450.0
        df = build_custom_aligned_telemetry()
        boundary = build_corner_boundary(1, start=start_m, end=end_m)
        metrics = extract_corner_metrics(df, [boundary])
        m = metrics[0]

        # Independent manual integration over [150m, 450m]
        sub = df[(df["Distance"] >= start_m) & (df["Distance"] <= end_m)]
        dist = sub["Distance"].to_numpy()
        d_speed_ms = sub["DriverSpeed"].to_numpy() / 3.6
        r_speed_ms = sub["ReferenceSpeed"].to_numpy() / 3.6
        integrand = (1.0 / d_speed_ms) - (1.0 / r_speed_ms)
        dx = np.diff(dist)
        expected_time_loss = float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dx))

        assert m.time_loss_s == pytest.approx(expected_time_loss, rel=1e-5)


class TestFallbackBehaviors:
    """Test fallback paths when brake or throttle never exceed threshold."""

    def test_flat_out_corner_falls_back_to_corner_start(self) -> None:
        """When brake never exceeds threshold, braking_point_m defaults to start_distance_m."""
        df = build_flat_out_aligned_telemetry()
        boundary = build_corner_boundary(1, start=120.0, end=350.0)
        metrics = extract_corner_metrics(df, [boundary])
        m = metrics[0]

        assert m.braking_point_m == 120.0
        assert m.braking_delta_m == 0.0

    def test_no_throttle_corner_falls_back_to_corner_end(self) -> None:
        """When throttle never exceeds threshold, throttle_application_m defaults to end_distance_m."""
        df = build_no_throttle_aligned_telemetry()
        boundary = build_corner_boundary(1, start=150.0, end=400.0)
        metrics = extract_corner_metrics(df, [boundary])
        m = metrics[0]

        assert m.throttle_application_m == 400.0
        assert m.throttle_delay_m == 0.0

    def test_custom_thresholds_applied(self) -> None:
        """Passing higher brake and throttle thresholds shifts detected crossing distances."""
        df = build_custom_aligned_telemetry()
        boundary = build_corner_boundary(1, start=150.0, end=450.0)

        # Standard thresholds: brake > 10% is 200m
        std_metric = extract_corner_metrics(df, [boundary], brake_threshold_pct=10.0)[0]
        assert std_metric.braking_point_m == 200.0

        # Impossible threshold: brake > 150% triggers fallback to start (150m)
        high_metric = extract_corner_metrics(df, [boundary], brake_threshold_pct=150.0)[0]
        assert high_metric.braking_point_m == 150.0


class TestExtractCornerMetricsMultiCorner:
    """Test multi-corner sorting and overlapping corner windows."""

    def test_corners_returned_sorted_when_passed_in_reverse(self) -> None:
        df = build_custom_aligned_telemetry()
        b1 = build_corner_boundary(1, start=100.0, end=300.0)
        b2 = build_corner_boundary(2, start=350.0, end=550.0)
        b3 = build_corner_boundary(3, start=600.0, end=800.0)

        # Pass in reverse order [C3, C2, C1]
        results = extract_corner_metrics(df, [b3, b2, b1])

        assert len(results) == 3
        assert [m.corner_number for m in results] == [1, 2, 3]
        for m in results:
            assert isinstance(m, CornerMetric)

    def test_overlapping_corner_windows_supported(self) -> None:
        """Overlapping corner boundaries (e.g. chicane complex) are evaluated independently."""
        df = build_custom_aligned_telemetry()
        # C1 and C2 overlap between 250m and 350m
        b1 = build_corner_boundary(1, start=150.0, end=350.0)
        b2 = build_corner_boundary(2, start=250.0, end=450.0)

        results = extract_corner_metrics(df, [b1, b2])

        assert len(results) == 2
        assert results[0].corner_number == 1
        assert results[1].corner_number == 2
        assert results[0].start_distance_m == 150.0
        assert results[1].start_distance_m == 250.0
