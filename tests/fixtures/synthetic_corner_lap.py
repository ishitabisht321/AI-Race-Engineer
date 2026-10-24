"""
tests/fixtures/synthetic_corner_lap.py

Deterministic synthetic telemetry generation with independent control points
for driver and reference channels.

Designed specifically for unit and integration testing of:
    - analysis/corners.py (requiring asymmetric braking, apex, and throttle points)
    - analysis/alignment.py (requiring FastF1-style raw car data with 0/1 brake values)

Builders provided:
    1. build_custom_aligned_telemetry: 8-column aligned DataFrame with 0-100 Brake and Throttle.
    2. build_raw_car_data: FastF1-style Distance/Speed/Throttle/Brake with Brake as 0/1.
    3. build_flat_out_aligned_telemetry: Aligned DataFrame where brake never exceeds threshold.
    4. build_no_throttle_aligned_telemetry: Aligned DataFrame where throttle never exceeds threshold.
    5. build_corner_boundary: Plain dict with keys {"corner", "start", "end"}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_KMH_TO_MS: float = 1.0 / 3.6


def _compute_delta_time(
    distance: np.ndarray,
    driver_speed_kph: np.ndarray,
    reference_speed_kph: np.ndarray,
) -> np.ndarray:
    """Compute cumulative delta time Δt = ∫ (1/v_driver - 1/v_ref) ds in seconds."""
    if distance.size == 0:
        return np.array([], dtype=float)

    driver_ms = np.maximum(driver_speed_kph, 1.0) * _KMH_TO_MS
    ref_ms = np.maximum(reference_speed_kph, 1.0) * _KMH_TO_MS
    integrand = (1.0 / driver_ms) - (1.0 / ref_ms)

    dx = np.diff(distance)
    segment_areas = 0.5 * (integrand[:-1] + integrand[1:]) * dx
    return np.concatenate(([0.0], np.cumsum(segment_areas)))


def build_corner_boundary(
    corner: int = 1,
    start: float = 150.0,
    end: float = 450.0,
) -> dict[str, float | int]:
    """Build a plain-dict corner boundary definition {"corner": int, "start": float, "end": float}."""
    return {"corner": corner, "start": float(start), "end": float(end)}


def build_custom_aligned_telemetry(
    distance_end: float = 1000.0,
    step_m: float = 1.0,
    driver_brake_start_m: float = 200.0,
    reference_brake_start_m: float = 220.0,
    driver_apex_m: float = 300.0,
    driver_apex_speed_kph: float = 110.0,
    reference_apex_m: float = 300.0,
    reference_apex_speed_kph: float = 125.0,
    driver_throttle_start_m: float = 330.0,
    reference_throttle_start_m: float = 315.0,
    driver_exit_speed_kph: float = 190.0,
    reference_exit_speed_kph: float = 205.0,
    straight_speed_kph: float = 300.0,
) -> pd.DataFrame:
    """Generate an 8-column aligned DataFrame with 0-100 Brake and Throttle channels.

    Allows independent control of driver and reference braking points, apex points,
    apex speeds, throttle re-application points, and exit speeds.
    """
    distance = np.arange(0.0, distance_end, step_m)
    n = distance.size

    # Baseline profiles
    d_speed = np.full(n, straight_speed_kph, dtype=float)
    r_speed = np.full(n, straight_speed_kph, dtype=float)
    d_brake = np.zeros(n, dtype=float)
    r_brake = np.zeros(n, dtype=float)
    d_throttle = np.full(n, 100.0, dtype=float)
    r_throttle = np.full(n, 100.0, dtype=float)

    # Driver Braking (driver_brake_start_m to driver_apex_m)
    d_brake_mask = (distance >= driver_brake_start_m) & (distance <= driver_apex_m)
    d_brake[d_brake_mask] = 100.0
    d_dec_mask = (distance >= driver_brake_start_m) & (distance <= driver_apex_m)
    if np.any(d_dec_mask):
        frac = (distance[d_dec_mask] - driver_brake_start_m) / max(driver_apex_m - driver_brake_start_m, 1.0)
        d_speed[d_dec_mask] = straight_speed_kph - frac * (straight_speed_kph - driver_apex_speed_kph)
        d_throttle[d_dec_mask] = 0.0

    # Reference Braking (reference_brake_start_m to reference_apex_m)
    r_brake_mask = (distance >= reference_brake_start_m) & (distance <= reference_apex_m)
    r_brake[r_brake_mask] = 100.0
    r_dec_mask = (distance >= reference_brake_start_m) & (distance <= reference_apex_m)
    if np.any(r_dec_mask):
        frac = (distance[r_dec_mask] - reference_brake_start_m) / max(reference_apex_m - reference_brake_start_m, 1.0)
        r_speed[r_dec_mask] = straight_speed_kph - frac * (straight_speed_kph - reference_apex_speed_kph)
        r_throttle[r_dec_mask] = 0.0

    # Driver Acceleration (driver_apex_m to corner exit ~450m)
    corner_exit_m = max(driver_throttle_start_m + 120.0, 450.0)
    d_acc_mask = (distance > driver_apex_m) & (distance <= corner_exit_m)
    if np.any(d_acc_mask):
        frac = (distance[d_acc_mask] - driver_apex_m) / max(corner_exit_m - driver_apex_m, 1.0)
        d_speed[d_acc_mask] = driver_apex_speed_kph + frac * (driver_exit_speed_kph - driver_apex_speed_kph)
        # Throttle hits 100% at driver_throttle_start_m
        d_th_ramp = np.clip((distance[d_acc_mask] - driver_apex_m) / max(driver_throttle_start_m - driver_apex_m, 1.0) * 100.0, 0.0, 100.0)
        d_throttle[d_acc_mask] = d_th_ramp

    # Reference Acceleration (reference_apex_m to corner exit ~450m)
    r_acc_mask = (distance > reference_apex_m) & (distance <= corner_exit_m)
    if np.any(r_acc_mask):
        frac = (distance[r_acc_mask] - reference_apex_m) / max(corner_exit_m - reference_apex_m, 1.0)
        r_speed[r_acc_mask] = reference_apex_speed_kph + frac * (reference_exit_speed_kph - reference_apex_speed_kph)
        r_th_ramp = np.clip((distance[r_acc_mask] - reference_apex_m) / max(reference_throttle_start_m - reference_apex_m, 1.0) * 100.0, 0.0, 100.0)
        r_throttle[r_acc_mask] = r_th_ramp

    delta_time = _compute_delta_time(distance, d_speed, r_speed)

    return pd.DataFrame(
        {
            "Distance": distance,
            "DriverSpeed": d_speed,
            "ReferenceSpeed": r_speed,
            "DriverBrake": d_brake,
            "ReferenceBrake": r_brake,
            "DriverThrottle": d_throttle,
            "ReferenceThrottle": r_throttle,
            "DeltaTime": delta_time,
        }
    )


def build_raw_car_data(
    distance_end: float = 1000.0,
    step_m: float = 1.0,
    straight_speed_kph: float = 300.0,
    apex_m: float = 300.0,
    apex_speed_kph: float = 120.0,
    brake_start_m: float = 200.0,
    throttle_start_m: float = 320.0,
) -> pd.DataFrame:
    """Generate FastF1-style raw telemetry with Brake channel as boolean 0/1.

    Matches the exact schema and conventions returned by FastF1's
    `lap.get_car_data().add_distance()` before alignment.
    """
    distance = np.arange(0.0, distance_end, step_m)
    n = distance.size

    speed = np.full(n, straight_speed_kph, dtype=float)
    # FastF1 uses 0/1 (or False/True) for Brake
    brake = np.zeros(n, dtype=float)
    throttle = np.full(n, 100.0, dtype=float)

    # Braking phase
    brake_mask = (distance >= brake_start_m) & (distance <= apex_m)
    brake[brake_mask] = 1.0  # FastF1 0/1 indicator
    throttle[brake_mask] = 0.0
    if np.any(brake_mask):
        frac = (distance[brake_mask] - brake_start_m) / max(apex_m - brake_start_m, 1.0)
        speed[brake_mask] = straight_speed_kph - frac * (straight_speed_kph - apex_speed_kph)

    # Exit phase
    exit_m = max(throttle_start_m + 100.0, 450.0)
    exit_mask = (distance > apex_m) & (distance <= exit_m)
    if np.any(exit_mask):
        frac = (distance[exit_mask] - apex_m) / max(exit_m - apex_m, 1.0)
        speed[exit_mask] = apex_speed_kph + frac * (straight_speed_kph - apex_speed_kph)
        throttle[exit_mask] = np.clip((distance[exit_mask] - apex_m) / max(throttle_start_m - apex_m, 1.0) * 100.0, 0.0, 100.0)

    return pd.DataFrame(
        {
            "Distance": distance,
            "Speed": speed,
            "Throttle": throttle,
            "Brake": brake,
        }
    )


def build_flat_out_aligned_telemetry(
    distance_end: float = 1000.0,
    step_m: float = 1.0,
    speed_kph: float = 280.0,
) -> pd.DataFrame:
    """Generate an aligned DataFrame where Brake is 0.0 everywhere.

    Used by test_corners.py to explicitly verify the fallback behavior:
    when brake never exceeds threshold, braking_point_m defaults to corner start.
    """
    distance = np.arange(0.0, distance_end, step_m)
    n = distance.size

    return pd.DataFrame(
        {
            "Distance": distance,
            "DriverSpeed": np.full(n, speed_kph, dtype=float),
            "ReferenceSpeed": np.full(n, speed_kph, dtype=float),
            "DriverBrake": np.zeros(n, dtype=float),
            "ReferenceBrake": np.zeros(n, dtype=float),
            "DriverThrottle": np.full(n, 100.0, dtype=float),
            "ReferenceThrottle": np.full(n, 100.0, dtype=float),
            "DeltaTime": np.zeros(n, dtype=float),
        }
    )


def build_no_throttle_aligned_telemetry(
    distance_end: float = 1000.0,
    step_m: float = 1.0,
    apex_speed_kph: float = 100.0,
) -> pd.DataFrame:
    """Generate an aligned DataFrame where Throttle never exceeds 30% (under 80% threshold).

    Used by test_corners.py to explicitly verify the fallback behavior:
    when throttle never exceeds threshold, throttle_application_m defaults to corner end.
    """
    distance = np.arange(0.0, distance_end, step_m)
    n = distance.size

    return pd.DataFrame(
        {
            "Distance": distance,
            "DriverSpeed": np.full(n, apex_speed_kph, dtype=float),
            "ReferenceSpeed": np.full(n, apex_speed_kph, dtype=float),
            "DriverBrake": np.full(n, 50.0, dtype=float),
            "ReferenceBrake": np.full(n, 50.0, dtype=float),
            "DriverThrottle": np.full(n, 30.0, dtype=float),  # Never reaches 80%
            "ReferenceThrottle": np.full(n, 30.0, dtype=float),
            "DeltaTime": np.zeros(n, dtype=float),
        }
    )
