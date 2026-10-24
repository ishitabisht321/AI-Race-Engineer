"""
utils/demo_data.py

Deterministic synthetic data generator for Demo Mode.

Provides offline telemetry, corner boundaries, and pre-calibrated session
metadata so the entire AI Race Engineer dashboard can run end-to-end without
network access, FastF1 downloads, or a Gemini API key.

Strict architectural constraints:
    - Never imports from tests/
    - No FastF1, Streamlit, or Gemini dependencies
    - Pure mathematical synthesis based on physics principles (1/v integration)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from utils.helpers import kmh_to_ms

#: Canonical demo session identifier
DEMO_SESSION_ID: str = "DEMO_Synthetic_Q"

#: Primary driver code for demo mode
DEMO_DRIVER_CODE: str = "DEM"

#: Reference driver code for demo mode
DEMO_REFERENCE_DRIVER_CODE: str = "REF"

#: Pre-defined corner boundaries spanning a 4800m synthetic circuit (6 corners)
DEMO_CORNER_BOUNDARIES: list[dict[str, float | int]] = [
    {"corner": 1, "start": 460.0, "end": 760.0},
    {"corner": 2, "start": 1180.0, "end": 1420.0},
    {"corner": 3, "start": 1920.0, "end": 2280.0},
    {"corner": 4, "start": 2780.0, "end": 3020.0},
    {"corner": 5, "start": 3540.0, "end": 3860.0},
    {"corner": 6, "start": 4260.0, "end": 4540.0},
]

#: Canned race engineer radio coaching text exactly calibrated to the Rule Engine metrics
DEMO_COACHING_FEEDBACK: str = (
    "• Corner 3 Early Braking (Worst Corner): You braked 18.0m earlier than REF into Corner 3, "
    "dropping apex speed by 15.0 km/h and conceding 0.353s. Move your brake marker closer to corner entry.\n"
    "• Corner 5 Late Throttle: Throttle application was delayed by 14.0m after the apex, "
    "costing 0.034s on corner exit. Pick up maintenance throttle earlier at the apex kerb.\n"
    "• Corner 4 Strong Apex (Best Corner): Carried +5.0 km/h more apex speed than REF, "
    "gaining 0.044s. Use this positive minimum-speed reference across the rest of the lap."
)


class CornerSpec(NamedTuple):
    """Specification of a synthetic corner dip."""

    center_m: float
    half_width_m: float
    driver_apex_kph: float
    ref_apex_kph: float
    brake_shift_m: float
    throttle_shift_m: float


_CORNER_SPECS: tuple[CornerSpec, ...] = (
    CornerSpec(600.0, 100.0, 115.0, 115.0, 0.0, 0.0),      # C1: clean / even
    CornerSpec(1300.0, 90.0, 145.0, 145.0, 0.0, 0.0),     # C2: clean / even
    CornerSpec(2100.0, 120.0, 95.0, 110.0, -18.0, 0.0),   # C3: early braking (-18m) & slow apex (-15kph) -> worst corner
    CornerSpec(2900.0, 90.0, 165.0, 160.0, 0.0, 0.0),     # C4: fast apex (+5kph) -> best corner (gain)
    CornerSpec(3700.0, 110.0, 120.0, 122.0, 0.0, 14.0),   # C5: late throttle delay (+14m)
    CornerSpec(4400.0, 100.0, 135.0, 135.0, 0.0, 0.0),    # C6: clean / even
)


def _raised_cosine_dip(
    distance: np.ndarray,
    center: float,
    half_width: float,
) -> np.ndarray:
    """Evaluate a smooth raised-cosine dip centered at `center` with width `2 * half_width`."""
    offset = (distance - center) / half_width
    return np.where(
        np.abs(offset) <= 1.0,
        0.5 * (1.0 + np.cos(np.pi * offset)),
        0.0,
    )


def generate_demo_telemetry(
    lap_length_m: float = 4800.0,
    step_m: float = 1.0,
) -> pd.DataFrame:
    """Generate aligned synthetic telemetry across 6 corners for DEM vs REF.

    Produces the exact column schema expected by analysis.corners.extract_corner_metrics()
    and utils.plotting:
        Distance, DriverSpeed, ReferenceSpeed, DriverBrake, ReferenceBrake,
        DriverThrottle, ReferenceThrottle, DeltaTime

    Args:
        lap_length_m: Total lap distance in meters (default: 4800m).
        step_m: Spatial grid resolution in meters (default: 1.0m).

    Returns:
        A DataFrame with the aligned telemetry grid.
    """
    distance = np.arange(0.0, lap_length_m, step_m)

    driver_speed = np.full_like(distance, 315.0)
    ref_speed = np.full_like(distance, 315.0)
    driver_throttle = np.full_like(distance, 100.0)
    ref_throttle = np.full_like(distance, 100.0)
    driver_brake = np.zeros_like(distance)
    ref_brake = np.zeros_like(distance)

    for spec in _CORNER_SPECS:
        # Reference speed dip & pedal inputs
        r_dip = _raised_cosine_dip(distance, spec.center_m, spec.half_width_m)
        ref_speed -= r_dip * (315.0 - spec.ref_apex_kph)
        ref_throttle = np.minimum(ref_throttle, 100.0 * (1.0 - r_dip))
        ref_brake = np.maximum(
            ref_brake,
            np.where(distance <= spec.center_m, r_dip * 100.0, 0.0),
        )

        # Driver speed dip & pedal inputs with spatial shifts
        d_center = spec.center_m + spec.brake_shift_m
        d_dip = _raised_cosine_dip(distance, d_center, spec.half_width_m)
        driver_speed -= d_dip * (315.0 - spec.driver_apex_kph)
        driver_brake = np.maximum(
            driver_brake,
            np.where(distance <= d_center, d_dip * 100.0, 0.0),
        )

        if spec.throttle_shift_m != 0.0:
            d_exit_dip = _raised_cosine_dip(
                distance,
                d_center + spec.throttle_shift_m,
                spec.half_width_m,
            )
            driver_throttle = np.minimum(
                driver_throttle,
                100.0 * (1.0 - np.maximum(d_dip, d_exit_dip)),
            )
        else:
            driver_throttle = np.minimum(driver_throttle, 100.0 * (1.0 - d_dip))

    # Cumulative delta time integration: Δt = ∫ (1/v_driver - 1/v_ref) ds
    driver_ms = np.maximum(kmh_to_ms(driver_speed), 1.0)
    ref_ms = np.maximum(kmh_to_ms(ref_speed), 1.0)
    integrand = (1.0 / driver_ms) - (1.0 / ref_ms)

    dx = np.diff(distance)
    segment_areas = 0.5 * (integrand[:-1] + integrand[1:]) * dx
    delta_time = np.concatenate(([0.0], np.cumsum(segment_areas)))

    return pd.DataFrame(
        {
            "Distance": distance,
            "DriverSpeed": driver_speed,
            "ReferenceSpeed": ref_speed,
            "DriverBrake": driver_brake,
            "ReferenceBrake": ref_brake,
            "DriverThrottle": driver_throttle,
            "ReferenceThrottle": ref_throttle,
            "DeltaTime": delta_time,
        }
    )
