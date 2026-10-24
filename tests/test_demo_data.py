"""
tests/test_demo_data.py

Unit tests verifying the integrity and mathematical validity of utils/demo_data.py.
"""

from __future__ import annotations

import pandas as pd

from analysis.corners import extract_corner_metrics
from analysis.engine import build_session_analysis
from models.schemas import IssueType
from utils.demo_data import (
    DEMO_CORNER_BOUNDARIES,
    DEMO_DRIVER_CODE,
    DEMO_REFERENCE_DRIVER_CODE,
    DEMO_SESSION_ID,
    generate_demo_telemetry,
)


def test_demo_telemetry_schema_and_determinism() -> None:
    """Verify demo telemetry matches expected columns, is non-empty, and has no NaN values."""
    df1 = generate_demo_telemetry()
    df2 = generate_demo_telemetry()

    pd.testing.assert_frame_equal(df1, df2)

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
    assert list(df1.columns) == expected_columns
    assert len(df1) > 0
    assert not df1.isna().any().any()


def test_demo_telemetry_through_pipeline() -> None:
    """Verify demo telemetry passes through corners and engine to produce expected outcomes."""
    df = generate_demo_telemetry()
    metrics = extract_corner_metrics(df, DEMO_CORNER_BOUNDARIES)
    assert len(metrics) == 6

    analysis = build_session_analysis(
        corner_metrics=metrics,
        session_id=DEMO_SESSION_ID,
        driver_code=DEMO_DRIVER_CODE,
        reference_driver_code=DEMO_REFERENCE_DRIVER_CODE,
    )

    assert analysis.session_id == DEMO_SESSION_ID
    assert analysis.driver_summary.driver_code == DEMO_DRIVER_CODE
    assert analysis.driver_summary.reference_driver_code == DEMO_REFERENCE_DRIVER_CODE
    assert analysis.worst_corner == 3
    assert analysis.best_corner == 4
    assert df["DeltaTime"].iloc[-1] > 0.0
    assert 80 <= analysis.driver_summary.performance_score <= 90

    # Verify corner-specific classifications matching the canned coaching text
    corner_3 = next(c for c in analysis.corner_metrics if c.corner_number == 3)
    assert corner_3.issue_type == IssueType.EARLY_BRAKING

    corner_5 = next(c for c in analysis.corner_metrics if c.corner_number == 5)
    assert corner_5.issue_type == IssueType.LATE_THROTTLE
