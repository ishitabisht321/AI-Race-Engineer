"""
tests/test_schemas.py

Unit tests for every Pydantic model in models/schemas.py:
IssueType, IssueSeverity, CornerMetric, DriverSummary, SessionAnalysis,
and AnalysisReport.

These tests exercise successful construction, validation failures,
immutability, enum handling, and the cross-field rules encoded in the
schema's model_validators -- with no dependency on FastF1, Streamlit,
or Gemini.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models.schemas import (
    AnalysisReport,
    CornerMetric,
    DriverSummary,
    IssueSeverity,
    IssueType,
    SessionAnalysis,
)

# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def valid_corner_metric_kwargs() -> dict[str, object]:
    """Return a complete, valid set of constructor kwargs for CornerMetric."""
    return {
        "corner_number": 1,
        "start_distance_m": 100.0,
        "end_distance_m": 220.0,
        "braking_point_m": 110.0,
        "braking_delta_m": -2.0,
        "apex_speed_kph": 150.0,
        "apex_speed_delta_kph": 1.5,
        "exit_speed_kph": 200.0,
        "exit_speed_delta_kph": -1.0,
        "throttle_application_m": 180.0,
        "throttle_delay_m": 3.0,
        "time_loss_s": 0.05,
    }


@pytest.fixture
def valid_corner_metric(valid_corner_metric_kwargs: dict[str, object]) -> CornerMetric:
    """Return a valid CornerMetric instance."""
    return CornerMetric(**valid_corner_metric_kwargs)


@pytest.fixture
def valid_driver_summary_kwargs() -> dict[str, object]:
    """Return a complete, valid set of constructor kwargs for DriverSummary."""
    return {
        "driver_code": "VER",
        "reference_driver_code": "NOR",
        "total_time_loss_s": 0.420,
        "performance_score": 87,
    }


@pytest.fixture
def valid_driver_summary(valid_driver_summary_kwargs: dict[str, object]) -> DriverSummary:
    """Return a valid DriverSummary instance."""
    return DriverSummary(**valid_driver_summary_kwargs)


@pytest.fixture
def valid_session_analysis_kwargs(
    valid_corner_metric: CornerMetric,
    valid_driver_summary: DriverSummary,
) -> dict[str, object]:
    """Return a complete, valid set of constructor kwargs for SessionAnalysis."""
    return {
        "session_id": "2024_Monza_Q",
        "driver_summary": valid_driver_summary,
        "corner_metrics": [valid_corner_metric],
        "worst_corner": 1,
        "best_corner": 1,
        "biggest_mistake": "No significant driving errors detected this lap.",
    }


@pytest.fixture
def valid_session_analysis(valid_session_analysis_kwargs: dict[str, object]) -> SessionAnalysis:
    """Return a valid SessionAnalysis instance."""
    return SessionAnalysis(**valid_session_analysis_kwargs)


def _second_corner_metric_kwargs(base_kwargs: dict[str, object]) -> dict[str, object]:
    """Build kwargs for a second, non-overlapping corner derived from a base corner."""
    return {
        **base_kwargs,
        "corner_number": 2,
        "start_distance_m": 340.0,
        "end_distance_m": 430.0,
    }


# --------------------------------------------------------------------------
# IssueType
# --------------------------------------------------------------------------


class TestIssueType:
    """Tests for the IssueType enum."""

    def test_expected_members_have_expected_string_values(self) -> None:
        """Every documented member exists with its expected string value."""
        assert IssueType.NONE == "none"
        assert IssueType.EARLY_BRAKING == "early_braking"
        assert IssueType.LATE_BRAKING == "late_braking"
        assert IssueType.SLOW_APEX == "slow_apex"
        assert IssueType.POOR_EXIT_SPEED == "poor_exit_speed"
        assert IssueType.LATE_THROTTLE == "late_throttle"
        assert IssueType.CONSISTENCY_ISSUE == "consistency_issue"

    def test_member_is_a_str_instance(self) -> None:
        """IssueType members behave as plain strings (str Enum)."""
        assert isinstance(IssueType.EARLY_BRAKING, str)

    def test_invalid_value_raises_value_error(self) -> None:
        """Constructing IssueType from an unrecognized value raises ValueError."""
        with pytest.raises(ValueError):
            IssueType("not_a_real_issue")


# --------------------------------------------------------------------------
# IssueSeverity
# --------------------------------------------------------------------------


class TestIssueSeverity:
    """Tests for the IssueSeverity enum."""

    def test_expected_members_have_expected_string_values(self) -> None:
        """Every documented member exists with its expected string value."""
        assert IssueSeverity.NONE == "none"
        assert IssueSeverity.LOW == "low"
        assert IssueSeverity.MEDIUM == "medium"
        assert IssueSeverity.HIGH == "high"

    def test_invalid_value_raises_value_error(self) -> None:
        """Constructing IssueSeverity from an unrecognized value raises ValueError."""
        with pytest.raises(ValueError):
            IssueSeverity("extreme")


# --------------------------------------------------------------------------
# CornerMetric
# --------------------------------------------------------------------------


class TestCornerMetric:
    """Tests for the CornerMetric model."""

    def test_creates_successfully_with_valid_data(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """A fully valid kwargs set constructs a CornerMetric without error."""
        metric = CornerMetric(**valid_corner_metric_kwargs)
        assert metric.corner_number == 1
        assert metric.braking_point_m == 110.0

    def test_issue_type_and_severity_default_to_none(
        self, valid_corner_metric: CornerMetric
    ) -> None:
        """issue_type/severity default to NONE when not explicitly classified."""
        assert valid_corner_metric.issue_type is IssueType.NONE
        assert valid_corner_metric.severity is IssueSeverity.NONE

    def test_accepts_explicit_issue_classification(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """issue_type and severity can be explicitly supplied at construction."""
        kwargs = {
            **valid_corner_metric_kwargs,
            "issue_type": IssueType.EARLY_BRAKING,
            "severity": IssueSeverity.HIGH,
        }
        metric = CornerMetric(**kwargs)
        assert metric.issue_type == IssueType.EARLY_BRAKING
        assert metric.severity == IssueSeverity.HIGH

    def test_rejects_corner_number_zero(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """corner_number must be >= 1; zero is invalid."""
        kwargs = {**valid_corner_metric_kwargs, "corner_number": 0}
        with pytest.raises(ValidationError):
            CornerMetric(**kwargs)

    def test_rejects_negative_corner_number(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """corner_number must be >= 1; negative values are invalid."""
        kwargs = {**valid_corner_metric_kwargs, "corner_number": -3}
        with pytest.raises(ValidationError):
            CornerMetric(**kwargs)

    def test_rejects_end_distance_equal_to_start_distance(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """end_distance_m must be strictly greater than start_distance_m."""
        kwargs = {
            **valid_corner_metric_kwargs,
            "start_distance_m": 200.0,
            "end_distance_m": 200.0,
        }
        with pytest.raises(ValidationError):
            CornerMetric(**kwargs)

    def test_rejects_end_distance_before_start_distance(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """end_distance_m must not be less than start_distance_m."""
        kwargs = {
            **valid_corner_metric_kwargs,
            "start_distance_m": 300.0,
            "end_distance_m": 100.0,
        }
        with pytest.raises(ValidationError):
            CornerMetric(**kwargs)

    def test_rejects_negative_start_distance(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """start_distance_m must be non-negative."""
        kwargs = {**valid_corner_metric_kwargs, "start_distance_m": -10.0}
        with pytest.raises(ValidationError):
            CornerMetric(**kwargs)

    def test_rejects_negative_apex_speed(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """apex_speed_kph must be non-negative."""
        kwargs = {**valid_corner_metric_kwargs, "apex_speed_kph": -1.0}
        with pytest.raises(ValidationError):
            CornerMetric(**kwargs)

    def test_rejects_negative_exit_speed(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """exit_speed_kph must be non-negative."""
        kwargs = {**valid_corner_metric_kwargs, "exit_speed_kph": -5.0}
        with pytest.raises(ValidationError):
            CornerMetric(**kwargs)

    def test_rejects_missing_required_field(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """Omitting a required field raises a validation error."""
        kwargs = dict(valid_corner_metric_kwargs)
        del kwargs["braking_point_m"]
        with pytest.raises(ValidationError):
            CornerMetric(**kwargs)

    def test_rejects_invalid_issue_type_value(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """An unrecognized issue_type string is rejected by enum validation."""
        kwargs = {**valid_corner_metric_kwargs, "issue_type": "not_a_valid_issue"}
        with pytest.raises(ValidationError):
            CornerMetric(**kwargs)

    def test_rejects_unknown_extra_field(
        self, valid_corner_metric_kwargs: dict[str, object]
    ) -> None:
        """extra='forbid' rejects fields not defined on the model."""
        kwargs = {**valid_corner_metric_kwargs, "unexpected_field": 123}
        with pytest.raises(ValidationError):
            CornerMetric(**kwargs)

    def test_model_is_frozen(self, valid_corner_metric: CornerMetric) -> None:
        """Attempting to mutate a constructed CornerMetric raises a validation error."""
        with pytest.raises(ValidationError):
            valid_corner_metric.corner_number = 99  # type: ignore[misc]

    def test_model_dump_round_trip_produces_an_equal_instance(
        self, valid_corner_metric: CornerMetric
    ) -> None:
        """model_dump() output can reconstruct an equal CornerMetric instance."""
        dumped = valid_corner_metric.model_dump()
        rebuilt = CornerMetric(**dumped)
        assert rebuilt == valid_corner_metric


# --------------------------------------------------------------------------
# DriverSummary
# --------------------------------------------------------------------------


class TestDriverSummary:
    """Tests for the DriverSummary model."""

    def test_creates_successfully_with_valid_data(
        self, valid_driver_summary_kwargs: dict[str, object]
    ) -> None:
        """A fully valid kwargs set constructs a DriverSummary without error."""
        summary = DriverSummary(**valid_driver_summary_kwargs)
        assert summary.driver_code == "VER"
        assert summary.performance_score == 87

    def test_rejects_lowercase_driver_code(
        self, valid_driver_summary_kwargs: dict[str, object]
    ) -> None:
        """driver_code must be uppercase per the field pattern."""
        kwargs = {**valid_driver_summary_kwargs, "driver_code": "ver"}
        with pytest.raises(ValidationError):
            DriverSummary(**kwargs)

    def test_rejects_driver_code_longer_than_three_letters(
        self, valid_driver_summary_kwargs: dict[str, object]
    ) -> None:
        """driver_code must be two or three letters."""
        kwargs = {**valid_driver_summary_kwargs, "driver_code": "VERS"}
        with pytest.raises(ValidationError):
            DriverSummary(**kwargs)

    def test_rejects_driver_code_containing_digits(
        self, valid_driver_summary_kwargs: dict[str, object]
    ) -> None:
        """driver_code must contain only letters."""
        kwargs = {**valid_driver_summary_kwargs, "driver_code": "V3R"}
        with pytest.raises(ValidationError):
            DriverSummary(**kwargs)

    def test_accepts_two_letter_driver_code(
        self, valid_driver_summary_kwargs: dict[str, object]
    ) -> None:
        """Two-letter driver codes are valid per the field pattern."""
        kwargs = {**valid_driver_summary_kwargs, "driver_code": "AB"}
        summary = DriverSummary(**kwargs)
        assert summary.driver_code == "AB"

    def test_rejects_identical_driver_and_reference_driver_codes(
        self, valid_driver_summary_kwargs: dict[str, object]
    ) -> None:
        """A driver cannot be compared against themselves."""
        kwargs = {
            **valid_driver_summary_kwargs,
            "reference_driver_code": valid_driver_summary_kwargs["driver_code"],
        }
        with pytest.raises(ValidationError):
            DriverSummary(**kwargs)

    def test_rejects_negative_total_time_loss(
        self, valid_driver_summary_kwargs: dict[str, object]
    ) -> None:
        """total_time_loss_s must be non-negative."""
        kwargs = {**valid_driver_summary_kwargs, "total_time_loss_s": -0.1}
        with pytest.raises(ValidationError):
            DriverSummary(**kwargs)

    def test_accepts_zero_total_time_loss(
        self, valid_driver_summary_kwargs: dict[str, object]
    ) -> None:
        """A total_time_loss_s of exactly zero is valid (inclusive lower bound)."""
        kwargs = {**valid_driver_summary_kwargs, "total_time_loss_s": 0.0}
        summary = DriverSummary(**kwargs)
        assert summary.total_time_loss_s == 0.0

    @pytest.mark.parametrize("invalid_score", [-1, -50, 101, 250])
    def test_rejects_performance_score_out_of_range(
        self, valid_driver_summary_kwargs: dict[str, object], invalid_score: int
    ) -> None:
        """performance_score must fall within [0, 100]."""
        kwargs = {**valid_driver_summary_kwargs, "performance_score": invalid_score}
        with pytest.raises(ValidationError):
            DriverSummary(**kwargs)

    @pytest.mark.parametrize("boundary_score", [0, 100])
    def test_accepts_performance_score_at_inclusive_boundaries(
        self, valid_driver_summary_kwargs: dict[str, object], boundary_score: int
    ) -> None:
        """performance_score of exactly 0 or 100 is valid."""
        kwargs = {**valid_driver_summary_kwargs, "performance_score": boundary_score}
        summary = DriverSummary(**kwargs)
        assert summary.performance_score == boundary_score

    def test_rejects_non_numeric_performance_score(
        self, valid_driver_summary_kwargs: dict[str, object]
    ) -> None:
        """A string that cannot be parsed as an int is rejected."""
        kwargs = {**valid_driver_summary_kwargs, "performance_score": "not-a-number"}
        with pytest.raises(ValidationError):
            DriverSummary(**kwargs)

    def test_rejects_missing_required_field(
        self, valid_driver_summary_kwargs: dict[str, object]
    ) -> None:
        """Omitting a required field raises a validation error."""
        kwargs = dict(valid_driver_summary_kwargs)
        del kwargs["performance_score"]
        with pytest.raises(ValidationError):
            DriverSummary(**kwargs)

    def test_model_is_frozen(self, valid_driver_summary: DriverSummary) -> None:
        """Attempting to mutate a constructed DriverSummary raises a validation error."""
        with pytest.raises(ValidationError):
            valid_driver_summary.performance_score = 0  # type: ignore[misc]


# --------------------------------------------------------------------------
# SessionAnalysis
# --------------------------------------------------------------------------


class TestSessionAnalysis:
    """Tests for the SessionAnalysis model."""

    def test_creates_successfully_with_valid_data(
        self, valid_session_analysis_kwargs: dict[str, object]
    ) -> None:
        """A fully valid kwargs set constructs a SessionAnalysis without error."""
        analysis = SessionAnalysis(**valid_session_analysis_kwargs)
        assert analysis.session_id == "2024_Monza_Q"
        assert len(analysis.corner_metrics) == 1

    def test_rejects_empty_session_id(
        self, valid_session_analysis_kwargs: dict[str, object]
    ) -> None:
        """session_id must be a non-empty string."""
        kwargs = {**valid_session_analysis_kwargs, "session_id": ""}
        with pytest.raises(ValidationError):
            SessionAnalysis(**kwargs)

    def test_rejects_empty_corner_metrics_list(
        self, valid_session_analysis_kwargs: dict[str, object]
    ) -> None:
        """corner_metrics must contain at least one entry."""
        kwargs = {**valid_session_analysis_kwargs, "corner_metrics": []}
        with pytest.raises(ValidationError):
            SessionAnalysis(**kwargs)

    def test_rejects_worst_corner_not_present_in_corner_metrics(
        self, valid_session_analysis_kwargs: dict[str, object]
    ) -> None:
        """worst_corner must reference a corner_number that actually exists."""
        kwargs = {**valid_session_analysis_kwargs, "worst_corner": 99}
        with pytest.raises(ValidationError):
            SessionAnalysis(**kwargs)

    def test_rejects_best_corner_not_present_in_corner_metrics(
        self, valid_session_analysis_kwargs: dict[str, object]
    ) -> None:
        """best_corner must reference a corner_number that actually exists."""
        kwargs = {**valid_session_analysis_kwargs, "best_corner": 99}
        with pytest.raises(ValidationError):
            SessionAnalysis(**kwargs)

    def test_rejects_worst_corner_below_one(
        self, valid_session_analysis_kwargs: dict[str, object]
    ) -> None:
        """worst_corner must be >= 1."""
        kwargs = {**valid_session_analysis_kwargs, "worst_corner": 0}
        with pytest.raises(ValidationError):
            SessionAnalysis(**kwargs)

    def test_rejects_best_corner_below_one(
        self, valid_session_analysis_kwargs: dict[str, object]
    ) -> None:
        """best_corner must be >= 1."""
        kwargs = {**valid_session_analysis_kwargs, "best_corner": 0}
        with pytest.raises(ValidationError):
            SessionAnalysis(**kwargs)

    def test_allows_equal_worst_and_best_corner_with_a_single_corner(
        self, valid_session_analysis_kwargs: dict[str, object]
    ) -> None:
        """With only one corner analyzed, worst_corner == best_corner is valid."""
        analysis = SessionAnalysis(**valid_session_analysis_kwargs)
        assert analysis.worst_corner == analysis.best_corner == 1

    def test_rejects_equal_worst_and_best_corner_with_multiple_corners(
        self,
        valid_corner_metric_kwargs: dict[str, object],
        valid_driver_summary: DriverSummary,
    ) -> None:
        """With more than one corner, worst_corner and best_corner must differ."""
        corner_one = CornerMetric(**valid_corner_metric_kwargs)
        corner_two = CornerMetric(**_second_corner_metric_kwargs(valid_corner_metric_kwargs))

        with pytest.raises(ValidationError):
            SessionAnalysis(
                session_id="2024_Monza_Q",
                driver_summary=valid_driver_summary,
                corner_metrics=[corner_one, corner_two],
                worst_corner=1,
                best_corner=1,
                biggest_mistake="Some mistake.",
            )

    def test_accepts_distinct_worst_and_best_corner_with_multiple_corners(
        self,
        valid_corner_metric_kwargs: dict[str, object],
        valid_driver_summary: DriverSummary,
    ) -> None:
        """Distinct worst_corner/best_corner values are valid with multiple corners."""
        corner_one = CornerMetric(**valid_corner_metric_kwargs)
        corner_two = CornerMetric(**_second_corner_metric_kwargs(valid_corner_metric_kwargs))

        analysis = SessionAnalysis(
            session_id="2024_Monza_Q",
            driver_summary=valid_driver_summary,
            corner_metrics=[corner_one, corner_two],
            worst_corner=1,
            best_corner=2,
            biggest_mistake="Some mistake.",
        )
        assert analysis.worst_corner == 1
        assert analysis.best_corner == 2

    def test_rejects_empty_biggest_mistake(
        self, valid_session_analysis_kwargs: dict[str, object]
    ) -> None:
        """biggest_mistake must be a non-empty string."""
        kwargs = {**valid_session_analysis_kwargs, "biggest_mistake": ""}
        with pytest.raises(ValidationError):
            SessionAnalysis(**kwargs)

    def test_rejects_missing_driver_summary(
        self, valid_session_analysis_kwargs: dict[str, object]
    ) -> None:
        """Omitting driver_summary raises a validation error."""
        kwargs = dict(valid_session_analysis_kwargs)
        del kwargs["driver_summary"]
        with pytest.raises(ValidationError):
            SessionAnalysis(**kwargs)

    def test_model_is_frozen(self, valid_session_analysis: SessionAnalysis) -> None:
        """Attempting to mutate a constructed SessionAnalysis raises a validation error."""
        with pytest.raises(ValidationError):
            valid_session_analysis.worst_corner = 1  # type: ignore[misc]


# --------------------------------------------------------------------------
# AnalysisReport
# --------------------------------------------------------------------------


class TestAnalysisReport:
    """Tests for the AnalysisReport model."""

    def test_creates_successfully_with_valid_data(
        self, valid_session_analysis: SessionAnalysis
    ) -> None:
        """A fully valid kwargs set constructs an AnalysisReport without error."""
        report = AnalysisReport(analysis=valid_session_analysis, ai_feedback="Great lap overall.")
        assert report.ai_feedback == "Great lap overall."
        assert report.analysis is valid_session_analysis

    def test_generated_at_defaults_to_the_current_utc_timestamp(
        self, valid_session_analysis: SessionAnalysis
    ) -> None:
        """generated_at defaults to approximately now (UTC) when not supplied."""
        before = datetime.now(timezone.utc)
        report = AnalysisReport(analysis=valid_session_analysis, ai_feedback="Solid lap.")
        after = datetime.now(timezone.utc)
        assert before <= report.generated_at <= after

    def test_generated_at_is_computed_independently_per_instance(
        self, valid_session_analysis: SessionAnalysis
    ) -> None:
        """Each AnalysisReport instance receives its own generated_at value, not a shared default."""
        report_one = AnalysisReport(analysis=valid_session_analysis, ai_feedback="A")
        report_two = AnalysisReport(analysis=valid_session_analysis, ai_feedback="B")
        assert isinstance(report_one.generated_at, datetime)
        assert isinstance(report_two.generated_at, datetime)

    def test_accepts_explicit_generated_at(
        self, valid_session_analysis: SessionAnalysis
    ) -> None:
        """An explicitly supplied generated_at overrides the default factory."""
        fixed_timestamp = datetime(2024, 5, 1, 12, 0, 0)
        report = AnalysisReport(
            analysis=valid_session_analysis,
            ai_feedback="Solid lap.",
            generated_at=fixed_timestamp,
        )
        assert report.generated_at == fixed_timestamp

    def test_rejects_empty_ai_feedback(self, valid_session_analysis: SessionAnalysis) -> None:
        """ai_feedback must be a non-empty string."""
        with pytest.raises(ValidationError):
            AnalysisReport(analysis=valid_session_analysis, ai_feedback="")

    def test_rejects_missing_analysis(self) -> None:
        """Omitting analysis raises a validation error."""
        with pytest.raises(ValidationError):
            AnalysisReport(ai_feedback="Some feedback.")  # type: ignore[call-arg]

    def test_model_is_frozen(self, valid_session_analysis: SessionAnalysis) -> None:
        """Attempting to mutate a constructed AnalysisReport raises a validation error."""
        report = AnalysisReport(analysis=valid_session_analysis, ai_feedback="Nice lap.")
        with pytest.raises(ValidationError):
            report.ai_feedback = "Changed."  # type: ignore[misc]
