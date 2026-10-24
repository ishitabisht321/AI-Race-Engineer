"""
tests/test_engine.py

Unit tests for analysis/engine.py.

Every function in engine.py is pure and deterministic, so tests can
verify exact outputs without mocking -- identical CornerMetric inputs
must always produce identical SessionAnalysis outputs.

Coverage targets:
    - _severity_from_excess (via classify_corner_metrics / _classify_corner)
    - _classify_corner priority order (braking > apex > exit > throttle)
    - classify_corner_metrics (sorted output, all-clear corner)
    - compute_performance_score (all three penalty terms, clamping)
    - _determine_biggest_mistake (single vs repeated, all-clear)
    - build_session_analysis (happy path, empty-input guard, score bounds)
"""

from __future__ import annotations

import pytest

from analysis.engine import (
    EARLY_BRAKING_THRESHOLD_M,
    LATE_BRAKING_THRESHOLD_M,
    LATE_THROTTLE_THRESHOLD_M,
    POOR_EXIT_THRESHOLD_KPH,
    SLOW_APEX_THRESHOLD_KPH,
    TIME_LOSS_PENALTY_CAP,
    TIME_LOSS_PENALTY_PER_SECOND,
    InvalidCornerMetricsError,
    build_session_analysis,
    classify_corner_metrics,
    compute_performance_score,
)
from models.schemas import CornerMetric, IssueSeverity, IssueType, SessionAnalysis


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _make_corner(
    *,
    corner_number: int = 1,
    start_distance_m: float = 100.0,
    end_distance_m: float = 300.0,
    braking_delta_m: float = 0.0,
    apex_speed_delta_kph: float = 0.0,
    exit_speed_delta_kph: float = 0.0,
    throttle_delay_m: float = 0.0,
    time_loss_s: float = 0.0,
    apex_speed_kph: float = 150.0,
    exit_speed_kph: float = 200.0,
) -> CornerMetric:
    """Build a CornerMetric with controllable delta fields; defaults are all-clear."""
    return CornerMetric(
        corner_number=corner_number,
        start_distance_m=start_distance_m,
        end_distance_m=end_distance_m,
        braking_point_m=start_distance_m + 10.0,
        braking_delta_m=braking_delta_m,
        apex_speed_kph=apex_speed_kph,
        apex_speed_delta_kph=apex_speed_delta_kph,
        exit_speed_kph=exit_speed_kph,
        exit_speed_delta_kph=exit_speed_delta_kph,
        throttle_application_m=start_distance_m + 150.0,
        throttle_delay_m=throttle_delay_m,
        time_loss_s=time_loss_s,
    )


def _make_two_corners() -> tuple[CornerMetric, CornerMetric]:
    """Return two non-overlapping corners with distinct time losses."""
    c1 = _make_corner(corner_number=1, start_distance_m=100.0, end_distance_m=250.0, time_loss_s=0.5)
    c2 = _make_corner(corner_number=2, start_distance_m=400.0, end_distance_m=550.0, time_loss_s=0.2)
    return c1, c2


# ---------------------------------------------------------------------------
# TestClassifyCornerMetrics – priority order and sorting
# ---------------------------------------------------------------------------


class TestClassifyCornerMetrics:
    """Tests for classify_corner_metrics and the per-corner classifier."""

    def test_all_clear_corner_gets_none_issue_and_severity(self) -> None:
        """A corner with all deltas inside thresholds is classified as NONE/NONE."""
        corner = _make_corner()
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.NONE
        assert result[0].severity == IssueSeverity.NONE

    # --- Braking rules ---

    def test_early_braking_is_detected_when_delta_crosses_threshold(self) -> None:
        """A braking_delta_m below EARLY_BRAKING_THRESHOLD_M triggers EARLY_BRAKING."""
        corner = _make_corner(braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 1.0)
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.EARLY_BRAKING

    def test_early_braking_boundary_is_exclusive(self) -> None:
        """braking_delta_m exactly at EARLY_BRAKING_THRESHOLD_M does NOT trigger the rule."""
        corner = _make_corner(braking_delta_m=EARLY_BRAKING_THRESHOLD_M)
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.NONE

    def test_late_braking_is_detected_when_delta_crosses_threshold(self) -> None:
        """A braking_delta_m above LATE_BRAKING_THRESHOLD_M triggers LATE_BRAKING."""
        corner = _make_corner(braking_delta_m=LATE_BRAKING_THRESHOLD_M + 1.0)
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.LATE_BRAKING

    def test_late_braking_boundary_is_exclusive(self) -> None:
        """braking_delta_m exactly at LATE_BRAKING_THRESHOLD_M does NOT trigger the rule."""
        corner = _make_corner(braking_delta_m=LATE_BRAKING_THRESHOLD_M)
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.NONE

    # --- Apex rule ---

    def test_slow_apex_is_detected_when_delta_crosses_threshold(self) -> None:
        """An apex_speed_delta_kph below SLOW_APEX_THRESHOLD_KPH triggers SLOW_APEX."""
        corner = _make_corner(apex_speed_delta_kph=SLOW_APEX_THRESHOLD_KPH - 1.0)
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.SLOW_APEX

    def test_slow_apex_boundary_is_exclusive(self) -> None:
        """apex_speed_delta_kph exactly at SLOW_APEX_THRESHOLD_KPH does NOT trigger the rule."""
        corner = _make_corner(apex_speed_delta_kph=SLOW_APEX_THRESHOLD_KPH)
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.NONE

    # --- Exit rule ---

    def test_poor_exit_is_detected_when_delta_crosses_threshold(self) -> None:
        """An exit_speed_delta_kph below POOR_EXIT_THRESHOLD_KPH triggers POOR_EXIT_SPEED."""
        corner = _make_corner(exit_speed_delta_kph=POOR_EXIT_THRESHOLD_KPH - 1.0)
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.POOR_EXIT_SPEED

    def test_poor_exit_boundary_is_exclusive(self) -> None:
        """exit_speed_delta_kph exactly at POOR_EXIT_THRESHOLD_KPH does NOT trigger the rule."""
        corner = _make_corner(exit_speed_delta_kph=POOR_EXIT_THRESHOLD_KPH)
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.NONE

    # --- Throttle rule ---

    def test_late_throttle_is_detected_when_delay_crosses_threshold(self) -> None:
        """A throttle_delay_m above LATE_THROTTLE_THRESHOLD_M triggers LATE_THROTTLE."""
        corner = _make_corner(throttle_delay_m=LATE_THROTTLE_THRESHOLD_M + 1.0)
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.LATE_THROTTLE

    def test_late_throttle_boundary_is_exclusive(self) -> None:
        """throttle_delay_m exactly at LATE_THROTTLE_THRESHOLD_M does NOT trigger the rule."""
        corner = _make_corner(throttle_delay_m=LATE_THROTTLE_THRESHOLD_M)
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.NONE

    # --- Priority: braking beats apex, apex beats exit, exit beats throttle ---

    def test_braking_takes_priority_over_slow_apex(self) -> None:
        """When both early-braking and slow-apex are triggered, EARLY_BRAKING wins."""
        corner = _make_corner(
            braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 5.0,
            apex_speed_delta_kph=SLOW_APEX_THRESHOLD_KPH - 5.0,
        )
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.EARLY_BRAKING

    def test_apex_takes_priority_over_poor_exit(self) -> None:
        """When both slow-apex and poor-exit are triggered, SLOW_APEX wins."""
        corner = _make_corner(
            apex_speed_delta_kph=SLOW_APEX_THRESHOLD_KPH - 5.0,
            exit_speed_delta_kph=POOR_EXIT_THRESHOLD_KPH - 5.0,
        )
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.SLOW_APEX

    def test_exit_takes_priority_over_late_throttle(self) -> None:
        """When both poor-exit and late-throttle are triggered, POOR_EXIT_SPEED wins."""
        corner = _make_corner(
            exit_speed_delta_kph=POOR_EXIT_THRESHOLD_KPH - 5.0,
            throttle_delay_m=LATE_THROTTLE_THRESHOLD_M + 5.0,
        )
        result = classify_corner_metrics([corner])
        assert result[0].issue_type == IssueType.POOR_EXIT_SPEED

    # --- Severity tiers ---

    def test_low_severity_is_assigned_just_past_threshold(self) -> None:
        """A value just over the threshold (ratio < 1.5×) receives LOW severity."""
        # ratio = abs(value) / abs(threshold) < 1.5 → LOW
        margin = abs(EARLY_BRAKING_THRESHOLD_M) * 1.4  # ratio ≈ 1.4
        corner = _make_corner(braking_delta_m=-margin)
        result = classify_corner_metrics([corner])
        assert result[0].severity == IssueSeverity.LOW

    def test_medium_severity_is_assigned_at_1_5x_threshold(self) -> None:
        """A value at 1.7× threshold magnitude receives MEDIUM severity."""
        margin = abs(EARLY_BRAKING_THRESHOLD_M) * 1.7  # ratio 1.7 → MEDIUM
        corner = _make_corner(braking_delta_m=-margin)
        result = classify_corner_metrics([corner])
        assert result[0].severity == IssueSeverity.MEDIUM

    def test_high_severity_is_assigned_at_2x_threshold(self) -> None:
        """A value at ≥ 2× threshold magnitude receives HIGH severity."""
        margin = abs(EARLY_BRAKING_THRESHOLD_M) * 2.5  # ratio 2.5 → HIGH
        corner = _make_corner(braking_delta_m=-margin)
        result = classify_corner_metrics([corner])
        assert result[0].severity == IssueSeverity.HIGH

    # --- Output ordering ---

    def test_output_is_sorted_by_corner_number(self) -> None:
        """classify_corner_metrics returns corners sorted by corner_number, not insertion order."""
        c3 = _make_corner(corner_number=3, start_distance_m=600.0, end_distance_m=750.0)
        c1 = _make_corner(corner_number=1, start_distance_m=100.0, end_distance_m=250.0)
        c2 = _make_corner(corner_number=2, start_distance_m=400.0, end_distance_m=550.0)
        result = classify_corner_metrics([c3, c1, c2])
        assert [r.corner_number for r in result] == [1, 2, 3]

    def test_returns_new_corner_metric_objects(self) -> None:
        """The classified list contains new CornerMetric instances (frozen originals unchanged)."""
        corner = _make_corner()
        result = classify_corner_metrics([corner])
        assert result[0] is not corner  # model_copy produces a new object

    def test_original_corner_metric_is_not_mutated(self) -> None:
        """The input CornerMetric is unchanged after classification (frozen model)."""
        corner = _make_corner(braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 20.0)
        classify_corner_metrics([corner])
        # The original must still have the unclassified defaults
        assert corner.issue_type == IssueType.NONE
        assert corner.severity == IssueSeverity.NONE


# ---------------------------------------------------------------------------
# TestComputePerformanceScore
# ---------------------------------------------------------------------------


class TestComputePerformanceScore:
    """Tests for compute_performance_score."""

    def test_perfect_lap_scores_100(self) -> None:
        """No issues, zero time loss → score of 100."""
        corner = _make_corner()  # all deltas at zero, no classification
        classified = classify_corner_metrics([corner])
        score = compute_performance_score(classified, total_time_loss_s=0.0)
        assert score == 100

    def test_score_decreases_with_time_loss(self) -> None:
        """A non-zero time loss produces a score below 100."""
        corner = _make_corner()
        classified = classify_corner_metrics([corner])
        score = compute_performance_score(classified, total_time_loss_s=1.0)
        # 10 pts/s * 1 s = 10 point penalty → expect 90 (no issue penalty)
        assert score == 90

    def test_time_loss_penalty_is_capped(self) -> None:
        """The time-loss penalty cannot exceed TIME_LOSS_PENALTY_CAP."""
        corner = _make_corner()
        classified = classify_corner_metrics([corner])
        # Any time loss ≥ cap/penalty_rate should hit the cap
        excessive_loss = TIME_LOSS_PENALTY_CAP / TIME_LOSS_PENALTY_PER_SECOND + 10.0
        score = compute_performance_score(classified, total_time_loss_s=excessive_loss)
        # Only cap applies; no severity or mistake penalty on a clean corner
        expected = int(round(100.0 - TIME_LOSS_PENALTY_CAP))
        assert score == expected

    def test_high_severity_issue_subtracts_10_points(self) -> None:
        """A single HIGH-severity issue removes 10 extra penalty points."""
        corner = _make_corner(braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 30.0)  # HIGH
        classified = classify_corner_metrics([corner])
        assert classified[0].severity == IssueSeverity.HIGH
        score = compute_performance_score(classified, total_time_loss_s=0.0)
        # 10 (severity) + 1 (mistake count) = 11 pts deducted
        assert score == 89

    def test_medium_severity_issue_subtracts_5_points(self) -> None:
        """A single MEDIUM-severity issue removes 5 severity penalty points."""
        margin = abs(EARLY_BRAKING_THRESHOLD_M) * 1.7  # MEDIUM
        corner = _make_corner(braking_delta_m=-margin)
        classified = classify_corner_metrics([corner])
        assert classified[0].severity == IssueSeverity.MEDIUM
        score = compute_performance_score(classified, total_time_loss_s=0.0)
        # 5 (severity) + 1 (mistake count) = 6 pts deducted
        assert score == 94

    def test_low_severity_issue_subtracts_2_points(self) -> None:
        """A single LOW-severity issue removes 2 severity penalty points."""
        margin = abs(EARLY_BRAKING_THRESHOLD_M) * 1.4  # LOW
        corner = _make_corner(braking_delta_m=-margin)
        classified = classify_corner_metrics([corner])
        assert classified[0].severity == IssueSeverity.LOW
        score = compute_performance_score(classified, total_time_loss_s=0.0)
        # 2 (severity) + 1 (mistake count) = 3 pts deducted
        assert score == 97

    def test_score_is_clamped_to_zero_minimum(self) -> None:
        """Score never goes below 0, even with overwhelming penalties."""
        corners = [
            _make_corner(
                corner_number=i,
                start_distance_m=i * 200.0,
                end_distance_m=i * 200.0 + 100.0,
                braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 50.0,
                time_loss_s=2.0,
            )
            for i in range(1, 20)
        ]
        classified = classify_corner_metrics(corners)
        total_loss = sum(c.time_loss_s for c in classified)
        score = compute_performance_score(classified, total_time_loss_s=total_loss)
        assert score == 0

    def test_score_is_clamped_to_100_maximum(self) -> None:
        """Score never exceeds 100 (e.g. zero loss + no issues)."""
        corner = _make_corner()
        classified = classify_corner_metrics([corner])
        score = compute_performance_score(classified, total_time_loss_s=0.0)
        assert score <= 100

    def test_score_is_an_integer(self) -> None:
        """compute_performance_score always returns an int."""
        corner = _make_corner(time_loss_s=0.123)
        classified = classify_corner_metrics([corner])
        score = compute_performance_score(classified, total_time_loss_s=0.123)
        assert isinstance(score, int)

    def test_multiple_issues_accumulate_penalties(self) -> None:
        """Two corners with HIGH issues accumulate more deductions than one."""
        c1 = _make_corner(
            corner_number=1,
            start_distance_m=100.0,
            end_distance_m=250.0,
            braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 30.0,
        )
        c2 = _make_corner(
            corner_number=2,
            start_distance_m=400.0,
            end_distance_m=550.0,
            braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 30.0,
        )
        classified = classify_corner_metrics([c1, c2])
        score_two = compute_performance_score(classified, total_time_loss_s=0.0)

        classified_one = classify_corner_metrics([c1])
        score_one = compute_performance_score(classified_one, total_time_loss_s=0.0)

        assert score_two < score_one


# ---------------------------------------------------------------------------
# TestBuildSessionAnalysis – orchestrator
# ---------------------------------------------------------------------------


class TestBuildSessionAnalysis:
    """Tests for the top-level build_session_analysis function."""

    def test_raises_on_empty_corner_metrics(self) -> None:
        """build_session_analysis raises InvalidCornerMetricsError for an empty list."""
        with pytest.raises(InvalidCornerMetricsError):
            build_session_analysis([], "2024_Monaco_Q", "VER", "NOR")

    def test_returns_session_analysis_for_valid_input(self) -> None:
        """Happy-path single corner returns a fully populated SessionAnalysis."""
        corner = _make_corner(time_loss_s=0.1)
        result = build_session_analysis([corner], "2024_Monaco_Q", "VER", "NOR")
        assert isinstance(result, SessionAnalysis)
        assert result.session_id == "2024_Monaco_Q"

    def test_driver_summary_codes_are_propagated(self) -> None:
        """driver_code and reference_driver_code are stored on the DriverSummary."""
        corner = _make_corner(time_loss_s=0.0)
        result = build_session_analysis([corner], "2024_Monza_R", "HAM", "RUS")
        assert result.driver_summary.driver_code == "HAM"
        assert result.driver_summary.reference_driver_code == "RUS"

    def test_performance_score_is_within_bounds(self) -> None:
        """performance_score is always in [0, 100]."""
        c1, c2 = _make_two_corners()
        result = build_session_analysis([c1, c2], "2024_Monaco_Q", "VER", "NOR")
        assert 0 <= result.driver_summary.performance_score <= 100

    def test_total_time_loss_is_non_negative(self) -> None:
        """total_time_loss_s is floored at zero even if the driver is net faster."""
        # Both corners have negative time_loss (driver faster than reference)
        c1 = _make_corner(corner_number=1, start_distance_m=100.0, end_distance_m=250.0, time_loss_s=-0.5)
        c2 = _make_corner(corner_number=2, start_distance_m=400.0, end_distance_m=550.0, time_loss_s=-0.3)
        result = build_session_analysis([c1, c2], "2024_Monaco_Q", "VER", "NOR")
        assert result.driver_summary.total_time_loss_s == 0.0

    def test_worst_corner_has_largest_time_loss(self) -> None:
        """worst_corner matches the corner_number of the corner with the highest time_loss_s."""
        c1 = _make_corner(corner_number=1, start_distance_m=100.0, end_distance_m=250.0, time_loss_s=0.1)
        c2 = _make_corner(corner_number=2, start_distance_m=400.0, end_distance_m=550.0, time_loss_s=0.9)
        result = build_session_analysis([c1, c2], "2024_Monaco_Q", "VER", "NOR")
        assert result.worst_corner == 2

    def test_best_corner_has_smallest_time_loss(self) -> None:
        """best_corner matches the corner_number of the corner with the lowest time_loss_s."""
        c1 = _make_corner(corner_number=1, start_distance_m=100.0, end_distance_m=250.0, time_loss_s=0.1)
        c2 = _make_corner(corner_number=2, start_distance_m=400.0, end_distance_m=550.0, time_loss_s=0.9)
        result = build_session_analysis([c1, c2], "2024_Monaco_Q", "VER", "NOR")
        assert result.best_corner == 1

    def test_corner_metrics_are_classified_in_output(self) -> None:
        """Returned corner_metrics have issue_type populated (not left as NONE when an issue exists)."""
        corner = _make_corner(braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 20.0)
        result = build_session_analysis([corner], "2024_Monaco_Q", "VER", "NOR")
        assert result.corner_metrics[0].issue_type == IssueType.EARLY_BRAKING

    def test_biggest_mistake_is_populated(self) -> None:
        """biggest_mistake is a non-empty string for both clean and faulty laps."""
        corner = _make_corner(time_loss_s=0.1)
        result = build_session_analysis([corner], "2024_Monaco_Q", "VER", "NOR")
        assert isinstance(result.biggest_mistake, str)
        assert len(result.biggest_mistake) > 0

    def test_no_errors_detected_message_on_clean_lap(self) -> None:
        """A lap with no issues produces the 'no significant errors' biggest_mistake text."""
        corner = _make_corner()
        result = build_session_analysis([corner], "2024_Monaco_Q", "VER", "NOR")
        assert "No significant driving errors detected" in result.biggest_mistake

    def test_repeated_issue_description_in_biggest_mistake(self) -> None:
        """Multiple corners with the same issue produce a 'Repeated ...' biggest_mistake."""
        c1 = _make_corner(
            corner_number=1,
            start_distance_m=100.0,
            end_distance_m=250.0,
            braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 20.0,
            time_loss_s=0.3,
        )
        c2 = _make_corner(
            corner_number=2,
            start_distance_m=400.0,
            end_distance_m=550.0,
            braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 15.0,
            time_loss_s=0.2,
        )
        result = build_session_analysis([c1, c2], "2024_Monaco_Q", "VER", "NOR")
        assert result.biggest_mistake.startswith("Repeated")

    def test_single_issue_uses_detected_phrasing(self) -> None:
        """A single corner issue produces '... detected in corner N.' phrasing."""
        c1 = _make_corner(
            corner_number=1,
            start_distance_m=100.0,
            end_distance_m=250.0,
            braking_delta_m=EARLY_BRAKING_THRESHOLD_M - 20.0,
            time_loss_s=0.3,
        )
        c2 = _make_corner(
            corner_number=2,
            start_distance_m=400.0,
            end_distance_m=550.0,
            time_loss_s=0.1,
        )
        result = build_session_analysis([c1, c2], "2024_Monaco_Q", "VER", "NOR")
        assert "detected in corner" in result.biggest_mistake

    def test_session_analysis_validates_worst_and_best_against_corner_list(self) -> None:
        """The assembled SessionAnalysis is consistent: worst/best refer to real corners."""
        c1, c2 = _make_two_corners()
        result = build_session_analysis([c1, c2], "2024_Monaco_Q", "VER", "NOR")
        known = {m.corner_number for m in result.corner_metrics}
        assert result.worst_corner in known
        assert result.best_corner in known

    def test_single_corner_allows_worst_equals_best(self) -> None:
        """With only one corner, worst_corner and best_corner may be the same corner."""
        corner = _make_corner(time_loss_s=0.5)
        result = build_session_analysis([corner], "2024_Monaco_Q", "VER", "NOR")
        assert result.worst_corner == result.best_corner == 1

    @pytest.mark.parametrize("session_id", ["2024_Monaco_Q", "2025_Monza_R", "1950_Silverstone_R"])
    def test_session_id_is_preserved_exactly(self, session_id: str) -> None:
        """The session_id passed in is stored unchanged on the returned SessionAnalysis."""
        corner = _make_corner()
        result = build_session_analysis([corner], session_id, "VER", "NOR")
        assert result.session_id == session_id
