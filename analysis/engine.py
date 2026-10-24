"""
analysis/engine.py

The deterministic Rule Engine of the AI Race Engineer project.

Responsible ONLY for converting a list of CornerMetric objects (produced
by analysis/corners.py) into a complete, validated SessionAnalysis:
classifying each corner's issue and severity, determining the worst and
best corner, computing total time loss, identifying the session's
biggest driving mistake, and computing an overall performance score.

This module must NOT:
    - call Gemini (that is llm/feedback.py)
    - contain Streamlit code (that is app/main.py)
    - load FastF1 telemetry (that is analysis/alignment.py)
    - calculate telemetry metrics (that is analysis/corners.py)
    - contain machine learning

Every function here is pure and deterministic: identical CornerMetric
input always produces an identical SessionAnalysis output.
"""

from __future__ import annotations

from collections import Counter

from pydantic import ValidationError

from models.schemas import CornerMetric, DriverSummary, IssueSeverity, IssueType, SessionAnalysis
from utils.helpers import setup_logger

_logger = setup_logger(__name__)


class EngineError(Exception):
    """Base exception for all errors raised by analysis/engine.py."""


class InvalidCornerMetricsError(EngineError):
    """Raised when the supplied corner metrics list is empty or malformed."""


# --------------------------------------------------------------------------
# Rule Engine thresholds (configurable, never hardcoded inline in logic)
# --------------------------------------------------------------------------

#: If braking_delta_m is less than this, the driver braked meaningfully
#: earlier than the reference driver.
EARLY_BRAKING_THRESHOLD_M: float = -10.0

#: If braking_delta_m is greater than this, the driver braked meaningfully
#: later than the reference driver.
LATE_BRAKING_THRESHOLD_M: float = 10.0

#: If apex_speed_delta_kph is less than this, the driver's apex speed was
#: meaningfully slower than the reference driver's.
SLOW_APEX_THRESHOLD_KPH: float = -5.0

#: If exit_speed_delta_kph is less than this, the driver's exit speed was
#: meaningfully slower than the reference driver's.
POOR_EXIT_THRESHOLD_KPH: float = -5.0

#: If throttle_delay_m is greater than this, the driver applied throttle
#: meaningfully later than the reference driver.
LATE_THROTTLE_THRESHOLD_M: float = 8.0

# --------------------------------------------------------------------------
# Severity tiering (ratio of actual value to threshold magnitude)
# --------------------------------------------------------------------------

#: Below this ratio of threshold magnitude, severity is LOW.
_SEVERITY_LOW_MAX_RATIO: float = 1.5

#: Below this ratio of threshold magnitude (and >= low), severity is MEDIUM.
#: At or above this ratio, severity is HIGH.
_SEVERITY_MEDIUM_MAX_RATIO: float = 2.0

# --------------------------------------------------------------------------
# Performance score constants (every penalty documented at its use site)
# --------------------------------------------------------------------------

#: Points subtracted from the score per second of total time lost.
TIME_LOSS_PENALTY_PER_SECOND: float = 10.0

#: Maximum total points the time-loss penalty may contribute, so one very
#: bad lap does not let a single metric dominate the entire score.
TIME_LOSS_PENALTY_CAP: float = 50.0

#: Points subtracted per corner for each severity tier.
SEVERITY_PENALTY_POINTS: dict[IssueSeverity, float] = {
    IssueSeverity.LOW: 2.0,
    IssueSeverity.MEDIUM: 5.0,
    IssueSeverity.HIGH: 10.0,
}

#: Flat points subtracted per corner that has any detected issue, regardless
#: of severity -- penalizes frequency/consistency problems distinctly from
#: severity.
MISTAKE_COUNT_PENALTY_PER_ISSUE: float = 1.0

#: Human-readable descriptions used to build the "biggest mistake" sentence.
_ISSUE_DESCRIPTIONS: dict[IssueType, str] = {
    IssueType.EARLY_BRAKING: "early braking",
    IssueType.LATE_BRAKING: "late braking",
    IssueType.SLOW_APEX: "a slow apex speed",
    IssueType.POOR_EXIT_SPEED: "poor corner exit speed",
    IssueType.LATE_THROTTLE: "late throttle application",
    IssueType.CONSISTENCY_ISSUE: "inconsistent corner execution",
}


# --------------------------------------------------------------------------
# Corner classification
# --------------------------------------------------------------------------


def _severity_from_excess(value: float, threshold: float) -> IssueSeverity:
    """Compute an issue's severity tier from how far a value exceeds its threshold.

    Severity scales with the ratio of the value's magnitude to the
    threshold's magnitude, so the same tiering logic applies consistently
    across every rule (braking, apex, exit, throttle) instead of being
    reimplemented per rule.

    Args:
        value: The metric value that triggered a rule (e.g. braking_delta_m).
        threshold: The threshold that was exceeded.

    Returns:
        IssueSeverity.LOW, MEDIUM, or HIGH based on the excess ratio.
    """
    if threshold == 0:
        # Defensive guard; none of the configured thresholds are zero.
        return IssueSeverity.HIGH

    ratio = abs(value) / abs(threshold)

    if ratio < _SEVERITY_LOW_MAX_RATIO:
        return IssueSeverity.LOW
    if ratio < _SEVERITY_MEDIUM_MAX_RATIO:
        return IssueSeverity.MEDIUM
    return IssueSeverity.HIGH


def _classify_corner(metric: CornerMetric) -> CornerMetric:
    """Classify a single corner's issue type and severity using ordered threshold rules.

    Rules are checked in a fixed priority order -- braking, then apex,
    then exit, then throttle -- because a single corner can trigger more
    than one condition at once (e.g. early braking often also produces a
    slow apex), but CornerMetric can only carry one issue_type. Checking
    braking first attributes the classification to the more likely root
    cause rather than a downstream symptom. See the module-level design
    notes for the full rationale.

    CornerMetric is immutable, so this returns a new instance via
    model_copy() rather than mutating the input.

    Args:
        metric: The corner metric to classify.

    Returns:
        A new CornerMetric with issue_type and severity populated
        (IssueType.NONE / IssueSeverity.NONE if no rule matched).
    """
    issue_type = IssueType.NONE
    severity = IssueSeverity.NONE

    if metric.braking_delta_m < EARLY_BRAKING_THRESHOLD_M:
        issue_type = IssueType.EARLY_BRAKING
        severity = _severity_from_excess(metric.braking_delta_m, EARLY_BRAKING_THRESHOLD_M)
    elif metric.braking_delta_m > LATE_BRAKING_THRESHOLD_M:
        issue_type = IssueType.LATE_BRAKING
        severity = _severity_from_excess(metric.braking_delta_m, LATE_BRAKING_THRESHOLD_M)
    elif metric.apex_speed_delta_kph < SLOW_APEX_THRESHOLD_KPH:
        issue_type = IssueType.SLOW_APEX
        severity = _severity_from_excess(metric.apex_speed_delta_kph, SLOW_APEX_THRESHOLD_KPH)
    elif metric.exit_speed_delta_kph < POOR_EXIT_THRESHOLD_KPH:
        issue_type = IssueType.POOR_EXIT_SPEED
        severity = _severity_from_excess(metric.exit_speed_delta_kph, POOR_EXIT_THRESHOLD_KPH)
    elif metric.throttle_delay_m > LATE_THROTTLE_THRESHOLD_M:
        issue_type = IssueType.LATE_THROTTLE
        severity = _severity_from_excess(metric.throttle_delay_m, LATE_THROTTLE_THRESHOLD_M)

    return metric.model_copy(update={"issue_type": issue_type, "severity": severity})


def classify_corner_metrics(corner_metrics: list[CornerMetric]) -> list[CornerMetric]:
    """Classify every corner in a session using the deterministic Rule Engine.

    Args:
        corner_metrics: Unclassified corner metrics from analysis/corners.py.

    Returns:
        A new list of CornerMetric objects, each with issue_type and
        severity populated, sorted by corner_number.
    """
    classified = [_classify_corner(metric) for metric in corner_metrics]
    classified.sort(key=lambda metric: metric.corner_number)
    return classified


# --------------------------------------------------------------------------
# Performance score
# --------------------------------------------------------------------------


def compute_performance_score(
    classified_corners: list[CornerMetric],
    total_time_loss_s: float,
) -> int:
    """Compute the deterministic 0-100 performance score for a session.

    Score = 100
        - time_loss_penalty   (total_time_loss_s * 10, capped at 50)
        - severity_penalty    (sum of per-corner severity points)
        - mistake_count_penalty (1 point per corner with any issue)
    clamped to [0, 100].

    See the module-level design notes for the rationale behind each term.

    Args:
        classified_corners: Corner metrics with issue_type/severity already set.
        total_time_loss_s: The session's total (non-negative) time loss, in seconds.

    Returns:
        An integer performance score between 0 and 100 inclusive.
    """
    time_loss_penalty = min(
        total_time_loss_s * TIME_LOSS_PENALTY_PER_SECOND, TIME_LOSS_PENALTY_CAP
    )

    severity_penalty = sum(
        SEVERITY_PENALTY_POINTS.get(corner.severity, 0.0) for corner in classified_corners
    )

    mistake_count = sum(
        1 for corner in classified_corners if corner.issue_type != IssueType.NONE
    )
    mistake_count_penalty = mistake_count * MISTAKE_COUNT_PENALTY_PER_ISSUE

    raw_score = 100.0 - time_loss_penalty - severity_penalty - mistake_count_penalty
    clamped_score = max(0.0, min(100.0, raw_score))

    _logger.debug(
        "Score breakdown: time_loss_penalty=%.2f, severity_penalty=%.2f, "
        "mistake_count_penalty=%.2f (n=%d) -> raw=%.2f, clamped=%.2f.",
        time_loss_penalty,
        severity_penalty,
        mistake_count_penalty,
        mistake_count,
        raw_score,
        clamped_score,
    )

    return int(round(clamped_score))


# --------------------------------------------------------------------------
# Biggest mistake
# --------------------------------------------------------------------------


def _determine_biggest_mistake(classified_corners: list[CornerMetric]) -> str:
    """Determine the session's dominant issue and describe it in one sentence.

    Tallies issue types (excluding NONE) across every classified corner;
    the most frequent issue type is treated as the session's defining
    problem. Ties are broken by which issue type was encountered first
    among the classified corners (stable insertion order).

    Args:
        classified_corners: Corner metrics with issue_type already set.

    Returns:
        A concise, human-readable description of the biggest mistake, or
        a neutral message if no issues were detected.
    """
    issue_counts: Counter[IssueType] = Counter(
        corner.issue_type for corner in classified_corners if corner.issue_type != IssueType.NONE
    )

    if not issue_counts:
        return "No significant driving errors detected this lap."

    dominant_issue, occurrence_count = issue_counts.most_common(1)[0]
    description = _ISSUE_DESCRIPTIONS.get(dominant_issue, dominant_issue.value)

    affected_corners = [
        corner.corner_number
        for corner in classified_corners
        if corner.issue_type == dominant_issue
    ]
    corner_list = ", ".join(str(corner_number) for corner_number in affected_corners)

    if occurrence_count == 1:
        return f"{description.capitalize()} detected in corner {corner_list}."

    return f"Repeated {description} detected across corners {corner_list}."


# --------------------------------------------------------------------------
# Public orchestration function
# --------------------------------------------------------------------------


def build_session_analysis(
    corner_metrics: list[CornerMetric],
    session_id: str,
    driver_code: str,
    reference_driver_code: str,
) -> SessionAnalysis:
    """Build a complete SessionAnalysis from a list of (unclassified) corner metrics.

    Orchestrates the full Rule Engine pipeline: classify every corner,
    compute total time loss, identify the worst/best corner, determine
    the biggest mistake, compute the performance score, and assemble the
    validated SessionAnalysis.

    Args:
        corner_metrics: Corner metrics from analysis/corners.py (issue_type
            and severity may be unset -- this function will classify them).
        session_id: Unique session identifier, e.g. "2025_Monaco_Q".
        driver_code: FIA driver code of the analyzed driver.
        reference_driver_code: FIA driver code of the reference driver.

    Returns:
        A fully populated, validated SessionAnalysis.

    Raises:
        InvalidCornerMetricsError: If `corner_metrics` is empty.
        EngineError: If assembling DriverSummary or SessionAnalysis fails
            Pydantic validation (wraps the underlying ValidationError).
    """
    if not corner_metrics:
        raise InvalidCornerMetricsError("corner_metrics must not be empty.")

    classified_corners = classify_corner_metrics(corner_metrics)

    raw_total_time_loss = sum(corner.time_loss_s for corner in classified_corners)
    # Total time loss represents time actually lost, not gained -- if the
    # driver was net faster than the reference across all corners, there is
    # no "loss" to report, so the aggregate is floored at zero (this also
    # satisfies DriverSummary.total_time_loss_s's non-negative constraint).
    total_time_loss = max(raw_total_time_loss, 0.0)

    worst_corner_metric = max(classified_corners, key=lambda corner: corner.time_loss_s)
    best_corner_metric = min(classified_corners, key=lambda corner: corner.time_loss_s)

    performance_score = compute_performance_score(classified_corners, total_time_loss)
    biggest_mistake = _determine_biggest_mistake(classified_corners)

    try:
        driver_summary = DriverSummary(
            driver_code=driver_code,
            reference_driver_code=reference_driver_code,
            total_time_loss_s=total_time_loss,
            performance_score=performance_score,
        )

        session_analysis = SessionAnalysis(
            session_id=session_id,
            driver_summary=driver_summary,
            corner_metrics=classified_corners,
            worst_corner=worst_corner_metric.corner_number,
            best_corner=best_corner_metric.corner_number,
            biggest_mistake=biggest_mistake,
        )
    except ValidationError as exc:
        raise EngineError(f"Failed to build SessionAnalysis: {exc}") from exc

    _logger.info(
        "Built SessionAnalysis for session '%s': score=%d, total_time_loss=%.3fs, "
        "worst_corner=%d, best_corner=%d.",
        session_id,
        performance_score,
        total_time_loss,
        worst_corner_metric.corner_number,
        best_corner_metric.corner_number,
    )

    return session_analysis
